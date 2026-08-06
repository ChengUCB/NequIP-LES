"""Evaluate one structure through each inference engine a compiled model can reach.

Shared by check_consistency.py (exports fresh from a checkpoint) and
run_inference_matrix.py (reuses whatever run_gpu.sh / run_compile.sh already produced).

Each `eval_*` returns `(energy_eV, forces_eV_per_A)` for the structure it is given, so the
callers can just difference them. `eval_lammps` returns a third element, a dict of any
per-atom model outputs that were asked for (e.g. LES's latent charges).

Docs:
    https://nequip.readthedocs.io/en/latest/integrations/ase.html
    https://nequip.readthedocs.io/en/latest/integrations/torchsim.html
    https://nequip.readthedocs.io/en/latest/integrations/lammps/pair_styles.html
"""

import os
import re
import shlex
import subprocess
from pathlib import Path

import numpy as np
import torch
from ase.io import write

# --- targets and modes that appear in artefact filenames -----------------------------
TARGETS = ("ase", "batch", "pair_nequip", "pair_allegro")
MODES = ("aotinductor", "torchscript")


# --------------------------------------------------------------- checkpoint reading ----
def _walk_for_key(obj, key):
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if key in cur:
                return cur[key]
            stack.extend(cur.values())
    return None


def type_names(ckpt_path):
    """The model's atom type names, in the order LAMMPS types must be mapped to."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    names = _walk_for_key(ckpt.get("hyper_parameters", {}), "type_names")
    if names is None:
        raise SystemExit(f"could not find type_names in {ckpt_path}")
    return list(names)


def _collect_key(obj, key):
    """Every value stored under `key`, anywhere in a nested dict/list."""
    found, stack = [], [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if key in cur:
                found.append(cur[key])
            stack.extend(cur.values())
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)
    return found


def backbone(ckpt_path):
    """Which backbone a checkpoint holds: ("nequip" | "allegro", evidence).

    Not a substring search for "allegro" over the whole config -- nequip records its
    installed extension packages in the checkpoint, so that name is present whenever allegro
    is merely installed, and every model looked like an Allegro one. Read the model's own
    declaration instead.
    """
    hp = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("hyper_parameters", {})

    # LES models name it directly
    for val in _collect_key(hp, "base_model"):
        if isinstance(val, str) and val.strip().lower() in ("nequip", "allegro"):
            return val.strip().lower(), f"base_model: {val.strip()}"

    # otherwise the model builder's own target
    targets = [t for t in _collect_key(hp, "_target_")
               if isinstance(t, str) and ".model" in t]
    for t in targets:
        if t.startswith("allegro."):
            return "allegro", f"_target_: {t}"
    for t in targets:
        if t.startswith("nequip."):
            return "nequip", f"_target_: {t}"

    raise SystemExit(
        f"cannot tell the backbone of {ckpt_path} from its config "
        f"(model targets seen: {targets or 'none'}). Pass --backbone nequip|allegro."
    )


# ------------------------------------------------------------------------- preload ----
def preload_for(artefact_name):
    """Import the kernel package an accelerated artefact needs before it is loaded.

    The nequip acceleration docs are explicit that these imports must happen *before* the
    compiled model is loaded, so a plain `torch.load` of an OpenEquivariance artefact fails
    without them.
    """
    name = artefact_name.lower()
    if "openequivariance" in name:
        import openequivariance  # noqa: F401
    if "cuequivariance" in name:
        import cuequivariance_torch  # noqa: F401
    # Triton kernels need no explicit import


# ------------------------------------------------------------------------- engines ----
def eval_ase(artefact, atoms, device, species):
    from nequip.integrations.ase import NequIPCalculator

    preload_for(Path(artefact).name)
    atoms = atoms.copy()
    atoms.calc = NequIPCalculator.from_compiled_model(
        compile_path=str(artefact),
        device=device,
        chemical_species_to_atom_type_map={s: s for s in species},
    )
    return float(atoms.get_potential_energy()), np.asarray(atoms.get_forces(), dtype=float)


def eval_torchsim(artefact, atoms, device, species):
    import torch_sim as ts
    from nequip.integrations.torchsim import NequIPTorchSimCalc

    preload_for(Path(artefact).name)
    calc = NequIPTorchSimCalc.from_compiled_model(
        compile_path=str(artefact),
        device=device,
        chemical_species_to_atom_type_map={s: s for s in species},
    )
    # torch-sim is batched by construction: one structure in, a length-1 energy out
    state = ts.io.atoms_to_state([atoms.copy()], device=device, dtype=torch.float64)
    out = calc(state)
    energy = float(np.asarray(out["energy"].detach().cpu()).reshape(-1)[0])
    forces = np.asarray(out["forces"].detach().cpu(), dtype=float)
    return energy, forces


# LAMMPS thermo output carries only a handful of significant digits, which is coarser than
# the differences we are trying to measure. pair_nequip_allegro's own repro tests get around
# it by printing `1e6 * pe` to a file; the same trick is used here.
PRECISION_CONST = 1.0e6


def eval_lammps(lmp, atoms, species, workdir, pair_lines, extra_args=(),
                newton=None, extract=()):
    """Run `lmp` for zero timesteps and read the potential energy and forces back.

    `pair_lines` is the only difference between the pair styles and ML-IAP; the rest of the
    input script is shared on purpose, so a discrepancy cannot come from the setup.

    `extra_args` goes on the lmp command line, e.g. Kokkos runtime flags.

    `newton` defaults to whatever the pair style demands, because the two disagree
    (pair_nequip_allegro.cpp:149-150):

        pair_style nequip   -> newton pair off, or it errors out
        pair_style allegro  -> newton pair on,  or it errors out

    `extract` names per-atom keys to pull out of the model's own output dictionary via
    `compute <style>/atom <key> <n_components> 0`; they come back in the third element of the
    result. Left empty by default, and LES quantities cannot go here: `nequip-compile`'s
    targets fix the exported outputs to

        pair_nequip / pair_allegro : per_atom_energy, forces, virial
        ase / batch                : per_atom_energy, total_energy, forces, stress

    so a compiled artefact does not carry `LES_q`, let alone `LES_BEC` -- which is not even
    computed unless BEC is switched on. The mechanism is kept for `atomic_energy` and
    `forces`, which the pair styles' own repro tests use to cross-check LAMMPS against the
    model's own arrays.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # LAMMPS' read_data rejects atoms that lie outside the declared box
    # ("Did not assign all atoms correctly"), which is easy to hit: an isolated structure
    # carries a dummy cell its coordinates were never fitted into. Both adjustments below
    # leave energies and forces untouched -- one wraps into an equivalent periodic image,
    # the other only translates and resizes a box the model ignores anyway.
    atoms = atoms.copy()
    if bool(atoms.pbc.all()):
        atoms.wrap()
    else:
        pos = atoms.positions
        margin = 10.0
        atoms.set_cell(np.diag(pos.max(axis=0) - pos.min(axis=0) + 2.0 * margin))
        atoms.positions = pos - pos.min(axis=0) + margin

    data = workdir / "data.lmp"
    # `specorder` fixes LAMMPS type 1..n to `species`, and the pair_coeff line names the
    # same list in the same order. Both come from one variable because a mismatch here is
    # the classic "model runs, forces are nonsense" failure.
    # No masses in the data file: the keyword's name has changed across ASE versions, and
    # LAMMPS only insists that *some* mass exists. They play no part in a 0-step energy and
    # force evaluation, so they are set in the input script instead -- which is what
    # pair_nequip_allegro's own repro tests do.
    write(data, atoms, format="lammps-data", specorder=list(species), atom_style="atomic")

    style = "allegro" if "pair_style      allegro" in pair_lines else "nequip"
    if newton is None:
        newton = "off" if "pair_style      nequip" in pair_lines else "on"
    extract_cmds, dump_cols = [], []
    for key, ncomp in extract:
        cid = "x_" + re.sub(r"\W", "_", key)
        # the reverse-communicate flag must be 0 for pair nequip (compute/README.md)
        extract_cmds.append(f"compute {cid} all {style}/atom {key} {ncomp} 0")
        dump_cols += ([f"c_{cid}"] if ncomp == 1
                      else [f"c_{cid}[{i + 1}]" for i in range(ncomp)])

    lines = [
        "units           metal",
        "atom_style      atomic",
        # `s` (shrink-wrapped) rather than `f` for the non-periodic case, as in
        # pair_nequip_allegro's repro tests: a fixed box loses atoms that sit outside it,
        # while a shrink-wrapped one always encloses them
        f"boundary        {'p p p' if bool(atoms.pbc.all()) else 's s s'}",
        "atom_modify     map yes",
        f"newton          {newton}",
        f"read_data       {data.name}",
        pair_lines,
        *[f"mass            {i + 1} 1.0" for i in range(len(species))],
        # a neighbour list rebuilt unconditionally, so `run 0` cannot reuse a stale one
        "neighbor        1.0 bin",
        "neigh_modify    delay 0 every 1 check no",
        *extract_cmds,
        "thermo_style    custom step pe",
        "thermo          1",
        "run             0",
        # full precision, via a file rather than the thermo table, which rounds
        f"print           $({PRECISION_CONST} * pe) file pe.dat",
        "write_dump      all custom out.dump id fx fy fz "
        + " ".join(dump_cols) + " modify format float %20.15g",
    ]
    script = workdir / "in.check"
    script.write_text("\n".join(lines) + "\n")

    # LAMMPS is built against a real MPI (the pair styles require one). Started as a plain
    # subprocess from inside an `srun` step, its MPI_Init sees the outer step's PMI/PMIX
    # variables, aborts on a NULL communicator, and takes the whole SLURM step down. Hiding
    # those variables makes it initialise as an ordinary single-process job instead.
    # Set LMP_LAUNCHER="srun -n 1" to launch it properly instead, in which case the
    # environment is left alone.
    launcher = shlex.split(os.environ.get("LMP_LAUNCHER", ""))
    env = dict(os.environ)
    if not launcher:
        for key in [k for k in env if k.startswith(("PMI_", "PMIX_", "SLURM_"))]:
            del env[key]

    # A `run 0` on a few hundred atoms is seconds of work. Anything much longer means it is
    # stuck rather than busy -- MPI startup is the usual culprit -- so fail instead of hanging.
    timeout = float(os.environ.get("LMP_TIMEOUT", "300"))

    log = workdir / "lmp.log"
    with open(log, "w") as fh:
        cmd = [*launcher, str(lmp), "-in", script.name, *[str(a) for a in extra_args]]
        fh.write("# " + " ".join(cmd) + "\n")
        fh.flush()
        try:
            rc = subprocess.run(cmd, cwd=workdir, stdout=fh, env=env,
                                stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"lmp still running after {timeout:.0f}s -- see {log}. "
                f"If it is stuck at startup, try LMP_LAUNCHER='srun -n 1', "
                f"or raise LMP_TIMEOUT."
            ) from None
    if rc != 0:
        # the last few lines are usually LAMMPS' setup chatter, not the failure; pick out the
        # lines that actually say something went wrong, and fall back to the tail
        text = log.read_text().splitlines()
        hits = [ln.strip() for ln in text
                if re.search(r"ERROR|Error|Exception|Traceback|error:", ln)]
        detail = " | ".join(hits[-3:]) if hits else " | ".join(t.strip() for t in text[-3:])
        raise RuntimeError(f"lmp exited {rc}: {detail}  (full log: {log})")

    pe_file = workdir / "pe.dat"
    if not pe_file.exists():
        tail = "\n".join(log.read_text().splitlines()[-6:])
        raise RuntimeError(f"lmp produced no pe.dat: {tail}")
    energy = float(pe_file.read_text().split()[0]) / PRECISION_CONST

    # parse the dump by column name and sort by id ourselves, so the result does not depend
    # on whether `write_dump ... modify sort id` is honoured
    dump = (workdir / "out.dump").read_text().splitlines()
    hdr = next(i for i, ln in enumerate(dump) if ln.startswith("ITEM: ATOMS"))
    cols = dump[hdr].split()[2:]
    rows = np.array([[float(c) for c in ln.split()] for ln in dump[hdr + 1:] if ln.strip()])
    rows = rows[np.argsort(rows[:, cols.index("id")])]

    forces = np.stack([rows[:, cols.index(c)] for c in ("fx", "fy", "fz")], axis=1)
    extras = {}
    for key, ncomp in extract:
        cid = "x_" + re.sub(r"\W", "_", key)
        names = [f"c_{cid}"] if ncomp == 1 else [f"c_{cid}[{i + 1}]" for i in range(ncomp)]
        extras[key] = np.stack([rows[:, cols.index(nm)] for nm in names], axis=1)
    return energy, forces, extras


def lammps_pair_lines(artefact, species, allegro):
    style = "allegro" if allegro else "nequip"
    return (f"pair_style      {style}\n"
            f"pair_coeff      * * {artefact} {' '.join(species)}")

