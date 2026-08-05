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
    https://nequip.readthedocs.io/en/latest/integrations/lammps/mliap.html
"""

import re
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


def is_allegro(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return "allegro" in str(ckpt.get("hyper_parameters", {})).lower()


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
                newton="off", extract=()):
    """Run `lmp` for zero timesteps and read the potential energy and forces back.

    `pair_lines` is the only difference between the pair styles and ML-IAP; the rest of the
    input script is shared on purpose, so a discrepancy cannot come from the setup.

    `extra_args` goes on the lmp command line. ML-IAP needs the Kokkos runtime flags there
    (`-k on g 1 -sf kk ...`), as the nequip docs' example shows -- that integration is built
    on the KOKKOS package, and without them the styles it installs are not the ones used.

    `newton` is "off" for the pair styles, following pair_nequip_allegro's own repro tests,
    and "on" for ML-IAP, following the nequip ML-IAP docs.

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
    data = workdir / "data.lmp"
    # `specorder` fixes LAMMPS type 1..n to `species`, and the pair_coeff line names the
    # same list in the same order. Both come from one variable because a mismatch here is
    # the classic "model runs, forces are nonsense" failure.
    write(data, atoms, format="lammps-data", specorder=list(species), masses=True,
          atom_style="atomic")

    style = "allegro" if "pair_style      allegro" in pair_lines else "nequip"
    extract_cmds, dump_cols = [], []
    for key, ncomp in extract:
        cid = "x_" + re.sub(r"\W", "_", key)
        # the reverse-communicate flag must be 0 for pair nequip (compute/README.md)
        extract_cmds.append(f"compute {cid} all {style}/atom {key} {ncomp} 0")
        dump_cols += ([f"c_{cid}"] if ncomp == 1
                      else [f"c_{cid}[{i + 1}]" for i in range(ncomp)])

    script = workdir / "in.check"
    script.write_text(
        "units           metal\n"
        "atom_style      atomic\n"
        # `s` (shrink-wrapped) rather than `f` for the non-periodic case, as in
        # pair_nequip_allegro's repro tests: a fixed box loses atoms that sit outside it,
        # while a shrink-wrapped one always encloses them
        f"boundary        {'p p p' if bool(atoms.pbc.all()) else 's s s'}\n"
        "atom_modify     map yes\n"
        f"newton          {newton}\n"
        f"read_data       {data.name}\n"
        f"{pair_lines}\n"
        # a neighbour list rebuilt unconditionally, so `run 0` cannot reuse a stale one
        "neighbor        1.0 bin\n"
        "neigh_modify    delay 0 every 1 check no\n"
        + "".join(c + "\n" for c in extract_cmds) +
        "thermo_style    custom step pe\n"
        "thermo          1\n"
        "run             0\n"
        # full precision, via a file rather than the thermo table
        f"print           $({PRECISION_CONST} * pe) file pe.dat\n"
        "write_dump      all custom out.dump id fx fy fz "
        + " ".join(dump_cols) +
        " modify format float %20.15g\n"
    )

    log = workdir / "lmp.log"
    with open(log, "w") as fh:
        cmd = [str(lmp), "-in", script.name, *[str(a) for a in extra_args]]
        fh.write("# " + " ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=workdir, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    if rc != 0:
        tail = "\n".join(log.read_text().splitlines()[-6:])
        raise RuntimeError(f"lmp exited {rc}: {tail}")

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


def lammps_mliap_lines(artefact, species):
    return (f"pair_style      mliap unified {artefact} 0\n"
            f"pair_coeff      * * {' '.join(species)}")


# from the nequip ML-IAP docs' run example:
#   srun -n 1 lmp -in in.lammps -k on g 1 -sf kk -pk kokkos newton on neigh half
MLIAP_KOKKOS_ARGS = ("-k", "on", "g", "1", "-sf", "kk",
                     "-pk", "kokkos", "newton", "on", "neigh", "half")
