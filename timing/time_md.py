"""Time NVE molecular dynamics through ASE or torch-sim, one (model, engine) per invocation.

    python time_md.py --model models/nequip_les.ckpt --engine ase-aoti
    python time_md.py --model models/nequip_les.ckpt --engine ase-aoti --policy scaled

The calculator is built once and the size ladder is walked inside this one process, so the
model is loaded a handful of times rather than once per size.

What is and is not inside the clock
-----------------------------------
Inside: the MD steps only -- force evaluation plus the neighbour-list rebuild the ASE path
does on every call. Outside: model load, the AOTInductor first call, structure construction,
cuBLAS autotuning, and anything else that only happens once, all absorbed by a warm-up run.
`torch.cuda.synchronize()` brackets the timed segment, because without it the wall clock
straddles kernels still queued on the device.
"""

import argparse
import pathlib
import statistics
import sys
import time

import numpy as np
import torch
from ase import units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

import common as C

sys.path.insert(0, str(C.TESTS))
import engines  # noqa: E402


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


# ----------------------------------------------------------------- calculators ----
def build_ase(engine, ckpt, artefact, device, species):
    from nequip.integrations.ase import NequIPCalculator

    mapping = {s: s for s in species}
    if engine in ("ase-eager", "ase-compile"):
        return NequIPCalculator._from_saved_model(
            str(ckpt), device=device,
            chemical_species_to_atom_type_map=mapping,
            allow_tf32=False,
            compile_mode="compile" if engine == "ase-compile" else "eager",
            neighborlist_backend=C.NL_BACKEND,
        )
    engines.preload_for(artefact.name)  # OEQ/cuEq must be imported before the artefact loads
    return NequIPCalculator.from_compiled_model(
        compile_path=str(artefact), device=device,
        chemical_species_to_atom_type_map=mapping,
    )


def build_torchsim(artefact, device, species):
    from nequip.integrations.torchsim import NequIPTorchSimCalc

    engines.preload_for(artefact.name)
    return NequIPTorchSimCalc.from_compiled_model(
        compile_path=str(artefact), device=device,
        chemical_species_to_atom_type_map={s: s for s in species},
    )


# ------------------------------------------------------------------- MD timing ----
def run_ase(calc, atoms, steps, device):
    atoms = atoms.copy()
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=C.TEMPERATURE_K, rng=np.random.default_rng(C.SEED))
    dyn = VelocityVerlet(atoms, timestep=C.TIMESTEP_FS * units.fs)

    dyn.run(C.WARMUP)                       # not timed
    out = []
    for _ in range(C.REPEATS):
        sync(device)
        t0 = time.perf_counter()
        dyn.run(steps)
        sync(device)
        out.append(time.perf_counter() - t0)
    return out


def run_torchsim(calc, atoms, steps, device):
    import torch_sim as ts

    state = ts.io.atoms_to_state([atoms.copy()], device=device, dtype=torch.float64)
    kw = dict(model=calc, integrator=ts.Integrator.nve,
              timestep=C.TIMESTEP_FS, temperature=C.TEMPERATURE_K)

    state = ts.integrate(system=state, n_steps=C.WARMUP, **kw)   # not timed
    out = []
    for _ in range(C.REPEATS):
        sync(device)
        t0 = time.perf_counter()
        state = ts.integrate(system=state, n_steps=steps, **kw)
        sync(device)
        out.append(time.perf_counter() - t0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--policy", default="fixed", choices=C.POLICIES)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=C.STEPS)
    ap.add_argument("--sizes", default="", help='e.g. "1 1 1;2 2 2"; default is the full ladder')
    args = ap.parse_args()

    ckpt = pathlib.Path(args.model)
    if not ckpt.is_absolute():
        ckpt = C.HERE / ckpt
    ckpt = ckpt.resolve()
    if not ckpt.exists():
        raise SystemExit(f"no such checkpoint: {ckpt}")
    stem = ckpt.stem
    backbone = engines.backbone(ckpt)[0]
    variant = stem[len(backbone) + 1:]
    species = engines.type_names(ckpt)
    model_dtype = next(
        (v for v in engines._collect_key(
            torch.load(ckpt, map_location="cpu", weights_only=False)
            .get("hyper_parameters", {}), "model_dtype") if isinstance(v, str)),
        "unknown")

    spec = next((e for e in C.ENGINES[backbone] if e[0] == args.engine), None)
    if spec is None:
        raise SystemExit(f"{args.engine} is not an engine for a {backbone} model")
    _, target, modifier, runner = spec
    if runner != "md":
        raise SystemExit(f"{args.engine} is a {runner} engine; use time_lammps.py")

    base = read(C.BASE_XYZ, index=0)
    lengths = base.cell.lengths()
    sizes = ([tuple(int(v) for v in s.split()) for s in args.sizes.split(";") if s.strip()]
             or C.SIZES)

    done = C.done_keys()
    common_row = dict(model=stem, backbone=backbone, variant=variant, engine=args.engine,
                      accel=modifier or "", nmax_policy=args.policy, steps=args.steps,
                      nl_backend=C.NL_BACKEND, dtype=model_dtype, device=args.device,
                      gpu_name=C.gpu_name(), torch_version=torch.__version__)

    calc_cache = {}
    for nxyz in sizes:
        nx, ny, nz = nxyz
        n_max = C.n_max_for(lengths, nxyz) if args.policy == "scaled" else ""
        k_vectors = (2 * n_max + 1) ** 3 if n_max != "" else ""

        key = (stem, args.engine, args.policy, str(nx), str(ny), str(nz), "0")
        if key in done:
            print(f"  {nx}{ny}{nz}  already in {C.CSV_PATH.name}, skipping")
            continue

        atoms = base.repeat(nxyz)
        row = dict(common_row, natoms=len(atoms), nx=nx, ny=ny, nz=nz,
                   n_max=n_max, k_vectors=k_vectors)

        try:
            if n_max not in calc_cache:
                if args.policy == "scaled":
                    src = C.DERIVED_DIR / f"{stem}_nmax{n_max}.ckpt"
                    art = C.artefact_path(stem, target, modifier, n_max) if target else None
                else:
                    src = ckpt
                    art = C.artefact_path(stem, target, modifier) if target else None
                if art is not None and not art.exists():
                    raise FileNotFoundError(f"{art} -- run prepare.py first")
                calc_cache[n_max] = (
                    build_torchsim(art, args.device, species)
                    if args.engine.startswith("torchsim")
                    else build_ase(args.engine, src, art, args.device, species))
            calc = calc_cache[n_max]

            walls = (run_torchsim(calc, atoms, args.steps, args.device)
                     if args.engine.startswith("torchsim")
                     else run_ase(calc, atoms, args.steps, args.device))
            for i, wall in enumerate(walls):
                C.append_row(dict(row, repeat=i, wall_s=f"{wall:.6f}",
                                  s_per_step=f"{wall / args.steps:.8f}",
                                  steps_per_s=f"{args.steps / wall:.6f}",
                                  atom_steps_per_s=f"{len(atoms) * args.steps / wall:.1f}",
                                  status="ok"))
            med = statistics.median(walls)
            print(f"  {nx}{ny}{nz}  {len(atoms):6d} atoms  "
                  f"{med / args.steps * 1e3:8.3f} ms/step  "
                  f"{args.steps / med:7.2f} steps/s"
                  + (f"  N_max={n_max}" if n_max != "" else ""))

        except torch.cuda.OutOfMemoryError:
            C.append_row(dict(row, repeat=0, status="oom"))
            print(f"  {nx}{ny}{nz}  {len(atoms):6d} atoms  OOM -- stopping this ladder")
            calc_cache.clear()
            torch.cuda.empty_cache()
            break
        except Exception as exc:  # noqa: BLE001 -- record and stop, do not lose the sweep
            C.append_row(dict(row, repeat=0, status=f"error: {type(exc).__name__}"))
            print(f"  {nx}{ny}{nz}  {len(atoms):6d} atoms  ERROR {type(exc).__name__}: {exc}")
            break


if __name__ == "__main__":
    main()
