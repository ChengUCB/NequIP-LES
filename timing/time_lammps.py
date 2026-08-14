"""Time NVE molecular dynamics in LAMMPS, one model per invocation.

    export LMP=/path/to/lmp
    python time_lammps.py --model models/nequip_les.ckpt --engine lammps

Uses LAMMPS' own `Loop time`, read from a **second** `run` after a warm-up `run`. That number
already excludes setup, the first neighbour build and the model load, so it is the cleanest
per-step time the engine can give.

`pair_style nequip` for both backbones: it is the only LAMMPS target that passes a cell, which
a periodic Ewald sum needs, and it is limited to one MPI rank -- which LES is anyway, since the
structure factor is a sum over every atom in the cell.
"""

import argparse
import os
import pathlib
import statistics
import sys

import torch
from ase.io import read

import common as C

sys.path.insert(0, str(C.TESTS))
import engines  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", default="lammps")
    ap.add_argument("--policy", default="fixed", choices=C.POLICIES)
    ap.add_argument("--steps", type=int, default=C.STEPS)
    ap.add_argument("--sizes", default="")
    args = ap.parse_args()

    lmp = os.environ.get("LMP")
    if not lmp:
        raise SystemExit("$LMP is not set -- build it with ../tests/build_lammps.sh pair")

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

    spec = next((e for e in C.ENGINES[backbone] if e[0] == args.engine), None)
    if spec is None:
        raise SystemExit(f"{args.engine} is not an engine for a {backbone} model")
    _, target, modifier, runner = spec
    if runner != "lammps":
        raise SystemExit(f"{args.engine} is a {runner} engine; use time_md.py")

    base = read(C.BASE_XYZ, index=0)
    lengths = base.cell.lengths()
    sizes = ([tuple(int(v) for v in s.split()) for s in args.sizes.split(";") if s.strip()]
             or C.SIZES)

    done = C.done_keys()
    common_row = dict(model=stem, backbone=backbone, variant=variant, engine=args.engine,
                      accel=modifier or "", nmax_policy=args.policy, steps=args.steps,
                      # LAMMPS builds its own neighbour list, so the ASE-side backend is not
                      # part of this measurement
                      nl_backend="lammps", dtype="", device="cuda",
                      gpu_name=C.gpu_name(), torch_version=torch.__version__)

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

        art = (C.artefact_path(stem, target, modifier, n_max) if args.policy == "scaled"
               else C.artefact_path(stem, target, modifier))
        if not art.exists():
            C.append_row(dict(row, repeat=0, status="error: artefact missing"))
            print(f"  {nx}{ny}{nz}  artefact missing: {art.name} -- run prepare.py")
            break

        pair_lines = engines.lammps_pair_lines(art, species, allegro=False)
        walls = []
        try:
            # a fresh directory per repeat: LAMMPS writes log.lammps and the data file in cwd,
            # and reusing one directory would let a stale log be parsed
            for i in range(C.REPEATS):
                wd = C.HERE / "lmp_runs" / f"{stem}_{args.engine}_{args.policy}_{nx}{ny}{nz}_{i}"
                wall, nsteps, natoms = engines.time_lammps_md(
                    lmp, atoms, species, wd, pair_lines,
                    steps=args.steps, warmup=C.WARMUP,
                    timestep_fs=C.TIMESTEP_FS, temperature_K=C.TEMPERATURE_K, seed=C.SEED,
                    timeout=float(os.environ.get("LMP_TIMEOUT", "3600")),
                )
                walls.append(wall)
                C.append_row(dict(row, repeat=i, natoms=natoms, wall_s=f"{wall:.6f}",
                                  s_per_step=f"{wall / nsteps:.8f}",
                                  steps_per_s=f"{nsteps / wall:.6f}",
                                  atom_steps_per_s=f"{natoms * nsteps / wall:.1f}",
                                  status="ok"))
            med = statistics.median(walls)
            print(f"  {nx}{ny}{nz}  {len(atoms):6d} atoms  "
                  f"{med / args.steps * 1e3:8.3f} ms/step  "
                  f"{args.steps / med:7.2f} steps/s"
                  + (f"  N_max={n_max}" if n_max != "" else ""))
        except Exception as exc:  # noqa: BLE001 -- record and stop this ladder
            msg = str(exc)
            status = "oom" if "out of memory" in msg.lower() else f"error: {type(exc).__name__}"
            C.append_row(dict(row, repeat=0, status=status))
            print(f"  {nx}{ny}{nz}  {len(atoms):6d} atoms  {status}: {msg[:160]}")
            break


if __name__ == "__main__":
    main()
