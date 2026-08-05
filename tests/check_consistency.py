"""One checkpoint, exported fresh to every target, run through every engine.

    python check_consistency.py model.ckpt structure.xyz [--outdir consistency_out]

Use this for a model of your own. To cross-check the artefacts the test suite already
produced (KEEP_OUTPUTS=1 ./run_gpu.sh), use run_inference_matrix.py instead -- it reuses them
rather than exporting again.

    engine     artefact                                needs
    ase        --target ase                             -
    torchsim   --target batch                           pip install torch-sim-atomistic
    lammps     --target pair_nequip / pair_allegro      $LMP
    mliap      nequip-prepare-lmp-mliap                 $LMP_MLIAP

Engines whose artefact or binary is missing are skipped and reported, so this is useful
before the LAMMPS builds exist as well as after.

Why bother: `nequip-compile` already checks each artefact against the eager model, but one
at a time and only through its own interface. It cannot catch a wrong unit, a wrong type
mapping, or a cell that never reached the graph -- the failures that surface as "the same
model gives different forces in LAMMPS". Those only show up when the engines sit side by
side.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read

import engines

HERE = Path(__file__).resolve().parent


def run(cmd, log):
    with open(log, "w") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode == 0


def compile_cmd():
    """`nequip-compile`, through the TF32 wrapper when it is present.

    On torch 2.13 + CUDA every export otherwise dies reading a legacy cuDNN flag; see
    compile_tf32fix.py.
    """
    fix = HERE / "compile_tf32fix.py"
    return [sys.executable, str(fix)] if fix.exists() else ["nequip-compile"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("structure", help="xyz/extxyz file; the first frame is used")
    ap.add_argument("--outdir", default="consistency_out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--etol", type=float, default=1e-4, help="eV/atom (default 1e-4)")
    ap.add_argument("--ftol", type=float, default=1e-3, help="eV/A (default 1e-3)")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    out = Path(args.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    atoms = read(args.structure, index=0)
    species = engines.type_names(ckpt)
    allegro = engines.is_allegro(ckpt)
    pair_target = "pair_allegro" if allegro else "pair_nequip"

    print(f"checkpoint : {ckpt}")
    print(f"structure  : {args.structure}  ({len(atoms)} atoms, pbc={atoms.pbc.all()})")
    print(f"type names : {species}")
    print(f"backbone   : {'allegro' if allegro else 'nequip'}  ->  --target {pair_target}")
    print(f"device     : {args.device}")

    # ---------------------------------------------------------------- export ----
    print("\n== exporting ==")
    artefacts = {}
    for name, target in (("ase", "ase"), ("torchsim", "batch"), ("lammps", pair_target)):
        path = out / f"{name}.nequip.pt2"
        if path.exists():
            print(f"  {name:9s} reusing {path.name}")
            artefacts[name] = path
            continue
        cmd = compile_cmd() + [str(ckpt), str(path), "--device", args.device,
                               "--mode", "aotinductor", "--target", target]
        print(f"  {name:9s} $ {' '.join(cmd)}")
        if run(cmd, out / f"export_{name}.log") and path.exists():
            artefacts[name] = path
        else:
            print(f"  {name:9s} EXPORT FAILED (see export_{name}.log)")

    mliap_file = out / "model.nequip.lmp.pt"
    if shutil.which("nequip-prepare-lmp-mliap") and not mliap_file.exists():
        cmd = ["nequip-prepare-lmp-mliap", str(ckpt), str(mliap_file)]
        print(f"  {'mliap':9s} $ {' '.join(cmd)}")
        run(cmd, out / "export_mliap.log")
    if mliap_file.exists():
        artefacts["mliap"] = mliap_file

    # ------------------------------------------------------------- evaluate ----
    print("\n== evaluating ==")
    results, skipped = {}, {}

    def attempt(name, fn):
        try:
            e, f = fn()
        except Exception as exc:                           # noqa: BLE001 - report, continue
            skipped[name] = f"{type(exc).__name__}: {exc}"
            print(f"  {name:9s} skipped ({skipped[name][:70]})")
            return
        results[name] = (e, f)
        print(f"  {name:9s} E = {e:18.8f} eV   |F|max = {np.abs(f).max():.8f} eV/A")

    if "ase" in artefacts:
        attempt("ase", lambda: engines.eval_ase(artefacts["ase"], atoms, args.device, species))
    if "torchsim" in artefacts:
        attempt("torchsim",
                lambda: engines.eval_torchsim(artefacts["torchsim"], atoms, args.device, species))

    lmp = os.environ.get("LMP")
    if "lammps" in artefacts:
        if lmp:
            lines = engines.lammps_pair_lines(artefacts["lammps"], species, allegro)
            attempt("lammps",
                    lambda: engines.eval_lammps(lmp, atoms, species, out / "lammps", lines)[:2])
        else:
            skipped["lammps"] = "$LMP unset (./build_lammps.sh pair)"
            print(f"  {'lammps':9s} skipped ({skipped['lammps']})")

    lmp_mliap = os.environ.get("LMP_MLIAP")
    if "mliap" in artefacts:
        if lmp_mliap:
            lines = engines.lammps_mliap_lines(artefacts["mliap"], species)
            attempt("mliap",
                    lambda: engines.eval_lammps(lmp_mliap, atoms, species, out / "mliap", lines,
                                                engines.MLIAP_KOKKOS_ARGS,
                                                newton="on")[:2])
        else:
            skipped["mliap"] = "$LMP_MLIAP unset (./build_lammps.sh mliap)"
            print(f"  {'mliap':9s} skipped ({skipped['mliap']})")

    # -------------------------------------------------------------- compare ----
    if len(results) < 2:
        print("\nfewer than two engines ran; nothing to compare.")
        for name, why in skipped.items():
            print(f"  {name}: {why}")
        return 0 if results else 1

    ref = "ase" if "ase" in results else sorted(results)[0]
    ref_e, ref_f = results[ref]
    n = len(atoms)

    print(f"\n== differences against {ref} ==")
    print(f"  tolerances: {args.etol:g} eV/atom, {args.ftol:g} eV/A")
    failures = []
    for name in sorted(results):
        if name == ref:
            continue
        e, f = results[name]
        de = abs(e - ref_e) / n
        df = float(np.abs(f - ref_f).max()) if f.shape == ref_f.shape else float("inf")
        ok = de <= args.etol and df <= args.ftol
        print(f"  {name:10s} dE/atom = {de:10.3e} eV   dFmax = {df:10.3e} eV/A   "
              f"{'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(name)

    print("\n==================== SUMMARY ====================")
    print(f"  ran      : {', '.join(sorted(results))}")
    for name, why in skipped.items():
        print(f"  skipped  : {name} -- {why}")
    print(f"  agreeing : {len(results) - 1 - len(failures)}/{len(results) - 1} vs {ref}")
    print("=================================================")
    if failures:
        print(f"MISMATCH in: {', '.join(failures)}")
        return 1
    print("all engines agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
