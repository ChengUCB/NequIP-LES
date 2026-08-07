"""One checkpoint, exported fresh to every target, run through every engine.

    python check_consistency.py model.ckpt structure.xyz [--outdir consistency_out]

Use this for a model of your own. To cross-check the artefacts the test suite already
produced (KEEP_OUTPUTS=1 ./run_gpu.sh), use run_inference_matrix.py instead -- it reuses them
rather than exporting again.

    engine     artefact                                needs
    ase        --target ase                             -
    torchsim   --target batch                           pip install torch-sim-atomistic
    lammps     --target pair_nequip / pair_allegro      $LMP

Engines whose artefact or binary is missing are skipped and reported, so this is useful
before the LAMMPS build exists as well as after.

KNOWN GAP -- LAMMPS ML-IAP is not exercised.
    The ML-IAP wrapper hands the model `edge_vectors` but neither absolute positions nor the
    cell, so a LES model raises KeyError: 'pos' -- the Ewald sum has nothing to sum over. Its
    run-time torch.compile also fails on torch 2.13 inside nequip's cutoff function
    (InductorError: KeyError 'unbacked_bindings'). Both are upstream matters in what nequip
    documents as a beta integration, so nothing is run here until they are resolved there.

Why bother: `nequip-compile` already checks each artefact against the eager model, but one
at a time and only through its own interface. It cannot catch a wrong unit, a wrong type
mapping, or a cell that never reached the graph -- the failures that surface as "the same
model gives different forces in LAMMPS". Those only show up when the engines sit side by
side.
"""

import argparse
import importlib.util
import os
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
    ap.add_argument("--outdir", default=None,
                    help="default: consistency_out/<checkpoint name>")
    ap.add_argument("--backbone", choices=("auto", "nequip", "allegro"), default="auto",
                    help="override the backbone detected from the checkpoint")
    ap.add_argument("--pair-target", choices=("auto", "pair_nequip", "pair_allegro"),
                    default="auto",
                    help="which LAMMPS target to export. `auto` follows the backbone. Worth "
                         "overriding to pair_nequip for a PERIODIC Allegro+LES model: only "
                         "that target passes a cell, and it feeds the model only the local "
                         "atoms, which is what an Ewald sum needs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--etol", type=float, default=1e-4, help="eV/atom (default 1e-4)")
    ap.add_argument("--ftol", type=float, default=1e-3, help="eV/A (default 1e-3)")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    # keyed on the checkpoint: artefacts are named by target only, so a shared directory
    # would silently hand a different model's exports to the comparison. The stem alone is
    # not enough -- every lightning run produces `epoch=N-step=M.ckpt`, so all models would
    # collide -- hence the nearest meaningful ancestor directory is included.
    noise = {"checkpoints", "lightning_logs", "outputs", "ckpt", "run"}
    tag = next((d for d in reversed(ckpt.parent.parts)
                if d not in noise and not d.startswith("version_")), "model")
    out = Path(args.outdir or Path("consistency_out") / f"{tag}_{ckpt.stem}").resolve()
    out.mkdir(parents=True, exist_ok=True)

    # and a stamp, in case the same checkpoint path is rewritten by a later training run
    stamp_file = out / "source.txt"
    stamp = f"{ckpt}\n{ckpt.stat().st_mtime_ns}\n"
    if stamp_file.exists() and stamp_file.read_text() != stamp:
        print(f"(checkpoint changed since {out.name} was built; re-exporting)")
        for old in list(out.glob("*.nequip.pt2")) + list(out.glob("*.nequip.lmp.pt")):
            old.unlink()
    stamp_file.write_text(stamp)

    atoms = read(args.structure, index=0)
    species = engines.type_names(ckpt)
    if args.backbone == "auto":
        base, why = engines.backbone(ckpt)
    else:
        base, why = args.backbone, "--backbone"
    allegro = base == "allegro"
    pair_target = ("pair_allegro" if allegro else "pair_nequip") \
        if args.pair_target == "auto" else args.pair_target
    # the LAMMPS pair style follows the artefact's target, not the backbone
    pair_style_is_allegro = pair_target == "pair_allegro"

    print(f"checkpoint : {ckpt}")
    print(f"structure  : {args.structure}  ({len(atoms)} atoms, pbc={atoms.pbc.all()})")
    print(f"type names : {species}")
    print(f"backbone   : {base}  ->  --target {pair_target}   ({why})")
    print(f"device     : {args.device}")
    print(f"outdir     : {out}")

    # ---------------------------------------------------------------- export ----
    print("\n== exporting ==")
    artefacts = {}
    wanted = [("ase", "ase"), ("lammps", pair_target)]
    # no point exporting the batched target with nothing able to load it (torch-sim needs
    # python >= 3.11, so it is simply absent on some environments)
    if importlib.util.find_spec("torch_sim") is not None:
        wanted.insert(1, ("torchsim", "batch"))
    else:
        print("  torchsim  not exported: torch_sim not importable "
              "(pip install torch-sim-atomistic, needs python >= 3.11)")
    for name, target in wanted:
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
            lines = engines.lammps_pair_lines(artefacts["lammps"], species,
                                              pair_style_is_allegro)
            attempt("lammps",
                    lambda: engines.eval_lammps(lmp, atoms, species, out / "lammps", lines)[:2])
        else:
            skipped["lammps"] = "$LMP unset (./build_lammps.sh pair)"
            print(f"  {'lammps':9s} skipped ({skipped['lammps']})")


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
