"""Every artefact run_gpu.sh / run_compile.sh left behind, evaluated and cross-checked.

    KEEP_OUTPUTS=1 ./run_gpu.sh          # produces compiled/ and ckpt/
    python run_inference_matrix.py       # then this

Those scripts prove each export *succeeds*. They do not prove the exports agree: a wrong
unit, a wrong type mapping, or a cell that never reached the graph all produce a perfectly
valid artefact that returns different numbers. This script closes that gap -- it loads every
artefact through the interface it was built for and differences them against each other,
per model.

Nothing is exported here (except optionally from a `.nequip.zip`); the artefacts are reused
as they are. To export fresh from a checkpoint instead, use check_consistency.py.

    artefact                                       engine
    <tag>_ase_<mode>_<dev>[_extra].nequip.pt2      ASE calculator
    <tag>_batch_<mode>_<dev>[_extra].nequip.pt2    torch-sim
    <tag>_pair_nequip_...                          LAMMPS pair_style nequip    ($LMP)
    <tag>_pair_allegro_...                         LAMMPS pair_style allegro   ($LMP)
    <tag>_mliap[...].nequip.lmp.pt                 LAMMPS pair_style mliap     ($LMP_MLIAP)
    <tag>.nequip.zip                               compiled to ASE, then ASE

`extra` is `tf32` or an acceleration modifier. Those are expected to differ slightly -- they
change the arithmetic on purpose -- so they are checked against a looser tolerance and
labelled in the output rather than being held to the strict one.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read

import engines

HERE = Path(__file__).resolve().parent

ARTEFACT_RE = re.compile(
    r"^(?P<tag>.+?)_(?P<target>" + "|".join(engines.TARGETS) + r")"
    r"_(?P<mode>" + "|".join(engines.MODES) + r")"
    r"_(?P<dev>cuda|cpu)(?:_(?P<extra>.+))?$"
)
MLIAP_RE = re.compile(r"^(?P<tag>.+?)_mliap(?:_(?P<extra>.+))?$")

# which frame to feed a given model; the species have to match what it was trained on
STRUCTURES = {
    "water": "data/water_train.xyz",
    # the dummy-cell copy: LES ignores the cell for a non-periodic model, but LAMMPS needs
    # a box and nequip needs a finite volume for the stress
    "dipep": "data/dipep_test_dummycell.xyz",
}


def structure_for(tag, override=None):
    if override:
        return override
    for key, path in STRUCTURES.items():
        if key in tag:
            return path
    raise SystemExit(f"no structure known for tag '{tag}'; pass --structure")


def find_checkpoint(tag, ckpt_dir):
    hits = sorted(Path(ckpt_dir).glob(f"{tag}/**/*.ckpt"))
    return hits[0] if hits else None


def discover(compiled_dir):
    """Group the artefacts in `compiled_dir` by model tag."""
    models = {}

    def slot(tag):
        return models.setdefault(tag, {"exports": [], "mliap": [], "package": None})

    for path in sorted(Path(compiled_dir).iterdir()):
        name = path.name
        if name.endswith(".nequip.pt2") or name.endswith(".nequip.pth"):
            stem = name.replace(".nequip.pt2", "").replace(".nequip.pth", "")
            m = ARTEFACT_RE.match(stem)
            if m:
                slot(m["tag"])["exports"].append((path, m.groupdict()))
        elif name.endswith(".nequip.lmp.pt"):
            m = MLIAP_RE.match(name.replace(".nequip.lmp.pt", ""))
            if m:
                slot(m["tag"])["mliap"].append((path, m.groupdict()))
        elif name.endswith(".nequip.zip"):
            slot(name.replace(".nequip.zip", ""))["package"] = path
    return models


def compile_cmd():
    fix = HERE / "compile_tf32fix.py"
    return [sys.executable, str(fix)] if fix.exists() else ["nequip-compile"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="compiled", help="artefact directory (default compiled)")
    ap.add_argument("--ckpt-dir", default="ckpt", help="checkpoint directory (default ckpt)")
    ap.add_argument("--filter", default="", help="only models whose tag contains this")
    ap.add_argument("--structure", default=None, help="override the structure file")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--etol", type=float, default=1e-4, help="eV/atom (default 1e-4)")
    ap.add_argument("--ftol", type=float, default=1e-3, help="eV/A (default 1e-3)")
    # tf32 and the fused kernels change float32 arithmetic deliberately; holding them to the
    # strict tolerance would report a difference that is the feature working
    ap.add_argument("--loose-etol", type=float, default=1e-2, help="eV/atom for tf32/modifiers")
    ap.add_argument("--loose-ftol", type=float, default=1e-1, help="eV/A for tf32/modifiers")
    ap.add_argument("--no-package", action="store_true", help="skip the .nequip.zip route")
    args = ap.parse_args()

    compiled = Path(args.dir)
    if not compiled.is_dir():
        raise SystemExit(f"{compiled} not found. Run: KEEP_OUTPUTS=1 ./run_gpu.sh")

    models = discover(compiled)
    if args.filter:
        models = {t: v for t, v in models.items() if args.filter in t}
    if not models:
        raise SystemExit(f"no artefacts in {compiled} (filter={args.filter!r})")

    lmp = os.environ.get("LMP")
    lmp_mliap = os.environ.get("LMP_MLIAP")
    print(f"artefacts  : {compiled.resolve()}")
    print(f"models     : {len(models)}")
    print(f"device     : {args.device}")
    print(f"$LMP       : {lmp or '(unset -- pair styles will be skipped)'}")
    print(f"$LMP_MLIAP : {lmp_mliap or '(unset -- ML-IAP will be skipped)'}")

    all_rows, mismatches, skipped = [], [], []

    for tag in sorted(models):
        art = models[tag]
        ckpt = find_checkpoint(tag, args.ckpt_dir)
        if ckpt is None:
            print(f"\n=== {tag} ===\n  no checkpoint under {args.ckpt_dir}/{tag}; skipped")
            skipped.append(f"{tag} (no checkpoint -- type names unknown)")
            continue

        species = engines.type_names(ckpt)
        allegro = engines.backbone(ckpt)[0] == "allegro"
        struct = structure_for(tag, args.structure)
        atoms = read(struct, index=0)
        n = len(atoms)

        print(f"\n=== {tag} ===")
        print(f"  structure {struct}  ({n} atoms, pbc={atoms.pbc.all()})  types {species}")

        results = {}   # label -> (energy, forces, loose?)

        def attempt(label, fn, loose=False):
            try:
                e, f = fn()
            except Exception as exc:                      # noqa: BLE001 - report, continue
                msg = f"{type(exc).__name__}: {exc}"
                print(f"  {label:52s} skipped ({msg[:60]})")
                skipped.append(f"{tag}/{label}: {msg[:120]}")
                return
            results[label] = (e, f, loose)
            print(f"  {label:52s} E={e:16.8f}  |F|max={np.abs(f).max():.8f}")

        for path, info in art["exports"]:
            extra = info["extra"] or ""
            loose = bool(extra)          # tf32 or an acceleration modifier
            label = f"{info['target']}/{info['mode']}/{info['dev']}" + (f"/{extra}" if extra else "")
            target = info["target"]

            if target == "ase":
                attempt(label, lambda p=path: engines.eval_ase(p, atoms, args.device, species), loose)
            elif target == "batch":
                attempt(label, lambda p=path: engines.eval_torchsim(p, atoms, args.device, species), loose)
            elif target.startswith("pair_"):
                if not lmp:
                    print(f"  {label:52s} skipped ($LMP unset)")
                    continue
                lines = engines.lammps_pair_lines(path.resolve(), species, allegro)
                wd = compiled / "run" / tag / label.replace("/", "_")
                attempt(label, lambda l=lines, w=wd: engines.eval_lammps(lmp, atoms, species, w, l)[:2], loose)

        for path, info in art["mliap"]:
            extra = info["extra"] or ""
            label = "mliap" + (f"/{extra}" if extra else "")
            if not lmp_mliap:
                print(f"  {label:52s} skipped ($LMP_MLIAP unset)")
                continue
            lines = engines.lammps_mliap_lines(path.resolve(), species)
            wd = compiled / "run" / tag / label.replace("/", "_")
            attempt(label, lambda l=lines, w=wd: engines.eval_lammps(lmp_mliap, atoms, species, w, l,
                                             engines.MLIAP_KOKKOS_ARGS)[:2],
                    bool(extra))

        if art["package"] and not args.no_package:
            # a packaged model is a deployment route of its own: compile it, then run it
            out = compiled / f"{tag}_frompackage_ase.nequip.pt2"
            if not out.exists():
                cmd = compile_cmd() + [str(art["package"]), str(out), "--device", args.device,
                                       "--mode", "aotinductor", "--target", "ase"]
                log = compiled / f"{tag}_frompackage.log"
                with open(log, "w") as fh:
                    subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
            if out.exists():
                attempt("package->ase", lambda p=out: engines.eval_ase(p, atoms, args.device, species))
            else:
                print(f"  {'package->ase':52s} skipped (compile from package failed)")
                skipped.append(f"{tag}/package->ase: compile failed")

        # ---- compare everything in this model against one reference ----
        if len(results) < 2:
            print("  (fewer than two engines ran; nothing to compare)")
            continue

        strict = [k for k, v in results.items() if not v[2]]
        ref = next((k for k in strict if k.startswith("ase/")), (strict or sorted(results))[0])
        ref_e, ref_f, _ = results[ref]

        print(f"  --- vs {ref} ---")
        for label in sorted(results):
            if label == ref:
                continue
            e, f, loose = results[label]
            de = abs(e - ref_e) / n
            df = float(np.abs(f - ref_f).max()) if f.shape == ref_f.shape else float("inf")
            etol = args.loose_etol if loose else args.etol
            ftol = args.loose_ftol if loose else args.ftol
            ok = de <= etol and df <= ftol
            mark = "ok" if ok else "MISMATCH"
            note = " [loose]" if loose else ""
            print(f"  {label:52s} dE/atom={de:10.3e}  dFmax={df:10.3e}  {mark}{note}")
            all_rows.append((tag, label, de, df, ok))
            if not ok:
                mismatches.append(f"{tag}/{label}: dE/atom={de:.3e} dFmax={df:.3e}")

    print("\n==================== SUMMARY ====================")
    print(f"  compared : {len(all_rows)} artefact pairs")
    print(f"  agreeing : {sum(1 for r in all_rows if r[4])}/{len(all_rows)}")
    if skipped:
        print(f"  skipped  : {len(skipped)}")
        for s in skipped:
            print(f"      {s}")
    for m in mismatches:
        print(f"  MISMATCH  {m}")
    print("=================================================")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
