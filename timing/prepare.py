"""One-time setup for the timing benchmark: the base cell, the N_max variants, the artefacts.

    python prepare.py

Everything it produces is skipped on a second run, so it is safe to re-run after adding a
checkpoint. Nothing here is timed -- the point is that no timing run ever pays for a
relaxation, a model rebuild or an export.
"""

import argparse
import shutil
import subprocess
import sys

import torch
from ase.io import read, write

import common as C

sys.path.insert(0, str(C.TESTS))
import engines  # noqa: E402  (after the path insert)


# ------------------------------------------------------------------- base cell ----
def make_base_cell(force=False):
    """Frame 0 of the test water set, relaxed once with the NequIP SR model.

    One structure is used by every row: a timing comparison across engines is meaningless if
    they are not looking at the same atoms.
    """
    if C.BASE_XYZ.exists() and not force:
        atoms = read(C.BASE_XYZ, index=0)
        print(f"  base cell     reusing {C.BASE_XYZ.name}  "
              f"({len(atoms)} atoms, {atoms.cell.lengths()[0]:.3f} A)")
        return atoms

    src = C.TESTS / "data" / "water_train.xyz"
    atoms = read(src, index=0)
    sr = C.MODELS_DIR / "nequip_sr.ckpt"
    if sr.exists():
        from nequip.integrations.ase import NequIPCalculator
        from ase.optimize import BFGS
        atoms.calc = NequIPCalculator._from_saved_model(
            str(sr), device="cuda" if torch.cuda.is_available() else "cpu",
            chemical_species_to_atom_type_map=True, allow_tf32=False,
            neighborlist_backend=C.NL_BACKEND)
        BFGS(atoms, logfile=str(C.HERE / "relax.log")).run(fmax=0.03, steps=200)
        atoms.calc = None
        print(f"  base cell     relaxed with nequip_sr.ckpt -> {C.BASE_XYZ.name}")
    else:
        print(f"  base cell     nequip_sr.ckpt absent, using {src.name} frame 0 unrelaxed")
    write(C.BASE_XYZ, atoms)
    return atoms


# ------------------------------------------------- N_max-patched checkpoints ----
def patch_n_max(ckpt, n_max):
    """A copy of `ckpt` with `les_args.N_max` overridden.

    Safe to do: the k-grid `N_max` builds is registered with `persistent=False`, so it is not
    in the state dict and not learned. `dl` sets the physical cutoff; `N_max` only has to be
    large enough to reach it.
    """
    out = C.DERIVED_DIR / f"{ckpt.stem}_nmax{n_max}.ckpt"
    if out.exists():
        return out
    C.DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)

    patched = 0
    stack = [obj.get("hyper_parameters", {})]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if "les_args" in cur and isinstance(cur["les_args"], dict):
                cur["les_args"]["N_max"] = n_max
                patched += 1
            stack.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            stack.extend(v for v in cur if isinstance(v, (dict, list)))
    if patched == 0:
        raise SystemExit(f"no les_args block found in {ckpt.name}; is it an SR model?")

    torch.save(obj, out)
    print(f"  derived       {out.name}  (N_max={n_max}, {patched} les_args block(s))")
    return out


# ------------------------------------------------------------------- artefacts ----
def export(ckpt, target, modifier=None, n_max=None, device="cuda"):
    out = C.artefact_path(ckpt.stem.replace(f"_nmax{n_max}", "") if n_max else ckpt.stem,
                          target, modifier, n_max)
    if out.exists():
        print(f"  artefact      reusing {out.name}")
        return out
    C.ARTEFACTS.mkdir(parents=True, exist_ok=True)
    cmd = C.compile_cmd() + [str(ckpt), str(out),
                             "--device", device, "--mode", "aotinductor",
                             "--target", target]
    if modifier:
        # positionals first: `--modifiers` takes nargs="+" and would swallow them
        cmd += ["--modifiers", modifier]
    log = C.ARTEFACTS / (out.stem + ".log")
    with open(log, "w") as fh:
        fh.write("# " + " ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not out.exists():
        print(f"  artefact      FAILED {out.name}  (see {log.name})")
        return None
    print(f"  artefact      {out.name}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--relax", action="store_true",
                    help="re-relax the base cell even if water.xyz exists")
    ap.add_argument("--policies", default=",".join(C.POLICIES),
                    help="which N_max policies to prepare artefacts for")
    ap.add_argument("--models", default="",
                    help="substring filter on the checkpoint name, as MODELS= in run_timing.sh")
    ap.add_argument("--engines", default="",
                    help="space-separated engine names; only their artefacts are built, "
                         "as ENGINES= in run_timing.sh")
    args = ap.parse_args()
    policies = args.policies.split(",")
    want_engines = set(args.engines.split()) if args.engines else None

    models = [m for m in C.model_files() if args.models in m[0].stem]
    if not models:
        raise SystemExit(
            f"no checkpoints in {C.MODELS_DIR} matching --models {args.models!r}. Expected "
            f"<backbone>_<variant>.ckpt with backbone in (nequip, allegro) and variant in "
            f"{C.VARIANTS}.")

    print("== base cell ==")
    atoms = make_base_cell(force=args.relax)
    lengths = atoms.cell.lengths()

    print("\n== sizes ==")
    for nxyz in C.SIZES:
        n = len(atoms) * nxyz[0] * nxyz[1] * nxyz[2]
        need = C.n_max_for(lengths, nxyz)
        print(f"  {nxyz[0]} {nxyz[1]} {nxyz[2]}   {n:6d} atoms   "
              f"N_max needed {need:3d}  (K = {(2 * need + 1) ** 3})")

    print("\n== artefacts, nmax-fixed (N_max as trained) ==")
    for ckpt, backbone, variant in models:
        print(f"-- {ckpt.stem} --")
        wanted = {(t, m) for n, t, m, _ in C.ENGINES[backbone]
                  if t is not None and (want_engines is None or n in want_engines)}
        if not wanted:
            print("  (no artefacts needed for the selected engines)")
        for target, modifier in sorted(wanted, key=lambda x: (x[0], x[1] or "")):
            export(ckpt, target, modifier, device=args.device)

    if "scaled" in policies:
        print("\n== artefacts, nmax-scaled (LES models only, two engines) ==")
        needed = sorted({C.n_max_for(lengths, s) for s in C.SIZES})
        print(f"  distinct N_max values: {needed}")
        for ckpt, backbone, variant in models:
            if variant == "sr":
                print(f"-- {ckpt.stem} -- skipped: no Ewald sum, N_max cannot affect it")
                continue
            print(f"-- {ckpt.stem} --")
            for n_max in needed:
                patched = patch_n_max(ckpt, n_max)
                for name, target, modifier, _ in C.ENGINES[backbone]:
                    if (name in C.SCALED_ENGINES and target is not None
                            and (want_engines is None or name in want_engines)):
                        export(patched, target, modifier, n_max=n_max, device=args.device)

    print("\nready. next: ./run_timing.sh")


if __name__ == "__main__":
    main()
