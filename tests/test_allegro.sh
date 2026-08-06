#!/usr/bin/env bash
# Every deployment path a trained ALLEGRO-backbone model can take, in order.
#
#   ./test_allegro.sh path/to/model.ckpt [outdir]
#
# Each step prints the exact command it runs, so you can copy any single one and
# investigate it on its own. Steps that need something you do not have installed
# (openequivariance, cuequivariance, LAMMPS ML-IAP, torch-sim) are skipped with a
# reason rather than failed.
#
# Reference:
#   https://nequip.readthedocs.io/en/latest/integrations/ase.html
#   https://nequip.readthedocs.io/projects/allegro/en/latest/guide/triton.html
#   https://nequip.readthedocs.io/projects/allegro/en/latest/guide/cuequivariance.html
#   https://nequip.readthedocs.io/projects/allegro/en/latest/guide/lammps.html
set -uo pipefail

CKPT="${1:-}"
OUT="${2:-integration_out}"
[[ -z "$CKPT" || ! -f "$CKPT" ]] && { echo "usage: $0 <model.ckpt> [outdir]"; exit 2; }
CKPT=$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")
HERE=$(cd "$(dirname "$0")" && pwd)
# Stay in the caller's directory: nequip-compile rebuilds the datamodule from the
# checkpoint's config, whose data paths are usually relative to where you trained.
# Only the artefacts go to $OUT.
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

PY=$(command -v python)
DEVICE=$("$PY" -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
TS_OK=$("$PY" -c "
from packaging.version import parse; import torch
print('yes' if parse(torch.__version__.split('+')[0]) < parse('2.10') else 'no')")
has() { "$PY" -c "import $1" 2>/dev/null; }

# On torch 2.13 every nequip-compile fails on CUDA: torch.export reads the legacy
# cuDNN TF32 flag, which cannot represent the "not tf32" that nequip sets. The wrapper
# only makes that read answer instead of raising.
COMPILE=(nequip-compile)
[[ -f "$HERE/compile_tf32fix.py" ]] && COMPILE=("$PY" "$HERE/compile_tf32fix.py")

echo "checkpoint : $CKPT"
echo "device     : $DEVICE"
echo "torch      : $("$PY" -c 'import torch; print(torch.__version__)')  (torchscript: $TS_OK)"
echo "output dir : $OUT"
echo "run from   : $(pwd)  (data paths in the checkpoint config resolve from here)"

PASS=(); FAIL=(); SKIP=()
step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
run() {   # run <label> <command...>
    local label="$1"; shift
    printf '   $ %s\n' "$*"
    if "$@" > "$OUT/log_${label}.txt" 2>&1; then
        echo "   -> ok"; PASS+=("$label")
    else
        echo "   -> FAILED (see $(pwd)/log_${label}.txt)"
        grep -iE "error|exception" "$OUT/log_${label}.txt" | head -2 | sed 's/^/      /'
        FAIL+=("$label")
    fi
}
skip() { echo "   -> skipped: $2"; SKIP+=("$1"); }

# ---------------------------------------------------------------- packaging ----
step "1. nequip-package -- a portable model file, independent of the source tree"
run package nequip-package build "$CKPT" "$OUT"/model.nequip.zip

# ------------------------------------------------------------------- export ----
step "2. nequip-compile -- ASE target"
run ase_aoti "${COMPILE[@]}" "$CKPT" "$OUT"/ase.nequip.pt2 \
    --device "$DEVICE" --mode aotinductor --target ase
if [[ "$TS_OK" == yes ]]; then
    run ase_ts "${COMPILE[@]}" "$CKPT" "$OUT"/ase.nequip.pth \
        --device "$DEVICE" --mode torchscript --target ase
else
    skip ase_ts "nequip refuses TorchScript on torch >= 2.10"
fi

step "3. nequip-compile -- LAMMPS pair_allegro target"
# NOTE: pair_allegro passes no cell, so a PERIODIC LES model is refused here by design
# (the reciprocal-space sum has no lattice). That refusal is correct; use pair_nequip or
# ASE for periodic LES models. A short-range or non-periodic model exports fine.
# `pair_allegro` is registered by the allegro package through its `nequip.extension`
# entry point; if this row fails with "invalid choice", that registration is broken --
# not something to skip past.
run pair_aoti "${COMPILE[@]}" "$CKPT" "$OUT"/pair.nequip.pt2 \
    --device "$DEVICE" --mode aotinductor --target pair_allegro
echo "   (a PERIODIC LES model must use pair_nequip instead: pair_allegro passes no cell)"
run pair_nequip_aoti "${COMPILE[@]}" "$CKPT" "$OUT"/pair_nequip.nequip.pt2 \
    --device "$DEVICE" --mode aotinductor --target pair_nequip
if [[ "$TS_OK" == yes ]]; then
    run pair_ts "${COMPILE[@]}" "$CKPT" "$OUT"/pair.nequip.pth \
        --device "$DEVICE" --mode torchscript --target pair_allegro
else
    skip pair_ts "nequip refuses TorchScript on torch >= 2.10"
fi

step "4. nequip-compile -- batch target (batched inference)"
run batch_aoti "${COMPILE[@]}" "$CKPT" "$OUT"/batch.nequip.pt2 \
    --device "$DEVICE" --mode aotinductor --target batch

step "5. nequip-compile from the PACKAGED file rather than the checkpoint"
if [[ -f "$OUT"/model.nequip.zip ]]; then
    run ase_from_package "${COMPILE[@]}" "$OUT"/model.nequip.zip "$OUT"/ase_pkg.nequip.pt2 \
        --device "$DEVICE" --mode aotinductor --target ase
else
    skip ase_from_package "packaging failed"
fi

# ------------------------------------------------------------ accelerations ----
step "6. accelerations -- Triton contracter (CUDA only; triton ships with torch)"
if [[ "$DEVICE" != cuda ]]; then
    skip triton "not a CUDA device"
elif ! has triton; then
    skip triton "no triton in this environment"
else
    run triton_ase "${COMPILE[@]}" "$CKPT" "$OUT"/ase_triton.nequip.pt2 \
        --device cuda --mode aotinductor --target ase --modifiers enable_TritonContracter
    run triton_pair "${COMPILE[@]}" "$CKPT" "$OUT"/pair_triton.nequip.pt2 \
        --device cuda --mode aotinductor --target pair_allegro --modifiers enable_TritonContracter
fi

step "7. accelerations -- CuEquivarianceContracter (CUDA only, needs cuequivariance-torch)"
if [[ "$DEVICE" != cuda ]]; then
    skip cueq "not a CUDA device"
elif ! has cuequivariance_torch; then
    skip cueq "pip install cuequivariance-torch cuequivariance-ops-torch-cu12"
else
    run cueq_ase "${COMPILE[@]}" "$CKPT" "$OUT"/ase_cueq.nequip.pt2 \
        --device cuda --mode aotinductor --target ase --modifiers enable_CuEquivarianceContracter
    run cueq_pair "${COMPILE[@]}" "$CKPT" "$OUT"/pair_cueq.nequip.pt2 \
        --device cuda --mode aotinductor --target pair_allegro --modifiers enable_CuEquivarianceContracter
fi


# -------------------------------------------------------------------- LAMMPS ---
step "9. LAMMPS ML-IAP -- KNOWN GAP, not attempted"
cat <<'NOTE'
   The ML-IAP wrapper passes `edge_vectors` but neither absolute positions nor the cell,
   so a LES model raises KeyError: 'pos' -- the Ewald sum has nothing to sum over. Its
   run-time torch.compile also fails on torch 2.13 inside nequip's cutoff function. Both
   are upstream issues in a beta integration; see
   https://nequip.readthedocs.io/en/latest/integrations/lammps/mliap.html
NOTE
SKIP+=(mliap)

# ------------------------------------------------------------------ run it -----
step "10. ASE -- actually run the compiled model and compare with the checkpoint"
cat > "$OUT/check_ase.py" <<'PYEOF'
"""Run the compiled model once and check the prediction is finite.

The structure is built from the model's own type names, read out of the checkpoint --
a physically meaningless but valid configuration, which is all a smoke test needs.
"""
import sys

import numpy as np
import torch
from ase import Atoms

compiled, device, ckpt = sys.argv[1], sys.argv[2], sys.argv[3]
if "oeq" in compiled:
    import openequivariance  # noqa: F401  (must precede loading, see the docs)
if "cueq" in compiled:
    import cuequivariance_torch  # noqa: F401
if "triton" in compiled:
    pass  # triton kernels need no explicit import

from nequip.integrations.ase import NequIPCalculator  # noqa: E402


def type_names(path):
    hp = torch.load(path, map_location="cpu", weights_only=False).get("hyper_parameters", {})
    stack = [hp]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            if "type_names" in o:
                return list(o["type_names"])
            stack.extend(o.values())
    raise SystemExit("could not find type_names in the checkpoint")


names = type_names(ckpt)
print("model type names:", names)

# a loose cubic arrangement of those species, well inside a big cell
n = max(8, len(names) * 4)
rng = np.random.default_rng(0)
side = 12.0
atoms = Atoms(
    symbols=[names[i % len(names)] for i in range(n)],
    positions=rng.uniform(1.0, side - 1.0, size=(n, 3)),
    cell=np.eye(3) * side,
    pbc=True,
)

atoms.calc = NequIPCalculator.from_compiled_model(
    compile_path=compiled, device=device, chemical_species_to_atom_type_map=True
)
e = atoms.get_potential_energy()
f = atoms.get_forces()
print(f"energy      = {e:.6f} eV")
print(f"|forces|max = {np.abs(f).max():.6f} eV/A")
assert np.isfinite(e) and np.isfinite(f).all(), "non-finite prediction"
print("finite: OK")
PYEOF
if [[ -f "$OUT"/ase.nequip.pt2 ]]; then
    run ase_inference "$PY" "$OUT"/check_ase.py "$OUT"/ase.nequip.pt2 "$DEVICE" "$CKPT"
    [[ -f "$OUT/log_ase_inference.txt" ]] && sed -n '1,4p' "$OUT/log_ase_inference.txt" | sed 's/^/      /'
else
    skip ase_inference "ASE export failed"
fi

step "11. torch-sim -- a few MD steps through the same compiled model"
if ! has torch_sim; then
    skip torchsim "pip install torch-sim"
elif [[ ! -f "$OUT"/ase.nequip.pt2 ]]; then
    skip torchsim "ASE export failed"
else
    cat > "$OUT/check_torchsim.py" <<'PYEOF'
import sys
import torch
from ase.build import bulk
from nequip.integrations.torchsim import NequIPTorchSimCalc

compiled, device = sys.argv[1], sys.argv[2]
model = NequIPTorchSimCalc.from_compiled_model(compile_path=compiled, device=device)
print("loaded:", type(model).__name__)
PYEOF
    run torchsim "$PY" "$OUT"/check_torchsim.py "$OUT"/ase.nequip.pt2 "$DEVICE"
fi

# ------------------------------------------------------------------ summary ----
printf '\n\033[1m==================== SUMMARY ====================\033[0m\n'
for t in ${PASS[@]+"${PASS[@]}"}; do echo "  ok      $t"; done
for t in ${SKIP[@]+"${SKIP[@]}"}; do echo "  skip    $t"; done
for t in ${FAIL[@]+"${FAIL[@]}"}; do echo "  FAILED  $t"; done
echo "================================================="
echo "${#PASS[@]} ok, ${#SKIP[@]} skipped, ${#FAIL[@]} failed   (artefacts and logs in $OUT)"
[[ ${#FAIL[@]} -eq 0 ]] || exit 1
