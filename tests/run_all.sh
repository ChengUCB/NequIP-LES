#!/usr/bin/env bash
# Smoke-test NequIP-LES and Allegro-LES end to end.
#
# Every run trains for a couple of epochs on a handful of frames and then runs
# the test phase, which is where ToggleLESCallback turns BEC inference on. The
# point is that nothing errors and that BECs come out -- not accuracy.
#
# 12 configurations, one yaml each in configs/ (the LES settings are written out
# in the files rather than overridden here, so each config is a complete,
# copy-able example):
#
#   backbone   : nequip | allegro
#   periodicity: water (periodic) | dipep (non-periodic)
#   LES path   : compiled = vectorized Ewald + compile_mode: compile
#                eager    = vectorized Ewald, compile_mode: eager
#                legacy   = is_periodic/N_max absent -> loop-based Ewald
#
# All of them enable every long-range term (dipoles, quadrupoles, induced
# charges, induced dipoles, anisotropic polarizability): if something is going
# to break, it should break here.
#
# Usage:
#   ./run_all.sh                  # all 12
#   ./run_all.sh nequip           # only configs whose name contains "nequip"
#   ./run_all.sh water_compiled   # ... or any other substring
#   KEEP_OUTPUTS=1 ./run_all.sh   # keep predictions/ and outputs/ afterwards
set -uo pipefail
cd "$(dirname "$0")"

FILTER="${1:-}"
PY=$(command -v python)

# GPU only when CUDA is really present. Deliberately not lightning's "auto",
# which selects MPS on Macs, where NequIP's float64 buffers fail.
if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    ACCELERATOR=gpu
else
    ACCELERATOR=cpu
fi
echo "torch $("$PY" -c 'import torch; print(torch.__version__)')  |  accelerator: $ACCELERATOR"

mkdir -p predictions outputs
PASSED=(); FAILED=()

for cfg in nequip_water nequip_dipep allegro_water allegro_dipep; do
  for mode in compiled eager legacy; do
    tag="${cfg}_${mode}"
    [[ -n "$FILTER" && "$tag" != *"$FILTER"* ]] && continue

    printf '\n=== %-30s ===\n' "$tag"
    if nequip-train -cn "$tag" --config-dir configs \
            "hydra.run.dir=outputs/$tag" \
            "++trainer.accelerator=$ACCELERATOR" > "outputs/$tag.log" 2>&1; then
        # trained -- but it only counts if BEC inference produced values
        if grep -q "LES_BEC" "predictions/${tag}_dataset0.xyz" 2>/dev/null; then
            echo "PASS  (trained + BEC written)"; PASSED+=("$tag")
        else
            echo "FAIL  (trained, but no LES_BEC in predictions)"; FAILED+=("$tag")
        fi
    else
        echo "FAIL  (see outputs/$tag.log)"
        grep -iE "Error:|Exception|RuntimeError|TypeError|ValueError|NotImplemented" \
            "outputs/$tag.log" | head -3 | sed 's/^/      /'
        FAILED+=("$tag")
    fi
  done
done

echo
echo "==================== SUMMARY ===================="
for t in ${PASSED[@]+"${PASSED[@]}"}; do echo "  PASS  $t"; done
for t in ${FAILED[@]+"${FAILED[@]}"}; do echo "  FAIL  $t"; done
echo "================================================="
echo "${#PASSED[@]} passed, ${#FAILED[@]} failed"

if [[ -z "${KEEP_OUTPUTS:-}" ]]; then
    # Leave only the inputs behind. A keep-list rather than a list of things to
    # delete, so whatever the training stack happens to drop here -- outputs/,
    # predictions/, lightning_logs/, .hydra/, wandb/ -- goes away too.
    find . -mindepth 1 -maxdepth 1 \
        ! -name configs ! -name data ! -name run_all.sh ! -name README.md \
        -exec rm -rf {} +
    echo "(cleaned generated files; KEEP_OUTPUTS=1 to keep them)"
fi

[[ ${#FAILED[@]} -eq 0 ]] || exit 1
