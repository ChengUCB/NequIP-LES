#!/usr/bin/env bash
# Smoke-test NequIP-LES and Allegro-LES end to end.
#
# Every run trains for a couple of epochs on a handful of frames. The point is
# that nothing errors -- not accuracy. BEC inference is a separate step, so that
# a training failure and a BEC failure cannot be confused: see run_bec.sh.
#
# 20 configurations, one yaml each in configs/ (the LES settings are written out
# in the files rather than overridden here, so each config is a complete,
# copy-able example):
#
#   backbone   : nequip | allegro
#   periodicity: water (periodic) | dipep (non-periodic)
#   LES path   : compiled = vectorized Ewald + compile_mode: compile
#                eager    = vectorized Ewald, compile_mode: eager
#                legacy   = is_periodic/N_max absent -> loop-based Ewald
#                sr_compiled | sr_eager = no LES at all, for attribution
#
# All of them enable every long-range term (dipoles, quadrupoles, induced
# charges, induced dipoles, anisotropic polarizability): if something is going
# to break, it should break here.
#
# Usage:
#   ./run_all.sh                  # all 20
#   ./run_all.sh nequip           # only configs whose name contains "nequip"
#   ./run_all.sh water_compiled   # ... or any other substring
#   KEEP_OUTPUTS=1 ./run_all.sh   # keep outputs/ afterwards
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

mkdir -p outputs
PASSED=(); FAILED=()

# `sr_*` are the same backbones with LES removed. They run every time on purpose:
# when something breaks, the SR row says immediately whether it is a LES problem or
# nequip's -- twice already it was nequip's.
for cfg in nequip_water nequip_dipep allegro_water allegro_dipep; do
  for mode in compiled eager legacy sr_compiled sr_eager; do
    tag="${cfg}_${mode}"
    [[ -n "$FILTER" && "$tag" != *"$FILTER"* ]] && continue

    printf '\n=== %-30s ===\n' "$tag"

    # train_probed.py is nequip-train plus a report of whether a graph was really
    # traced. Checking the config is not enough: CompileGraphModel.forward falls
    # back to eager for batches with fewer than 2 frames, so a `compiled` config
    # can train without ever compiling -- which is how this suite once passed
    # while real training crashed.
    if "$PY" train_probed.py -cn "$tag" --config-dir configs \
            "hydra.run.dir=outputs/$tag" \
            "++trainer.logger.save_dir=outputs/$tag" \
            "++trainer.accelerator=$ACCELERATOR" > "outputs/$tag.log" 2>&1; then
        traced=$(grep -c "^TRACED" "outputs/$tag.log")
        # A NaN loss raises nothing, so exiting 0 is not enough: a compiled
        # non-periodic run trained "successfully" while every gradient was NaN,
        # and only the export caught it. Read the logged metrics.
        csv=$(find "outputs/$tag" -name metrics.csv 2>/dev/null | head -1)
        if [[ -n "$csv" ]] && grep -qi "nan" "$csv"; then
            echo "FAIL  (trained, but the logged loss is NaN)"; FAILED+=("$tag"); continue
        fi
        if [[ "$mode" == *compiled && "$traced" -eq 0 ]]; then
            echo "FAIL  (trained, but nothing was compiled)"; FAILED+=("$tag")
        elif [[ "$mode" != *compiled && "$traced" -ne 0 ]]; then
            echo "FAIL  (compiled although the config asks for eager)"; FAILED+=("$tag")
        else
            echo "PASS  (trained; $(grep -hE '^(TRACED|NOT-TRACED)' "outputs/$tag.log" | tail -1))"
            PASSED+=("$tag")
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
        ! -name configs ! -name data ! -name '*.sh' ! -name '*.py' ! -name README.md \
        -exec rm -rf {} +
    echo "(cleaned generated files; KEEP_OUTPUTS=1 to keep them)"
fi

[[ ${#FAILED[@]} -eq 0 ]] || exit 1
