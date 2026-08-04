#!/usr/bin/env bash
# Check that BEC inference works for every trained model.
#
# Training and BEC are separate steps: run_all.sh only trains, and this script
# loads each resulting checkpoint through `nequip.model.ModelFromCheckpoint` and
# runs the test phase, where ToggleLESCallback turns BEC on. It trains a model
# itself if no checkpoint is there yet, so it can be run on its own.
#
# A run passes when `nequip-train` exits 0 AND the written xyz actually contains
# an LES_BEC column -- a run that silently stops producing BECs is a failure, not
# a pass. Accuracy is not checked: these are two-epoch models.
#
# Usage:
#   ./run_bec.sh                  # all 12
#   ./run_bec.sh nequip           # only configs whose name contains "nequip"
#   KEEP_OUTPUTS=1 ./run_bec.sh
set -uo pipefail
cd "$(dirname "$0")"

FILTER="${1:-}"
PY=$(command -v python)

if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    ACCELERATOR=gpu
else
    ACCELERATOR=cpu
fi
echo "torch $("$PY" -c 'import torch; print(torch.__version__)')  |  accelerator: $ACCELERATOR"

mkdir -p ckpt predictions
PASSED=(); FAILED=()

train_once() {   # $1 tag -> prints the checkpoint path on stdout
    local tag="$1"
    if [[ -z "$(find "ckpt/$tag" -name '*.ckpt' 2>/dev/null | head -1)" ]]; then
        echo "  training $tag ..." >&2
        nequip-train -cn "$tag" --config-dir configs \
            "hydra.run.dir=ckpt/$tag/run" \
            "++trainer.logger.save_dir=ckpt/$tag" \
            "++trainer.accelerator=$ACCELERATOR" \
            > "ckpt/$tag/train.log" 2>&1 \
            || { echo "  TRAINING FAILED (ckpt/$tag/train.log)" >&2; return 1; }
    fi
    find "ckpt/$tag" -name '*.ckpt' | head -1
}

for cfg in nequip_water nequip_dipep allegro_water allegro_dipep; do
  # the BEC config follows the data, not the backbone
  case "$cfg" in
      *_water) bec=bec_water ;;
      *_dipep) bec=bec_dipep ;;
  esac
  for mode in compiled eager legacy; do
    tag="${cfg}_${mode}"
    [[ -n "$FILTER" && "$tag" != *"$FILTER"* ]] && continue

    printf '\n=== %-30s ===\n' "$tag"
    mkdir -p "ckpt/$tag"
    CK=$(train_once "$tag") || { FAILED+=("$tag/train"); continue; }

    # the checkpoint path is quoted for hydra: lightning's filenames contain '='
    # (epoch=1-step=12.ckpt), which the override parser would otherwise choke on
    if nequip-train -cn "$bec" --config-dir configs \
            "hydra.run.dir=ckpt/$tag/bec" \
            "++trainer.accelerator=$ACCELERATOR" \
            "++training_module.model.checkpoint_path='$CK'" \
            "++trainer.callbacks.0.out_file=predictions/$tag" \
            > "ckpt/$tag/bec.log" 2>&1; then
        if grep -q "LES_BEC" "predictions/${tag}_dataset0.xyz" 2>/dev/null; then
            echo "PASS  (BEC written)"; PASSED+=("$tag")
        else
            echo "FAIL  (test ran, but no LES_BEC in predictions)"; FAILED+=("$tag")
        fi
    else
        echo "FAIL  (see ckpt/$tag/bec.log)"
        grep -iE "Error:|Exception|RuntimeError|TypeError|ValueError|KeyError" \
            "ckpt/$tag/bec.log" | head -3 | sed 's/^/      /'
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
    find . -mindepth 1 -maxdepth 1 \
        ! -name configs ! -name data ! -name '*.sh' ! -name '*.py' ! -name README.md \
        -exec rm -rf {} +
    echo "(cleaned generated files; KEEP_OUTPUTS=1 to keep them)"
fi

[[ ${#FAILED[@]} -eq 0 ]] || exit 1
