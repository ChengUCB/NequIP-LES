#!/usr/bin/env bash
# CUDA-only checks: GPU training, TF32, the GPU kernel accelerations, and export
# of the accelerated models.
#
# Everything here needs a CUDA device, and each kernel library needs to be
# installed, so the script skips what is unavailable instead of failing. Run it on
# the cluster; run_all.sh / run_bec.sh / run_compile.sh cover what a laptop can do.
#
#   nequip  : enable_OpenEquivariance   pip install openequivariance   (NVIDIA or AMD/HIP)
#             enable_CuEquivariance     pip install cuequivariance-torch cuequivariance-ops-torch-cu12
#   allegro : enable_TritonContracter          (triton ships with torch)
#             enable_CuEquivarianceContracter  (same cuequivariance install)
#
# Docs:
#   https://nequip.readthedocs.io/en/latest/guide/accelerations/openequivariance.html
#   https://nequip.readthedocs.io/en/latest/guide/accelerations/cuequivariance.html
#   https://nequip.readthedocs.io/projects/allegro/en/latest/guide/cuequivariance.html
#   https://nequip.readthedocs.io/projects/allegro/en/latest/guide/triton.html
#
# NOTE: train-time compilation needs torch 2.9.x. On 2.12 it fails on CUDA with
# "derivative for aten::silu_backward is not implemented", which is a nequip/torch
# issue and not LES-specific (see README).
#
# Usage:
#   ./run_gpu.sh              # everything available
#   ./run_gpu.sh nequip       # only rows matching "nequip"
#   KEEP_OUTPUTS=1 ./run_gpu.sh
set -uo pipefail
cd "$(dirname "$0")"

FILTER="${1:-}"
PY=$(command -v python)

if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "no CUDA device: nothing to do here (use run_all.sh / run_compile.sh)"
    exit 0
fi
echo "torch $("$PY" -c 'import torch; print(torch.__version__)')  |  $("$PY" -c 'import torch; print(torch.cuda.get_device_name(0))')"

# Which kernel modifiers can we exercise? The names differ per backbone: nequip
# swaps whole tensor products, Allegro swaps its contracter.
NEQUIP_MODS=(); ALLEGRO_MODS=()
if "$PY" -c "import openequivariance" 2>/dev/null; then
    NEQUIP_MODS+=(enable_OpenEquivariance)
fi
if "$PY" -c "import cuequivariance_torch" 2>/dev/null; then
    NEQUIP_MODS+=(enable_CuEquivariance)
    ALLEGRO_MODS+=(enable_CuEquivarianceContracter)
fi
if "$PY" -c "import triton" 2>/dev/null; then
    ALLEGRO_MODS+=(enable_TritonContracter)     # triton ships with torch
fi
echo "nequip modifiers : ${NEQUIP_MODS[*]:-(none installed)}"
echo "allegro modifiers: ${ALLEGRO_MODS[*]:-(none installed)}"

mods_for() {   # $1 tag -> echoes the modifier names that apply to it
    case "$1" in
        nequip_*)  echo "${NEQUIP_MODS[*]:-}" ;;
        allegro_*) echo "${ALLEGRO_MODS[*]:-}" ;;
    esac
}

# export modes nequip still accepts
if "$PY" - <<'EOF' 2>/dev/null
import sys, torch
from packaging.version import parse
sys.exit(0 if parse(torch.__version__.split("+")[0]) < parse("2.10") else 1)
EOF
then MODES=(aotinductor torchscript); else
    MODES=(aotinductor)
    echo "(skipping TorchScript: unsupported by nequip on torch >= 2.10)"
fi

export NEQUIP_FLOAT32_MODEL_TOL="${NEQUIP_FLOAT32_MODEL_TOL:-1e-3}"
export NEQUIP_FLOAT64_MODEL_TOL="${NEQUIP_FLOAT64_MODEL_TOL:-1e-6}"

# nequip-compile goes through compile_tf32fix.py: on torch 2.13 + CUDA the per-backend
# TF32 flags do not start out consistent, and nequip reads the aggregate
# torch.backends.fp32_precision, which torch then refuses to report. The wrapper
# normalises the flags first; see its docstring.
NEQUIP_COMPILE=("$PY" compile_tf32fix.py)

mkdir -p ckpt compiled
PASSED=(); FAILED=()

row_skipped() {   # $1 row -> 0 when the filter excludes it
    [[ -n "$FILTER" && "$1" != *"$FILTER"* ]]
}

record() {   # $1 row, $2 rc, $3 log
    if [[ "$2" -eq 0 ]]; then
        printf '  %-60s ok\n' "$1"; PASSED+=("$1")
    else
        printf '  %-60s FAILED (%s)\n' "$1" "$3"; FAILED+=("$1")
        grep -iE "Error|Exception|not supported|no kernel" "$3" 2>/dev/null | head -2 | sed 's/^/        /'
    fi
}

train_gpu() {   # $1 tag, $2... extra overrides -> prints the checkpoint path
    local tag="$1"; shift
    if [[ -z "$(find "ckpt/$tag" -name '*.ckpt' 2>/dev/null | head -1)" ]]; then
        echo "  training $tag on cuda ..." >&2
        nequip-train -cn "${tag%%+*}" --config-dir configs \
            "hydra.run.dir=ckpt/$tag/run" \
            "++trainer.logger.save_dir=ckpt/$tag" \
            "++trainer.accelerator=gpu" "$@" \
            > "ckpt/$tag/train.log" 2>&1 \
            || { echo "  TRAINING FAILED (ckpt/$tag/train.log)" >&2; return 1; }
    fi
    find "ckpt/$tag" -name '*.ckpt' | head -1
}

export_ckpt() {   # $1 ckpt, $2 row, $3 target, $4 mode, $5... extra nequip-compile args
    local ck="$1" row="$2" target="$3" mode="$4"; shift 4
    row_skipped "$row" && return 0
    local name; name=$(echo "$row" | tr '/ ' '__')
    local ext=pt2; [[ "$mode" == torchscript ]] && ext=pth
    local log="compiled/$name.log"
    "${NEQUIP_COMPILE[@]}" --mode "$mode" --device cuda --target "$target" "$@" \
        "$ck" "compiled/$name.nequip.$ext" > "$log" 2>&1
    record "$row" $? "$log"
}

# ---------------------------------------------------------------------------
# 1. plain CUDA: train both LES paths, then export for ASE and for LAMMPS
# ---------------------------------------------------------------------------
# both periodicities: a periodic model is rejected by pair_allegro (no cell), so
# Allegro's LAMMPS pair style can only be exercised by the non-periodic ones
for tag in nequip_water_eager nequip_water_compiled nequip_dipep_eager nequip_dipep_compiled \
           allegro_water_eager allegro_water_compiled allegro_dipep_eager allegro_dipep_compiled; do
    [[ -n "$FILTER" && "$tag" != *"${FILTER%%/*}"* ]] && continue
    printf '\n=== %s ===\n' "$tag"
    mkdir -p "ckpt/$tag"

    CK=$(train_gpu "$tag") || { FAILED+=("$tag/train"); continue; }
    PASSED+=("$tag/train/cuda")
    printf '  %-60s ok\n' "$tag/train/cuda"

    if [[ "$tag" == nequip_* ]]; then pair=pair_nequip; else pair=pair_allegro; fi
    # pair_allegro declares no cell, so a periodic model there is a rejection case
    # (run_compile.sh covers that); here only the non-periodic ones are exported
    skip_pair=no
    [[ "$pair" == pair_allegro && "$tag" == *_water_* ]] && skip_pair=yes

    for mode in "${MODES[@]}"; do
        export_ckpt "$CK" "$tag/ase/$mode/cuda" ase "$mode"
        [[ "$skip_pair" == no ]] && export_ckpt "$CK" "$tag/$pair/$mode/cuda" "$pair" "$mode"
        # TF32 changes float32 arithmetic and the Ewald k-space sum is sensitive to it
        export_ckpt "$CK" "$tag/ase/$mode/cuda/tf32" ase "$mode" --tf32
        # LAMMPS ML-IAP: a packaging path of its own, separate from the pair styles
        export_ckpt "$CK" "$tag/mliap/$mode/cuda" lammps_mliap "$mode"
    done

    # ---- accelerations, on every target that matters for inference ----
    for mod in $(mods_for "$tag"); do
        for mode in "${MODES[@]}"; do
            export_ckpt "$CK" "$tag/ase/$mode/cuda/$mod" ase "$mode" --modifiers "$mod"
            export_ckpt "$CK" "$tag/mliap/$mode/cuda/$mod" lammps_mliap "$mode" --modifiers "$mod"
            [[ "$skip_pair" == no ]] && \
                export_ckpt "$CK" "$tag/$pair/$mode/cuda/$mod" "$pair" "$mode" --modifiers "$mod"
        done
    done
done

# ---------------------------------------------------------------------------
# 2. accelerations at TRAINING time -- the modifier wraps the model, so this is a
#    different code path from exporting an already-trained checkpoint
# ---------------------------------------------------------------------------
for tag in nequip_water_eager allegro_water_eager; do
    for mod in $(mods_for "$tag"); do
        row="$tag+$mod/train/cuda"
        row_skipped "$row" && continue
        printf '\n=== %s ===\n' "$row"
        name="$tag+$mod"
        mkdir -p "ckpt/$name"
        # the modifier nests the model, which a command-line override cannot do,
        # so generate a wrapped copy of the same config
        "$PY" wrap_modifier.py "$tag" "$mod" "ckpt/$name/config" > /dev/null || {
            record "$row" 1 ""; continue; }
        if CK=$(nequip-train -cn "$name" --config-dir "ckpt/$name/config" \
                    "hydra.run.dir=ckpt/$name/run" \
                    "++trainer.logger.save_dir=ckpt/$name" \
                    "++trainer.accelerator=gpu" \
                    > "ckpt/$name/train.log" 2>&1 \
                && find "ckpt/$name" -name '*.ckpt' | head -1); then
            record "$row" 0 "ckpt/$name/train.log"
            for mode in "${MODES[@]}"; do
                export_ckpt "$CK" "$name/ase/$mode/cuda" ase "$mode" --modifiers "$mod"
            done
        else
            record "$row" 1 "ckpt/$name/train.log"
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
