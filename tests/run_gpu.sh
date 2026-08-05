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
#   ./run_gpu.sh                        # everything available
#   ./run_gpu.sh nequip                 # only models whose tag contains "nequip"
#   ROWS=enable_ ./run_gpu.sh           # only the acceleration rows
#   ROWS=mliap ./run_gpu.sh             # only the ML-IAP rows
#   KEEP_OUTPUTS=1 ./run_gpu.sh         # keep checkpoints, so a rerun reuses them
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
# Which kernel modifiers can we exercise? The names differ per backbone, and the docs
# are explicit about which ones work at training time:
#   enable_OpenEquivariance          nequip   train + inference
#   enable_CuEquivariance            nequip   inference only (training is WIP)
#   enable_CuEquivarianceContracter  allegro  train + inference
#   enable_TritonContracter          allegro  inference only
NEQUIP_INFER_MODS=(); NEQUIP_TRAIN_MODS=()
ALLEGRO_INFER_MODS=(); ALLEGRO_TRAIN_MODS=()
if "$PY" -c "import openequivariance" 2>/dev/null; then
    NEQUIP_INFER_MODS+=(enable_OpenEquivariance)
    NEQUIP_TRAIN_MODS+=(enable_OpenEquivariance)
fi
if "$PY" -c "import cuequivariance_torch" 2>/dev/null; then
    NEQUIP_INFER_MODS+=(enable_CuEquivariance)
    ALLEGRO_INFER_MODS+=(enable_CuEquivarianceContracter)
    ALLEGRO_TRAIN_MODS+=(enable_CuEquivarianceContracter)
fi
if "$PY" -c "import triton" 2>/dev/null; then
    ALLEGRO_INFER_MODS+=(enable_TritonContracter)   # triton ships with torch
fi
echo "nequip  modifiers: infer=[${NEQUIP_INFER_MODS[*]:-none}] train=[${NEQUIP_TRAIN_MODS[*]:-none}]"
echo "allegro modifiers: infer=[${ALLEGRO_INFER_MODS[*]:-none}] train=[${ALLEGRO_TRAIN_MODS[*]:-none}]"

mods_for() {         # $1 tag -> modifiers valid when EXPORTING it
    case "$1" in
        nequip_*)  echo "${NEQUIP_INFER_MODS[*]:-}" ;;
        allegro_*) echo "${ALLEGRO_INFER_MODS[*]:-}" ;;
    esac
}
train_mods_for() {   # $1 tag -> modifiers valid when TRAINING with it
    case "$1" in
        nequip_*)  echo "${NEQUIP_TRAIN_MODS[*]:-}" ;;
        allegro_*) echo "${ALLEGRO_TRAIN_MODS[*]:-}" ;;
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

# Which --target values does this environment actually offer? `pair_allegro` is
# registered by the allegro package through the `nequip.extension` entry point, so it is
# missing whenever allegro is installed without its metadata -- skip those rows instead
# of failing them.
TARGETS=$("$PY" -c "
from nequip.scripts._compile_utils import COMPILE_TARGET_DICT as d
print(' '.join(sorted(d)))" 2>/dev/null)
echo "available --target values: ${TARGETS:-<could not query>}"
has_target() { [[ " $TARGETS " == *" $1 "* ]]; }

mkdir -p ckpt compiled
PASSED=(); FAILED=()

row_skipped() {   # $1 row -> 0 when this row is excluded
    [[ -n "$FILTER" && "$1" != *"$FILTER"* ]] && return 0
    # ROWS narrows to individual rows regardless of the model filter, e.g.
    #   ROWS=enable_ ./run_gpu.sh          only the acceleration rows
    #   ROWS=mliap   ./run_gpu.sh          only the ML-IAP rows
    [[ -n "${ROWS:-}" && "$1" != *"${ROWS}"* ]] && return 0
    return 1
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
    # positionals FIRST, as the docs show: `--modifiers` takes nargs="+" and would
    # otherwise swallow the input and output paths
    "${NEQUIP_COMPILE[@]}" "$ck" "compiled/$name.nequip.$ext" \
        --mode "$mode" --device cuda --target "$target" "$@" > "$log" 2>&1
    record "$row" $? "$log"
}

# ---------------------------------------------------------------------------
# 1. plain CUDA: train both LES paths, package, then export for ASE and for LAMMPS
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

    # a portable model file, independent of the source tree; this broke once because a
    # LES module imported the e3nn package roots and dragged sympy into torch.package
    if ! row_skipped "$tag/package"; then
        log="compiled/${tag}_package.log"
        nequip-package build "$CK" "compiled/${tag}.nequip.zip" > "$log" 2>&1
        record "$tag/package" $? "$log"
    fi

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
    done

    # LAMMPS ML-IAP is not a --target: it has its own CLI (nequip-prepare-lmp-mliap),
    # takes the same --modifiers, and needs LAMMPS built with ML-IAP in this env
    if "$PY" -c "import lammps" 2>/dev/null; then
        mliap() {   # $1 row suffix, $2... extra args
            local suffix="$1"; shift
            row_skipped "$tag/mliap$suffix" && return 0
            local name="${tag}_mliap${suffix//\//_}"
            nequip-prepare-lmp-mliap "$CK" "compiled/${name}.nequip.lmp.pt" "$@" \
                > "compiled/${name}.log" 2>&1
            record "$tag/mliap$suffix" $? "compiled/${name}.log"
        }
        mliap ""
        for mod in $(mods_for "$tag"); do mliap "/$mod" --modifiers "$mod"; done
    else
        printf '  %-60s skipped (no LAMMPS ML-IAP in this env)\n' "$tag/mliap"
    fi

    # ---- accelerations, on every target that matters for inference ----
    for mod in $(mods_for "$tag"); do
        for mode in "${MODES[@]}"; do
            export_ckpt "$CK" "$tag/ase/$mode/cuda/$mod" ase "$mode" --modifiers "$mod"
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
    for mod in $(train_mods_for "$tag"); do
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
