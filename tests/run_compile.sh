#!/usr/bin/env bash
# Export trained LES models for deployment with `nequip-compile` and check the results.
#
# This is a DIFFERENT thing from `compile_mode: compile` in the training configs:
# that one compiles the model for the training loop, this one produces the artefact
# LAMMPS / ASE load. Both are tested, and every model is exported regardless of
# which training path produced it -- a checkpoint trained eagerly must export just
# as well as one trained compiled.
#
# Compilation works on the CPU, so this runs anywhere; CUDA is used as well when
# present.
#
# Two of the rows are expected to FAIL, and that is the point:
#
#   periodic model + pair_allegro   ->  must be rejected
#
# `pair_allegro` declares only positions, edge indices and atom types, so the
# graph has no cell and the reciprocal-space sum cannot be evaluated. Before this
# was caught, the cell was silently replaced by zeros and the long-range physics
# quietly became non-periodic (ChengUCB/NequIP-LES#15). A non-periodic model on
# the same target is fine, and is checked too.
#
# TorchScript is not attempted on PyTorch >= 2.10, where nequip itself refuses it.
#
# Usage:
#   ./run_compile.sh              # everything
#   ./run_compile.sh allegro      # only rows matching "allegro"
#   KEEP_OUTPUTS=1 ./run_compile.sh
set -uo pipefail
cd "$(dirname "$0")"

FILTER="${1:-}"
PY=$(command -v python)
TORCH=$("$PY" -c 'import torch; print(torch.__version__)')

# Train and export on the same device: exporting a GPU-trained model for the CPU is
# not a case anyone deploys, so it is not worth the runtime.
if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICES=(cuda); ACCELERATOR=gpu
else
    DEVICES=(cpu);  ACCELERATOR=cpu
fi
echo "torch $TORCH  |  training on: $ACCELERATOR  |  exporting for: ${DEVICES[*]}"

# nequip raises on TorchScript for >= 2.10; only worth trying below that
if "$PY" - <<'EOF' 2>/dev/null
import sys, torch
from packaging.version import parse
sys.exit(0 if parse(torch.__version__.split("+")[0]) < parse("2.10") else 1)
EOF
then MODES=(aotinductor torchscript); else
    MODES=(aotinductor)
    echo "(skipping TorchScript: unsupported by nequip on torch >= 2.10)"
fi

# nequip-compile checks the compiled model against the eager one with an ABSOLUTE
# tolerance (5e-5 for float32). These models train for two epochs on a handful of
# frames, so their forces are still of order 10^3 eV/A and that tolerance is
# unreachable: the non-periodic runs failed at 6e-4 on predictions whose largest
# entry was 1475, i.e. 4e-7 relative -- float32 round-off. Loosen it enough to
# clear the round-off and no further; a genuine miscompile in this code shows up
# orders of magnitude out, not at the seventh digit.
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

check() {   # $1 ckpt, $2 model tag, $3 target, $4 mode, $5 device, $6 expect(pass|reject)
    local ck="$1" tag="$2" target="$3" mode="$4" dev="$5" expect="$6"
    local row="$tag/$target/$mode/$dev"
    [[ -n "$FILTER" && "$row" != *"$FILTER"* ]] && return 0

    local ext=pt2; [[ "$mode" == torchscript ]] && ext=pth
    local out="compiled/${tag}_${target}_${mode}_${dev}.nequip.$ext"
    local log="compiled/${tag}_${target}_${mode}_${dev}.log"
    "${NEQUIP_COMPILE[@]}" --mode "$mode" --device "$dev" --target "$target" \
        "$ck" "$out" > "$log" 2>&1
    local rc=$?

    if [[ "$expect" == reject ]]; then
        if [[ $rc -ne 0 ]] && grep -q "contains no cell" "$log"; then
            printf '  %-52s rejected as expected\n' "$row"; PASSED+=("$row")
        else
            printf '  %-52s SHOULD HAVE BEEN REJECTED (rc=%s)\n' "$row" "$rc"; FAILED+=("$row")
        fi
    else
        if [[ $rc -eq 0 && -s "$out" ]]; then
            printf '  %-52s compiled\n' "$row"; PASSED+=("$row")
        else
            printf '  %-52s FAILED (%s)\n' "$row" "$log"; FAILED+=("$row")
            grep -iE "Error|guard" "$log" | head -2 | sed 's/^/        /'
        fi
    fi
}

# both training paths: an eagerly trained checkpoint and a compile-trained one must
# both export. `legacy` is deliberately absent -- the loop-based Ewald cannot be
# traced at all, which is why the vectorized one exists.
# `*_sr_eager` carry no LES: if an export breaks, they say whether LES is involved
for tag in nequip_water_eager nequip_water_compiled nequip_dipep_eager nequip_dipep_compiled \
           allegro_water_eager allegro_water_compiled allegro_dipep_eager allegro_dipep_compiled \
           nequip_water_sr_eager nequip_dipep_sr_eager \
           allegro_water_sr_eager allegro_dipep_sr_eager; do
    # skip whole models the filter excludes, so a filtered run does not train them
    [[ -n "$FILTER" && "$tag" != *"${FILTER%%/*}"* ]] && continue
    mkdir -p "ckpt/$tag"
    printf '\n=== %s ===\n' "$tag"
    CK=$(train_once "$tag") || { FAILED+=("$tag/train"); continue; }

    # ASE always passes the cell, so it works for periodic and non-periodic alike
    targets=(ase)
    # the LAMMPS pair style follows the backbone
    if [[ "$tag" == nequip_* ]]; then targets+=(pair_nequip); else targets+=(pair_allegro); fi

    for target in "${targets[@]}"; do
        for mode in "${MODES[@]}"; do
            expect=pass
            # pair_allegro carries no cell, so a periodic LES model must be refused --
            # but only the modes that trace with example inputs can notice. `torchscript`
            # merely compiles the source, so the branch that raises never runs and the
            # export succeeds; the guard is compiled into the artefact and fires when
            # LAMMPS calls it without a cell. An SR model needs no cell at all.
            if [[ "$target" == pair_allegro && "$tag" == *_water_* \
                  && "$tag" != *_sr_* && "$mode" == aotinductor ]]; then
                expect=reject
            fi
            for dev in "${DEVICES[@]}"; do
                check "$CK" "$tag" "$target" "$mode" "$dev" "$expect"
            done
        done
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
