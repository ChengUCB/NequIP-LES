#!/usr/bin/env bash
# Drive the whole MD timing sweep: every model x every engine x both N_max policies.
#
#   ./run_timing.sh
#
# Resumable: rows already in timing.csv are skipped, so this can be chunked across jobs and
# restarted after an OOM. One process at a time -- two MD jobs on one GPU invalidate every
# number in the file.
#
# Filters, for dry runs and for re-doing one slice:
#   MODELS="nequip_les"        substring match on the checkpoint name
#   ENGINES="ase-aoti lammps"  exact engine names
#   POLICIES="fixed"           fixed | scaled
#   SIZES="1 1 1"              semicolon-separate several: "1 1 1;2 2 2"
#
# LAMMPS rows need $LMP. They are skipped with a note if it is unset.
set -uo pipefail
cd "$(dirname "$0")"

PY=$(command -v python)
MODELS="${MODELS:-}"
ENGINES="${ENGINES:-}"
POLICIES="${POLICIES:-fixed scaled}"
SIZES="${SIZES:-}"

[[ -f water.xyz ]] || { echo "water.xyz missing -- run: python prepare.py"; exit 2; }

# engines per backbone, and which of them the scaled policy uses, both read from common.py so
# this script cannot drift from the tables there
engines_for() {   # $1 backbone, $2 policy
    "$PY" - "$1" "$2" <<'EOF'
import sys
import common as C
backbone, policy = sys.argv[1], sys.argv[2]
names = [e[0] for e in C.ENGINES[backbone]]
if policy == "scaled":
    names = [n for n in names if n in C.SCALED_ENGINES]
print(" ".join(names))
EOF
}
runner_for() {    # $1 backbone, $2 engine -> md | lammps
    "$PY" - "$1" "$2" <<'EOF'
import sys
import common as C
backbone, engine = sys.argv[1], sys.argv[2]
print(next(e[3] for e in C.ENGINES[backbone] if e[0] == engine))
EOF
}

echo "torch    : $("$PY" -c 'import torch; print(torch.__version__)')"
echo "gpu      : $("$PY" -c 'import common as C; print(C.gpu_name())')"
echo "\$LMP     : ${LMP:-<unset, LAMMPS rows will be skipped>}"
echo "policies : $POLICIES"
[[ -n "$SIZES" ]] && echo "sizes    : $SIZES"

for ckpt in models/*.ckpt; do
    [[ -e "$ckpt" ]] || { echo "no checkpoints in models/"; exit 2; }
    stem=$(basename "$ckpt" .ckpt)
    [[ -n "$MODELS" && "$stem" != *"$MODELS"* ]] && continue

    backbone="${stem%%_*}"
    variant="${stem#*_}"

    for policy in $POLICIES; do
        # SR has no Ewald sum, so N_max cannot affect it -- the scaled policy is meaningless
        [[ "$policy" == scaled && "$variant" == sr ]] && continue

        for engine in $(engines_for "$backbone" "$policy"); do
            [[ -n "$ENGINES" && " $ENGINES " != *" $engine "* ]] && continue

            runner=$(runner_for "$backbone" "$engine")
            if [[ "$runner" == lammps && -z "${LMP:-}" ]]; then
                echo "=== $stem / $engine / $policy === skipped: \$LMP unset"
                continue
            fi

            echo "=== $stem / $engine / $policy ==="
            script=$([[ "$runner" == lammps ]] && echo time_lammps.py || echo time_md.py)
            cmd=("$PY" "$script" --model "$ckpt" --engine "$engine" --policy "$policy")
            [[ -n "$SIZES" ]] && cmd+=(--sizes "$SIZES")
            "${cmd[@]}"
        done
    done
done

echo
echo "done. rows in timing.csv: $(( $(wc -l < timing.csv) - 1 ))"
echo "next: python plot.py"
