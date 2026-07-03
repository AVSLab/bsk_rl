#!/bin/bash

# Reproduce the LEO-trained alpha sweep in the mixed LEO/MEO/GEO environment.
#
# Usage:
#   bash examples/breckenridge2026/submit_alpha_sweep_mc.sh [max-concurrent-per-cell]

set -euo pipefail

MAX_CONCURRENT=${1:-10}

if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "max-concurrent-per-cell must be a positive integer" >&2
    exit 2
fi

cd /projects/$USER/bsk_rl
source /projects/$USER/.venv/bin/activate
VENV_PYTHON=/projects/$USER/.venv/bin/python

source examples/breckenridge2026/alpine_runtime.sh
configure_breckenridge_alpine_runtime "$VENV_PYTHON"

CAMPAIGN_ID=${BRECK_MC_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${BRECK_MC_OUTPUT_ROOT:-/scratch/alpine/$USER/breckenridge2026_mc/alpha_sweep_mixed_${CAMPAIGN_ID}}
MANIFEST=${BRECK_MC_MANIFEST:-$OUTPUT_ROOT/breckenridge2026_alpha_sweep_manifest.json}
TIME_LIMIT=${BRECK_MC_TIME:-02:00:00}
MEMORY=${BRECK_MC_MEM:-16G}
CPUS_PER_TASK=${BRECK_MC_CPUS_PER_TASK:-4}
ARRAY_SPEC="0-99%${MAX_CONCURRENT}"

mkdir -p "$OUTPUT_ROOT" /scratch/alpine/$USER/job_output

if [[ -f "$MANIFEST" ]]; then
    echo "Reusing frozen manifest: $MANIFEST"
else
    "$VENV_PYTHON" -u examples/breckenridge2026/prepare_mc_manifest.py \
        --policy-set alpha_sweep \
        --output "$MANIFEST"
fi

labels=(
    0d100i
    10d90i
    20d80i
    30d70i
    40d60i
    50d50i
    60d40i
    70d30i
    80d20i
    90d10i
    100d00i
)

echo
echo "Submitting independent Breckenridge alpha-sweep MC arrays"
echo "  output root: $OUTPUT_ROOT"
echo "  manifest:    $MANIFEST"
echo "  seeds:       0-99 in every alpha cell"
echo "  array limit: $MAX_CONCURRENT tasks per cell"
echo "  dependencies: none"
echo

for label in "${labels[@]}"; do
    cell="leo_trained_${label}__mixed_eval"
    job_name="br26_alpha_${label}"
    job_id=$(sbatch \
        --parsable \
        --job-name="$job_name" \
        --array="$ARRAY_SPEC" \
        --time="$TIME_LIMIT" \
        --mem="$MEMORY" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --export=ALL,BRECK_MC_MANIFEST="$MANIFEST",BRECK_MC_CELL="$cell",BRECK_MC_OUTPUT_ROOT="$OUTPUT_ROOT" \
        examples/breckenridge2026/sbatch_mc_cell.sh)
    echo "Submitted $cell as job $job_id"
done

echo
echo "Monitor:"
echo "  squeue -u $USER"
echo
echo "Audit and summarize after all arrays finish:"
echo "  python3 examples/breckenridge2026/audit_mc_campaign.py --input-root \"$OUTPUT_ROOT\""
echo "  python3 examples/breckenridge2026/summarize_mc_campaign.py --input-root \"$OUTPUT_ROOT\""
