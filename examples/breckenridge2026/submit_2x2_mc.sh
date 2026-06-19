#!/bin/bash

# Submit the complete 2x2 policy-training/evaluation-environment comparison.
#
# Usage:
#   bash examples/breckenridge2026/submit_2x2_mc.sh \
#       /path/to/october_leo_policy_or_checkpoint \
#       /path/to/new_mixed_policy_or_checkpoint \
#       [max-concurrent-per-cell]
#
# Four independent arrays are submitted, with no Slurm dependencies:
#   leo_trained__leo_eval
#   leo_trained__mixed_eval
#   mixed_trained__leo_eval
#   mixed_trained__mixed_eval

set -euo pipefail

LEO_POLICY=${1:?First argument must be the October LEO-trained policy/checkpoint}
MIXED_POLICY=${2:?Second argument must be the new mixed-trained policy/checkpoint}
MAX_CONCURRENT=${3:-10}

if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "max-concurrent-per-cell must be a positive integer" >&2
    exit 2
fi

cd /projects/$USER/bsk_rl
source /projects/$USER/.venv/bin/activate

CAMPAIGN_ID=${BRECK_MC_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${BRECK_MC_OUTPUT_ROOT:-/scratch/alpine/$USER/breckenridge2026_mc/leo_vs_mixed_10d90i_${CAMPAIGN_ID}}
MANIFEST=${BRECK_MC_MANIFEST:-$OUTPUT_ROOT/breckenridge2026_2x2_manifest.json}
TIME_LIMIT=${BRECK_MC_TIME:-02:00:00}
MEMORY=${BRECK_MC_MEM:-16G}
CPUS_PER_TASK=${BRECK_MC_CPUS_PER_TASK:-4}
ARRAY_SPEC="0-99%${MAX_CONCURRENT}"

mkdir -p "$OUTPUT_ROOT" /scratch/alpine/$USER/job_output

if [[ -f "$MANIFEST" ]]; then
    echo "Reusing frozen manifest: $MANIFEST"
else
    python3 -u examples/breckenridge2026/prepare_mc_manifest.py \
        --leo-policy "$LEO_POLICY" \
        --mixed-policy "$MIXED_POLICY" \
        --output "$MANIFEST"
fi

cells=(
    leo_trained__leo_eval
    leo_trained__mixed_eval
    mixed_trained__leo_eval
    mixed_trained__mixed_eval
)

echo
echo "Submitting four independent Breckenridge MC arrays"
echo "  output root: $OUTPUT_ROOT"
echo "  manifest:    $MANIFEST"
echo "  seeds:       0-99 in every cell"
echo "  array limit: $MAX_CONCURRENT tasks per cell"
echo "  dependencies: none"
echo

for cell in "${cells[@]}"; do
    job_name="br26_${cell}"
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
echo "Summarize after all four arrays finish:"
echo "  python3 examples/breckenridge2026/summarize_2x2_mc.py --input-root \"$OUTPUT_ROOT\""
