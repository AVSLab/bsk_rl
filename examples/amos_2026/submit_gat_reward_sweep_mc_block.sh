#!/bin/bash

# Freeze one checkpoint per policy, then submit one ten-seed Monte Carlo block.
# Usage:
#   bash examples/amos_2026/submit_gat_reward_sweep_mc_block.sh 0
#   bash examples/amos_2026/submit_gat_reward_sweep_mc_block.sh 10

set -euo pipefail

SEED_START=${1:-0}
MAX_CONCURRENT=${2:-4}
if ! [[ "$SEED_START" =~ ^[0-9]+$ ]] || ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 <seed-start: 0,10,...,90> [max-concurrent: default 4]" >&2
    exit 2
fi
if (( SEED_START % 10 != 0 || SEED_START > 90 )); then
    echo "seed-start must be one of: 0, 10, 20, ..., 90" >&2
    exit 2
fi

cd /projects/$USER/bsk_rl
source /projects/$USER/.venv/bin/activate

OUTPUT_ROOT=${BSK_RL_MC_OUTPUT_ROOT:-/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i}
MANIFEST_DIR="$OUTPUT_ROOT/manifests"
SEED_END=$((SEED_START + 9))
MANIFEST=${BSK_RL_MC_MANIFEST:-$MANIFEST_DIR/gat_full_actions_obs_v9_eval100d00i_frozen.json}
JOB_NAME="gat_mc_s$(printf '%03d' "$SEED_START")_$(printf '%03d' "$SEED_END")"

mkdir -p "$MANIFEST_DIR" /scratch/alpine/$USER/job_output
if [[ "${BSK_RL_MC_REFRESH_MANIFEST:-0}" == "1" && -e "$MANIFEST" ]]; then
    ARCHIVED_MANIFEST="${MANIFEST%.json}_archived_$(date -u +%Y%m%dT%H%M%SZ).json"
    mv "$MANIFEST" "$ARCHIVED_MANIFEST"
    echo "Archived previous checkpoint manifest:"
    echo "  $ARCHIVED_MANIFEST"
fi
if [[ -f "$MANIFEST" ]]; then
    echo "Reusing frozen checkpoint manifest:"
    echo "  $MANIFEST"
else
    python3 -u examples/amos_2026/evaluate_gat_reward_sweep_mc.py \
        --write-manifest "$MANIFEST"
fi

echo
echo "Submitting $JOB_NAME with campaign manifest:"
echo "  $MANIFEST"
sbatch \
    --job-name="$JOB_NAME" \
    --array="0-79%${MAX_CONCURRENT}" \
    --export=ALL,BSK_RL_MC_SEED_START="$SEED_START",BSK_RL_MC_OUTPUT_ROOT="$OUTPUT_ROOT",BSK_RL_MC_MANIFEST="$MANIFEST" \
    examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh
