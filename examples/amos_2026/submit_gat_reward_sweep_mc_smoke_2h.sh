#!/bin/bash

# Freeze the currently available complete checkpoints and submit the immediate
# two-hour AMOS 2026 GAT Monte Carlo smoke test.
# Usage:
#   bash examples/amos_2026/submit_gat_reward_sweep_mc_smoke_2h.sh
#   bash examples/amos_2026/submit_gat_reward_sweep_mc_smoke_2h.sh 2

set -euo pipefail

MAX_CONCURRENT=${1:-4}
if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [max-concurrent: default 4]" >&2
    exit 2
fi

cd /projects/$USER/bsk_rl
source /projects/$USER/.venv/bin/activate

CAMPAIGN_ID=${BSK_RL_MC_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${BSK_RL_MC_OUTPUT_ROOT:-/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i_smoke_2h_$CAMPAIGN_ID}
MANIFEST_DIR="$OUTPUT_ROOT/manifests"
MANIFEST="$MANIFEST_DIR/gat_full_actions_obs_v9_eval100d00i_nonalpha48h_frozen.json"
JOB_NAME="gat_mc_smoke_2h"

mkdir -p "$MANIFEST_DIR" /scratch/alpine/$USER/job_output
python3 -u examples/amos_2026/evaluate_gat_reward_sweep_mc.py \
    --write-manifest "$MANIFEST"

echo
echo "Submitting one Slurm array job: $JOB_NAME"
echo "  policies:       12 non-alpha 48-hour GAT runs"
echo "  seeds:          0..9 inclusive"
echo "  array tasks:    12 policy jobs, each running 10 fresh evaluator subprocesses"
echo "  max concurrent: $MAX_CONCURRENT"
echo "  task time cap:  02:00:00"
echo "  common score:   100d00i"
echo "  output root:    $OUTPUT_ROOT"
echo "  manifest:       $MANIFEST"
echo
echo "Analyze after completion with:"
echo "  python3 examples/amos_2026/analyze_gat_reward_sweep_mc.py --input-root \"$OUTPUT_ROOT\" --expected-seeds 0:10"
echo

sbatch \
    --job-name="$JOB_NAME" \
    --array="0-11%${MAX_CONCURRENT}" \
    --export=ALL,BSK_RL_MC_SEED_START=0,BSK_RL_MC_SEEDS_PER_BLOCK=10,BSK_RL_MC_OUTPUT_ROOT="$OUTPUT_ROOT",BSK_RL_MC_MANIFEST="$MANIFEST" \
    examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_smoke_2h.sh
