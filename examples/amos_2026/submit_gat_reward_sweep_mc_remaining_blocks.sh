#!/bin/bash

# Submit the remaining AMOS 2026 GAT full-action Monte Carlo seed blocks.
# This is intended to continue from a smoke folder that already contains
# seeds 0..10, reusing the same frozen checkpoint manifest.
#
# Usage:
#   BSK_RL_MC_OUTPUT_ROOT=/scratch/alpine/$USER/amos2026_mc/<smoke-folder> \
#   BSK_RL_MC_MANIFEST=/scratch/alpine/$USER/amos2026_mc/<smoke-folder>/manifests/gat_full_actions_obs_v9_eval100d00i_nonalpha48h_frozen.json \
#     bash examples/amos_2026/submit_gat_reward_sweep_mc_remaining_blocks.sh 2
#
# Optional env vars:
#   BSK_RL_MC_START_BLOCK=10
#   BSK_RL_MC_END_BLOCK=90

set -euo pipefail

MAX_CONCURRENT=${1:-2}
START_BLOCK=${BSK_RL_MC_START_BLOCK:-10}
END_BLOCK=${BSK_RL_MC_END_BLOCK:-90}

if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [max-concurrent: default 2]" >&2
    exit 2
fi
if ! [[ "$START_BLOCK" =~ ^[0-9]+$ ]] || ! [[ "$END_BLOCK" =~ ^[0-9]+$ ]]; then
    echo "BSK_RL_MC_START_BLOCK and BSK_RL_MC_END_BLOCK must be numeric" >&2
    exit 2
fi
if (( START_BLOCK % 10 != 0 || END_BLOCK % 10 != 0 || START_BLOCK < 0 || END_BLOCK > 90 || START_BLOCK > END_BLOCK )); then
    echo "Blocks must be multiples of 10 with 0 <= start <= end <= 90" >&2
    exit 2
fi

cd /projects/$USER/bsk_rl
source /projects/$USER/.venv/bin/activate

OUTPUT_ROOT=${BSK_RL_MC_OUTPUT_ROOT:?Set BSK_RL_MC_OUTPUT_ROOT to the smoke/full campaign folder to continue}
MANIFEST=${BSK_RL_MC_MANIFEST:-$OUTPUT_ROOT/manifests/gat_full_actions_obs_v9_eval100d00i_nonalpha48h_frozen.json}
if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found: $MANIFEST" >&2
    exit 1
fi

mkdir -p /scratch/alpine/$USER/job_output "$OUTPUT_ROOT"

echo "Submitting chained AMOS 2026 GAT MC blocks"
echo "  output root:    $OUTPUT_ROOT"
echo "  manifest:       $MANIFEST"
echo "  blocks:         $START_BLOCK..$END_BLOCK"
echo "  max concurrent: $MAX_CONCURRENT policy tasks per block"
echo "  dependency:     afterany chain, so only one 10-seed block is active at once"
echo "  skip behavior:  completed matching policy/seed runs under output root are skipped"
echo

previous_job_id=""
for ((seed_start = START_BLOCK; seed_start <= END_BLOCK; seed_start += 10)); do
    seed_end=$((seed_start + 9))
    job_name="gat_mc_s$(printf '%03d' "$seed_start")_$(printf '%03d' "$seed_end")"
    sbatch_args=(
        --parsable
        --job-name="$job_name"
        --array="0-7%${MAX_CONCURRENT}"
        --export=ALL,BSK_RL_MC_SEED_START="$seed_start",BSK_RL_MC_SEEDS_PER_BLOCK=10,BSK_RL_MC_OUTPUT_ROOT="$OUTPUT_ROOT",BSK_RL_MC_MANIFEST="$MANIFEST"
    )
    if [[ -n "$previous_job_id" ]]; then
        sbatch_args+=(--dependency="afterany:$previous_job_id")
    fi

    job_id=$(sbatch "${sbatch_args[@]}" examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh)
    echo "Submitted $job_name as job $job_id"
    previous_job_id="$job_id"
done

echo
echo "Analyze after all blocks finish with:"
echo "  python examples/amos_2026/analyze_gat_reward_sweep_mc.py --input-root \"$OUTPUT_ROOT\" --expected-seeds 0:100"
