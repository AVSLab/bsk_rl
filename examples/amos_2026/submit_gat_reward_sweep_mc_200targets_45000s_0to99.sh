#!/bin/bash

# Submit the full 200-target / 45,000-second AMOS 2026 GAT MC campaign.
#
# This submits ten Slurm array jobs:
#   seeds 000..009, 010..019, ..., 090..099
# Each array job has twelve tasks, one per trained policy. Each task runs the ten
# seeds for that policy as fresh evaluator subprocesses. Blocks are chained by
# dependency so only one ten-seed block is active at a time; within an active
# block, up to MAX_CONCURRENT policy tasks run at once.
#
# Usage from /projects/$USER/bsk_rl:
#   bash examples/amos_2026/submit_gat_reward_sweep_mc_200targets_45000s_0to99.sh
#   bash examples/amos_2026/submit_gat_reward_sweep_mc_200targets_45000s_0to99.sh 5
#
# Restart behavior:
#   Re-run the same command with BSK_RL_MC_OUTPUT_ROOT pointing at the existing
#   campaign folder. Completed policy/seed pairs are skipped automatically.

set -euo pipefail

MAX_CONCURRENT=${1:-5}
START_BLOCK=${BSK_RL_MC_START_BLOCK:-0}
END_BLOCK=${BSK_RL_MC_END_BLOCK:-90}
SEEDS_PER_BLOCK=${BSK_RL_MC_SEEDS_PER_BLOCK:-10}
POLICY_TAGS=${BSK_RL_MC_POLICY_TAGS:-00d100i,10d90i,20d80i,30d70i,40d60i,50d50i,60d40i,70d30i,75d25i,80d20i,90d10i,100d00i}
N_TARGETS=${BSK_RL_MC_N_TARGETS:-200}
N_TARGETS_AHEAD=${BSK_RL_MC_N_TARGETS_AHEAD:-10}
TOTAL_TIME_SEC=${BSK_RL_MC_TOTAL_TIME_SEC:-45000}
EXTRA_TIME_FACTOR=${BSK_RL_MC_EXTRA_TIME_FACTOR:-1.5}
TIME_LIMIT=${BSK_RL_MC_TIME:-04:00:00}
MEMORY=${BSK_RL_MC_MEM:-12G}
CPUS_PER_TASK=${BSK_RL_MC_CPUS_PER_TASK:-4}

if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [max-concurrent: default 5]" >&2
    exit 2
fi
if ! [[ "$START_BLOCK" =~ ^[0-9]+$ ]] || ! [[ "$END_BLOCK" =~ ^[0-9]+$ ]]; then
    echo "BSK_RL_MC_START_BLOCK and BSK_RL_MC_END_BLOCK must be numeric" >&2
    exit 2
fi
if (( START_BLOCK % SEEDS_PER_BLOCK != 0 || END_BLOCK % SEEDS_PER_BLOCK != 0 || START_BLOCK < 0 || END_BLOCK > 90 || START_BLOCK > END_BLOCK )); then
    echo "Blocks must be seed block starts with 0 <= start <= end <= 90" >&2
    exit 2
fi
POLICY_COUNT=$(python3 - <<PY
tags = [tag.strip() for tag in """$POLICY_TAGS""".split(",") if tag.strip()]
if not tags:
    raise SystemExit("BSK_RL_MC_POLICY_TAGS cannot be empty")
print(len(tags))
PY
)
ARRAY_END=$((POLICY_COUNT - 1))

cd /projects/$USER/bsk_rl
source /projects/$USER/.venv/bin/activate

CAMPAIGN_ID=${BSK_RL_MC_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${BSK_RL_MC_OUTPUT_ROOT:-/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i_${N_TARGETS}targets_${TOTAL_TIME_SEC}s_${CAMPAIGN_ID}}
MANIFEST_DIR="$OUTPUT_ROOT/manifests"
MANIFEST=${BSK_RL_MC_MANIFEST:-$MANIFEST_DIR/gat_full_actions_obs_v9_eval100d00i_nonalpha48h_frozen.json}

mkdir -p "$MANIFEST_DIR" /scratch/alpine/$USER/job_output "$OUTPUT_ROOT"
if [[ -f "$MANIFEST" ]]; then
    echo "Reusing frozen checkpoint manifest:"
    echo "  $MANIFEST"
else
    python3 -u examples/amos_2026/evaluate_gat_reward_sweep_mc.py \
        --policy-tags "$POLICY_TAGS" \
        --write-manifest "$MANIFEST"
fi

echo
echo "Submitting full AMOS 2026 GAT MC campaign"
echo "  output root:       $OUTPUT_ROOT"
echo "  manifest:          $MANIFEST"
echo "  policy tags:       $POLICY_TAGS"
echo "  policy count:      $POLICY_COUNT"
echo "  seed blocks:       $START_BLOCK..$END_BLOCK"
echo "  seeds/block:       $SEEDS_PER_BLOCK"
echo "  n_targets:         $N_TARGETS"
echo "  n_targets_ahead:   $N_TARGETS_AHEAD"
echo "  total_time_sec:    $TOTAL_TIME_SEC"
echo "  max concurrent:    $MAX_CONCURRENT policy tasks per active block"
echo "  resources/task:    $CPUS_PER_TASK CPUs, $MEMORY, $TIME_LIMIT"
echo "  dependency:        afterany chain, so only one ten-seed block is active at once"
echo "  skip behavior:     completed matching policy/seed/target-count/time runs are skipped"
echo

previous_job_id=""
for ((seed_start = START_BLOCK; seed_start <= END_BLOCK; seed_start += SEEDS_PER_BLOCK)); do
    seed_end=$((seed_start + SEEDS_PER_BLOCK - 1))
    job_name="gat_mc_${N_TARGETS}t_s$(printf '%03d' "$seed_start")_$(printf '%03d' "$seed_end")"
    sbatch_args=(
        --parsable
        --job-name="$job_name"
        --array="0-${ARRAY_END}%${MAX_CONCURRENT}"
        --time="$TIME_LIMIT"
        --mem="$MEMORY"
        --cpus-per-task="$CPUS_PER_TASK"
        --qos=normal
        --export=ALL,BSK_RL_MC_SEED_START="$seed_start",BSK_RL_MC_SEEDS_PER_BLOCK="$SEEDS_PER_BLOCK",BSK_RL_MC_POLICY_TAGS="$POLICY_TAGS",BSK_RL_MC_N_TARGETS="$N_TARGETS",BSK_RL_MC_N_TARGETS_AHEAD="$N_TARGETS_AHEAD",BSK_RL_MC_EXTRA_TIME_FACTOR="$EXTRA_TIME_FACTOR",BSK_RL_MC_TOTAL_TIME_SEC="$TOTAL_TIME_SEC",BSK_RL_MC_OUTPUT_ROOT="$OUTPUT_ROOT",BSK_RL_MC_MANIFEST="$MANIFEST"
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
echo
echo "Detailed analysis after completion:"
echo "  python examples/amos_2026/analyze_gat_reward_sweep_mc_detailed.py --input-root \"$OUTPUT_ROOT\" --expected-seeds 0:100"
