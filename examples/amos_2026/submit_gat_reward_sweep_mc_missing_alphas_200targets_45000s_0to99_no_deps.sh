#!/bin/bash

# Submit the missing AMOS 2026 GAT reward-sweep Monte Carlo policies.
#
# Default policies:
#   60d40i, 70d30i, 80d20i, 90d10i
#
# The script keeps the existing seed-block layout:
#   <OUTPUT_ROOT>/seeds_000_009/<policy>/seed_000/...
#
# Unlike the full campaign submitters, seed blocks are NOT chained with Slurm
# dependencies. All ten seed-block array jobs are submitted immediately so the
# scheduler can start whichever blocks have available resources.
#
# Usage from /projects/$USER/bsk_rl:
#   bash examples/amos_2026/submit_gat_reward_sweep_mc_missing_alphas_200targets_45000s_0to99_no_deps.sh
#   bash examples/amos_2026/submit_gat_reward_sweep_mc_missing_alphas_200targets_45000s_0to99_no_deps.sh 4
#
# Important overrides:
#   BSK_RL_MC_TARGET_ENV=mixed|leo
#   BSK_RL_MC_OUTPUT_ROOT=/scratch/alpine/$USER/amos2026_mc/<existing-root>
#   BSK_RL_MC_REUSE_LATEST_ROOT=0   # create a new timestamped root instead

set -euo pipefail

MAX_CONCURRENT=${1:-4}
START_BLOCK=${BSK_RL_MC_START_BLOCK:-0}
END_BLOCK=${BSK_RL_MC_END_BLOCK:-90}
SEEDS_PER_BLOCK=${BSK_RL_MC_SEEDS_PER_BLOCK:-10}
POLICY_TAGS=${BSK_RL_MC_POLICY_TAGS:-60d40i,70d30i,80d20i,90d10i}
N_TARGETS=${BSK_RL_MC_N_TARGETS:-200}
N_TARGETS_AHEAD=${BSK_RL_MC_N_TARGETS_AHEAD:-10}
TOTAL_TIME_SEC=${BSK_RL_MC_TOTAL_TIME_SEC:-45000}
EXTRA_TIME_FACTOR=${BSK_RL_MC_EXTRA_TIME_FACTOR:-1.5}
TIME_LIMIT=${BSK_RL_MC_TIME:-04:00:00}
MEMORY=${BSK_RL_MC_MEM:-12G}
CPUS_PER_TASK=${BSK_RL_MC_CPUS_PER_TASK:-4}
TARGET_ENV=${BSK_RL_MC_TARGET_ENV:-mixed}
MIX_WEIGHTS=${BSK_RL_MC_MIX_WEIGHTS:-'{"LEO":0.5,"MEO":0.3,"GEO":0.2}'}
DYNAMIC_PRIORITY_EVENT=${BSK_RL_MC_DYNAMIC_PRIORITY_EVENT:-on}
HIO_COUNT=${BSK_RL_MC_HIO_COUNT:-5}
HIO_PRIORITY=${BSK_RL_MC_HIO_PRIORITY:-5.0}
SHIO_COUNT=${BSK_RL_MC_SHIO_COUNT:-3}
SHIO_PRIORITY=${BSK_RL_MC_SHIO_PRIORITY:-10.0}
REUSE_LATEST_ROOT=${BSK_RL_MC_REUSE_LATEST_ROOT:-1}
MANIFEST_TAG=${BSK_RL_MC_MANIFEST_TAG:-missing60to90}

if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [max-concurrent-per-block: default 4]" >&2
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
if [[ "$TARGET_ENV" != "mixed" && "$TARGET_ENV" != "leo" ]]; then
    echo "BSK_RL_MC_TARGET_ENV must be 'mixed' or 'leo'." >&2
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
POLICY_TAGS_EXPORT=${POLICY_TAGS//,/:}

cd /projects/$USER/bsk_rl
source /projects/$USER/.venv/bin/activate

export BSK_RL_MC_POLICY_TAGS="$POLICY_TAGS"
export BSK_RL_MC_TARGET_ENV="$TARGET_ENV"
export BSK_RL_MC_MIX_WEIGHTS="$MIX_WEIGHTS"
export BSK_RL_MC_DYNAMIC_PRIORITY_EVENT="$DYNAMIC_PRIORITY_EVENT"
export BSK_RL_MC_HIO_COUNT="$HIO_COUNT"
export BSK_RL_MC_HIO_PRIORITY="$HIO_PRIORITY"
export BSK_RL_MC_SHIO_COUNT="$SHIO_COUNT"
export BSK_RL_MC_SHIO_PRIORITY="$SHIO_PRIORITY"

CAMPAIGN_ID=${BSK_RL_MC_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}

if [[ -n "${BSK_RL_MC_OUTPUT_ROOT:-}" ]]; then
    OUTPUT_ROOT="$BSK_RL_MC_OUTPUT_ROOT"
else
    if [[ "$TARGET_ENV" == "mixed" ]]; then
        ROOT_GLOB="gat_full_actions_eval_100d00i_mixed_50LEO30MEO20GEO_${N_TARGETS}targets_${TOTAL_TIME_SEC}s_HIO${HIO_COUNT}_SHIO${SHIO_COUNT}_*"
        NEW_ROOT="/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i_mixed_50LEO30MEO20GEO_${N_TARGETS}targets_${TOTAL_TIME_SEC}s_HIO${HIO_COUNT}_SHIO${SHIO_COUNT}_${CAMPAIGN_ID}"
    else
        ROOT_GLOB="gat_full_actions_eval_100d00i_${N_TARGETS}targets_${TOTAL_TIME_SEC}s_*"
        NEW_ROOT="/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i_${N_TARGETS}targets_${TOTAL_TIME_SEC}s_${CAMPAIGN_ID}"
    fi
    LATEST_ROOT=$(
        { find "/scratch/alpine/$USER/amos2026_mc" -mindepth 1 -maxdepth 1 -type d \
            -name "$ROOT_GLOB" -printf '%T@ %p\n' 2>/dev/null || true; } \
            | sort -nr | head -n 1 | cut -d' ' -f2-
    )
    if [[ "$REUSE_LATEST_ROOT" == "1" && -n "$LATEST_ROOT" ]]; then
        OUTPUT_ROOT="$LATEST_ROOT"
    else
        OUTPUT_ROOT="$NEW_ROOT"
    fi
fi

MANIFEST_DIR="$OUTPUT_ROOT/manifests"
MANIFEST=${BSK_RL_MC_MANIFEST:-$MANIFEST_DIR/gat_full_actions_obs_v9_eval100d00i_${MANIFEST_TAG}_frozen.json}

mkdir -p "$MANIFEST_DIR" /scratch/alpine/$USER/job_output "$OUTPUT_ROOT"
if [[ -f "$MANIFEST" ]]; then
    echo "Reusing frozen missing-alpha checkpoint manifest:"
    echo "  $MANIFEST"
else
    python3 -u examples/amos_2026/evaluate_gat_reward_sweep_mc.py \
        --policy-tags "$POLICY_TAGS" \
        --write-manifest "$MANIFEST"
fi

echo
echo "Submitting missing-alpha AMOS 2026 GAT MC campaign"
echo "  output root:       $OUTPUT_ROOT"
echo "  manifest:          $MANIFEST"
echo "  policy tags:       $POLICY_TAGS"
echo "  policy count:      $POLICY_COUNT"
echo "  seed blocks:       $START_BLOCK..$END_BLOCK"
echo "  seeds/block:       $SEEDS_PER_BLOCK"
echo "  target env:        $TARGET_ENV"
echo "  mix weights:       $MIX_WEIGHTS"
echo "  dynamic priority:  $DYNAMIC_PRIORITY_EVENT"
echo "  HIO/SHIO:          ${HIO_COUNT}x${HIO_PRIORITY}, ${SHIO_COUNT}x${SHIO_PRIORITY}"
echo "  n_targets:         $N_TARGETS"
echo "  n_targets_ahead:   $N_TARGETS_AHEAD"
echo "  total_time_sec:    $TOTAL_TIME_SEC"
echo "  max concurrent:    $MAX_CONCURRENT policy tasks per seed block"
echo "  resources/task:    $CPUS_PER_TASK CPUs, $MEMORY, $TIME_LIMIT"
echo "  dependency:        none; all seed blocks are submitted immediately"
echo "  skip behavior:     completed matching policy/seed/config runs are skipped"
echo

for ((seed_start = START_BLOCK; seed_start <= END_BLOCK; seed_start += SEEDS_PER_BLOCK)); do
    seed_end=$((seed_start + SEEDS_PER_BLOCK - 1))
    if [[ "$TARGET_ENV" == "mixed" ]]; then
        job_name="gat_mc_miss_mix_${N_TARGETS}t_s$(printf '%03d' "$seed_start")_$(printf '%03d' "$seed_end")"
    else
        job_name="gat_mc_miss_leo_${N_TARGETS}t_s$(printf '%03d' "$seed_start")_$(printf '%03d' "$seed_end")"
    fi
    sbatch_args=(
        --parsable
        --job-name="$job_name"
        --array="0-${ARRAY_END}%${MAX_CONCURRENT}"
        --time="$TIME_LIMIT"
        --mem="$MEMORY"
        --cpus-per-task="$CPUS_PER_TASK"
        --qos=normal
        --export=ALL,BSK_RL_MC_SEED_START="$seed_start",BSK_RL_MC_SEEDS_PER_BLOCK="$SEEDS_PER_BLOCK",BSK_RL_MC_POLICY_TAGS="$POLICY_TAGS_EXPORT",BSK_RL_MC_N_TARGETS="$N_TARGETS",BSK_RL_MC_N_TARGETS_AHEAD="$N_TARGETS_AHEAD",BSK_RL_MC_EXTRA_TIME_FACTOR="$EXTRA_TIME_FACTOR",BSK_RL_MC_TOTAL_TIME_SEC="$TOTAL_TIME_SEC",BSK_RL_MC_OUTPUT_ROOT="$OUTPUT_ROOT",BSK_RL_MC_MANIFEST="$MANIFEST",BSK_RL_MC_TARGET_ENV="$TARGET_ENV",BSK_RL_MC_DYNAMIC_PRIORITY_EVENT="$DYNAMIC_PRIORITY_EVENT",BSK_RL_MC_HIO_COUNT="$HIO_COUNT",BSK_RL_MC_HIO_PRIORITY="$HIO_PRIORITY",BSK_RL_MC_SHIO_COUNT="$SHIO_COUNT",BSK_RL_MC_SHIO_PRIORITY="$SHIO_PRIORITY"
    )

    job_id=$(sbatch "${sbatch_args[@]}" examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh)
    echo "Submitted $job_name as job $job_id"
done

echo
echo "Analyze after all blocks finish with:"
echo "  python examples/amos_2026/analyze_gat_reward_sweep_mc.py --input-root \"$OUTPUT_ROOT\" --expected-seeds 0:100"
echo
echo "Detailed analysis after completion:"
echo "  python examples/amos_2026/analyze_gat_reward_sweep_mc_detailed.py --input-root \"$OUTPUT_ROOT\" --expected-seeds 0:100"
