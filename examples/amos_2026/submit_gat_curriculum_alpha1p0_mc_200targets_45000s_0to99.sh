#!/bin/bash

# Submit a curriculum-policy AMOS 2026 GAT MC campaign.
#
# Default policy:
#   final checkpoint of the 0.0->1.0 curriculum run, evaluated at alpha=1.0.
#
# This submits ten chained Slurm array jobs:
#   seeds 000..009, 010..019, ..., 090..099
# Each array job has one task for the custom curriculum policy. The task runs
# its ten seeds as fresh evaluator subprocesses, matching the reward-sweep MC
# layout while keeping the custom policy visually/analytically distinct.
#
# Usage from /projects/$USER/bsk_rl:
#   bash examples/amos_2026/submit_gat_curriculum_alpha1p0_mc_200targets_45000s_0to99.sh
#
# To add the curriculum runs directly under an existing MC root, set:
#   BSK_RL_MC_OUTPUT_ROOT=/scratch/alpine/$USER/amos2026_mc/<existing-root> \
#     bash examples/amos_2026/submit_gat_curriculum_alpha1p0_mc_200targets_45000s_0to99.sh
#
# To evaluate the 24-hour alpha=1.0 continuation later, point
# BSK_RL_MC_CURRICULUM_FROM at that copied continued-training run and use a
# different BSK_RL_MC_CURRICULUM_TAG.

set -euo pipefail

MAX_CONCURRENT=${1:-1}
START_BLOCK=${BSK_RL_MC_START_BLOCK:-0}
END_BLOCK=${BSK_RL_MC_END_BLOCK:-90}
SEEDS_PER_BLOCK=${BSK_RL_MC_SEEDS_PER_BLOCK:-10}
POLICY_TAG=${BSK_RL_MC_CURRICULUM_TAG:-curriculum_alpha1p0}
POLICY_TAGS=${BSK_RL_MC_POLICY_TAGS:-$POLICY_TAG}
POLICY_LABEL=${BSK_RL_MC_CURRICULUM_LABEL:-Curriculum alpha=1.0}
POLICY_COLOR=${BSK_RL_MC_CURRICULUM_COLOR:-#5BC5DB}
N_TARGETS=${BSK_RL_MC_N_TARGETS:-200}
N_TARGETS_AHEAD=${BSK_RL_MC_N_TARGETS_AHEAD:-10}
TOTAL_TIME_SEC=${BSK_RL_MC_TOTAL_TIME_SEC:-45000}
EXTRA_TIME_FACTOR=${BSK_RL_MC_EXTRA_TIME_FACTOR:-1.5}
TIME_LIMIT=${BSK_RL_MC_TIME:-04:00:00}
MEMORY=${BSK_RL_MC_MEM:-12G}
CPUS_PER_TASK=${BSK_RL_MC_CPUS_PER_TASK:-4}
TARGET_ENV=${BSK_RL_MC_TARGET_ENV:-leo}
MIX_WEIGHTS=${BSK_RL_MC_MIX_WEIGHTS:-'{"LEO":0.5,"MEO":0.3,"GEO":0.2}'}
DYNAMIC_PRIORITY_EVENT=${BSK_RL_MC_DYNAMIC_PRIORITY_EVENT:-on}
HIO_COUNT=${BSK_RL_MC_HIO_COUNT:-5}
HIO_PRIORITY=${BSK_RL_MC_HIO_PRIORITY:-5.0}
SHIO_COUNT=${BSK_RL_MC_SHIO_COUNT:-3}
SHIO_PRIORITY=${BSK_RL_MC_SHIO_PRIORITY:-10.0}

DEFAULT_CURRICULUM_POLICY_PATH="/scratch/alpine/$USER/rllib_results/amos2026_LEO_GAT_fullActions_curriculum00d100iTo100d00i_4200batch_restrictedResources_obs-v9_hold10s_reimage2orb_prioritySum100_1781685771.185523/amos2026_LEO_GAT_fullActions_curriculum00d100iTo100d00i_4200batch_restrictedResources_obs-v9_hold10s_reimage2orb_prioritySum100.out_0"
CURRICULUM_POLICY_PATH=${BSK_RL_MC_CURRICULUM_FROM:-$DEFAULT_CURRICULUM_POLICY_PATH}

if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [max-concurrent: default 1]" >&2
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
if [[ ! -d "$CURRICULUM_POLICY_PATH" ]]; then
    echo "Curriculum policy path not found: $CURRICULUM_POLICY_PATH" >&2
    echo "Set BSK_RL_MC_CURRICULUM_FROM to the model dir, run dir, or checkpoint dir to evaluate." >&2
    exit 1
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
OUTPUT_ROOT=${BSK_RL_MC_OUTPUT_ROOT:-/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i_${TARGET_ENV}_${N_TARGETS}targets_${TOTAL_TIME_SEC}s_${POLICY_TAG}_${CAMPAIGN_ID}}
MANIFEST_DIR="$OUTPUT_ROOT/manifests"
MANIFEST=${BSK_RL_MC_MANIFEST:-$MANIFEST_DIR/gat_full_actions_obs_v9_eval100d00i_${POLICY_TAG}_frozen.json}
CUSTOM_POLICY_JSON_PATH="$MANIFEST_DIR/${POLICY_TAG}_policy.json"

mkdir -p "$MANIFEST_DIR" /scratch/alpine/$USER/job_output "$OUTPUT_ROOT"

export CURRICULUM_POLICY_PATH POLICY_TAG POLICY_LABEL POLICY_COLOR CUSTOM_POLICY_JSON_PATH
python3 - <<'PY'
import json
import os
from pathlib import Path

payload = {
    os.environ["POLICY_TAG"]: {
        "checkpoint_dir": os.environ["CURRICULUM_POLICY_PATH"],
        "alpha": 1.0,
        "color": os.environ["POLICY_COLOR"],
        "label": os.environ["POLICY_LABEL"],
    }
}
path = Path(os.environ["CUSTOM_POLICY_JSON_PATH"])
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
export BSK_RL_MC_CUSTOM_POLICIES_JSON="@$CUSTOM_POLICY_JSON_PATH"

if [[ -f "$MANIFEST" && "${BSK_RL_MC_REFRESH_MANIFEST:-0}" != "1" ]]; then
    echo "Reusing frozen checkpoint manifest:"
    echo "  $MANIFEST"
else
    python3 -u examples/amos_2026/evaluate_gat_reward_sweep_mc.py \
        --policy-tags "$POLICY_TAGS" \
        --custom-policies-json "$BSK_RL_MC_CUSTOM_POLICIES_JSON" \
        --write-manifest "$MANIFEST"
fi

echo
echo "Submitting curriculum AMOS 2026 GAT MC campaign"
echo "  output root:       $OUTPUT_ROOT"
echo "  manifest:          $MANIFEST"
echo "  custom policy json:$CUSTOM_POLICY_JSON_PATH"
echo "  policy path:       $CURRICULUM_POLICY_PATH"
echo "  policy tags:       $POLICY_TAGS"
echo "  policy label/color:$POLICY_LABEL / $POLICY_COLOR"
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
echo "  max concurrent:    $MAX_CONCURRENT policy tasks per active block"
echo "  resources/task:    $CPUS_PER_TASK CPUs, $MEMORY, $TIME_LIMIT"
echo "  dependency:        afterany chain, so only one ten-seed block is active at once"
echo "  skip behavior:     completed matching policy/seed/config runs are skipped"
echo

previous_job_id=""
for ((seed_start = START_BLOCK; seed_start <= END_BLOCK; seed_start += SEEDS_PER_BLOCK)); do
    seed_end=$((seed_start + SEEDS_PER_BLOCK - 1))
    job_name="gat_mc_curr_s$(printf '%03d' "$seed_start")_$(printf '%03d' "$seed_end")"
    sbatch_args=(
        --parsable
        --job-name="$job_name"
        --array="0-${ARRAY_END}%${MAX_CONCURRENT}"
        --time="$TIME_LIMIT"
        --mem="$MEMORY"
        --cpus-per-task="$CPUS_PER_TASK"
        --qos=normal
        --export=ALL,BSK_RL_MC_SEED_START="$seed_start",BSK_RL_MC_SEEDS_PER_BLOCK="$SEEDS_PER_BLOCK",BSK_RL_MC_POLICY_TAGS="$POLICY_TAGS_EXPORT",BSK_RL_MC_CUSTOM_POLICIES_JSON="$BSK_RL_MC_CUSTOM_POLICIES_JSON",BSK_RL_MC_N_TARGETS="$N_TARGETS",BSK_RL_MC_N_TARGETS_AHEAD="$N_TARGETS_AHEAD",BSK_RL_MC_EXTRA_TIME_FACTOR="$EXTRA_TIME_FACTOR",BSK_RL_MC_TOTAL_TIME_SEC="$TOTAL_TIME_SEC",BSK_RL_MC_OUTPUT_ROOT="$OUTPUT_ROOT",BSK_RL_MC_MANIFEST="$MANIFEST",BSK_RL_MC_TARGET_ENV="$TARGET_ENV",BSK_RL_MC_MIX_WEIGHTS="$MIX_WEIGHTS",BSK_RL_MC_DYNAMIC_PRIORITY_EVENT="$DYNAMIC_PRIORITY_EVENT",BSK_RL_MC_HIO_COUNT="$HIO_COUNT",BSK_RL_MC_HIO_PRIORITY="$HIO_PRIORITY",BSK_RL_MC_SHIO_COUNT="$SHIO_COUNT",BSK_RL_MC_SHIO_PRIORITY="$SHIO_PRIORITY"
    )
    if [[ -n "$previous_job_id" ]]; then
        sbatch_args+=(--dependency="afterany:$previous_job_id")
    fi

    job_id=$(sbatch "${sbatch_args[@]}" examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh)
    echo "Submitted $job_name as job $job_id"
    previous_job_id="$job_id"
done

echo
echo "Quick status after submission:"
echo "  squeue -u $USER"
echo
echo "Detailed analysis after completion:"
echo "  python examples/amos_2026/analyze_gat_reward_sweep_mc_detailed.py --input-root \"$OUTPUT_ROOT\" --expected-seeds 0:100"
