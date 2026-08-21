#!/usr/bin/env bash

# Submit corrected LEO-200 and mixed-regime-200 AMOS 2026 density evaluations.
# Every catalog begins with total priority 200, so its mean initial target
# priority is one. All twenty seed-block arrays are submitted independently.

set -euo pipefail

MAX_CONCURRENT_PER_BLOCK=${1:-2}
POLICY_TAGS=${BSK_RL_MC_POLICY_TAGS:-00d100i,10d90i,20d80i,30d70i,40d60i,50d50i,75d25i,100d00i}
N_TARGETS=200
PRIORITY_SUM=200.0
N_TARGETS_AHEAD=10
TOTAL_TIME_SEC=45000
SEEDS_PER_BLOCK=10
TIME_LIMIT=${BSK_RL_MC_TIME:-04:00:00}
MEMORY=${BSK_RL_MC_MEM:-12G}
CPUS_PER_TASK=${BSK_RL_MC_CPUS_PER_TASK:-4}
MIX_WEIGHTS='{"LEO":0.5,"MEO":0.3,"GEO":0.2}'
CAMPAIGN_ID=${BSK_RL_MC_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
BASE_ROOT=${BSK_RL_MC_BASE_ROOT:-/scratch/alpine/$USER/amos2026_mc/corrected_density_prioritySum200_${CAMPAIGN_ID}}
LEO_OUTPUT_ROOT="$BASE_ROOT/gat_leo_200targets_prioritySum200_${TOTAL_TIME_SEC}s_HIO5_SHIO3"
MIXED_OUTPUT_ROOT="$BASE_ROOT/gat_mixed_200targets_prioritySum200_${TOTAL_TIME_SEC}s_HIO5_SHIO3"
MANIFEST="$BASE_ROOT/manifests/gat_full_actions_obs_v9_eval100d00i_eight_policy_frozen.json"

if ! [[ "$MAX_CONCURRENT_PER_BLOCK" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [max-concurrent-per-block: default 2]" >&2
    exit 2
fi

POLICY_COUNT=$(python3 - "$POLICY_TAGS" <<'PY'
import re
import sys

tags = [tag.strip() for tag in re.split(r"[,;:]", sys.argv[1]) if tag.strip()]
if not tags:
    raise SystemExit("BSK_RL_MC_POLICY_TAGS cannot be empty")
print(len(tags))
PY
)
ARRAY_END=$((POLICY_COUNT - 1))
POLICY_TAGS_EXPORT=${POLICY_TAGS//,/:}

cd "/projects/$USER/bsk_rl"
source "/projects/$USER/.venv/bin/activate"

mkdir -p "$BASE_ROOT/manifests" "$LEO_OUTPUT_ROOT" "$MIXED_OUTPUT_ROOT" \
    "/scratch/alpine/$USER/job_output"

if [[ ! -f "$MANIFEST" ]]; then
    python3 -u examples/amos_2026/evaluate_gat_reward_sweep_mc.py \
        --policy-tags "$POLICY_TAGS" \
        --write-manifest "$MANIFEST"
fi

printf '%s\n' \
    "CAMPAIGN_ID='$CAMPAIGN_ID'" \
    "BASE_ROOT='$BASE_ROOT'" \
    "LEO_OUTPUT_ROOT='$LEO_OUTPUT_ROOT'" \
    "MIXED_OUTPUT_ROOT='$MIXED_OUTPUT_ROOT'" \
    "MANIFEST='$MANIFEST'" \
    "POLICY_TAGS='$POLICY_TAGS'" \
    "N_TARGETS='$N_TARGETS'" \
    "PRIORITY_SUM='$PRIORITY_SUM'" \
    > "$BASE_ROOT/campaign_paths.env"

# These values contain punctuation that is awkward inside Slurm's comma-delimited
# --export argument. Exporting them here lets --export=ALL carry them unchanged.
export BSK_RL_MC_POLICY_TAGS="$POLICY_TAGS"
export BSK_RL_MC_PRIORITY_SUM="$PRIORITY_SUM"
export BSK_RL_MC_N_TARGETS="$N_TARGETS"
export BSK_RL_MC_N_TARGETS_AHEAD="$N_TARGETS_AHEAD"
export BSK_RL_MC_TOTAL_TIME_SEC="$TOTAL_TIME_SEC"
export BSK_RL_MC_MIX_WEIGHTS="$MIX_WEIGHTS"
export BSK_RL_MC_EXACT_MIX_COUNTS=0
export BSK_RL_MC_DYNAMIC_PRIORITY_EVENT=on
export BSK_RL_MC_HIO_COUNT=5
export BSK_RL_MC_HIO_PRIORITY=5.0
export BSK_RL_MC_SHIO_COUNT=3
export BSK_RL_MC_SHIO_PRIORITY=10.0
export BSK_RL_MC_PRIORITY_CONTROL_COUNT=0

echo "Submitting corrected density campaigns"
echo "  base root:             $BASE_ROOT"
echo "  policies:              $POLICY_TAGS"
echo "  target count:          $N_TARGETS"
echo "  initial priority sum:  $PRIORITY_SUM"
echo "  mean initial priority: 1.0"
echo "  seed blocks:           000-009 through 090-099"
echo "  dependencies:          none"
echo "  max per block:         $MAX_CONCURRENT_PER_BLOCK"
echo

submission_log="$BASE_ROOT/submitted_jobs.tsv"
printf 'environment\tseed_start\tseed_end\tjob_id\tjob_name\n' > "$submission_log"

submit_environment() {
    local target_env=$1
    local output_root=$2
    local short_env=$3

    for seed_start in 0 10 20 30 40 50 60 70 80 90; do
        local seed_end=$((seed_start + SEEDS_PER_BLOCK - 1))
        local job_name
        job_name="ps200_${short_env}_s$(printf '%03d' "$seed_start")_$(printf '%03d' "$seed_end")"
        local job_id
        job_id=$(sbatch \
            --parsable \
            --job-name="$job_name" \
            --array="0-${ARRAY_END}%${MAX_CONCURRENT_PER_BLOCK}" \
            --time="$TIME_LIMIT" \
            --mem="$MEMORY" \
            --cpus-per-task="$CPUS_PER_TASK" \
            --qos=normal \
            --export=ALL,BSK_RL_MC_SEED_START="$seed_start",BSK_RL_MC_SEEDS_PER_BLOCK="$SEEDS_PER_BLOCK",BSK_RL_MC_POLICY_TAGS="$POLICY_TAGS_EXPORT",BSK_RL_MC_TARGET_ENV="$target_env",BSK_RL_MC_OUTPUT_ROOT="$output_root",BSK_RL_MC_MANIFEST="$MANIFEST" \
            examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh)
        printf '%s\t%d\t%d\t%s\t%s\n' \
            "$target_env" "$seed_start" "$seed_end" "$job_id" "$job_name" \
            | tee -a "$submission_log"
    done
}

submit_environment leo "$LEO_OUTPUT_ROOT" l
submit_environment mixed "$MIXED_OUTPUT_ROOT" m

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$BASE_ROOT/SUBMISSION_COMPLETE_UTC.txt"

echo
echo "Submitted 20 independent arrays. Slurm may hold excess tasks for QOS limits."
echo "Campaign paths: $BASE_ROOT/campaign_paths.env"
echo "Job record:     $submission_log"
echo
echo "Queue view:"
echo "  squeue -u \"$USER\" --format='%.18i %.26j %.2t %.10M %.10l %.6D %R' | grep -E 'JOBID|ps200_'"
