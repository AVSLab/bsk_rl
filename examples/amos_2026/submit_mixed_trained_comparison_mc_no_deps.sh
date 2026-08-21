#!/usr/bin/env bash

# Evaluate the eight mixed-fixed-100 GAT policies on the exact mixed-100
# catalog used for the LEO-trained transfer baseline. All seed-block arrays are
# independent, and completed matching runs are skipped on restart.

set -euo pipefail

MAX_CONCURRENT_PER_BLOCK=${1:-2}
CUSTOM_POLICIES_JSON=${BSK_RL_MIXED_CUSTOM_POLICIES_JSON:?Set BSK_RL_MIXED_CUSTOM_POLICIES_JSON}
POLICY_TAGS=${BSK_RL_MIXED_POLICY_TAGS:?Set BSK_RL_MIXED_POLICY_TAGS}
OUTPUT_ROOT=${BSK_RL_MIXED_COMPARISON_ROOT:-/scratch/alpine/$USER/amos2026_mc/gat_mixed_trained_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_45000s_HIO5_SHIO3_20260801}
MANIFEST="$OUTPUT_ROOT/manifests/mixed_trained_eight_policy_frozen.json"
SEEDS_PER_BLOCK=10

if ! [[ "$MAX_CONCURRENT_PER_BLOCK" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [max-concurrent-per-block: default 2]" >&2
    exit 2
fi

POLICY_COUNT=$(python3 - "$POLICY_TAGS" <<'PY'
import re
import sys
tags = [tag.strip() for tag in re.split(r"[,;:]", sys.argv[1]) if tag.strip()]
print(len(tags))
PY
)
if [[ "$POLICY_COUNT" != "8" ]]; then
    echo "Expected eight mixed-trained policies, found $POLICY_COUNT." >&2
    exit 3
fi
ARRAY_END=$((POLICY_COUNT - 1))
POLICY_TAGS_EXPORT=${POLICY_TAGS//,/:}

cd "/projects/$USER/bsk_rl"
source "/projects/$USER/.venv/bin/activate"
mkdir -p "$OUTPUT_ROOT/manifests" "/scratch/alpine/$USER/job_output"

if [[ ! -f "$MANIFEST" ]]; then
    python3 -u examples/amos_2026/evaluate_gat_reward_sweep_mc.py \
        --policy-tags "$POLICY_TAGS" \
        --custom-policies-json "@$CUSTOM_POLICIES_JSON" \
        --write-manifest "$MANIFEST"
fi

export BSK_RL_MC_CUSTOM_POLICIES_JSON="@$CUSTOM_POLICIES_JSON"
export BSK_RL_MC_POLICY_TAGS="$POLICY_TAGS"
export BSK_RL_MC_N_TARGETS=100
export BSK_RL_MC_PRIORITY_SUM=100.0
export BSK_RL_MC_N_TARGETS_AHEAD=10
export BSK_RL_MC_TOTAL_TIME_SEC=45000
export BSK_RL_MC_TARGET_ENV=mixed
export BSK_RL_MC_MIX_WEIGHTS='{"LEO":0.5,"MEO":0.3,"GEO":0.2}'
export BSK_RL_MC_EXACT_MIX_COUNTS=1
export BSK_RL_MC_DYNAMIC_PRIORITY_EVENT=on
export BSK_RL_MC_HIO_COUNT=5
export BSK_RL_MC_HIO_PRIORITY=5.0
export BSK_RL_MC_SHIO_COUNT=3
export BSK_RL_MC_SHIO_PRIORITY=10.0
export BSK_RL_MC_PRIORITY_CONTROL_COUNT=0

submission_log="$OUTPUT_ROOT/manifests/submitted_jobs.tsv"
printf 'seed_start\tseed_end\tjob_id\tjob_name\n' > "$submission_log"
for seed_start in 0 10 20 30 40 50 60 70 80 90; do
    seed_end=$((seed_start + SEEDS_PER_BLOCK - 1))
    job_name="mixtr_s$(printf '%03d' "$seed_start")_$(printf '%03d' "$seed_end")"
    job_id=$(sbatch \
        --parsable \
        --job-name="$job_name" \
        --array="0-${ARRAY_END}%${MAX_CONCURRENT_PER_BLOCK}" \
        --time=04:00:00 \
        --mem=12G \
        --cpus-per-task=4 \
        --qos=normal \
        --export=ALL,BSK_RL_MC_SEED_START="$seed_start",BSK_RL_MC_SEEDS_PER_BLOCK="$SEEDS_PER_BLOCK",BSK_RL_MC_POLICY_TAGS="$POLICY_TAGS_EXPORT",BSK_RL_MC_OUTPUT_ROOT="$OUTPUT_ROOT",BSK_RL_MC_MANIFEST="$MANIFEST" \
        examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh)
    printf '%d\t%d\t%s\t%s\n' "$seed_start" "$seed_end" "$job_id" "$job_name" \
        | tee -a "$submission_log"
done

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$OUTPUT_ROOT/manifests/SUBMISSION_COMPLETE_UTC.txt"
echo "Submitted ten independent mixed-trained comparison arrays."
echo "Output root: $OUTPUT_ROOT"
