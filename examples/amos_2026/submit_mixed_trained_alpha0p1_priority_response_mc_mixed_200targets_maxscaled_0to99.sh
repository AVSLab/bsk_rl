#!/usr/bin/env bash

# Evaluate the mixed-100-trained alpha=0.1 policy on 100 independent mixed-200
# episodes. Initial priorities are sampled uniformly on [0, 2] and rescaled to
# sum 200. At the midpoint event, HIO and SHIO priorities become 5x and 10x the
# realized maximum initial priority for that seed. Eight unpromoted controls are
# tracked from the same event time.

set -euo pipefail

START_BLOCK=${BSK_RL_MC_START_BLOCK:-0}
END_BLOCK=${BSK_RL_MC_END_BLOCK:-90}
SEEDS_PER_BLOCK=${BSK_RL_MC_SEEDS_PER_BLOCK:-10}
POLICY_TAG=mixed_a0p1
N_TARGETS=200
N_TARGETS_AHEAD=10
PRIORITY_SUM=200.0
PRIORITY_UNIFORM_LOW=0.0
PRIORITY_UNIFORM_HIGH=2.0
TOTAL_TIME_SEC=45000
TARGET_ENV=mixed
MIX_WEIGHTS='{"LEO":0.5,"MEO":0.3,"GEO":0.2}'
HIO_COUNT=5
HIO_MAX_MULTIPLIER=5.0
SHIO_COUNT=3
SHIO_MAX_MULTIPLIER=10.0
CONTROL_COUNT=8
TIME_LIMIT=${BSK_RL_MC_TIME:-04:00:00}
MEMORY=${BSK_RL_MC_MEM:-24G}
CPUS_PER_TASK=${BSK_RL_MC_CPUS_PER_TASK:-4}

SOURCE_MC_ROOT=${BSK_RL_MIXED_TRAINED_SOURCE_ROOT:-/scratch/alpine/$USER/amos2026_mc/gat_mixed_trained_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_45000s_HIO5_SHIO3_20260802}
CUSTOM_POLICIES_JSON=${BSK_RL_MIXED_CUSTOM_POLICIES_JSON:-$SOURCE_MC_ROOT/manifests/mixed_fixed100_custom_policies.json}
CAMPAIGN_ID=${BSK_RL_MC_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${BSK_RL_MC_OUTPUT_ROOT:-/scratch/alpine/$USER/amos2026_mc/gat_mixedtrained_a0p1_priority_response_mixed_exact100LEO60MEO40GEO_200targets_prioritySum200_HIO5xMax_SHIO10xMax_controls8_45000s_$CAMPAIGN_ID}
MANIFEST=${BSK_RL_MC_MANIFEST:-$OUTPUT_ROOT/manifests/mixed_trained_a0p1_frozen.json}

if (( START_BLOCK % SEEDS_PER_BLOCK != 0 || END_BLOCK % SEEDS_PER_BLOCK != 0 || START_BLOCK < 0 || END_BLOCK > 90 || START_BLOCK > END_BLOCK )); then
    echo "Seed blocks must satisfy 0 <= start <= end <= 90 in increments of $SEEDS_PER_BLOCK." >&2
    exit 2
fi

cd "/projects/$USER/bsk_rl"
source "/projects/$USER/.venv/bin/activate"

if [[ ! -f "$CUSTOM_POLICIES_JSON" ]]; then
    echo "Mixed-trained policy specification not found: $CUSTOM_POLICIES_JSON" >&2
    exit 3
fi

python3 - "$CUSTOM_POLICIES_JSON" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
policy = payload.get("policies", {}).get("mixed_a0p1")
if policy is None:
    raise SystemExit(f"mixed_a0p1 is missing from {path}")
if float(policy.get("alpha", -1.0)) != 0.1:
    raise SystemExit(f"mixed_a0p1 has unexpected alpha in {path}: {policy.get('alpha')}")
checkpoint = pathlib.Path(policy["checkpoint_dir"])
if not checkpoint.is_dir():
    raise SystemExit(f"mixed_a0p1 checkpoint directory is missing: {checkpoint}")
print(f"Validated mixed_a0p1 checkpoint: {checkpoint}")
PY

mkdir -p "$OUTPUT_ROOT/manifests" "/scratch/alpine/$USER/job_output"
if [[ ! -f "$MANIFEST" ]]; then
    python3 -u examples/amos_2026/evaluate_gat_reward_sweep_mc.py \
        --policy-tags "$POLICY_TAG" \
        --custom-policies-json "@$CUSTOM_POLICIES_JSON" \
        --write-manifest "$MANIFEST"
fi

cat > "$OUTPUT_ROOT/manifests/campaign_configuration.env" <<EOF
POLICY_TAG=$POLICY_TAG
POLICY_TRAINING_CATALOG=exact_50LEO_30MEO_20GEO_100targets
EVALUATION_CATALOG=exact_100LEO_60MEO_40GEO_200targets
N_TARGETS=$N_TARGETS
N_TARGETS_AHEAD=$N_TARGETS_AHEAD
BASELINE_PRIORITY_DISTRIBUTION=uniform_raw_then_rescaled
PRIORITY_UNIFORM_LOW=$PRIORITY_UNIFORM_LOW
PRIORITY_UNIFORM_HIGH=$PRIORITY_UNIFORM_HIGH
PRIORITY_SUM=$PRIORITY_SUM
INITIAL_MEAN_PRIORITY=1.0
HIO_COUNT=$HIO_COUNT
HIO_PRIORITY_FORMULA=${HIO_MAX_MULTIPLIER}x_realized_initial_priority_max
SHIO_COUNT=$SHIO_COUNT
SHIO_PRIORITY_FORMULA=${SHIO_MAX_MULTIPLIER}x_realized_initial_priority_max
CONTROL_COUNT=$CONTROL_COUNT
TOTAL_TIME_SEC=$TOTAL_TIME_SEC
SEEDS=0:100
SLURM_DEPENDENCIES=none
PRIORITY_RESPONSE_TELEMETRY=eligible_visible_access_v1
EOF

export BSK_RL_MC_CUSTOM_POLICIES_JSON="@$CUSTOM_POLICIES_JSON"
export BSK_RL_MC_POLICY_TAGS="$POLICY_TAG"
export BSK_RL_MC_N_TARGETS="$N_TARGETS"
export BSK_RL_MC_PRIORITY_SUM="$PRIORITY_SUM"
export BSK_RL_MC_PRIORITY_UNIFORM_LOW="$PRIORITY_UNIFORM_LOW"
export BSK_RL_MC_PRIORITY_UNIFORM_HIGH="$PRIORITY_UNIFORM_HIGH"
export BSK_RL_MC_N_TARGETS_AHEAD="$N_TARGETS_AHEAD"
export BSK_RL_MC_TOTAL_TIME_SEC="$TOTAL_TIME_SEC"
export BSK_RL_MC_TARGET_ENV="$TARGET_ENV"
export BSK_RL_MC_MIX_WEIGHTS="$MIX_WEIGHTS"
export BSK_RL_MC_EXACT_MIX_COUNTS=1
export BSK_RL_MC_DYNAMIC_PRIORITY_EVENT=on
export BSK_RL_MC_HIO_COUNT="$HIO_COUNT"
export BSK_RL_MC_HIO_PRIORITY_MAX_MULTIPLIER="$HIO_MAX_MULTIPLIER"
export BSK_RL_MC_SHIO_COUNT="$SHIO_COUNT"
export BSK_RL_MC_SHIO_PRIORITY_MAX_MULTIPLIER="$SHIO_MAX_MULTIPLIER"
export BSK_RL_MC_PRIORITY_CONTROL_COUNT="$CONTROL_COUNT"

submission_log="$OUTPUT_ROOT/manifests/submitted_jobs.tsv"
printf 'seed_start\tseed_end\tjob_id\tjob_name\n' > "$submission_log"
for ((seed_start = START_BLOCK; seed_start <= END_BLOCK; seed_start += SEEDS_PER_BLOCK)); do
    seed_end=$((seed_start + SEEDS_PER_BLOCK - 1))
    job_name="mxmaxpr_s$(printf '%03d' "$seed_start")_$(printf '%03d' "$seed_end")"
    job_id=$(sbatch \
        --parsable \
        --job-name="$job_name" \
        --array=0-0 \
        --time="$TIME_LIMIT" \
        --mem="$MEMORY" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --qos=normal \
        --export=ALL,BSK_RL_MC_SEED_START="$seed_start",BSK_RL_MC_SEEDS_PER_BLOCK="$SEEDS_PER_BLOCK",BSK_RL_MC_OUTPUT_ROOT="$OUTPUT_ROOT",BSK_RL_MC_MANIFEST="$MANIFEST" \
        examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh)
    printf '%d\t%d\t%s\t%s\n' "$seed_start" "$seed_end" "$job_id" "$job_name" \
        | tee -a "$submission_log"
done

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$OUTPUT_ROOT/manifests/SUBMISSION_COMPLETE_UTC.txt"
echo
echo "Submitted ten independent seed-block jobs with no Slurm dependencies."
echo "Output root: $OUTPUT_ROOT"
echo "After completion, run:"
echo "  python examples/amos_2026/analyze_gat_priority_response_mc.py --input-root '$OUTPUT_ROOT' --expected-seeds 0:100"
