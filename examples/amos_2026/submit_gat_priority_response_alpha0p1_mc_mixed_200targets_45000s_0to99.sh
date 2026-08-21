#!/bin/bash

# Focused AMOS 2026 priority-response campaign:
#   alpha=0.1 policy (10d90i), 100 seeds, mixed 200-target catalog,
#   5 HIOs, 3 SHIOs, and 8 unboosted controls tracked from injection.
#
# Run from /projects/$USER/bsk_rl:
#   bash examples/amos_2026/submit_gat_priority_response_alpha0p1_mc_mixed_200targets_45000s_0to99.sh
#
# The evaluator writes priority_response_targets.csv and verified_deliveries.csv
# for every seed. All ten seed blocks are independent and submitted immediately.

set -euo pipefail

START_BLOCK=${BSK_RL_MC_START_BLOCK:-0}
END_BLOCK=${BSK_RL_MC_END_BLOCK:-90}
SEEDS_PER_BLOCK=${BSK_RL_MC_SEEDS_PER_BLOCK:-10}
POLICY_TAGS=10d90i
N_TARGETS=${BSK_RL_MC_N_TARGETS:-200}
N_TARGETS_AHEAD=${BSK_RL_MC_N_TARGETS_AHEAD:-10}
TOTAL_TIME_SEC=${BSK_RL_MC_TOTAL_TIME_SEC:-45000}
EXTRA_TIME_FACTOR=${BSK_RL_MC_EXTRA_TIME_FACTOR:-1.5}
TIME_LIMIT=${BSK_RL_MC_TIME:-04:00:00}
MEMORY=${BSK_RL_MC_MEM:-12G}
CPUS_PER_TASK=${BSK_RL_MC_CPUS_PER_TASK:-4}
TARGET_ENV=mixed
MIX_WEIGHTS=${BSK_RL_MC_MIX_WEIGHTS:-'{"LEO":0.5,"MEO":0.3,"GEO":0.2}'}
HIO_COUNT=${BSK_RL_MC_HIO_COUNT:-5}
HIO_PRIORITY=${BSK_RL_MC_HIO_PRIORITY:-5.0}
SHIO_COUNT=${BSK_RL_MC_SHIO_COUNT:-3}
SHIO_PRIORITY=${BSK_RL_MC_SHIO_PRIORITY:-10.0}
CONTROL_COUNT=${BSK_RL_MC_PRIORITY_CONTROL_COUNT:-8}

if (( START_BLOCK % SEEDS_PER_BLOCK != 0 || END_BLOCK % SEEDS_PER_BLOCK != 0 || START_BLOCK < 0 || END_BLOCK > 90 || START_BLOCK > END_BLOCK )); then
    echo "Seed blocks must satisfy 0 <= start <= end <= 90 in increments of $SEEDS_PER_BLOCK." >&2
    exit 2
fi

cd /projects/$USER/bsk_rl
source /projects/$USER/.venv/bin/activate

export BSK_RL_MC_POLICY_TAGS="$POLICY_TAGS"
export BSK_RL_MC_TARGET_ENV="$TARGET_ENV"
export BSK_RL_MC_MIX_WEIGHTS="$MIX_WEIGHTS"
export BSK_RL_MC_DYNAMIC_PRIORITY_EVENT=on
export BSK_RL_MC_HIO_COUNT="$HIO_COUNT"
export BSK_RL_MC_HIO_PRIORITY="$HIO_PRIORITY"
export BSK_RL_MC_SHIO_COUNT="$SHIO_COUNT"
export BSK_RL_MC_SHIO_PRIORITY="$SHIO_PRIORITY"
export BSK_RL_MC_PRIORITY_CONTROL_COUNT="$CONTROL_COUNT"

CAMPAIGN_ID=${BSK_RL_MC_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${BSK_RL_MC_OUTPUT_ROOT:-/scratch/alpine/$USER/amos2026_mc/gat_priority_response_alpha0p1_mixed_200targets_45000s_HIO5_SHIO3_controls${CONTROL_COUNT}_${CAMPAIGN_ID}}
MANIFEST=${BSK_RL_MC_MANIFEST:-$OUTPUT_ROOT/manifests/gat_full_actions_obs_v9_10d90i_frozen.json}

mkdir -p "$OUTPUT_ROOT/manifests" /scratch/alpine/$USER/job_output
if [[ ! -f "$MANIFEST" ]]; then
    python3 -u examples/amos_2026/evaluate_gat_reward_sweep_mc.py \
        --policy-tags "$POLICY_TAGS" \
        --write-manifest "$MANIFEST"
fi

echo "Submitting focused alpha=0.1 priority-response campaign"
echo "  output root: $OUTPUT_ROOT"
echo "  manifest:    $MANIFEST"
echo "  controls:    $CONTROL_COUNT"
echo "  seed blocks: $START_BLOCK..$END_BLOCK"

for ((seed_start = START_BLOCK; seed_start <= END_BLOCK; seed_start += SEEDS_PER_BLOCK)); do
    seed_end=$((seed_start + SEEDS_PER_BLOCK - 1))
    job_name="gat_prio_a01_s$(printf '%03d' "$seed_start")_$(printf '%03d' "$seed_end")"
    job_id=$(
        sbatch \
            --parsable \
            --job-name="$job_name" \
            --array=0-0 \
            --time="$TIME_LIMIT" \
            --mem="$MEMORY" \
            --cpus-per-task="$CPUS_PER_TASK" \
            --qos=normal \
            --export=ALL,BSK_RL_MC_SEED_START="$seed_start",BSK_RL_MC_SEEDS_PER_BLOCK="$SEEDS_PER_BLOCK",BSK_RL_MC_N_TARGETS="$N_TARGETS",BSK_RL_MC_N_TARGETS_AHEAD="$N_TARGETS_AHEAD",BSK_RL_MC_EXTRA_TIME_FACTOR="$EXTRA_TIME_FACTOR",BSK_RL_MC_TOTAL_TIME_SEC="$TOTAL_TIME_SEC",BSK_RL_MC_OUTPUT_ROOT="$OUTPUT_ROOT",BSK_RL_MC_MANIFEST="$MANIFEST" \
            examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh
    )
    echo "Submitted $job_name as job $job_id"
done

echo
echo "After completion, package the exact response outputs with:"
echo "  find \"$OUTPUT_ROOT\" -type f \\( -name 'priority_response_targets.csv' -o -name 'verified_deliveries.csv' -o -name 'metrics_*.json' -o -name 'mc_status.json' \\) -print0 | tar --null -T - -czf \"$OUTPUT_ROOT/priority_response_alpha0p1_exact_outputs.tgz\""
echo
echo "Analyze the completed campaign with:"
echo "  python examples/amos_2026/analyze_gat_priority_response_mc.py --input-root \"$OUTPUT_ROOT\" --expected-seeds 0:100"
