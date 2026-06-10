#!/bin/bash

# Freeze one checkpoint per GAT full-action reward-mix policy and submit a
# 200-target Monte Carlo block. Each array task owns one policy and runs ten
# seeds sequentially as fresh evaluator subprocesses.
#
# Usage from /projects/$USER/bsk_rl:
#   bash examples/amos_2026/submit_gat_reward_sweep_mc_200targets_10seeds.sh
#   bash examples/amos_2026/submit_gat_reward_sweep_mc_200targets_10seeds.sh 0 5
#   bash examples/amos_2026/submit_gat_reward_sweep_mc_200targets_10seeds.sh 10 5
#
# Optional resource overrides:
#   BSK_RL_MC_TIME=06:00:00 BSK_RL_MC_MEM=16G BSK_RL_MC_CPUS_PER_TASK=4 bash ...

set -euo pipefail

SEED_START=${1:-0}
MAX_CONCURRENT=${2:-5}
SEEDS_PER_BLOCK=${BSK_RL_MC_SEEDS_PER_BLOCK:-10}
N_TARGETS=${BSK_RL_MC_N_TARGETS:-200}
N_TARGETS_AHEAD=${BSK_RL_MC_N_TARGETS_AHEAD:-10}
EXTRA_TIME_FACTOR=${BSK_RL_MC_EXTRA_TIME_FACTOR:-1.5}
TIME_LIMIT=${BSK_RL_MC_TIME:-04:00:00}
MEMORY=${BSK_RL_MC_MEM:-12G}
CPUS_PER_TASK=${BSK_RL_MC_CPUS_PER_TASK:-4}

if ! [[ "$SEED_START" =~ ^[0-9]+$ ]] || ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 <seed-start: 0,10,...,90> [max-concurrent: default 5]" >&2
    exit 2
fi
if ! [[ "$SEEDS_PER_BLOCK" =~ ^[1-9][0-9]*$ ]] || ! [[ "$N_TARGETS" =~ ^[1-9][0-9]*$ ]]; then
    echo "BSK_RL_MC_SEEDS_PER_BLOCK and BSK_RL_MC_N_TARGETS must be positive integers" >&2
    exit 2
fi
if (( SEED_START % SEEDS_PER_BLOCK != 0 || SEED_START > 90 )); then
    echo "seed-start should be a block boundary, e.g. 0, 10, 20, ..., 90" >&2
    exit 2
fi

cd /projects/$USER/bsk_rl
source /projects/$USER/.venv/bin/activate

SEED_END=$((SEED_START + SEEDS_PER_BLOCK - 1))
CAMPAIGN_ID=${BSK_RL_MC_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${BSK_RL_MC_OUTPUT_ROOT:-/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i_${N_TARGETS}targets_${CAMPAIGN_ID}}
MANIFEST_DIR="$OUTPUT_ROOT/manifests"
MANIFEST=${BSK_RL_MC_MANIFEST:-$MANIFEST_DIR/gat_full_actions_obs_v9_eval100d00i_nonalpha48h_frozen.json}
JOB_NAME="gat_mc_${N_TARGETS}t_s$(printf '%03d' "$SEED_START")_$(printf '%03d' "$SEED_END")"

mkdir -p "$MANIFEST_DIR" /scratch/alpine/$USER/job_output
if [[ -f "$MANIFEST" ]]; then
    echo "Reusing frozen checkpoint manifest:"
    echo "  $MANIFEST"
else
    python3 -u examples/amos_2026/evaluate_gat_reward_sweep_mc.py \
        --write-manifest "$MANIFEST"
fi

echo
echo "Submitting one AMOS 2026 GAT MC block: $JOB_NAME"
echo "  policies:          8 non-alpha 48-hour GAT runs"
echo "  seeds:             $SEED_START..$SEED_END inclusive"
echo "  n_targets:         $N_TARGETS"
echo "  n_targets_ahead:   $N_TARGETS_AHEAD"
echo "  extra_time_factor: $EXTRA_TIME_FACTOR"
echo "  array tasks:       8 policy jobs, each running $SEEDS_PER_BLOCK fresh evaluator subprocesses"
echo "  max concurrent:    $MAX_CONCURRENT"
echo "  resources/task:    $CPUS_PER_TASK CPUs, $MEMORY, $TIME_LIMIT"
echo "  common score:      100d00i"
echo "  output root:       $OUTPUT_ROOT"
echo "  manifest:          $MANIFEST"
echo
echo "Analyze after completion with:"
echo "  python examples/amos_2026/analyze_gat_reward_sweep_mc.py --input-root \"$OUTPUT_ROOT\" --expected-seeds ${SEED_START}:$((SEED_END + 1))"
echo

sbatch \
    --job-name="$JOB_NAME" \
    --array="0-7%${MAX_CONCURRENT}" \
    --time="$TIME_LIMIT" \
    --mem="$MEMORY" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --qos=normal \
    --export=ALL,BSK_RL_MC_SEED_START="$SEED_START",BSK_RL_MC_SEEDS_PER_BLOCK="$SEEDS_PER_BLOCK",BSK_RL_MC_N_TARGETS="$N_TARGETS",BSK_RL_MC_N_TARGETS_AHEAD="$N_TARGETS_AHEAD",BSK_RL_MC_EXTRA_TIME_FACTOR="$EXTRA_TIME_FACTOR",BSK_RL_MC_OUTPUT_ROOT="$OUTPUT_ROOT",BSK_RL_MC_MANIFEST="$MANIFEST" \
    examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh
