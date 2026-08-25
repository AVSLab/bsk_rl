#!/usr/bin/env bash

# Submit paired seed-0..99 evaluations for:
#   1. minimum-angle eligible-target heuristic + safety shield,
#   2. maximum-priority target from the common candidate-10 set + safety shield,
#   3. uniform-random choice from the candidate-10 imaging actions + safety shield.

set -euo pipefail

max_concurrent=${1:-2}
if ! [[ "$max_concurrent" =~ ^[1-9][0-9]*$ ]] || (( max_concurrent > 30 )); then
    echo "Usage: $0 [max-concurrent-array-tasks: 1..30, default 2]" >&2
    exit 2
fi

workdir=${BSK_RL_CLUSTER_WORKDIR:-/projects/$USER/bsk_rl_amos2026}
venv_dir=${BSK_RL_CLUSTER_VENV:-/projects/$USER/.venv}
expected_branch=${BSK_RL_EXPECTED_BRANCH:-amos-2026-space-imaging}
partition=${BSK_RL_HEUR_PARTITION:-acpu}
account=${BSK_RL_HEUR_ACCOUNT:-ucb550_asc2}
qos=${BSK_RL_HEUR_QOS:-normal}
time_limit=${BSK_RL_HEUR_TIME:-08:00:00}
memory=${BSK_RL_HEUR_MEM:-24G}
cpus_per_task=${BSK_RL_HEUR_CPUS_PER_TASK:-4}

if [[ ! -d "$workdir/.git" && ! -f "$workdir/.git" ]]; then
    echo "Isolated AMOS 2026 checkout not found: $workdir" >&2
    exit 3
fi

cd "$workdir"
# shellcheck source=/dev/null
source "$venv_dir/bin/activate"
export PYTHONPATH="$workdir/src${PYTHONPATH:+:$PYTHONPATH}"

actual_branch=$(git branch --show-current)
if [[ "$actual_branch" != "$expected_branch" ]]; then
    echo "Refusing to submit from branch '$actual_branch'; expected '$expected_branch'." >&2
    exit 3
fi

for required_file in \
    examples/amos_2026/evaluate_heuristic_mc.py \
    examples/amos_2026/sbatch_evaluate_baseline_mc_120targets_oneorbit.sh \
    examples/updated_policy_evaluation.py; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required AMOS 2026 file is missing: $workdir/$required_file" >&2
        exit 3
    fi
done

imported_bsk_rl=$(python3 -c 'import pathlib, bsk_rl; print(pathlib.Path(bsk_rl.__file__).resolve())')
if [[ "$imported_bsk_rl" != "$workdir"/src/bsk_rl/* ]]; then
    echo "Refusing to submit: bsk_rl imports from $imported_bsk_rl, not $workdir/src." >&2
    exit 3
fi

campaign_id=${BSK_RL_HEUR_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
output_root=${BSK_RL_HEUR_OUTPUT_ROOT:-/scratch/alpine/$USER/amos2026_mc/baselines_100d00i_mixed_exact60LEO36MEO24GEO_120targets_prioritySum120_45000s_HIO5targets_5xMax_SHIO3targets_10xMax_1orbitCooldown_shieldOnly_${campaign_id}}
manifest_dir="$output_root/manifests"
mkdir -p "$manifest_dir" "/scratch/alpine/$USER/job_output"

export BSK_RL_HEUR_MODES="angle,candidate_priority,random"
export BSK_RL_HEUR_SEED_ORIGIN=0
export BSK_RL_HEUR_TOTAL_SEEDS=100
export BSK_RL_HEUR_SEEDS_PER_BLOCK=10
export BSK_RL_HEUR_OUTPUT_ROOT="$output_root"
export BSK_RL_HEUR_TARGET_ENV=mixed
export BSK_RL_HEUR_MIX_WEIGHTS='{"LEO":0.5,"MEO":0.3,"GEO":0.2}'
export BSK_RL_HEUR_EXACT_MIX_COUNTS=1
export BSK_RL_HEUR_N_TARGETS=120
export BSK_RL_HEUR_N_TARGETS_AHEAD=10
export BSK_RL_HEUR_PRIORITY_SUM=120.0
export BSK_RL_HEUR_TOTAL_TIME_SEC=45000
export BSK_RL_HEUR_REIMAGE_COOLDOWN_ORBITS=1.0
export BSK_RL_HEUR_REWARD_MIX=100d00i
export BSK_RL_HEUR_DYNAMIC_PRIORITY_EVENT=on
export BSK_RL_HEUR_HIO_COUNT=5
export BSK_RL_HEUR_HIO_PRIORITY=5.0
export BSK_RL_HEUR_HIO_PRIORITY_MAX_MULTIPLIER=5.0
export BSK_RL_HEUR_SHIO_COUNT=3
export BSK_RL_HEUR_SHIO_PRIORITY=10.0
export BSK_RL_HEUR_SHIO_PRIORITY_MAX_MULTIPLIER=10.0
export BSK_RL_HEUR_SHIELD_ONLY=1
export BSK_RL_CLUSTER_WORKDIR="$workdir"
export BSK_RL_CLUSTER_VENV="$venv_dir"
export BSK_RL_EXPECTED_BRANCH="$expected_branch"

cat > "$manifest_dir/campaign.json" <<EOF
{
  "campaign_id": "$campaign_id",
  "controllers": [
    "minimum_angle_visible_eligible_with_shield",
    "maximum_priority_from_candidate_10_with_shield",
    "uniform_random_from_candidate_10_with_shield"
  ],
  "seeds": "0:100",
  "paired_catalog_seeds": true,
  "target_environment": "mixed",
  "target_count": 120,
  "exact_regime_counts": {"LEO": 60, "MEO": 36, "GEO": 24},
  "candidate_count": 10,
  "initial_priority_sum": 120.0,
  "episode_duration_sec": 45000,
  "reimage_cooldown_orbits": 1.0,
  "evaluation_reward_mix": "100d00i",
  "dynamic_priority_event_fraction": 0.5,
  "HIO": {"count": 5, "initial_max_multiplier": 5.0},
  "SHIO": {"count": 3, "initial_max_multiplier": 10.0},
  "standard_safety_shield": true,
  "slurm_array_tasks": 30,
  "maximum_concurrent_array_tasks": $max_concurrent
}
EOF

echo "Submitting the 120-target, one-orbit baseline campaign"
echo "  output root:       $output_root"
echo "  controllers:       $BSK_RL_HEUR_MODES"
echo "  seeds/controller:  100"
echo "  evaluations:       300"
echo "  array concurrency: $max_concurrent"
echo "  source checkout:   $workdir"
echo "  source branch:     $actual_branch"
echo "  source commit:     $(git rev-parse HEAD)"
echo "  bsk_rl import:     $imported_bsk_rl"
echo "  partition:         $partition"
echo "  account:           $account"

job_id=$(sbatch \
    --parsable \
    --array="0-29%${max_concurrent}" \
    --partition="$partition" \
    --account="$account" \
    --qos="$qos" \
    --time="$time_limit" \
    --mem="$memory" \
    --cpus-per-task="$cpus_per_task" \
    --chdir="$workdir" \
    --export=ALL \
    examples/amos_2026/sbatch_evaluate_baseline_mc_120targets_oneorbit.sh)

printf '%s\n' "$job_id" > "$manifest_dir/slurm_job_id.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$manifest_dir/SUBMISSION_COMPLETE_UTC.txt"

echo
echo "Submitted Slurm job array: $job_id"
echo "OUTPUT_ROOT=$output_root"
echo "Monitor with:"
echo "  squeue -j $job_id -o '%.18i %.9P %.22j %.2t %.10M %.6D %R'"
echo "Count completed seed records with:"
echo "  find '$output_root' -name mc_status.json -exec grep -l '\"state\": \"completed\"' {} + | wc -l"
