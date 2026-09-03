#!/usr/bin/env bash

# Submit two matched AMOS 2026 initial-priority allocation campaigns:
#   - 50 seeds with ground-confirmation-only re-imaging;
#   - the same 50 seeds with a one-observer-orbit cooldown.
# A dependent job validates, aggregates, tests, and plots both cases.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
default_repo_dir=$(cd "$script_dir/../.." && pwd)
repo_dir=${BSK_RL_REPO_DIR:-$default_repo_dir}
source_root=${BSK_RL_MIXED_TRAINED_SOURCE_ROOT:-/scratch/alpine/$USER/amos2026_mc/gat_mixed_trained_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_45000s_HIO5_SHIO3_20260802}
policy_spec=${AMOS_INITIAL_PRIORITY_POLICY_SPEC:-$source_root/manifests/mixed_fixed100_custom_policies.json}
campaign_id=${AMOS_INITIAL_PRIORITY_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
output_root=${AMOS_INITIAL_PRIORITY_OUTPUT_ROOT:-/scratch/alpine/$USER/amos2026_mc/initial_priority_10pctHIO_10pctSHIO_mixed200_45000s_50seeds_${campaign_id}}
array_limit=${AMOS_INITIAL_PRIORITY_ARRAY_LIMIT:-20}
partition=${AMOS_INITIAL_PRIORITY_PARTITION:-acpu}
qos=${AMOS_INITIAL_PRIORITY_QOS:-cpu-normal}
constraint=${AMOS_INITIAL_PRIORITY_CONSTRAINT:-epyc-7713}

cd "$repo_dir"
# shellcheck source=/dev/null
source "/projects/$USER/.venv/bin/activate"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"

branch=$(git branch --show-current)
if [[ "$branch" != "amos-2026-space-imaging" ]]; then
    echo "Expected branch amos-2026-space-imaging, found $branch" >&2
    exit 2
fi
if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
    echo "Tracked files in $repo_dir are modified; refusing an irreproducible submission." >&2
    git status --short --untracked-files=no >&2
    exit 3
fi
if [[ ! -f "$repo_dir/src/bsk_rl/__init__.py" ]]; then
    echo "Refusing to submit: bsk_rl source is missing from $repo_dir/src." >&2
    exit 4
fi
if [[ ! -f "$policy_spec" ]]; then
    echo "Mixed-trained policy specification not found: $policy_spec" >&2
    exit 5
fi

python3 - "$policy_spec" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
policy = payload.get("policies", {}).get("mixed_a0p1")
if policy is None:
    raise SystemExit(f"mixed_a0p1 is missing from {path}")
if float(policy.get("alpha", -1.0)) != 0.1:
    raise SystemExit(f"mixed_a0p1 has unexpected alpha: {policy.get('alpha')}")
checkpoint = pathlib.Path(policy["checkpoint_dir"])
if not checkpoint.is_dir():
    raise SystemExit(f"mixed_a0p1 checkpoint is missing: {checkpoint}")
print(f"Validated mixed_a0p1 checkpoint: {checkpoint}")
PY

mkdir -p "$output_root/manifests" "/scratch/alpine/$USER/job_output"
cp "$policy_spec" "$output_root/manifests/source_mixed_policy_spec.json"
cat > "$output_root/manifests/campaign_configuration.env" <<EOF
BRANCH=$branch
COMMIT=$(git rev-parse HEAD)
POLICY=mixed_a0p1
POLICY_ALPHA=0.1
POLICY_TRAINING_COOLDOWN_ORBITS=2
EVALUATION_TARGETS=200
EVALUATION_CATALOG=exact_100LEO_60MEO_40GEO
EPISODE_DURATION_SEC=45000
SEEDS_PER_CASE=50
SEEDS=0-49_IN_BOTH_CASES
ARRAY_TASK_MAPPING=interleaved_matched_pairs_even_ground_confirmation_odd_one_orbit
PRIORITY_ASSIGNMENT_TIME_SEC=0
HIO_COUNT=20
HIO_FRACTION=0.10
HIO_PRIORITY=5x_realized_initial_priority_maximum
SHIO_COUNT=20
SHIO_FRACTION=0.10
SHIO_PRIORITY=10x_realized_initial_priority_maximum
NORMAL_TARGET_COUNT=160
NORMAL_BASELINE_PRIORITY_DISTRIBUTION=uniform_0_to_2_rescaled_to_sum_200
NORMAL_GROUPS=within_seed_lower_middle_upper_priority_tertiles
CASE_1=ground_confirmation_only
CASE_2=one_orbit_cooldown
VIZARD=disabled
PER_SEED_PLOTS=disabled
AGGREGATE_PLOTS=enabled_after_successful_array
SLURM_PARTITION=$partition
SLURM_QOS=$qos
SLURM_CONSTRAINT=$constraint
EOF

export BSK_RL_REPO_DIR="$repo_dir"
export AMOS_INITIAL_PRIORITY_OUTPUT_ROOT="$output_root"
export AMOS_INITIAL_PRIORITY_POLICY_SPEC="$policy_spec"

evaluation_exports=ALL,BSK_RL_REPO_DIR="$repo_dir",AMOS_INITIAL_PRIORITY_OUTPUT_ROOT="$output_root",AMOS_INITIAL_PRIORITY_POLICY_SPEC="$policy_spec"
analysis_exports=ALL,BSK_RL_REPO_DIR="$repo_dir",AMOS_INITIAL_PRIORITY_OUTPUT_ROOT="$output_root"

# Validate both resource requests before submitting either real job.  This
# catches stale partition/QoS/constraint combinations without leaving a
# partially submitted campaign.
sbatch \
    --test-only \
    --partition="$partition" \
    --qos="$qos" \
    --constraint="$constraint" \
    --array="0-99%${array_limit}" \
    --export="$evaluation_exports" \
    examples/amos_2026/sbatch_initial_priority_allocation_mc.sbatch
sbatch \
    --test-only \
    --partition="$partition" \
    --qos="$qos" \
    --constraint="$constraint" \
    --export="$analysis_exports" \
    examples/amos_2026/sbatch_analyze_initial_priority_allocation_mc.sbatch

evaluation_job=$(sbatch \
    --parsable \
    --partition="$partition" \
    --qos="$qos" \
    --constraint="$constraint" \
    --array="0-99%${array_limit}" \
    --export="$evaluation_exports" \
    examples/amos_2026/sbatch_initial_priority_allocation_mc.sbatch)

analysis_job=$(sbatch \
    --parsable \
    --partition="$partition" \
    --qos="$qos" \
    --constraint="$constraint" \
    --dependency="afterok:${evaluation_job}" \
    --export="$analysis_exports" \
    examples/amos_2026/sbatch_analyze_initial_priority_allocation_mc.sbatch)

cat > "$output_root/manifests/submitted_jobs.tsv" <<EOF
role\tjob_id\tdependency
evaluation_array\t$evaluation_job\tnone
aggregate_analysis\t$analysis_job\tafterok:$evaluation_job
EOF
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$output_root/manifests/SUBMISSION_COMPLETE_UTC.txt"

echo
echo "Submitted initial-priority allocation campaign."
echo "  source worktree: $repo_dir"
echo "  Slurm resources: partition=$partition qos=$qos constraint=$constraint"
echo "  evaluation array: $evaluation_job (100 tasks, max $array_limit concurrent)"
echo "  aggregate analysis: $analysis_job (after successful array completion)"
echo "  output root: $output_root"
echo
echo "Monitor:"
echo "  squeue -u $USER --i 5"
echo "  sacct -X -j $evaluation_job,$analysis_job --array --format=JobID,JobName%25,State,ExitCode,Elapsed -P"
echo
echo "After completion:"
echo "  sed -n '1,220p' '$output_root/analysis_initial_priority_allocation/STATISTICAL_SUMMARY.md'"
echo "  ls -lh '$output_root/analysis_initial_priority_allocation/'"
