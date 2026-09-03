#!/usr/bin/env bash

# Submit a bounded post-processing job for an existing completed campaign.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
default_repo_dir=$(cd "$script_dir/../.." && pwd)
repo_dir=${BSK_RL_REPO_DIR:-$default_repo_dir}
input_root=${AMOS_INITIAL_PRIORITY_OUTPUT_ROOT:?Set AMOS_INITIAL_PRIORITY_OUTPUT_ROOT}
sbatch_file=examples/amos_2026/sbatch_postprocess_initial_priority_episode_plots.sbatch
exports=ALL,BSK_RL_REPO_DIR="$repo_dir",AMOS_INITIAL_PRIORITY_OUTPUT_ROOT="$input_root"

cd "$repo_dir"
if [[ $(git branch --show-current) != "amos-2026-space-imaging" ]]; then
    echo "Expected branch amos-2026-space-imaging" >&2
    exit 2
fi
if [[ ! -f "$input_root/analysis_initial_priority_allocation/campaign_audit.csv" ]]; then
    echo "Completed campaign audit not found under: $input_root" >&2
    exit 3
fi

sbatch --test-only --export="$exports" "$sbatch_file"
job_id=$(sbatch --parsable --export="$exports" "$sbatch_file")

echo "Submitted per-seed plot postprocessing: $job_id"
echo "Campaign root: $input_root"
echo "Output directory: $input_root/per_seed_plots"
echo "The job also archives every episode CSV/JSON together with these plots."
echo "Monitor: squeue -j $job_id --i 5"
