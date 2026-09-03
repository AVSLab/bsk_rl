#!/usr/bin/env bash

# Aggregate and plot both completed 50-seed initial-priority cases.

#SBATCH --account=ucb550_asc2
#SBATCH --job-name=amos_initprio_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
#SBATCH --mem=12G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --qos=normal

set -euo pipefail

repo_dir=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl}
output_root=${AMOS_INITIAL_PRIORITY_OUTPUT_ROOT:?Set AMOS_INITIAL_PRIORITY_OUTPUT_ROOT}

module purge
module --ignore_cache load python/3.10.2 || true
source "/projects/$USER/.venv/bin/activate"
cd "$repo_dir"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="/scratch/alpine/$USER/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

python3 -u examples/amos_2026/analyze_initial_priority_allocation_mc.py \
    --input-root "$output_root" \
    --expected-seeds 50 \
    --n-targets 200 \
    --hio-count 20 \
    --shio-count 20

date -u +'%Y-%m-%dT%H:%M:%SZ' \
    > "$output_root/analysis_initial_priority_allocation/ANALYSIS_COMPLETE_UTC.txt"
