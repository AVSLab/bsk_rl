#!/usr/bin/env bash

# Recreate per-seed plots from completed AMOS initial-priority episode files.

#SBATCH --account=ucb550_asc2
#SBATCH --job-name=amos_initplots
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --partition=acpu
#SBATCH --mem=16G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --qos=cpu-normal

set -euo pipefail

repo_dir=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-amos2026-campaign}
input_root=${AMOS_INITIAL_PRIORITY_OUTPUT_ROOT:?Set AMOS_INITIAL_PRIORITY_OUTPUT_ROOT}

module purge
# shellcheck source=/dev/null
source "$repo_dir/examples/amos_2026/alpine_runtime.sh"
cd "$repo_dir"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="/scratch/alpine/$USER/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

python -u examples/amos_2026/postprocess_initial_priority_episode_plots.py \
    --input-root "$input_root" \
    --expected-episodes 100

archive_path="$(dirname "$input_root")/$(basename "$input_root")_per_seed_plots.tar.gz"
tar -C "$input_root" -czf "$archive_path" per_seed_plots
echo "Per-seed plot archive: $archive_path"

combined_archive_path="$(dirname "$input_root")/$(basename "$input_root")_episode_csv_and_plots.tar.gz"
(
    cd "$input_root"
    {
        find ground_confirmation one_orbit -type f \
            \( -name '*.csv' -o -name '*.json' \) -print0
        find per_seed_plots -type f -print0
    } | tar --null -czf "$combined_archive_path" -T -
)
echo "Episode CSV/JSON and plot archive: $combined_archive_path"
