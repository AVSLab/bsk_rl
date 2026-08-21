#!/usr/bin/env bash

# Aggregate the completed closest-angle heuristic campaigns.

#SBATCH --account=ucb550_asc2
#SBATCH --job-name=heur_mc_analyze
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

root=${BSK_RL_HEUR_OUTPUT_ROOT:?Set BSK_RL_HEUR_OUTPUT_ROOT}
source "/projects/$USER/.venv/bin/activate"
cd "/projects/$USER/bsk_rl"
export MPLBACKEND=Agg
export MPLCONFIGDIR="/scratch/alpine/$USER/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

python3 examples/amos_2026/analyze_gat_reward_sweep_mc_detailed.py \
    --input-root "$root" \
    --expected-seeds 0:100 \
    --policy-tags heur_angle,heur_priority_angle

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$root/analysis_detailed/ANALYSIS_COMPLETE_UTC.txt"
echo "Heuristic analysis complete: $root/analysis_detailed"
