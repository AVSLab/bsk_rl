#!/bin/bash

# AMOS 2026 GAT full-action Monte Carlo block:
#   8 trained reward mixes x 10 seeds = 80 isolated evaluation tasks.
# Use submit_gat_reward_sweep_mc_block.sh so a frozen manifest is created first
# and the job name stays aligned with the selected ten-seed block.

#SBATCH --account=ucb550_asc2
#SBATCH --job-name=gat_mc_s000_009
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-79%4
#SBATCH --partition=amilan
#SBATCH --mem=24G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --qos=normal

set -euo pipefail

module purge
if ! module --ignore_cache load gcc/14.2.0; then
    echo "WARNING: gcc/14.2.0 module not found; falling back to /curc/sw/install/gcc/14.2.0"
    export PATH="/curc/sw/install/gcc/14.2.0/bin:${PATH}"
fi
if ! module --ignore_cache load python/3.10.2; then
    echo "WARNING: python/3.10.2 module not found; continuing with the virtualenv python"
fi
source /projects/$USER/.venv/bin/activate

cd /projects/$USER/bsk_rl

if [ -d /curc/sw/install/gcc/14.2.0/lib64 ]; then
    export LD_LIBRARY_PATH="/curc/sw/install/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}"
fi
if command -v gcc >/dev/null 2>&1; then
    export LD_LIBRARY_PATH="$(dirname "$(gcc -print-file-name=libstdc++.so.6)"):${LD_LIBRARY_PATH:-}"
fi

export BSK_RL_MC_SEED_START=${BSK_RL_MC_SEED_START:-0}
export BSK_RL_MC_OUTPUT_ROOT=${BSK_RL_MC_OUTPUT_ROOT:-/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i}
export BSK_RL_MC_MANIFEST=${BSK_RL_MC_MANIFEST:?Set BSK_RL_MC_MANIFEST to a frozen checkpoint manifest}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1

mkdir -p /scratch/alpine/$USER/job_output "$BSK_RL_MC_OUTPUT_ROOT"

echo "===== AMOS 2026 Monte Carlo task ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "BSK_RL_MC_SEED_START=$BSK_RL_MC_SEED_START"
echo "BSK_RL_MC_MANIFEST=$BSK_RL_MC_MANIFEST"
echo "BSK_RL_MC_OUTPUT_ROOT=$BSK_RL_MC_OUTPUT_ROOT"
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo "commit: $(git rev-parse --short HEAD)"
git status --short --untracked-files=no

python3 -u examples/amos_2026/evaluate_gat_reward_sweep_mc.py
