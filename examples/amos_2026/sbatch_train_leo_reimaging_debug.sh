#!/bin/bash

# One-hour CURC debug job for the AMOS 2026 LEO-to-LEO reimaging trainer.
# Before submitting, create the Slurm log directory once:
#   mkdir -p /scratch/alpine/$USER/job_output

#SBATCH --account=ucb550_asc2
#SBATCH --job-name=amos2026_leo_dbg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --array=0-0
#SBATCH --partition=amilan
#SBATCH --mem=100G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --mail-type=ALL

set -euo pipefail

module purge

echo "Loading modules"
module load python/3.10.2
module load gcc

echo "Activating virtual environment"
source /projects/$USER/.venv/bin/activate

cd /projects/$USER/bsk_rl

export BSK_RL_SCRATCH=/scratch/alpine/$USER
export BSK_RL_OUTPUT_DIR=/scratch/alpine/$USER/rllib_results
export BSK_RL_RAY_TMPDIR=/scratch/alpine/$USER/tmp/amos2026_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
export TMPDIR=$BSK_RL_RAY_TMPDIR

# Keep these small/configurable for a first startup-failure check.
export BSK_RL_BATCH_MULTIPLIER=${BSK_RL_BATCH_MULTIPLIER:-150}
export BSK_RL_TOTAL_TIMESTEPS=${BSK_RL_TOTAL_TIMESTEPS:-500000}
export BSK_RL_CHECKPOINT_FREQUENCY=${BSK_RL_CHECKPOINT_FREQUENCY:-1}
export BSK_RL_TORCH_THREADS=${BSK_RL_TORCH_THREADS:-11}

mkdir -p "$BSK_RL_OUTPUT_DIR" "$BSK_RL_RAY_TMPDIR"

echo "===== Job context ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "BSK_RL_OUTPUT_DIR=$BSK_RL_OUTPUT_DIR"
echo "BSK_RL_RAY_TMPDIR=$BSK_RL_RAY_TMPDIR"
echo "BSK_RL_BATCH_MULTIPLIER=$BSK_RL_BATCH_MULTIPLIER"
echo "BSK_RL_TOTAL_TIMESTEPS=$BSK_RL_TOTAL_TIMESTEPS"
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --short

echo "Running AMOS 2026 LEO-to-LEO training script"
python3 examples/updated_train_Polaris.py

echo "== End of Job =="
