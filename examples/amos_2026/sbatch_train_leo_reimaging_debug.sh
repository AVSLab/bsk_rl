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
#SBATCH --qos=long

# Copy this script as sweep.sh to modify it for your own use.

module purge

echo "Loading modules"
module load python/3.10.2
module load gcc

echo "Activating virtual environment"
source /projects/$USER/.venv/bin/activate
export TMPDIR=/scratch/alpine/$USER/temp_dir

echo "Running AMOS 2026 LEO-to-LEO training script"
python3 /projects/$USER/bsk_rl/examples/updated_train_Polaris.py

echo "== End of Job =="
