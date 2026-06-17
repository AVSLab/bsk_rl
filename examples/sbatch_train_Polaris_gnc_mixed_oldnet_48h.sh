#!/bin/bash

# Allocation account name
#SBATCH --account=ucb550_asc2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=47:59:59
# Default: run only the imaging-only mixed-trained policy. Override with
# sbatch --array=0-1 for 0d100i and 10d90i, or --array=0-10 for the full sweep.
#SBATCH --array=0-0
#SBATCH --partition=amilan
#SBATCH --mem=100G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/gnc_mixed_oldnet_%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --qos=long

set -euo pipefail

BONUSES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
LABELS=(0d100i 10d90i 20d80i 30d70i 40d60i 50d50i 60d40i 70d30i 80d20i 90d10i 100d00i)

BONUS="${BONUSES[$SLURM_ARRAY_TASK_ID]}"
LABEL="${LABELS[$SLURM_ARRAY_TASK_ID]}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_Polaris_gnc_mixed_oldnet.py"

module purge

echo "Loading modules"
module load python/3.10.2
module load gcc

echo "Activating virtual environment"
source /projects/$USER/.venv/bin/activate
export TMPDIR=/scratch/alpine/$USER/temp_dir
mkdir -p "$TMPDIR" /scratch/alpine/$USER/tmp /scratch/alpine/$USER/job_output

echo "Running mixed old-network GNC training for ${LABEL}"
python3 "$TRAIN_SCRIPT" \
  --downlink-bonus "$BONUS" \
  --mix-weights "LEO=0.5,MEO=0.3,GEO=0.2" \
  --run-prefix "gnc_MIXED_oldnet" \
  --total-timesteps 20000000 \
  --batch-multiplier 350 \
  --checkpoint-frequency 3 \
  --checkpoints-to-keep 3 \
  --failure-penalty -100.0

echo "== End of Job =="
