#!/bin/bash

# One independent 100-seed Breckenridge Monte Carlo cell.
# submit_2x2_mc.sh submits this file twice without dependencies.

#SBATCH --account=ucb550_asc2
#SBATCH --job-name=br26_mc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-99%10
#SBATCH --partition=amilan
#SBATCH --mem=16G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --qos=normal

set -euo pipefail

cd /projects/$USER/bsk_rl
VENV_PYTHON=/projects/$USER/.venv/bin/python

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Virtual-environment Python is missing: $VENV_PYTHON" >&2
    exit 1
fi

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

: "${BRECK_MC_MANIFEST:?Set BRECK_MC_MANIFEST}"
: "${BRECK_MC_CELL:?Set BRECK_MC_CELL}"
: "${BRECK_MC_OUTPUT_ROOT:?Set BRECK_MC_OUTPUT_ROOT}"

mkdir -p /scratch/alpine/$USER/job_output "$BRECK_MC_OUTPUT_ROOT"

echo "===== Breckenridge 2026 Monte Carlo task ====="
date
hostname
echo "cell=$BRECK_MC_CELL"
echo "seed=${SLURM_ARRAY_TASK_ID}"
echo "manifest=$BRECK_MC_MANIFEST"
echo "output_root=$BRECK_MC_OUTPUT_ROOT"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "commit=$(git rev-parse --short HEAD)"
echo "python=$VENV_PYTHON"
"$VENV_PYTHON" --version

"$VENV_PYTHON" -u examples/breckenridge2026/run_mc_task.py \
    --manifest "$BRECK_MC_MANIFEST" \
    --cell "$BRECK_MC_CELL" \
    --seed "$SLURM_ARRAY_TASK_ID" \
    --output-root "$BRECK_MC_OUTPUT_ROOT"
