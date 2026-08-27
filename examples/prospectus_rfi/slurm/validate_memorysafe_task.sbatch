#!/usr/bin/env bash
# One held-out episode per task: safely below the wall time and independently resumable.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_memvalid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --partition=acpu
#SBATCH --mem=64G
#SBATCH --constraint=epyc-7713
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --mail-type=FAIL,END
#SBATCH --qos=cpu-normal
#SBATCH --nice=10000

set -euo pipefail
REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
MANIFEST=${BSK_RL_VALIDATION_MANIFEST:?BSK_RL_VALIDATION_MANIFEST is required}
TASK_MAP=${BSK_RL_VALIDATION_TASK_MAP:?BSK_RL_VALIDATION_TASK_MAP is required}
LOCAL_TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
TASK_ID=$(sed -n "$((LOCAL_TASK_ID + 1))p" "$TASK_MAP")
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
  echo "No validation task ID for local array index $LOCAL_TASK_ID in $TASK_MAP" >&2
  exit 4
fi

module purge
source "$REPO_DIR/examples/prospectus_rfi/slurm/alpine_runtime.sh"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u examples/prospectus_rfi/run_memorysafe_validation_task.py \
  --manifest "$MANIFEST" --task-id "$TASK_ID"
