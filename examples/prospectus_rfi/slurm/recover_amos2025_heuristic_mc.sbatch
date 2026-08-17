#!/usr/bin/env bash
# One missing closest-angle episode per task. The submitter supplies exact task IDs.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_heurrc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --array=0-299%30
#SBATCH --partition=acpu
#SBATCH --mem=80G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --mail-type=FAIL,END
#SBATCH --qos=cpu-normal
#SBATCH --nice=10000

set -euo pipefail

module purge
REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
source "$REPO_DIR/examples/prospectus_rfi/slurm/alpine_runtime.sh"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ "$(git branch --show-current)" != "amos2025-architecture-comparison" ]]; then
    echo "Wrong branch: $(git branch --show-current)" >&2
    exit 2
fi

OUTPUT_ROOT=${BSK_RL_HEURISTIC_MC_OUTPUT_ROOT:?Set BSK_RL_HEURISTIC_MC_OUTPUT_ROOT}
mkdir -p "$OUTPUT_ROOT" "/scratch/alpine/$USER/job_output"

echo "branch=$(git branch --show-current) commit=$(git rev-parse HEAD)"
echo "array_task=$SLURM_ARRAY_TASK_ID output_root=$OUTPUT_ROOT"
python -u examples/prospectus_rfi/heuristic_mc_independent.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --output-root "$OUTPUT_ROOT"
