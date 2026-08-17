#!/usr/bin/env bash
# Paired timeline replay: 2 methods x N={100,200,400} x seeds 0..99.
# Each of the 600 independent tasks records one complete decision-epoch history.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_tline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --array=0-599%30
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

HEURISTIC_ROOT=${BSK_RL_HEURISTIC_MC_OUTPUT_ROOT:?Set BSK_RL_HEURISTIC_MC_OUTPUT_ROOT}
POLICY_ROOT=${BSK_RL_LEGACY_POLICY_MC_OUTPUT_ROOT:?Set BSK_RL_LEGACY_POLICY_MC_OUTPUT_ROOT}
CHECKPOINT=${BSK_RL_AMOS2025_POLICY_CHECKPOINT:?Set BSK_RL_AMOS2025_POLICY_CHECKPOINT}
mkdir -p "/scratch/alpine/$USER/job_output"

echo "branch=$(git branch --show-current) commit=$(git rev-parse HEAD)"
echo "array_task=$SLURM_ARRAY_TASK_ID"
echo "heuristic_root=$HEURISTIC_ROOT"
echo "policy_root=$POLICY_ROOT"
python -u examples/prospectus_rfi/acquisition_timeline_mc.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --heuristic-root "$HEURISTIC_ROOT" \
    --policy-root "$POLICY_ROOT" \
    --checkpoint "$CHECKPOINT"
