#!/usr/bin/env bash
# Four matched methods x seeds 0..99; every task runs exactly one episode.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_a25mc3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-399%30
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ "$(git branch --show-current)" != "amos2025-architecture-comparison" ]]; then
    echo "Wrong branch: $(git branch --show-current)" >&2
    exit 2
fi

LEGACY_CHECKPOINT=${BSK_RL_BRECKENRIDGE_ALPHA0_CHECKPOINT:?Set BSK_RL_BRECKENRIDGE_ALPHA0_CHECKPOINT}
ATTENTION_CHECKPOINT=${BSK_RL_AMOS2025_ATTENTION_CHECKPOINT:?Set BSK_RL_AMOS2025_ATTENTION_CHECKPOINT}
OUTPUT_ROOT=${BSK_RL_AMOS2025_MATCHED_300S_OUTPUT_ROOT:?Set BSK_RL_AMOS2025_MATCHED_300S_OUTPUT_ROOT}
mkdir -p "$OUTPUT_ROOT" "/scratch/alpine/$USER/job_output"

python -u examples/prospectus_rfi/amos2025_matched_300s_mc.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --legacy-checkpoint "$LEGACY_CHECKPOINT" \
    --attention-checkpoint "$ATTENTION_CHECKPOINT" \
    --output-root "$OUTPUT_ROOT"
