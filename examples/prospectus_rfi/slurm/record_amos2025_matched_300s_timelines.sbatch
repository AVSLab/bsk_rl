#!/usr/bin/env bash
# Replay one accepted episode and record its decision-epoch acquisition history.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_a25time
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

CAMPAIGN_ROOT=${BSK_RL_AMOS2025_MATCHED_300S_OUTPUT_ROOT:?Set campaign root}
LEGACY_CHECKPOINT=${BSK_RL_BRECKENRIDGE_ALPHA0_CHECKPOINT:?Set legacy checkpoint}
ATTENTION_CHECKPOINT=${BSK_RL_AMOS2025_ATTENTION_CHECKPOINT:?Set attention checkpoint}
mkdir -p "/scratch/alpine/$USER/job_output"

python -u examples/prospectus_rfi/amos2025_matched_300s_timeline.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --campaign-root "$CAMPAIGN_ROOT" \
    --legacy-checkpoint "$LEGACY_CHECKPOINT" \
    --attention-checkpoint "$ATTENTION_CHECKPOINT"
