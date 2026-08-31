#!/usr/bin/env bash
# Validate and analyze all 400 matched acquisition histories.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_a25tcol
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=acpu
#SBATCH --mem=64G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --qos=cpu-normal

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
python -u examples/prospectus_rfi/analyze_amos2025_matched_300s_timelines.py \
    --input-root "$CAMPAIGN_ROOT"
