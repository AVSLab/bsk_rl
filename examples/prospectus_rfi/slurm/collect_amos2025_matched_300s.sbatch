#!/usr/bin/env bash
# Strictly collect only after all 400 matched episodes succeed.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_a25col
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --partition=acpu
#SBATCH --mem=16G
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

if [[ "$(git branch --show-current)" != "amos2025-architecture-comparison" ]]; then
    echo "Wrong branch: $(git branch --show-current)" >&2
    exit 2
fi

OUTPUT_ROOT=${BSK_RL_AMOS2025_MATCHED_300S_OUTPUT_ROOT:?Set BSK_RL_AMOS2025_MATCHED_300S_OUTPUT_ROOT}
python -u examples/prospectus_rfi/collect_amos2025_matched_300s.py \
    --input-root "$OUTPUT_ROOT"
