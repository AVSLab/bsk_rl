#!/usr/bin/env bash
# Select the 300-second attention checkpoint on five predeclared held-out seeds.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_a25val
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=acpu
#SBATCH --mem=80G
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

if [[ "$(git branch --show-current)" != "amos2025-architecture-comparison" ]]; then
    echo "Wrong branch: $(git branch --show-current)" >&2
    exit 2
fi

RUN_DIR=${BSK_RL_AMOS2025_ATTENTION_RUN_DIR:?Set BSK_RL_AMOS2025_ATTENTION_RUN_DIR}
if [[ ! -d "$RUN_DIR/checkpoints/final" ]]; then
    echo "Missing final attention checkpoint: $RUN_DIR/checkpoints/final" >&2
    exit 3
fi

python -u examples/prospectus_rfi/validate_checkpoints.py \
    --architecture attention \
    --candidate-count 10 \
    --run-dir "$RUN_DIR" \
    --base-config examples/prospectus_rfi/configs/base_amos2025_attention_control.yaml \
    --include-final \
    --no-wheel-guard

test -L "$RUN_DIR/checkpoints/best_validation"
echo "SELECTED_ATTENTION_CHECKPOINT=$RUN_DIR/checkpoints/best_validation"
