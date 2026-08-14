#!/usr/bin/env bash
# Select each run's best retained checkpoint using held-out physical performance.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_validate_ckpt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-5%6
#SBATCH --partition=amilan
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --mail-type=FAIL,END
#SBATCH --qos=normal

set -euo pipefail
ARCHITECTURES=(mlp mlp mlp attention attention attention)
CANDIDATE_COUNTS=(5 10 20 5 10 20)
INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
ARCHITECTURE=${ARCHITECTURES[$INDEX]}
K=${CANDIDATE_COUNTS[$INDEX]}

module purge
module --ignore_cache load gcc/14.2.0 || export PATH="/curc/sw/install/gcc/14.2.0/bin:${PATH}"
module --ignore_cache load python/3.10.2 || true
source "/projects/$USER/.venv/bin/activate"
REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl}
cd "$REPO_DIR"
RUN_DIR="/scratch/alpine/$USER/prospectus_rfi/training/${ARCHITECTURE}_k${K}_seed10001"
python -u examples/prospectus_rfi/validate_checkpoints.py \
    --architecture "$ARCHITECTURE" \
    --candidate-count "$K" \
    --run-dir "$RUN_DIR" \
    --include-final
