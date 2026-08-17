#!/usr/bin/env bash
# Select each v2 run's best checkpoint on held-out N=100/150/200 seeds.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_memvalid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-5%6
#SBATCH --partition=acpu
#SBATCH --mem=80G
#SBATCH --constraint=epyc-7713
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --mail-type=FAIL,END
#SBATCH --qos=cpu-normal

set -euo pipefail
ARCHITECTURES=(mlp mlp mlp attention attention attention)
CANDIDATE_COUNTS=(5 10 20 5 10 20)
INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
ARCHITECTURE=${ARCHITECTURES[$INDEX]}
K=${CANDIDATE_COUNTS[$INDEX]}

module purge
REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
source "$REPO_DIR/examples/prospectus_rfi/slurm/alpine_runtime.sh"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN_DIR="/scratch/alpine/$USER/prospectus_rfi/memorysafe_100_200_v2/training/${ARCHITECTURE}_k${K}_seed10001"
python -u examples/prospectus_rfi/validate_checkpoints.py \
    --architecture "$ARCHITECTURE" \
    --candidate-count "$K" \
    --base-config examples/prospectus_rfi/configs/base_memorysafe_100_200.yaml \
    --run-dir "$RUN_DIR" \
    --include-final
