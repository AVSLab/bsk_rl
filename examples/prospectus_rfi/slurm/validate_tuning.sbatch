#!/usr/bin/env bash
# Validate the 24 equal-budget tuning trials on predeclared held-out seeds.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_validate_tune
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-23%12
#SBATCH --partition=acpu
#SBATCH --mem=80G
#SBATCH --constraint=epyc-7713
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --mail-type=FAIL,END
#SBATCH --qos=cpu-normal

set -euo pipefail
INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
if (( INDEX < 12 )); then
    ARCHITECTURE=mlp
    TRIAL=$INDEX
else
    ARCHITECTURE=attention
    TRIAL=$((INDEX - 12))
fi
SEED=$((50000 + TRIAL))

module purge
REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl}
source "$REPO_DIR/examples/prospectus_rfi/slurm/alpine_runtime.sh"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
RUN_DIR=$(printf "/scratch/alpine/%s/prospectus_rfi/tuning/training/%s_k10_seed%s_tune%02d" "$USER" "$ARCHITECTURE" "$SEED" "$TRIAL")
python -u examples/prospectus_rfi/validate_checkpoints.py \
    --architecture "$ARCHITECTURE" \
    --candidate-count 10 \
    --run-dir "$RUN_DIR" \
    --include-final
