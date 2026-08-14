#!/usr/bin/env bash
# Equal tuning allowance: 12 trials x 8 h for each architecture.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_tune_equal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --array=0-23%8
#SBATCH --partition=amilan
#SBATCH --mem=200G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --mail-type=FAIL,END
#SBATCH --qos=normal
#SBATCH --signal=B:TERM@300

set -euo pipefail
INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
if (( INDEX < 12 )); then
    ARCHITECTURE=mlp
    TRIAL_INDEX=$INDEX
    TABLE=examples/prospectus_rfi/configs/mlp_tuning_table.csv
else
    ARCHITECTURE=attention
    TRIAL_INDEX=$((INDEX - 12))
    TABLE=examples/prospectus_rfi/configs/attention_tuning_table.csv
fi
SEED=$((50000 + TRIAL_INDEX))

module purge
module --ignore_cache load gcc/14.2.0 || export PATH="/curc/sw/install/gcc/14.2.0/bin:${PATH}"
module --ignore_cache load python/3.10.2 || true
source "/projects/$USER/.venv/bin/activate"
REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl}
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
TASK_TEMP=$(mktemp -d "/tmp/bsk_rfi_tune_${SLURM_JOB_ID}_${INDEX}_XXXXXX")
trap 'rm -rf "$TASK_TEMP"' EXIT
export TMPDIR=$TASK_TEMP RAY_TMPDIR=$TASK_TEMP PYTHONUNBUFFERED=1 BSK_RL_NUM_GPUS=0
OUTPUT_ROOT="/scratch/alpine/$USER/prospectus_rfi/tuning"
export BSK_RL_USE_WANDB=1
export BSK_RL_REQUIRE_WANDB=1
export BSK_RL_WANDB_PROJECT=${BSK_RL_RFI_WANDB_PROJECT:-amos2025-architecture-comparison}
export BSK_RL_WANDB_GROUP=${BSK_RL_RFI_TUNING_WANDB_GROUP:-rfi-alpha0-100s-architecture-tuning}
export BSK_RL_WANDB_KEY_PATH=${BSK_RL_WANDB_KEY_PATH:-/projects/$USER/bsk_rl/examples/wandb_key.txt}
export WANDB_DIR=${BSK_RL_RFI_WANDB_DIR:-/scratch/alpine/$USER/prospectus_rfi/wandb/$BSK_RL_WANDB_PROJECT}
mkdir -p "/scratch/alpine/$USER/job_output" "$OUTPUT_ROOT" "$WANDB_DIR"

echo "wandb_project=$BSK_RL_WANDB_PROJECT wandb_group=$BSK_RL_WANDB_GROUP"
python -c "import wandb; print('wandb import ok')"

python -u examples/prospectus_rfi/train.py \
    --architecture "$ARCHITECTURE" \
    --candidate-count 10 \
    --seed "$SEED" \
    --wall-hours 8 \
    --n-env-runners 28 \
    --temp-dir "$TASK_TEMP" \
    --output-root "$OUTPUT_ROOT" \
    --tuning-table "$TABLE" \
    --trial-index "$TRIAL_INDEX"
