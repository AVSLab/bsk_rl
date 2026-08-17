#!/usr/bin/env bash
# Worst-case memory/throughput gate: both architectures at K=20 and N<=200.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_memstress
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --partition=acpu
#SBATCH --mem=230G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --mail-type=FAIL,END
#SBATCH --qos=cpu-normal
#SBATCH --signal=B:TERM@1800
#SBATCH --nice=10000

set -euo pipefail

ARCHITECTURES=(mlp attention)
INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
ARCHITECTURE=${ARCHITECTURES[$INDEX]}
SEED=99001

module purge
REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
source "$REPO_DIR/examples/prospectus_rfi/slurm/alpine_runtime.sh"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$(git branch --show-current)" != "amos2025-architecture-comparison" ]]; then
    echo "Wrong branch: $(git branch --show-current)" >&2
    exit 2
fi

TASK_TEMP=$(mktemp -d "/tmp/bsk_rfi_memstress_${SLURM_JOB_ID}_${INDEX}_XXXXXX")
export TMPDIR=$TASK_TEMP
export RAY_TMPDIR=$TASK_TEMP
export PYTHONUNBUFFERED=1
export BSK_RL_NUM_GPUS=0
export BSK_RL_OBJECT_STORE_MEMORY=${BSK_RL_OBJECT_STORE_MEMORY:-4000000000}
export BSK_RL_TORCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

OUTPUT_ROOT="/scratch/alpine/$USER/prospectus_rfi/memorysafe_100_200_v2_stress/${SLURM_ARRAY_JOB_ID}"
export BSK_RL_USE_WANDB=1
export BSK_RL_REQUIRE_WANDB=1
export BSK_RL_WANDB_PROJECT=${BSK_RL_RFI_WANDB_PROJECT:-amos2025-architecture-comparison}
export BSK_RL_WANDB_GROUP=${BSK_RL_RFI_WANDB_GROUP:-rfi-alpha0-100s-n100-200-memorysafe-v2-stress}
export BSK_RL_WANDB_RUN_PREFIX="amos2025-rfi-alpha0-100s-n100-200-v2-stress-${SLURM_ARRAY_JOB_ID}"
export BSK_RL_WANDB_KEY_PATH=${BSK_RL_WANDB_KEY_PATH:-/projects/$USER/bsk_rl/examples/wandb_key.txt}
export WANDB_DIR=${BSK_RL_RFI_WANDB_DIR:-$OUTPUT_ROOT/wandb/$BSK_RL_WANDB_PROJECT}
mkdir -p "/scratch/alpine/$USER/job_output" "$OUTPUT_ROOT" "$WANDB_DIR"

if [[ "$ARCHITECTURE" == "mlp" ]]; then
    ARCH_CONFIG=examples/prospectus_rfi/configs/mlp_selected.yaml
else
    ARCH_CONFIG=examples/prospectus_rfi/configs/attention_selected.yaml
fi

echo "branch=$(git branch --show-current) commit=$(git rev-parse HEAD)"
echo "stress_job=$SLURM_ARRAY_JOB_ID architecture=$ARCHITECTURE K=20 seed=$SEED"
echo "catalog_range=100-200 physical_targets_per_runner=200 env_runners=12"
echo "cpus=$SLURM_CPUS_PER_TASK memory_mb=${SLURM_MEM_PER_NODE:-unknown} torch_threads=1"
echo "wandb_project=$BSK_RL_WANDB_PROJECT wandb_group=$BSK_RL_WANDB_GROUP"
python -c "import wandb; print('wandb import ok')"

# exec ensures Slurm's advance TERM signal reaches train.py directly.
exec python -u examples/prospectus_rfi/train.py \
    --architecture "$ARCHITECTURE" \
    --architecture-config "$ARCH_CONFIG" \
    --base-config examples/prospectus_rfi/configs/base_memorysafe_100_200.yaml \
    --candidate-count 20 \
    --seed "$SEED" \
    --wall-hours 1.5 \
    --segment-wall-hours 1.5 \
    --segment-index 0 \
    --n-env-runners 12 \
    --torch-threads 1 \
    --shutdown-buffer-minutes 10 \
    --temp-dir "$TASK_TEMP" \
    --output-root "$OUTPUT_ROOT"
