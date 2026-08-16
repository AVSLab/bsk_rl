#!/usr/bin/env bash
# Restartable Research Focus I training segment on the normal 24-hour QoS.
# Array indexes retain the original mapping; migration uses indexes 1 through 5.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_arch24
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=1-5%5
#SBATCH --partition=acpu
#SBATCH --mem=200G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --mail-type=FAIL,END
#SBATCH --qos=cpu-normal
#SBATCH --signal=TERM@300

set -euo pipefail

ARCHITECTURES=(mlp mlp mlp attention attention attention)
CANDIDATE_COUNTS=(5 10 20 5 10 20)
INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
SEGMENT_INDEX=${BSK_RL_SEGMENT_INDEX:?BSK_RL_SEGMENT_INDEX is required}
if (( INDEX < 1 || INDEX > 5 )); then
    echo "This migration script only accepts original array indexes 1 through 5" >&2
    exit 2
fi
if (( SEGMENT_INDEX < 0 || SEGMENT_INDEX > 2 )); then
    echo "BSK_RL_SEGMENT_INDEX must be 0, 1, or 2" >&2
    exit 3
fi

ARCHITECTURE=${ARCHITECTURES[$INDEX]}
CANDIDATE_COUNT=${CANDIDATE_COUNTS[$INDEX]}
SEED=10001

module purge
REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
source "$REPO_DIR/examples/prospectus_rfi/slurm/alpine_runtime.sh"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

TASK_TEMP=$(mktemp -d "/tmp/bsk_rfi_24h_${SLURM_JOB_ID}_${INDEX}_XXXXXX")
trap 'rm -rf "$TASK_TEMP"' EXIT
export TMPDIR=$TASK_TEMP
export RAY_TMPDIR=$TASK_TEMP
export PYTHONUNBUFFERED=1
export BSK_RL_NUM_GPUS=0
export BSK_RL_OBJECT_STORE_MEMORY=${BSK_RL_OBJECT_STORE_MEMORY:-4000000000}
OUTPUT_ROOT=/scratch/alpine/$USER/prospectus_rfi
export BSK_RL_USE_WANDB=1
export BSK_RL_REQUIRE_WANDB=1
export BSK_RL_WANDB_PROJECT=${BSK_RL_RFI_WANDB_PROJECT:-amos2025-architecture-comparison}
export BSK_RL_WANDB_GROUP=${BSK_RL_RFI_WANDB_GROUP:-rfi-alpha0-100s-candidate-sweep}
export BSK_RL_WANDB_KEY_PATH=${BSK_RL_WANDB_KEY_PATH:-/projects/$USER/bsk_rl/examples/wandb_key.txt}
export WANDB_DIR=${BSK_RL_RFI_WANDB_DIR:-$OUTPUT_ROOT/wandb/$BSK_RL_WANDB_PROJECT}
mkdir -p "/scratch/alpine/$USER/job_output" "$OUTPUT_ROOT" "$WANDB_DIR"

if [[ "$(git branch --show-current)" != "amos2025-architecture-comparison" ]]; then
    echo "Wrong branch: $(git branch --show-current)" >&2
    exit 4
fi

if [[ "$ARCHITECTURE" == "mlp" ]]; then
    ARCH_CONFIG=${BSK_RL_MLP_CONFIG:-examples/prospectus_rfi/configs/mlp_selected.yaml}
else
    ARCH_CONFIG=${BSK_RL_ATTENTION_CONFIG:-examples/prospectus_rfi/configs/attention_selected.yaml}
fi

RUN_DIR="$OUTPUT_ROOT/training/${ARCHITECTURE}_k${CANDIDATE_COUNT}_seed${SEED}"
FINAL_CHECKPOINT="$RUN_DIR/checkpoints/final"
TRAIN_ARGS=(
    --architecture "$ARCHITECTURE"
    --architecture-config "$ARCH_CONFIG"
    --candidate-count "$CANDIDATE_COUNT"
    --seed "$SEED"
    --wall-hours 48
    --segment-index "$SEGMENT_INDEX"
    --n-env-runners 28
    --temp-dir "$TASK_TEMP"
    --output-root "$OUTPUT_ROOT"
)
if (( SEGMENT_INDEX == 0 )); then
    if [[ -e "$RUN_DIR/training_metrics.csv" || -e "$FINAL_CHECKPOINT" ]]; then
        echo "Refusing to overwrite an existing initial run: $RUN_DIR" >&2
        exit 5
    fi
    SEGMENT_WALL_HOURS=23.5
else
    if [[ ! -d "$FINAL_CHECKPOINT" ]]; then
        echo "Continuation checkpoint not found: $FINAL_CHECKPOINT" >&2
        exit 6
    fi
    TRAIN_ARGS+=(--resume "$FINAL_CHECKPOINT")
    if (( SEGMENT_INDEX == 1 )); then
        SEGMENT_WALL_HOURS=23.5
    else
        # The first two segments provide about 47 training hours. This cleanup
        # allocation supplies only the remaining time needed to reach 48 hours.
        SEGMENT_WALL_HOURS=1.5
    fi
fi
TRAIN_ARGS+=(--segment-wall-hours "$SEGMENT_WALL_HOURS")

echo "branch=$(git branch --show-current) commit=$(git rev-parse HEAD)"
echo "architecture=$ARCHITECTURE K=$CANDIDATE_COUNT seed=$SEED segment=$SEGMENT_INDEX"
echo "target_wall_h=48 segment_cap_h=$SEGMENT_WALL_HOURS"
echo "wandb_project=$BSK_RL_WANDB_PROJECT wandb_group=$BSK_RL_WANDB_GROUP"
python -c "import wandb; print('wandb import ok')"
python -u examples/prospectus_rfi/train.py "${TRAIN_ARGS[@]}"
