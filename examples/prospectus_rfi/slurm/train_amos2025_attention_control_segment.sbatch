#!/usr/bin/env bash
# Restartable 48-hour historical attention control in 24-hour allocations.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_a25attn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=acpu
#SBATCH --mem=120G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --qos=cpu-normal
#SBATCH --signal=B:TERM@1800

set -euo pipefail

REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
CAMPAIGN_ID=${BSK_RL_CONTROL_CAMPAIGN:?BSK_RL_CONTROL_CAMPAIGN is required}
SEGMENT_INDEX=${BSK_RL_SEGMENT_INDEX:?BSK_RL_SEGMENT_INDEX is required}
if (( SEGMENT_INDEX < 0 || SEGMENT_INDEX > 2 )); then
    echo "BSK_RL_SEGMENT_INDEX must be 0, 1, or 2" >&2
    exit 2
fi

module purge
source "$REPO_DIR/examples/prospectus_rfi/slurm/alpine_runtime.sh"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$(git branch --show-current)" != "amos2025-architecture-comparison" ]]; then
    echo "Wrong branch: $(git branch --show-current)" >&2
    exit 3
fi

TASK_TEMP=$(mktemp -d "/tmp/bsk_rfi_a25attn_${SLURM_JOB_ID}_XXXXXX")
export TMPDIR=$TASK_TEMP
export RAY_TMPDIR=$TASK_TEMP
export PYTHONUNBUFFERED=1
export BSK_RL_NUM_GPUS=0
export BSK_RL_OBJECT_STORE_MEMORY=${BSK_RL_OBJECT_STORE_MEMORY:-3000000000}
export BSK_RL_TORCH_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

OUTPUT_ROOT="/scratch/alpine/$USER/prospectus_rfi/amos2025_attention_control_300s/$CAMPAIGN_ID"
RUN_DIR="$OUTPUT_ROOT/training/attention_k10_seed10001"
FINAL_CHECKPOINT="$RUN_DIR/checkpoints/final"
export BSK_RL_USE_WANDB=1
export BSK_RL_REQUIRE_WANDB=1
export BSK_RL_WANDB_PROJECT=${BSK_RL_RFI_WANDB_PROJECT:-amos2025-architecture-comparison}
export BSK_RL_WANDB_GROUP="rfi-amos2025-attention-k10-300s-control"
export BSK_RL_WANDB_RUN_PREFIX="amos2025-attention-k10-300s-control-$CAMPAIGN_ID"
export BSK_RL_WANDB_KEY_PATH=${BSK_RL_WANDB_KEY_PATH:-/projects/$USER/bsk_rl/examples/wandb_key.txt}
export WANDB_DIR="$OUTPUT_ROOT/wandb/$BSK_RL_WANDB_PROJECT"
mkdir -p "/scratch/alpine/$USER/job_output" "$OUTPUT_ROOT" "$WANDB_DIR"

TRAIN_ARGS=(
    --architecture attention
    --architecture-config examples/prospectus_rfi/configs/attention_amos2025_control.yaml
    --base-config examples/prospectus_rfi/configs/base_amos2025_attention_control.yaml
    --seed 10001
    --wall-hours 48
    --segment-index "$SEGMENT_INDEX"
    --n-env-runners 4
    --torch-threads 1
    --shutdown-buffer-minutes 15
    --temp-dir "$TASK_TEMP"
    --output-root "$OUTPUT_ROOT"
)
if (( SEGMENT_INDEX == 0 )); then
    if [[ -e "$RUN_DIR" ]]; then
        echo "Refusing to overwrite existing control run: $RUN_DIR" >&2
        exit 4
    fi
    SEGMENT_WALL_HOURS=22
else
    if [[ ! -d "$FINAL_CHECKPOINT" ]]; then
        echo "Continuation checkpoint not found: $FINAL_CHECKPOINT" >&2
        exit 5
    fi
    TRAIN_ARGS+=(--resume "$FINAL_CHECKPOINT")
    if (( SEGMENT_INDEX == 1 )); then
        SEGMENT_WALL_HOURS=22
    else
        SEGMENT_WALL_HOURS=5
    fi
fi
TRAIN_ARGS+=(--segment-wall-hours "$SEGMENT_WALL_HOURS")

echo "branch=$(git branch --show-current) commit=$(git rev-parse HEAD)"
echo "control=N100 K10 image300s charge300s downlink180s desat150s"
echo "ppo=batch180 epochs10 lr1e-6 gamma0.9997 env_runners=4"
echo "campaign=$CAMPAIGN_ID segment=$SEGMENT_INDEX target_wall_h=48"
echo "wandb_project=$BSK_RL_WANDB_PROJECT wandb_group=$BSK_RL_WANDB_GROUP"

exec python -u examples/prospectus_rfi/train.py "${TRAIN_ARGS[@]}"
