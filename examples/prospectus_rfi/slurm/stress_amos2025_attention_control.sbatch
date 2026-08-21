#!/usr/bin/env bash
# One-iteration integration gate for the N=100, K=10, 300 s attention control.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_a25gate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=acpu
#SBATCH --mem=120G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --qos=cpu-normal

set -euo pipefail

REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
CAMPAIGN_ID=${BSK_RL_CONTROL_CAMPAIGN:?BSK_RL_CONTROL_CAMPAIGN is required}
module purge
source "$REPO_DIR/examples/prospectus_rfi/slurm/alpine_runtime.sh"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$(git branch --show-current)" != "amos2025-architecture-comparison" ]]; then
    echo "Wrong branch: $(git branch --show-current)" >&2
    exit 2
fi

TASK_TEMP=$(mktemp -d "/tmp/bsk_rfi_a25gate_${SLURM_JOB_ID}_XXXXXX")
export TMPDIR=$TASK_TEMP
export RAY_TMPDIR=$TASK_TEMP
export PYTHONUNBUFFERED=1
export BSK_RL_NUM_GPUS=0
export BSK_RL_OBJECT_STORE_MEMORY=${BSK_RL_OBJECT_STORE_MEMORY:-3000000000}
export BSK_RL_TORCH_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

OUTPUT_ROOT="/scratch/alpine/$USER/prospectus_rfi/amos2025_attention_control_300s/$CAMPAIGN_ID/gate"
export BSK_RL_USE_WANDB=1
export BSK_RL_REQUIRE_WANDB=1
export BSK_RL_WANDB_PROJECT=${BSK_RL_RFI_WANDB_PROJECT:-amos2025-architecture-comparison}
export BSK_RL_WANDB_GROUP="rfi-amos2025-attention-k10-300s-control-gate"
export BSK_RL_WANDB_RUN_PREFIX="amos2025-attention-k10-300s-control-gate-$CAMPAIGN_ID"
export BSK_RL_WANDB_KEY_PATH=${BSK_RL_WANDB_KEY_PATH:-/projects/$USER/bsk_rl/examples/wandb_key.txt}
export WANDB_DIR="$OUTPUT_ROOT/wandb/$BSK_RL_WANDB_PROJECT"
mkdir -p "/scratch/alpine/$USER/job_output" "$OUTPUT_ROOT" "$WANDB_DIR"

echo "branch=$(git branch --show-current) commit=$(git rev-parse HEAD)"
echo "control=N100 K10 image300s charge300s downlink180s desat150s"
echo "ppo=batch180 epochs10 lr1e-6 gamma0.9997 env_runners=4"
echo "campaign=$CAMPAIGN_ID output_root=$OUTPUT_ROOT"

python -u examples/prospectus_rfi/train.py \
    --architecture attention \
    --architecture-config examples/prospectus_rfi/configs/attention_amos2025_control.yaml \
    --base-config examples/prospectus_rfi/configs/base_amos2025_attention_control.yaml \
    --seed 99025 \
    --wall-hours 48 \
    --segment-index 0 \
    --n-env-runners 4 \
    --torch-threads 1 \
    --shutdown-buffer-minutes 0 \
    --max-iterations 1 \
    --temp-dir "$TASK_TEMP" \
    --output-root "$OUTPUT_ROOT"

STATUS="$OUTPUT_ROOT/training/attention_k10_seed99025/status.json"
python - "$STATUS" <<'PY'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text())
if int(status.get("training_iteration", 0)) < 1:
    raise SystemExit("control gate did not complete one PPO iteration")
if int(status.get("environment_steps", 0)) < 180:
    raise SystemExit("control gate did not collect the configured 180 samples")
print("PASS: AMOS 2025 attention control completed one PPO iteration")
PY
