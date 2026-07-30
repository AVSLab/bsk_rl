#!/usr/bin/env bash
set -euo pipefail

MAX_CONCURRENT=${1:-4}
SBATCH_SCRIPT=examples/amos_2026/sbatch_train_polaris_gat_full_actions_mixed_fixed_100targets_reward_sweep_48h.sh
JOB_NAME=gat_mix100_fixed_sweep

if ! [[ "$MAX_CONCURRENT" =~ ^[1-8]$ ]]; then
    echo "Usage: $0 [max-concurrent: 1..8, default 4]" >&2
    exit 2
fi

cd /projects/$USER/bsk_rl
source /projects/$USER/.venv/bin/activate

WANDB_KEY_PATH=${BSK_RL_WANDB_KEY_PATH:-/projects/$USER/bsk_rl/examples/wandb_key.txt}
if [[ ! -s "$WANDB_KEY_PATH" ]]; then
    echo "W&B key file is missing or empty: $WANDB_KEY_PATH" >&2
    exit 1
fi
python3 -c "import wandb; print('wandb import ok:', wandb.__version__)"

active_jobs=$(
    squeue \
        --noheader \
        --user="$USER" \
        --name="$JOB_NAME" \
        --format='%A %T' \
        | sort -u
)
if [[ -n "$active_jobs" ]]; then
    echo "An active fixed mixed-100 training sweep already exists:"
    echo "$active_jobs"
    if [[ ${BSK_RL_ALLOW_DUPLICATE_TRAINING:-0} != "1" ]]; then
        echo "No job submitted. This guard prevents duplicate training." >&2
        echo "Set BSK_RL_ALLOW_DUPLICATE_TRAINING=1 only for an intentional rerun." >&2
        exit 3
    fi
    echo "WARNING: duplicate protection explicitly disabled."
fi

command=(sbatch --array="0-7%${MAX_CONCURRENT}" "$SBATCH_SCRIPT")
echo "Submitting fixed mixed-100 reward sweep:"
printf '  %q' "${command[@]}"
echo
echo "No dependencies will be added; Slurm will schedule these tasks independently."

if [[ ${BSK_RL_DRY_RUN:-0} == "1" ]]; then
    echo "Dry run only; no job submitted."
    exit 0
fi

job_id=$("${command[@]}")
echo "Submitted: $job_id"
echo "Monitor with: squeue -u $USER -n $JOB_NAME"
