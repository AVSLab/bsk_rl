#!/bin/bash

# Continue an existing AMOS 2026 Polaris GAT full-action 10d90i policy for 24h.
# The original run is copied before restore/training, so it is not modified.
#
# Submit from /projects/$USER/bsk_rl with:
#   sbatch --export=ALL,BSK_RL_CONTINUE_FROM=/scratch/alpine/$USER/rllib_results/<old_output>/<old_run> \
#     examples/amos_2026/sbatch_continue_polaris_gat_full_actions_10d90i_24h.sh
#
# BSK_RL_CONTINUE_FROM may point to a run directory, a checkpoint directory, or an
# output directory containing exactly one run directory.

#SBATCH --account=ucb550_asc2
#SBATCH --job-name=gat_cont_10d90i_24h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=0-0
#SBATCH --partition=amilan
#SBATCH --mem=100G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --qos=normal

set -euo pipefail

: "${BSK_RL_CONTINUE_FROM:?Set BSK_RL_CONTINUE_FROM to the original run or checkpoint directory.}"

module purge

echo "Loading modules"
if ! module --ignore_cache load gcc/14.2.0; then
    echo "WARNING: gcc/14.2.0 module not found; falling back to /curc/sw/install/gcc/14.2.0"
    export PATH="/curc/sw/install/gcc/14.2.0/bin:${PATH}"
fi
if ! module --ignore_cache load python/3.10.2; then
    echo "WARNING: python/3.10.2 module not found; continuing with the virtualenv python"
fi

echo "Activating virtual environment"
source /projects/$USER/.venv/bin/activate

cd /projects/$USER/bsk_rl

export BSK_RL_SCRATCH=/scratch/alpine/$USER
export BSK_RL_OUTPUT_DIR=/scratch/alpine/$USER/rllib_results
export BSK_RL_RAY_TMPDIR=/tmp/bskray_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
export TMPDIR=$BSK_RL_RAY_TMPDIR
trap 'rm -rf "$BSK_RL_RAY_TMPDIR"' EXIT
if [ -d /curc/sw/install/gcc/14.2.0/lib64 ]; then
    export LD_LIBRARY_PATH="/curc/sw/install/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}"
fi
if command -v gcc >/dev/null 2>&1; then
    export LD_LIBRARY_PATH="$(dirname "$(gcc -print-file-name=libstdc++.so.6)"):${LD_LIBRARY_PATH:-}"
fi

export BSK_RL_WANDB_KEY_PATH=${BSK_RL_WANDB_KEY_PATH:-/projects/$USER/bsk_rl/examples/wandb_key.txt}
export BSK_RL_USE_WANDB=${BSK_RL_USE_WANDB:-1}
export BSK_RL_REQUIRE_WANDB=${BSK_RL_REQUIRE_WANDB:-1}
export BSK_RL_WANDB_PROJECT=${BSK_RL_WANDB_PROJECT:-amos2026-bsk-rl}
export BSK_RL_WANDB_GROUP=polaris-gat-full-actions-obs-v9-10d90i-continue24h

export BSK_RL_DYNAMIC_PRIORITY_EVENT=${BSK_RL_DYNAMIC_PRIORITY_EVENT:-1}
export BSK_RL_DYNAMIC_PRIORITY_EVENT_FRACTION=${BSK_RL_DYNAMIC_PRIORITY_EVENT_FRACTION:-0.5}
export BSK_RL_DYNAMIC_PRIORITY_EVENT_TIME_SEC=${BSK_RL_DYNAMIC_PRIORITY_EVENT_TIME_SEC:-}
export BSK_RL_HIO_COUNT=${BSK_RL_HIO_COUNT:-5}
export BSK_RL_HIO_PRIORITY=${BSK_RL_HIO_PRIORITY:-5.0}
export BSK_RL_SHIO_COUNT=${BSK_RL_SHIO_COUNT:-3}
export BSK_RL_SHIO_PRIORITY=${BSK_RL_SHIO_PRIORITY:-10.0}

export BSK_RL_DOWNLINK_BONUS=0.1
export BSK_RL_BATCH_MULTIPLIER=${BSK_RL_BATCH_MULTIPLIER:-150}
export BSK_RL_TOTAL_TIMESTEPS=${BSK_RL_TOTAL_TIMESTEPS:-20000000}
export BSK_RL_DISABLE_TIMESTEP_LIMIT=${BSK_RL_DISABLE_TIMESTEP_LIMIT:-1}
export BSK_RL_TRAIN_TIMEOUT_SEC=${BSK_RL_TRAIN_TIMEOUT_SEC:-84600}
export BSK_RL_CONTINUE_SUFFIX=${BSK_RL_CONTINUE_SUFFIX:-continue24h_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}}
export BSK_RL_CHECKPOINT_FREQUENCY=${BSK_RL_CHECKPOINT_FREQUENCY:-3}
export BSK_RL_TORCH_THREADS=${BSK_RL_TORCH_THREADS:-11}
export BSK_RL_BATTERY_LIFE_MULTIPLIER=${BSK_RL_BATTERY_LIFE_MULTIPLIER:-1}
export BSK_RL_IMAGE_STORAGE_CAPACITY_IMAGES=${BSK_RL_IMAGE_STORAGE_CAPACITY_IMAGES:-50}
export PYTHONUNBUFFERED=1

mkdir -p /scratch/alpine/$USER/job_output "$BSK_RL_OUTPUT_DIR" "$BSK_RL_RAY_TMPDIR"

echo "===== Job context ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "BSK_RL_CONTINUE_FROM=$BSK_RL_CONTINUE_FROM"
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo "commit: $(git rev-parse --short HEAD)"
git status --short --untracked-files=no

echo "gcc path: $(which gcc)"
gcc --version
echo "libstdc++ path: $(gcc -print-file-name=libstdc++.so.6)"
strings "$(gcc -print-file-name=libstdc++.so.6)" | grep GLIBCXX_3.4.29 || true
python3 -c "import bsk_rl; import bsk_rl.sim.simulator; print('bsk_rl import ok')"
python3 -c "import wandb; print('wandb import ok')"

echo "Continuing copied AMOS 2026 Polaris GAT full-action obs-v9 10d90i policy"
python3 -u examples/train_Polaris_gat_full_actions_10d90i_wandb.py

echo "== End of Job =="
