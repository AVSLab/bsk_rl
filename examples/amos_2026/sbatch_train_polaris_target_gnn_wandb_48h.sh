#!/bin/bash

# 48-hour CURC training job for the AMOS 2026 Polaris Target-GNN + W&B trainer.
# Submit from /projects/$USER/bsk_rl with:
#   sbatch examples/amos_2026/sbatch_train_polaris_target_gnn_wandb_48h.sh

#SBATCH --account=ucb550_asc2
#SBATCH --job-name=polaris_tgn_48h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --array=0-0
#SBATCH --partition=amilan
#SBATCH --mem=100G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --qos=long

set -euo pipefail

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

export BSK_RL_SCRATCH=/scratch/alpine/$USER # cluster scratch root; local script uses ~/rllib_results/...
export BSK_RL_OUTPUT_DIR=/scratch/alpine/$USER/rllib_results # cluster TensorBoard/checkpoints; local uses ~/rllib_results/may_results/...
# Ray creates AF_UNIX sockets below this directory; keep the path short.
export BSK_RL_RAY_TMPDIR=/tmp/bskray_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0} # cluster; local uses ~/rllib_results/ray_tmp/...
export TMPDIR=$BSK_RL_RAY_TMPDIR # Ray also checks TMPDIR internally.
trap 'rm -rf "$BSK_RL_RAY_TMPDIR"' EXIT
if [ -d /curc/sw/install/gcc/14.2.0/lib64 ]; then
    export LD_LIBRARY_PATH="/curc/sw/install/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}"
fi
if command -v gcc >/dev/null 2>&1; then
    export LD_LIBRARY_PATH="$(dirname "$(gcc -print-file-name=libstdc++.so.6)"):${LD_LIBRARY_PATH:-}"
fi

# Put the token file here on the cluster, or override this path before sbatch.
export BSK_RL_WANDB_KEY_PATH=${BSK_RL_WANDB_KEY_PATH:-/projects/$USER/bsk_rl/examples/wandb_key.txt} # local: examples/wandb_key.txt
export BSK_RL_USE_WANDB=${BSK_RL_USE_WANDB:-1} # set to 0 before sbatch for a no-W&B run.
export BSK_RL_REQUIRE_WANDB=${BSK_RL_REQUIRE_WANDB:-1} # fail fast if key/dependency is missing on cluster.
export BSK_RL_WANDB_PROJECT=${BSK_RL_WANDB_PROJECT:-amos2026-bsk-rl}
export BSK_RL_WANDB_GROUP=${BSK_RL_WANDB_GROUP:-polaris-target-gnn-imaging-only-obs-v9-48h}
export BSK_RL_DYNAMIC_PRIORITY_EVENT=${BSK_RL_DYNAMIC_PRIORITY_EVENT:-1}
export BSK_RL_DYNAMIC_PRIORITY_EVENT_FRACTION=${BSK_RL_DYNAMIC_PRIORITY_EVENT_FRACTION:-0.5}
export BSK_RL_DYNAMIC_PRIORITY_EVENT_TIME_SEC=${BSK_RL_DYNAMIC_PRIORITY_EVENT_TIME_SEC:-}
export BSK_RL_HIO_COUNT=${BSK_RL_HIO_COUNT:-5}
export BSK_RL_HIO_PRIORITY=${BSK_RL_HIO_PRIORITY:-5.0}
export BSK_RL_SHIO_COUNT=${BSK_RL_SHIO_COUNT:-3}
export BSK_RL_SHIO_PRIORITY=${BSK_RL_SHIO_PRIORITY:-10.0}

export BSK_RL_BATCH_MULTIPLIER=${BSK_RL_BATCH_MULTIPLIER:-150} # local default inside Python: 32
export BSK_RL_TOTAL_TIMESTEPS=${BSK_RL_TOTAL_TIMESTEPS:-20000000} # full cluster train; local default inside Python: 10000
export BSK_RL_CHECKPOINT_FREQUENCY=${BSK_RL_CHECKPOINT_FREQUENCY:-3}
export BSK_RL_TORCH_THREADS=${BSK_RL_TORCH_THREADS:-11}
export BSK_RL_BATTERY_LIFE_MULTIPLIER=${BSK_RL_BATTERY_LIFE_MULTIPLIER:-1000}
export BSK_RL_IMAGE_STORAGE_CAPACITY_IMAGES=${BSK_RL_IMAGE_STORAGE_CAPACITY_IMAGES:-500}
export PYTHONUNBUFFERED=1

mkdir -p /scratch/alpine/$USER/job_output "$BSK_RL_OUTPUT_DIR" "$BSK_RL_RAY_TMPDIR"

echo "===== Job context ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo "commit: $(git rev-parse --short HEAD)"
git status --short --untracked-files=no

echo "gcc path: $(which gcc)"
gcc --version
echo "libstdc++ path: $(gcc -print-file-name=libstdc++.so.6)"
strings "$(gcc -print-file-name=libstdc++.so.6)" | grep GLIBCXX_3.4.29 || true
python3 -c "import bsk_rl; import bsk_rl.sim.simulator; print('bsk_rl import ok')"
python3 -c "import wandb; print('wandb import ok')"

echo "Running AMOS 2026 Polaris Target-GNN obs-v9 + W&B training script"
python3 -u examples/train_Polaris_target_gnn_wandb.py

echo "== End of Job =="
