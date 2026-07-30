#!/usr/bin/env bash

# Eight-policy AMOS 2026 training sweep:
#   alpha = 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0
#   100 targets/catalog, exactly 50 LEO + 30 MEO + 20 GEO
#   10 target candidates exposed to the obs-v9 GAT policy

#SBATCH --account=ucb550_asc2
#SBATCH --job-name=gat_mix100_fixed_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --array=0-7%4
#SBATCH --partition=amilan
#SBATCH --mem=100G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --qos=long

set -euo pipefail

POLICY_TAGS=(
    00d100i
    10d90i
    20d80i
    30d70i
    40d60i
    50d50i
    75d25i
    100d00i
)
ALPHAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.75 1.0)
ALPHA_NAMES=(0p0 0p1 0p2 0p3 0p4 0p5 0p75 1p0)

SWEEP_INDEX=${SLURM_ARRAY_TASK_ID:-0}
if (( SWEEP_INDEX < 0 || SWEEP_INDEX >= ${#POLICY_TAGS[@]} )); then
    echo "SLURM_ARRAY_TASK_ID must be in [0, 7], got $SWEEP_INDEX" >&2
    exit 2
fi
POLICY_TAG=${POLICY_TAGS[$SWEEP_INDEX]}
ALPHA=${ALPHAS[$SWEEP_INDEX]}
ALPHA_NAME=${ALPHA_NAMES[$SWEEP_INDEX]}

module purge
echo "Loading modules"
if ! module --ignore_cache load gcc/14.2.0; then
    echo "WARNING: gcc/14.2.0 not found; using /curc/sw/install/gcc/14.2.0"
    export PATH="/curc/sw/install/gcc/14.2.0/bin:${PATH}"
fi
if ! module --ignore_cache load python/3.10.2; then
    echo "WARNING: python/3.10.2 not found; continuing with virtualenv Python"
fi

source /projects/$USER/.venv/bin/activate
cd /projects/$USER/bsk_rl

export BSK_RL_SCRATCH=/scratch/alpine/$USER
export BSK_RL_OUTPUT_DIR=/scratch/alpine/$USER/rllib_results
export BSK_RL_RAY_TMPDIR=/tmp/bskray_${SLURM_JOB_ID}_${SWEEP_INDEX}
export TMPDIR=$BSK_RL_RAY_TMPDIR
trap 'rm -rf "$BSK_RL_RAY_TMPDIR"' EXIT

if [[ -d /curc/sw/install/gcc/14.2.0/lib64 ]]; then
    export LD_LIBRARY_PATH="/curc/sw/install/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}"
fi
if command -v gcc >/dev/null 2>&1; then
    export LD_LIBRARY_PATH="$(dirname "$(gcc -print-file-name=libstdc++.so.6)"):${LD_LIBRARY_PATH:-}"
fi

export BSK_RL_WANDB_KEY_PATH=${BSK_RL_WANDB_KEY_PATH:-/projects/$USER/bsk_rl/examples/wandb_key.txt}
export BSK_RL_USE_WANDB=${BSK_RL_USE_WANDB:-1}
export BSK_RL_REQUIRE_WANDB=${BSK_RL_REQUIRE_WANDB:-1}
export BSK_RL_WANDB_PROJECT=${BSK_RL_WANDB_PROJECT:-amos2026-bsk-rl}
export BSK_RL_WANDB_GROUP=polaris-gat-full-actions-obs-v9-mixed-fixed-50leo30meo20geo-100targets-reward-sweep
export WANDB_MODE=${WANDB_MODE:-online}

export BSK_RL_DOWNLINK_BONUS=$ALPHA
export BSK_RL_REWARD_SPLIT_TAG=${POLICY_TAG}MixedFixed50LEO30MEO20GEO100Targets
export BSK_RL_ALPHA_TAG=alpha${ALPHA_NAME}_mixedFixed50LEO30MEO20GEO100Targets
export BSK_RL_TARGET_ENV=mixed
export BSK_RL_RANDOMIZE_MIX_WEIGHTS=0
export BSK_RL_MIX_WEIGHTS='{"LEO":0.5,"MEO":0.3,"GEO":0.2}'
export BSK_RL_EXACT_MIX_COUNTS=1
export BSK_RL_RANDOMIZE_N_TARGETS=0
export BSK_RL_N_TARGETS=100
export BSK_RL_N_TARGETS_AHEAD=10
# The Slurm array selects alpha; the trainer itself still has one RLlib job.
export BSK_RL_JOB_INDEX=0

export BSK_RL_DYNAMIC_PRIORITY_EVENT=${BSK_RL_DYNAMIC_PRIORITY_EVENT:-1}
export BSK_RL_DYNAMIC_PRIORITY_EVENT_FRACTION=${BSK_RL_DYNAMIC_PRIORITY_EVENT_FRACTION:-0.5}
export BSK_RL_DYNAMIC_PRIORITY_EVENT_TIME_SEC=${BSK_RL_DYNAMIC_PRIORITY_EVENT_TIME_SEC:-}
export BSK_RL_HIO_COUNT=${BSK_RL_HIO_COUNT:-5}
export BSK_RL_HIO_PRIORITY=${BSK_RL_HIO_PRIORITY:-5.0}
export BSK_RL_SHIO_COUNT=${BSK_RL_SHIO_COUNT:-3}
export BSK_RL_SHIO_PRIORITY=${BSK_RL_SHIO_PRIORITY:-10.0}

export BSK_RL_BATCH_MULTIPLIER=${BSK_RL_BATCH_MULTIPLIER:-150}
export BSK_RL_TOTAL_TIMESTEPS=${BSK_RL_TOTAL_TIMESTEPS:-20000000}
export BSK_RL_CHECKPOINT_FREQUENCY=${BSK_RL_CHECKPOINT_FREQUENCY:-3}
export BSK_RL_TORCH_THREADS=${BSK_RL_TORCH_THREADS:-11}
export BSK_RL_BATTERY_LIFE_MULTIPLIER=${BSK_RL_BATTERY_LIFE_MULTIPLIER:-1}
export BSK_RL_IMAGE_STORAGE_CAPACITY_IMAGES=${BSK_RL_IMAGE_STORAGE_CAPACITY_IMAGES:-50}
export PYTHONUNBUFFERED=1

mkdir -p /scratch/alpine/$USER/job_output "$BSK_RL_OUTPUT_DIR" "$BSK_RL_RAY_TMPDIR"

if [[ "$BSK_RL_USE_WANDB" == "1" ]]; then
    if [[ ! -s "$BSK_RL_WANDB_KEY_PATH" ]]; then
        echo "W&B key file is missing or empty: $BSK_RL_WANDB_KEY_PATH" >&2
        exit 1
    fi
    python3 -c "import wandb; print('wandb import ok:', wandb.__version__)"
fi
python3 -c "import bsk_rl; import bsk_rl.sim.simulator; print('bsk_rl import ok')"

echo "===== AMOS 2026 fixed mixed-100 training ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SWEEP_INDEX=$SWEEP_INDEX"
echo "POLICY_TAG=$POLICY_TAG"
echo "DOWNLINK_REWARD_WEIGHT_ALPHA=$ALPHA"
echo "CATALOG=100 targets: exactly 50 LEO, 30 MEO, 20 GEO"
echo "N_TARGETS_AHEAD=$BSK_RL_N_TARGETS_AHEAD"
echo "WANDB_PROJECT=$BSK_RL_WANDB_PROJECT"
echo "WANDB_GROUP=$BSK_RL_WANDB_GROUP"
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo "commit: $(git rev-parse --short HEAD)"
git status --short --untracked-files=no

python3 -u examples/train_Polaris_gat_full_actions_wandb.py

echo "== End of Job =="
