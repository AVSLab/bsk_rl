#!/bin/bash

# AMOS 2026 immediate GAT full-action Monte Carlo smoke test:
#   12 non-alpha 48-hour reward-sweep policies x seeds 0..9 = 120 evaluations.
# Each array task owns one policy and launches ten fresh evaluator subprocesses.
# Use submit_gat_reward_sweep_mc_smoke_2h.sh so the current complete checkpoints
# are frozen immediately and the timestamped output folder is exported.

#SBATCH --account=ucb550_asc2
#SBATCH --job-name=gat_mc_smoke_2h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-11%4
#SBATCH --partition=amilan
#SBATCH --mem=24G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --qos=normal

set -euo pipefail

module purge
if ! module --ignore_cache load gcc/14.2.0; then
    echo "WARNING: gcc/14.2.0 module not found; falling back to /curc/sw/install/gcc/14.2.0"
    export PATH="/curc/sw/install/gcc/14.2.0/bin:${PATH}"
fi
if ! module --ignore_cache load python/3.10.2; then
    echo "WARNING: python/3.10.2 module not found; continuing with the virtualenv python"
fi
source /projects/$USER/.venv/bin/activate

cd /projects/$USER/bsk_rl

if [ -d /curc/sw/install/gcc/14.2.0/lib64 ]; then
    export LD_LIBRARY_PATH="/curc/sw/install/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}"
fi
if command -v gcc >/dev/null 2>&1; then
    export LD_LIBRARY_PATH="$(dirname "$(gcc -print-file-name=libstdc++.so.6)"):${LD_LIBRARY_PATH:-}"
fi

export BSK_RL_MC_SEED_START=${BSK_RL_MC_SEED_START:-0}
export BSK_RL_MC_SEEDS_PER_BLOCK=${BSK_RL_MC_SEEDS_PER_BLOCK:-10}
export BSK_RL_MC_N_TARGETS=${BSK_RL_MC_N_TARGETS:-100}
export BSK_RL_MC_N_TARGETS_AHEAD=${BSK_RL_MC_N_TARGETS_AHEAD:-10}
export BSK_RL_MC_EXTRA_TIME_FACTOR=${BSK_RL_MC_EXTRA_TIME_FACTOR:-1.5}
export BSK_RL_MC_TOTAL_TIME_SEC=${BSK_RL_MC_TOTAL_TIME_SEC:-}
export BSK_RL_MC_OUTPUT_ROOT=${BSK_RL_MC_OUTPUT_ROOT:?Set BSK_RL_MC_OUTPUT_ROOT to the timestamped smoke-test folder}
export BSK_RL_MC_MANIFEST=${BSK_RL_MC_MANIFEST:?Set BSK_RL_MC_MANIFEST to a frozen checkpoint manifest}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export MPLBACKEND=${MPLBACKEND:-Agg}
export PYTHONUNBUFFERED=1

mkdir -p /scratch/alpine/$USER/job_output "$BSK_RL_MC_OUTPUT_ROOT"

echo "===== AMOS 2026 two-hour Monte Carlo smoke task ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "BSK_RL_MC_SEED_START=$BSK_RL_MC_SEED_START"
echo "BSK_RL_MC_SEEDS_PER_BLOCK=$BSK_RL_MC_SEEDS_PER_BLOCK"
echo "BSK_RL_MC_N_TARGETS=$BSK_RL_MC_N_TARGETS"
echo "BSK_RL_MC_N_TARGETS_AHEAD=$BSK_RL_MC_N_TARGETS_AHEAD"
echo "BSK_RL_MC_EXTRA_TIME_FACTOR=$BSK_RL_MC_EXTRA_TIME_FACTOR"
echo "BSK_RL_MC_TOTAL_TIME_SEC=$BSK_RL_MC_TOTAL_TIME_SEC"
echo "BSK_RL_MC_MANIFEST=$BSK_RL_MC_MANIFEST"
echo "BSK_RL_MC_OUTPUT_ROOT=$BSK_RL_MC_OUTPUT_ROOT"
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo "commit: $(git rev-parse --short HEAD)"
git status --short --untracked-files=no

policy_task_id=${SLURM_ARRAY_TASK_ID:-0}
overall_status=0
for ((seed_offset = 0; seed_offset < BSK_RL_MC_SEEDS_PER_BLOCK; seed_offset++)); do
    evaluator_task_id=$((policy_task_id * BSK_RL_MC_SEEDS_PER_BLOCK + seed_offset))
    echo
    echo "===== Running policy task $policy_task_id seed offset $seed_offset ====="
    if ! python3 -u examples/amos_2026/evaluate_gat_reward_sweep_mc.py \
        --task-id "$evaluator_task_id" \
        --seeds-per-block "$BSK_RL_MC_SEEDS_PER_BLOCK"; then
        echo "WARNING: evaluator task $evaluator_task_id failed; continuing with the remaining seeds" >&2
        overall_status=1
    fi
done

exit "$overall_status"
