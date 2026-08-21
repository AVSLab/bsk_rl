#!/usr/bin/env bash

# Two AMOS 2026 heuristics x ten seeds. Each array task owns one heuristic
# and launches ten isolated Basilisk evaluators.

#SBATCH --account=ucb550_asc2
#SBATCH --job-name=heur_mc_s000_009
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
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
    export PATH="/curc/sw/install/gcc/14.2.0/bin:${PATH}"
fi
module --ignore_cache load python/3.10.2 || true
source "/projects/$USER/.venv/bin/activate"
cd "/projects/$USER/bsk_rl"

if [[ -d /curc/sw/install/gcc/14.2.0/lib64 ]]; then
    export LD_LIBRARY_PATH="/curc/sw/install/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}"
fi
if command -v gcc >/dev/null 2>&1; then
    export LD_LIBRARY_PATH="$(dirname "$(gcc -print-file-name=libstdc++.so.6)"):${LD_LIBRARY_PATH:-}"
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

export BSK_RL_HEUR_SEED_START=${BSK_RL_HEUR_SEED_START:-0}
export BSK_RL_HEUR_SEEDS_PER_BLOCK=${BSK_RL_HEUR_SEEDS_PER_BLOCK:-10}
export BSK_RL_HEUR_MODES=${BSK_RL_HEUR_MODES:-angle,priority_angle}
export BSK_RL_HEUR_OUTPUT_ROOT=${BSK_RL_HEUR_OUTPUT_ROOT:?Set BSK_RL_HEUR_OUTPUT_ROOT}

mkdir -p "/scratch/alpine/$USER/job_output" "$BSK_RL_HEUR_OUTPUT_ROOT"

mode_task_id=${SLURM_ARRAY_TASK_ID:-0}
overall_status=0
for ((seed_offset = 0; seed_offset < BSK_RL_HEUR_SEEDS_PER_BLOCK; seed_offset++)); do
    evaluator_task_id=$((mode_task_id * BSK_RL_HEUR_SEEDS_PER_BLOCK + seed_offset))
    if ! python3 -u examples/amos_2026/evaluate_heuristic_mc.py \
        --task-id "$evaluator_task_id" \
        --seeds-per-block "$BSK_RL_HEUR_SEEDS_PER_BLOCK"; then
        echo "WARNING: heuristic task $evaluator_task_id failed" >&2
        overall_status=1
    fi
done

exit "$overall_status"
