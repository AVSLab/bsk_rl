#!/usr/bin/env bash

# AMOS 2026 paired-controller Monte Carlo campaign:
#   3 controllers x 10 seed blocks x 10 seeds = 300 evaluations.
# Each Slurm array element owns one controller/seed-block pair and runs its ten
# seeds sequentially. The array throttle therefore limits the complete campaign,
# including when other jobs are already using the same allocation.

#SBATCH --account=ucb550_asc2
#SBATCH --job-name=base120_1orb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-29%4
#SBATCH --partition=acpu
#SBATCH --mem=24G
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --qos=normal

set -euo pipefail

module purge
if ! module --ignore_cache load gcc/14.2.0; then
    export PATH="/curc/sw/install/gcc/14.2.0/bin:${PATH}"
fi
module --ignore_cache load python/3.10.2 || true

workdir=${BSK_RL_CLUSTER_WORKDIR:?Set BSK_RL_CLUSTER_WORKDIR to the isolated AMOS 2026 checkout}
venv_dir=${BSK_RL_CLUSTER_VENV:-/projects/$USER/.venv}
expected_branch=${BSK_RL_EXPECTED_BRANCH:-amos-2026-space-imaging}

# shellcheck source=/dev/null
source "$venv_dir/bin/activate"
cd "$workdir"
export PYTHONPATH="$workdir/src${PYTHONPATH:+:$PYTHONPATH}"

actual_branch=$(git branch --show-current)
if [[ "$actual_branch" != "$expected_branch" ]]; then
    echo "Refusing to run from branch '$actual_branch'; expected '$expected_branch'." >&2
    exit 3
fi

imported_bsk_rl=$(python3 -c 'import pathlib, bsk_rl; print(pathlib.Path(bsk_rl.__file__).resolve())')
if [[ "$imported_bsk_rl" != "$workdir"/src/bsk_rl/* ]]; then
    echo "Refusing to run: bsk_rl imports from $imported_bsk_rl, not $workdir/src." >&2
    exit 3
fi

if [[ -d /curc/sw/install/gcc/14.2.0/lib64 ]]; then
    export LD_LIBRARY_PATH="/curc/sw/install/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}"
fi
if command -v gcc >/dev/null 2>&1; then
    gcc_libstdcpp=$(gcc -print-file-name=libstdc++.so.6)
    gcc_lib_dir=$(dirname "$gcc_libstdcpp")
    LD_LIBRARY_PATH="$gcc_lib_dir:${LD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

controller_csv=${BSK_RL_HEUR_MODES:-angle,candidate_priority,random}
seed_origin=${BSK_RL_HEUR_SEED_ORIGIN:-0}
total_seeds=${BSK_RL_HEUR_TOTAL_SEEDS:-100}
seeds_per_block=${BSK_RL_HEUR_SEEDS_PER_BLOCK:-10}
output_root=${BSK_RL_HEUR_OUTPUT_ROOT:?Set BSK_RL_HEUR_OUTPUT_ROOT}

IFS=',' read -r -a controllers <<< "$controller_csv"
controller_count=${#controllers[@]}
if (( total_seeds <= 0 || seeds_per_block <= 0 || total_seeds % seeds_per_block != 0 )); then
    echo "Total seeds must be positive and divisible by seeds per block." >&2
    exit 2
fi

blocks_per_controller=$((total_seeds / seeds_per_block))
expected_tasks=$((controller_count * blocks_per_controller))
array_task_id=${SLURM_ARRAY_TASK_ID:-0}
if (( array_task_id < 0 || array_task_id >= expected_tasks )); then
    echo "SLURM_ARRAY_TASK_ID must be in [0, $((expected_tasks - 1))]." >&2
    exit 2
fi

controller_index=$((array_task_id / blocks_per_controller))
block_index=$((array_task_id % blocks_per_controller))
controller=${controllers[$controller_index]//[[:space:]]/}
seed_start=$((seed_origin + block_index * seeds_per_block))
seed_stop=$((seed_start + seeds_per_block - 1))

mkdir -p "/scratch/alpine/$USER/job_output" "$output_root"

echo "===== AMOS 2026 baseline Monte Carlo block ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=$array_task_id"
echo "controller=$controller"
echo "seeds=$seed_start..$seed_stop"
echo "output_root=$output_root"
echo "workdir=$workdir"
echo "venv_dir=$venv_dir"
echo "bsk_rl_import=$imported_bsk_rl"
echo "n_targets=${BSK_RL_HEUR_N_TARGETS:-120}"
echo "candidate_count=${BSK_RL_HEUR_N_TARGETS_AHEAD:-10}"
echo "cooldown_orbits=${BSK_RL_HEUR_REIMAGE_COOLDOWN_ORBITS:-1}"
echo "shield_only=${BSK_RL_HEUR_SHIELD_ONLY:-1}"
echo "branch: $actual_branch"
echo "commit: $(git rev-parse --short HEAD)"
git status --short --untracked-files=no

overall_status=0
for ((seed_offset = 0; seed_offset < seeds_per_block; seed_offset++)); do
    echo
    echo "===== Running $controller, seed $((seed_start + seed_offset)) ====="
    if ! python3 -u examples/amos_2026/evaluate_heuristic_mc.py \
        --task-id "$seed_offset" \
        --seed-start "$seed_start" \
        --seeds-per-block "$seeds_per_block" \
        --modes "$controller" \
        --output-root "$output_root" \
        --shield-only; then
        echo "WARNING: $controller seed $((seed_start + seed_offset)) failed." >&2
        overall_status=1
    fi
done

exit "$overall_status"
