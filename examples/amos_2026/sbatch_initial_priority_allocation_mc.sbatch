#!/usr/bin/env bash

# One array task runs one AMOS 2026 initial-priority allocation episode.
# Even tasks: ground-confirmation cooldown, seeds 0--49.
# Odd tasks: one-orbit cooldown, seeds 0--49.
# Adjacent tasks therefore form a matched cooldown pair for the same seed.

#SBATCH --account=ucb550_asc2
#SBATCH --job-name=amos_initprio
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-99%20
#SBATCH --partition=amilan
#SBATCH --mem=24G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --qos=normal

set -euo pipefail

repo_dir=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl}
output_root=${AMOS_INITIAL_PRIORITY_OUTPUT_ROOT:?Set AMOS_INITIAL_PRIORITY_OUTPUT_ROOT}
policy_spec=${AMOS_INITIAL_PRIORITY_POLICY_SPEC:?Set AMOS_INITIAL_PRIORITY_POLICY_SPEC}

module purge
if ! module --ignore_cache load gcc/14.2.0; then
    export PATH="/curc/sw/install/gcc/14.2.0/bin:${PATH}"
fi
module --ignore_cache load python/3.10.2 || true
source "/projects/$USER/.venv/bin/activate"

cd "$repo_dir"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
imported_bsk_rl=$(python3 -c 'import pathlib, bsk_rl; print(pathlib.Path(bsk_rl.__file__).resolve())')
if [[ "$imported_bsk_rl" != "$repo_dir"/src/bsk_rl/* ]]; then
    echo "Refusing to run: bsk_rl imports from $imported_bsk_rl, not $repo_dir/src." >&2
    exit 3
fi
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
export MPLCONFIGDIR="/scratch/alpine/$USER/.cache/matplotlib"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR" "$output_root" "/scratch/alpine/$USER/job_output"

echo "===== AMOS initial-priority allocation task ====="
date -u
hostname
echo "job=${SLURM_JOB_ID:-} array_task=${SLURM_ARRAY_TASK_ID:-}"
echo "repo=$repo_dir"
echo "bsk_rl_import=$imported_bsk_rl"
echo "commit=$(git rev-parse HEAD)"
echo "output_root=$output_root"
echo "policy_spec=$policy_spec"

python3 -u examples/amos_2026/run_initial_priority_allocation_mc_task.py \
    --task-id "${SLURM_ARRAY_TASK_ID:?}" \
    --seeds-per-case 50 \
    --repo-dir "$repo_dir" \
    --output-root "$output_root" \
    --policy-spec "$policy_spec"
