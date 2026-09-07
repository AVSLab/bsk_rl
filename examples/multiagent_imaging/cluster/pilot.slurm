#!/usr/bin/env bash
#SBATCH --job-name=bsk-pilot
#SBATCH --account=ucb550_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=bsk-pilot-%j.log

# Run only after explicit authorization. This file does not submit another job.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through an authorized Slurm allocation}"
: "${BSK_PROJECT_ROOT:?Set the reviewed completion-v2 project path}"
: "${BSK_RL_PYTHON:?Set the validated Python 3.11 interpreter}"
: "${BASILISK_SOURCE_ROOT:?Set the clean tested Basilisk source checkout}"
: "${BSK_OUTPUT:?Set a fresh output directory}"
cd "$BSK_PROJECT_ROOT"
export PYTHONPATH="$BSK_PROJECT_ROOT/src:$BSK_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Load the C++ runtime used to build the isolated Basilisk modules.
BSK_GCC_ROOT="${BSK_GCC_ROOT:-/curc/sw/install/gcc/14.2.0}"
export LD_LIBRARY_PATH="$BSK_GCC_ROOT/lib64:$BSK_GCC_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Short, job-specific Unix socket paths; no shared Ray head or global ray stop.
export BSK_RL_RAY_TMPDIR="/tmp/bsk-ray-${SLURM_JOB_ID}"
mkdir -p "$BSK_RL_RAY_TMPDIR"
pilot_args=(--stage "${BSK_STAGE:-validate}" --output "$BSK_OUTPUT"
            --updates "${BSK_UPDATES:-8}")
if [[ "${BSK_STAGE:-validate}" == learn ]]; then
  : "${BSK_VALIDATION_GATE:?Review a passed one-worker gate before four workers}"
  pilot_args+=(--validation-gate "$BSK_VALIDATION_GATE")
fi
exec "$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.learning_pilot "${pilot_args[@]}"
