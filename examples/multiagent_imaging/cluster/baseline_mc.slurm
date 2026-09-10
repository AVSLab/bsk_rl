#!/usr/bin/env bash
#SBATCH --job-name=bsk-baseline-mc
#SBATCH --account=ucb550_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --array=0-199%8
#SBATCH --output=baseline-mc-%A_%a.log

# Each array element propagates one independent 103-spacecraft simulation:
# three sensing agents and 100 passive Basilisk/Vizard RSO spacecraft.
# No Ray, learner, GPU, checkpoint, radio action, or MPI job is required.
# Account/partition/QOS verified on the cluster via Termius on 2026-09-06.
set -euo pipefail
: "${BSK_PROJECT_ROOT:?Set BSK_PROJECT_ROOT to the deployed source directory}"
: "${BSK_RL_PYTHON:?Set BSK_RL_PYTHON to the validated cluster Python executable}"
: "${BSK_MC_MANIFEST:?Set BSK_MC_MANIFEST to the reviewed campaign manifest}"
: "${BSK_MC_OUTPUT:?Set BSK_MC_OUTPUT to the episode output directory}"
: "${SLURM_JOB_ID:?Do not run simulations on a login node; submit after authorization}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"
cd "$BSK_PROJECT_ROOT"
export PYTHONPATH="$BSK_PROJECT_ROOT/src:$BSK_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Load the C++ runtime used to build the isolated Basilisk modules.
BSK_GCC_ROOT="${BSK_GCC_ROOT:-/curc/sw/install/gcc/14.2.0}"
export LD_LIBRARY_PATH="$BSK_GCC_ROOT/lib64:$BSK_GCC_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$BSK_MC_OUTPUT"
"$BSK_RL_PYTHON" examples/multiagent_imaging/cluster/audit_runtime.py \
  --output "$BSK_MC_OUTPUT/runtime-${SLURM_ARRAY_TASK_ID}.json"
exec "$BSK_RL_PYTHON" -m examples.multiagent_imaging.baseline_monte_carlo run \
  --manifest "$BSK_MC_MANIFEST" --task-id "$SLURM_ARRAY_TASK_ID" \
  --output-dir "$BSK_MC_OUTPUT"
