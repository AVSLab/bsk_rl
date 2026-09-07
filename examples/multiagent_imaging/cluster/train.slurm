#!/usr/bin/env bash
#SBATCH --job-name=bsk-completion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=bsk-completion-%j.log

# Submit only after local preflight and explicit approval to use the cluster.
# Load the site's compiler/Python modules and activate a validated Basilisk/Ray
# environment before submission. BSK_RL_PYTHON must identify that interpreter.
set -euo pipefail
: "${BSK_RL_PYTHON:?Set BSK_RL_PYTHON to the cluster environment Python executable}"
BSK_PROJECT_ROOT="${BSK_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BSK_PROJECT_ROOT"
export PYTHONPATH="$BSK_PROJECT_ROOT/src:$BSK_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# One Basilisk environment per process; no vectorized simulators or Vizard files.
# Defaults are one bounded update, CPU learner, and one rollout worker.
# Prefer pilot.slurm for the reviewed one-worker gate followed by four workers.
exec "$BSK_RL_PYTHON" -m examples.multiagent_imaging.train \
  --config "${BSK_CONFIG:-examples/multiagent_imaging/configs/mission_preflight_conflict.json}" \
  --output "${BSK_OUTPUT:-results/multiagent_imaging/cluster-${SLURM_JOB_ID:-local}}" \
  --iterations "${BSK_ITERATIONS:-1}" \
  --train-batch-size "${BSK_BATCH_SIZE:-64}" \
  --num-env-runners "${BSK_ENV_RUNNERS:-1}" \
  --num-learners "${BSK_LEARNERS:-0}" \
  --gpus-per-learner "${BSK_LEARNER_GPUS:-0}" \
  --ray-cpus "${SLURM_CPUS_PER_TASK:-8}" \
  --torch-threads 1 --sample-timeout-s "${BSK_SAMPLE_TIMEOUT_S:-1800}"
