#!/usr/bin/env bash
#SBATCH --job-name=bsk-learning-repair
#SBATCH --account=ucb550_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=bsk-learning-repair-%j.log

# Bounded diagnostic; this file never submits another job.
set -euo pipefail
: "${SLURM_JOB_ID:?Use a Slurm allocation}"
: "${BSK_PROJECT_ROOT:?Set the isolated reviewed source directory}"
: "${BSK_RL_PYTHON:?Set the validated Python interpreter}"
: "${BSK_RUNTIME_EXTRAS:?Set the isolated W&B dependency directory}"
: "${BSK_OUTPUT:?Set a fresh output directory}"
: "${BSK_RL_WANDB_KEY_PATH:?Set the existing private credential file path}"
: "${BSK_SUPPORT_DATA_CACHE:?Set the physical support-data cache}"
: "${BASILISK_SOURCE_ROOT:?Set the validated Basilisk source directory}"
cd "$BSK_PROJECT_ROOT"
export PYTHONPATH="$BSK_PROJECT_ROOT/src:$BSK_PROJECT_ROOT:$BSK_RUNTIME_EXTRAS"
export LD_LIBRARY_PATH="/curc/sw/install/gcc/14.2.0/lib64:/curc/sw/install/gcc/14.2.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
export BSK_RL_RAY_TMPDIR="/tmp/bsk-repair-${SLURM_JOB_ID}"
mkdir -p "$BSK_RL_RAY_TMPDIR"
test ! -e "$BSK_OUTPUT"
BSK_PREFLIGHT="${BSK_OUTPUT}.preflight"
test ! -e "$BSK_PREFLIGHT"
mkdir -p "$BSK_PREFLIGHT"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.audit_runtime \
  --require-allocation --output "$BSK_PREFLIGHT/runtime.json"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.prepare_support_data \
  --output "$BSK_PREFLIGHT/support-data.json"
BSK_RECOVERY_CONFIG="${BSK_CONFIG:-examples/multiagent_imaging/configs/walker4_mixed_preflight_conflict.json}"
BSK_RECOVERY_ITERATIONS="${BSK_ITERATIONS:-3}"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.train \
  --config "$BSK_RECOVERY_CONFIG" --output "$BSK_OUTPUT" \
  --iterations "$BSK_RECOVERY_ITERATIONS" --train-batch-size "${BSK_BATCH_SIZE:-1024}" \
  --num-env-runners "${BSK_ENV_RUNNERS:-1}" --num-learners 0 \
  --ray-cpus "$SLURM_CPUS_PER_TASK" --torch-threads 1 --sample-timeout-s 3600 \
  --ppo-profile learning_recovery --save-initial --wandb-mode online --require-wandb
printf -v BSK_LAST_CHECKPOINT 'checkpoint_%04d' "$BSK_RECOVERY_ITERATIONS"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.compare_learned_checkpoints \
  --config "$BSK_RECOVERY_CONFIG" \
  --checkpoint "initial=$BSK_OUTPUT/checkpoint_0000" \
  --checkpoint "trained=$BSK_OUTPUT/$BSK_LAST_CHECKPOINT" \
  --seeds 401 402 --output "$BSK_OUTPUT/heldout_validation"
cp "$BSK_PREFLIGHT/runtime.json" "$BSK_OUTPUT/runtime-audit.json"
cp "$BSK_PREFLIGHT/support-data.json" "$BSK_OUTPUT/support-data-audit.json"
