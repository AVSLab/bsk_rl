#!/usr/bin/env bash
#SBATCH --job-name=bsk-amos-matched
#SBATCH --account=ucb550_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=bsk-amos-matched-%j.log

# ONE diagnostic with the selected AMOS run's optimizer and discount settings.
# Resource topology and multi-agent mission/private observations remain declared changes.
set -euo pipefail
: "${SLURM_JOB_ID:?Use a Slurm allocation}"
: "${BSK_PROJECT_ROOT:?Set the NEW reviewed source directory}"
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
export BSK_RL_RAY_TMPDIR="/tmp/bsk-amos-${SLURM_JOB_ID}"
mkdir -p "$BSK_RL_RAY_TMPDIR"
test ! -e "$BSK_OUTPUT"
BSK_PREFLIGHT="${BSK_OUTPUT}.preflight"
test ! -e "$BSK_PREFLIGHT"
mkdir -p "$BSK_PREFLIGHT"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.audit_runtime \
  --require-allocation --output "$BSK_PREFLIGHT/runtime.json"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.prepare_support_data \
  --output "$BSK_PREFLIGHT/support-data.json"
"$BSK_RL_PYTHON" - "$BSK_PREFLIGHT" <<'PY'
import hashlib, json, os, pathlib, sys
root = pathlib.Path.cwd()
release = json.loads((root/'completion-release.json').read_text())
for name, expected in release['source_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
import wandb
assert wandb.__version__ == '0.26.1'
wandb.login(key=pathlib.Path(os.environ['BSK_RL_WANDB_KEY_PATH']).read_text().strip(), relogin=False)
run = wandb.Api(timeout=60).run('daniel-hutererprats-university-of-colorado-boulder/amos2026-bsk-rl/4osb53df')
record = dict(source_hashes_verified=len(release['source_sha256']), online_wandb=True, wandb_version=wandb.__version__, reference_run_id=run.id)
(pathlib.Path(sys.argv[1])/'source-online-gates.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record), flush=True)
PY
# Gate data boundaries, temporal targets, masks, physical products and restore
# using the actual compute-node source/native/runtime before mission training.
"$BSK_RL_PYTHON" -m pytest -q \
  tests/unittest/multiagent/test_optimization_profiles.py \
  tests/unittest/multiagent/test_async_discounting.py \
  tests/unittest/multiagent/test_completion_catalog.py \
  tests/unittest/multiagent/test_completion_products.py \
  tests/unittest/multiagent/test_data_boundaries.py \
  tests/unittest/multiagent/test_completion_policy_mask.py \
  tests/unittest/multiagent/test_completion_set_attention.py \
  tests/unittest/multiagent/test_directed_hold.py \
  tests/unittest/multiagent/test_wandb_logging.py \
  tests/integration/multiagent/test_checkpoint_resume.py \
  > "$BSK_PREFLIGHT/focused-tests.log" 2>&1
BSK_CONFIG="examples/multiagent_imaging/configs/walker4_mixed_amos_best_conflict.json"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.train \
  --config "$BSK_CONFIG" --output "$BSK_OUTPUT" \
  --iterations 3 --train-batch-size 4200 --num-env-runners 1 --num-learners 0 \
  --ray-cpus "$SLURM_CPUS_PER_TASK" --torch-threads 1 --sample-timeout-s 3600 \
  --ppo-profile amos_best --save-initial --capture-learner-batch \
  --wandb-mode online --require-wandb
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.fixed_batch_diagnostic \
  --config "$BSK_CONFIG" --output "$BSK_OUTPUT" --checkpoint "$BSK_OUTPUT/checkpoint_0000"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.compare_learned_checkpoints \
  --config "$BSK_CONFIG" --checkpoint "initial=$BSK_OUTPUT/checkpoint_0000" \
  --checkpoint "trained=$BSK_OUTPUT/checkpoint_0003" --seeds 401 402 \
  --output "$BSK_OUTPUT/heldout_validation"
# Bounded restore check, zero extra PPO updates, same online W&B identity.
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.train \
  --config "$BSK_CONFIG" --output "$BSK_OUTPUT/resume_validation" \
  --resume "$BSK_OUTPUT/checkpoint_0003" --iterations 0 --train-batch-size 4200 \
  --num-env-runners 1 --num-learners 0 --ray-cpus "$SLURM_CPUS_PER_TASK" \
  --torch-threads 1 --sample-timeout-s 3600 --ppo-profile amos_best \
  --wandb-mode online --require-wandb
cp "$BSK_PREFLIGHT/runtime.json" "$BSK_OUTPUT/runtime-audit.json"
cp "$BSK_PREFLIGHT/support-data.json" "$BSK_OUTPUT/support-data-audit.json"
cp "$BSK_PREFLIGHT/source-online-gates.json" "$BSK_OUTPUT/source-online-gates.json"
cp "$BSK_PREFLIGHT/focused-tests.log" "$BSK_OUTPUT/focused-tests.log"
date -u '+Diagnostic complete at %Y-%m-%dT%H:%M:%SZ'
