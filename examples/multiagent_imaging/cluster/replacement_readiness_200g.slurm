#!/usr/bin/env bash
#SBATCH --job-name=multiagent-checkpoint-communication-gate
#SBATCH --account=ucb550_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=multiagent-study-readiness-%j.log
# PROPOSAL ONLY. A fresh hashed release and explicit approval are required.
# 56 allocated CPUs for 200 GiB; keep 28 remote samplers and local learner.
# One allocation only. Canary must pass before PPO.
# <=2 PPO updates, <=120k true decisions total, 16 held-out missions.
set -euo pipefail
: "${BSK_PROFILED_READINESS_APPROVED:?Explicit approval and fresh reviewed release required}"
export BSK_RESERVATION_CAP=112
export BSK_EXPECTED_CPUS=56 BSK_EXPECTED_WALL_S=7200 BSK_EXPECTED_MEMORY_MIB=204800
source "${BSK_PROJECT_ROOT:?}/examples/multiagent_imaging/cluster/study_common.sh"
capture_gate_exit() {
  export BSK_EXIT_CODE=$?
  trap - EXIT
  "$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.capture_ray_evidence --root "$BSK_RL_RAY_TMPDIR" --output "$BSK_OUTPUT/ray-evidence" || true
  "$BSK_RL_PYTHON" - "$BSK_OUTPUT" <<'PY'
import json,os,pathlib,sys,time
elapsed=time.time()-int(os.environ['BSK_JOB_START_UNIX'])
(pathlib.Path(sys.argv[1])/'job-exit.json').write_text(json.dumps(dict(exit_code=int(os.environ['BSK_EXIT_CODE']),elapsed_wall_s=elapsed,
allocated_cpus=int(os.environ['BSK_ALLOCATED_CPUS']),actual_allocated_cpu_h=int(os.environ['BSK_ALLOCATED_CPUS'])*elapsed/3600,
accounting_semantics='Provisional elapsed allocation cost; reconcile final sacct'),indent=2)+'\n')
PY
  exit "$BSK_EXIT_CODE"
}
trap capture_gate_exit EXIT
export BSK_RL_RAY_OBJECT_STORE_BYTES=8589934592
export BSK_MEMORY_ALLOCATION_BYTES=214748364800
export BSK_RL_MEMORY_PROFILE_INTERVAL_S=20
# Require the actual Slurm memory cgroup, including tighter job ancestors.
"$BSK_RL_PYTHON" - "$BSK_OUTPUT" <<'PYMEM'
import json,os,pathlib,sys
from examples.multiagent_imaging.memory_profile import read_cgroup_memory
records=read_cgroup_memory()
limits=[]
for row in records:
 for name in ('memory.max','memory.limit_in_bytes'):
  value=row['values'].get(name,'max')
  if value.isdigit() and 0<int(value)<2**60:limits.append(int(value))
actual=min(limits) if limits else None
expected=int(os.environ['BSK_MEMORY_ALLOCATION_BYTES'])
record=dict(cgroups=records,effective_limit_bytes=actual,expected_bytes=expected,passed=actual==expected)
with (pathlib.Path(sys.argv[1])/'memory-allocation-profile.json').open('x') as f:json.dump(record,f,indent=2)
if actual!=expected:raise RuntimeError('Missing/mismatched effective job memory cgroup; stop before mission or Ray')
PYMEM
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.catalog_communication_canary --output "$BSK_OUTPUT/communication-canary"
"$BSK_RL_PYTHON" -m pytest -q tests/unittest/multiagent/test_study_protocol.py tests/unittest/multiagent/test_study_metrics.py tests/unittest/multiagent/test_data_boundaries.py tests/unittest/multiagent/test_async_discounting.py tests/unittest/multiagent/test_gate_repairs.py tests/unittest/multiagent/test_checkpoint_topology.py > "$BSK_OUTPUT/focused-tests.log" 2>&1
for BSK_ARM in catalog_tx geometry_only; do
  BSK_CONFIG="examples/multiagent_imaging/configs/walker4_t200_cd1_${BSK_ARM}.json"
  BSK_LABEL=with_communication; if [[ "$BSK_ARM" == geometry_only ]]; then BSK_LABEL=without_communication; fi
  export BSK_RL_WANDB_NAME="${BSK_LABEL}_s3105_readiness_${SLURM_JOB_ID}"
  study_run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.train --config "$BSK_CONFIG" --output "$BSK_OUTPUT/$BSK_ARM" --iterations 1 --train-batch-size 4200 --num-env-runners 28 --num-learners 0 --ray-cpus "$SLURM_CPUS_PER_TASK" --torch-threads 1 --sample-timeout-s 3600 --ppo-profile amos_best --save-initial --capture-learner-batch --max-training-episodes 32 --max-sampled-true-decisions 60000 --wandb-mode online --require-wandb
  "$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_checkpoint_bytes --checkpoint "$BSK_OUTPUT/$BSK_ARM/checkpoint_0001" --output "$BSK_OUTPUT/$BSK_ARM/checkpoint-before.json"
  study_run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.train --config "$BSK_CONFIG" --output "$BSK_OUTPUT/${BSK_ARM}-resume" --resume "$BSK_OUTPUT/$BSK_ARM/checkpoint_0001" --iterations 0 --train-batch-size 4200 --num-env-runners 28 --num-learners 0 --ray-cpus "$SLURM_CPUS_PER_TASK" --torch-threads 1 --sample-timeout-s 3600 --ppo-profile amos_best --wandb-mode online --require-wandb
  "$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_checkpoint_bytes --checkpoint "$BSK_OUTPUT/$BSK_ARM/checkpoint_0001" --compare "$BSK_OUTPUT/$BSK_ARM/checkpoint-before.json" --output "$BSK_OUTPUT/$BSK_ARM/checkpoint-after.json"
  "$BSK_RL_PYTHON" -m examples.multiagent_imaging.compare_learned_checkpoints --config "$BSK_CONFIG" --checkpoint "initial=$BSK_OUTPUT/$BSK_ARM/checkpoint_0000" --checkpoint "trained=$BSK_OUTPUT/$BSK_ARM/checkpoint_0001" --seeds 401 402 --output "$BSK_OUTPUT/$BSK_ARM/heldout" --workers 8
done
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_gate --stage verify --output "$BSK_OUTPUT" --parallel
