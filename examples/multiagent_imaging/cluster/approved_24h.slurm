#!/usr/bin/env bash
#SBATCH --job-name=multiagent-approved-24h
#SBATCH --account=ucb550_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --no-requeue
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@1800
# User approved direct 24h training with inspection after 20 running minutes.
# Startup, first real update, and full restore gates run inside this allocation.
set -euo pipefail
test "${BSK_DIRECT_24H_APPROVED:?Explicit direct long-launch approval required}" = 1
case "${BSK_ARM:?Choose the information variant}" in catalog_tx) BSK_LABEL=with_communication;; geometry_only) BSK_LABEL=without_communication;; *) exit 2;; esac
# Corrective attempts use the reviewed remaining allocation time, in whole minutes.
# This is an administrative budget; mission and PPO settings remain unchanged.
export BSK_APPROVED_WALL_S="${BSK_APPROVED_WALL_S:-86400}"
[[ "$BSK_APPROVED_WALL_S" =~ ^[0-9]+$ ]]
test "$BSK_APPROVED_WALL_S" -ge 10800
test "$BSK_APPROVED_WALL_S" -le 86400
export SLURM_EXPORT_ENV=ALL
export BSK_RESERVATION_CAP=1344 BSK_EXPECTED_CPUS=56 BSK_EXPECTED_WALL_S="$BSK_APPROVED_WALL_S" BSK_EXPECTED_MEMORY_MIB=204800
source "${BSK_PROJECT_ROOT:?}/examples/multiagent_imaging/cluster/study_common.sh"
long_exit() {
  export BSK_EXIT_CODE=$?
  trap - EXIT
  "$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.capture_ray_evidence --root "$BSK_RL_RAY_TMPDIR" --output "$BSK_OUTPUT/ray-evidence" || true
  "$BSK_RL_PYTHON" - "$BSK_OUTPUT" <<'COST'
import json,os,pathlib,sys,time
elapsed=time.time()-int(os.environ['BSK_JOB_START_UNIX'])
with (pathlib.Path(sys.argv[1])/'job-exit.json').open('x') as f:
 json.dump(dict(exit_code=int(os.environ['BSK_EXIT_CODE']),elapsed_wall_s=elapsed,allocated_cpus=int(os.environ['BSK_ALLOCATED_CPUS']),actual_allocated_cpu_h=int(os.environ['BSK_ALLOCATED_CPUS'])*elapsed/3600,accounting_semantics='Provisional; reconcile final sacct'),f,indent=2)
COST
  exit "$BSK_EXIT_CODE"
}
trap long_exit EXIT
export BSK_RL_RAY_OBJECT_STORE_BYTES=8589934592 BSK_MEMORY_ALLOCATION_BYTES=214748364800 BSK_RL_MEMORY_PROFILE_INTERVAL_S=20
"$BSK_RL_PYTHON" - "$BSK_OUTPUT" <<'MEM'
import json,pathlib,sys
from examples.multiagent_imaging.memory_profile import read_cgroup_memory
records=read_cgroup_memory(); limits=[]
for row in records:
 for name in ('memory.max','memory.limit_in_bytes'):
  value=row['values'].get(name,'max')
  if value.isdigit() and 0<int(value)<2**60: limits.append(int(value))
actual=min(limits) if limits else None
record=dict(cgroups=records,effective_limit_bytes=actual,expected_bytes=214748364800,passed=actual==214748364800)
with (pathlib.Path(sys.argv[1])/'memory-allocation-profile.json').open('x') as f: json.dump(record,f,indent=2)
if not record['passed']: raise RuntimeError('Effective 200 GiB cgroup required before native/Ray work')
MEM
BSK_CONFIG="examples/multiagent_imaging/configs/walker4_t200_cd1_${BSK_ARM}.json"
export BSK_RL_WANDB_NAME="${BSK_LABEL}_s3105_24h_${SLURM_JOB_ID}"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.catalog_communication_canary --output "$BSK_OUTPUT/communication-canary"
study_run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.train --config "$BSK_CONFIG" --output "$BSK_OUTPUT/bootstrap" --iterations 1 --train-batch-size 4200 --num-env-runners 28 --num-learners 0 --ray-cpus "$SLURM_CPUS_PER_TASK" --torch-threads 1 --sample-timeout-s 3600 --ppo-profile amos_best --save-initial --capture-learner-batch --max-training-episodes 56 --max-sampled-true-decisions 240000 --wandb-mode online --require-wandb
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.approved_long_evidence --root "$BSK_OUTPUT" --phase bootstrap --config "$BSK_CONFIG"
BSK_CHECKPOINT="$BSK_OUTPUT/bootstrap/checkpoint_0001"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_checkpoint_bytes --checkpoint "$BSK_CHECKPOINT" --output "$BSK_OUTPUT/bootstrap-checkpoint-before.json"
study_run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.train --config "$BSK_CONFIG" --output "$BSK_OUTPUT/bootstrap-resume" --resume "$BSK_CHECKPOINT" --iterations 0 --train-batch-size 4200 --num-env-runners 28 --num-learners 0 --ray-cpus "$SLURM_CPUS_PER_TASK" --torch-threads 1 --sample-timeout-s 3600 --ppo-profile amos_best --wandb-mode online --require-wandb
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_checkpoint_bytes --checkpoint "$BSK_CHECKPOINT" --compare "$BSK_OUTPUT/bootstrap-checkpoint-before.json" --output "$BSK_OUTPUT/bootstrap-checkpoint-after.json"
# Reserve two hours for update overshoot, final restore and eight held-out cases.
# This is an administrative wall budget, not a predicted experience milestone.
BSK_REMAINING="$("$BSK_RL_PYTHON" -c 'import os,time; print(max(0,int(os.environ["BSK_JOB_START_UNIX"])+int(os.environ["BSK_APPROVED_WALL_S"])-7200-int(time.time())))')"
test "$BSK_REMAINING" -ge 3600
study_run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.train --config "$BSK_CONFIG" --output "$BSK_OUTPUT/training" --resume "$BSK_CHECKPOINT" --iterations 10000 --train-batch-size 4200 --num-env-runners 28 --num-learners 0 --ray-cpus "$SLURM_CPUS_PER_TASK" --torch-threads 1 --sample-timeout-s 3600 --ppo-profile amos_best --max-wall-seconds "$BSK_REMAINING" --wandb-mode online --require-wandb
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.approved_long_evidence --root "$BSK_OUTPUT" --phase training --config "$BSK_CONFIG"
BSK_FINAL="$("$BSK_RL_PYTHON" -c 'import json,os; print(json.load(open(os.environ["BSK_OUTPUT"]+"/training/tranche-stop.json"))["checkpoint"])')"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_checkpoint_bytes --checkpoint "$BSK_FINAL" --output "$BSK_OUTPUT/final-checkpoint-before.json"
study_run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.train --config "$BSK_CONFIG" --output "$BSK_OUTPUT/final-resume" --resume "$BSK_FINAL" --iterations 0 --train-batch-size 4200 --num-env-runners 28 --num-learners 0 --ray-cpus "$SLURM_CPUS_PER_TASK" --torch-threads 1 --sample-timeout-s 3600 --ppo-profile amos_best --wandb-mode online --require-wandb
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_checkpoint_bytes --checkpoint "$BSK_FINAL" --compare "$BSK_OUTPUT/final-checkpoint-before.json" --output "$BSK_OUTPUT/final-checkpoint-after.json"
study_run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.compare_learned_checkpoints --config "$BSK_CONFIG" --checkpoint "initial=$BSK_OUTPUT/bootstrap/checkpoint_0000" --checkpoint "trained=$BSK_FINAL" --seeds 401 402 --output "$BSK_OUTPUT/heldout" --workers 8
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.approved_long_evidence --root "$BSK_OUTPUT" --phase heldout --config "$BSK_CONFIG"
# No automatic successor/requeue. Review final evidence before any continuation.
