#!/usr/bin/env bash
# PROPOSED ONLY. Separate approval and a new verified immutable release required.
# No training/sample/evaluation/optimizer/checkpoint operation; never submits.
#SBATCH --job-name=multiagent-fresh-policy-startup
#SBATCH --account=ucb550_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=200G
#SBATCH --time=00:15:00
set -euo pipefail
: "${BSK_STARTUP_ONLY_APPROVED:?Separate startup-only approval required}"
test "$BSK_STARTUP_ONLY_APPROVED" = 1
export BSK_RESERVATION_CAP=14 BSK_EXPECTED_CPUS=56 BSK_EXPECTED_WALL_S=900 BSK_EXPECTED_MEMORY_MIB=204800
source "${BSK_PROJECT_ROOT:?}/examples/multiagent_imaging/cluster/study_common.sh"
# Preserve bounded Ray logs on success or failure, then retain final exit cost.
startup_exit() {
  export BSK_EXIT_CODE=$?
  trap - EXIT
  "$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.capture_ray_evidence \
    --root "$BSK_RL_RAY_TMPDIR" --output "$BSK_OUTPUT/ray-evidence" || true
  "$BSK_RL_PYTHON" - "$BSK_OUTPUT" <<'COST'
import json,os,pathlib,sys,time
elapsed=time.time()-int(os.environ['BSK_JOB_START_UNIX'])
record=dict(exit_code=int(os.environ['BSK_EXIT_CODE']),elapsed_wall_s=elapsed,allocated_cpus=int(os.environ['BSK_ALLOCATED_CPUS']),actual_allocated_cpu_h=int(os.environ['BSK_ALLOCATED_CPUS'])*elapsed/3600,accounting_semantics='Provisional; reconcile final sacct')
with (pathlib.Path(sys.argv[1])/'job-exit.json').open('x') as f:json.dump(record,f,indent=2)
COST
  exit "$BSK_EXIT_CODE"
}
trap startup_exit EXIT
export BSK_RL_RAY_OBJECT_STORE_BYTES=8589934592 BSK_MEMORY_ALLOCATION_BYTES=214748364800
export BSK_RL_MEMORY_PROFILE_INTERVAL_S=20
"$BSK_RL_PYTHON" - "$BSK_OUTPUT" <<'PY'
import json,pathlib,sys
from examples.multiagent_imaging.memory_profile import read_cgroup_memory
records=read_cgroup_memory();limits=[]
for row in records:
 for name in ('memory.max','memory.limit_in_bytes'):
  value=row['values'].get(name,'max')
  if value.isdigit() and 0<int(value)<2**60:limits.append(int(value))
actual=min(limits) if limits else None
record=dict(cgroups=records,effective_limit_bytes=actual,expected_bytes=214748364800,passed=actual==214748364800)
with (pathlib.Path(sys.argv[1])/'memory-allocation-profile.json').open('x') as f:json.dump(record,f,indent=2)
if not record['passed']:raise RuntimeError('Missing/mismatched effective 200 GiB cgroup')
PY
for BSK_ARM in catalog_tx geometry_only; do
  study_run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.startup_only \
    --config "examples/multiagent_imaging/configs/walker4_t200_cd1_${BSK_ARM}.json" \
    --output "$BSK_OUTPUT/$BSK_ARM"
done
"$BSK_RL_PYTHON" - "$BSK_OUTPUT" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);records={a:json.loads((root/a/'startup-only-result.json').read_text()) for a in ('catalog_tx','geometry_only')}
digests={a:r['initial_shared_policy']['authoritative_actor_sha256'] for a,r in records.items()}
result=dict(passed=all(r['passed'] for r in records.values()) and len(set(digests.values()))==1,authoritative_actor_digests=digests,new_sampled_episodes=0,ppo_updates=0)
with (root/'matched-startup-result.json').open('x') as f:json.dump(result,f,indent=2)
if not result['passed']:raise RuntimeError('Matched fresh actor weights/startup gate failed')
PY
