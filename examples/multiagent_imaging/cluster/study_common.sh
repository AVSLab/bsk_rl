#!/usr/bin/env bash
# Sourced inside a later explicitly approved allocation. Never submits a job.
set -euo pipefail
: "${SLURM_JOB_ID:?Compute allocation required}"
: "${BSK_PROJECT_ROOT:?New immutable reviewed deployment required}"
: "${BSK_OUTPUT:?Fresh output directory required}"
: "${BSK_RL_WANDB_KEY_PATH:?Existing owner-controlled private file path required}"
export BSK_RL_PYTHON=/projects/dahu1128/.venv-completion-v2/bin/python
# Historical installed runtime location, retained for compatibility.
export BSK_RUNTIME_EXTRAS=/projects/dahu1128/gnc-learning-runtime-extras-20261005
export BASILISK_SOURCE_ROOT=/projects/dahu1128/basilisk-completion-v2
export BSK_SUPPORT_DATA_CACHE=/projects/dahu1128/basilisk-completion-v2/.support-data-cache
export BSK_REQUIRE_EXTERNAL_PSUTIL=1
export PYTHONPATH="$BSK_PROJECT_ROOT/examples/multiagent_imaging/cluster/job_bootstrap:$BSK_PROJECT_ROOT/src:$BSK_PROJECT_ROOT:$BSK_RUNTIME_EXTRAS"
export LD_LIBRARY_PATH="/curc/sw/install/gcc/14.2.0/lib64:/curc/sw/install/gcc/14.2.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
export BSK_RL_RAY_TMPDIR="/tmp/bsk-study-${SLURM_JOB_ID}"
export BSK_RL_WANDB_PROJECT="${BSK_RL_WANDB_PROJECT:-multi-agent-space-imaging}"
export BSK_RL_WANDB_ENTITY=daniel-hutererprats-university-of-colorado-boulder
cd "$BSK_PROJECT_ROOT"
test ! -e "$BSK_OUTPUT"
mkdir "$BSK_OUTPUT" "$BSK_RL_RAY_TMPDIR"
scontrol show job "$SLURM_JOB_ID" -o > "$BSK_OUTPUT/allocation.txt"
export BSK_JOB_START_UNIX="$(date +%s)"
"$BSK_RL_PYTHON" - "$BSK_OUTPUT" "$BSK_RESERVATION_CAP" <<'PY'
import json,pathlib,re,sys,os
from examples.multiagent_imaging.cluster.study_gate import allocation
out=pathlib.Path(sys.argv[1]);raw=(out/'allocation.txt').read_text()
cpus=int(re.search(r'NumCPUs=(\d+)',raw)[1]);wall=re.search(r'TimeLimit=(\S+)',raw)[1]
days,sep,clock=wall.rpartition('-');clock=clock if sep else wall
parts=list(map(int,clock.split(':')));seconds=(int(days)*86400 if sep else 0)+sum(n*x for n,x in zip(parts,[3600,60,1]))
record=allocation(cpus,seconds,float(sys.argv[2]));(out/'allocation-gate.json').write_text(json.dumps(record,indent=2)+'\n')
for key,value in [('BSK_EXPECTED_CPUS',cpus),('BSK_EXPECTED_WALL_S',seconds),('BSK_EXPECTED_MEMORY_MIB',int(os.environ.get('SLURM_MEM_PER_NODE','0')))]:
 if key in os.environ and value!=int(os.environ[key]): raise ValueError('Allocation differs from reviewed '+key)
if 'Requeue=0' not in raw: raise ValueError('Automatic requeue must be disabled')
(out/'allocated-cpus.txt').write_text(str(cpus))
PY
export BSK_ALLOCATED_CPUS="$(cat "$BSK_OUTPUT/allocated-cpus.txt")"
study_exit() {
  BSK_EXIT_CODE=$?
  trap - EXIT
  export BSK_EXIT_CODE
  "$BSK_RL_PYTHON" - "$BSK_OUTPUT" <<'PY'
import json,os,pathlib,sys,time
out=pathlib.Path(sys.argv[1]);elapsed=time.time()-int(os.environ['BSK_JOB_START_UNIX'])
(out/'job-exit.json').write_text(json.dumps(dict(exit_code=int(os.environ['BSK_EXIT_CODE']),elapsed_wall_s=elapsed,
allocated_cpus=int(os.environ['BSK_ALLOCATED_CPUS']),actual_allocated_cpu_h=int(os.environ['BSK_ALLOCATED_CPUS'])*elapsed/3600,
accounting_semantics='In-job elapsed allocation cost; reconcile final sacct after job completion'),indent=2)+'\n')
PY
  exit "$BSK_EXIT_CODE"
}
trap study_exit EXIT
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_gate --stage preflight --output "$BSK_OUTPUT/preflight"
export BSK_STUDY_RELEASE_ID="$("$BSK_RL_PYTHON" -c 'import json; print(json.load(open("completion-release.json"))["release_id"])')"
export BSK_RL_WANDB_GROUP="walker4_t200_cd1_amosbest_${BSK_STUDY_RELEASE_ID}"
# Forward scheduler warning to the Python driver; finish the active update and
# checkpoint. A scheduler kill still remains a failed/partial artifact.
BSK_ACTIVE_CHILD=""
trap 'if [[ -n "$BSK_ACTIVE_CHILD" ]]; then kill -USR1 "$BSK_ACTIVE_CHILD" 2>/dev/null || true; fi' USR1 TERM
study_run() {
  "$@" &
  BSK_ACTIVE_CHILD=$!
  BSK_CHILD_STATUS=0
  while true; do
    wait "$BSK_ACTIVE_CHILD" && BSK_CHILD_STATUS=0 || BSK_CHILD_STATUS=$?
    if [[ "$BSK_CHILD_STATUS" -ge 128 ]] && kill -0 "$BSK_ACTIVE_CHILD" 2>/dev/null; then continue; fi
    break
  done
  BSK_ACTIVE_CHILD=""
  return "$BSK_CHILD_STATUS"
}
