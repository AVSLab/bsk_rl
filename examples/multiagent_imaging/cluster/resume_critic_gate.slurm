#!/usr/bin/env bash
#SBATCH --job-name=bsk-resume-critic
#SBATCH --account=ucb550_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${BSK_PROJECT_ROOT:?New immutable source required}"
: "${BSK_RL_PYTHON:?}"
: "${BSK_RUNTIME_EXTRAS:?}"
: "${BASILISK_SOURCE_ROOT:?}"
: "${BSK_SUPPORT_DATA_CACHE:?}"
: "${BSK_RL_WANDB_KEY_PATH:?}"
: "${BSK_GATE_INPUT:?}"
: "${BSK_OUTPUT:?Fresh output required}"
cd "$BSK_PROJECT_ROOT"
test ! -e "$BSK_OUTPUT"
mkdir "$BSK_OUTPUT"
export BSK_REQUIRE_EXTERNAL_PSUTIL=1
export PYTHONPATH="$BSK_PROJECT_ROOT/examples/multiagent_imaging/cluster/job_bootstrap:$BSK_PROJECT_ROOT/src:$BSK_PROJECT_ROOT:$BSK_RUNTIME_EXTRAS"
export LD_LIBRARY_PATH="/curc/sw/install/gcc/14.2.0/lib64:/curc/sw/install/gcc/14.2.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
gate_exit() {
    BSK_GATE_EXIT=$?
    trap - EXIT
    export BSK_GATE_EXIT
    "$BSK_RL_PYTHON" - "$BSK_OUTPUT" "$BSK_GATE_INPUT" <<'PYEXIT'
import json,pathlib,sys,os,time
from examples.multiagent_imaging.cluster.resume_critic_gate import input_hashes
out=pathlib.Path(sys.argv[1]); saved=out/'original-input-hashes.json'
record=dict(exit_code=int(os.environ['BSK_GATE_EXIT']),ended_at_unix=time.time(),original_inputs_verified=False)
if saved.exists():
    record['original_inputs_verified']=input_hashes(pathlib.Path(sys.argv[2]))==json.loads(saved.read_text())
(out/'job-exit.json').write_text(json.dumps(record,indent=2)+'\n')
if saved.exists() and not record['original_inputs_verified']:
    raise SystemExit('Original checkpoint/batch bytes changed')
PYEXIT
    exit "$BSK_GATE_EXIT"
}
trap gate_exit EXIT
export BSK_RL_RAY_TMPDIR="/tmp/bsk-gate-${SLURM_JOB_ID}"
mkdir "$BSK_RL_RAY_TMPDIR"
scontrol show job "$SLURM_JOB_ID" -o > "$BSK_OUTPUT/allocation.txt"
"$BSK_RL_PYTHON" - "$BSK_OUTPUT" <<'PYALLOC'
import json,pathlib,re,sys
out=pathlib.Path(sys.argv[1]); text=(out/'allocation.txt').read_text()
cpus=int(re.search(r'NumCPUs=(\d+)',text)[1]); wall=re.search(r'TimeLimit=(\S+)',text)[1]
days,sep,clock=wall.rpartition('-'); clock=clock if sep else wall
parts=list(map(int,clock.split(':')))
seconds=(int(days)*86400 if sep else 0)+sum(n*x for n,x in zip(parts,[3600,60,1]))
record=dict(allocated_cpus=cpus,time_limit_s=seconds,reserved_cpu_hours=cpus*seconds/3600,cap_cpu_hours=3)
(out/'allocation-gate.json').write_text(json.dumps(record,indent=2)+'\n')
assert seconds<=900 and record['reserved_cpu_hours']<=3, 'Effective reservation exceeds authorization'
PYALLOC
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.resume_critic_gate --stage preflight --input-root "$BSK_GATE_INPUT" --output "$BSK_OUTPUT"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.prepare_support_data --output "$BSK_OUTPUT/support-data.json"
"$BSK_RL_PYTHON" - "$BSK_OUTPUT" <<'PYSUPPORT'
import json,pathlib,sys
root=pathlib.Path.cwd(); actual=json.loads((pathlib.Path(sys.argv[1])/'support-data.json').read_text()); expected=json.loads((root/'examples/multiagent_imaging/cluster/gate-parent-support.manifest').read_text())
assert actual==expected, 'Physical support-data differs from parent diagnostic'
PYSUPPORT
"$BSK_RL_PYTHON" -m pytest -q tests/unittest/multiagent/test_gate_repairs.py tests/unittest/multiagent/test_data_boundaries.py tests/unittest/multiagent/test_completion_policy_mask.py tests/unittest/multiagent/test_async_discounting.py > "$BSK_OUTPUT/focused-tests.log" 2>&1
BSK_CONFIG="examples/multiagent_imaging/configs/walker4_mixed_amos_best_conflict.json"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.train --config "$BSK_CONFIG" --output "$BSK_OUTPUT/resume" --resume "$BSK_GATE_INPUT/checkpoint_0003" --iterations 0 --train-batch-size 4200 --num-env-runners 1 --num-learners 0 --ray-cpus "$SLURM_CPUS_PER_TASK" --torch-threads 1 --sample-timeout-s 3600 --ppo-profile amos_best --wandb-mode online --require-wandb
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.fixed_batch_diagnostic --config "$BSK_CONFIG" --input-root "$BSK_GATE_INPUT" --output "$BSK_OUTPUT/critic" --checkpoint "$BSK_GATE_INPUT/checkpoint_0000" --steps 5000 --max-wall-seconds 600 --require-fit
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.resume_critic_gate --stage verify --input-root "$BSK_GATE_INPUT" --output "$BSK_OUTPUT"
