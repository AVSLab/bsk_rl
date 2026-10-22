#!/usr/bin/env bash
# PREPARATION ONLY. Submit only after review of terminal qualification and budget.
# Supply account explicitly: sbatch --account=ucb550_asc3 ...
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SLURM_JOB_ID:?A separately approved compute allocation is required}"
: "${BSK_PROJECT_ROOT:?Fresh immutable reviewed EVALUATION source checkout required}"
: "${BSK_RL_WANDB_KEY_PATH:?Existing private W&B credential file path required}"
RELEASE="${1:?reviewed evaluation release manifest}"
PLAN="${2:?frozen A/B/C/D plan}"
QUALIFICATION="${3:?independent terminal pair qualification}"
OUTPUT="${4:?fresh evaluation output directory}"
export BSK_RL_PYTHON=/projects/dahu1128/.venv-completion-v2/bin/python
export BASILISK_SOURCE_ROOT=/projects/dahu1128/basilisk-completion-v2
export BSK_SUPPORT_DATA_CACHE="$BASILISK_SOURCE_ROOT/.support-data-cache"
export BSK_REQUIRE_EXTERNAL_PSUTIL=1
export PYTHONPATH="$BSK_PROJECT_ROOT/examples/multiagent_imaging/cluster/job_bootstrap:$BSK_PROJECT_ROOT/src:$BSK_PROJECT_ROOT:/projects/dahu1128/gnc-learning-runtime-extras-20261005"
export LD_LIBRARY_PATH="/curc/sw/install/gcc/14.2.0/lib64:/curc/sw/install/gcc/14.2.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 BSK_RL_TARGET_RECORDERS=0
export BSK_RL_WANDB_PROJECT=multi-agent-space-imaging
export BSK_RL_WANDB_ENTITY=daniel-hutererprats-university-of-colorado-boulder
export BSK_RL_WANDB_GROUP=cooperative-v1-c18-frozen-abcd-v1
export BSK_RL_WANDB_NAME="heldout_${SLURM_JOB_ID}"
export BSK_RL_MEMORY_PROFILE_INTERVAL_S=20
export BSK_ALLOCATED_CPUS=16
cd "$BSK_PROJECT_ROOT"
test ! -e "$OUTPUT"
mkdir -p "$OUTPUT"
scontrol show job "$SLURM_JOB_ID" -o > "$OUTPUT/allocation.txt"
finish() {
    status=$?
    trap - EXIT
    "$BSK_RL_PYTHON" - "$OUTPUT" "$status" <<'PY'
import json,pathlib,sys
(pathlib.Path(sys.argv[1])/'job-exit.json').write_text(json.dumps(dict(exit_code=int(sys.argv[2]),
    new_PPO_updates=0,final_sacct_required=True))+'\n')
PY
    sacct -nP -j "$SLURM_JOB_ID" --format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,MaxRSS > "$OUTPUT/sacct-at-exit.txt" || true
    exit "$status"
}
trap finish EXIT
"$BSK_RL_PYTHON" - "$OUTPUT" "$RELEASE" "$PLAN" "$QUALIFICATION" <<'PY'
import json,os,pathlib,re,sys
from examples.multiagent_imaging.cluster.study_gate import preflight
from examples.multiagent_imaging.cluster.cooperative_launch import validate_release
from examples.multiagent_imaging.frozen_evaluation import qualify,reserved_seed_check
out=pathlib.Path(sys.argv[1]); allocation=(out/'allocation.txt').read_text()
for pattern in (r'Account=ucb550_asc3(?:\s|$)',r'Partition=acpu(?:\s|$)',r'QOS=cpu-normal(?:\s|$)',
                r'NumCPUs=16(?:\s|$)',r'Requeue=0(?:\s|$)',r'TimeLimit=02:00:00(?:\s|$)'):
    assert re.search(pattern,allocation),pattern
assert int(os.environ['SLURM_CPUS_PER_TASK'])==16 and int(os.environ['SLURM_MEM_PER_NODE'])==65536
preflight(out/'preflight',source_validator=lambda:validate_release(sys.argv[2]))
qualify(json.loads(pathlib.Path(sys.argv[3]).read_text()),json.loads(pathlib.Path(sys.argv[4]).read_text()),artifact_root=pathlib.Path(sys.argv[4]).resolve().parent)
(out/'seed-preflight.json').write_text(json.dumps(reserved_seed_check(),indent=2)+'\n')
PY
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.frozen_evaluation run \
    --plan "$PLAN" --qualification "$QUALIFICATION" --output "$OUTPUT/heldout"
"$BSK_RL_PYTHON" - "$OUTPUT" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]);groups=json.loads((out/'post-evaluation-memory.json').read_text())['cgroups']
finite=[int(g['values'][k]) for g in groups for k in ('memory.max','memory.limit_in_bytes')
        if g['values'].get(k,'').isdigit() and 0<int(g['values'][k])<2**60]
assert finite and min(finite)==64*1024**3
assert all(int(g['values'].get('memory.failcnt','0'))==0 for g in groups)
PY
# No submission, continuation, optimization or dependency installation here.
