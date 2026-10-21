#!/usr/bin/env bash
# Resource/account arguments are supplied explicitly by cooperative_launch plan.
# Runs only inside a later user-submitted compute allocation. Never submits jobs.
set -euo pipefail
: "${SLURM_JOB_ID:?Compute allocation required}"
: "${BSK_PROJECT_ROOT:?Fresh immutable checkout at the reviewed commit required}"
: "${BSK_RL_WANDB_KEY_PATH:?Existing private W&B credential path required}"
PHASE="${1:?canary/train/evaluate}" RELEASE="${2:?release manifest}" CONFIG="${3:?config}"
OUTPUT="${4:?fresh output}" WALL_S="${5:?wall seconds}"
ITERATIONS="${6:?updates}" EPISODES="${7:?missions}" DECISIONS="${8:?decisions}"
shift 8
export BSK_RL_PYTHON=/projects/dahu1128/.venv-completion-v2/bin/python
export BASILISK_SOURCE_ROOT=/projects/dahu1128/basilisk-completion-v2
export BSK_SUPPORT_DATA_CACHE="$BASILISK_SOURCE_ROOT/.support-data-cache"
export BSK_REQUIRE_EXTERNAL_PSUTIL=1
export PYTHONPATH="$BSK_PROJECT_ROOT/examples/multiagent_imaging/cluster/job_bootstrap:$BSK_PROJECT_ROOT/src:$BSK_PROJECT_ROOT:/projects/dahu1128/gnc-learning-runtime-extras-20261005"
export LD_LIBRARY_PATH="/curc/sw/install/gcc/14.2.0/lib64:/curc/sw/install/gcc/14.2.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export BSK_RL_TARGET_RECORDERS=0 BSK_RL_MEMORY_PROFILE_INTERVAL_S=20
export BSK_RL_RAY_OBJECT_STORE_BYTES=8589934592 BSK_MEMORY_ALLOCATION_BYTES=214748364800
export BSK_RL_RAY_TMPDIR="/tmp/cooperative-v1-${SLURM_JOB_ID}"
export BSK_RL_WANDB_PROJECT=multi-agent-space-imaging
export BSK_RL_WANDB_ENTITY=daniel-hutererprats-university-of-colorado-boulder
GROUP_LABEL="$(basename "$CONFIG" .json)"
GROUP_LABEL="${GROUP_LABEL%_catalog_tx}"
GROUP_LABEL="${GROUP_LABEL%_geometry_only}"
export BSK_RL_WANDB_GROUP="${GROUP_LABEL}_c18opt_$("$BSK_RL_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"][:12])' "$RELEASE")"
export BSK_RL_WANDB_NAME="${PHASE}_${SLURM_JOB_ID}_$(basename "$CONFIG" .json)"
export BSK_ALLOCATED_CPUS=56
export BSK_STUDY_RELEASE_ID="$("$BSK_RL_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$RELEASE")"
cd "$BSK_PROJECT_ROOT"

: "${BSK_HPO_ENVELOPE_ROOT:?Explicit envelope path required because Slurm spools its script}"
HPO_ENVELOPE="$BSK_HPO_ENVELOPE_ROOT"
HPO_DRIVER="$HPO_ENVELOPE/train_hpo_transfer.py"
HPO_SETTINGS="$HPO_ENVELOPE/hpo-transfer.json"
: "${BSK_HPO_ENVELOPE_SHA256:?Reviewed external HPO manifest digest required}"
"$BSK_RL_PYTHON" - "$HPO_ENVELOPE" <<'VERIFY_HPO_ENVELOPE'
import hashlib,json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]);raw=(root/'envelope-manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==os.environ['BSK_HPO_ENVELOPE_SHA256']
record=json.loads(raw)
assert record['source_commit']=='b205fc1603b5ca887aaadcb3c34de2700d563c0c'
for name,digest in record['files'].items():
 assert pathlib.Path(name).name==name
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
print(json.dumps(dict(hpo_envelope_passed=True,profile=record['profile'],manifest_sha256=hashlib.sha256(raw).hexdigest())))
VERIFY_HPO_ENVELOPE

test ! -e "$OUTPUT"
mkdir "$OUTPUT" "$BSK_RL_RAY_TMPDIR"
scontrol show job "$SLURM_JOB_ID" -o > "$OUTPUT/allocation.txt"
START_UNIX="$(date +%s)"
CHILD=""
finish() {
    status=$?
    trap - EXIT
    "$BSK_RL_PYTHON" - "$OUTPUT" "$START_UNIX" "$status" <<'PY'
import json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]);elapsed=time.time()-int(sys.argv[2])
(out/'job-exit.json').write_text(json.dumps(dict(exit_code=int(sys.argv[3]),elapsed_wall_s=elapsed,
 allocated_cpus=56,provisional_allocated_cpu_hours=56*elapsed/3600,
 final_sacct_required=True),indent=2)+'\n')
PY
    sacct -nP -j "$SLURM_JOB_ID" --format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,MaxRSS > "$OUTPUT/sacct-at-exit.txt" || true
    exit "$status"
}
trap finish EXIT
trap 'if [[ -n "$CHILD" ]]; then kill -USR1 "$CHILD" 2>/dev/null || true; fi' USR1 TERM
run() {
    "$@" & CHILD=$!
    local result=0
    while true; do
        wait "$CHILD" && result=0 || result=$?
        if [[ "$result" -ge 128 ]] && kill -0 "$CHILD" 2>/dev/null; then continue; fi
        break
    done
    CHILD=""
    return "$result"
}
"$BSK_RL_PYTHON" - "$RELEASE" "$OUTPUT" "$WALL_S" <<'PY'
import json,os,pathlib,re,sys
from examples.multiagent_imaging.cluster.cooperative_launch import validate_release
from examples.multiagent_imaging.cluster.study_gate import preflight
from examples.multiagent_imaging.memory_profile import read_cgroup_memory
out=pathlib.Path(sys.argv[2]);raw=(out/'allocation.txt').read_text()
assert re.search(r'Account=ucb550_asc3(?:\s|$)',raw)
assert re.search(r'Partition=acpu(?:\s|$)',raw) and re.search(r'QOS=cpu-normal(?:\s|$)',raw)
assert re.search(r'NumCPUs=56(?:\s|$)',raw) and 'Requeue=0' in raw
assert int(os.environ['SLURM_CPUS_PER_TASK'])==56 and int(os.environ['SLURM_MEM_PER_NODE'])==204800
wall=re.search(r'TimeLimit=(\S+)',raw)[1]
days,sep,clock=wall.rpartition('-');parts=list(map(int,(clock if sep else wall).split(':')))
seconds=(int(days)*86400 if sep else 0)+sum(n*x for n,x in zip(parts,[3600,60,1]))
assert seconds==int(sys.argv[3]) and seconds<=86400
groups=read_cgroup_memory();limits=[]
for group in groups:
 for field in ('memory.max','memory.limit_in_bytes'):
  value=group['values'].get(field,'max')
  if value.isdigit() and 0<int(value)<2**60:limits.append(int(value))
assert limits and min(limits)==214748364800
(out/'allocation-gate.json').write_text(json.dumps(dict(passed=True,cgroups=groups,wall_s=seconds,allocated_cpu_hours=56*seconds/3600),indent=2)+'\n')
preflight(out/'preflight',source_validator=lambda:validate_release(sys.argv[1]))
PY

if [[ "$PHASE" == evaluate ]]; then
    CHECKPOINTS=() SEEDS=()
    for token in "$@"; do
        case "$token" in
            checkpoint=*) CHECKPOINTS+=(--checkpoint "${token#checkpoint=}");;
            seed=*) SEEDS+=("${token#seed=}");;
            *) exit 2;;
        esac
    done
    test "${#CHECKPOINTS[@]}" -gt 0
    test "${#SEEDS[@]}" -gt 0
    run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.compare_learned_checkpoints \
        --config "$CONFIG" "${CHECKPOINTS[@]}" --seeds "${SEEDS[@]}" \
        --workers 8 --output "$OUTPUT/heldout"
    run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.cooperative_eval_logging --root "$OUTPUT" --config "$CONFIG"
    exit 0
fi
[[ "$PHASE" == canary || "$PHASE" == train ]]
if [[ "$PHASE" == canary ]]; then
    test "$WALL_S" -eq 5400
    test "$ITERATIONS" -eq 2
    test "$EPISODES" -eq 56
    test "$DECISIONS" -eq 120000
    run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.catalog_communication_canary --output "$OUTPUT/native-transport"
fi
RESOURCES=(--num-env-runners 28 --num-learners 0 --ray-cpus 56 --torch-threads 1 --sample-timeout-s 3600)
LOGGING=(--wandb-mode online --require-wandb)
# Reserve 15 minutes for checkpoint and full-topology zero-update restore.
TRAIN_WALL=$((WALL_S - 900 - $(date +%s) + START_UNIX))
test "$TRAIN_WALL" -ge 300
run "$BSK_RL_PYTHON" "$HPO_DRIVER" --config "$CONFIG" \
    --output "$OUTPUT/training" --iterations "$ITERATIONS" --train-batch-size 4200 \
    "${RESOURCES[@]}" --ppo-settings "$HPO_SETTINGS" --save-initial --capture-learner-batch \
    --max-training-episodes "$EPISODES" --max-sampled-true-decisions "$DECISIONS" \
    --max-wall-seconds "$TRAIN_WALL" "${LOGGING[@]}"
CHECKPOINT="$("$BSK_RL_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$OUTPUT/training/tranche-stop.json")"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_checkpoint_bytes --checkpoint "$CHECKPOINT" --output "$OUTPUT/checkpoint-before.json"
run "$BSK_RL_PYTHON" "$HPO_DRIVER" --config "$CONFIG" \
    --output "$OUTPUT/zero-update-restore" --resume "$CHECKPOINT" --iterations 0 \
    --train-batch-size 4200 "${RESOURCES[@]}" --ppo-settings "$HPO_SETTINGS" "${LOGGING[@]}"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_checkpoint_bytes --checkpoint "$CHECKPOINT" \
    --compare "$OUTPUT/checkpoint-before.json" --output "$OUTPUT/checkpoint-after.json"
GATE=()
if [[ "$PHASE" == canary ]]; then GATE+=(--canary); fi
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.cooperative_accept --root "$OUTPUT" --config "$CONFIG" "${GATE[@]}"
# No successor, automatic continuation, dependency installation or submission.
