#!/usr/bin/env bash
# Validate the two-task K=20 stress gate before launching the full campaign.
set -euo pipefail

JOB_ID=${1:?Usage: $0 STRESS_ARRAY_JOB_ID}
if [[ ! "$JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Stress array job ID must be numeric: $JOB_ID" >&2
    exit 2
fi

REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
cd "$REPO_DIR"
OUTPUT_ROOT="/scratch/alpine/$USER/prospectus_rfi/memorysafe_100_200_v2_stress/$JOB_ID"
LOG_ROOT="/scratch/alpine/$USER/job_output"

echo "===== scheduler states ====="
sacct -X -j "$JOB_ID" --format=JobIDRaw,JobName,State,ExitCode,Elapsed,MaxRSS,MaxVMSize

TASK_STATES=$(sacct -X -n -P -j "$JOB_ID" --format=JobIDRaw,State | awk -F'|' '$1 ~ /_[01]$/ {print $2}')
if [[ "$(wc -l <<< "$TASK_STATES" | tr -d ' ')" != "2" ]]; then
    echo "Expected two stress task states; the array may still be running" >&2
    exit 3
fi
while IFS= read -r STATE; do
    if [[ "$STATE" != "COMPLETED" ]]; then
        echo "Stress gate failed: task state is $STATE" >&2
        exit 4
    fi
done <<< "$TASK_STATES"

FAIL_PATTERN='OutOfMemory|out of memory|OOM|worker (died|was killed)|actor.*unavailable|SYSTEM_ERROR|Ray.*killed'
for INDEX in 0 1; do
    LOG="$LOG_ROOT/rfi_memstress_${JOB_ID}_${INDEX}.out"
    if [[ ! -s "$LOG" ]]; then
        echo "Missing stress log: $LOG" >&2
        exit 5
    fi
    if grep -Eiq "$FAIL_PATTERN" "$LOG"; then
        echo "Stress gate failed: memory/worker failure in $LOG" >&2
        grep -Ein "$FAIL_PATTERN" "$LOG" | tail -20 >&2
        exit 6
    fi
done

PYTHON=${BSK_RL_VENV_ROOT:-/projects/$USER/.venv}/bin/python
"$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "training"
for run_name in ("mlp_k20_seed99001", "attention_k20_seed99001"):
    run = root / run_name
    status_path = run / "status.json"
    final_checkpoint = run / "checkpoints" / "final"
    if not status_path.is_file() or not final_checkpoint.is_dir():
        raise SystemExit(f"missing status/final checkpoint for {run}")
    status = json.loads(status_path.read_text())
    if status.get("state") == "failed" or int(status.get("environment_steps", 0)) <= 0:
        raise SystemExit(f"invalid stress status for {run_name}: {status}")
    print(
        f"{run_name}: state={status['state']} iterations={status['training_iteration']} "
        f"steps={status['environment_steps']} wall_s={status['wall_clock_s']:.1f}"
    )
PY

echo "PASS: both N<=200, K=20 stress jobs completed with no detected OOM/worker failure."
