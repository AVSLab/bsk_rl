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
sacct -X -j "$JOB_ID" --format=JobID,JobIDRaw,JobName,State,ExitCode,Elapsed

TASK_RECORDS=$(sacct -X -n -P -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed |
    awk -F'|' -v job="$JOB_ID" '$1 == job "_0" || $1 == job "_1"')
TASK_COUNT=$(awk 'NF {count++} END {print count+0}' <<< "$TASK_RECORDS")
if (( TASK_COUNT != 2 )); then
    echo "Stress array is not ready to audit; no failure is implied." >&2
    echo "Wait until both tasks leave squeue, then run this audit again." >&2
    squeue -j "$JOB_ID" -r -o "%.20i %.12j %.2t %.10M %R" >&2 || true
    exit 3
fi
FAILED_RECORDS=$(awk -F'|' '$2 != "COMPLETED"' <<< "$TASK_RECORDS")
if [[ -n "$FAILED_RECORDS" ]]; then
    echo "Stress gate failed; task records are JobID|State|ExitCode|Elapsed:" >&2
    printf '%s\n' "$TASK_RECORDS" >&2
    echo "Diagnostic log tails:" >&2
    for INDEX in 0 1; do
        LOG="$LOG_ROOT/rfi_memstress_${JOB_ID}_${INDEX}.out"
        echo "===== $LOG =====" >&2
        if [[ -s "$LOG" ]]; then
            tail -80 "$LOG" >&2
        else
            echo "missing or empty" >&2
        fi
    done
    exit 4
fi

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
