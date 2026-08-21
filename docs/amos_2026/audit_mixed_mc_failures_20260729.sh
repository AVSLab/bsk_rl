#!/usr/bin/env bash
set -euo pipefail

host="${AMOS_CLUSTER_HOST:-dahu1128@login-ci5.rc.colorado.edu}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
report="$repo_root/docs/amos_2026/cluster_mixed_failure_audit_20260729.txt"

ssh "$host" 'bash -s' <<'REMOTE' | tee "$report"
set -u

weighted_root="/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_all12_mixed_100targets_45000s_20260729T203255Z"
exact_root="/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_45000s_HIO5_SHIO3_20260729T232546Z"

python3 - "$weighted_root" "$exact_root" <<'PY'
import json
import sys
from pathlib import Path

for root_arg in sys.argv[1:]:
    root = Path(root_arg)
    failed = []
    for status_path in root.rglob("mc_status.json"):
        try:
            status = json.loads(status_path.read_text())
        except Exception:
            continue
        state = status.get("state") or status.get("status")
        if str(state or "").lower() == "failed":
            failed.append((status_path, status))
    print(f"ROOT={root}")
    print(f"failed_status_count={len(failed)}")
    if failed:
        status_path, status = sorted(failed, key=lambda item: str(item[0]))[0]
        print(f"sample_status={status_path}")
        print(json.dumps(status, indent=2, sort_keys=True))
    print()
PY

echo "Recent weighted-mix Slurm logs"
find "/scratch/alpine/$USER/job_output" -maxdepth 1 -type f \
  -name 'gat_mc_miss_mix_100t_*.out' -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr | head -n 2 | cut -d' ' -f2- \
  | while IFS= read -r log; do
      echo "===== $log ====="
      tail -n 120 "$log"
    done

echo
echo "Recent exact-mix Slurm logs"
find "/scratch/alpine/$USER/job_output" -maxdepth 1 -type f \
  -name 'gat_mc_mix_100t_*.out' -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr | head -n 4 | cut -d' ' -f2- \
  | while IFS= read -r log; do
      echo "===== $log ====="
      tail -n 120 "$log"
    done
REMOTE

echo
echo "Saved failure audit to: $report"
