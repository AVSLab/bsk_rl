#!/usr/bin/env bash
set -euo pipefail

host="${AMOS_CLUSTER_HOST:-dahu1128@login-ci5.rc.colorado.edu}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
report="$repo_root/docs/amos_2026/exact_mixed100_recovery_check_20260729.txt"

ssh "$host" 'bash -s' <<'REMOTE' | tee "$report"
set -u

root="/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_45000s_HIO5_SHIO3_20260729T232546Z"

echo "Queue"
squeue -h -u "$USER" -o '%A %a %T %j' \
  | grep -E 'gat_mc_mix_100t|gat_mix100_fixed' || true

echo
echo "Recovery jobs"
sacct -X -S 2026-07-29 -n -P \
  --name=gat_mc_mix_100t_s000_009,gat_mc_mix_100t_s010_019,gat_mc_mix_100t_s020_029,gat_mc_mix_100t_s030_039,gat_mc_mix_100t_s040_049,gat_mc_mix_100t_s050_059,gat_mc_mix_100t_s060_069,gat_mc_mix_100t_s070_079,gat_mc_mix_100t_s080_089,gat_mc_mix_100t_s090_099 \
  --format=JobIDRaw,JobName%32,State,ExitCode,Elapsed \
  | tail -n 60

echo
python3 - "$root" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
labels = Counter()
for path in root.rglob("mc_status.json"):
    try:
        payload = json.loads(path.read_text())
    except Exception:
        labels["unreadable"] += 1
        continue
    labels[str(payload.get("state") or payload.get("status") or "unknown").lower()] += 1

print(f"ROOT={root}")
print(f"mc_status_files={sum(labels.values())}")
print(f"status_labels={dict(sorted(labels.items()))}")
for name in ("steps.csv", "images.csv", "metrics_*.json"):
    print(f"{name}={sum(1 for _ in root.rglob(name))}")
PY

echo
echo "Latest first-block log signal"
find "/scratch/alpine/$USER/job_output" -maxdepth 1 -type f \
  -name 'gat_mc_mix_100t_s000_009_*.out' -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr | head -n 5 | cut -d' ' -f2- \
  | while IFS= read -r log; do
      echo "===== $log ====="
      grep -E 'Exact mixed target counts|Traceback|TypeError|JSONDecodeError|Run outputs|Running policy task' "$log" \
        | tail -n 20 || true
    done
REMOTE

echo
echo "Saved recovery check to: $report"
