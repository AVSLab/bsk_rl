#!/usr/bin/env bash
set -euo pipefail

host="${AMOS_CLUSTER_HOST:-dahu1128@login-ci5.rc.colorado.edu}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
report="$repo_root/docs/amos_2026/cluster_campaign_audit_20260729.txt"

ssh "$host" 'bash -s' <<'REMOTE' | tee "$report"
set -u

roots=(
  "/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_all12_LEO_100targets_45000s_20260729T203236Z"
  "/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_all12_mixed_100targets_45000s_20260729T203255Z"
  "/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_45000s_HIO5_SHIO3_20260729T232546Z"
)

echo "AMOS 2026 live campaign audit"
date -Is
echo

echo "Queue"
squeue -h -u "$USER" -o '%A %a %T %j' || true
echo

echo "Accounting"
sacct -X -j \
30562459,30562460,30562461,30562462,30562463,30562464,30562465,30562466,30562467,30562468,\
30562505,30562506,30562507,30562508,30562509,30562510,30562511,30562512,30562513,30562514,\
30568059,30568061,30568062,30568063,30568064,30568065,30568066,30568067,30568068,30568069,30568070 \
--format=JobIDRaw,JobName%32,State,ExitCode,Elapsed -n -P 2>/dev/null || true
echo

python3 - "${roots[@]}" <<'PY'
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def status_label(payload):
    for key in ("status", "state", "result"):
        value = payload.get(key)
        if isinstance(value, str):
            return value.lower()
    for key in ("success", "completed", "ok"):
        value = payload.get(key)
        if isinstance(value, bool):
            return "completed" if value else "failed"
    return "unknown"


for root_arg in sys.argv[1:]:
    root = Path(root_arg)
    print(f"ROOT={root}")
    if not root.is_dir():
        print("  exists=no")
        print()
        continue

    print("  exists=yes")
    statuses = list(root.rglob("mc_status.json"))
    labels = Counter()
    policy_seeds = defaultdict(set)
    unreadable = 0
    for path in statuses:
        try:
            payload = json.loads(path.read_text())
        except Exception:
            unreadable += 1
            continue
        labels[status_label(payload)] += 1
        policy = payload.get("policy_tag") or payload.get("policy")
        seed = payload.get("seed")
        if policy is not None and seed is not None:
            policy_seeds[str(policy)].add(int(seed))

    print(f"  mc_status_files={len(statuses)}")
    print(f"  status_labels={dict(sorted(labels.items()))}")
    print(f"  unreadable_status_files={unreadable}")
    if policy_seeds:
        print(
            "  seeds_by_policy="
            + ",".join(
                f"{policy}:{len(seeds)}"
                for policy, seeds in sorted(policy_seeds.items())
            )
        )

    for name in (
        "steps.csv",
        "images.csv",
        "downlinks.csv",
        "metrics_*.json",
        "summary_by_policy.csv",
        "per_run.csv",
        "missing_runs.csv",
        "failed_runs.csv",
        "analysis_report.json",
        "detailed_summary_by_policy.csv",
        "detailed_per_run.csv",
    ):
        print(f"  {name}={sum(1 for _ in root.rglob(name))}")
    print()
PY

echo "Disk usage"
for root in "${roots[@]}"; do
  if [[ -d "$root" ]]; then
    du -sh "$root"
  fi
done
echo

echo "Mixed-fixed training directories"
find "/scratch/alpine/$USER/rllib_results" -maxdepth 1 -type d \
  -name 'amos2026_MIXED_GAT_fullActions_*MixedFixed50LEO30MEO20GEO100Targets*' \
  -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort || true
REMOTE

echo
echo "Saved audit report to: $report"
