#!/usr/bin/env bash
set -euo pipefail

remote=${AMOS2026_CLUSTER_HOST:-dahu1128@login.rc.colorado.edu}
output=/Users/dahu1128/Downloads/AMOS2026_cluster_state_20260802.txt
socket="${TMPDIR:-/tmp}/amos2026_state_mux_$$"

cleanup() {
    ssh -S "$socket" -O exit "$remote" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Connecting to Alpine. Enter your CURC password when prompted."
ssh -M -S "$socket" -o ControlPersist=300 -fN "$remote"

ssh -S "$socket" "$remote" bash -s <<'REMOTE' | tee "$output"
set -euo pipefail

corrected_root="/scratch/alpine/$USER/amos2026_mc/corrected_density_prioritySum200_20260801T222325Z"
leo_root="$corrected_root/gat_leo_200targets_prioritySum200_45000s_HIO5_SHIO3"
mixed200_root="$corrected_root/gat_mixed_200targets_prioritySum200_45000s_HIO5_SHIO3"
mixed100_root="/scratch/alpine/$USER/amos2026_mc/gat_mixed_trained_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_45000s_HIO5_SHIO3_20260802"

echo "AMOS 2026 cluster state"
date -u +'%Y-%m-%dT%H:%M:%SZ'
echo
echo "===== Relevant queue ====="
squeue -u "$USER" --format='%.18i %.28j %.2t %.12M %.12l %.6D %R' \
    | grep -E 'JOBID|ps200_|gat_mix100_fixed|amos_mix_post|gat_mc_mix_100t' || true

echo
echo "===== Training and handoff accounting ====="
sacct -j 30568059,30688737,30703176 \
    --format=JobID,JobName,State,Elapsed,Timelimit,ExitCode -X || true

python3 - "$leo_root" "$mixed200_root" "$mixed100_root" <<'PY'
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

campaigns = {
    "corrected_leo_200": (Path(sys.argv[1]), 8, 200.0, 200),
    "corrected_mixed_200": (Path(sys.argv[2]), 8, 200.0, 200),
    "mixed_trained_mixed_100": (Path(sys.argv[3]), 8, 100.0, 100),
}

for label, (root, policy_count, expected_priority_sum, expected_targets) in campaigns.items():
    states = Counter()
    completed = defaultdict(set)
    invalid = []
    statuses = sorted(root.glob("seeds_*/*/seed_*/mc_status.json"))
    for path in statuses:
        try:
            status = json.loads(path.read_text())
        except Exception as exc:
            invalid.append(f"{path}: unreadable ({exc})")
            continue
        state = str(status.get("state", "unknown"))
        states[state] += 1
        tag = str(status.get("policy_tag", "unknown"))
        seed = int(status.get("seed", -1))
        if state == "completed" and status.get("returncode", 0) in (0, None):
            if abs(float(status.get("priority_sum", 100.0)) - expected_priority_sum) > 1e-9:
                invalid.append(f"{path}: priority_sum={status.get('priority_sum')}")
                continue
            if int(status.get("n_targets", 100)) != expected_targets:
                invalid.append(f"{path}: n_targets={status.get('n_targets')}")
                continue
            completed[tag].add(seed)
    completed_count = sum(len(seeds) for seeds in completed.values())
    expected_count = policy_count * 100
    print(f"\n===== {label} =====")
    print(f"root_exists={root.is_dir()}")
    print(f"root={root}")
    print(f"status_files={len(statuses)}")
    print(f"states={dict(sorted(states.items()))}")
    print(f"valid_completed={completed_count}/{expected_count}")
    print(f"completed_by_policy={dict(sorted((k, len(v)) for k, v in completed.items()))}")
    print(f"invalid_configuration_count={len(invalid)}")
    if invalid:
        print("invalid_examples=")
        for item in invalid[:10]:
            print(f"  {item}")
    print(f"submission_marker={(root / 'manifests' / 'SUBMISSION_COMPLETE_UTC.txt').is_file()}")
PY

echo
echo "===== Recent post-training log ====="
latest_post_log=$(find "/scratch/alpine/$USER/job_output" -maxdepth 1 -type f \
    -name 'amos_mix_post_*.out' -print | sort | tail -1)
if [[ -n "$latest_post_log" ]]; then
    echo "$latest_post_log"
    tail -80 "$latest_post_log"
else
    echo "No amos_mix_post log yet."
fi
REMOTE

echo
echo "Cluster audit saved to: $output"
