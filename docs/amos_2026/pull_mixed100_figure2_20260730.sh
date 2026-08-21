#!/usr/bin/env bash
set -euo pipefail

remote="dahu1128@login.rc.colorado.edu"
root="/scratch/alpine/dahu1128/amos2026_mc/gat_full_actions_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_45000s_HIO5_SHIO3_20260729T232546Z"
local_root="/Users/dahu1128/Documents/PhD/Conferences/AMOS2026/AMOS_paper_2026/data/cluster_20260730/mixed100_exact_all12"
paper_root="/Users/dahu1128/Documents/PhD/Conferences/AMOS2026/AMOS_paper_2026"
socket="${TMPDIR:-/tmp}/amos2026_mixed100_mux_$$"
remote_archive="/tmp/amos2026_mixed100_figure2_${USER}_$$.tgz"

cleanup() {
    ssh -S "$socket" -O exit "$remote" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Connecting to Alpine. Enter your CURC password when prompted."
ssh -M -S "$socket" -o ControlPersist=300 -fN "$remote"

ssh -S "$socket" "$remote" bash -s -- "$root" "$remote_archive" <<'REMOTE'
set -euo pipefail
root=$1
archive=$2

cd "/projects/$USER/bsk_rl"
source "/projects/$USER/.venv/bin/activate"
export MPLBACKEND=Agg
export MPLCONFIGDIR="/scratch/alpine/$USER/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

python3 examples/amos_2026/analyze_gat_reward_sweep_mc.py \
    --input-root "$root" --expected-seeds 0:100
python3 examples/amos_2026/analyze_gat_reward_sweep_mc_detailed.py \
    --input-root "$root" --expected-seeds 0:100

python3 - "$root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "analysis" / "analysis_report.json").read_text())
detailed = json.loads(
    (root / "analysis_detailed" / "detailed_analysis_report.json").read_text()
)

completed = int(report.get("completed_runs", 0))
missing = int(report.get("missing_runs", -1))
failed = int(report.get("failed_or_incomplete_runs", -1))
selected = int(detailed.get("selected_status_files", 0))
detailed_missing = int(detailed.get("missing_runs", -1))

print(
    "Mixed-100 audit:",
    f"completed={completed}",
    f"missing={missing}",
    f"failed={failed}",
    f"detailed={selected}",
    f"detailed_missing={detailed_missing}",
)
if completed != 1200 or missing != 0 or failed != 0:
    raise SystemExit("Mixed-100 campaign is not a complete 12-policy x 100-seed sweep")
if selected != 1200 or detailed_missing != 0:
    raise SystemExit("Detailed mixed-100 analysis is incomplete")
PY

tar -czf "$archive" -C "$root" manifests analysis analysis_detailed
REMOTE

mkdir -p "$local_root"
scp -o ControlPath="$socket" "$remote:$remote_archive" "$local_root/"
archive_name=$(basename "$remote_archive")
tar -xzf "$local_root/$archive_name" -C "$local_root"
ssh -S "$socket" "$remote" rm -f "$remote_archive"

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$local_root/TRANSFER_COMPLETE_UTC.txt"
cd "$paper_root"
python3 make_mixed100_figure2_candidate.py

echo "Mixed-100 aggregate results copied to:"
echo "$local_root"
echo "Candidate Figure 2 generated in:"
echo "$paper_root/figures/figure2_candidates_20260730"
