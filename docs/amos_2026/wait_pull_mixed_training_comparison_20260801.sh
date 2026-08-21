#!/usr/bin/env bash
set -euo pipefail

# Run on the Mac. Keep one authenticated Alpine session open until the matched
# mixed-trained campaign completes, then analyze, pull, and generate the local
# comparison figure automatically.

remote=${AMOS2026_CLUSTER_HOST:-dahu1128@login.rc.colorado.edu}
comparison_root=/scratch/alpine/dahu1128/amos2026_mc/gat_mixed_trained_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_45000s_HIO5_SHIO3_20260802
local_root=/Users/dahu1128/Downloads/AMOS2026_mixed_training_comparison_20260802/completed
paper_root=/Users/dahu1128/Documents/PhD/Conferences/AMOS2026/AMOS_paper_2026
socket="${TMPDIR:-/tmp}/amos2026_mixtr_wait_mux_$$"
remote_archive="$comparison_root/mixed_training_comparison_aggregate.tgz"
policy_tags=mixed_a0,mixed_a0p1,mixed_a0p2,mixed_a0p3,mixed_a0p4,mixed_a0p5,mixed_a0p75,mixed_a1

cleanup() {
    ssh -S "$socket" -O exit "$remote" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "$local_root"
echo "Connecting to Alpine. Enter your CURC password when prompted."
ssh -M -S "$socket" -o ControlPersist=7200 -fN "$remote"

ssh -S "$socket" "$remote" bash -s -- \
    "$comparison_root" "$remote_archive" "$policy_tags" <<'REMOTE' \
    | tee "$local_root/completion_monitor.log"
set -euo pipefail
root=$1
archive=$2
policy_tags=$3
cd "/projects/$USER/bsk_rl"
source "/projects/$USER/.venv/bin/activate"
export MPLBACKEND=Agg
export MPLCONFIGDIR="/scratch/alpine/$USER/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

deadline=$((SECONDS + 43200))
while true; do
    completed=$(python3 - "$root" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path
completed = defaultdict(set)
for path in Path(sys.argv[1]).glob("seeds_*/*/seed_*/mc_status.json"):
    try:
        status = json.loads(path.read_text())
    except Exception:
        continue
    if status.get("state") == "completed" and status.get("returncode", 0) in (0, None):
        completed[str(status.get("policy_tag"))].add(int(status.get("seed")))
print(sum(len(value) for value in completed.values()))
PY
)
    active=$(squeue -h -u "$USER" -o '%j' | grep -c '^mixtr_' || true)
    printf '%s completed=%s/800 active_array_entries=%s\n' \
        "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$completed" "$active"
    if [[ "$completed" == "800" ]]; then
        break
    fi
    if [[ "$active" == "0" ]]; then
        echo "No mixed-trained jobs remain, but the campaign is incomplete." >&2
        exit 2
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out after 12 hours while waiting for the comparison campaign." >&2
        exit 3
    fi
    sleep 120
done

python3 examples/amos_2026/analyze_gat_reward_sweep_mc.py \
    --input-root "$root" --expected-seeds 0:100 --policy-tags "$policy_tags"
python3 examples/amos_2026/analyze_gat_reward_sweep_mc_detailed.py \
    --input-root "$root" --expected-seeds 0:100 --policy-tags "$policy_tags"

tar -czf "$archive" -C "$root" manifests analysis analysis_detailed
echo "Completed analysis archive: $archive"
REMOTE

scp -o ControlPath="$socket" "$remote:$remote_archive" "$local_root/"
archive="$local_root/$(basename "$remote_archive")"
tar -xzf "$archive" -C "$local_root"

cd "$paper_root"
/Users/dahu1128/Repositories/bsk_rl/.venv/bin/python \
    analyze_training_environment_comparison.py \
    --mixed "$local_root/analysis_detailed/detailed_per_run.csv"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$local_root/COMPARISON_READY_UTC.txt"

echo "Comparison data and figure are ready."
echo "  data:   $local_root"
echo "  figure: $paper_root/data/training_environment_comparison/paper_training_environment_comparison.pdf"
