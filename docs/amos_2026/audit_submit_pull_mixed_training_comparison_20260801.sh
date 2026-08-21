#!/usr/bin/env bash
set -euo pipefail

# Run on the Mac. Audit mixed-fixed training and any existing mixed-trained
# evaluation, submit one no-dependency evaluation only when needed, and pull
# aggregate results immediately when all 800 runs are complete.

remote=${AMOS2026_CLUSTER_HOST:-dahu1128@login.rc.colorado.edu}
local_repo=/Users/dahu1128/Repositories/bsk_rl
remote_repo=/projects/dahu1128/bsk_rl
local_root=/Users/dahu1128/Downloads/AMOS2026_mixed_training_comparison_20260801
socket="${TMPDIR:-/tmp}/amos2026_mixtr_mux_$$"
remote_stage="/tmp/amos2026_mixtr_audit_${USER}_$$"
remote_archive="${remote_stage}.tgz"
comparison_root="/scratch/alpine/dahu1128/amos2026_mc/gat_mixed_trained_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_45000s_HIO5_SHIO3_20260801"

files=(
    examples/updated_policy_evaluation.py
    examples/amos_2026/audit_mixed_v9_training_runs.py
    examples/amos_2026/build_mixed_trained_policy_manifest.py
    examples/amos_2026/evaluate_gat_reward_sweep_mc.py
    examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh
    examples/amos_2026/analyze_gat_reward_sweep_mc.py
    examples/amos_2026/analyze_gat_reward_sweep_mc_detailed.py
    examples/amos_2026/submit_mixed_trained_comparison_mc_no_deps.sh
)

cleanup() {
    ssh -S "$socket" -O exit "$remote" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "$local_root"
cd "$local_repo"
echo "Connecting to Alpine. Enter your CURC password when prompted."
ssh -M -S "$socket" -o ControlPersist=300 -fN "$remote"
rsync -azR -e "ssh -S $socket" "${files[@]}" "$remote:$remote_repo/"

ssh -S "$socket" "$remote" bash -s -- \
    "$comparison_root" "$remote_stage" "$remote_archive" <<'REMOTE' \
    | tee "$local_root/cluster_audit.log"
set -euo pipefail
comparison_root=$1
stage=$2
archive=$3
repo="/projects/$USER/bsk_rl"
audit_root="/scratch/alpine/$USER/amos2026_mc/mixed_training_comparison_audit_20260801"
inventory="$audit_root/mixed_v9_training_inventory.csv"
custom_spec="$audit_root/mixed_fixed100_custom_policies.json"
custom_tags="$audit_root/mixed_fixed100_policy_tags.txt"

# Reuse any previously submitted matching campaign, including the post-training
# campaign dated 20260802, instead of creating duplicate evaluations.
existing_root=$(find "/scratch/alpine/$USER/amos2026_mc" -maxdepth 1 -type d \
    -name 'gat_mixed_trained_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_45000s_HIO5_SHIO3_*' \
    -exec test -f '{}/manifests/SUBMISSION_COMPLETE_UTC.txt' ';' -print \
    | sort | tail -1)
if [[ -n "$existing_root" ]]; then
    comparison_root=$existing_root
    echo "Reusing previously submitted comparison root: $comparison_root"
fi

cd "$repo"
source "/projects/$USER/.venv/bin/activate"
export MPLBACKEND=Agg
export MPLCONFIGDIR="/scratch/alpine/$USER/.cache/matplotlib"
mkdir -p "$audit_root" "$MPLCONFIGDIR"

python3 -m py_compile \
    examples/amos_2026/audit_mixed_v9_training_runs.py \
    examples/amos_2026/build_mixed_trained_policy_manifest.py \
    examples/amos_2026/evaluate_gat_reward_sweep_mc.py \
    examples/amos_2026/analyze_gat_reward_sweep_mc.py \
    examples/amos_2026/analyze_gat_reward_sweep_mc_detailed.py
bash -n examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh
bash -n examples/amos_2026/submit_mixed_trained_comparison_mc_no_deps.sh

python3 examples/amos_2026/audit_mixed_v9_training_runs.py \
    --policy-root "/scratch/alpine/$USER/rllib_results" \
    --output-dir "$audit_root"

manifest_complete=0
if python3 examples/amos_2026/build_mixed_trained_policy_manifest.py \
    --inventory "$inventory" \
    --output-json "$custom_spec" \
    --output-tags "$custom_tags"; then
    manifest_complete=1
fi

status_summary() {
    python3 - "$comparison_root" <<'PY'
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1])
states = Counter()
completed = defaultdict(set)
for path in root.glob("seeds_*/*/seed_*/mc_status.json"):
    try:
        status = json.loads(path.read_text())
    except Exception:
        states["unreadable"] += 1
        continue
    state = str(status.get("state", "unknown"))
    states[state] += 1
    if state == "completed" and status.get("returncode", 0) in (0, None):
        completed[str(status.get("policy_tag"))].add(int(status.get("seed")))
print(json.dumps({
    "root": str(root),
    "states": dict(states),
    "valid_completed": sum(len(value) for value in completed.values()),
    "completed_by_policy": {key: len(value) for key, value in sorted(completed.items())},
}, indent=2, sort_keys=True))
PY
}

echo "===== Existing mixed-trained comparison status ====="
status_summary

submission_marker="$comparison_root/manifests/SUBMISSION_COMPLETE_UTC.txt"
if [[ "$manifest_complete" == "1" && ! -f "$submission_marker" ]]; then
    echo "===== Submitting mixed-trained comparison ====="
    BSK_RL_MIXED_CUSTOM_POLICIES_JSON="$custom_spec" \
    BSK_RL_MIXED_POLICY_TAGS=$(cat "$custom_tags") \
    BSK_RL_MIXED_COMPARISON_ROOT="$comparison_root" \
        bash examples/amos_2026/submit_mixed_trained_comparison_mc_no_deps.sh 2
elif [[ -f "$submission_marker" ]]; then
    echo "Comparison campaign was already submitted. No duplicate jobs created."
elif [[ "$manifest_complete" != "1" ]]; then
    echo "The eight-policy mixed training sweep is not complete. Evaluation was not submitted."
fi

echo "===== Current mixed-trained comparison jobs ====="
squeue -u "$USER" --format='%.18i %.26j %.2t %.10M %.10l %.6D %R' \
    | grep -E 'JOBID|mixtr_' || true

valid_completed=$(python3 - "$comparison_root" <<'PY'
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

rm -rf "$stage"
mkdir -p "$stage/training_audit"
cp "$audit_root"/*.json "$stage/training_audit/" 2>/dev/null || true
cp "$audit_root"/*.csv "$stage/training_audit/" 2>/dev/null || true
cp "$audit_root"/*.txt "$stage/training_audit/" 2>/dev/null || true

if [[ "$valid_completed" == "800" ]]; then
    policy_tags=$(cat "$custom_tags")
    python3 examples/amos_2026/analyze_gat_reward_sweep_mc.py \
        --input-root "$comparison_root" --expected-seeds 0:100 \
        --policy-tags "$policy_tags"
    python3 examples/amos_2026/analyze_gat_reward_sweep_mc_detailed.py \
        --input-root "$comparison_root" --expected-seeds 0:100 \
        --policy-tags "$policy_tags"
    cp -R "$comparison_root/analysis" "$stage/"
    cp -R "$comparison_root/analysis_detailed" "$stage/"
    cp -R "$comparison_root/manifests" "$stage/"
    printf 'complete\n' > "$stage/CAMPAIGN_STATE.txt"
else
    printf 'incomplete: %s of 800 valid completed runs\n' "$valid_completed" \
        > "$stage/CAMPAIGN_STATE.txt"
fi

tar -czf "$archive" -C "$(dirname "$stage")" "$(basename "$stage")"
echo "Packaged current audit state: $archive"
REMOTE

scp -o ControlPath="$socket" "$remote:$remote_archive" "$local_root/"
archive="$local_root/$(basename "$remote_archive")"
tar -xzf "$archive" -C "$local_root"
ssh -S "$socket" "$remote" rm -rf "$remote_stage" "$remote_archive"

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$local_root/TRANSFER_COMPLETE_UTC.txt"
echo "Audit package copied to: $local_root"
