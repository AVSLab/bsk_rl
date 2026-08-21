#!/usr/bin/env bash
set -euo pipefail

remote="dahu1128@login.rc.colorado.edu"
mixed100_root="/scratch/alpine/dahu1128/amos2026_mc/gat_full_actions_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_45000s_HIO5_SHIO3_20260729T232546Z"
local_root="/Users/dahu1128/Downloads/AMOS2026_cluster_results_20260731"
paper_data="/Users/dahu1128/Documents/PhD/Conferences/AMOS2026/AMOS_paper_2026/data/cluster_20260731"
socket="${TMPDIR:-/tmp}/amos2026_audit_mux_$$"
remote_stage="/tmp/amos2026_audit_${USER}_$$"
remote_archive="${remote_stage}.tgz"
repo="/Users/dahu1128/Repositories/bsk_rl"
remote_repo="/projects/dahu1128/bsk_rl"
sync_files=(
    examples/updated_policy_evaluation.py
    examples/amos_2026/evaluate_gat_reward_sweep_mc.py
    examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh
    examples/amos_2026/submit_gat_reward_sweep_mc_mixed_100targets_45000s_0to99.sh
    examples/amos_2026/submit_gat_reward_sweep_mc_mixed_200targets_45000s_0to99.sh
    examples/amos_2026/evaluate_heuristic_mc.py
    examples/amos_2026/sbatch_evaluate_heuristics_mc_10seeds.sh
    examples/amos_2026/submit_heuristics_mc_mixed_100targets_45000s_0to99.sh
    src/bsk_rl/act/discrete_actions.py
    src/bsk_rl/sim/dyn.py
)

cleanup() {
    ssh -S "$socket" -O exit "$remote" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Connecting to Alpine. Enter your CURC password when prompted."
ssh -M -S "$socket" -o ControlPersist=300 -fN "$remote"

cd "$repo"
rsync -azR -e "ssh -S $socket" "${sync_files[@]}" "$remote:$remote_repo/"

ssh -S "$socket" "$remote" bash -s -- \
    "$mixed100_root" "$remote_stage" "$remote_archive" <<'REMOTE'
set -euo pipefail
mixed100_root=$1
stage=$2
archive=$3

repo="/projects/$USER/bsk_rl"
rm -rf "$stage"
mkdir -p "$stage/mixed100" "$stage/mixed_training"

cd "$repo"
source "/projects/$USER/.venv/bin/activate"

squeue -u "$USER" --format='%.18i %.35j %.2t %.12M %.12l %.6D %R' \
    > "$stage/slurm_queue.txt"
sacct -j 30568059 --format=JobID,JobName,State,Elapsed,ExitCode,Start,End -X \
    > "$stage/mixed_training/slurm_job_30568059.txt" 2>&1 || true

python3 examples/amos_2026/audit_mixed_v9_training_runs.py \
    --policy-root "/scratch/alpine/$USER/rllib_results" \
    --output-dir "$stage/mixed_training"

mixed_policy_spec="/scratch/alpine/$USER/amos2026_mc/manifests/mixed_fixed100_reward_sweep_20260731.json"
mixed_policy_tags="${mixed_policy_spec%.json}_tags.txt"
mkdir -p "$(dirname "$mixed_policy_spec")"
python3 - "$stage/mixed_training/mixed_v9_training_inventory.csv" \
    "$mixed_policy_spec" "$mixed_policy_tags" <<'PY'
import csv
import json
import sys
from pathlib import Path

inventory = Path(sys.argv[1])
output = Path(sys.argv[2])
tags_output = Path(sys.argv[3])
expected = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
required_group = (
    "polaris-gat-full-actions-obs-v9-mixed-fixed-"
    "50leo30meo20geo-100targets-reward-sweep"
)

with inventory.open(newline="") as handle:
    rows = list(csv.DictReader(handle))

selected = {}
for row in rows:
    if row.get("wandb_group", "").lower() != required_group:
        continue
    if row.get("fixed_target_count_100", "").lower() != "true":
        continue
    if not row.get("latest_checkpoint"):
        continue
    try:
        alpha = round(float(row["alpha"]), 6)
        rank = (
            int(float(row.get("progress_iteration") or -1)),
            int(float(row.get("latest_checkpoint_iteration") or -1)),
            row.get("run_dir", ""),
        )
    except (TypeError, ValueError):
        continue
    if alpha not in expected:
        continue
    if alpha not in selected or rank > selected[alpha][0]:
        selected[alpha] = (rank, row)

policies = {}
tags = []
for alpha in expected:
    if alpha not in selected:
        continue
    row = selected[alpha][1]
    alpha_text = (f"{alpha:g}").replace(".", "p")
    tag = f"mixed_a{alpha_text}"
    tags.append(tag)
    policies[tag] = {
        "checkpoint_dir": row["latest_checkpoint"],
        "alpha": alpha,
        "label": f"Mixed-trained alpha={alpha:g}",
        "training_run_dir": row["run_dir"],
        "training_iteration": row.get("progress_iteration"),
    }

payload = {
    "schema_version": 1,
    "training_environment": "mixed_exact_50LEO_30MEO_20GEO_100targets",
    "required_wandb_group": required_group,
    "expected_alphas": list(expected),
    "found_alphas": sorted(selected),
    "missing_alphas": [alpha for alpha in expected if alpha not in selected],
    "policies": policies,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tags_output.write_text(",".join(tags) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
cp "$mixed_policy_spec" "$stage/mixed_training/"
cp "$mixed_policy_tags" "$stage/mixed_training/"

mixed_policy_count=$(python3 - "$mixed_policy_spec" <<'PY'
import json
import sys
print(len(json.load(open(sys.argv[1]))["policies"]))
PY
)
mixed_eval_root="/scratch/alpine/$USER/amos2026_mc/gat_mixed_trained_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_45000s_HIO5_SHIO3_20260731"
if [[ "$mixed_policy_count" == "8" ]]; then
    if [[ ! -f "$mixed_eval_root/manifests/SUBMISSION_COMPLETE_UTC.txt" ]]; then
        echo "Submitting eight-policy mixed-trained evaluation campaign" \
            | tee "$stage/mixed_training/evaluation_submission.txt"
        BSK_RL_MC_POLICY_TAGS=$(cat "$mixed_policy_tags") \
        BSK_RL_MC_CUSTOM_POLICIES_JSON="@$mixed_policy_spec" \
        BSK_RL_MC_OUTPUT_ROOT="$mixed_eval_root" \
        BSK_RL_MC_CHAIN_BLOCKS=0 \
        bash examples/amos_2026/submit_gat_reward_sweep_mc_mixed_100targets_45000s_0to99.sh 4 \
            | tee -a "$stage/mixed_training/evaluation_submission.txt"
        date -u +"%Y-%m-%dT%H:%M:%SZ" \
            > "$mixed_eval_root/manifests/SUBMISSION_COMPLETE_UTC.txt"
    else
        echo "Mixed-trained evaluation campaign was already submitted: $mixed_eval_root" \
            | tee "$stage/mixed_training/evaluation_submission.txt"
    fi
else
    echo "Found $mixed_policy_count of 8 required mixed-trained policies; evaluation not submitted." \
        | tee "$stage/mixed_training/evaluation_submission.txt"
fi

python3 - "$mixed100_root" "$stage/mixed100/status_audit.json" <<'PY'
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
expected_tags = (
    "00d100i", "10d90i", "20d80i", "30d70i", "40d60i", "50d50i",
    "60d40i", "70d30i", "75d25i", "80d20i", "90d10i", "100d00i",
)
states = Counter()
completed_by_policy = defaultdict(set)
failed_by_policy = defaultdict(set)
status_files = []

if root.is_dir():
    status_files = sorted(root.glob("seeds_*/*/seed_*/mc_status.json"))
for path in status_files:
    try:
        status = json.loads(path.read_text())
    except Exception:
        states["unreadable"] += 1
        continue
    state = str(status.get("state", "unknown"))
    tag = str(status.get("policy_tag", path.parent.parent.name))
    seed = int(status.get("seed", path.parent.name.replace("seed_", "")))
    states[state] += 1
    if state == "completed" and status.get("returncode", 0) in (0, None):
        completed_by_policy[tag].add(seed)
    elif state == "failed" or status.get("returncode") not in (0, None):
        failed_by_policy[tag].add(seed)

policies = {}
for tag in expected_tags:
    completed = sorted(completed_by_policy[tag])
    failed = sorted(failed_by_policy[tag])
    policies[tag] = {
        "completed_count": len(completed),
        "completed_seeds": completed,
        "failed_count": len(failed),
        "failed_seeds": failed,
        "missing_seeds": sorted(set(range(100)) - set(completed)),
    }

payload = {
    "root": str(root),
    "root_exists": root.is_dir(),
    "status_file_count": len(status_files),
    "states": dict(sorted(states.items())),
    "completed_count": sum(len(v) for v in completed_by_policy.values()),
    "expected_count": len(expected_tags) * 100,
    "complete": all(len(completed_by_policy[tag]) == 100 for tag in expected_tags),
    "policies": policies,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: payload[k] for k in (
    "root_exists", "status_file_count", "states", "completed_count",
    "expected_count", "complete"
)}, indent=2, sort_keys=True))
PY

if [[ -d "$mixed100_root" ]]; then
    export MPLBACKEND=Agg
    export MPLCONFIGDIR="/scratch/alpine/$USER/.cache/matplotlib"
    mkdir -p "$MPLCONFIGDIR"

    python3 examples/amos_2026/analyze_gat_reward_sweep_mc.py \
        --input-root "$mixed100_root" --expected-seeds 0:100
    python3 examples/amos_2026/analyze_gat_reward_sweep_mc_detailed.py \
        --input-root "$mixed100_root" --expected-seeds 0:100

    for name in manifests analysis analysis_detailed; do
        if [[ -d "$mixed100_root/$name" ]]; then
            cp -R "$mixed100_root/$name" "$stage/mixed100/"
        fi
    done
fi

echo "===== Heuristic smoke tests =====" | tee "$stage/heuristic_submission.txt"
module purge
if ! module --ignore_cache load gcc/14.2.0; then
    export PATH="/curc/sw/install/gcc/14.2.0/bin:${PATH}"
fi
module --ignore_cache load python/3.10.2 || true
export LD_LIBRARY_PATH="/curc/sw/install/gcc/14.2.0/lib64:$(dirname "$(gcc -print-file-name=libstdc++.so.6)"):${LD_LIBRARY_PATH:-}"
python3 -m py_compile \
    examples/updated_policy_evaluation.py \
    examples/amos_2026/evaluate_heuristic_mc.py \
    src/bsk_rl/act/discrete_actions.py \
    src/bsk_rl/sim/dyn.py
bash -n examples/amos_2026/sbatch_evaluate_heuristics_mc_10seeds.sh
bash -n examples/amos_2026/submit_heuristics_mc_mixed_100targets_45000s_0to99.sh

smoke_root="/scratch/alpine/$USER/amos2026_mc/heuristics_smoke_20260731"
rm -rf "$smoke_root"
for mode in angle priority_angle; do
    echo "Smoke mode: $mode" | tee -a "$stage/heuristic_submission.txt"
    python3 -u examples/updated_policy_evaluation.py \
        --heuristic_mode "$mode" \
        --policy_layout gat_full \
        --obs_v 9 \
        --seed 0 \
        --reward_mix_tag 100d00i \
        --target_env mixed \
        --mix_weights '{"LEO":0.5,"MEO":0.3,"GEO":0.2}' \
        --exact_mix_counts \
        --n_targets 3 \
        --n_targets_ahead 3 \
        --total_time_sec 60 \
        --dynamic_priority_event off \
        --hio_count 0 \
        --shio_count 0 \
        --output_dir "$smoke_root/$mode" \
        --no_save_data \
        --quiet \
        --skip_plots \
        --no_show_plots \
        --no_shield \
        >> "$stage/heuristic_submission.txt" 2>&1
done

active_heuristic_jobs=$(
    squeue --noheader --user="$USER" --format='%j' | grep '^heur_mc_' || true
)
if [[ -n "$active_heuristic_jobs" ]]; then
    echo "Existing heuristic jobs found; no duplicate submission:" \
        | tee -a "$stage/heuristic_submission.txt"
    echo "$active_heuristic_jobs" | tee -a "$stage/heuristic_submission.txt"
else
    bash examples/amos_2026/submit_heuristics_mc_mixed_100targets_45000s_0to99.sh 2 \
        | tee -a "$stage/heuristic_submission.txt"
fi
squeue -u "$USER" --format='%.18i %.28j %.2t %.10M %.10l %.6D %R' \
    | grep -E 'JOBID|heur_mc_' > "$stage/heuristic_queue.txt" || true

tar -czf "$archive" -C "$(dirname "$stage")" "$(basename "$stage")"
REMOTE

mkdir -p "$local_root" "$paper_data"
scp -o ControlPath="$socket" "$remote:$remote_archive" "$local_root/"
archive_name=$(basename "$remote_archive")
tar -xzf "$local_root/$archive_name" -C "$local_root"
extracted="$local_root/$(basename "$remote_stage")"

rm -rf "$paper_data/mixed100" "$paper_data/mixed_training"
cp -R "$extracted/mixed100" "$paper_data/"
cp -R "$extracted/mixed_training" "$paper_data/"
cp "$extracted/slurm_queue.txt" "$paper_data/"
cp "$extracted/heuristic_submission.txt" "$paper_data/"
cp "$extracted/heuristic_queue.txt" "$paper_data/"

ssh -S "$socket" "$remote" rm -rf "$remote_stage" "$remote_archive"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$paper_data/TRANSFER_COMPLETE_UTC.txt"

echo
echo "Audit and aggregate results copied to:"
echo "  $local_root"
echo "Paper data copy:"
echo "  $paper_data"
echo "TRANSFER_COMPLETE" > "$local_root/TRANSFER_COMPLETE.txt"
