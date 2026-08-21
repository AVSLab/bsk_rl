#!/usr/bin/env bash
set -euo pipefail

# Run on the Mac. This opens one authenticated SSH connection, synchronizes the
# corrected evaluator, validates it on Alpine, and submits both 200-target
# campaigns without Slurm dependencies.

remote=${AMOS2026_CLUSTER_HOST:-dahu1128@login.rc.colorado.edu}
local_repo=/Users/dahu1128/Repositories/bsk_rl
remote_repo=/projects/dahu1128/bsk_rl
local_status=/Users/dahu1128/Downloads/AMOS2026_prioritysum200_submission_20260801
socket="${TMPDIR:-/tmp}/amos2026_ps200_mux_$$"
max_concurrent_per_block=${1:-2}

files=(
    examples/updated_policy_evaluation.py
    examples/amos_2026/evaluate_gat_reward_sweep_mc.py
    examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh
    examples/amos_2026/analyze_gat_reward_sweep_mc.py
    examples/amos_2026/analyze_gat_reward_sweep_mc_detailed.py
    examples/amos_2026/submit_corrected_density_200targets_prioritysum200_no_deps.sh
    docs/amos_2026/audit_and_package_corrected_density_prioritysum200.sh
    docs/amos_2026/pull_corrected_density_prioritysum200.sh
)

cleanup() {
    ssh -S "$socket" -O exit "$remote" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "$local_status"
cd "$local_repo"

echo "Connecting to Alpine. Enter your CURC password when prompted."
ssh -M -S "$socket" -o ControlPersist=300 -fN "$remote"

rsync -azR -e "ssh -S $socket" "${files[@]}" "$remote:$remote_repo/"

ssh -S "$socket" "$remote" bash -s -- "$max_concurrent_per_block" <<'REMOTE' \
    | tee "$local_status/submission.log"
set -euo pipefail
max_concurrent_per_block=$1

cd "/projects/$USER/bsk_rl"
source "/projects/$USER/.venv/bin/activate"
export MPLBACKEND=Agg

python3 -m py_compile \
    examples/updated_policy_evaluation.py \
    examples/amos_2026/evaluate_gat_reward_sweep_mc.py \
    examples/amos_2026/analyze_gat_reward_sweep_mc.py \
    examples/amos_2026/analyze_gat_reward_sweep_mc_detailed.py
bash -n examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh
bash -n examples/amos_2026/submit_corrected_density_200targets_prioritysum200_no_deps.sh
bash -n docs/amos_2026/audit_and_package_corrected_density_prioritysum200.sh

echo "===== Submitting corrected LEO-200 and mixed-200 campaigns ====="
bash examples/amos_2026/submit_corrected_density_200targets_prioritysum200_no_deps.sh \
    "$max_concurrent_per_block"

echo "===== Current corrected-density jobs ====="
squeue -u "$USER" --format='%.18i %.26j %.2t %.10M %.10l %.6D %R' \
    | grep -E 'JOBID|ps200_' || true
REMOTE

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$local_status/SUBMISSION_COMPLETE_UTC.txt"
echo
echo "Submission record: $local_status"
