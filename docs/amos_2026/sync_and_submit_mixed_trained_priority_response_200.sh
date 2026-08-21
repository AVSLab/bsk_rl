#!/usr/bin/env bash

# Run on the Mac. This opens one authenticated SSH connection, synchronizes the
# focused mixed-trained priority-response campaign, validates the scripts, and
# submits ten independent 10-seed jobs without Slurm dependencies.

set -euo pipefail

remote=${AMOS2026_CLUSTER_HOST:-dahu1128@login-ci5.rc.colorado.edu}
local_repo=/Users/dahu1128/Repositories/bsk_rl
remote_repo=/projects/dahu1128/bsk_rl
local_status=/Users/dahu1128/Downloads/AMOS2026_mixedtrained_priority_response_200_submission_20260803
socket="${TMPDIR:-/tmp}/amos2026_mixpr200_mux_$$"

files=(
    examples/sim_config.py
    src/bsk_rl/scene/rso_targets.py
    src/bsk_rl/obs/observations.py
    examples/updated_policy_evaluation.py
    examples/amos_2026/evaluate_gat_reward_sweep_mc.py
    examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh
    examples/amos_2026/analyze_gat_priority_response_mc.py
    examples/amos_2026/submit_mixed_trained_alpha0p1_priority_response_mc_mixed_200targets_0to99.sh
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

ssh -S "$socket" "$remote" bash -s <<'REMOTE' | tee "$local_status/submission.log"
set -euo pipefail

cd "/projects/$USER/bsk_rl"
source "/projects/$USER/.venv/bin/activate"
export MPLBACKEND=Agg

python3 -m py_compile \
    examples/updated_policy_evaluation.py \
    examples/amos_2026/evaluate_gat_reward_sweep_mc.py \
    examples/amos_2026/analyze_gat_priority_response_mc.py
bash -n examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh
bash -n examples/amos_2026/submit_mixed_trained_alpha0p1_priority_response_mc_mixed_200targets_0to99.sh

echo "===== Submitting mixed-trained alpha=0.1 priority-response campaign ====="
bash examples/amos_2026/submit_mixed_trained_alpha0p1_priority_response_mc_mixed_200targets_0to99.sh

echo "===== Current campaign jobs ====="
squeue -u "$USER" --format='%.18i %.26j %.2t %.10M %.10l %.6D %R' \
    | grep -E 'JOBID|mixpr200_' || true
REMOTE

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$local_status/SUBMISSION_COMPLETE_UTC.txt"
echo
echo "Submission record: $local_status"
