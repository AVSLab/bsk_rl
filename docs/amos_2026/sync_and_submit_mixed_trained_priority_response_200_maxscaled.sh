#!/usr/bin/env bash

# Run on the Mac. This opens one authenticated SSH connection, synchronizes the
# max-scaled mixed-trained priority-response campaign, validates it on Alpine,
# and submits ten independent 10-seed jobs without Slurm dependencies.

set -euo pipefail

remote=${AMOS2026_CLUSTER_HOST:-dahu1128@login-ci5.rc.colorado.edu}
local_repo=/Users/dahu1128/Repositories/bsk_rl
remote_repo=/projects/dahu1128/bsk_rl
local_status=/Users/dahu1128/Downloads/AMOS2026_mixedtrained_priority_response_200_visibility_submission_20260803
socket="${TMPDIR:-/tmp}/amos2026_mixmaxpr200_mux_$$"

files=(
    examples/sim_config.py
    src/bsk_rl/obs/observations.py
    src/bsk_rl/scene/rso_targets.py
    examples/updated_policy_evaluation.py
    examples/amos_2026/evaluate_gat_reward_sweep_mc.py
    examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh
    examples/amos_2026/analyze_gat_priority_response_mc.py
    examples/amos_2026/submit_mixed_trained_alpha0p1_priority_response_mc_mixed_200targets_maxscaled_0to99.sh
    tests/unittest/obs/test_priority_candidate_tracking.py
    tests/unittest/scene/test_rso_priority_controls.py
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
module purge
if ! module --ignore_cache load gcc/14.2.0; then
    export PATH="/curc/sw/install/gcc/14.2.0/bin:${PATH}"
fi
if ! module --ignore_cache load python/3.10.2; then
    echo "WARNING: python/3.10.2 module not found; using the virtualenv Python"
fi
source "/projects/$USER/.venv/bin/activate"
if [[ -d /curc/sw/install/gcc/14.2.0/lib64 ]]; then
    export LD_LIBRARY_PATH="/curc/sw/install/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}"
fi
if command -v gcc >/dev/null 2>&1; then
    export LD_LIBRARY_PATH="$(dirname "$(gcc -print-file-name=libstdc++.so.6)"):${LD_LIBRARY_PATH:-}"
fi
export MPLBACKEND=Agg

python3 -m py_compile \
    examples/sim_config.py \
    src/bsk_rl/obs/observations.py \
    src/bsk_rl/scene/rso_targets.py \
    examples/updated_policy_evaluation.py \
    examples/amos_2026/evaluate_gat_reward_sweep_mc.py \
    examples/amos_2026/analyze_gat_priority_response_mc.py
bash -n examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh
bash -n examples/amos_2026/submit_mixed_trained_alpha0p1_priority_response_mc_mixed_200targets_maxscaled_0to99.sh
python3 -m pytest -q \
    tests/unittest/obs/test_priority_candidate_tracking.py \
    tests/unittest/scene/test_rso_priority_controls.py

echo "===== Submitting max-scaled mixed-trained alpha=0.1 campaign ====="
bash examples/amos_2026/submit_mixed_trained_alpha0p1_priority_response_mc_mixed_200targets_maxscaled_0to99.sh

echo "===== Current campaign jobs ====="
squeue -u "$USER" --format='%.18i %.26j %.2t %.10M %.10l %.6D %R' \
    | grep -E 'JOBID|mxmaxpr_' || true
REMOTE

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$local_status/SUBMISSION_COMPLETE_UTC.txt"
echo
echo "Submission record: $local_status"
