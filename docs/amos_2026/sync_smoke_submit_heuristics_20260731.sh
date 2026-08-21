#!/usr/bin/env bash
set -euo pipefail

remote="dahu1128@login.rc.colorado.edu"
repo="/Users/dahu1128/Repositories/bsk_rl"
remote_repo="/projects/dahu1128/bsk_rl"
socket="${TMPDIR:-/tmp}/amos2026_heur_mux_$$"
local_status="/Users/dahu1128/Downloads/AMOS2026_heuristic_submission_20260731"

files=(
    examples/updated_policy_evaluation.py
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

mkdir -p "$local_status"
cd "$repo"

echo "Connecting to Alpine. Enter your CURC password when prompted."
ssh -M -S "$socket" -o ControlPersist=300 -fN "$remote"

rsync -azR -e "ssh -S $socket" "${files[@]}" "$remote:$remote_repo/"

ssh -S "$socket" "$remote" bash -s <<'REMOTE' | tee "$local_status/submission.log"
set -euo pipefail
cd "/projects/$USER/bsk_rl"
source "/projects/$USER/.venv/bin/activate"
export MPLBACKEND=Agg
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
    echo "===== Smoke test: $mode ====="
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
        --no_shield
done

echo "===== Submitting full heuristic Monte Carlo campaign ====="
bash examples/amos_2026/submit_heuristics_mc_mixed_100targets_45000s_0to99.sh 2

echo "===== Current heuristic jobs ====="
squeue -u "$USER" --format='%.18i %.28j %.2t %.10M %.10l %.6D %R' \
    | grep -E 'JOBID|heur_mc_' || true
REMOTE

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$local_status/SUBMISSION_COMPLETE_UTC.txt"
echo "Heuristic submission record: $local_status"
