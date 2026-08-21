#!/usr/bin/env bash
set -euo pipefail

remote=${AMOS2026_CLUSTER_HOST:-dahu1128@login-ci5.rc.colorado.edu}
local_repo=/Users/dahu1128/Repositories/bsk_rl
remote_repo=/projects/dahu1128/bsk_rl
base_root=/scratch/alpine/dahu1128/amos2026_mc/corrected_density_prioritySum200_20260801T222325Z
local_root=/Users/dahu1128/Downloads/AMOS2026_corrected_density_prioritySum200_20260802
socket="${TMPDIR:-/tmp}/amos2026_finalize_mux_$$"
duplicate_handoff=30703176
retained_handoff=30688737

files=(
    examples/amos_2026/analyze_gat_reward_sweep_mc.py
    examples/amos_2026/analyze_gat_reward_sweep_mc_detailed.py
    docs/amos_2026/audit_and_package_corrected_density_prioritysum200.sh
)

cleanup() {
    ssh -S "$socket" -O exit "$remote" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "$local_root"
cd "$local_repo"
echo "Connecting to Alpine. Enter your CURC password when prompted."
ssh -M -S "$socket" -o ControlPersist=600 -fN "$remote"
rsync -azR -e "ssh -S $socket" "${files[@]}" "$remote:$remote_repo/"

ssh -S "$socket" "$remote" bash -s -- \
    "$base_root" "$duplicate_handoff" "$retained_handoff" <<'REMOTE' \
    | tee "$local_root/finalize.log"
set -euo pipefail
base_root=$1
duplicate_handoff=$2
retained_handoff=$3

if squeue -h -j "$duplicate_handoff" | grep -q .; then
    scancel "$duplicate_handoff"
    echo "Canceled duplicate handoff job: $duplicate_handoff"
else
    echo "Duplicate handoff job was no longer queued: $duplicate_handoff"
fi

echo "Retained handoff job:"
squeue -j "$retained_handoff" -o '%.18i %.24j %.2t %.10M %R' || true
scontrol show job "$retained_handoff" \
    | tr ' ' '\n' \
    | grep -E '^(JobId|JobName|JobState|Dependency|Command)=' || true

cd "/projects/$USER/bsk_rl"
source "/projects/$USER/.venv/bin/activate"
bash docs/amos_2026/audit_and_package_corrected_density_prioritysum200.sh \
    "$base_root"
REMOTE

remote_archive="$base_root/amos2026_corrected_density_prioritySum200_results.tgz"
scp -o ControlPath="$socket" "$remote:$remote_archive" "$local_root/"
archive="$local_root/$(basename "$remote_archive")"
tar -xzf "$archive" -C "$local_root"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$local_root/TRANSFER_COMPLETE_UTC.txt"

echo
echo "Corrected density results copied to: $local_root"
