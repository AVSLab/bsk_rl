#!/usr/bin/env bash
set -euo pipefail

# Run on the Mac after the Alpine audit reports all_complete=true.

remote=${AMOS2026_CLUSTER_HOST:-dahu1128@login.rc.colorado.edu}
date_tag=${1:-$(date +%Y%m%d)}
destination=${2:-/Users/dahu1128/Downloads/AMOS2026_corrected_density_prioritySum200_${date_tag}}
socket="${TMPDIR:-/tmp}/amos2026_ps200_pull_mux_$$"

cleanup() {
    ssh -S "$socket" -O exit "$remote" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "$destination"
echo "Connecting to Alpine. Enter your CURC password when prompted."
ssh -M -S "$socket" -o ControlPersist=300 -fN "$remote"

remote_archive=$(ssh -S "$socket" "$remote" \
    "find /scratch/alpine/\$USER/amos2026_mc -maxdepth 2 -type f -name 'amos2026_corrected_density_prioritySum200_results.tgz' -print | sort | tail -1")
if [[ -z "$remote_archive" ]]; then
    echo "No packaged corrected-density archive found. Run the audit/package script first." >&2
    exit 1
fi

scp -o ControlPath="$socket" "$remote:$remote_archive" "$destination/"
archive="$destination/$(basename "$remote_archive")"
tar -xzf "$archive" -C "$destination"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$destination/TRANSFER_COMPLETE_UTC.txt"

echo "Copied and unpacked corrected results to:"
echo "  $destination"
