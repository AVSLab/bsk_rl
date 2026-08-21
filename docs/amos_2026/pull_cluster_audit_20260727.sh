#!/usr/bin/env bash
set -euo pipefail

AUDIT_DATE=${1:-20260727}
DEST="/Users/dahu1128/Downloads/AMOS2026_cluster_results_${AUDIT_DATE}"
REMOTE="dahu1128@login-ci5.rc.colorado.edu:/scratch/alpine/dahu1128/amos2026_mc/amos2026_cluster_audit_${AUDIT_DATE}.tgz"

mkdir -p "$DEST"
scp "$REMOTE" "$DEST/"
tar -xzf "$DEST/amos2026_cluster_audit_${AUDIT_DATE}.tgz" -C "$DEST/"

echo "Copied and unpacked to: $DEST"
