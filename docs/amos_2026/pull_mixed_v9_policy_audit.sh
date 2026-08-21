#!/usr/bin/env bash
set -euo pipefail

HOST=${AMOS2026_CLUSTER_HOST:-dahu1128@login-ci5.rc.colorado.edu}
DATE_TAG=${1:-$(date +%Y%m%d)}
DEST=${2:-/Users/dahu1128/Downloads/AMOS2026_mixed_v9_policies_${DATE_TAG}}
REMOTE="/scratch/alpine/dahu1128/amos2026_policy_audits/mixed_v9_policy_audit_latest.tgz"

mkdir -p "$DEST"
scp "$HOST:$REMOTE" "$DEST/"
tar -xzf "$DEST/mixed_v9_policy_audit_latest.tgz" -C "$DEST"

echo "Copied and unpacked to: $DEST"
