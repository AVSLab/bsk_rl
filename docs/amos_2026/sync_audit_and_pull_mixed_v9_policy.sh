#!/usr/bin/env bash
set -euo pipefail

# Run on the Mac. This uses one authenticated SSH connection to:
#   1. sync the audit code,
#   2. inventory Alpine mixed obs-v9 runs,
#   3. package fixed-100 checkpoints, and
#   4. stream the package back and unpack it locally.

HOST=${AMOS2026_CLUSTER_HOST:-dahu1128@login-ci5.rc.colorado.edu}
DATE_TAG=${1:-$(date +%Y%m%d)}
LOCAL_REPO=/Users/dahu1128/Repositories/bsk_rl
REMOTE_REPO=/projects/dahu1128/bsk_rl
DEST=${2:-/Users/dahu1128/Downloads/AMOS2026_mixed_v9_policies_${DATE_TAG}}
ARCHIVE="$DEST/mixed_v9_policy_audit_latest.tgz"
PARTIAL="$ARCHIVE.part"

cd "$LOCAL_REPO"
mkdir -p "$DEST"
rm -f "$PARTIAL"

echo "Connecting to $HOST."
echo "Enter your Alpine password when prompted."

COPYFILE_DISABLE=1 tar -czf - \
    examples/amos_2026/audit_mixed_v9_training_runs.py \
    docs/amos_2026/package_mixed_v9_policy_audit.sh \
  | ssh "$HOST" "set -euo pipefail
      tar -xzf - -C '$REMOTE_REPO'
      cd '$REMOTE_REPO'
      INCLUDE_CHECKPOINTS=1 \
        bash docs/amos_2026/package_mixed_v9_policy_audit.sh '$DATE_TAG' >&2
      cat /scratch/alpine/dahu1128/amos2026_policy_audits/mixed_v9_policy_audit_latest.tgz
    " >"$PARTIAL"

mv "$PARTIAL" "$ARCHIVE"
tar -xzf "$ARCHIVE" -C "$DEST"

echo
echo "Copied and unpacked to: $DEST"
echo "Inventory files:"
find "$DEST" -maxdepth 3 -type f \
    \( -name 'summary.json' \
    -o -name 'mixed_v9_training_inventory.csv' \
    -o -name 'copy_candidates.txt' \) \
    -print
