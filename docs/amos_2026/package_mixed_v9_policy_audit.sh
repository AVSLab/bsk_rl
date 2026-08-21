#!/usr/bin/env bash
set -euo pipefail

# Run on Alpine. By default, include the latest valid checkpoint from every
# fixed-100-target mixed obs-v9 GAT run so the package can be copied to a Mac.

INCLUDE_CHECKPOINTS=${INCLUDE_CHECKPOINTS:-1}
POLICY_ROOT=${BSK_RL_POLICY_ROOT:-/scratch/alpine/$USER/rllib_results}
AUDIT_ROOT=${BSK_RL_POLICY_AUDIT_ROOT:-/scratch/alpine/$USER/amos2026_policy_audits}
STAMP=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
NAME="mixed_v9_policy_audit_${STAMP}"
OUTPUT_DIR="$AUDIT_ROOT/$NAME"
ARCHIVE="$AUDIT_ROOT/${NAME}.tgz"
LATEST_ARCHIVE="$AUDIT_ROOT/mixed_v9_policy_audit_latest.tgz"

cd /projects/$USER/bsk_rl
source /projects/$USER/.venv/bin/activate
mkdir -p "$AUDIT_ROOT"

args=(
    --policy-root "$POLICY_ROOT"
    --output-dir "$OUTPUT_DIR"
)
if [[ "$INCLUDE_CHECKPOINTS" == "1" ]]; then
    args+=(--copy-fixed100-checkpoints)
fi

python3 examples/amos_2026/audit_mixed_v9_training_runs.py "${args[@]}"
tar -C "$AUDIT_ROOT" -czf "$ARCHIVE" "$NAME"
ln -f "$ARCHIVE" "$LATEST_ARCHIVE"

echo
echo "Mixed-V9 audit complete."
echo "  output:  $OUTPUT_DIR"
echo "  archive: $ARCHIVE"
echo "  latest:  $LATEST_ARCHIVE"
