#!/usr/bin/env bash
# Historical plan only. Submission is retired after the reference pair failed.
# Existing sealed archives retain their original bytes and are not approved.
set -euo pipefail
BSK_SWEEP_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  '') exec python3 "$BSK_SWEEP_SCRIPT_DIR/campaign.py" plan ;;
  --submit) exec python3 "$BSK_SWEEP_SCRIPT_DIR/campaign.py" submit ;;
  *) echo 'Usage: submit-alpha-sweep.sh [--submit]' >&2; exit 2 ;;
esac
