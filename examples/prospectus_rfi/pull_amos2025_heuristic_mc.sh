#!/usr/bin/env bash
# Run this on the local workstation, not on Alpine.
set -euo pipefail

CAMPAIGN_NAME=${1:-amos2025_closest_angle_100s_20260815T183838Z}
REMOTE_LOGIN=${BSK_RL_ALPINE_LOGIN:-dahu1128@login.rc.colorado.edu}
REMOTE_ROOT=${BSK_RL_HEURISTIC_MC_REMOTE_ROOT:-/scratch/alpine/dahu1128/prospectus_rfi/heuristic_mc/$CAMPAIGN_NAME}

REPO_DIR=$(git rev-parse --show-toplevel)
LOCAL_PARENT=${BSK_RL_LOCAL_RESULTS_ROOT:-$REPO_DIR/results/prospectus_rfi/cluster_downloads}
LOCAL_ROOT="$LOCAL_PARENT/heuristic_mc/$CAMPAIGN_NAME"
PYTHON=${BSK_RL_LOCAL_PYTHON:-$REPO_DIR/.venv/bin/python}

if [[ ! -x "$PYTHON" ]]; then
    echo "Local study Python is not executable: $PYTHON" >&2
    echo "Set BSK_RL_LOCAL_PYTHON to a Python with the study dependencies." >&2
    exit 2
fi

mkdir -p "$LOCAL_ROOT"
echo "Pulling $REMOTE_LOGIN:$REMOTE_ROOT/"
echo "Into    $LOCAL_ROOT/"
rsync -av --partial --checksum \
    "$REMOTE_LOGIN:$REMOTE_ROOT/" \
    "$LOCAL_ROOT/"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" examples/prospectus_rfi/collect_heuristic_mc.py \
    --input-root "$LOCAL_ROOT"

echo "Validated local heuristic campaign:"
echo "$LOCAL_ROOT"
echo "Combined episodes: $LOCAL_ROOT/analysis/episodes_combined.csv"
echo "Summary statistics: $LOCAL_ROOT/analysis/summary_statistics.csv"
