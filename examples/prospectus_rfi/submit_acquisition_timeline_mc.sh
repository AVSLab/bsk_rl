#!/usr/bin/env bash
set -euo pipefail

HEURISTIC_ROOT=${1:-}
POLICY_ROOT=${2:-}
MAX_CONCURRENT=${3:-30}
if [[ -z "$HEURISTIC_ROOT" || -z "$POLICY_ROOT" ]]; then
    echo "Usage: $0 <heuristic-root> <legacy-policy-root> [max-concurrent: 1..60]" >&2
    exit 2
fi
if ! [[ "$MAX_CONCURRENT" =~ ^([1-9]|[1-5][0-9]|60)$ ]]; then
    echo "max-concurrent must be in 1..60" >&2
    exit 3
fi

REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
CHECKPOINT=${BSK_RL_AMOS2025_POLICY_CHECKPOINT:-}
VENV_ROOT=${BSK_RL_VENV_ROOT:-/projects/$USER/.venv}
PYTHON="$VENV_ROOT/bin/python"
if [[ -z "$CHECKPOINT" ]]; then
    echo "Set BSK_RL_AMOS2025_POLICY_CHECKPOINT." >&2
    exit 4
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "Required Python environment not found: $PYTHON" >&2
    exit 5
fi
cd "$REPO_DIR"
if [[ "$(git branch --show-current)" != "amos2025-architecture-comparison" ]]; then
    echo "Wrong branch: $(git branch --show-current)" >&2
    exit 6
fi
for ROOT in "$HEURISTIC_ROOT" "$POLICY_ROOT"; do
    if [[ ! -d "$ROOT/raw" ]]; then
        echo "Completed campaign raw directory is missing: $ROOT/raw" >&2
        exit 7
    fi
done

EXPECTED_SHA256=6db5bcd4fda20205977dfab377441f625051ef9e9dfaebde5e8db5ec1ab0e2c4
STATE_PATH="$CHECKPOINT/module_state.pt"
if [[ ! -s "$STATE_PATH" ]]; then
    echo "Policy state is missing: $STATE_PATH" >&2
    exit 8
fi
ACTUAL_SHA256=$(sha256sum "$STATE_PATH" | awk '{print $1}')
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
    echo "Wrong AMOS 2025 policy checksum: $ACTUAL_SHA256" >&2
    exit 9
fi

MISSING_OUTPUT=$(
    "$PYTHON" - "$HEURISTIC_ROOT" "$POLICY_ROOT" <<'PY'
import sys
from pathlib import Path

roots = {
    "heuristic_historical": Path(sys.argv[1]),
    "legacy_amos2025_alpha0_policy": Path(sys.argv[2]),
}
methods = tuple(roots)
catalog_sizes = (100, 200, 400)
for method_index, method in enumerate(methods):
    root = roots[method]
    for catalog_index, catalog_size in enumerate(catalog_sizes):
        for seed in range(100):
            task_id = method_index * 300 + catalog_index * 100 + seed
            stem = f"{method}_n{catalog_size}_seed{seed:03d}"
            accepted = root / "raw" / f"n{catalog_size}" / f"{stem}.csv"
            timeline = (
                root / "timeline" / "raw" / f"n{catalog_size}"
                / f"{stem}.timeline.csv"
            )
            metadata = timeline.with_suffix(".metadata.json")
            if not accepted.is_file():
                raise SystemExit(f"accepted raw episode is missing: {accepted}")
            if not (timeline.is_file() and metadata.is_file()):
                print(task_id)
PY
)
MISSING_IDS=()
if [[ -n "$MISSING_OUTPUT" ]]; then
    while IFS= read -r TASK_ID; do
        [[ -n "$TASK_ID" ]] && MISSING_IDS+=("$TASK_ID")
    done <<< "$MISSING_OUTPUT"
fi
if [[ ${#MISSING_IDS[@]} -eq 0 ]]; then
    echo "All 600 acquisition timelines already exist; nothing to submit."
    exit 0
fi
ARRAY_IDS=$(IFS=,; echo "${MISSING_IDS[*]}")
if [[ ${BSK_RL_TIMELINE_SCAN_ONLY:-0} == 1 ]]; then
    echo "MISSING_TIMELINES=${#MISSING_IDS[@]}"
    echo "MISSING_TASK_IDS=$ARRAY_IDS"
    exit 0
fi

REPLAY_ID=$(date -u +%Y%m%dT%H%M%SZ)
MANIFEST_DIR="$POLICY_ROOT/timeline/manifests"
MANIFEST="$MANIFEST_DIR/replay_${REPLAY_ID}.json"
mkdir -p "$MANIFEST_DIR" "/scratch/alpine/$USER/job_output"
COMMIT=$(git rev-parse HEAD)
"$PYTHON" - "$MANIFEST" "$ARRAY_IDS" <<PY
import json
import sys

task_ids = [int(value) for value in sys.argv[2].split(",")]
payload = {
    "replay_id": "$REPLAY_ID",
    "branch": "amos2025-architecture-comparison",
    "commit": "$COMMIT",
    "heuristic_root": "$HEURISTIC_ROOT",
    "legacy_policy_root": "$POLICY_ROOT",
    "policy_checkpoint": "$CHECKPOINT",
    "policy_module_state_sha256": "$ACTUAL_SHA256",
    "methods": ["heuristic_historical", "legacy_amos2025_alpha0_policy"],
    "catalog_sizes": [100, 200, 400],
    "seeds": {"start": 0, "stop_inclusive": 99},
    "recording": "every decision epoch",
    "analysis_grid_interval_s": 100.0,
    "table_checkpoints_s": [15000.0, 30000.0, 45000.0],
    "missing_task_ids": task_ids,
    "missing_timeline_count": len(task_ids),
    "max_concurrent": int("$MAX_CONCURRENT"),
    "slurm_nice": 10000,
    "dependencies": [],
    "raw_episode_overwrite": False,
}
with open(sys.argv[1], "w") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

export BSK_RL_HEURISTIC_MC_OUTPUT_ROOT="$HEURISTIC_ROOT"
export BSK_RL_LEGACY_POLICY_MC_OUTPUT_ROOT="$POLICY_ROOT"
JOB_ID=$(sbatch --parsable \
    --array="${ARRAY_IDS}%${MAX_CONCURRENT}" \
    --export=ALL,BSK_RL_REPO_DIR,BSK_RL_AMOS2025_POLICY_CHECKPOINT,BSK_RL_HEURISTIC_MC_OUTPUT_ROOT,BSK_RL_LEGACY_POLICY_MC_OUTPUT_ROOT \
    examples/prospectus_rfi/slurm/record_acquisition_timelines.sbatch)

"$PYTHON" - "$MANIFEST" "$JOB_ID" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["slurm_job_id"] = sys.argv[2]
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo "Submitted dependency-free, low-priority acquisition timeline replays."
echo "JOB_ID=$JOB_ID"
echo "MISSING_TIMELINES=${#MISSING_IDS[@]}"
echo "MAX_CONCURRENT=$MAX_CONCURRENT"
echo "HEURISTIC_ROOT=$HEURISTIC_ROOT"
echo "POLICY_ROOT=$POLICY_ROOT"
echo "DEPENDENCIES=none"
echo "RAW_EPISODE_OVERWRITE=false"
