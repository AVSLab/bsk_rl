#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT=${1:-}
MAX_CONCURRENT=${2:-30}
if [[ -z "$OUTPUT_ROOT" ]]; then
    echo "Usage: $0 <existing-output-root> [max-concurrent: 1..60]" >&2
    exit 2
fi
if ! [[ "$MAX_CONCURRENT" =~ ^([1-9]|[1-5][0-9]|60)$ ]]; then
    echo "max-concurrent must be in 1..60" >&2
    exit 3
fi

REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
export BSK_RL_REPO_DIR="$REPO_DIR"
VENV_ROOT=${BSK_RL_VENV_ROOT:-/projects/$USER/.venv}
PYTHON="$VENV_ROOT/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Required Python environment not found: $PYTHON" >&2
    exit 4
fi
cd "$REPO_DIR"
if [[ "$(git branch --show-current)" != "amos2025-architecture-comparison" ]]; then
    echo "Wrong branch: $(git branch --show-current)" >&2
    exit 5
fi
if [[ ! -s "$OUTPUT_ROOT/manifests/campaign.json" ]]; then
    echo "Not an existing heuristic campaign: $OUTPUT_ROOT" >&2
    exit 6
fi

MISSING_OUTPUT=$(
    "$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
method = "heuristic_historical"
for catalog_index, catalog_size in enumerate((100, 200, 400)):
    for seed in range(100):
        stem = f"{method}_n{catalog_size}_seed{seed:03d}"
        csv_path = root / "raw" / f"n{catalog_size}" / f"{stem}.csv"
        metadata_path = root / "raw" / f"n{catalog_size}" / f"{stem}.metadata.json"
        if not (csv_path.is_file() and metadata_path.is_file()):
            print(catalog_index * 100 + seed)
PY
)
MISSING_IDS=()
if [[ -n "$MISSING_OUTPUT" ]]; then
    while IFS= read -r TASK_ID; do
        [[ -n "$TASK_ID" ]] && MISSING_IDS+=("$TASK_ID")
    done <<< "$MISSING_OUTPUT"
fi

if [[ ${#MISSING_IDS[@]} -eq 0 ]]; then
    echo "Campaign already has all 300 CSV/metadata pairs; nothing to submit."
    exit 0
fi
ARRAY_IDS=$(IFS=,; echo "${MISSING_IDS[*]}")
if [[ ${BSK_RL_RECOVERY_SCAN_ONLY:-0} == 1 ]]; then
    echo "MISSING_EPISODES=${#MISSING_IDS[@]}"
    echo "MISSING_TASK_IDS=$ARRAY_IDS"
    exit 0
fi
RECOVERY_ID=$(date -u +%Y%m%dT%H%M%SZ)
MANIFEST="$OUTPUT_ROOT/manifests/recovery_${RECOVERY_ID}.json"
COMMIT=$(git rev-parse HEAD)
"$PYTHON" - "$MANIFEST" "$ARRAY_IDS" <<PY
import json
import sys

task_ids = [int(value) for value in sys.argv[2].split(",")]
payload = {
    "recovery_id": "$RECOVERY_ID",
    "branch": "amos2025-architecture-comparison",
    "commit": "$COMMIT",
    "existing_output_root": "$OUTPUT_ROOT",
    "missing_task_ids": task_ids,
    "missing_episode_count": len(task_ids),
    "task_mapping": "0..99=N100, 100..199=N200, 200..299=N400",
    "episodes_per_task": 1,
    "max_concurrent": int("$MAX_CONCURRENT"),
    "slurm_nice": 10000,
    "dependencies": [],
}
with open(sys.argv[1], "w") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

export BSK_RL_HEURISTIC_MC_OUTPUT_ROOT="$OUTPUT_ROOT"
JOB_ID=$(sbatch --parsable \
    --array="${ARRAY_IDS}%${MAX_CONCURRENT}" \
    --export=ALL,BSK_RL_REPO_DIR,BSK_RL_HEURISTIC_MC_OUTPUT_ROOT \
    examples/prospectus_rfi/slurm/recover_amos2025_heuristic_mc.sbatch)

"$PYTHON" - "$MANIFEST" "$JOB_ID" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["slurm_job_id"] = sys.argv[2]
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo "Submitted independent recovery tasks for the incomplete heuristic campaign."
echo "JOB_ID=$JOB_ID"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "MISSING_EPISODES=${#MISSING_IDS[@]}"
echo "MAX_CONCURRENT=$MAX_CONCURRENT"
echo "DEPENDENCIES=none"
