#!/usr/bin/env bash
# Replay the accepted campaign with acquisition histories, then analyze them.
set -euo pipefail

CAMPAIGN_ROOT=${1:-}
MAX_CONCURRENT=${2:-30}
if [[ -z "$CAMPAIGN_ROOT" ]]; then
    echo "Usage: $0 <completed-campaign-root> [max-concurrent: 1..60]" >&2
    exit 2
fi
if ! [[ "$MAX_CONCURRENT" =~ ^([1-9]|[1-5][0-9]|60)$ ]]; then
    echo "max-concurrent must be in 1..60" >&2
    exit 3
fi

REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
PYTHON=${BSK_RL_VENV_ROOT:-/projects/$USER/.venv}/bin/python
CAMPAIGN_ROOT=$(cd "$CAMPAIGN_ROOT" && pwd)
MANIFEST="$CAMPAIGN_ROOT/manifests/submission.json"
if [[ ! -f "$CAMPAIGN_ROOT/analysis/completion.json" ]]; then
    echo "Completed collector gate is missing: $CAMPAIGN_ROOT/analysis/completion.json" >&2
    exit 4
fi
if [[ ! -f "$MANIFEST" ]]; then
    echo "Campaign manifest is missing: $MANIFEST" >&2
    exit 5
fi
cd "$REPO_DIR"

mapfile -t CHECKPOINTS < <(
    "$PYTHON" - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print(payload["legacy_checkpoint"])
print(payload.get("attention_checkpoint") or str(Path(payload["attention_run_dir"]) / "checkpoints" / "final"))
PY
)
LEGACY_CHECKPOINT=${CHECKPOINTS[0]}
ATTENTION_CHECKPOINT=${CHECKPOINTS[1]}
[[ -d "$LEGACY_CHECKPOINT" ]] || { echo "Missing $LEGACY_CHECKPOINT" >&2; exit 6; }
[[ -d "$ATTENTION_CHECKPOINT" ]] || { echo "Missing $ATTENTION_CHECKPOINT" >&2; exit 7; }

MISSING_OUTPUT=$(
	"$PYTHON" - "$CAMPAIGN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
methods = (
    "breckenridge2026_alpha0_mlp",
    "target_set_attention",
    "smallest_angle_heuristic",
    "closest_distance_heuristic",
)
for method_index, method in enumerate(methods):
    for seed in range(100):
        task_id = method_index * 100 + seed
        stem = f"{method}_seed{seed:03d}"
        accepted = root / "raw" / method / f"{stem}.csv"
        timeline = root / "timeline" / "raw" / method / f"{stem}.timeline.csv"
        metadata = timeline.with_suffix(".metadata.json")
        if not accepted.is_file():
            raise SystemExit(f"accepted raw episode is missing: {accepted}")
        complete = timeline.is_file() and metadata.is_file()
        if complete:
            try:
                payload = json.loads(metadata.read_text())
            except json.JSONDecodeError:
                payload = {}
            final_metrics = payload.get("final_replay_metrics", {})
            header = timeline.open().readline()
            complete = (
                "illuminated_target_selection_count" in final_metrics
                and "illuminated_target_selection_count" in header
            )
        if not complete:
            print(task_id)
PY
)
MISSING_IDS=()
if [[ -n "$MISSING_OUTPUT" ]]; then
    while IFS= read -r TASK_ID; do
        [[ -n "$TASK_ID" ]] && MISSING_IDS+=("$TASK_ID")
    done <<< "$MISSING_OUTPUT"
fi

export BSK_RL_REPO_DIR="$REPO_DIR"
export BSK_RL_AMOS2025_MATCHED_300S_OUTPUT_ROOT="$CAMPAIGN_ROOT"
export BSK_RL_BRECKENRIDGE_ALPHA0_CHECKPOINT="$LEGACY_CHECKPOINT"
export BSK_RL_AMOS2025_ATTENTION_CHECKPOINT="$ATTENTION_CHECKPOINT"
EXPORTS=ALL,BSK_RL_REPO_DIR,BSK_RL_AMOS2025_MATCHED_300S_OUTPUT_ROOT,BSK_RL_BRECKENRIDGE_ALPHA0_CHECKPOINT,BSK_RL_AMOS2025_ATTENTION_CHECKPOINT

if [[ ${#MISSING_IDS[@]} -gt 0 ]]; then
    ARRAY_IDS=$(IFS=,; echo "${MISSING_IDS[*]}")
    TIMELINE_JOB=$(sbatch --parsable \
        --array="${ARRAY_IDS}%${MAX_CONCURRENT}" \
        --export="$EXPORTS" \
        examples/prospectus_rfi/slurm/record_amos2025_matched_300s_timelines.sbatch)
    ANALYSIS_JOB=$(sbatch --parsable --dependency="afterok:$TIMELINE_JOB" \
        --export="$EXPORTS" \
        examples/prospectus_rfi/slurm/analyze_amos2025_matched_300s_timelines.sbatch)
else
    TIMELINE_JOB=none
    ANALYSIS_JOB=$(sbatch --parsable --export="$EXPORTS" \
        examples/prospectus_rfi/slurm/analyze_amos2025_matched_300s_timelines.sbatch)
fi

echo "Submitted matched 300-second acquisition timeline analysis."
echo "TIMELINE_JOB=$TIMELINE_JOB"
echo "ANALYSIS_JOB=$ANALYSIS_JOB"
echo "MISSING_TIMELINES=${#MISSING_IDS[@]}"
echo "CAMPAIGN_ROOT=$CAMPAIGN_ROOT"
