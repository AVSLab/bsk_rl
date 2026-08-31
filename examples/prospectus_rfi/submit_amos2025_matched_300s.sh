#!/usr/bin/env bash
# Select the attention checkpoint, run 400 paired episodes, then validate/collect.
set -euo pipefail

MAX_CONCURRENT=${1:-30}
if ! [[ "$MAX_CONCURRENT" =~ ^([1-9]|[1-5][0-9]|60)$ ]]; then
    echo "Usage: $0 [max-concurrent: 1..60, default 30]" >&2
    exit 2
fi

REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
LEGACY_CHECKPOINT=${BSK_RL_BRECKENRIDGE_ALPHA0_CHECKPOINT:-/projects/$USER/bsk_rl/policies/breckenridge2026_alpha_sweep/0d100i/checkpoint_000145}
ATTENTION_RUN_DIR=${BSK_RL_AMOS2025_ATTENTION_RUN_DIR:-}
if [[ -z "$ATTENTION_RUN_DIR" ]]; then
    ATTENTION_RUN_DIR=$(python - <<'PY'
import json
import os
from pathlib import Path

root = (
    Path("/scratch/alpine")
    / os.environ["USER"]
    / "prospectus_rfi"
    / "amos2025_attention_control_300s"
)
candidates = []
for status_path in root.glob("*/training/attention_k10_seed10001/status.json"):
    try:
        status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    run_dir = status_path.parent
    if (
        status.get("state") == "target_reached"
        and (run_dir / "checkpoints" / "final").is_dir()
    ):
        candidates.append(run_dir)
if not candidates:
    raise SystemExit(
        "No completed target_reached 300-second attention run was found under "
        f"{root}"
    )
print(max(candidates, key=lambda path: path.stat().st_mtime))
PY
    )
fi
if [[ ! -d "$LEGACY_CHECKPOINT" ]]; then
    LEGACY_CHECKPOINT=$(find "/projects/$USER" -type d \
        -path '*/policies/breckenridge2026_alpha_sweep/0d100i/checkpoint_000145' \
        -print -quit 2>/dev/null)
fi
cd "$REPO_DIR"
if [[ "$(git branch --show-current)" != "amos2025-architecture-comparison" ]]; then
    echo "Wrong branch: $(git branch --show-current)" >&2
    exit 4
fi
if [[ ! -d "$ATTENTION_RUN_DIR/checkpoints/final" ]]; then
    echo "Missing final attention checkpoint under $ATTENTION_RUN_DIR" >&2
    exit 5
fi
if [[ ! -d "$LEGACY_CHECKPOINT" ]]; then
    echo "Missing Breckenridge alpha=0 checkpoint: $LEGACY_CHECKPOINT" >&2
    exit 6
fi

echo "ATTENTION_RUN_DIR=$ATTENTION_RUN_DIR"
echo "BRECKENRIDGE_ALPHA0_CHECKPOINT=$LEGACY_CHECKPOINT"

mapfile -t LEGACY_STATES < <(find "$LEGACY_CHECKPOINT" -type f -name module_state.pt -print)
if [[ ${#LEGACY_STATES[@]} -ne 1 ]]; then
    echo "Expected one archived MLP module_state.pt; found ${#LEGACY_STATES[@]}" >&2
    exit 7
fi
EXPECTED_SHA256=0d8033272f14cdd408192d7ab6ee819b18691c9385fca87be24044fc950464d2
ACTUAL_SHA256=$(sha256sum "${LEGACY_STATES[0]}" | awk '{print $1}')
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
    echo "Wrong Breckenridge 2026 alpha=0 MLP checksum: $ACTUAL_SHA256" >&2
    exit 8
fi

CAMPAIGN_ID=${BSK_RL_AMOS2025_MATCHED_300S_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${BSK_RL_AMOS2025_MATCHED_300S_OUTPUT_ROOT:-/scratch/alpine/$USER/prospectus_rfi/amos2025_matched_300s/$CAMPAIGN_ID}
MANIFEST_DIR="$OUTPUT_ROOT/manifests"
mkdir -p "$MANIFEST_DIR" "/scratch/alpine/$USER/job_output"

BASE_EXPORTS="ALL,BSK_RL_REPO_DIR,BSK_RL_BRECKENRIDGE_ALPHA0_CHECKPOINT=$LEGACY_CHECKPOINT,BSK_RL_AMOS2025_ATTENTION_RUN_DIR=$ATTENTION_RUN_DIR,BSK_RL_AMOS2025_MATCHED_300S_OUTPUT_ROOT=$OUTPUT_ROOT"
sbatch --test-only --export="$BASE_EXPORTS" \
    examples/prospectus_rfi/slurm/validate_amos2025_attention_control.sbatch
sbatch --test-only --array="0-399%${MAX_CONCURRENT}" \
    --export="$BASE_EXPORTS,BSK_RL_AMOS2025_ATTENTION_CHECKPOINT=$ATTENTION_RUN_DIR/checkpoints/best_validation" \
    examples/prospectus_rfi/slurm/evaluate_amos2025_matched_300s.sbatch

VALIDATION_JOB=$(sbatch --parsable --export="$BASE_EXPORTS" \
    examples/prospectus_rfi/slurm/validate_amos2025_attention_control.sbatch)
MC_JOB=$(sbatch --parsable --dependency="afterok:$VALIDATION_JOB" \
    --array="0-399%${MAX_CONCURRENT}" \
    --export="$BASE_EXPORTS,BSK_RL_AMOS2025_ATTENTION_CHECKPOINT=$ATTENTION_RUN_DIR/checkpoints/best_validation" \
    examples/prospectus_rfi/slurm/evaluate_amos2025_matched_300s.sbatch)
COLLECTOR_JOB=$(sbatch --parsable --dependency="afterok:$MC_JOB" \
    --export="$BASE_EXPORTS" \
    examples/prospectus_rfi/slurm/collect_amos2025_matched_300s.sbatch)

COMMIT=$(git rev-parse HEAD)
python - "$MANIFEST_DIR/submission.json" <<PY
import json
import sys

payload = {
    "campaign_id": "$CAMPAIGN_ID",
    "branch": "amos2025-architecture-comparison",
    "commit": "$COMMIT",
    "legacy_checkpoint": "$LEGACY_CHECKPOINT",
    "legacy_module_state_sha256": "$ACTUAL_SHA256",
    "attention_run_dir": "$ATTENTION_RUN_DIR",
    "attention_selection": "maximum held-out physical validation score over seeds 91001..91005",
    "methods": [
        "breckenridge2026_alpha0_mlp",
        "target_set_attention",
        "smallest_angle_heuristic",
        "closest_distance_heuristic",
    ],
    "conditions": {
        "catalog_size": 100,
        "candidate_count": 10,
        "episode_duration_s": 45000,
        "imaging_charge_downlink_desaturation_s": [300, 300, 180, 150],
        "initial_battery_fraction": [0.10, 0.40],
        "reward_alpha": 0.0,
        "resource_shield": True,
        "wheel_guard": False,
        "seeds": [0, 99],
    },
    "validation_job": "$VALIDATION_JOB",
    "monte_carlo_job": "$MC_JOB",
    "collector_job": "$COLLECTOR_JOB",
    "max_concurrent": int("$MAX_CONCURRENT"),
}
with open(sys.argv[1], "w") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

echo "Submitted matched AMOS 2025 300-second Research Focus I comparison."
echo "VALIDATION_JOB=$VALIDATION_JOB"
echo "MC_JOB=$MC_JOB"
echo "COLLECTOR_JOB=$COLLECTOR_JOB"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
