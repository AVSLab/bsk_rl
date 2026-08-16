#!/usr/bin/env bash
set -euo pipefail

MAX_CONCURRENT=${1:-30}
if ! [[ "$MAX_CONCURRENT" =~ ^([1-9]|[1-5][0-9]|60)$ ]]; then
    echo "Usage: $0 [max-concurrent: 1..60, default 30]" >&2
    exit 2
fi

REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
CHECKPOINT=${BSK_RL_AMOS2025_POLICY_CHECKPOINT:-}
if [[ -z "$CHECKPOINT" ]]; then
    echo "Set BSK_RL_AMOS2025_POLICY_CHECKPOINT to the inspector module directory." >&2
    exit 3
fi
cd "$REPO_DIR"
if [[ "$(git branch --show-current)" != "amos2025-architecture-comparison" ]]; then
    echo "Wrong branch: $(git branch --show-current)" >&2
    exit 4
fi

mapfile -t STATE_FILES < <(find "$CHECKPOINT" -type f -name module_state.pt -print)
if [[ ${#STATE_FILES[@]} -ne 1 ]]; then
    echo "Expected exactly one module_state.pt under $CHECKPOINT; found ${#STATE_FILES[@]}" >&2
    exit 5
fi
EXPECTED_SHA256=6db5bcd4fda20205977dfab377441f625051ef9e9dfaebde5e8db5ec1ab0e2c4
ACTUAL_SHA256=$(sha256sum "${STATE_FILES[0]}" | awk '{print $1}')
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
    echo "Wrong AMOS 2025 module_state.pt checksum: $ACTUAL_SHA256" >&2
    echo "Expected: $EXPECTED_SHA256" >&2
    exit 6
fi
for REQUIRED in class_and_ctor_args.pkl metadata.json; do
    if [[ ! -s "$(dirname "${STATE_FILES[0]}")/$REQUIRED" ]]; then
        echo "Missing module artifact file: $(dirname "${STATE_FILES[0]}")/$REQUIRED" >&2
        exit 7
    fi
done

CAMPAIGN_ID=${BSK_RL_LEGACY_POLICY_MC_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${BSK_RL_LEGACY_POLICY_MC_OUTPUT_ROOT:-/scratch/alpine/$USER/prospectus_rfi/legacy_policy_mc/amos2025_alpha0_300s_to_100s_${CAMPAIGN_ID}}
MANIFEST_DIR="$OUTPUT_ROOT/manifests"
mkdir -p "$MANIFEST_DIR" "/scratch/alpine/$USER/job_output"

COMMIT=$(git rev-parse HEAD)
python - "$MANIFEST_DIR/campaign.json" <<PY
import json
import sys

payload = {
    "campaign_id": "$CAMPAIGN_ID",
    "campaign_name": "amos2025_alpha0_policy_transfer_300s_to_100s",
    "branch": "amos2025-architecture-comparison",
    "commit": "$COMMIT",
    "method": "legacy_amos2025_alpha0_policy",
    "checkpoint": "$CHECKPOINT",
    "module_state_sha256": "$ACTUAL_SHA256",
    "policy_best_iteration": 427,
    "policy_training_imaging_duration_s": 300.0,
    "evaluation_imaging_duration_s": 100.0,
    "catalog_sizes": [100, 200, 400],
    "seeds": {"start": 0, "stop_inclusive": 99},
    "episodes_per_catalog_size": 100,
    "candidate_count": 10,
    "resource_shield": True,
    "array_tasks": 300,
    "episodes_per_task": 1,
    "max_concurrent": int("$MAX_CONCURRENT"),
    "slurm_nice": 10000,
    "dependencies": [],
}
with open(sys.argv[1], "w") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

export BSK_RL_LEGACY_POLICY_MC_OUTPUT_ROOT="$OUTPUT_ROOT"
JOB_ID=$(sbatch --parsable \
    --array="0-299%${MAX_CONCURRENT}" \
    --export=ALL,BSK_RL_REPO_DIR,BSK_RL_AMOS2025_POLICY_CHECKPOINT,BSK_RL_LEGACY_POLICY_MC_OUTPUT_ROOT \
    examples/prospectus_rfi/slurm/evaluate_amos2025_legacy_policy_mc.sbatch)

printf '%s\n' "$JOB_ID" > "$MANIFEST_DIR/slurm_job_id.txt"
printf '%s\n' "$OUTPUT_ROOT" > "$MANIFEST_DIR/output_root.txt"
echo "Submitted dependency-free frozen AMOS 2025 policy MC at low scheduler priority."
echo "JOB_ID=$JOB_ID"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "TASKS=300 MAX_CONCURRENT=$MAX_CONCURRENT EPISODES=300"
echo "CHECKPOINT_SHA256=$ACTUAL_SHA256"
