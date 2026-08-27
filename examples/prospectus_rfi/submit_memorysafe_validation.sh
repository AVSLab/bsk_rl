#!/usr/bin/env bash
# Submit only missing per-episode validation work, followed by a collector.
set -euo pipefail

REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
ROOT=${BSK_RL_RFI_EVALUATION_ROOT:-/scratch/alpine/$USER/prospectus_rfi/memorysafe_100_200_v2}
BASE_CONFIG=${BSK_RL_RFI_EVALUATION_BASE_CONFIG:-examples/prospectus_rfi/configs/base_memorysafe_100_200.yaml}
MAX_CONCURRENT=${1:-20}
if ! [[ "$MAX_CONCURRENT" =~ ^([1-9]|[1-5][0-9]|60)$ ]]; then
  echo "max concurrent must be an integer in 1..60" >&2
  exit 2
fi

cd "$REPO_DIR"
if [[ "$(git branch --show-current)" != "amos2025-architecture-comparison" ]]; then
  echo "Wrong branch: $(git branch --show-current)" >&2
  exit 3
fi
PYTHON=${BSK_RL_VENV_ROOT:-/projects/$USER/.venv}/bin/python
MANIFEST="$ROOT/validation/manifest.json"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
MISSING_COUNT=$("$PYTHON" examples/prospectus_rfi/prepare_memorysafe_validation.py \
  --root "$ROOT" --base-config "$BASE_CONFIG" --manifest "$MANIFEST" --print-missing-count)
if [[ "$MISSING_COUNT" == 0 ]]; then
  echo "All validation episodes already exist; submit collector only."
  TASK_JOB=""
  COLLECT_JOB=$(sbatch --parsable \
    --export="ALL,BSK_RL_REPO_DIR=$REPO_DIR,BSK_RL_VALIDATION_MANIFEST=$MANIFEST" \
    examples/prospectus_rfi/slurm/collect_memorysafe_validation.sbatch)
else
  mkdir -p "/scratch/alpine/$USER/job_output"
  SUBMISSION_ID=$(date -u +%Y%m%dT%H%M%SZ)
  SHARD_DIR="$ROOT/validation/task_maps/$SUBMISSION_ID"
  SHARDS=$("$PYTHON" examples/prospectus_rfi/prepare_memorysafe_validation.py \
    --root "$ROOT" --base-config "$BASE_CONFIG" --manifest "$MANIFEST" \
    --write-shards "$SHARD_DIR" --shard-size 400)
  TASK_JOBS=()
  while IFS=$'\t' read -r TASK_MAP TASK_COUNT; do
    [[ -n "$TASK_MAP" ]] || continue
    ARRAY_MAX=$((TASK_COUNT - 1))
    EXPORTS="ALL,BSK_RL_REPO_DIR=$REPO_DIR,BSK_RL_VALIDATION_MANIFEST=$MANIFEST,BSK_RL_VALIDATION_TASK_MAP=$TASK_MAP"
    sbatch --test-only --array="0-${ARRAY_MAX}%${MAX_CONCURRENT}" --export="$EXPORTS" \
      examples/prospectus_rfi/slurm/validate_memorysafe_task.sbatch
    TASK_JOBS+=("$(sbatch --parsable --array="0-${ARRAY_MAX}%${MAX_CONCURRENT}" --export="$EXPORTS" \
      examples/prospectus_rfi/slurm/validate_memorysafe_task.sbatch)")
  done <<< "$SHARDS"
  TASK_JOB=$(IFS=,; echo "${TASK_JOBS[*]}")
  DEPENDENCY=$(IFS=:; echo "${TASK_JOBS[*]}")
  COLLECT_EXPORTS="ALL,BSK_RL_REPO_DIR=$REPO_DIR,BSK_RL_VALIDATION_MANIFEST=$MANIFEST"
  COLLECT_JOB=$(sbatch --parsable --dependency="afterok:$DEPENDENCY" --export="$COLLECT_EXPORTS" \
    examples/prospectus_rfi/slurm/collect_memorysafe_validation.sbatch)
fi
echo "VALIDATION_TASK_JOB=$TASK_JOB"
echo "VALIDATION_COLLECT_JOB=$COLLECT_JOB"
echo "MANIFEST=$MANIFEST"
echo "MISSING_TASKS=$MISSING_COUNT"
