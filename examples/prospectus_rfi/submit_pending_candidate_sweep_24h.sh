#!/usr/bin/env bash
# Migrate only pending indexes 1..5 from a cpu-long array to restartable
# cpu-normal segments. The running index 0 is inspected and never modified.
set -euo pipefail

ORIGINAL_JOB_ID=${1:?Usage: $0 ORIGINAL_ARRAY_JOB_ID}
REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
cd "$REPO_DIR"
if [[ "$(git branch --show-current)" != "amos2025-architecture-comparison" ]]; then
    echo "Wrong branch: $(git branch --show-current)" >&2
    exit 2
fi
if [[ ! "$ORIGINAL_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Original job ID must be numeric: $ORIGINAL_JOB_ID" >&2
    exit 3
fi
if [[ ! -s "${BSK_RL_WANDB_KEY_PATH:-/projects/$USER/bsk_rl/examples/wandb_key.txt}" ]]; then
    echo "W&B key file is missing or empty" >&2
    exit 6
fi

RUNNING_TASK="${ORIGINAL_JOB_ID}_0"
RUNNING_STATE=$(squeue -h -j "$RUNNING_TASK" -o '%T')
if [[ "$RUNNING_STATE" != "RUNNING" ]]; then
    echo "Expected $RUNNING_TASK to be RUNNING, found ${RUNNING_STATE:-missing}" >&2
    exit 4
fi

PENDING_TASKS=()
for INDEX in 1 2 3 4 5; do
    TASK_ID="${ORIGINAL_JOB_ID}_${INDEX}"
    STATE=$(squeue -h -j "$TASK_ID" -o '%T')
    if [[ "$STATE" != "PENDING" ]]; then
        echo "Expected $TASK_ID to be PENDING, found ${STATE:-missing}" >&2
        exit 5
    fi
    PENDING_TASKS+=("$TASK_ID")
done

OUTPUT_ROOT=/scratch/alpine/$USER/prospectus_rfi/training
for RUN_NAME in \
    mlp_k10_seed10001 \
    mlp_k20_seed10001 \
    attention_k5_seed10001 \
    attention_k10_seed10001 \
    attention_k20_seed10001; do
    if [[ -e "$OUTPUT_ROOT/$RUN_NAME/training_metrics.csv" || \
          -e "$OUTPUT_ROOT/$RUN_NAME/checkpoints/final" ]]; then
        echo "Replacement output already exists: $OUTPUT_ROOT/$RUN_NAME" >&2
        exit 7
    fi
done

# Validate the replacement resource request before changing any live job state.
sbatch --test-only \
    --array=1 \
    --export=ALL,BSK_RL_REPO_DIR,BSK_RL_WANDB_KEY_PATH,BSK_RL_SEGMENT_INDEX=0 \
    examples/prospectus_rfi/slurm/train_candidate_sweep_segment_24h.sbatch

NEW_JOBS=()
SEGMENT0_JOBS=()
HELD_TASKS=()
rollback() {
    local job_id task_id
    if (( ${#NEW_JOBS[@]} > 0 )); then
        for job_id in "${NEW_JOBS[@]}"; do
            scancel "$job_id" 2>/dev/null || true
        done
    fi
    if (( ${#HELD_TASKS[@]} > 0 )); then
        for task_id in "${HELD_TASKS[@]}"; do
            scontrol release "$task_id" 2>/dev/null || true
        done
    fi
}
trap rollback ERR INT TERM

# Holding first closes the race in which an original pending task starts while
# its replacement chain is being submitted. Index 0 is deliberately absent.
for TASK_ID in "${PENDING_TASKS[@]}"; do
    scontrol hold "$TASK_ID"
    HELD_TASKS+=("$TASK_ID")
done

MANIFEST_ROOT=/scratch/alpine/$USER/prospectus_rfi/manifests
CAMPAIGN_ID=$(date -u +%Y%m%dT%H%M%SZ)
MANIFEST_DIR="$MANIFEST_ROOT/candidate_sweep_24h_${CAMPAIGN_ID}"
mkdir -p "$MANIFEST_DIR" "/scratch/alpine/$USER/job_output"
JOBS_TSV="$MANIFEST_DIR/jobs.tsv"
printf 'array_index\tsegment0_job\tsegment1_job\tcleanup_job\n' > "$JOBS_TSV"

for INDEX in 1 2 3 4 5; do
    SEGMENT0=$(sbatch --parsable \
        --hold \
        --array="$INDEX" \
        --export=ALL,BSK_RL_REPO_DIR,BSK_RL_WANDB_KEY_PATH,BSK_RL_SEGMENT_INDEX=0 \
        examples/prospectus_rfi/slurm/train_candidate_sweep_segment_24h.sbatch)
    NEW_JOBS+=("$SEGMENT0")
    SEGMENT0_JOBS+=("$SEGMENT0")
    SEGMENT1=$(sbatch --parsable \
        --array="$INDEX" \
        --dependency="afterok:$SEGMENT0" \
        --export=ALL,BSK_RL_REPO_DIR,BSK_RL_WANDB_KEY_PATH,BSK_RL_SEGMENT_INDEX=1 \
        examples/prospectus_rfi/slurm/train_candidate_sweep_segment_24h.sbatch)
    NEW_JOBS+=("$SEGMENT1")
    CLEANUP=$(sbatch --parsable \
        --array="$INDEX" \
        --dependency="afterok:$SEGMENT1" \
        --time=02:00:00 \
        --export=ALL,BSK_RL_REPO_DIR,BSK_RL_WANDB_KEY_PATH,BSK_RL_SEGMENT_INDEX=2 \
        examples/prospectus_rfi/slurm/train_candidate_sweep_segment_24h.sbatch)
    NEW_JOBS+=("$CLEANUP")
    printf '%s\t%s\t%s\t%s\n' \
        "$INDEX" "$SEGMENT0" "$SEGMENT1" "$CLEANUP" >> "$JOBS_TSV"
done

cat > "$MANIFEST_DIR/metadata.json" <<EOF
{
  "branch": "amos2025-architecture-comparison",
  "commit": "$(git rev-parse HEAD)",
  "original_array_job": "$ORIGINAL_JOB_ID",
  "preserved_running_task": "$RUNNING_TASK",
  "replaced_pending_indexes": [1, 2, 3, 4, 5],
  "qos": "cpu-normal",
  "partition": "acpu",
  "constraint": "epyc-7713",
  "target_training_wall_hours": 48.0,
  "segment_training_caps_hours": [23.5, 23.5, 1.5],
  "wandb_project": "amos2025-architecture-comparison"
}
EOF

# Replacements now exist, so remove only the five held/pending originals. The
# original running index 0 is neither held nor cancelled by this script.
for TASK_ID in "${PENDING_TASKS[@]}"; do
    scancel "$TASK_ID"
done
HELD_TASKS=()
trap - ERR INT TERM

RELEASE_FAILURE=0
for JOB_ID in "${SEGMENT0_JOBS[@]}"; do
    if ! scontrol release "$JOB_ID"; then
        echo "Could not release replacement job $JOB_ID; release it manually" >&2
        RELEASE_FAILURE=1
    fi
done

echo "Preserved running task: $RUNNING_TASK"
echo "Replaced only pending tasks: ${PENDING_TASKS[*]}"
echo "Manifest: $MANIFEST_DIR"
cat "$JOBS_TSV"
exit "$RELEASE_FAILURE"
