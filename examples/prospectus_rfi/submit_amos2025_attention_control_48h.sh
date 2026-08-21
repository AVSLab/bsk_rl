#!/usr/bin/env bash
# Submit a one-iteration gate and one independent 48-hour attention control.
set -euo pipefail

REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
cd "$REPO_DIR"
if [[ "$(git branch --show-current)" != "amos2025-architecture-comparison" ]]; then
    echo "Wrong branch: $(git branch --show-current)" >&2
    exit 2
fi
if [[ ! -s "${BSK_RL_WANDB_KEY_PATH:-/projects/$USER/bsk_rl/examples/wandb_key.txt}" ]]; then
    echo "W&B key file is missing or empty" >&2
    exit 3
fi

CAMPAIGN_ID=${BSK_RL_CONTROL_CAMPAIGN:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT="/scratch/alpine/$USER/prospectus_rfi/amos2025_attention_control_300s/$CAMPAIGN_ID"
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to reuse existing campaign output: $OUTPUT_ROOT" >&2
    exit 4
fi
mkdir -p "/scratch/alpine/$USER/job_output" "$OUTPUT_ROOT/manifests"

EXPORTS="ALL,BSK_RL_REPO_DIR,BSK_RL_WANDB_KEY_PATH,BSK_RL_CONTROL_CAMPAIGN=$CAMPAIGN_ID"
sbatch --test-only --export="$EXPORTS" \
    examples/prospectus_rfi/slurm/stress_amos2025_attention_control.sbatch

SUBMITTED_JOBS=()
rollback() {
    local job_id
    for job_id in "${SUBMITTED_JOBS[@]}"; do
        scancel "$job_id" 2>/dev/null || true
    done
}
trap rollback ERR INT TERM

GATE_JOB=$(sbatch --parsable --export="$EXPORTS" \
    examples/prospectus_rfi/slurm/stress_amos2025_attention_control.sbatch)
SUBMITTED_JOBS+=("$GATE_JOB")
SEGMENT0_JOB=$(sbatch --parsable --dependency="afterok:$GATE_JOB" \
    --export="$EXPORTS,BSK_RL_SEGMENT_INDEX=0" \
    examples/prospectus_rfi/slurm/train_amos2025_attention_control_segment.sbatch)
SUBMITTED_JOBS+=("$SEGMENT0_JOB")
SEGMENT1_JOB=$(sbatch --parsable --dependency="afterok:$SEGMENT0_JOB" \
    --export="$EXPORTS,BSK_RL_SEGMENT_INDEX=1" \
    examples/prospectus_rfi/slurm/train_amos2025_attention_control_segment.sbatch)
SUBMITTED_JOBS+=("$SEGMENT1_JOB")
CLEANUP_JOB=$(sbatch --parsable --dependency="afterok:$SEGMENT1_JOB" \
    --time=06:00:00 \
    --export="$EXPORTS,BSK_RL_SEGMENT_INDEX=2" \
    examples/prospectus_rfi/slurm/train_amos2025_attention_control_segment.sbatch)
SUBMITTED_JOBS+=("$CLEANUP_JOB")

MANIFEST="$OUTPUT_ROOT/manifests/submission.json"
cat > "$MANIFEST" <<EOF
{
  "branch": "$(git branch --show-current)",
  "commit": "$(git rev-parse HEAD)",
  "campaign_id": "$CAMPAIGN_ID",
  "gate_job": "$GATE_JOB",
  "segment0_job": "$SEGMENT0_JOB",
  "segment1_job": "$SEGMENT1_JOB",
  "cleanup_job": "$CLEANUP_JOB",
  "dependency_type": "afterok",
  "control": "attention, N=100, K=10, image=300s",
  "environment_runners": 4,
  "memory_gib": 120,
  "wandb_group": "rfi-amos2025-attention-k10-300s-control"
}
EOF
trap - ERR INT TERM

echo "Submitted an isolated AMOS 2025 attention control; existing jobs were untouched."
echo "CAMPAIGN_ID=$CAMPAIGN_ID"
echo "GATE_JOB=$GATE_JOB"
echo "SEGMENT0_JOB=$SEGMENT0_JOB"
echo "SEGMENT1_JOB=$SEGMENT1_JOB"
echo "CLEANUP_JOB=$CLEANUP_JOB"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "MANIFEST=$MANIFEST"
