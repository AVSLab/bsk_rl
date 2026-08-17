#!/usr/bin/env bash
# Submit all six independent v2 runs after a passing two-task stress gate.
set -euo pipefail

STRESS_JOB_ID=${1:?Usage: $0 PASSING_STRESS_ARRAY_JOB_ID}
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

# The full campaign cannot be submitted from an unaudited or failed stress run.
bash examples/prospectus_rfi/audit_memorysafe_stress.sh "$STRESS_JOB_ID"

OUTPUT_ROOT="/scratch/alpine/$USER/prospectus_rfi/memorysafe_100_200_v2"
for RUN_NAME in \
    mlp_k5_seed10001 mlp_k10_seed10001 mlp_k20_seed10001 \
    attention_k5_seed10001 attention_k10_seed10001 attention_k20_seed10001; do
    if [[ -e "$OUTPUT_ROOT/training/$RUN_NAME" ]]; then
        echo "Refusing to overwrite existing v2 output: $OUTPUT_ROOT/training/$RUN_NAME" >&2
        exit 4
    fi
done

mkdir -p "/scratch/alpine/$USER/job_output" "$OUTPUT_ROOT/manifests"
sbatch --test-only \
    --array=0 \
    --export=ALL,BSK_RL_REPO_DIR,BSK_RL_WANDB_KEY_PATH,BSK_RL_SEGMENT_INDEX=0 \
    examples/prospectus_rfi/slurm/train_candidate_sweep_memorysafe_segment.sbatch

SUBMITTED_JOBS=()
rollback() {
    local job_id
    for job_id in "${SUBMITTED_JOBS[@]}"; do
        scancel "$job_id" 2>/dev/null || true
    done
}
trap rollback ERR INT TERM

SEGMENT0=$(sbatch --parsable \
    --array=0-5%6 \
    --export=ALL,BSK_RL_REPO_DIR,BSK_RL_WANDB_KEY_PATH,BSK_RL_SEGMENT_INDEX=0 \
    examples/prospectus_rfi/slurm/train_candidate_sweep_memorysafe_segment.sbatch)
SUBMITTED_JOBS+=("$SEGMENT0")
SEGMENT1=$(sbatch --parsable \
    --array=0-5%6 \
    --dependency="aftercorr:$SEGMENT0" \
    --export=ALL,BSK_RL_REPO_DIR,BSK_RL_WANDB_KEY_PATH,BSK_RL_SEGMENT_INDEX=1 \
    examples/prospectus_rfi/slurm/train_candidate_sweep_memorysafe_segment.sbatch)
SUBMITTED_JOBS+=("$SEGMENT1")
CLEANUP=$(sbatch --parsable \
    --array=0-5%6 \
    --dependency="aftercorr:$SEGMENT1" \
    --time=06:00:00 \
    --export=ALL,BSK_RL_REPO_DIR,BSK_RL_WANDB_KEY_PATH,BSK_RL_SEGMENT_INDEX=2 \
    examples/prospectus_rfi/slurm/train_candidate_sweep_memorysafe_segment.sbatch)
SUBMITTED_JOBS+=("$CLEANUP")

CAMPAIGN_ID=$(date -u +%Y%m%dT%H%M%SZ)
MANIFEST="$OUTPUT_ROOT/manifests/submission_${CAMPAIGN_ID}.json"
{
    printf '{\n'
    printf '  "branch": "%s",\n' "$(git branch --show-current)"
    printf '  "commit": "%s",\n' "$(git rev-parse HEAD)"
    printf '  "passing_stress_job": "%s",\n' "$STRESS_JOB_ID"
    printf '  "segment0_array": "%s",\n' "$SEGMENT0"
    printf '  "segment1_array": "%s",\n' "$SEGMENT1"
    printf '  "cleanup_array": "%s",\n' "$CLEANUP"
    printf '  "dependency_type": "aftercorr",\n'
    printf '  "catalog_range": [100, 200],\n'
    printf '  "environment_runners": 12,\n'
    printf '  "memory_gib": 230,\n'
    printf '  "wandb_group": "rfi-alpha0-100s-n100-200-memorysafe-v2"\n'
    printf '}\n'
} > "$MANIFEST"
trap - ERR INT TERM

echo "Submitted all six memory-safe v2 policies with task-correlated continuations."
echo "SEGMENT0_JOB=$SEGMENT0"
echo "SEGMENT1_JOB=$SEGMENT1"
echo "CLEANUP_JOB=$CLEANUP"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "MANIFEST=$MANIFEST"
