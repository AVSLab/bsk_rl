#!/usr/bin/env bash
# Gate and submit the paired Monte Carlo for the memory-safe RFI campaign.
set -euo pipefail

REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
ROOT=${BSK_RL_RFI_EVALUATION_ROOT:-/scratch/alpine/$USER/prospectus_rfi/memorysafe_100_200_v2}
BASE_CONFIG=${BSK_RL_RFI_EVALUATION_BASE_CONFIG:-examples/prospectus_rfi/configs/base_memorysafe_100_200.yaml}

cd "$REPO_DIR"
if [[ "$(git branch --show-current)" != "amos2025-architecture-comparison" ]]; then
    echo "Wrong branch: $(git branch --show-current)" >&2
    exit 2
fi

for METHOD in mlp attention; do
    for K in 5 10 20; do
        CHECKPOINT="$ROOT/training/${METHOD}_k${K}_seed10001/checkpoints/best_validation"
        if [[ ! -e "$CHECKPOINT" ]]; then
            echo "Missing validated checkpoint: $CHECKPOINT" >&2
            exit 3
        fi
    done
done

if find "$ROOT/evaluation/raw" -maxdepth 1 -type f -name '*.csv' -print -quit \
    2>/dev/null | grep -q .; then
    echo "Refusing to overwrite an existing paired evaluation under $ROOT/evaluation/raw" >&2
    exit 4
fi

mkdir -p "$ROOT/manifests" "/scratch/alpine/$USER/job_output"
EXPORTS="ALL,BSK_RL_REPO_DIR=$REPO_DIR,BSK_RL_RFI_EVALUATION_ROOT=$ROOT,BSK_RL_RFI_EVALUATION_BASE_CONFIG=$BASE_CONFIG"
SBATCH_FILE=examples/prospectus_rfi/slurm/evaluate_paired_mc.sbatch

sbatch --test-only --export="$EXPORTS" "$SBATCH_FILE"
MC_JOB=$(sbatch --parsable --export="$EXPORTS" "$SBATCH_FILE")

MANIFEST="$ROOT/manifests/paired_mc_submission_${MC_JOB}.json"
cat > "$MANIFEST" <<EOF
{
  "branch": "$(git branch --show-current)",
  "commit": "$(git rev-parse HEAD)",
  "job_id": "$MC_JOB",
  "base_config": "$BASE_CONFIG",
  "evaluation_root": "$ROOT",
  "methods": ["mlp", "attention", "heuristic_historical", "heuristic_distance_historical", "heuristic_matched"],
  "candidate_counts": [5, 10, 20],
  "catalog_sizes": [100, 200, 300, 400],
  "scenario_seeds": [700000, 700099],
  "expected_array_tasks": 240,
  "expected_episode_rows": 6000
}
EOF

echo "MC_JOB=$MC_JOB"
echo "OUTPUT_ROOT=$ROOT"
echo "MANIFEST=$MANIFEST"
