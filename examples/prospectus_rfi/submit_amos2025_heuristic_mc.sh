#!/usr/bin/env bash
set -euo pipefail

MAX_CONCURRENT=${1:-12}
if ! [[ "$MAX_CONCURRENT" =~ ^([1-9]|[12][0-9]|30)$ ]]; then
    echo "Usage: $0 [max-concurrent: 1..30, default 12]" >&2
    exit 2
fi

REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
cd "$REPO_DIR"
if [[ "$(git branch --show-current)" != "amos2025-architecture-comparison" ]]; then
    echo "Wrong branch: $(git branch --show-current)" >&2
    exit 3
fi

CAMPAIGN_ID=${BSK_RL_HEURISTIC_MC_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${BSK_RL_HEURISTIC_MC_OUTPUT_ROOT:-/scratch/alpine/$USER/prospectus_rfi/heuristic_mc/amos2025_closest_angle_100s_${CAMPAIGN_ID}}
MANIFEST_DIR="$OUTPUT_ROOT/manifests"
mkdir -p "$MANIFEST_DIR" "/scratch/alpine/$USER/job_output"

COMMIT=$(git rev-parse HEAD)
cat > "$MANIFEST_DIR/campaign.json" <<EOF
{
  "campaign_id": "$CAMPAIGN_ID",
  "campaign_name": "amos2025_closest_angle_mc_100s",
  "branch": "amos2025-architecture-comparison",
  "commit": "$COMMIT",
  "method": "heuristic_historical",
  "heuristic_mode": "angle",
  "information_scope": "full_visible_eligible_catalog",
  "resource_shield": true,
  "candidate_count": 10,
  "catalog_sizes": [100, 200, 400],
  "seeds": {"start": 0, "stop_inclusive": 99},
  "episodes_per_catalog_size": 100,
  "episode_duration_s": 45000.0,
  "imaging_duration_s": 100.0,
  "array_tasks": 30,
  "seeds_per_task": 10,
  "max_concurrent": $MAX_CONCURRENT,
  "dependencies": []
}
EOF

export BSK_RL_HEURISTIC_MC_OUTPUT_ROOT="$OUTPUT_ROOT"
JOB_ID=$(sbatch --parsable \
    --array="0-29%${MAX_CONCURRENT}" \
    --export=ALL,BSK_RL_REPO_DIR,BSK_RL_HEURISTIC_MC_OUTPUT_ROOT \
    examples/prospectus_rfi/slurm/evaluate_amos2025_heuristic_mc.sbatch)

printf '%s\n' "$JOB_ID" > "$MANIFEST_DIR/slurm_job_id.txt"
printf '%s\n' "$OUTPUT_ROOT" > "$MANIFEST_DIR/output_root.txt"
echo "Submitted dependency-free AMOS 2025 closest-angle heuristic MC."
echo "JOB_ID=$JOB_ID"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "TASKS=30 MAX_CONCURRENT=$MAX_CONCURRENT EPISODES=300"
