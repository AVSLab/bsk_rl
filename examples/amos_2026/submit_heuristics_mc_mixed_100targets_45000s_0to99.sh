#!/usr/bin/env bash
set -euo pipefail

max_concurrent=${1:-4}
if ! [[ "$max_concurrent" =~ ^[1-8]$ ]]; then
    echo "Usage: $0 [max-concurrent: 1..8, default 4]" >&2
    exit 2
fi

cd "/projects/$USER/bsk_rl"
source "/projects/$USER/.venv/bin/activate"

campaign_id=${BSK_RL_HEUR_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
output_root=${BSK_RL_HEUR_OUTPUT_ROOT:-/scratch/alpine/$USER/amos2026_mc/heuristics_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_45000s_HIO5_SHIO3_${campaign_id}}
mkdir -p "$output_root/manifests"

cat > "$output_root/manifests/campaign.json" <<EOF
{
  "campaign_id": "$campaign_id",
  "heuristic_modes": ["angle", "priority_angle"],
  "seeds": "0:100",
  "target_environment": "mixed",
  "target_count": 100,
  "exact_regime_counts": {"LEO": 50, "MEO": 30, "GEO": 20},
  "candidate_count": 10,
  "total_time_sec": 45000,
  "evaluation_reward_mix": "100d00i",
  "dynamic_priority_event": true,
  "HIO_count": 5,
  "SHIO_count": 3
}
EOF

echo "Submitting heuristic campaign to $output_root"
job_ids=()
for seed_start in $(seq 0 10 90); do
    seed_stop=$((seed_start + 9))
    job_name=$(printf 'heur_mc_s%03d_%03d' "$seed_start" "$seed_stop")
    submission=$(
        sbatch \
            --job-name="$job_name" \
            --array="0-1%${max_concurrent}" \
            --export="ALL,BSK_RL_HEUR_SEED_START=$seed_start,BSK_RL_HEUR_SEEDS_PER_BLOCK=10,BSK_RL_HEUR_OUTPUT_ROOT=$output_root" \
            examples/amos_2026/sbatch_evaluate_heuristics_mc_10seeds.sh
    )
    job_id=${submission##* }
    job_ids+=("$job_id")
    echo "$job_name: $submission"
done

printf '%s\n' "${job_ids[@]}" > "$output_root/manifests/slurm_job_ids.txt"
echo "No dependencies were added. Slurm may run every seed block concurrently."
echo "OUTPUT_ROOT=$output_root"
