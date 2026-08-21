#!/usr/bin/env bash

# Audit the completed mixed-fixed training sweep, freeze all eight checkpoints,
# and submit their exact mixed-100 Monte Carlo evaluations.

#SBATCH --account=ucb550_asc2
#SBATCH --job-name=amos_mix_post
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --partition=amilan
#SBATCH --mem=8G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --qos=normal

set -euo pipefail

source "/projects/$USER/.venv/bin/activate"
cd "/projects/$USER/bsk_rl"

output_root=${BSK_RL_MIXED_TRAINED_MC_ROOT:-/scratch/alpine/$USER/amos2026_mc/gat_mixed_trained_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_45000s_HIO5_SHIO3_20260802}
manifest_dir="$output_root/manifests"
audit_dir="$manifest_dir/training_audit"
custom_spec="$manifest_dir/mixed_fixed100_custom_policies.json"
custom_tags="$manifest_dir/mixed_fixed100_policy_tags.txt"
submission_marker="$manifest_dir/SUBMISSION_COMPLETE_UTC.txt"

if [[ -f "$submission_marker" ]]; then
    echo "Mixed-trained campaign already submitted: $output_root"
    exit 0
fi

mkdir -p "$audit_dir" "$manifest_dir" "/scratch/alpine/$USER/job_output"
python3 examples/amos_2026/audit_mixed_v9_training_runs.py \
    --policy-root "/scratch/alpine/$USER/rllib_results" \
    --output-dir "$audit_dir"
python3 examples/amos_2026/build_mixed_trained_policy_manifest.py \
    --inventory "$audit_dir/mixed_v9_training_inventory.csv" \
    --output-json "$custom_spec" \
    --output-tags "$custom_tags"

BSK_RL_MC_POLICY_TAGS=$(cat "$custom_tags") \
BSK_RL_MC_CUSTOM_POLICIES_JSON="@$custom_spec" \
BSK_RL_MC_OUTPUT_ROOT="$output_root" \
BSK_RL_MC_CHAIN_BLOCKS=0 \
bash examples/amos_2026/submit_gat_reward_sweep_mc_mixed_100targets_45000s_0to99.sh 4

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$submission_marker"
echo "Mixed-trained evaluation submitted: $output_root"
