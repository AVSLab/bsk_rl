#!/usr/bin/env bash
set -euo pipefail

# Run on the Mac. One authenticated SSH connection transfers the tested files
# and invokes the Alpine launch helper.

HOST=${AMOS2026_CLUSTER_HOST:-dahu1128@login-ci5.rc.colorado.edu}
LOCAL_REPO=/Users/dahu1128/Repositories/bsk_rl
REMOTE_REPO=/projects/dahu1128/bsk_rl
TRAIN_MAX_CONCURRENT=${TRAIN_MAX_CONCURRENT:-4}
MC_MAX_CONCURRENT=${MC_MAX_CONCURRENT:-5}
RUN_TRAINING=${RUN_TRAINING:-1}
RUN_MIXED100_MC=${RUN_MIXED100_MC:-1}

cd "$LOCAL_REPO"

echo "Connecting to $HOST."
echo "Enter your Alpine password when prompted."
echo "RUN_TRAINING=$RUN_TRAINING; RUN_MIXED100_MC=$RUN_MIXED100_MC"

COPYFILE_DISABLE=1 tar -czf - \
    examples/train_Polaris_gat_full_actions_wandb.py \
    examples/updated_policy_evaluation.py \
    examples/amos_2026/evaluate_gat_reward_sweep_mc.py \
    examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh \
    examples/amos_2026/sbatch_train_polaris_gat_full_actions_mixed_fixed_100targets_reward_sweep_48h.sh \
    examples/amos_2026/submit_mixed_fixed_100targets_reward_sweep_training.sh \
    examples/amos_2026/submit_gat_reward_sweep_mc_mixed_100targets_45000s_0to99.sh \
    docs/amos_2026/cluster_launch_mixed_fixed_training_and_mixed100_mc.sh \
  | ssh "$HOST" "set -euo pipefail
      tar -xzf - -C '$REMOTE_REPO'
      cd '$REMOTE_REPO'
      RUN_TRAINING='$RUN_TRAINING' \
      RUN_MIXED100_MC='$RUN_MIXED100_MC' \
      TRAIN_MAX_CONCURRENT='$TRAIN_MAX_CONCURRENT' \
      MC_MAX_CONCURRENT='$MC_MAX_CONCURRENT' \
        bash docs/amos_2026/cluster_launch_mixed_fixed_training_and_mixed100_mc.sh
    "
