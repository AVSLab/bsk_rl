#!/usr/bin/env bash
set -euo pipefail

# Run on Alpine after the companion Mac sync script transfers the files.

RUN_TRAINING=${RUN_TRAINING:-1}
RUN_MIXED100_MC=${RUN_MIXED100_MC:-1}
TRAIN_MAX_CONCURRENT=${TRAIN_MAX_CONCURRENT:-4}
MC_MAX_CONCURRENT=${MC_MAX_CONCURRENT:-5}

cd /projects/$USER/bsk_rl
source /projects/$USER/.venv/bin/activate

test -s examples/wandb_key.txt
python3 -c "import wandb; print('wandb import ok:', wandb.__version__)"
python3 -m py_compile \
    examples/train_Polaris_gat_full_actions_wandb.py \
    examples/updated_policy_evaluation.py \
    examples/amos_2026/evaluate_gat_reward_sweep_mc.py
bash -n \
    examples/amos_2026/sbatch_train_polaris_gat_full_actions_mixed_fixed_100targets_reward_sweep_48h.sh \
    examples/amos_2026/submit_mixed_fixed_100targets_reward_sweep_training.sh \
    examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh \
    examples/amos_2026/submit_gat_reward_sweep_mc_mixed_100targets_45000s_0to99.sh

echo
echo "===== Training submission preflight ====="
BSK_RL_DRY_RUN=1 \
    bash examples/amos_2026/submit_mixed_fixed_100targets_reward_sweep_training.sh \
    "$TRAIN_MAX_CONCURRENT"

if [[ "$RUN_TRAINING" == "1" ]]; then
    echo
    echo "===== Submitting mixed-fixed training sweep ====="
    bash examples/amos_2026/submit_mixed_fixed_100targets_reward_sweep_training.sh \
        "$TRAIN_MAX_CONCURRENT"
fi

if [[ "$RUN_MIXED100_MC" == "1" ]]; then
    echo
    echo "===== Submitting LEO-policy mixed-100 MC campaign ====="
    bash examples/amos_2026/submit_gat_reward_sweep_mc_mixed_100targets_45000s_0to99.sh \
        "$MC_MAX_CONCURRENT"
fi

echo
echo "===== Current AMOS jobs ====="
squeue -u "$USER" --format='%.18i %.30j %.2t %.10M %.10l %.6D %R' \
    | grep -E 'JOBID|gat_mix100_fixed|gat_mc_mix_100t' || true
