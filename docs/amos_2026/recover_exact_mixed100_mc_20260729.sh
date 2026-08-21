#!/usr/bin/env bash
set -euo pipefail

host="${AMOS2026_CLUSTER_HOST:-dahu1128@login-ci5.rc.colorado.edu}"
local_repo="/Users/dahu1128/Repositories/bsk_rl"
remote_repo="/projects/dahu1128/bsk_rl"
output_root="/scratch/alpine/dahu1128/amos2026_mc/gat_full_actions_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_45000s_HIO5_SHIO3_20260729T232546Z"

cd "$local_repo"

COPYFILE_DISABLE=1 tar -czf - \
    examples/updated_policy_evaluation.py \
    examples/amos_2026/submit_gat_reward_sweep_mc_missing_alphas_200targets_45000s_0to99_no_deps.sh \
  | ssh "$host" "set -euo pipefail
      tar -xzf - -C '$remote_repo'
      cd '$remote_repo'
      source /projects/\$USER/.venv/bin/activate

      scancel 30568061 30568062 30568063 30568064 30568065 \
              30568066 30568067 30568068 30568069 30568070 || true
      sleep 3

      python3 -m py_compile \
          examples/updated_policy_evaluation.py \
          examples/amos_2026/evaluate_gat_reward_sweep_mc.py
      bash -n \
          examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh \
          examples/amos_2026/submit_gat_reward_sweep_mc_mixed_100targets_45000s_0to99.sh

      export BSK_RL_MC_OUTPUT_ROOT='$output_root'
      bash examples/amos_2026/submit_gat_reward_sweep_mc_mixed_100targets_45000s_0to99.sh 5
    "
