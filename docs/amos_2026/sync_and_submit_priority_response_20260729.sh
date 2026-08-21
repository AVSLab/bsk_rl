#!/bin/bash

# Sync the focused priority-response instrumentation to Alpine and submit it.
# This prompts once for the Alpine password, then immediately queues 100 seeds.

set -euo pipefail

LOCAL_REPO=${LOCAL_REPO:-/Users/dahu1128/Repositories/bsk_rl}
REMOTE_HOST=${REMOTE_HOST:-dahu1128@login-ci5.rc.colorado.edu}
REMOTE_REPO=${REMOTE_REPO:-/projects/dahu1128/bsk_rl}

FILES=(
    examples/sim_config.py
    src/bsk_rl/scene/rso_targets.py
    src/bsk_rl/obs/observations.py
    examples/updated_policy_evaluation.py
    examples/amos_2026/evaluate_gat_reward_sweep_mc.py
    examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh
    examples/amos_2026/submit_gat_priority_response_alpha0p1_mc_mixed_200targets_45000s_0to99.sh
    examples/amos_2026/analyze_gat_priority_response_mc.py
)

cd "$LOCAL_REPO"
tar -czf - "${FILES[@]}" | ssh "$REMOTE_HOST" \
    "set -euo pipefail
     cd '$REMOTE_REPO'
     tar -xzf -
     chmod +x examples/amos_2026/submit_gat_priority_response_alpha0p1_mc_mixed_200targets_45000s_0to99.sh
     chmod +x examples/amos_2026/analyze_gat_priority_response_mc.py
     bash examples/amos_2026/submit_gat_priority_response_alpha0p1_mc_mixed_200targets_45000s_0to99.sh"
