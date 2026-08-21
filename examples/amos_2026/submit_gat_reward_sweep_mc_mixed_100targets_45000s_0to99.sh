#!/usr/bin/env bash
set -euo pipefail

# Evaluate frozen LEO-trained obs-v9 GAT policies in the missing mixed-100
# scenario. Every seed contains exactly 50 LEO, 30 MEO, and 20 GEO targets.
# All policies are scored with the common 100d00i (alpha=1.0) ground-value
# objective used in the AMOS 2026 paper comparisons.

export BSK_RL_MC_POLICY_TAGS=${BSK_RL_MC_POLICY_TAGS:-00d100i,10d90i,20d80i,30d70i,40d60i,50d50i,60d40i,70d30i,75d25i,80d20i,90d10i,100d00i}
export BSK_RL_MC_N_TARGETS=100
export BSK_RL_MC_N_TARGETS_AHEAD=10
export BSK_RL_MC_TOTAL_TIME_SEC=45000
export BSK_RL_MC_TARGET_ENV=mixed
export BSK_RL_MC_MIX_WEIGHTS='{"LEO":0.5,"MEO":0.3,"GEO":0.2}'
export BSK_RL_MC_EXACT_MIX_COUNTS=1
export BSK_RL_MC_DYNAMIC_PRIORITY_EVENT=on
export BSK_RL_MC_HIO_COUNT=5
export BSK_RL_MC_HIO_PRIORITY=5.0
export BSK_RL_MC_SHIO_COUNT=3
export BSK_RL_MC_SHIO_PRIORITY=10.0
export BSK_RL_MC_CHAIN_BLOCKS=${BSK_RL_MC_CHAIN_BLOCKS:-0}

CAMPAIGN_ID=${BSK_RL_MC_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
export BSK_RL_MC_OUTPUT_ROOT=${BSK_RL_MC_OUTPUT_ROOT:-/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_45000s_HIO5_SHIO3_${CAMPAIGN_ID}}

exec bash examples/amos_2026/submit_gat_reward_sweep_mc_mixed_200targets_45000s_0to99.sh "${1:-5}"
