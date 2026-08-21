#!/usr/bin/env bash
set -euo pipefail

host="${AMOS2026_CLUSTER_HOST:-dahu1128@login-ci5.rc.colorado.edu}"
local_repo="/Users/dahu1128/Repositories/bsk_rl"
remote_repo="/projects/dahu1128/bsk_rl"

cd "$local_repo"

COPYFILE_DISABLE=1 tar -czf - \
    examples/amos_2026/submit_gat_reward_sweep_mc_mixed_100targets_45000s_0to99.sh \
    examples/amos_2026/submit_gat_reward_sweep_mc_mixed_200targets_45000s_0to99.sh \
  | ssh "$host" "set -euo pipefail
      tar -xzf - -C '$remote_repo'
      cd '$remote_repo'
      bash -n \
          examples/amos_2026/submit_gat_reward_sweep_mc_mixed_100targets_45000s_0to99.sh \
          examples/amos_2026/submit_gat_reward_sweep_mc_mixed_200targets_45000s_0to99.sh

      for job_id in 30568319 30568320 30568321 30568322 30568323 \
                    30568324 30568325 30568326 30568327; do
          if squeue -h -j \"\$job_id\" | grep -q .; then
              scontrol update JobId=\"\$job_id\" Dependency=
              echo \"Released dependency for \$job_id\"
          else
              echo \"Job \$job_id is no longer queued; no dependency update needed\"
          fi
      done

      echo
      squeue -u \"\$USER\" \
        --format='%.18i %.30j %.2t %.10M %.10l %.6D %R' \
        | grep -E 'JOBID|gat_mix100_fixed|gat_mc_mix_100t' || true
    "
