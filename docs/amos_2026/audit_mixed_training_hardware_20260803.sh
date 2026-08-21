#!/usr/bin/env bash
set -euo pipefail

output=${1:-/Users/dahu1128/Downloads/AMOS2026_mixed_training_hardware_20260803.txt}
remote_host=${AMOS2026_CLUSTER_HOST:-dahu1128@login-ci5.rc.colorado.edu}

mkdir -p "$(dirname "$output")"

ssh "$remote_host" 'bash -s' >"$output" <<'REMOTE'
set -euo pipefail

echo "AMOS 2026 mixed-training hardware audit"
date -u +'%Y-%m-%dT%H:%M:%SZ'

echo
echo "===== Slurm accounting for training array 30568059 ====="
sacct -X -j 30568059 \
    --starttime 2026-07-29 \
    --endtime 2026-08-04 \
    --format=JobIDRaw,JobName%30,State,Elapsed,ElapsedRaw,Timelimit,AllocCPUS,ReqMem,Partition,QOS,NodeList%80,ExitCode

echo
echo "===== Machine-readable array-task accounting ====="
sacct -X -n -P -j 30568059 \
    --starttime 2026-07-29 \
    --endtime 2026-08-04 \
    --format=JobIDRaw,State,ElapsedRaw,AllocCPUS,ReqMem,Partition,QOS,NodeList

echo
echo "===== Realized compute-node records ====="
mapfile -t nodes < <(
    sacct -X -n -P -j 30568059 \
        --starttime 2026-07-29 \
        --endtime 2026-08-04 \
        --format=NodeList \
        | sed '/^$/d;/None/d;/Unknown/d' \
        | sort -u
)
for node in "${nodes[@]}"; do
    echo "--- $node ---"
    scontrol show node "$node" | grep -E \
        'NodeName=|Arch=|CfgTRES=|ActiveFeatures=|AvailableFeatures=|RealMemory=|ThreadsPerCore=|CPUTot='
done

echo
echo "===== Submission resource directives ====="
grep '^#SBATCH' \
    /projects/$USER/bsk_rl/examples/amos_2026/sbatch_train_polaris_gat_full_actions_mixed_fixed_100targets_reward_sweep_48h.sh
REMOTE

echo "Saved Alpine hardware audit to: $output"
