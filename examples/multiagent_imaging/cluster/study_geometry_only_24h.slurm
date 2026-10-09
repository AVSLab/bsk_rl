#!/usr/bin/env bash
#SBATCH --job-name=multiagent-without_communication-t200
#SBATCH --account=ucb550_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=multiagent-without_communication-%j.log
# PREPARED ONLY. Requires separately approved readiness AND this 24h tranche.
set -euo pipefail
: "${BSK_READINESS_ACCEPTANCE:?Path to passing same-release readiness JSON required}"
: "${BSK_LONG_TRANCHE_APPROVED:?Set only after explicit user approval}"
export BSK_RESERVATION_CAP=768
export BSK_EXPECTED_CPUS=32 BSK_EXPECTED_WALL_S=86400 BSK_EXPECTED_MEMORY_MIB=102400
source "${BSK_PROJECT_ROOT:?}/examples/multiagent_imaging/cluster/study_common.sh"
BSK_ARM=geometry_only
BSK_CONFIG="examples/multiagent_imaging/configs/walker4_t200_cd1_${BSK_ARM}.json"
BSK_CUTOFF="$("$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_long_budget --readiness "$BSK_READINESS_ACCEPTANCE" --arm "$BSK_ARM" --output "$BSK_OUTPUT/budget.json" --parallel)"
read -r BSK_MAX_UPDATES BSK_SOFT_TRUE BSK_HARD_TRUE BSK_EPISODE_CAP < <("$BSK_RL_PYTHON" -c 'import json,os; p=json.load(open(os.environ["BSK_OUTPUT"]+"/budget.json")); print(p["max_updates"],p["soft_true_decision_target"],p["hard_new_sampled_decision_cap"],p["max_training_episodes"])')
export BSK_RL_WANDB_NAME="without_communication_s3105_tranche1_${SLURM_JOB_ID}"
study_run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.train --config "$BSK_CONFIG" --output "$BSK_OUTPUT/training" --iterations "$BSK_MAX_UPDATES" --train-batch-size 4200 --num-env-runners 28 --num-learners 0 --ray-cpus "$SLURM_CPUS_PER_TASK" --torch-threads 1 --sample-timeout-s 3600 --ppo-profile amos_best --save-initial --max-true-decisions "$BSK_SOFT_TRUE" --max-sampled-true-decisions "$BSK_HARD_TRUE" --max-training-episodes "$BSK_EPISODE_CAP" --max-wall-seconds "$BSK_CUTOFF" --wandb-mode online --require-wandb
BSK_FINAL_CHECKPOINT="$("$BSK_RL_PYTHON" -c 'import json,os; print(json.load(open(os.environ["BSK_OUTPUT"]+"/training/tranche-stop.json"))["checkpoint"])')"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_checkpoint_bytes --checkpoint "$BSK_FINAL_CHECKPOINT" --output "$BSK_OUTPUT/checkpoint-before.json"
study_run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.train --config "$BSK_CONFIG" --output "$BSK_OUTPUT/resume-check" --resume "$BSK_FINAL_CHECKPOINT" --iterations 0 --train-batch-size 4200 --num-env-runners 28 --num-learners 0 --ray-cpus "$SLURM_CPUS_PER_TASK" --torch-threads 1 --sample-timeout-s 3600 --ppo-profile amos_best --wandb-mode online --require-wandb
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_checkpoint_bytes --checkpoint "$BSK_FINAL_CHECKPOINT" --compare "$BSK_OUTPUT/checkpoint-before.json" --output "$BSK_OUTPUT/checkpoint-after.json"
study_run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.compare_learned_checkpoints --config "$BSK_CONFIG" --checkpoint "initial=$BSK_OUTPUT/training/checkpoint_0000" --checkpoint "trained=$BSK_FINAL_CHECKPOINT" --seeds 401 402 --output "$BSK_OUTPUT/heldout" --workers 8
# Common milestone checkpoints are recorded in updates.json. Evaluate additional
# milestones only under a separately reviewed evaluation budget.
