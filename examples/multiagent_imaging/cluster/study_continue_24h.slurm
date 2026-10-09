#!/usr/bin/env bash
#SBATCH --job-name=multiagent-walker4-continue
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
#SBATCH --output=multiagent-study-continue-%j.log
# PREPARED ONLY. Requires a reviewed checkpoint and separate continuation approval.
set -euo pipefail
: "${BSK_READINESS_ACCEPTANCE:?Passing same-source readiness JSON required}"
: "${BSK_CONTINUATION_APPROVED:?Set only after explicit continuation approval}"
: "${BSK_ARM:?catalog_tx or geometry_only}"
case "$BSK_ARM" in catalog_tx|geometry_only) ;; *) exit 2 ;; esac
: "${BSK_RESUME_CHECKPOINT:?Reviewed completed-update checkpoint required}"
: "${BSK_INITIAL_CHECKPOINT:?Original fresh study checkpoint_0000 required}"
export BSK_RESERVATION_CAP=768
export BSK_EXPECTED_CPUS=32 BSK_EXPECTED_WALL_S=86400 BSK_EXPECTED_MEMORY_MIB=102400
source "${BSK_PROJECT_ROOT:?}/examples/multiagent_imaging/cluster/study_common.sh"
BSK_CONFIG="examples/multiagent_imaging/configs/walker4_t200_cd1_${BSK_ARM}.json"
BSK_CUTOFF="$("$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_long_budget --readiness "$BSK_READINESS_ACCEPTANCE" --arm "$BSK_ARM" --output "$BSK_OUTPUT/budget.json" --parallel)"
read -r BSK_MAX_UPDATES BSK_SOFT_TRUE BSK_HARD_TRUE BSK_EPISODE_CAP < <("$BSK_RL_PYTHON" -c 'import json,os; p=json.load(open(os.environ["BSK_OUTPUT"]+"/budget.json")); print(p["max_updates"],p["soft_true_decision_target"],p["hard_new_sampled_decision_cap"],p["max_training_episodes"])')
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_checkpoint_bytes --checkpoint "$BSK_RESUME_CHECKPOINT" --output "$BSK_OUTPUT/predecessor-before.json"
study_run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.train --config "$BSK_CONFIG" --output "$BSK_OUTPUT/pre-restore" --resume "$BSK_RESUME_CHECKPOINT" --iterations 0 --train-batch-size 4200 --num-env-runners 28 --num-learners 0 --ray-cpus "$SLURM_CPUS_PER_TASK" --torch-threads 1 --sample-timeout-s 3600 --ppo-profile amos_best --wandb-mode online --require-wandb
study_run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.train --config "$BSK_CONFIG" --output "$BSK_OUTPUT/training" --resume "$BSK_RESUME_CHECKPOINT" --iterations "$BSK_MAX_UPDATES" --train-batch-size 4200 --num-env-runners 28 --num-learners 0 --ray-cpus "$SLURM_CPUS_PER_TASK" --torch-threads 1 --sample-timeout-s 3600 --ppo-profile amos_best --max-true-decisions "$BSK_SOFT_TRUE" --max-sampled-true-decisions "$BSK_HARD_TRUE" --max-training-episodes "$BSK_EPISODE_CAP" --max-wall-seconds "$BSK_CUTOFF" --wandb-mode online --require-wandb
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_checkpoint_bytes --checkpoint "$BSK_RESUME_CHECKPOINT" --compare "$BSK_OUTPUT/predecessor-before.json" --output "$BSK_OUTPUT/predecessor-after.json"
BSK_FINAL_CHECKPOINT="$("$BSK_RL_PYTHON" -c 'import json,os; print(json.load(open(os.environ["BSK_OUTPUT"]+"/training/tranche-stop.json"))["checkpoint"])')"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_checkpoint_bytes --checkpoint "$BSK_FINAL_CHECKPOINT" --output "$BSK_OUTPUT/final-before.json"
study_run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.train --config "$BSK_CONFIG" --output "$BSK_OUTPUT/post-restore" --resume "$BSK_FINAL_CHECKPOINT" --iterations 0 --train-batch-size 4200 --num-env-runners 28 --num-learners 0 --ray-cpus "$SLURM_CPUS_PER_TASK" --torch-threads 1 --sample-timeout-s 3600 --ppo-profile amos_best --wandb-mode online --require-wandb
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_checkpoint_bytes --checkpoint "$BSK_FINAL_CHECKPOINT" --compare "$BSK_OUTPUT/final-before.json" --output "$BSK_OUTPUT/final-after.json"
study_run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.compare_learned_checkpoints --config "$BSK_CONFIG" --checkpoint "initial=$BSK_INITIAL_CHECKPOINT" --checkpoint "trained=$BSK_FINAL_CHECKPOINT" --seeds 401 402 --output "$BSK_OUTPUT/heldout" --workers 8
# Resume retains run ID/name/group, optimizer, seed stream/RNG and lifetime metrics.
# Stop here. Another tranche needs another reviewed checkpoint and approval.
