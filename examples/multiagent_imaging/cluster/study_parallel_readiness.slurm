#!/usr/bin/env bash
#SBATCH --job-name=multiagent-walker4-parallel-readiness
#SBATCH --account=ucb550_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=multiagent-study-readiness-%j.log
# PREPARED ONLY. Needs explicit review of expanded mission caps: <=32 per arm.
# One allocation, <=2 PPO updates, <=120k true decisions total, 16 held-out missions.
# Never substitute this script for the approved unchanged v4 readiness launcher.
set -euo pipefail
export BSK_RESERVATION_CAP=64
export BSK_EXPECTED_CPUS=32 BSK_EXPECTED_WALL_S=7200 BSK_EXPECTED_MEMORY_MIB=102400
source "${BSK_PROJECT_ROOT:?}/examples/multiagent_imaging/cluster/study_common.sh"
"$BSK_RL_PYTHON" -m pytest -q tests/unittest/multiagent/test_study_protocol.py tests/unittest/multiagent/test_study_metrics.py tests/unittest/multiagent/test_data_boundaries.py tests/unittest/multiagent/test_async_discounting.py tests/unittest/multiagent/test_gate_repairs.py > "$BSK_OUTPUT/focused-tests.log" 2>&1
for BSK_ARM in catalog_tx geometry_only; do
  BSK_CONFIG="examples/multiagent_imaging/configs/walker4_t200_cd1_${BSK_ARM}.json"
  BSK_LABEL=with_communication; if [[ "$BSK_ARM" == geometry_only ]]; then BSK_LABEL=without_communication; fi
  export BSK_RL_WANDB_NAME="${BSK_LABEL}_s3105_readiness_${SLURM_JOB_ID}"
  study_run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.train --config "$BSK_CONFIG" --output "$BSK_OUTPUT/$BSK_ARM" --iterations 1 --train-batch-size 4200 --num-env-runners 28 --num-learners 0 --ray-cpus "$SLURM_CPUS_PER_TASK" --torch-threads 1 --sample-timeout-s 3600 --ppo-profile amos_best --save-initial --capture-learner-batch --max-training-episodes 32 --max-sampled-true-decisions 60000 --wandb-mode online --require-wandb
  "$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_checkpoint_bytes --checkpoint "$BSK_OUTPUT/$BSK_ARM/checkpoint_0001" --output "$BSK_OUTPUT/$BSK_ARM/checkpoint-before.json"
  study_run "$BSK_RL_PYTHON" -m examples.multiagent_imaging.train --config "$BSK_CONFIG" --output "$BSK_OUTPUT/${BSK_ARM}-resume" --resume "$BSK_OUTPUT/$BSK_ARM/checkpoint_0001" --iterations 0 --train-batch-size 4200 --num-env-runners 28 --num-learners 0 --ray-cpus "$SLURM_CPUS_PER_TASK" --torch-threads 1 --sample-timeout-s 3600 --ppo-profile amos_best --wandb-mode online --require-wandb
  "$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_checkpoint_bytes --checkpoint "$BSK_OUTPUT/$BSK_ARM/checkpoint_0001" --compare "$BSK_OUTPUT/$BSK_ARM/checkpoint-before.json" --output "$BSK_OUTPUT/$BSK_ARM/checkpoint-after.json"
  "$BSK_RL_PYTHON" -m examples.multiagent_imaging.compare_learned_checkpoints --config "$BSK_CONFIG" --checkpoint "initial=$BSK_OUTPUT/$BSK_ARM/checkpoint_0000" --checkpoint "trained=$BSK_OUTPUT/$BSK_ARM/checkpoint_0001" --seeds 401 402 --output "$BSK_OUTPUT/$BSK_ARM/heldout" --workers 8
done
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.cluster.study_gate --stage verify --output "$BSK_OUTPUT" --parallel
