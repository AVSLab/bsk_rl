#!/usr/bin/env bash
#SBATCH --job-name=multiagent-alpha-sweep
#SBATCH --account=ucb550_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=200G
#SBATCH --time=23:59:00
#SBATCH --array=0-15%2
#SBATCH --no-requeue
#SBATCH --export=ALL
#SBATCH --signal=B:USR1@1800
set -euo pipefail
: "${BSK_SWEEP_ROOT:?Path to the sealed prepared campaign}"
: "${BSK_SWEEP_OUTPUT_ROOT:?Fresh scratch campaign path}"
: "${BSK_RL_WANDB_KEY_PATH:?Existing private key file path}"
: "${SLURM_ARRAY_TASK_ID:?Run using the prepared array submitter}"
exec python3 "$BSK_SWEEP_ROOT/campaign.py" run "$SLURM_ARRAY_TASK_ID"
