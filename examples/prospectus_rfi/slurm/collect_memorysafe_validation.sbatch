#!/usr/bin/env bash
# Combine atomic validation rows and create the six best_validation symlinks.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_memvcollect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=acpu
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --qos=cpu-normal
#SBATCH --nice=10000

set -euo pipefail
REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
MANIFEST=${BSK_RL_VALIDATION_MANIFEST:?BSK_RL_VALIDATION_MANIFEST is required}
module purge
source "$REPO_DIR/examples/prospectus_rfi/slurm/alpine_runtime.sh"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
python -u examples/prospectus_rfi/collect_memorysafe_validation.py --manifest "$MANIFEST"
