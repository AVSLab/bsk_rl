#!/usr/bin/env bash
# Read-only snapshot diagnostic for the active N=100--200 training campaign.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_diag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --partition=acpu
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --qos=cpu-normal
#SBATCH --nice=10000

set -euo pipefail

REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
INPUT_ROOT=${BSK_RL_DIAGNOSTIC_INPUT_ROOT:-/scratch/alpine/$USER/prospectus_rfi/memorysafe_100_200_v2}
module purge
source "$REPO_DIR/examples/prospectus_rfi/slurm/alpine_runtime.sh"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "/scratch/alpine/$USER/job_output"

echo "branch=$(git branch --show-current) commit=$(git rev-parse HEAD)"
echo "read_only_input=$INPUT_ROOT"
python -u examples/prospectus_rfi/diagnose_training.py --input-root "$INPUT_ROOT"
