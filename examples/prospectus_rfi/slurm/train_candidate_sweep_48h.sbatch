#!/usr/bin/env bash
# Six exploratory runs: two architectures x K={5,10,20} x one seed.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_arch_k_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --array=0-5%6
#SBATCH --partition=amilan
#SBATCH --mem=200G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --qos=long
#SBATCH --signal=B:TERM@300

set -euo pipefail

ARCHITECTURES=(mlp mlp mlp attention attention attention)
CANDIDATE_COUNTS=(5 10 20 5 10 20)
INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
ARCHITECTURE=${ARCHITECTURES[$INDEX]}
CANDIDATE_COUNT=${CANDIDATE_COUNTS[$INDEX]}
SEED=10001

module purge
module --ignore_cache load gcc/14.2.0 || export PATH="/curc/sw/install/gcc/14.2.0/bin:${PATH}"
module --ignore_cache load python/3.10.2 || true
source "/projects/$USER/.venv/bin/activate"
REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl}
cd "$REPO_DIR"

if [[ "$(git branch --show-current)" != "codex/amos2025-architecture-comparison-100s" ]]; then
    echo "Wrong branch: $(git branch --show-current)" >&2
    exit 2
fi

TASK_TEMP=$(mktemp -d "/tmp/bsk_rfi_${SLURM_JOB_ID}_${INDEX}_XXXXXX")
trap 'rm -rf "$TASK_TEMP"' EXIT
export TMPDIR=$TASK_TEMP
export RAY_TMPDIR=$TASK_TEMP
export PYTHONUNBUFFERED=1
export BSK_RL_NUM_GPUS=0
export BSK_RL_OBJECT_STORE_MEMORY=${BSK_RL_OBJECT_STORE_MEMORY:-4000000000}
OUTPUT_ROOT="/scratch/alpine/$USER/prospectus_rfi"
mkdir -p "/scratch/alpine/$USER/job_output" "$OUTPUT_ROOT"

echo "branch=$(git branch --show-current) commit=$(git rev-parse HEAD)"
echo "architecture=$ARCHITECTURE K=$CANDIDATE_COUNT seed=$SEED"
if [[ "$ARCHITECTURE" == "mlp" ]]; then
    ARCH_CONFIG=${BSK_RL_MLP_CONFIG:-examples/prospectus_rfi/configs/mlp_selected.yaml}
else
    ARCH_CONFIG=${BSK_RL_ATTENTION_CONFIG:-examples/prospectus_rfi/configs/attention_selected.yaml}
fi
python -u examples/prospectus_rfi/train.py \
    --architecture "$ARCHITECTURE" \
    --architecture-config "$ARCH_CONFIG" \
    --candidate-count "$CANDIDATE_COUNT" \
    --seed "$SEED" \
    --wall-hours 48 \
    --n-env-runners 28 \
    --temp-dir "$TASK_TEMP" \
    --output-root "$OUTPUT_ROOT"
