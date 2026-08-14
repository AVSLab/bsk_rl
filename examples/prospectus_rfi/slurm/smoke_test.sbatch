#!/usr/bin/env bash
# Full 45,000 s episode smoke: random MLP, random attention, and heuristic at N=100/400.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=amilan
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --qos=normal

set -euo pipefail

module purge
module --ignore_cache load gcc/14.2.0 || export PATH="/curc/sw/install/gcc/14.2.0/bin:${PATH}"
module --ignore_cache load python/3.10.2 || true
source "/projects/$USER/.venv/bin/activate"
REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl-rfi}
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

if [[ "$(git branch --show-current)" != "amos2025-architecture-comparison" ]]; then
    echo "Wrong branch: $(git branch --show-current)" >&2
    exit 2
fi

CANDIDATE_COUNT=${BSK_RL_SMOKE_CANDIDATE_COUNT:-10}
OUTPUT_ROOT="/scratch/alpine/$USER/prospectus_rfi/smoke"
OUTPUT="$OUTPUT_ROOT/smoke_k${CANDIDATE_COUNT}_job${SLURM_JOB_ID}.csv"
mkdir -p "/scratch/alpine/$USER/job_output" "$OUTPUT_ROOT"

echo "branch=$(git branch --show-current) commit=$(git rev-parse HEAD)"
python -c "import bsk_rl; print('bsk_rl source:', bsk_rl.__file__)"
python -m pytest -q tests/unittest/prospectus_rfi tests/integration/prospectus_rfi
python -u examples/prospectus_rfi/smoke_test.py \
    --candidate-count "$CANDIDATE_COUNT" \
    --output "$OUTPUT"
echo "smoke_output=$OUTPUT"
