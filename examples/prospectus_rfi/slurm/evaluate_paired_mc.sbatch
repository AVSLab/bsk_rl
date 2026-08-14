#!/usr/bin/env bash
# 4 methods x 3 K values x 4 N values x 4 blocks of 25 paired seeds.
#SBATCH --account=ucb550_asc2
#SBATCH --job-name=rfi_paired_mc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-191%32
#SBATCH --partition=acpu
#SBATCH --mem=80G
#SBATCH --constraint=epyc-7713
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/%x_%A_%a.out
#SBATCH --mail-type=FAIL,END
#SBATCH --qos=cpu-normal

set -euo pipefail
METHODS=(mlp attention heuristic_historical heuristic_matched)
KS=(5 10 20)
NS=(100 200 300 400)
INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
BLOCK=$((INDEX % 4))
N_INDEX=$(((INDEX / 4) % 4))
K_INDEX=$(((INDEX / 16) % 3))
METHOD_INDEX=$(((INDEX / 48) % 4))
METHOD=${METHODS[$METHOD_INDEX]}
K=${KS[$K_INDEX]}
N=${NS[$N_INDEX]}
SEED_START=$((700000 + BLOCK * 25))

module purge
REPO_DIR=${BSK_RL_REPO_DIR:-/projects/$USER/bsk_rl}
source "$REPO_DIR/examples/prospectus_rfi/slurm/alpine_runtime.sh"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
ROOT="/scratch/alpine/$USER/prospectus_rfi"
OUT="$ROOT/evaluation/raw/${METHOD}_k${K}_n${N}_block${BLOCK}.csv"
mkdir -p "$(dirname "$OUT")" "/scratch/alpine/$USER/job_output"

CHECKPOINT_ARGS=()
if [[ "$METHOD" == "mlp" || "$METHOD" == "attention" ]]; then
    CHECKPOINT="$ROOT/training/${METHOD}_k${K}_seed10001/checkpoints/best_validation"
    [[ -e "$CHECKPOINT" ]] || { echo "Missing $CHECKPOINT" >&2; exit 3; }
    CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT")
fi

python -u examples/prospectus_rfi/evaluate.py \
    --method "$METHOD" \
    --candidate-count "$K" \
    --catalog-sizes "$N" \
    --episodes 25 \
    --seed-start "$SEED_START" \
    --output "$OUT" \
    "${CHECKPOINT_ARGS[@]}"
