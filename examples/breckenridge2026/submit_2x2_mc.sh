#!/bin/bash

# Submit only the missing mixed-trained row of the 2x2 comparison.
#
# Usage:
#   bash examples/breckenridge2026/submit_2x2_mc.sh [max-concurrent-per-cell]
#
# The existing LEO-trained Monte Carlos are retained as published baselines.
# Two independent arrays are submitted, with no Slurm dependencies:
#   mixed_trained__leo_eval
#   mixed_trained__mixed_eval

set -euo pipefail

MAX_CONCURRENT=${1:-10}

if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "max-concurrent-per-cell must be a positive integer" >&2
    exit 2
fi

cd /projects/$USER/bsk_rl
source /projects/$USER/.venv/bin/activate
VENV_PYTHON=/projects/$USER/.venv/bin/python

source examples/breckenridge2026/alpine_runtime.sh
configure_breckenridge_alpine_runtime "$VENV_PYTHON"

MIXED_POLICY=${BRECK_MC_MIXED_POLICY:-$PWD/policies/breckenridge2026_mixed_10d90i/checkpoint_000160}
CAMPAIGN_ID=${BRECK_MC_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${BRECK_MC_OUTPUT_ROOT:-/scratch/alpine/$USER/breckenridge2026_mc/mixed_trained_row_10d90i_${CAMPAIGN_ID}}
MANIFEST=${BRECK_MC_MANIFEST:-$OUTPUT_ROOT/breckenridge2026_2x2_manifest.json}
TIME_LIMIT=${BRECK_MC_TIME:-02:00:00}
MEMORY=${BRECK_MC_MEM:-16G}
CPUS_PER_TASK=${BRECK_MC_CPUS_PER_TASK:-4}
ARRAY_SPEC="0-99%${MAX_CONCURRENT}"

mkdir -p "$OUTPUT_ROOT" /scratch/alpine/$USER/job_output

if [[ ! -f "$MIXED_POLICY/learner_group/learner/rl_module/inspector/module_state.pt" ]]; then
    echo "Bundled mixed-trained policy is missing or incomplete: $MIXED_POLICY" >&2
    exit 1
fi

echo "Checking the bundled policy with the Alpine virtual environment..."
"$VENV_PYTHON" - "$MIXED_POLICY" <<'PY'
from pathlib import Path
import sys

from ray.rllib.core.rl_module.rl_module import RLModule

checkpoint = Path(sys.argv[1])
module_dir = checkpoint / "learner_group" / "learner" / "rl_module" / "inspector"
module = RLModule.from_checkpoint(module_dir)
print(f"Loaded {type(module).__name__} from {module_dir}")
PY

if [[ -f "$MANIFEST" ]]; then
    echo "Reusing frozen manifest: $MANIFEST"
else
    "$VENV_PYTHON" -u examples/breckenridge2026/prepare_mc_manifest.py \
        --mixed-policy "$MIXED_POLICY" \
        --output "$MANIFEST"
fi

cells=(
    mixed_trained__leo_eval
    mixed_trained__mixed_eval
)

echo
echo "Submitting two independent mixed-trained Breckenridge MC arrays"
echo "  output root: $OUTPUT_ROOT"
echo "  manifest:    $MANIFEST"
echo "  seeds:       0-99 in every cell"
echo "  array limit: $MAX_CONCURRENT tasks per cell"
echo "  dependencies: none"
echo

for cell in "${cells[@]}"; do
    job_name="br26_${cell}"
    job_id=$(sbatch \
        --parsable \
        --job-name="$job_name" \
        --array="$ARRAY_SPEC" \
        --time="$TIME_LIMIT" \
        --mem="$MEMORY" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --export=ALL,BRECK_MC_MANIFEST="$MANIFEST",BRECK_MC_CELL="$cell",BRECK_MC_OUTPUT_ROOT="$OUTPUT_ROOT" \
        examples/breckenridge2026/sbatch_mc_cell.sh)
    echo "Submitted $cell as job $job_id"
done

echo
echo "Monitor:"
echo "  squeue -u $USER"
echo
echo "Audit and summarize after both arrays finish:"
echo "  python3 examples/breckenridge2026/audit_mc_campaign.py --input-root \"$OUTPUT_ROOT\""
echo "  python3 examples/breckenridge2026/summarize_2x2_mc.py --input-root \"$OUTPUT_ROOT\""
