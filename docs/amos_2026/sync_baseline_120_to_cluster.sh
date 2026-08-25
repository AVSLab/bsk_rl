#!/usr/bin/env bash

# Create or validate an isolated AMOS 2026 cluster checkout, then synchronize
# only the source files needed by the 120-target baseline Monte Carlo campaign.
# The existing /projects/$USER/bsk_rl checkout is never switched or modified.

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

remote_host=${AMOS2026_REMOTE_HOST:-dahu1128@login.rc.colorado.edu}
remote_base=${AMOS2026_REMOTE_BASE:-/projects/dahu1128/bsk_rl}
remote_checkout=${AMOS2026_REMOTE_CHECKOUT:-/projects/dahu1128/bsk_rl_amos2026}
remote_venv=${AMOS2026_REMOTE_VENV:-/projects/dahu1128/.venv}
branch=${AMOS2026_BRANCH:-amos-2026-space-imaging}

echo "Preparing isolated cluster checkout: $remote_host:$remote_checkout"
ssh "$remote_host" bash -s -- "$remote_base" "$remote_checkout" "$branch" <<'REMOTE_SETUP'
set -euo pipefail
base=$1
checkout=$2
branch=$3

if [[ ! -d "$base/.git" && ! -f "$base/.git" ]]; then
    echo "Existing cluster repository not found: $base" >&2
    exit 3
fi

if [[ ! -e "$checkout" ]]; then
    origin=$(git -C "$base" remote get-url origin)
    git clone --single-branch --branch "$branch" "$origin" "$checkout"
fi

if [[ ! -d "$checkout/.git" && ! -f "$checkout/.git" ]]; then
    echo "Cluster path is not a Git checkout: $checkout" >&2
    exit 3
fi

actual_branch=$(git -C "$checkout" branch --show-current)
if [[ "$actual_branch" != "$branch" ]]; then
    echo "Refusing to use $checkout: branch is '$actual_branch', expected '$branch'." >&2
    exit 3
fi

if git -C "$checkout" diff --quiet && git -C "$checkout" diff --cached --quiet; then
    git -C "$checkout" pull --ff-only origin "$branch"
else
    echo "NOTE: $checkout has tracked working-tree changes; skipping git pull before rsync."
fi
REMOTE_SETUP

rsync -av --relative \
    src/bsk_rl/act/discrete_actions.py \
    src/bsk_rl/sim/dyn.py \
    examples/updated_policy_evaluation.py \
    examples/amos_2026/evaluate_heuristic_mc.py \
    examples/amos_2026/audit_paired_heuristic_mc.py \
    examples/amos_2026/sbatch_evaluate_heuristics_mc_10seeds.sh \
    examples/amos_2026/submit_heuristics_mc_mixed_100targets_45000s_0to99.sh \
    examples/amos_2026/sbatch_analyze_heuristic_mc.sh \
    examples/amos_2026/sbatch_evaluate_baseline_mc_120targets_oneorbit.sh \
    examples/amos_2026/submit_baseline_mc_mixed_120targets_oneorbit_0to99.sh \
    examples/amos_2026/README.md \
    "$remote_host:$remote_checkout/"

ssh "$remote_host" bash -s -- "$remote_checkout" "$remote_venv" "$branch" <<'REMOTE_VALIDATE'
set -euo pipefail
checkout=$1
venv=$2
branch=$3

test "$(git -C "$checkout" branch --show-current)" = "$branch"
source "$venv/bin/activate"
export PYTHONPATH="$checkout/src${PYTHONPATH:+:$PYTHONPATH}"

python3 -m py_compile \
    "$checkout/examples/amos_2026/evaluate_heuristic_mc.py" \
    "$checkout/examples/amos_2026/audit_paired_heuristic_mc.py" \
    "$checkout/examples/updated_policy_evaluation.py" \
    "$checkout/src/bsk_rl/act/discrete_actions.py" \
    "$checkout/src/bsk_rl/sim/dyn.py"
bash -n "$checkout/examples/amos_2026/sbatch_evaluate_baseline_mc_120targets_oneorbit.sh"
bash -n "$checkout/examples/amos_2026/submit_baseline_mc_mixed_120targets_oneorbit_0to99.sh"
bash -n "$checkout/examples/amos_2026/sbatch_evaluate_heuristics_mc_10seeds.sh"
bash -n "$checkout/examples/amos_2026/submit_heuristics_mc_mixed_100targets_45000s_0to99.sh"
bash -n "$checkout/examples/amos_2026/sbatch_analyze_heuristic_mc.sh"

imported=$(python3 -c 'import pathlib, bsk_rl; print(pathlib.Path(bsk_rl.__file__).resolve())')
case "$imported" in
    "$checkout"/src/bsk_rl/*) ;;
    *)
        echo "bsk_rl imports from $imported instead of $checkout/src." >&2
        exit 3
        ;;
esac

echo "Cluster checkout validated:"
echo "  path:   $checkout"
echo "  branch: $(git -C "$checkout" branch --show-current)"
echo "  commit: $(git -C "$checkout" rev-parse --short HEAD)"
echo "  import: $imported"
REMOTE_VALIDATE

echo
echo "Synchronization complete. On the cluster, submit with:"
echo "  cd $remote_checkout"
echo "  BSK_RL_CLUSTER_WORKDIR=$remote_checkout bash examples/amos_2026/submit_baseline_mc_mixed_120targets_oneorbit_0to99.sh 2"
echo "For the paper-matched 100-target/two-orbit heuristic campaign, use:"
echo "  BSK_RL_CLUSTER_WORKDIR=$remote_checkout bash examples/amos_2026/submit_heuristics_mc_mixed_100targets_45000s_0to99.sh 4"
