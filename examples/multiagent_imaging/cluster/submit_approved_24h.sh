#!/usr/bin/env bash
# Submit one explicitly approved arm; called through Termius after hash verification.
set -euo pipefail
test "${BSK_DIRECT_24H_APPROVED:?Explicit long-launch approval required}" = 1
: "${BSK_PROJECT_ROOT:?Immutable verified source required}"
: "${BSK_OUTPUT:?Fresh diagnostic path required}"
: "${BSK_RL_WANDB_KEY_PATH:?Existing private key-file path required}"
: "${BSK_APPROVED_WALL_S:?Reviewed remaining allocation seconds required}"
case "${BSK_ARM:?Information variant required}" in
  catalog_tx) BSK_LABEL=with_communication;;
  geometry_only) BSK_LABEL=without_communication;;
  *) exit 2;;
esac
[[ "$BSK_APPROVED_WALL_S" =~ ^[0-9]+$ ]]
test "$BSK_APPROVED_WALL_S" -ge 10800
test "$BSK_APPROVED_WALL_S" -le 86400
test "$((BSK_APPROVED_WALL_S % 60))" = 0
test "${BSK_OUTPUT##*/}" = diagnostic
BSK_PARENT="${BSK_OUTPUT%/diagnostic}"
test ! -e "$BSK_PARENT"
test -f "$BSK_PROJECT_ROOT/completion-release.json"
test -f "$BSK_PROJECT_ROOT/examples/multiagent_imaging/cluster/approved_24h.slurm"
printf -v BSK_WALL '%02d:%02d:00' "$((BSK_APPROVED_WALL_S / 3600))" "$((BSK_APPROVED_WALL_S % 3600 / 60))"
mkdir "$BSK_PARENT"
# Explicit values avoid dependence on a prior subshell's exported environment.
# This contains paths and reviewed settings, never credential contents.
sbatch --parsable --export="ALL,BSK_DIRECT_24H_APPROVED=1,BSK_ARM=$BSK_ARM,BSK_PROJECT_ROOT=$BSK_PROJECT_ROOT,BSK_OUTPUT=$BSK_OUTPUT,BSK_RL_WANDB_KEY_PATH=$BSK_RL_WANDB_KEY_PATH,BSK_APPROVED_WALL_S=$BSK_APPROVED_WALL_S" \
  --time="$BSK_WALL" --job-name="multiagent-${BSK_LABEL}-24h" \
  --chdir="$BSK_PROJECT_ROOT" --output="$BSK_PARENT/slurm-%j.log" \
  "$BSK_PROJECT_ROOT/examples/multiagent_imaging/cluster/approved_24h.slurm" \
  > "$BSK_PARENT/submission.log"
cat "$BSK_PARENT/submission.log"
