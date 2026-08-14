# Research Focus I reproducibility guide

This directory runs a matched comparison of the fixed-input monolithic MLP, the
target-set attention policy, the historical smallest-pointing-angle heuristic, and an
information-matched heuristic. Read `PROVENANCE.md` for the historical audit.

## What is fixed and what is swept

- Episode duration: 45,000 s for every N.
- Training catalog: integer N sampled uniformly and inclusively from 100 through 400 on
  every reset, followed by a new AMOS-2025 LEO target realization.
- Presented target candidates: K in {5, 10, 20}, explicitly separate from N.
- Imaging decision/action: fixed 100 s. Charge/downlink/desaturation remain 300/300/150 s.
- Initial scanner battery: uniform 20–60% of 1.8 MJ.
- Reward: alpha=0, meaning observation-only (not the AlphaZero algorithm).
- Re-imaging: disabled for the full episode.
- Exploratory campaign: one seed (10001) for each architecture/K combination, six
  48-hour runs or 288 node-hours. These runs do not measure training-seed variance.
- Confirmatory campaign retained in configuration: seeds 10001, 20001, and 30001 after
  the exploratory results identify comparisons worth replicating.

## Branch workflow on Alpine

Do not merge or copy these files manually onto `IA_Polaris_imaging_june10`. That branch's
name is historical but its current tip moved in 2026, and it lacks the maintained RLModule
interfaces used here. Pull the dedicated study branch, which reconstructs and tests the
AMOS 2025 physical configuration explicitly.

First push the study branch from the machine where it was prepared:

```bash
git switch amos2025-architecture-comparison
git push -u origin amos2025-architecture-comparison
```

On Alpine, preserve an existing AMOS 2026 checkout by creating a separate worktree:

```bash
cd /projects/$USER/bsk_rl
git fetch origin
git worktree add /projects/$USER/bsk_rl-rfi \
  origin/amos2025-architecture-comparison
cd /projects/$USER/bsk_rl-rfi
git switch -c amos2025-architecture-comparison \
  --track origin/amos2025-architecture-comparison
export BSK_RL_REPO_DIR=/projects/$USER/bsk_rl-rfi
git rev-parse HEAD
git status --short
```

The existing `/projects/$USER/.venv` may be reused if it imports Basilisk, bsk_rl,
Ray 2.35, Torch 2.4, pandas, scipy, matplotlib, and pyarrow. Verify before submission:

```bash
source /projects/$USER/.venv/bin/activate
cd "$BSK_RL_REPO_DIR"
export PYTHONPATH="$BSK_RL_REPO_DIR/src:$BSK_RL_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
python -c "import Basilisk, bsk_rl, ray, torch; print(ray.__version__, torch.__version__)"
python -c "import bsk_rl; print(bsk_rl.__file__)"  # Must resolve inside bsk_rl-rfi/src.
python -m pytest -q tests/unittest/prospectus_rfi tests/integration/prospectus_rfi
SMOKE_JOB=$(sbatch --parsable examples/prospectus_rfi/slurm/smoke_test.sbatch)
echo "$SMOKE_JOB"
```

The smoke job completes six full 45,000-second episodes on a compute node: random
initialized MLP, random initialized attention, and the historical heuristic at N=100
and N=400. Do not run this full simulation on an Alpine login node.

## Weights & Biases isolation

Training runs use a dedicated W&B project rather than the AMOS 2026 project:

- project: `amos2025-architecture-comparison`;
- final-run group: `rfi-alpha0-100s-candidate-sweep`;
- tuning group: `rfi-alpha0-100s-architecture-tuning`;
- run names: `amos2025-rfi-alpha0-100s__<architecture>_k<K>_seed<seed>`;
- local W&B files: `/scratch/alpine/$USER/prospectus_rfi/wandb/amos2025-architecture-comparison`.

Run IDs are deterministic, so resuming a checkpoint reconnects to the same W&B run.
Both training Slurm scripts set `BSK_RL_REQUIRE_WANDB=1`; a missing key or package stops
the job instead of silently producing an untracked policy. By default they use the
existing key at `/projects/$USER/bsk_rl/examples/wandb_key.txt`. Override it if needed:

```bash
export BSK_RL_WANDB_KEY_PATH=/projects/$USER/secure/wandb_key.txt
```

The project can be redirected deliberately with `BSK_RL_RFI_WANDB_PROJECT`, without
inheriting an unrelated `BSK_RL_WANDB_PROJECT` left over from an AMOS 2026 campaign.

## Recommended campaign order

The checked-in `*_selected.yaml` files are historically informed starting points. The
prospectus protocol calls for separate tuning before final training.

1. Submit equal tuning compute (12 × 8 h for each architecture):

```bash
TUNE_JOB=$(sbatch --parsable examples/prospectus_rfi/slurm/tune_equal_budget_8h.sbatch)
echo "$TUNE_JOB"
```

2. Validate the retained tuning checkpoints after training finishes:

```bash
VALIDATE_TUNE_JOB=$(sbatch --parsable --dependency=afterok:$TUNE_JOB \
  examples/prospectus_rfi/slurm/validate_tuning.sbatch)
echo "$VALIDATE_TUNE_JOB"
```

3. Collect validation rows and select one configuration per architecture:

```bash
ROOT=/scratch/alpine/$USER/prospectus_rfi
python examples/prospectus_rfi/tuning.py collect \
  --run-root "$ROOT/tuning" --architecture mlp \
  --output "$ROOT/tuning/mlp_validation.csv"
python examples/prospectus_rfi/tuning.py select \
  --table examples/prospectus_rfi/configs/mlp_tuning_table.csv \
  --validation "$ROOT/tuning/mlp_validation.csv" \
  --output-dir "$ROOT/tuning/selection/mlp"
python examples/prospectus_rfi/tuning.py collect \
  --run-root "$ROOT/tuning" --architecture attention \
  --output "$ROOT/tuning/attention_validation.csv"
python examples/prospectus_rfi/tuning.py select \
  --table examples/prospectus_rfi/configs/attention_tuning_table.csv \
  --validation "$ROOT/tuning/attention_validation.csv" \
  --output-dir "$ROOT/tuning/selection/attention"
```

4. Submit the six single-seed 48-hour candidate-sweep runs using the selected files:

```bash
export BSK_RL_MLP_CONFIG=$ROOT/tuning/selection/mlp/selected_from_tuning.yaml
export BSK_RL_ATTENTION_CONFIG=$ROOT/tuning/selection/attention/selected_from_tuning.yaml
TRAIN_JOB=$(sbatch --parsable \
  --export=ALL,BSK_RL_REPO_DIR,BSK_RL_MLP_CONFIG,BSK_RL_ATTENTION_CONFIG \
  examples/prospectus_rfi/slurm/train_candidate_sweep_48h.sbatch)
echo "$TRAIN_JOB"
```

5. Select the best held-out validation checkpoint for each final run:

```bash
VALIDATE_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB \
  --export=ALL,BSK_RL_REPO_DIR \
  examples/prospectus_rfi/slurm/validate_candidate_sweep.sbatch)
echo "$VALIDATE_JOB"
```

6. Launch the paired 100-episode-per-N evaluation after validation. The 192-task array
   splits each method/K/N cell into four blocks of 25 seeds while preserving identical
   seeds across methods:

```bash
MC_JOB=$(sbatch --parsable --dependency=afterok:$VALIDATE_JOB \
  --export=ALL,BSK_RL_REPO_DIR \
  examples/prospectus_rfi/slurm/evaluate_paired_mc.sbatch)
echo "$MC_JOB"
```

7. Produce statistics, SVG/PDF figures, and prospectus text after evaluation:

```bash
python examples/prospectus_rfi/analyze.py --input-root "$ROOT"
```

## Resume one interrupted run

Every periodic checkpoint is a complete RLlib Algorithm checkpoint. Resume with the same
architecture, K, seed, selected configuration, and a new 48-hour allocation:

```bash
python examples/prospectus_rfi/train.py \
  --architecture attention \
  --architecture-config "$BSK_RL_ATTENTION_CONFIG" \
  --candidate-count 10 --seed 10001 --wall-hours 48 --n-env-runners 28 \
  --output-root "$ROOT" \
  --resume "$ROOT/training/attention_k10_seed10001/checkpoints/final"
```

## Outputs

- `training/<method>_k<K>_seed<seed>/metadata.json`: commit, full configuration,
  observation/action/reward contract, parameter count, and W&B namespace.
- `training/<method>_k<K>_seed<seed>/wandb_run.json`: W&B project, group, stable run
  ID, local directory, and web URL.
- `training_metrics.csv` and `.jsonl`: raw environment steps, wall time, throughput, and
  RLlib training metrics.
- `validation_metrics.csv`: held-out physical metrics for retained checkpoints.
- `checkpoints/best_validation`: symlink to the selected validation checkpoint;
  `checkpoints/final`: final checkpoint.
- `evaluation/raw/*.csv` and optional Parquet counterparts: paired episode-level data.
- `analysis/summary_statistics.csv`: means, standard deviations, medians, and IQRs.
- `analysis/paired_differences.csv`: paired bootstrap intervals, Wilcoxon tests, Holm
  correction, and predeclared practical-equivalence classification.
- `figures/*.pdf` and `figures/*.svg`: training/validation curves, catalog-size results,
  paired differences, resource allocation, and computation cost.
- `prospectus_results.md`: generated numerical language and figure paths. It deliberately
  labels the one-seed campaign exploratory and does not assert architecture superiority.
