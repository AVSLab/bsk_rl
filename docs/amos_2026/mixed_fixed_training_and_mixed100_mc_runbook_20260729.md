# AMOS 2026 Mixed-Policy Training and Mixed-100 Evaluation

Date: 2026-07-29

## What is known now

- The local fixed-count mixed obs-v9 artifact is only an alpha-0.1 smoke
  configuration: 10 catalog targets, 2 candidates, W&B disabled, and no
  checkpoint.
- The local variable-count mixed obs-v9 artifact is also configuration-only:
  100--300 targets, 2 candidates, randomized orbital proportions, and no
  checkpoint.
- The Alpine audit found four checkpoint-bearing mixed obs-v9 runs. Two use a
  fixed 100-target count and two randomize the active count from 100--300. All
  four are alpha 0.1, use 10 candidates, and randomize catalog-level orbital
  proportions.
- The fixed-100 continuation has latest checkpoint 176 and training-selected
  best checkpoint 133. The original fixed-100 run has latest checkpoint 114
  and training-selected best checkpoint 59. All four are now copied locally.
- The older completed mixed-100 campaign is obs-v7 and is not compatible with
  the AMOS 2026 obs-v9 paper experiments.
- The previous mixed evaluator treated 50/30/20 as independent target-sampling
  probabilities, not guaranteed per-seed counts. The new exact-count option
  makes that distinction explicit.
- The new training sweep uses 100 targets, 10 candidates, and exactly 50 LEO,
  30 MEO, and 20 GEO targets in every training catalog.
- The new mixed-100 Monte Carlo campaign evaluates frozen LEO-trained policies,
  uses that same exact 50/30/20 composition, and scores every policy with the
  common alpha-1.0 delivered-ground-value objective.

## 1. Put the new scripts on Alpine

The complete sync-and-submit path is:

```bash
cd /Users/dahu1128/Repositories/bsk_rl
docs/amos_2026/sync_and_launch_mixed_fixed_training_and_mixed100_mc.sh
```

That command syncs the changed files, validates W&B and Python on Alpine, then
submits both workloads. The lower-level steps below remain useful for auditing,
rerunning, or submitting only one part.

If the branch changes have already been pushed, run this on Alpine:

```bash
cd /projects/$USER/bsk_rl
git fetch origin
git switch amos-2026-space-imaging
git pull --ff-only
source /projects/$USER/.venv/bin/activate
```

To transfer the current working-tree files directly from the Mac instead, run
this from the local repository:

```bash
cd /Users/dahu1128/Repositories/bsk_rl

rsync -av --relative \
  examples/train_Polaris_gat_full_actions_wandb.py \
  examples/updated_policy_evaluation.py \
  examples/amos_2026/audit_mixed_v9_training_runs.py \
  examples/amos_2026/evaluate_gat_reward_sweep_mc.py \
  examples/amos_2026/sbatch_evaluate_gat_reward_sweep_mc_10seeds.sh \
  examples/amos_2026/sbatch_train_polaris_gat_full_actions_mixed_fixed_100targets_reward_sweep_48h.sh \
  examples/amos_2026/submit_mixed_fixed_100targets_reward_sweep_training.sh \
  examples/amos_2026/submit_gat_reward_sweep_mc_mixed_100targets_45000s_0to99.sh \
  docs/amos_2026/package_mixed_v9_policy_audit.sh \
  dahu1128@login-ci5.rc.colorado.edu:/projects/dahu1128/bsk_rl/
```

## 2. Audit and copy existing mixed V9 policies

Run on Alpine:

```bash
cd /projects/$USER/bsk_rl
source /projects/$USER/.venv/bin/activate

INCLUDE_CHECKPOINTS=1 \
  bash docs/amos_2026/package_mixed_v9_policy_audit.sh
```

The audit writes a stable transfer path:

```text
/scratch/alpine/$USER/amos2026_policy_audits/mixed_v9_policy_audit_latest.tgz
```

Then run on the Mac:

```bash
cd /Users/dahu1128/Repositories/bsk_rl
bash docs/amos_2026/pull_mixed_v9_policy_audit.sh 20260729
```

Inspect these first:

```text
mixed_v9_training_inventory.csv
summary.json
copy_candidates.txt
fixed100_checkpoints/
```

The inventory distinguishes fixed target count from variable target count,
fixed weights from randomized weights, candidate count, alpha, W&B group,
checkpoint iteration, and final progress values.

## 3. Verify W&B and submit mixed-fixed training

Run on Alpine:

```bash
cd /projects/$USER/bsk_rl
source /projects/$USER/.venv/bin/activate

test -s examples/wandb_key.txt
python3 -c "import wandb; print(wandb.__version__)"

BSK_RL_DRY_RUN=1 \
  bash examples/amos_2026/submit_mixed_fixed_100targets_reward_sweep_training.sh 4

bash examples/amos_2026/submit_mixed_fixed_100targets_reward_sweep_training.sh 4
```

This submits eight 48-hour array tasks:

```text
alpha: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0
tags:  00d100i, 10d90i, 20d80i, 30d70i,
       40d60i, 50d50i, 75d25i, 100d00i
```

W&B destination:

```text
project: amos2026-bsk-rl
group: polaris-gat-full-actions-obs-v9-mixed-fixed-50leo30meo20geo-100targets-reward-sweep
```

Monitor:

```bash
squeue -u $USER -n gat_mix100_fixed_sweep
ls -1t /scratch/alpine/$USER/job_output/gat_mix100_fixed_sweep_* | head
tail -f "$(ls -1t /scratch/alpine/$USER/job_output/gat_mix100_fixed_sweep_* | head -n 1)"
```

## 4. Submit the missing mixed-100 Monte Carlo campaign

This campaign uses the already-trained LEO policies, not the new mixed-trained
policies. Run on Alpine:

```bash
cd /projects/$USER/bsk_rl
source /projects/$USER/.venv/bin/activate

bash \
  examples/amos_2026/submit_gat_reward_sweep_mc_mixed_100targets_45000s_0to99.sh \
  5
```

The default is twelve policies by 100 seeds, or 1,200 evaluations. Ten seed blocks
are dependency-chained, and up to five policy tasks run concurrently in the
active block.

For an alpha-0.1-only campaign instead:

```bash
BSK_RL_MC_POLICY_TAGS=10d90i \
  bash examples/amos_2026/submit_gat_reward_sweep_mc_mixed_100targets_45000s_0to99.sh 1
```

Monitor:

```bash
squeue -u $USER | grep gat_mc_mix_100t
find /scratch/alpine/$USER/amos2026_mc \
  -maxdepth 1 -type d \
  -name 'gat_full_actions_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_*' \
  -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort
```

## 5. Analyze after all MC jobs finish

Run on Alpine:

```bash
cd /projects/$USER/bsk_rl
source /projects/$USER/.venv/bin/activate

ROOT=$(find /scratch/alpine/$USER/amos2026_mc \
  -maxdepth 1 -type d \
  -name 'gat_full_actions_eval_100d00i_mixed_exact50LEO30MEO20GEO_100targets_*' \
  -print | sort | tail -n 1)

export MPLBACKEND=Agg
export MPLCONFIGDIR=/scratch/alpine/$USER/.cache/matplotlib
mkdir -p "$MPLCONFIGDIR"

python3 examples/amos_2026/analyze_gat_reward_sweep_mc.py \
  --input-root "$ROOT" --expected-seeds 0:100

python3 examples/amos_2026/analyze_gat_reward_sweep_mc_detailed.py \
  --input-root "$ROOT" --expected-seeds 0:100
```

Package the paper-facing outputs:

```bash
PACKAGE=/scratch/alpine/$USER/amos2026_mc/mixed100_exact_results_$(date +%Y%m%d).tgz

(
  cd "$ROOT"
  find . -type f \
    \( -name 'summary_by_policy.csv' \
    -o -name 'per_run.csv' \
    -o -name 'missing_runs.csv' \
    -o -name 'failed_runs.csv' \
    -o -name 'analysis_report.json' \
    -o -path '*/analysis_detailed/*' \) \
    -print0 | tar --null -czf "$PACKAGE" --files-from -
)

echo "$PACKAGE"
```

Once copied locally, compare the existing mixed checkpoint, the new
mixed-trained sweep, and the LEO-trained cross-regime campaign using common
seeds and the same alpha-1.0 evaluation score. Do not rank training runs from
W&B training reward alone.

## Launch record

Submitted 2026-07-29:

```text
mixed-fixed training array: 30568059 (tasks 0--7, at most 4 concurrent)
mixed-100 MC blocks:        30568061--30568070
MC task shape:              12 policies x 100 seeds
MC concurrency:             at most 5 policy tasks in each active seed block
```
