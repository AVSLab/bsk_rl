# Research Focus I reproducibility guide

This directory runs a matched comparison of the fixed-input monolithic MLP, the
target-set attention policy, the historical smallest-pointing-angle heuristic, and an
information-matched heuristic. Read `PROVENANCE.md` for the historical audit.

For the live Alpine campaign status, decision gates, and detailed prompts for each
remaining task, see `CAMPAIGN_TODO_PROMPTS.md`.

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
- Alpine resources: `acpu` with the exact `epyc-7713` feature; `cpu-normal` for jobs
  up to 24 hours and `cpu-long` for the 48-hour final runs.
- Runtime ABI: `slurm/alpine_runtime.sh` uses the existing GCC 14.2 installation at
  `/curc/sw/install/gcc/14.2.0` and verifies `GLIBCXX_3.4.29` before activating the
  shared virtual environment. The compiler is no longer registered as an Lmod module.
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

## Dependency-free AMOS 2025 heuristic Monte Carlo

The standalone closest-angle campaign is separate from the later paired-policy study.
It runs the historical full-catalog smallest-pointing-angle heuristic with the resource
shield at N in {100, 200, 400}, using the exact scenario seeds 0 through 99 at every N.
The study environment still has a 45,000 s horizon, fixed 100 s imaging actions,
observation-only alpha=0 reward, no re-imaging, and randomized 20--60% initial battery.
K=10 remains the action-interface size, but the historical heuristic intentionally sees
the full eligible target catalog; this information advantage is recorded in every row.

After the smoke job succeeds, submit all 300 episodes as 30 independent array tasks
(ten seeds per task). The `%12` throttle controls concurrency and is not a dependency:

```bash
cd "$BSK_RL_REPO_DIR"
git pull --ff-only
MC_SUBMISSION=$(bash examples/prospectus_rfi/submit_amos2025_heuristic_mc.sh 12)
printf '%s\n' "$MC_SUBMISSION"
```

The submission prints both `JOB_ID` and a timestamped `OUTPUT_ROOT`. Save the output
root, then monitor the job and logs with:

```bash
squeue -j <JOB_ID>
tail -f /scratch/alpine/$USER/job_output/rfi_heur_mc_<JOB_ID>_<ARRAY_TASK>.out
```

Each episode is committed independently under `raw/n100`, `raw/n200`, or `raw/n400`,
with adjacent JSON metadata and a separate status record. Re-submitting the same array
with the same `BSK_RL_HEURISTIC_MC_OUTPUT_ROOT` safely skips completed seeds. After all
tasks finish, validate the complete 300-pair design and create analysis-ready tables:

```bash
python examples/prospectus_rfi/collect_heuristic_mc.py \
  --input-root <OUTPUT_ROOT>
```

The collector refuses incomplete, duplicate, or mislabeled campaigns. Successful
collection writes `analysis/episodes_combined.csv`, optional Parquet, summary statistics,
and `analysis/completion.json`. No W&B login is required for this deterministic baseline.

The completed August 15 campaign can be pulled from a terminal on the local Mac and
validated in one command. The destination is under the repository's git-ignored
`results/prospectus_rfi/cluster_downloads/` tree, separate from source code and from the
historical checkpoint archive:

```bash
cd /Users/dahu1128/Repositories/bsk_rl
bash examples/prospectus_rfi/pull_amos2025_heuristic_mc.sh \
  amos2025_closest_angle_100s_20260815T183838Z
```

CURC may request the user's password or interactive authentication. On success, the
validated episode table is at
`results/prospectus_rfi/cluster_downloads/heuristic_mc/amos2025_closest_angle_100s_20260815T183838Z/analysis/episodes_combined.csv`.

If validation reports missing pairs, the original three-hour jobs exhausted their wall
time while processing ten seeds serially. Preserve every completed file and recover only
the missing pairs as independent one-seed tasks. On Alpine:

```bash
cd /projects/$USER/bsk_rl-rfi
git pull --ff-only
export BSK_RL_REPO_DIR=/projects/$USER/bsk_rl-rfi
HEURISTIC_ROOT=/scratch/alpine/$USER/prospectus_rfi/heuristic_mc/amos2025_closest_angle_100s_20260815T183838Z

bash examples/prospectus_rfi/submit_missing_amos2025_heuristic_mc.sh \
  "$HEURISTIC_ROOT" 30
```

The recovery submitter scans the existing raw files and schedules only missing
`(catalog size, seed)` pairs. Each gets its own three-hour allocation, has no dependency,
and runs with positive Slurm nice so current training remains higher priority. After the
recovery finishes, rerun the local pull helper; rsync transfers only new/changed files and
the collector must then validate exactly 300 episodes.

## Frozen AMOS 2025 policy transfer Monte Carlo

This separate campaign evaluates the archived best alpha=0 policy without retraining.
The exact artifact is iteration 427 of
`wGAE_balance0d100i_largepenalties_smallbatch_obs2`. The module has an 87-value input,
13 actions, `[1024, 1024]` separate actor/value MLPs with tanh, 2,293,774 parameters,
and inspector-state SHA-256
`6db5bcd4fda20205977dfab377441f625051ef9e9dfaebde5e8db5ec1ab0e2c4`.

The evaluator restores its exact historical observation order and normalization while
using the current heuristic's physical evaluation environment and shield. The policy was
trained at N=100 with 300 s imaging, 180 s downlink, and 10--40% initial battery. The
transfer test instead uses N in {100,200,400}, fixed 100 s imaging, 300 s downlink, and
20--60% initial battery. It must be described as a frozen-policy transfer baseline, not
as a policy trained for 100-second actions.

The self-contained inspector module is only 8.8 MB. From a terminal on the local Mac,
copy exactly its three files to persistent project storage on Alpine:

```bash
LOCAL_AMOS2025_MODULE='/Users/dahu1128/rllib_results/july_results/july30rllib_results/aug13_wGAE_smallbatch_halfnetwork_largepenalties_smallerICbattery_restrictedResources_obsv2_1e-6lr_0.15cp_gamma9997_0d100i_1755107128.629914/aug13_wGAE_smallbatch_halfnetwork_largepenalties_smallerICbattery_restrictedResources_obsv2_1e-6lr_0.15cp_gamma9997_0d100i.out_0/checkpoint_best/learner_group/learner/rl_module/inspector'
REMOTE_AMOS2025_MODULE='/projects/dahu1128/policy_artifacts/amos2025_alpha0_best_iter427/inspector'
ssh dahu1128@login.rc.colorado.edu "mkdir -p '$REMOTE_AMOS2025_MODULE'"
rsync -av --checksum "$LOCAL_AMOS2025_MODULE/" \
  "dahu1128@login.rc.colorado.edu:$REMOTE_AMOS2025_MODULE/"
```

Then, on an Alpine login node, update only the dedicated worktree and submit 300
one-episode array tasks. They have no dependencies and use a positive Slurm nice value,
so they neither alter nor cancel the active training jobs and are lower scheduler
priority than those jobs:

```bash
module unload slurm/blanca 2>/dev/null || true
module load slurm/alpine
cd /projects/$USER/bsk_rl-rfi
git pull --ff-only
export BSK_RL_REPO_DIR=/projects/$USER/bsk_rl-rfi
export BSK_RL_AMOS2025_POLICY_CHECKPOINT=/projects/$USER/policy_artifacts/amos2025_alpha0_best_iter427/inspector

sha256sum "$BSK_RL_AMOS2025_POLICY_CHECKPOINT/module_state.pt"
sbatch --test-only \
  --export=ALL,BSK_RL_REPO_DIR,BSK_RL_AMOS2025_POLICY_CHECKPOINT \
  examples/prospectus_rfi/slurm/evaluate_amos2025_legacy_policy_mc.sbatch

POLICY_MC_SUBMISSION=$(bash \
  examples/prospectus_rfi/submit_amos2025_legacy_policy_mc.sh 30)
printf '%s\n' "$POLICY_MC_SUBMISSION"
```

The submission prints `JOB_ID` and `OUTPUT_ROOT`. Save both. Array task 0 is N=100,
seed 0; task 100 is N=200, seed 0; and task 200 is N=400, seed 0. Monitor without
touching any training allocation:

```bash
squeue -j <JOB_ID> -o "%.20i %.16j %.10q %.2t %.10M %R"
tail -f /scratch/alpine/$USER/job_output/rfi_oldpol_<JOB_ID>_0.out
```

Each task atomically writes one CSV, optional Parquet, metadata JSON, and status JSON.
Reusing the same `BSK_RL_LEGACY_POLICY_MC_OUTPUT_ROOT` safely skips completed episodes.
After all 300 tasks finish, validate and combine them:

```bash
python examples/prospectus_rfi/collect_legacy_policy_mc.py \
  --input-root <OUTPUT_ROOT>
```

## Saturation-aware acquisition timelines

The one-row episode files above contain final totals, not the time at which each
illuminated target was acquired. Consequently, final-count CSVs cannot be used to infer
whether one method reaches a common plateau earlier. Record that evidence with a paired
timeline replay after both 300-episode campaigns have completed:

```bash
cd /projects/$USER/bsk_rl-rfi
git pull --ff-only
export BSK_RL_REPO_DIR=/projects/$USER/bsk_rl-rfi
export BSK_RL_AMOS2025_POLICY_CHECKPOINT=/projects/$USER/policy_artifacts/amos2025_alpha0_best_iter427/inspector
export BSK_RL_HEURISTIC_MC_OUTPUT_ROOT=/scratch/alpine/$USER/prospectus_rfi/heuristic_mc/amos2025_closest_angle_100s_20260815T183838Z
export BSK_RL_LEGACY_POLICY_MC_OUTPUT_ROOT=/scratch/alpine/$USER/prospectus_rfi/legacy_policy_mc/amos2025_alpha0_300s_to_100s_20260817T004436Z

BSK_RL_TIMELINE_SCAN_ONLY=1 \
  bash examples/prospectus_rfi/submit_acquisition_timeline_mc.sh \
  "$BSK_RL_HEURISTIC_MC_OUTPUT_ROOT" \
  "$BSK_RL_LEGACY_POLICY_MC_OUTPUT_ROOT" 30

TIMELINE_SUBMISSION=$(bash \
  examples/prospectus_rfi/submit_acquisition_timeline_mc.sh \
  "$BSK_RL_HEURISTIC_MC_OUTPUT_ROOT" \
  "$BSK_RL_LEGACY_POLICY_MC_OUTPUT_ROOT" 30)
printf '%s\n' "$TIMELINE_SUBMISSION"
```

The 600 tasks are dependency-free and independently map the two methods, three catalog
sizes, and seeds 0--99. They use positive Slurm nice and do not cancel, modify, or depend
on training jobs. Every task replays one accepted scenario, records every decision epoch,
and verifies the scenario fingerprint plus final observation/downlink counts against the
existing raw row before atomically writing under `<CAMPAIGN_ROOT>/timeline/`. The original
episode CSV is never overwritten. Re-running the submitter schedules only missing
timeline sidecars.

The analysis forward-fills those irregular decision epochs onto a shared 100-second grid
(451 points from 0 through 45,000 seconds) without interpolation or smoothing. The
15,000-, 30,000-, and 45,000-second values are table endpoints, while the figures use the
entire grid. After the timeline tasks finish, run on Alpine or after pulling both campaign
roots locally:

```bash
python examples/prospectus_rfi/analyze_acquisition_timelines.py \
  --heuristic-root "$BSK_RL_HEURISTIC_MC_OUTPUT_ROOT" \
  --policy-root "$BSK_RL_LEGACY_POLICY_MC_OUTPUT_ROOT" \
  --output-root /scratch/alpine/$USER/prospectus_rfi/acquisition_timeline_analysis/amos2025_frozen_policy_vs_heuristic
```

Outputs include the raw 100-second analysis grid, per-seed normalized acquisition AUC,
time to 50/80/90/95% of each episode's final illuminated count, checkpoint tables,
paired bootstrap intervals and Holm-adjusted tests, and PDF/SVG figures for absolute
cumulative counts, empirical plateau fractions, and paired policy-minus-heuristic curves.
The plots show 100-seed means with transparent 95% bootstrap bands. Final-plateau
normalization is explicitly an empirical saturation diagnostic; it does not assume that
every unobserved target was geometrically unreachable, and the analysis does not assume
in advance that the frozen policy is faster.

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

## Memory-safe N=100--200 v2 campaign

The original N=100--400, 28-runner candidate sweep suffered Ray worker OOM deaths and
must be treated as a failed pilot. Do not resume its checkpoints for the prospectus
comparison. The v2 campaign has a separate configuration, scratch root, W&B group, and
W&B run-ID prefix, so it cannot overwrite or silently resume the pilot.

The v2 training environment samples an integer catalog size uniformly and inclusively
from 100 through 200 at every episode reset. Each runner physically instantiates 200
target spacecraft. Training uses 12 environment runners, 16 allocated CPUs, one
PyTorch/BLAS thread per process, and 230 GiB on an Alpine 240-GiB node. Only the
`inspector` policy is updated; the target spacecraft retain their passive drift module.
The AMOS physical/action configuration remains otherwise unchanged, including 45,000 s
episodes and fixed 100 s image actions.

Do not submit the six long runs until both K=20 stress tasks complete one full PPO
iteration and pass. The stress allocation allows up to four hours because constructing
12 independent 200-target Basilisk environments took about one hour on Alpine; each
task exits early as soon as iteration 1 and its final checkpoint are complete. On
Alpine:

```bash
module unload slurm/blanca 2>/dev/null || true
module load slurm/alpine
cd /projects/$USER/bsk_rl-rfi
git pull --ff-only

export BSK_RL_REPO_DIR=/projects/$USER/bsk_rl-rfi
export BSK_RL_WANDB_KEY_PATH=/projects/$USER/bsk_rl/examples/wandb_key.txt

STRESS_JOB=$(sbatch --parsable \
  --export=ALL,BSK_RL_REPO_DIR,BSK_RL_WANDB_KEY_PATH \
  examples/prospectus_rfi/slurm/stress_candidate_sweep_memorysafe_2h.sbatch)
echo "Stress job: $STRESS_JOB"
```

After both array tasks leave `squeue`, run the mandatory gate:

```bash
bash examples/prospectus_rfi/audit_memorysafe_stress.sh "$STRESS_JOB"
```

The audit requires two `COMPLETED` task states, final checkpoints, positive environment
steps, and no detected Ray OOM, worker-death, or actor-unavailable messages. If it prints
`PASS`, submit all six configurations and their task-correlated continuation chains:

```bash
SUBMISSION=$(bash \
  examples/prospectus_rfi/submit_memorysafe_candidate_sweep_24h.sh \
  "$STRESS_JOB")
printf '%s\n' "$SUBMISSION"
CLEANUP_JOB=$(awk -F= '/^CLEANUP_JOB=/{print $2}' <<< "$SUBMISSION")
echo "Cleanup array: $CLEANUP_JOB"
```

The submission helper reports `SEGMENT0_JOB`, `SEGMENT1_JOB`, and `CLEANUP_JOB`.
`aftercorr` makes every policy's continuation depend only on the corresponding policy
task, so one failed architecture/K cell does not prevent the other independent cells
from continuing. The three allocation limits are 24, 24, and 6 hours; guarded training
caps are 22, 22, and at most the remaining 5 hours. Advance signals and the iteration
duration guard leave time for a complete final checkpoint.

Outputs are under:

```text
/scratch/alpine/$USER/prospectus_rfi/memorysafe_100_200_v2
```

W&B uses project `amos2025-architecture-comparison`, group
`rfi-alpha0-100s-n100-200-memorysafe-v2`, and prefix
`amos2025-rfi-alpha0-100s-n100-200-v2`. Checkpoint selection uses held-out seeds at
N=100, 150, and 200. N=300 and 400 remain predeclared out-of-distribution evaluations
and are not part of training-runner memory load.

After the reported cleanup array completes successfully, submit validation with:

The first serial validation array was retired because 90 full episodes per
configuration exceeded its 24-hour allocation. The replacement is a
restartable, one-episode-per-task campaign. It preserves completed rows and
selects a checkpoint only after the collector verifies the full held-out
design:

```bash
VALIDATION_SUBMISSION=$(bash examples/prospectus_rfi/submit_memorysafe_validation.sh 20)
printf '%s\n' "$VALIDATION_SUBMISSION"
```

It uses the predeclared N = 100, 150, 200 and five held-out seeds for every
retained checkpoint. `VALIDATION_TASK_JOB` is dependency-free; the short
collector runs `afterok` only after those missing task IDs finish. If an array
task fails or is cancelled, rerun the same submitter; it schedules only the
missing atomic CSV/metadata pairs.

## AMOS 2025 N=100, K=10, 300-second attention control

This control isolates the target-set attention policy in the archived August 13, 2025
physical and PPO regime. It uses N=100, K=10, a 45,000-second episode, fixed
image/charge/downlink/desaturation actions of 300/300/180/150 seconds, alpha=0,
10--40% initial battery, and the archived observation fields and normalization. A
validity bit is appended to each target row because the attention policy requires
masked target-set semantics. The attention architecture did not exist in 2025, so this
is a checkpoint-regime control rather than a bitwise historical reproduction.

The historical PPO cadence is restored: batch 180, 10 epochs, learning rate 1e-6,
clip 0.15, gamma 0.9997, lambda 0.95, entropy 0, and gradient clip 1. One gate job and
one 48-hour run are submitted. The run is split into 22-, 22-, and at-most-5-hour
training caps so it fits Alpine's 24-hour `cpu-normal` limit. Each continuation uses
`afterok`; a failed gate prevents long compute from starting. This submission does not
cancel, hold, or modify the active N=100--200 six-policy campaign.

```bash
module unload slurm/blanca 2>/dev/null || true
module load slurm/alpine
cd /projects/$USER/bsk_rl-rfi
git pull --ff-only

export BSK_RL_REPO_DIR=/projects/$USER/bsk_rl-rfi
export BSK_RL_WANDB_KEY_PATH=/projects/$USER/bsk_rl/examples/wandb_key.txt

CONTROL_SUBMISSION=$(
  bash examples/prospectus_rfi/submit_amos2025_attention_control_48h.sh
)
printf '%s\n' "$CONTROL_SUBMISSION"
```

W&B uses project `amos2025-architecture-comparison` and the dedicated group
`rfi-amos2025-attention-k10-300s-control`. Scratch outputs are placed under a unique UTC
campaign directory below:

```text
/scratch/alpine/$USER/prospectus_rfi/amos2025_attention_control_300s/
```

To audit the current N=100--200 training telemetry against environment steps and wall
time, submit the read-only, low-priority diagnostic from the login node:

```bash
DIAGNOSTIC_JOB=$(sbatch --parsable \
  --export=ALL,BSK_RL_REPO_DIR \
  examples/prospectus_rfi/slurm/diagnose_memorysafe_training.sbatch)
echo "Diagnostic job: $DIAGNOSTIC_JOB"
```

The diagnostic writes CSV and Markdown under
`analysis/training_diagnostic/`. In W&B, use
`prospectus_rfi/environment_steps` or `prospectus_rfi/wall_clock_h` for the x-axis.
Raw return should not be the sole y-axis for variable-N runs; use the successful and
illuminated observation fractions and show episode target count alongside them.

## Historical campaign order

The commands below document the original campaign. They are retained for provenance;
the N=100--400 training launch should not be repeated as the memory-safe v2 campaign.

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

If Alpine's `cpu-long` pool starts only array index 0 and reports
`QOSGrpNodeLimit` for indexes 1--5, the pending policies can instead use restartable
`cpu-normal` segments on the same `acpu`/`epyc-7713` hardware. The migration helper
preserves the running index 0, holds and replaces only pending indexes 1--5, and submits
an independent continuation chain for each policy. The first two allocations train for
up to 23.5 hours each; a short cleanup allocation supplies the remaining cumulative time
needed to reach the common 48-hour target. Checkpoint numbering, environment steps,
training curves, and the stable W&B run identity continue across segments.

Inspect the original array immediately before migration, then run:

```bash
squeue -j "$TRAIN_JOB" -o "%.20i %.2t %.10M %R"
bash examples/prospectus_rfi/submit_pending_candidate_sweep_24h.sh "$TRAIN_JOB"
```

The helper refuses to proceed unless `${TRAIN_JOB}_0` is running and every index from 1
through 5 is still pending. It never holds, cancels, or resubmits index 0. Its timestamped
manifest under `/scratch/alpine/$USER/prospectus_rfi/manifests` records all initial,
continuation, and cleanup job IDs. Do not submit the helper twice for the same array.

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
