# AMOS 2026 Experiments

This folder is the intended home for the cleaner AMOS 2026 experiment entry points.

The current working baseline still lives in the historical evaluation scripts:

- `examples/policy_evaluation_2026.py`
- `examples/updated_policy_evaluation.py`
- `examples/sim_config.py`

As the branch stabilizes, move paper-specific experiment code here in small modules:

- `config.py`: AMOS-specific simulation and campaign settings.
- `eval_reimaging.py`: single-run evaluation entry point.
- `metrics.py`: action durations, reimages, useful/failed downlinks, cooldown waits.
- `policies.py`: policy loading, heuristic policies, and shielding.
- `run_campaign.py`: Monte Carlo/scaling campaign launcher.

Design goal: keep reusable experiment logic here and leave notebooks or one-off plotting
scripts outside the core evaluation path.

## Run A Current Policy

From the repository root, activate the Python environment that has Basilisk, Gymnasium,
Ray/RLlib, and this package installed, then run:

```bash
python3 examples/policy_evaluation_2026.py \
  --seed 20 \
  --target_env mixed \
  --policy_name oct14_obsv7_1e_5lr_batch5000_gamma9997_20d80i \
  --policy_mode latest \
  --save_data
```

The `20d80i` tag is the current alpha-0.2 convention in the eval script:
`alpha_from_tag("20d80i") == 0.2`, meaning 20 percent downlink weighting and
80 percent imaging weighting.

Useful quick variants:

```bash
# LEO-only target catalog instead of mixed LEO/MEO/GEO targets
python3 examples/policy_evaluation_2026.py --seed 20 --target_env leo \
  --policy_name oct14_obsv7_1e_5lr_batch5000_gamma9997_20d80i --save_data

# Faster smoke run output: do not save the per-step numpy arrays/csv files
python3 examples/policy_evaluation_2026.py --seed 20 --target_env mixed \
  --policy_name oct14_obsv7_1e_5lr_batch5000_gamma9997_20d80i --no_save_data --quiet
```

Outputs are written under `examples/data/<policy>_seed<seed>_<timestamp>/`.
The historical plotting code also writes PDFs to `plots/`.

## Profile A Run

For a profile that includes the same simulation path:

```bash
python3 examples/amos_2026/profile_eval.py --stats 50 -- \
  --seed 20 \
  --target_env mixed \
  --policy_name oct14_obsv7_1e_5lr_batch5000_gamma9997_20d80i \
  --no_save_data \
  --quiet
```

The helper writes `examples/amos_2026/policy_evaluation_2026.prof` and prints the
top functions by cumulative time. To re-open any `.prof` file later:

```bash
python3 - <<'PY'
import pstats
pstats.Stats("examples/amos_2026/policy_evaluation_2026.prof") \
    .strip_dirs().sort_stats("cumtime").print_stats(50)
PY
```

Current profiling expectation: most wall time is usually inside Basilisk
`SimModel_StepUntilStop`, so true physics cadence changes are the highest-leverage but
also change fidelity. Low-risk speedups should first remove unnecessary debug printing,
reduce Python-side recorder/log reads, and keep expensive plotting/saving out of timing
runs.

## Target Priorities

Priority generation is configured in `examples/sim_config.py` and used by
`scene.RandomSatellites`.

Current defaults:

- `priority_mode="uniform"`
- `priority_sum=100.0`
- `rescale_priorities_to_sum=True`
- `priority_uniform_low=0.0`
- `priority_uniform_high=None`

With `priority_uniform_high=None`, the high end becomes `2 * priority_sum / n_targets`.
For 100 targets, priorities are sampled from `[0, 2]`, then rescaled so the sum is
exactly 100 before the first simulation step. Gaussian and constant modes are also
available through the same config fields.

## CURC Training Startup

Use `examples/train_Polaris_target_gnn_wandb.py` for the current AMOS 2026
target-wise GNN, W&B-tracked, imaging-only training run. It uses observation layout
`obs-v9`: target chunks only, with priority, relative target position/velocity in
Hill frame, target angle/distance, and illumination. Only imaging actions are
exposed to the policy.

Use `examples/updated_train_Polaris.py` for the BigNetwork fully-connected baseline
with the full action set (`ImageRSO`, `Charge`, `Downlink`, `Desat`). That path remains
available for comparison runs, now also using the organized `obs-v9` stack:
spacecraft/resource state with sun vector, eclipse timing, ground-station windows,
then target chunks. Its debug and 96-hour Slurm wrappers log to W&B by default.

Use `examples/updated_train_Polaris_ImagingOnly.py` for the BigNetwork image-only
baseline. It also uses target-only `obs-v9` observations, but keeps the scanner at
1000x baseline battery and 500-image storage so resource depletion does not drive
the learning signal.

The real GNN implementation lives in
`src/bsk_rl/utils/rllib/target_gnn_module.py`. The file
`examples/target_gnn_module.py` is only a compatibility wrapper for older example
imports.

The full-action BigNetwork resource-restricted entrypoint uses the baseline scanner
battery by default. The image-only BigNetwork and Target-GNN entrypoints use 1000x
baseline battery and 500-image storage by default. Target satellites are kept
passive/alive; they are not the learned spacecraft, and killing them at `t=0`
mostly creates log noise for these runs.

The AMOS branch makes the cluster-specific pieces configurable through environment
variables instead of hardcoding them in the Python script.

Useful environment variables:

- `BSK_RL_SCRATCH`: defaults to `/scratch/alpine/$USER` on Slurm.
- `BSK_RL_OUTPUT_DIR`: defaults to `$BSK_RL_SCRATCH/rllib_results` on Slurm.
- `BSK_RL_RAY_TMPDIR`: defaults to `$BSK_RL_SCRATCH/tmp` unless the sbatch script sets
  a job-specific directory.
- `BSK_RL_BATCH_MULTIPLIER`: defaults to `150`, matching the recent cluster script.
- `BSK_RL_TOTAL_TIMESTEPS`: defaults to `20000000`; the debug sbatch overrides this to
  `500000`.
- `BSK_RL_TORCH_THREADS`: defaults to `11`, matching the recent cluster script.
- `BSK_RL_BATTERY_LIFE_MULTIPLIER`: defaults to `1000`, giving the scanner 1000x the
  baseline battery capacity for image-only runs. The full-action resource-restricted
  Slurm wrappers set this to `1`.
- `BSK_RL_IMAGE_STORAGE_CAPACITY_IMAGES`: defaults to `500` for image-only runs.
- `BSK_RL_DYNAMIC_PRIORITY_EVENT`: defaults to `1` in the current Slurm wrappers,
  enabling the half-episode HIO/SHIO priority schedule.
- `BSK_RL_DYNAMIC_PRIORITY_EVENT_FRACTION`: defaults to `0.5`, so the boost applies
  after half of `sim_cfg.total_time`.
- `BSK_RL_DYNAMIC_PRIORITY_EVENT_TIME_SEC`: optional absolute boost time in seconds.
  If set, this overrides the fraction-based timing.
- `BSK_RL_HIO_COUNT`, `BSK_RL_HIO_PRIORITY`: default to `5` targets at priority `5`.
- `BSK_RL_SHIO_COUNT`, `BSK_RL_SHIO_PRIORITY`: default to `3` targets at priority `10`.
- `BSK_RL_DYNAMIC_PRIORITY_EVENT_SEED`: optional fixed seed for reproducible HIO/SHIO
  target selection.
- `BSK_RL_WANDB_KEY_PATH`: defaults to `/projects/$USER/bsk_rl/examples/wandb_key.txt`
  on the Slurm wrappers.

Before the first Slurm submission on CURC:

```bash
cd /projects/$USER/bsk_rl
mkdir -p /scratch/alpine/$USER/job_output \
         /scratch/alpine/$USER/rllib_results \
         /scratch/alpine/$USER/tmp
```

For a one-hour startup check:

```bash
cd /projects/$USER/bsk_rl
sbatch examples/amos_2026/sbatch_train_polaris_target_gnn_wandb_debug.sh
```

Watch the job:

```bash
squeue -u $USER
tail -f /scratch/alpine/$USER/job_output/amos2026_leo_dbg_<jobid>_0.out
```

If the debug job starts cleanly, increase the sbatch time and set
`BSK_RL_TOTAL_TIMESTEPS=20000000` for the real run. The ready-made current runs are:

```bash
# Current target-wise GNN obs-v9 + W&B, one-hour debug
sbatch examples/amos_2026/sbatch_train_polaris_target_gnn_wandb_debug.sh

# Current target-wise GNN obs-v9 + W&B, 24-hour training
sbatch examples/amos_2026/sbatch_train_polaris_target_gnn_wandb_24h.sh

# BigNetwork full-action baseline obs-v9, one-hour debug
sbatch examples/amos_2026/sbatch_updated_train_polaris_debug.sh

# BigNetwork full-action baseline obs-v9, 96-hour training
sbatch examples/amos_2026/sbatch_updated_train_polaris_96h.sh

# BigNetwork imaging-only baseline obs-v9, 96-hour training
sbatch examples/amos_2026/sbatch_updated_train_polaris_imaging_only_96h.sh
```

## GAT Reward-Sweep Monte Carlo Evaluation

Use the AMOS 2026 Monte Carlo workflow to compare the full-action GAT policies
trained with `00d100i`, `10d90i`, `20d80i`, `30d70i`, `40d60i`, `50d50i`,
`60d40i`, `70d30i`, `75d25i`, `80d20i`, `90d10i`, and `100d00i` reward mixes. Every policy is scored with the same
`100d00i` evaluation reward, representing the value of images delivered to the
ground. Checkpoint discovery intentionally excludes earlier `_alpha...`
24-hour pilot folders and selects only the later non-alpha 48-hour sweep runs.

Each Slurm array task owns one policy and runs a ten-seed block. Every seed is
still launched as a fresh Python evaluator subprocess. This keeps the queue
compact while avoiding the memory growth and CSPICE teardown issues seen when
many Basilisk episodes were evaluated sequentially inside one Python process.
Plots, tabular data, and summary metrics are saved below each seed's uniquely
named evaluation run folder.

To run an immediate two-hour smoke test without waiting for active training,
submit one array job covering seeds `0..9` inclusive for every policy:

```bash
cd /projects/$USER/bsk_rl
git pull --ff-only origin amos-2026-space-imaging
bash examples/amos_2026/submit_gat_reward_sweep_mc_smoke_2h.sh
```

This snapshots the latest complete checkpoint that exists at submission time
for each non-alpha policy. It submits `12` policy-level array tasks, each of
which evaluates `10` seeds using fresh subprocesses (`120` evaluations total).
At most `4` policy tasks run simultaneously, and each policy task has a
two-hour limit, `4` allocated CPUs, and `24G` of memory shared by its sequential
seed subprocesses. The helper prints its timestamped output folder and exact
analysis command.

For the later full `0..99` campaign, submit its first ten-seed block:

```bash
cd /projects/$USER/bsk_rl
git pull --ff-only origin amos-2026-space-imaging
bash examples/amos_2026/submit_gat_reward_sweep_mc_block.sh 0
```

The helper freezes one exact checkpoint per policy before the first submission,
then reuses that campaign manifest for later seed blocks. It submits `12`
policy-level array tasks, each running `10` fresh seed subprocesses (`80`
evaluations total), with at most `4` policy tasks active at once. Freezing
prevents different seeds from silently loading different checkpoints while a
training job is still advancing. Submit only after the desired training runs
have finished, and inspect the printed manifest if checkpoint choice matters.

After the first full-campaign block is healthy, submit its remaining blocks:

```bash
for start in 10 20 30 40 50 60 70 80 90; do
    bash examples/amos_2026/submit_gat_reward_sweep_mc_block.sh "$start"
done
```

If you are continuing from a successful smoke folder, reuse that exact output
root and frozen manifest. The remaining-block helper chains blocks `10..99` so
only one ten-seed block is active at a time, and completed matching policy/seed
runs already present under the root are skipped. This is useful when the smoke
folder already contains seed `10`:

```bash
ROOT=/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i_smoke_2h_20260601T184244Z
MANIFEST=$ROOT/manifests/gat_full_actions_obs_v9_eval100d00i_nonalpha48h_frozen.json

BSK_RL_MC_OUTPUT_ROOT="$ROOT" \
BSK_RL_MC_MANIFEST="$MANIFEST" \
    bash examples/amos_2026/submit_gat_reward_sweep_mc_remaining_blocks.sh 2
```

To lower or raise the concurrency cap, pass a second argument. For example,
this runs the first block with at most two simultaneous episodes:

```bash
bash examples/amos_2026/submit_gat_reward_sweep_mc_block.sh 0 2
```

To intentionally begin a new campaign with freshly discovered checkpoints,
refresh the manifest on the first block:

```bash
BSK_RL_MC_REFRESH_MANIFEST=1 \
    bash examples/amos_2026/submit_gat_reward_sweep_mc_block.sh 0
```

Results are organized below:

```text
/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i/
  manifests/
  seeds_000_009/
    00d100i/seed_000/
      <unique-evaluation-run>/plots/
    ...
    100d00i/seed_009/
  analysis/
```

Aggregate the first full-campaign block with:

```bash
python3 examples/amos_2026/analyze_gat_reward_sweep_mc.py --expected-seeds 0:10
```

Aggregate the eventual full `0..99` campaign with:

```bash
python3 examples/amos_2026/analyze_gat_reward_sweep_mc.py --expected-seeds 0:100
```

The analysis folder contains `per_run.csv`, `summary_by_policy.csv`,
`missing_runs.csv`, `failed_runs.csv`, `analysis_report.json`, and a
ground-value comparison plot.

For a richer AMOS 2026 policy comparison, run the detailed analyzer:

```bash
ROOT=/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i_smoke_2h_20260601T184244Z

python examples/amos_2026/analyze_gat_reward_sweep_mc_detailed.py \
    --input-root "$ROOT" \
    --expected-seeds 0:100 \
    --storage-capacity-images 50
```

This writes `analysis_detailed/` with per-run and policy-level metrics,
including target priority/illumination summaries, action distributions,
downlink-usefulness proxies, image-to-next-downlink latency proxies, and plots.
The latency metrics are labeled as proxies because the current evaluation files
do not store packet IDs tying each captured image to its exact downlink event.
