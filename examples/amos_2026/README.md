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
`obsB8`: spacecraft/global observations first, then target chunks, with only imaging
actions exposed to the policy.

Use `examples/updated_train_Polaris.py` for the old fully-connected-network baseline
with the full action set (`ImageRSO`, `Charge`, `Downlink`, `Desat`). That path remains
`obsv7` and is intended for comparison runs, not for target-wise GNN training. Its
debug and 96-hour Slurm wrappers also log to W&B by default.

The real GNN implementation lives in
`src/bsk_rl/utils/rllib/target_gnn_module.py`. The file
`examples/target_gnn_module.py` is only a compatibility wrapper for older example
imports.

The scanner gets 1000x baseline battery capacity by default in these training
entrypoints. Target satellites are kept passive/alive; they are not the learned
spacecraft, and killing them at `t=0` mostly creates log noise for these runs.

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
  baseline battery capacity for these training runs.
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
# Current target-wise GNN obsB8 + W&B, one-hour debug
sbatch examples/amos_2026/sbatch_train_polaris_target_gnn_wandb_debug.sh

# Current target-wise GNN obsB8 + W&B, 24-hour training
sbatch examples/amos_2026/sbatch_train_polaris_target_gnn_wandb_24h.sh

# Old FC-network full-action baseline, one-hour debug
sbatch examples/amos_2026/sbatch_updated_train_polaris_debug.sh

# Old FC-network full-action baseline, 96-hour training
sbatch examples/amos_2026/sbatch_updated_train_polaris_96h.sh
```
