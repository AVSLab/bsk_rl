# AMOS 2026 Experiments

This folder is the intended home for the cleaner AMOS 2026 experiment entry points.

## Current workstream layout

- `examples/amos_2026/`: AMOS-only training, evaluation, analysis, campaign, and
  visualization entry points.
- `examples/policy_evaluation_2026.py`: the current paper-matched single-episode
  evaluator.  AMOS entry points should call it rather than copy its environment setup.
- `src/bsk_rl/`: reusable simulator, action, observation, data, and visualization code.
- `docs/amos_2026/`: CURC campaign runbooks, audits, synchronization helpers, and
  reproducibility notes.
- `artifacts/amos_2026/`: local policy checkpoints, Vizard recordings, and other large
  generated files.  This folder is intentionally ignored by Git.
- The paper source and final PDF remain outside the repository in the canonical OneDrive
  `Conferences/AMOS2026/AMOS_conference_paper_2026/` folder.

The AMOS work is on branch `amos-2026-space-imaging`.  The submitted imaging-versus-
downlink journal work has its own clean worktree and branch,
`bsk_rl_breckenridge_public_clean` / `breckenridge2026-publication-clean`; it should not
be folded into this AMOS checkout.

## Run a smooth Vizard episode

The paper's selected mixed-trained alpha-0.1 checkpoint belongs at:

```text
artifacts/amos_2026/policies/mixed_a0p1/checkpoint_000119
```

It can instead be supplied through `--policy-path` or the `AMOS2026_POLICY_PATH`
environment variable.  From the repository root:

```bash
.venv/bin/python examples/amos_2026/run_vizard_episode.py --seed 0
```

The launcher records a 200-target, 45,000-second episode at 1 Hz by default. The
priority-event threshold is half the episode (22,500 s); at the first policy decision
boundary at or after that threshold, a reproducible, disjoint 10% of the catalog (20
targets) becomes HIO and another 10% (20 targets) becomes SHIO. Applying the event at a
decision boundary avoids pretending that a priority change can interrupt an action that
is already executing. Their priorities become five and ten times the episode's maximum
initial priority, respectively. The default additional orbital
cooldown is zero: a captured target remains unavailable while its image is onboard, then
becomes eligible as soon as useful ground verification removes that pending record. Use
`--reimage-cooldown-orbits 2` only to reproduce the older two-orbit behavior. Use
`--vizard-rate-hz 2` only when a denser playback is specifically needed. Target fill
uses light, medium, and dark blue for the lower, middle, and upper initial-priority
thirds. All targets begin as blue circles. Because Vizard treats sprite shape as an
initialization-only property, dedicated promotion sprites are initialized at Earth's
center and remain hidden before the midpoint event. They then replace the selected
targets as medium-purple HIO stars and deep-purple SHIO triangles; matching promotion
halos turn on at the same time. The launcher also
saves the action-allocation, resource-history,
and pointing-history PDFs under `examples/amos_2026/plots/`. The Space Surveillance
Inspector HUD also shows the current action,
catalog and image-count metrics, resource state, illuminated/non-illuminated storage
split sampled when each physical target partition grows, active
ground link, reaction-wheel state, thruster plumes, and the
yellow-to-green imaging pointing/hold line.  Ground-station cones use each station's
minimum elevation and the scanner's initial orbital radius. Automatic location links
from every target spacecraft are explicitly disabled. The custom green line and slow,
opaque-purple outward transmission rings appear only while the transmitter reports data
leaving onboard storage; an access window or downlink command by itself is insufficient.
During a desaturation action, a separate transceiver emits opaque red rings instead of
the former spacecraft halo; the purple communication rings remain off unless data is
actually leaving storage. Each target retains its light-to-dark blue priority-tier
fill. The cyan eligible, red cooldown, and green onboard lifecycle outlines are omitted
by default because they are not reliably visible in Vizard and add 240 ellipsoid objects
to the 200-target/40-promotion scene. Use `--target-status-outlines` to opt into them for
diagnostics. The native expandable storage panel is visible. Vizard automatically
appends `Storage` to that panel title and provides no title override. The separate
operations dialog remains `SPACE SURVEILLANCE`, and the spacecraft display name is
`SS1 Space Surveillance Inspector`.

The inspector uses a 1.5x body-fixed `bskSat` CAD model and a 2.5x planet-view
spacecraft scale, so its attitude motion remains easier to see and its distant sprite
transition occurs farther out. Vizard's sprite-enable setting is global rather than
per-spacecraft; disabling it would replace all 200 RSO circles/stars/triangles with CAD
spacecraft. The AMOS setup therefore preserves the target marker semantics instead of
using that global switch.

The AMOS launcher omits the reaction-wheel panel by default. Use `--rw-display all` for
Vizard's three-wheel actuator panel or `--rw-display off` to keep it hidden. Use
`--no-text-hud` to hide the text dialog, `--no-image-bars` to hide only the three image
count bars, or `--no-metric-bars` to omit every metric bar and skip the associated
200-target catalog calculations. Use
`--target-status-outlines` to restore lifecycle outlines for an A/B diagnostic, or
`--no-hud` to disable all overlays. Use `--dry-run` to inspect the exact paper-matched
evaluator command without executing it. Use `--overwrite` to replace the canonical
playback for the same seed, catalog size, and sampling rate after a successful rerun.
Per-action simulation-time progress is printed by default; `--quiet` is an explicit
opt-in when terminal progress is not wanted.

The default run opens both the `SPACE SURVEILLANCE` operations dialog and the `SS1
Space Surveillance Inspector Storage` metric panel. Do not pass `--no-text-hud` or
`--no-metric-bars` when those panels are wanted. If the storage panel is closed during
playback, reopen it from Vizard's `Devices` menu for the SS1 spacecraft. The operations
dialog is refreshed every recorded frame; advancing the timeline reopens a minimized
copy, while reloading the recording restores it after it has been destroyed.

Use `--n-targets 100` or `--n-targets 200` to select the exact 50/30/20 mixed-regime
catalog size; 200 is the default. `--interest-fraction` changes each of the two disjoint
promotion groups and must produce an integer target count. Both catalog sizes use the
paper's 45,000 s horizon, the selected mixed-trained `alpha=0.1` checkpoint, and save
into the single `artifacts/amos_2026/vizard/` folder. Scenario tags include catalog
size, interest fractions, cooldown mode, seed, and sampling rate so a new
ground-confirmation playback cannot overwrite an older two-orbit file accidentally.

### Vizard playback performance

The existing 200-target/1 Hz playback is about 2.5 GB, but RAM capacity is usually not
the limiting resource. Increasing Vizard's playback multiplier makes its main render
thread deserialize and update hundreds of spacecraft, promotion proxies, lifecycle
halos, orbit paths, cones, and HUD elements in less wall-clock time. CPU/GPU frame time
and Vizard's per-frame scene bookkeeping therefore saturate before system memory.
The AMOS recording starts with osculating orbit, true-trajectory, and ground-track
histories hidden because rendering histories for 241 visualized spacecraft is the
largest avoidable playback cost. They can still be enabled from Vizard's `View` menu.

The default launcher now avoids constructing the 240 lifecycle-outline ellipsoids. It
retains only the 40 HIO/SHIO promotion halos and the two action-ring transceivers. This
reduces scene load without changing target priority fills or simulation behavior.

For a faster review copy, record at `--vizard-rate-hz 0.5` (or `0.25` for rapid
screening), leave `--rw-display off`, and use `--no-text-hud --no-metric-bars`. The AMOS
monitor runs at this recording cadence rather than at every dynamics tick. In the
Vizard UI, hide osculating/true trajectory lines, spacecraft labels, and location cones
when they are not being inspected. Keep the 1 Hz full-HUD file as the presentation or
audit artifact rather than expecting the same file to remain smooth at every playback
multiplier.

`--vizard-rate-hz` is recording density, not playback speed. A 2.5 Hz, 45,000-second
episode contains 112,500 frames—five times the frames in a 0.5 Hz recording. The larger
file increases disk decoding and gives Vizard more scene updates to process when the `+`
playback multiplier is used, even though Vizard buffers the file instead of loading all
of it into RAM.

## Safe cleanup order

1. Preserve the current dirty AMOS checkout by committing or making a named patch before
   deleting or moving any experiment files.
2. Copy the frozen paper checkpoint and its manifest into `artifacts/amos_2026/`, then
   verify the checkpoint with one Vizard episode.
3. Keep source code in `examples/amos_2026/` or `src/bsk_rl/`, large outputs in
   `artifacts/amos_2026/`, campaign notes in `docs/amos_2026/`, and the manuscript in its
   OneDrive paper folder.
4. Only after the checkpoint, manifests, and final outputs are verified should redundant
   downloads, old result bundles, or the detached legacy journal worktree be archived or
   removed.

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

## Continue GAT Reward-Sweep Training

To add another 24 hours to an already-trained AMOS full-action GAT policy, use
one of the continuation wrappers and pass the original run/checkpoint through
`BSK_RL_CONTINUE_FROM`. The Python trainer copies that run directory into a new
output folder, restores from the checkpoint inside the copy, and writes all new
checkpoints to the copy.

```bash
cd /projects/$USER/bsk_rl

# Two-hour smoke test, alpha 0.1 / 10d90i
sbatch --export=ALL,BSK_RL_CONTINUE_FROM=/scratch/alpine/$USER/rllib_results/<old_output>/<old_run> \
  examples/amos_2026/sbatch_continue_polaris_gat_full_actions_10d90i_2h.sh

# Two-hour smoke test, alpha 0.2 / 20d80i
sbatch --export=ALL,BSK_RL_CONTINUE_FROM=/scratch/alpine/$USER/rllib_results/<old_output>/<old_run> \
  examples/amos_2026/sbatch_continue_polaris_gat_full_actions_20d80i_2h.sh

# Alpha 0.1 / 10d90i
sbatch --export=ALL,BSK_RL_CONTINUE_FROM=/scratch/alpine/$USER/rllib_results/<old_output>/<old_run> \
  examples/amos_2026/sbatch_continue_polaris_gat_full_actions_10d90i_24h.sh

# Alpha 0.2 / 20d80i
sbatch --export=ALL,BSK_RL_CONTINUE_FROM=/scratch/alpine/$USER/rllib_results/<old_output>/<old_run> \
  examples/amos_2026/sbatch_continue_polaris_gat_full_actions_20d80i_24h.sh

# Final 0.0 -> 1.0 curriculum policy, held at alpha 1.0 for another 24 hours
sbatch --export=ALL,BSK_RL_CONTINUE_FROM=/scratch/alpine/$USER/rllib_results/<curriculum_output>/<curriculum_run> \
  examples/amos_2026/sbatch_continue_polaris_gat_full_actions_curriculum_final_alpha1p0_24h.sh

# Randomized LEO/MEO/GEO mix with 100 targets
sbatch --export=ALL,BSK_RL_CONTINUE_FROM=/scratch/alpine/$USER/rllib_results/<mixed_output>/<mixed_run> \
  examples/amos_2026/sbatch_continue_polaris_gat_full_actions_10d90i_mixed_random_24h.sh

# Recovery test for randomized LEO/MEO/GEO mix with 100-300 targets (200G RAM)
sbatch --export=ALL,BSK_RL_CONTINUE_FROM=/scratch/alpine/$USER/rllib_results/<mixed_n300_output>/<mixed_n300_run> \
  examples/amos_2026/sbatch_continue_polaris_gat_full_actions_10d90i_mixed_random_100to300targets_2h.sh

# Full 24-hour continuation after the recovery test succeeds (200G RAM)
sbatch --export=ALL,BSK_RL_CONTINUE_FROM=/scratch/alpine/$USER/rllib_results/<mixed_n300_output>/<mixed_n300_run> \
  examples/amos_2026/sbatch_continue_polaris_gat_full_actions_10d90i_mixed_random_100to300targets_24h.sh
```

`BSK_RL_CONTINUE_FROM` may point directly at a run directory, a specific
`checkpoint_000123` directory, or a timestamped output directory containing
exactly one run. The two-hour wrappers set `BSK_RL_TRAIN_TIMEOUT_SEC=5400`;
the 24-hour wrappers set `BSK_RL_TRAIN_TIMEOUT_SEC=84600`. Both leave room
before Slurm wall time so the trainer can save a final checkpoint cleanly. Keep
the same branch, virtualenv, target-regime env vars, and reward-split wrapper as
the original run; RLlib restore expects the same observation/action spaces and
module config. The curriculum continuation wrapper is the intentional exception
to keeping the reward wrapper: it restores the curriculum policy and optimizer
state into the same GAT/PPO setup, disables the ramp, and holds the environment
reward at its final `alpha=1.0` task for the full continuation.

The randomized 100-300-target training job `28226894` ended after 1 day,
9 hours with Slurm state `OUT_OF_MEMORY`, using all `100 GiB` requested. Its
recovery and continuation wrappers request `200G` while preserving 32 CPUs,
28 Ray environments, and the 4,200-step training batch. The original 48-hour
training wrapper now also requests `200G`. Check a completed job and inspect
the available source checkpoints with:

```bash
seff <job_id>
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,ReqMem,MaxRSS
find <run_directory> -maxdepth 1 -type d -name 'checkpoint_[0-9]*' | sort -V
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

To evaluate the final `0.0 -> 1.0` curriculum policy on seeds `0..99` using
the same 200-target, 45,000-second MC setup, submit the dedicated custom-policy
campaign. It freezes the latest checkpoint from the curriculum run into a
manifest and records plotting metadata with `alpha=1.0` and color `#5BC5DB`:

```bash
cd /projects/$USER/bsk_rl
git pull --ff-only origin amos-2026-space-imaging
bash examples/amos_2026/submit_gat_curriculum_alpha1p0_mc_200targets_45000s_0to99.sh
```

If you want those curriculum seed folders to live directly under an existing
MC output root for later combined plotting, set `BSK_RL_MC_OUTPUT_ROOT` before
submitting. To evaluate a different curriculum-derived copy later, set
`BSK_RL_MC_CURRICULUM_FROM` to that run or checkpoint and optionally set a new
`BSK_RL_MC_CURRICULUM_TAG`.

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

## Seed-level timing diagnostics

Current policy evaluations save exact reset-time ground-station access intervals
to `ground_station_windows.csv` and, when `--save_data` is enabled, annotate each
downlink action in `downlink_ground_station_window_alignment.csv`. The resource
plot uses stepwise cumulative counts and distinguishes the downlink command start
from the full action interval. A separate target-availability diagnostic plots
eligible and imageable target counts together with Desat decisions and wheel state.

Use `--reimage_cooldown_orbits 1` for the one-orbit cooldown ablation. Its
availability plot is automatically titled `ONE-ORBIT COOLDOWN ABLATION`; the
standard AMOS configuration remains `--reimage_cooldown_orbits 2`.

An older saved run can be replotted without rerunning the episode:

```bash
python examples/amos_2026/plot_evaluation_timing_diagnostics.py RUN_DIR \
    --ground-station-windows RUN_DIR/ground_station_windows.csv \
    --plot-dir examples/amos_2026/plots \
    --name seed0_corrected_timing \
    --cooldown-orbits 2
```
