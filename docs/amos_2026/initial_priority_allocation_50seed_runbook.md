# AMOS 2026 initial-priority allocation campaign

This campaign evaluates whether the mixed-trained alpha=0.1 policy allocates
more successful images and useful deliveries to progressively more valuable
targets when the priority classes are present before the first decision.

## Matched design

- 200 targets in an exact 100 LEO / 60 MEO / 40 GEO catalog.
- Seeds 0--49 in both cases.
- 45,000 s episode duration and ten policy candidates.
- 20 HIOs (10% of the catalog) assigned five times the realized maximum initial
  priority.
- 20 SHIOs (10%) assigned ten times the realized maximum initial priority.
- The 160 remaining targets are ranked separately within every seed and split
  into lower, middle, and upper initial-priority thirds.
- Case 1 uses the ground-confirmation gate with no additional time cooldown.
- Case 2 adds one observer-orbit cooldown after ground confirmation.
- No Vizard recording. Per-seed plots are created after the simulations from
  the retained CSV files so plotting does not occupy the dynamics jobs.

The two cooldown cases use the same environment seed and the same dynamic
priority-event seed. The event is applied at simulation time zero before the
initial policy observation. This is an initial-priority allocation study, not a
mid-episode responsiveness study.

Array tasks are interleaved by matched seed: even tasks run the
ground-confirmation case and odd tasks run the one-orbit case. Thus tasks 0 and
1 both use seed 0, tasks 2 and 3 both use seed 1, and tasks 98 and 99 both use
seed 49. With the default 20-task concurrency cap, approximately ten matched
seed pairs can run concurrently.

Baseline priorities are drawn uniformly on [0, 2] and rescaled to sum to 200,
as in the existing 200-target evaluation. Rank-based tertiles preserve balanced
comparison groups even though the per-seed rescaling can move the exact numeric
third boundaries slightly.

The evaluated checkpoint was trained on 100 targets with a two-orbit cooldown
and midpoint priority changes. Both cases therefore measure zero-shot
catalog-size, cooldown, and priority-timing generalization; they do not estimate
the performance of policies retrained for either condition.

## Retained output per episode

Each seed retains `steps.csv`, `images.csv`, `target_catalog.csv`,
`verified_deliveries.csv`, `priority_response_targets.csv`, the metrics JSON,
and `mc_status.json`. The status file records the exact command, checkpoint,
case, seed, and validation result.

The dependent analysis job writes:

- `target_allocation_combined.csv`;
- `seed_class_summary.csv`;
- `paired_hio_shio_statistics.csv`;
- `campaign_audit.csv`;
- `STATISTICAL_SUMMARY.md`;
- vector PDF and PNG versions of the capture-allocation, service-metric, and
  paired HIO--SHIO figures.

The raw episode directories remain under the campaign root rather than under
the source worktree. Their layout is
`<campaign>/<cooldown_case>/seed_###/<evaluation_name>/`, and each evaluation
directory contains the retained CSV and NumPy products. The aggregate-analysis
archive therefore does not contain `steps.csv` or the other per-episode files.

## Per-seed plot postprocessing

After the aggregate analysis completes, generate four PDF/PNG diagnostics for
all 100 episodes without rerunning Basilisk or policy inference:

```bash
cd /projects/$USER/bsk_rl-amos2026-campaign
git pull --ff-only origin amos-2026-space-imaging
export AMOS_INITIAL_PRIORITY_OUTPUT_ROOT=/scratch/alpine/$USER/amos2026_mc/<campaign>
bash examples/amos_2026/submit_initial_priority_episode_plot_postprocessing.sh
```

The job writes `per_seed_plots/<cooldown_case>/seed_###/`, an inventory CSV,
and a summary JSON under the campaign root. It creates two downloadable files
beside the campaign directory: `<campaign>_per_seed_plots.tar.gz` contains only
the generated figures, while `<campaign>_episode_csv_and_plots.tar.gz` contains
every per-episode CSV/JSON file together with the figures.

To package all raw episode files as a separate archive, first check their size
and then archive the two case directories:

```bash
campaign_root=/scratch/alpine/$USER/amos2026_mc/<campaign>
du -sh "$campaign_root/ground_confirmation" "$campaign_root/one_orbit"
tar -C "$campaign_root" -czf "${campaign_root}_raw_episodes.tar.gz" \
  ground_confirmation one_orbit
```

## Cluster launch

```bash
module unload slurm/blanca 2>/dev/null || true
module load slurm/alpine
cd /projects/$USER/bsk_rl
git switch amos-2026-space-imaging
git pull --ff-only origin amos-2026-space-imaging
bash examples/amos_2026/submit_initial_priority_allocation_mc_50seeds.sh
```

The launcher uses the same current Alpine resource combination as the recent
successful Research Focus I jobs: partition `acpu`, QoS `cpu-normal`, and the
`epyc-7713` constraint. It runs `sbatch --test-only` for both the evaluation and
analysis jobs before submitting either real job. If CURC changes these names,
override them without editing the scripts:

```bash
AMOS_INITIAL_PRIORITY_PARTITION=<partition> \
AMOS_INITIAL_PRIORITY_QOS=<qos> \
AMOS_INITIAL_PRIORITY_CONSTRAINT=<constraint> \
  bash examples/amos_2026/submit_initial_priority_allocation_mc_50seeds.sh
```

The submission script resolves the repository from its own location, so it can
be launched safely from a linked worktree. It prints the evaluation-array job
ID, dependent analysis job ID, source worktree, and unique output root. The
submission script does not import Basilisk on the restricted login node. Each
evaluation task loads Alpine's default GCC module on its compute node, verifies
that the selected C++ runtime provides `GLIBCXX_3.4.29`, and then refuses to run
if Python resolves `bsk_rl` outside the selected worktree.
