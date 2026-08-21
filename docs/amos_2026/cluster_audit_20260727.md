# AMOS 2026 Cluster Results Audit

Created: 2026-07-27

This checklist is for auditing the AMOS 2026 space-imaging Monte Carlo results on Alpine/CURC from a login shell. It assumes the repository is available at `/projects/$USER/bsk_rl` and the main result roots live under `/scratch/alpine/$USER/amos2026_mc`.

## Expected reward-sweep campaign

Standard policy tags:

```text
00d100i 10d90i 20d80i 30d70i 40d60i 50d50i 60d40i 70d30i 75d25i 80d20i 90d10i 100d00i
```

Expected seeds: `0..99`

Expected standard runs: `12 policies * 100 seeds = 1200 completed mc_status.json files`

Main root conventions seen in the branch:

```text
/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i
/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i_200targets_45000s_<timestamp>
```

Curriculum root convention:

```text
/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i_leo_200targets_45000s_curriculum_alpha1p0_<timestamp>
```

## 1. Environment and repository sanity

Run on Alpine:

```bash
cd /projects/$USER/bsk_rl
git status --short
git branch --show-current
git log --oneline --decorate -n 5
source /projects/$USER/.venv/bin/activate
python --version
```

The intended branch is `amos-2026-space-imaging`.

## 2. Discover candidate MC roots

```bash
MC_BASE=/scratch/alpine/$USER/amos2026_mc

echo "Candidate AMOS 2026 MC roots:"
find "$MC_BASE" -maxdepth 1 -type d \
  \( -name 'gat_full_actions_eval_100d00i*' -o -name '*curriculum*' \) \
  -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort
```

If the fixed root exists, inspect it first:

```bash
export ROOT=/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i
test -d "$ROOT" && echo "Found fixed ROOT=$ROOT" || echo "Fixed root not found"
```

For timestamped 200-target roots, pick the newest candidate:

```bash
find "$MC_BASE" -maxdepth 1 -type d -name 'gat_full_actions_eval_100d00i_200targets_45000s_*' \
  -printf '%T@ %p\n' | sort -n | tail -10
```

Then set `ROOT` explicitly, for example:

```bash
export ROOT=/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i_200targets_45000s_YYYYMMDDTHHMMSSZ
```

## 3. Quick structural check

```bash
echo "ROOT=$ROOT"
test -d "$ROOT" || { echo "ROOT does not exist"; exit 1; }

echo "Manifest files:"
find "$ROOT/manifests" -maxdepth 1 -type f -name '*.json' -print 2>/dev/null | sort

echo "Seed-block folders:"
find "$ROOT" -maxdepth 1 -type d -name 'seeds_*' -print | sort

echo "Analysis folders:"
find "$ROOT" -maxdepth 1 -type d \( -name 'analysis*' -o -name '*plots*' \) -print | sort
```

## 4. Count completed, failed, and missing standard reward-sweep runs

This Python block checks the known branch layout: `seeds_XXX_YYY/<policy>/seed_ZZZ/mc_status.json`.

```bash
python3 - <<'PY'
from pathlib import Path
import json
import os
from collections import Counter, defaultdict

root = Path(os.environ["ROOT"]).expanduser()
policies = "00d100i 10d90i 20d80i 30d70i 40d60i 50d50i 60d40i 70d30i 75d25i 80d20i 90d10i 100d00i".split()
expected_seeds = set(range(100))

status_paths = sorted(root.glob("seeds_*/*/seed_*/mc_status.json"))
print(f"Root: {root}")
print(f"Status files found: {len(status_paths)}")
print(f"Expected standard status files: {len(policies) * len(expected_seeds)}")

by_policy = defaultdict(dict)
states = Counter()
configs = Counter()
for path in status_paths:
    try:
        status = json.loads(path.read_text())
    except Exception as exc:
        print(f"BAD_JSON {path}: {exc}")
        continue
    policy = str(status.get("policy_tag"))
    seed = int(status.get("seed", -1))
    state = status.get("state")
    states[state] += 1
    configs[(status.get("target_env"), status.get("n_targets"), status.get("total_time_sec"), status.get("evaluation_reward_mix"))] += 1
    if policy in policies and seed in expected_seeds:
        by_policy[policy][seed] = (state, path)

print("States:", dict(states))
print("Configs:")
for config, count in configs.most_common():
    print(f"  {config}: {count}")

print("\nPolicy completeness:")
missing_total = 0
failed_total = 0
for policy in policies:
    seen = by_policy.get(policy, {})
    missing = sorted(expected_seeds - set(seen))
    failed = sorted(seed for seed, (state, _path) in seen.items() if state != "completed")
    missing_total += len(missing)
    failed_total += len(failed)
    print(f"{policy:8s} completed={sum(1 for state,_ in seen.values() if state == 'completed'):3d} seen={len(seen):3d} missing={len(missing):3d} failed_or_incomplete={len(failed):3d}")
    if missing:
        print("  missing:", " ".join(f"{seed:03d}" for seed in missing[:40]), "..." if len(missing) > 40 else "")
    if failed:
        print("  failed/incomplete:", " ".join(f"{seed:03d}" for seed in failed[:40]), "..." if len(failed) > 40 else "")

print(f"\nMissing total: {missing_total}")
print(f"Failed/incomplete total: {failed_total}")
PY
```

## 5. Run or refresh standard analyses

Run these after all standard reward-sweep jobs have finished:

```bash
python3 examples/amos_2026/analyze_gat_reward_sweep_mc.py \
  --input-root "$ROOT" \
  --expected-seeds 0:100

python3 examples/amos_2026/analyze_gat_reward_sweep_mc_detailed.py \
  --input-root "$ROOT" \
  --expected-seeds 0:100 \
  --storage-capacity-images 50

python3 examples/amos_2026/plot_amos2026_gat_research_figures.py \
  --input-root "$ROOT" \
  --expected-seeds 0:100
```

Expected standard analysis outputs:

```text
$ROOT/analysis/per_run.csv
$ROOT/analysis/summary_by_policy.csv
$ROOT/analysis/missing_runs.csv
$ROOT/analysis/failed_runs.csv
$ROOT/analysis/analysis_report.json
$ROOT/analysis/ground_value_score_by_training_reward_mix.png
$ROOT/analysis_detailed/detailed_per_run.csv
$ROOT/analysis_detailed/detailed_summary_by_policy.csv
$ROOT/analysis_detailed/action_distribution_by_policy.csv
$ROOT/analysis_detailed/action_id_distribution_by_policy.csv
$ROOT/analysis_detailed/downlink_summary_by_policy.csv
$ROOT/analysis_detailed/downlink_events.csv
$ROOT/analysis_detailed/image_events.csv
$ROOT/analysis_detailed/image_candidate_slot_summary_by_policy.csv
$ROOT/analysis_detailed/missing_runs.csv
$ROOT/analysis_detailed/failed_runs.csv
$ROOT/analysis_detailed/detailed_analysis_report.json
$ROOT/analysis_detailed/metric_definitions.json
$ROOT/analysis_research_plots/
```

## 6. Inspect analysis summaries

```bash
echo "Basic analysis report:"
cat "$ROOT/analysis/analysis_report.json"

echo
echo "Detailed analysis report:"
cat "$ROOT/analysis_detailed/detailed_analysis_report.json"

echo
echo "Top policies by mean ground-value score:"
python3 - <<'PY'
from pathlib import Path
import os
import pandas as pd

root = Path(os.environ["ROOT"])
summary_path = root / "analysis" / "summary_by_policy.csv"
detailed_path = root / "analysis_detailed" / "detailed_summary_by_policy.csv"

summary = pd.read_csv(summary_path)
cols = ["policy_tag", "n_runs", "score_ground_value_100d00i_mean", "score_ground_value_100d00i_std", "score_ground_value_100d00i_ci95"]
print(summary[cols].sort_values("score_ground_value_100d00i_mean", ascending=False).to_string(index=False))

if detailed_path.exists():
    detailed = pd.read_csv(detailed_path)
    keep = [c for c in [
        "policy_tag",
        "n_runs",
        "score_ground_value_100d00i_mean",
        "confirmed_illuminated_images_mean",
        "target_imaging_count_mean",
        "frac_downlink_actions_mean",
        "downlink_success_rate_reward_proxy_mean",
        "mean_successful_imaging_action_duration_sec_mean",
        "mean_target_priority_mean",
    ] if c in detailed.columns]
    print("\nDetailed comparison:")
    print(detailed[keep].sort_values("score_ground_value_100d00i_mean", ascending=False).to_string(index=False))
PY
```

## 7. Check curriculum alpha1p0 campaign

Find curriculum roots:

```bash
find "$MC_BASE" -maxdepth 1 -type d \
  \( -name '*curriculum_alpha1p0*' -o -name '*curriculum*' \) \
  -printf '%T@ %p\n' | sort -n | tail -20
```

Set the curriculum root explicitly:

```bash
export CURR_ROOT=/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i_leo_200targets_45000s_curriculum_alpha1p0_YYYYMMDDTHHMMSSZ
```

Then run:

```bash
echo "CURR_ROOT=$CURR_ROOT"
test -d "$CURR_ROOT" || { echo "CURR_ROOT does not exist"; exit 1; }

find "$CURR_ROOT/manifests" -maxdepth 1 -type f -name '*.json' -print 2>/dev/null | sort
find "$CURR_ROOT" -path '*/mc_status.json' | wc -l

python3 - <<'PY'
from pathlib import Path
import json
import os

root = Path(os.environ["CURR_ROOT"]).expanduser()
statuses = sorted(root.glob("seeds_*/*/seed_*/mc_status.json"))
seen = {}
states = {}
for path in statuses:
    status = json.loads(path.read_text())
    tag = str(status.get("policy_tag"))
    seed = int(status.get("seed", -1))
    seen.setdefault(tag, set()).add(seed)
    states.setdefault(tag, {}).setdefault(status.get("state"), 0)
    states[tag][status.get("state")] += 1
print(f"Status files: {len(statuses)}")
for tag in sorted(seen):
    missing = sorted(set(range(100)) - seen[tag])
    print(f"{tag}: seen={len(seen[tag])} states={states[tag]} missing={len(missing)}")
    if missing:
        print("  missing:", " ".join(f"{seed:03d}" for seed in missing))
PY
```

The standard analyzers have hard-coded standard policy tags, so they are not the safest way to summarize the custom curriculum policy. Package the raw curriculum status/metrics files even if no analyzer summary exists.

## 8. Locate failed jobs and logs

Use Slurm accounting for the MC job names:

```bash
sacct -S 2026-05-01 -u $USER \
  --format=JobID,JobName%35,State,ExitCode,Elapsed,ReqMem,MaxRSS \
  | egrep 'gat_mc|curr|State|----'

echo "Recent job output logs:"
find /scratch/alpine/$USER/job_output -maxdepth 1 -type f \
  \( -name '*gat_mc*' -o -name '*curr*' -o -name '*amos2026*' \) \
  -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort | tail -80
```

If `missing_runs.csv` or the Python completeness report lists missing seeds, inspect their seed folder and logs:

```bash
POLICY=20d80i
SEED=042
find "$ROOT" -maxdepth 6 -path "*/$POLICY/seed_$(printf '%03d' "$SEED")/*" -print
grep -R "policy=$POLICY, seed=$SEED\\|seed $(printf '%03d' "$SEED")\\|Traceback\\|failed\\|OUT_OF_MEMORY" /scratch/alpine/$USER/job_output | tail -80
```

## 9. Package standard and curriculum outputs for copy-back

Use a date-stamped audit folder on scratch:

```bash
AUDIT_DATE=$(date +%Y%m%d)
AUDIT_DIR=/scratch/alpine/$USER/amos2026_mc/audit_export_$AUDIT_DATE
mkdir -p "$AUDIT_DIR"

echo "$ROOT" > "$AUDIT_DIR/standard_root.txt"
test -n "${CURR_ROOT:-}" && echo "$CURR_ROOT" > "$AUDIT_DIR/curriculum_root.txt"

rsync -a \
  "$ROOT/manifests" \
  "$ROOT/analysis" \
  "$ROOT/analysis_detailed" \
  "$ROOT/analysis_research_plots" \
  "$AUDIT_DIR/standard/" 2>/dev/null || true

if [[ -n "${CURR_ROOT:-}" && -d "$CURR_ROOT" ]]; then
  mkdir -p "$AUDIT_DIR/curriculum"
  rsync -a "$CURR_ROOT/manifests" "$AUDIT_DIR/curriculum/" 2>/dev/null || true
  find "$CURR_ROOT" \( -path '*/mc_status.json' -o -name 'metrics_*.json' -o -name 'steps.csv' -o -name 'images.csv' \) -print \
    | tar -czf "$AUDIT_DIR/curriculum_status_metrics.tgz" -T -
fi

tar -czf /scratch/alpine/$USER/amos2026_mc/amos2026_cluster_audit_$AUDIT_DATE.tgz -C "$AUDIT_DIR" .
ls -lh /scratch/alpine/$USER/amos2026_mc/amos2026_cluster_audit_$AUDIT_DATE.tgz
```

If the `analysis_research_plots` folder does not exist yet, run `plot_amos2026_gat_research_figures.py` first or remove that path from the `rsync` command.

## 10. Copy tarball back to Mac

From your Mac terminal, replace `login.rc.colorado.edu` with your actual CURC login host if different:

```bash
AUDIT_DATE=$(date +%Y%m%d)
mkdir -p /Users/dahu1128/Downloads/AMOS2026_cluster_results_$AUDIT_DATE

scp $USER@login.rc.colorado.edu:/scratch/alpine/$USER/amos2026_mc/amos2026_cluster_audit_$AUDIT_DATE.tgz \
  /Users/dahu1128/Downloads/AMOS2026_cluster_results_$AUDIT_DATE/

tar -xzf /Users/dahu1128/Downloads/AMOS2026_cluster_results_$AUDIT_DATE/amos2026_cluster_audit_$AUDIT_DATE.tgz \
  -C /Users/dahu1128/Downloads/AMOS2026_cluster_results_$AUDIT_DATE/
```

If you prefer `rsync`:

```bash
AUDIT_DATE=$(date +%Y%m%d)
mkdir -p /Users/dahu1128/Downloads/AMOS2026_cluster_results_$AUDIT_DATE

rsync -avP \
  $USER@login.rc.colorado.edu:/scratch/alpine/$USER/amos2026_mc/amos2026_cluster_audit_$AUDIT_DATE.tgz \
  /Users/dahu1128/Downloads/AMOS2026_cluster_results_$AUDIT_DATE/

tar -xzf /Users/dahu1128/Downloads/AMOS2026_cluster_results_$AUDIT_DATE/amos2026_cluster_audit_$AUDIT_DATE.tgz \
  -C /Users/dahu1128/Downloads/AMOS2026_cluster_results_$AUDIT_DATE/
```

## 11. Local analysis plan after copy-back

Once the tarball is copied back, use the copied CSV/JSON files to:

1. Verify data completeness: expected policy tags, expected seed count, no failed/incomplete runs, consistent `target_env`, `n_targets`, `total_time_sec`, `evaluation_reward_mix`, and dynamic-priority settings.
2. Rank the standard alpha policies by `score_ground_value_100d00i_mean`, with standard deviation and 95 percent confidence interval.
3. Compare the top policy to neighboring alpha values to avoid overclaiming a noisy one-seed or one-alpha advantage.
4. Compare curriculum `curriculum_alpha1p0` against the best fixed-alpha policies using the same scoring and seed set.
5. Generate paper tables: reward sweep summary, top-three policy comparison, action mix/downlink usefulness, imaging-duration savings, and HIO/SHIO response if instrumented fields exist.
6. Generate paper figures from `analysis_research_plots`, or regenerate local figures from `detailed_per_run.csv` and `detailed_summary_by_policy.csv`.
7. Mark any metrics as proxies where the repo already says they are proxies, especially image-to-downlink latency without packet IDs.
8. Update `papers/amos_2026_space_imaging/main.tex` with final values and replace local single-seed placeholders.
