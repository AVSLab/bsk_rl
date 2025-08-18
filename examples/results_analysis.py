#!/usr/bin/env python3
# examples/compile_results.py
#
# Aggregate all *_results_summary.json files under a chosen folder (preset toggle
# or --root path), compute overall metrics, print a report, and save plots.
#
# Usage examples (run from examples/):
#   python compile_results.py --preset heuristic
#   python compile_results.py --preset rl_restricted
#   python compile_results.py --root "./some_other_folder"
#   python compile_results.py --preset imaging_rl
#
# You can also combine: --preset rl_restricted --root "."  (root overrides preset)

import argparse
import csv
import json
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# --------------------------- PRESETS / TOGGLES ---------------------------
EXAMPLES_DIR = Path(__file__).resolve().parent

# Define simple, human names → default folders (edit as you create folders)
PRESETS = {
    # your existing heuristic MC runs
    "heuristic": EXAMPLES_DIR / "heuristic_policy_data",

    # your RL runs with restricted resources/downlinks shown in your screenshot;
    # if those JSONs live directly under examples/, point this to EXAMPLES_DIR
    "rl_restricted": EXAMPLES_DIR/ "RL-policy_data",  # change to a subfolder if you move them later

    # future imaging-only folders (no charge/downlink/desat required)
    "imaging_heuristic": EXAMPLES_DIR / "heuristic_imaging",
    "imaging_rl":        EXAMPLES_DIR / "RL-policy_imaging",
    "old_imaging_rl":        EXAMPLES_DIR / "old_RL-policy_imaging",
    "old_imaging_heuristic": EXAMPLES_DIR / "old_heuristic_imaging",
    "aug3_imaging_rl":        EXAMPLES_DIR / "aug3_RL-policy_imaging",
    "no_outlier_imaging_rl":        EXAMPLES_DIR / "no_outlier_RL-policy_imaging",
}

# --------------------------- Helpers ---------------------------
def _nums(vals):
    """Keep numeric values only (skip None/NaN)."""
    out = []
    for v in vals:
        if v is None:
            continue
        if isinstance(v, (int, float)) and not math.isnan(v):
            out.append(float(v))
    return out

def _sum_dicts_of_ints(dicts):
    """Sum a sequence of {str/int->int} dicts into one dict of ints with int keys."""
    total = defaultdict(int)
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            try:
                ik = int(k)
                total[ik] += int(v)
            except Exception:
                pass
    return dict(total)

def _safe_get(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

def mean_std_min_max(x):
    if not x:
        return (0.0, 0.0, 0.0, 0.0)
    arr = np.asarray(x, dtype=float)
    return (float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr)))

# --------------------------- Core ---------------------------
def gather_results(root: Path):
    """Find and load every *_results_summary.json under root (recursively)."""
    files = sorted(root.rglob("*results_summary.json"))
    if not files:
        raise FileNotFoundError(f"No results_summary.json files found under: {root}")
    print(f"Scanning {len(files)} file(s) under: {root}")

    per_run_records = []         # list[dict] for every run across all files
    per_file_action_totals = []  # list[dict] of action_counts_total from each file's aggregate

    for f in files:
        with open(f, "r") as fh:
            js = json.load(fh)

        # Try per_run first
        per_run = _safe_get(js, "per_run", None)
        if per_run is None:
            per_run = _safe_get(_safe_get(js, "aggregate", {}), "per_run", None)

        # Grab per-file aggregate action counts if present
        agg = _safe_get(js, "aggregate", {})
        act = _safe_get(agg, "action_counts_total", {})
        if act:
            per_file_action_totals.append(act)

        if isinstance(per_run, list) and per_run:
            for rec in per_run:
                r = dict(rec)
                r["_source_file"] = str(f)
                per_run_records.append(r)
        else:
            # Synthesize a run-like record from aggregate (best effort)
            synthesized = {}
            for k in (
                "final_cum_reward", "final_num_imaged", "final_num_imaged_illuminated",
                "final_num_downlinked_total", "final_num_downlinked_useful",
                "total_actions", "imaging_action_count", "non_imaging_action_count",
                "charging_events_count", "downlink_events_count", "desat_events_count",
                "shield_interventions_count", "shield_policy_disagreements_count",
                "mean_initial_ang_error", "std_initial_ang_error",
                "mean_target_distance", "std_target_distance",
                "mean_illumination_status", "num_target_above_illumination_threshold",
            ):
                if k in agg:
                    synthesized[k] = agg[k]
            if synthesized:
                synthesized["_source_file"] = str(f)
                per_run_records.append(synthesized)

    if not per_run_records:
        raise RuntimeError("Found result files, but no per-run records could be extracted.")
    return per_run_records, per_file_action_totals, files

def aggregate_metrics(per_run_records, per_file_action_totals):
    rewards            = _nums([r.get("final_cum_reward") for r in per_run_records])
    imaged             = _nums([r.get("final_num_imaged") for r in per_run_records])
    imaged_illum       = _nums([r.get("final_num_imaged_illuminated") for r in per_run_records])
    dl_total           = _nums([r.get("final_num_downlinked_total") for r in per_run_records])
    dl_useful          = _nums([r.get("final_num_downlinked_useful") for r in per_run_records])
    charging_events    = _nums([r.get("charging_events_count") for r in per_run_records])
    downlink_events    = _nums([r.get("downlink_events_count") for r in per_run_records])
    desat_events       = _nums([r.get("desat_events_count") for r in per_run_records])
    shield_interv      = _nums([r.get("shield_interventions_count") for r in per_run_records])
    shield_disagree    = _nums([r.get("shield_policy_disagreements_count") for r in per_run_records])
    mean_init_ang      = _nums([r.get("mean_initial_ang_error") for r in per_run_records])
    std_init_ang       = _nums([r.get("std_initial_ang_error") for r in per_run_records])
    mean_tgt_dist      = _nums([r.get("mean_target_distance") for r in per_run_records])
    std_tgt_dist       = _nums([r.get("std_target_distance") for r in per_run_records])
    mean_illum         = _nums([r.get("mean_illumination_status") for r in per_run_records])
    num_above_illum    = _nums([r.get("num_target_above_illumination_threshold") for r in per_run_records])
    img_counts         = _nums([r.get("imaging_action_count") for r in per_run_records])
    nonimg_counts      = _nums([r.get("non_imaging_action_count") for r in per_run_records])

    action_counts_total = _sum_dicts_of_ints(per_file_action_totals)

    overall = {}
    overall["n_runs"] = len(per_run_records)
    overall["reward_mean"], overall["reward_std"], overall["reward_min"], overall["reward_max"] = mean_std_min_max(rewards)
    overall["imaged_mean"], overall["imaged_std"], _, _ = mean_std_min_max(imaged)
    overall["imaged_illuminated_mean"], overall["imaged_illuminated_std"], _, _ = mean_std_min_max(imaged_illum)
    overall["downlinked_total_mean"], overall["downlinked_total_std"], _, _ = mean_std_min_max(dl_total)
    overall["downlinked_useful_mean"], overall["downlinked_useful_std"], _, _ = mean_std_min_max(dl_useful)

    # totals for event counters
    overall["charging_events_total"] = int(sum(charging_events))
    overall["downlink_events_total"] = int(sum(downlink_events))
    overall["desat_events_total"]    = int(sum(desat_events))

    overall["shield_interventions_total"]        = int(sum(shield_interv))
    overall["shield_policy_disagreements_total"] = int(sum(shield_disagree))

    if img_counts:
        overall["imaging_action_count_mean"]     = float(np.mean(img_counts))
    if nonimg_counts:
        overall["non_imaging_action_count_mean"] = float(np.mean(nonimg_counts))

    # angles, distances
    if mean_init_ang:
        overall["mean_initial_ang_error_mean"] = float(np.mean(mean_init_ang))
    if std_init_ang:
        overall["std_initial_ang_error_mean"]  = float(np.mean(std_init_ang))
    if mean_tgt_dist:
        overall["mean_target_distance_mean"]   = float(np.mean(mean_tgt_dist))
    if std_tgt_dist:
        overall["std_target_distance_mean"]    = float(np.mean(std_tgt_dist))

    # illumination (NEW: full stats)
    illum_mean, illum_std, illum_min, illum_max = mean_std_min_max(mean_illum)
    overall["illumination_fraction_mean"] = illum_mean        # fraction 0..1
    overall["illumination_fraction_std"]  = illum_std
    overall["illumination_fraction_min"]  = illum_min
    overall["illumination_fraction_max"]  = illum_max

    nta_mean, nta_std, nta_min, nta_max = mean_std_min_max(num_above_illum)
    overall["targets_above_illum_threshold_mean"] = nta_mean
    overall["targets_above_illum_threshold_std"]  = nta_std
    overall["targets_above_illum_threshold_min"]  = nta_min
    overall["targets_above_illum_threshold_max"]  = nta_max

    overall["action_counts_total"] = {str(k): int(v) for k, v in sorted(action_counts_total.items())}
    return overall, {
        "rewards": rewards,
        "shield_interv": shield_interv,
        "action_counts_total": action_counts_total
    }


def save_outputs(root: Path, overall: dict, per_run_records, files_scanned, series):
    out_dir = root / f"compiled__{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Print summary
    print("\n===== OVERALL RESULTS =====")
    print(f"Runs aggregated: {overall['n_runs']}")
    print(f"Reward: mean={overall['reward_mean']:.3f}  std={overall['reward_std']:.3f}  "
          f"min={overall['reward_min']:.3f}  max={overall['reward_max']:.3f}")
    print(f"Imaged (all): mean={overall['imaged_mean']:.3f}  std={overall['imaged_std']:.3f}")
    if "imaged_illuminated_mean" in overall:
        print(f"Imaged (illuminated): mean={overall['imaged_illuminated_mean']:.3f}  "
              f"std={overall['imaged_illuminated_std']:.3f}")
    print(f"Downlinks total: mean={overall['downlinked_total_mean']:.3f}  "
          f"std={overall['downlinked_total_std']:.3f}")
    print(f"Downlinks useful: mean={overall['downlinked_useful_mean']:.3f}  "
          f"std={overall['downlinked_useful_std']:.3f}")

    # NEW: illumination stats (both the per-run fraction and the absolute count over threshold)
    print("Illumination fraction (per-run mean over targets): "
          f"mean={overall['illumination_fraction_mean']*100:.2f}%  "
          f"std={overall['illumination_fraction_std']*100:.2f}%  "
          f"min={overall['illumination_fraction_min']*100:.2f}%  "
          f"max={overall['illumination_fraction_max']*100:.2f}%")
    print("Targets above illumination threshold: "
          f"mean={overall['targets_above_illum_threshold_mean']:.1f}  "
          f"std={overall['targets_above_illum_threshold_std']:.1f}  "
          f"min={overall['targets_above_illum_threshold_min']:.0f}  "
          f"max={overall['targets_above_illum_threshold_max']:.0f}")

    print(f"Events totals: charge={overall['charging_events_total']}, "
          f"downlink={overall['downlink_events_total']}, desat={overall['desat_events_total']}")
    print(f"Shield totals: interventions={overall['shield_interventions_total']}, "
          f"disagreements={overall['shield_policy_disagreements_total']}")
    print(f"Action counts total (keys are action IDs 0–12): {overall['action_counts_total']}")

    # JSON summary
    summary_path = out_dir / "overall_summary.json"
    with open(summary_path, "w") as fh:
        json.dump({
            "source_root": str(root),
            "overall": overall,
            "per_run": per_run_records,
            "files_scanned": [str(p) for p in files_scanned],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }, fh, indent=2)
    print(f"Wrote overall summary → {summary_path}")

    # Optional: CSV of per-run rows (you already include both columns in headers)
    csv_path = out_dir / "per_run_rows.csv"
    all_keys = set()
    for r in per_run_records:
        all_keys.update(r.keys())
    core = [
        "final_cum_reward", "final_num_imaged", "final_num_imaged_illuminated",
        "final_num_downlinked_total", "final_num_downlinked_useful",
        "total_actions", "imaging_action_count", "non_imaging_action_count",
        "charging_events_count", "downlink_events_count", "desat_events_count",
        "shield_interventions_count", "shield_policy_disagreements_count",
        "mean_initial_ang_error", "std_initial_ang_error",
        "mean_target_distance", "std_target_distance",
        "mean_illumination_status", "num_target_above_illumination_threshold",
        "seed", "run_index", "elapsed_seconds", "_source_file"
    ]
    rest = [k for k in sorted(all_keys) if k not in core]
    headers = core + rest
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=headers)
        w.writeheader()
        for r in per_run_records:
            w.writerow({k: r.get(k, "") for k in headers})
    print(f"Wrote per-run CSV → {csv_path}")

    # -------- Plots (unchanged) --------
    rewards = series["rewards"]
    if rewards:
        fig1, ax1 = plt.subplots(figsize=(9,5))
        ax1.hist(rewards, bins=min(20, max(5, len(rewards)//2)))
        ax1.set_title("Final reward distribution (all runs)")
        ax1.set_xlabel("final_cum_reward")
        ax1.set_ylabel("count")
        ax1.axvline(overall["reward_mean"], linestyle="--")
        fig1.savefig(out_dir / "plot_rewards_hist.png", bbox_inches="tight")
        plt.close(fig1)

    action_counts_total = series["action_counts_total"]
    if action_counts_total:
        img_total   = sum(v for k, v in action_counts_total.items() if 0 <= int(k) <= 9)
        charge_total   = action_counts_total.get(10, 0)
        downlink_total = action_counts_total.get(11, 0)
        desat_total    = action_counts_total.get(12, 0)
        total_all = max(1, img_total + charge_total + downlink_total + desat_total)

        avg_counts = np.array([
            img_total/overall["n_runs"],
            charge_total/overall["n_runs"],
            downlink_total/overall["n_runs"],
            desat_total/overall["n_runs"],
        ], dtype=float)

        pct = (np.array([img_total, charge_total, downlink_total, desat_total], dtype=float) / total_all) * 100.0
        labels = ["image (0–9)", "charge (10)", "downlink (11)", "desat (12)"]
        x = np.arange(4)

        fig2, axL = plt.subplots(figsize=(9,5))
        bars = axL.bar(x, avg_counts)
        axL.set_ylabel("Average count per run")
        axL.set_xticks(x)
        axL.set_xticklabels(labels)
        axL.set_title("Action distribution — avg count & % share")

        axR = axL.twinx()
        axR.plot(x, pct, marker="o", linewidth=2)
        axR.set_ylabel("Percentage of all actions (%)")
        axR.set_ylim(0, 100)

        for xi, b in zip(x, bars):
            h = b.get_height()
            axL.text(xi, h, f"{h:.1f}", ha="center", va="bottom", fontsize=9)
        for xi, p in zip(x, pct):
            axR.text(xi, p, f"{p:.1f}%", ha="center", va="bottom", fontsize=9)

        axL.grid(True, axis="y", linestyle="--", alpha=0.4)
        fig2.savefig(out_dir / "plot_actions_fourbar.png", bbox_inches="tight")
        plt.close(fig2)

    shield_interv = series["shield_interv"]
    if shield_interv and rewards and len(shield_interv) == len(rewards):
        fig3, ax3 = plt.subplots(figsize=(9,5))
        ax3.scatter(shield_interv, rewards)
        ax3.set_xlabel("shield_interventions_count")
        ax3.set_ylabel("final_cum_reward")
        ax3.set_title("Shield interventions vs. reward")
        fig3.savefig(out_dir / "plot_interventions_vs_reward.png", bbox_inches="tight")
        plt.close(fig3)

    print(f"Saved plots → {out_dir}")


# --------------------------- CLI ---------------------------
def main():
    p = argparse.ArgumentParser(description="Compile *_results_summary.json metrics.")
    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default="heuristic",
                   help="Choose a predefined data folder (you can override with --root).")
    p.add_argument("--root", type=str, default=None,
                   help="Custom root folder to scan (overrides preset).")
    args = p.parse_args()

    root = Path(args.root).resolve() if args.root else PRESETS[args.preset]
    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")

    per_run_records, per_file_action_totals, files_scanned = gather_results(root)
    overall, series = aggregate_metrics(per_run_records, per_file_action_totals)
    save_outputs(root, overall, per_run_records, files_scanned, series)

if __name__ == "__main__":
    main()
