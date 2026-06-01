#!/usr/bin/env python3
"""Aggregate AMOS 2026 GAT reward-sweep Monte Carlo outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICY_TAGS = (
    "00d100i",
    "10d90i",
    "20d80i",
    "30d70i",
    "40d60i",
    "50d50i",
    "75d25i",
    "100d00i",
)


def parse_args() -> argparse.Namespace:
    user = os.environ.get("USER", "unknown")
    default_root = Path(
        f"/scratch/alpine/{user}/amos2026_mc/gat_full_actions_eval_100d00i"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <input-root>/analysis.",
    )
    parser.add_argument(
        "--expected-seeds",
        default="0:100",
        help="Expected Python-style seed range, for example 0:10 for the smoke block.",
    )
    return parser.parse_args()


def nested_get(payload: dict[str, Any], *keys: str, default=np.nan):
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def latest_file(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def parse_expected_seeds(spec: str) -> list[int]:
    try:
        start_text, stop_text = spec.split(":", 1)
        start, stop = int(start_text), int(stop_text)
    except ValueError as exc:
        raise ValueError("--expected-seeds must look like 0:10 or 0:100") from exc
    if stop <= start:
        raise ValueError("--expected-seeds stop must be greater than start")
    return list(range(start, stop))


def step_metrics(steps_path: Path | None) -> dict[str, Any]:
    if steps_path is None or not steps_path.exists():
        return {}
    steps = pd.read_csv(steps_path)
    if steps.empty:
        return {}

    result: dict[str, Any] = {"num_steps": int(len(steps))}
    if "action_id" in steps:
        action = pd.to_numeric(steps["action_id"], errors="coerce").dropna().astype(int)
        denom = max(1, len(action))
        result.update(
            {
                "frac_image_actions": float(((action >= 3) & (action <= 12)).sum() / denom),
                "frac_charge_actions": float((action == 0).sum() / denom),
                "frac_downlink_actions": float((action == 1).sum() / denom),
                "frac_desat_actions": float((action == 2).sum() / denom),
            }
        )
    for source, output in (
        ("battery_frac_after", "final_battery_fraction"),
        ("storage_frac_after", "final_storage_fraction"),
        ("t_after", "final_sim_time_sec"),
    ):
        if source in steps:
            values = pd.to_numeric(steps[source], errors="coerce").dropna()
            if not values.empty:
                result[output] = float(values.iloc[-1])
    return result


def load_record(status_path: Path) -> dict[str, Any]:
    status = json.loads(status_path.read_text())
    seed_dir = status_path.parent
    row: dict[str, Any] = {
        "policy_tag": status.get("policy_tag"),
        "seed": status.get("seed"),
        "state": status.get("state"),
        "returncode": status.get("returncode"),
        "elapsed_seconds": status.get("elapsed_seconds"),
        "status_path": str(status_path),
        "checkpoint_iteration": nested_get(status, "policy", "checkpoint_iteration"),
        "checkpoint_dir": nested_get(status, "policy", "checkpoint_dir", default=None),
        "evaluation_reward_mix": status.get("evaluation_reward_mix"),
        "score_ground_value_100d00i": np.nan,
    }
    metrics_path = latest_file(list(seed_dir.rglob("metrics_*.json")))
    if metrics_path is None:
        return row

    payload = json.loads(metrics_path.read_text())
    row.update(
        {
            "metrics_path": str(metrics_path),
            "score_ground_value_100d00i": nested_get(
                payload, "data", "cumulativeRewardSS1"
            ),
            "illuminated_images": nested_get(payload, "data", "illuminated_images"),
            "confirmed_illuminated_images": nested_get(
                payload, "data", "confirmed_illuminated_images"
            ),
            "pending_illuminated_images_onboard": nested_get(
                payload, "data", "pending_illuminated_images_onboard"
            ),
            "pending_images_onboard": nested_get(
                payload, "data", "pending_images_onboard"
            ),
            "mean_chosen_target_priority": nested_get(
                payload, "data", "mean_target_priority"
            ),
            "target_imaging_count": nested_get(
                payload, "summary", "target_imaging_count"
            ),
            "charge_action_count": nested_get(
                payload, "summary", "charge_action_count"
            ),
            "downlink_action_count": nested_get(
                payload, "summary", "downlink_action_count"
            ),
            "desat_action_count": nested_get(payload, "summary", "desat_action_count"),
            "acq_success_rate": nested_get(payload, "summary", "acq_success_rate"),
        }
    )
    row.update(step_metrics(metrics_path.parent / "steps.csv"))
    return row


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "score_ground_value_100d00i",
        "illuminated_images",
        "target_imaging_count",
        "charge_action_count",
        "downlink_action_count",
        "desat_action_count",
        "acq_success_rate",
        "mean_chosen_target_priority",
        "final_battery_fraction",
        "final_storage_fraction",
        "elapsed_seconds",
    ]
    rows = []
    for tag in POLICY_TAGS:
        subset = records[
            (records["policy_tag"] == tag)
            & (records["state"] == "completed")
            & records["score_ground_value_100d00i"].notna()
        ]
        row: dict[str, Any] = {"policy_tag": tag, "n_runs": int(len(subset))}
        for metric in metric_columns:
            values = pd.to_numeric(subset.get(metric, pd.Series(dtype=float)), errors="coerce")
            values = values.dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        score_std = row["score_ground_value_100d00i_std"]
        row["score_ground_value_100d00i_ci95"] = (
            float(1.96 * score_std / math.sqrt(len(subset)))
            if len(subset) > 1 and np.isfinite(score_std)
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_plot(records: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    completed = records[
        (records["state"] == "completed")
        & records["score_ground_value_100d00i"].notna()
    ].copy()
    if completed.empty:
        return

    x = np.arange(len(POLICY_TAGS))
    fig, (ax_box, ax_mean) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    values = [
        pd.to_numeric(
            completed.loc[
                completed["policy_tag"] == tag, "score_ground_value_100d00i"
            ],
            errors="coerce",
        ).dropna()
        for tag in POLICY_TAGS
    ]
    ax_box.boxplot(values, positions=x, widths=0.55, showmeans=True)
    for index, tag_values in enumerate(values):
        if len(tag_values):
            jitter = np.linspace(-0.12, 0.12, len(tag_values))
            ax_box.scatter(
                np.full(len(tag_values), index) + jitter,
                tag_values,
                s=18,
                alpha=0.65,
            )
    ax_box.set_ylabel("Ground-value score")
    ax_box.set_title("AMOS 2026 GAT Policies Scored With Common 100d00i Reward")
    ax_box.grid(axis="y", alpha=0.25)

    means = summary["score_ground_value_100d00i_mean"].to_numpy(dtype=float)
    ci95 = summary["score_ground_value_100d00i_ci95"].to_numpy(dtype=float)
    ax_mean.errorbar(x, means, yerr=ci95, marker="o", capsize=4)
    ax_mean.set_ylabel("Mean score +/- 95% CI")
    ax_mean.set_xlabel("Training reward mix")
    ax_mean.set_xticks(x, POLICY_TAGS)
    ax_mean.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ground_value_score_by_training_reward_mix.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.input_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_seeds = parse_expected_seeds(args.expected_seeds)

    status_paths = sorted(args.input_root.rglob("mc_status.json"))
    records = pd.DataFrame(load_record(path) for path in status_paths)
    if records.empty:
        print(f"No mc_status.json files found below {args.input_root}")
        return 1

    records["seed"] = pd.to_numeric(records["seed"], errors="coerce")
    records.to_csv(output_dir / "per_run.csv", index=False)
    summary = summarize(records)
    summary.to_csv(output_dir / "summary_by_policy.csv", index=False)

    observed = {
        (str(row.policy_tag), int(row.seed))
        for row in records.itertuples()
        if pd.notna(row.seed)
    }
    expected = {(tag, seed) for tag in POLICY_TAGS for seed in expected_seeds}
    missing = sorted(expected - observed)
    failed = records[records["state"] != "completed"].copy()
    pd.DataFrame(missing, columns=["policy_tag", "seed"]).to_csv(
        output_dir / "missing_runs.csv", index=False
    )
    failed.to_csv(output_dir / "failed_runs.csv", index=False)
    write_plot(records, summary, output_dir)

    report = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_root": str(args.input_root.resolve()),
        "expected_seeds": args.expected_seeds,
        "status_files": len(status_paths),
        "completed_runs": int((records["state"] == "completed").sum()),
        "failed_or_incomplete_runs": int(len(failed)),
        "missing_runs": len(missing),
    }
    (output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    print(summary[["policy_tag", "n_runs", "score_ground_value_100d00i_mean", "score_ground_value_100d00i_std"]].to_string(index=False))
    print()
    print(f"Completed runs: {report['completed_runs']}")
    print(f"Failed or incomplete runs: {report['failed_or_incomplete_runs']}")
    print(f"Missing expected runs: {report['missing_runs']}")
    print(f"Analysis written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
