#!/usr/bin/env python3
"""Analyze exact AMOS 2026 priority-response Monte Carlo outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_ORDER = ("HIO", "SHIO", "Control")
CLASS_COLORS = {
    "HIO": "#355f8d",
    "SHIO": "#2a9d8f",
    "Control": "#5f5f5f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Focused campaign root containing priority_response_targets.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <input-root>/priority_response_analysis.",
    )
    parser.add_argument(
        "--expected-seeds",
        default="0:100",
        help="Half-open seed range START:STOP.",
    )
    return parser.parse_args()


def parse_seed_range(value: str) -> set[int]:
    start, stop = (int(part) for part in value.split(":", maxsplit=1))
    return set(range(start, stop))


def parse_bool(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(bool)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def ci95(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 2:
        return math.nan
    return float(1.96 * values.std(ddof=1) / math.sqrt(len(values)))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def finish_axis(ax: plt.Axes, *, ylabel: str | None = None) -> None:
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="0.86", linewidth=0.55, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.15")
        spine.set_linewidth(0.8)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


def load_target_outputs(input_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(input_root.rglob("priority_response_targets.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["source_path"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No priority_response_targets.csv files found below {input_root}"
        )
    data = pd.concat(frames, ignore_index=True)
    data["kind"] = data["response_class"].replace({"CONTROL": "Control"})
    for column in (
        "selected_on_first_presentation",
        "successful_image_selected_on_first_presentation",
    ):
        data[column] = parse_bool(data[column])
    return data


def validate(data: pd.DataFrame, expected_seeds: set[int]) -> None:
    required_columns = {
        "first_eligible_visible_delay_sec",
        "first_successful_image_delay_sec",
        "first_useful_downlink_delay_sec",
    }
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(
            "The campaign predates exact eligible-visible-access telemetry. "
            f"Missing columns: {missing_columns}"
        )
    observed_seeds = set(pd.to_numeric(data["seed"], errors="raise").astype(int))
    if observed_seeds != expected_seeds:
        missing = sorted(expected_seeds - observed_seeds)
        extra = sorted(observed_seeds - expected_seeds)
        raise ValueError(f"Seed mismatch: missing={missing}, extra={extra}")
    duplicates = data.duplicated(["seed", "target_id", "kind"]).sum()
    if duplicates:
        raise ValueError(f"Found {duplicates} duplicate seed/target/class rows.")
    counts = data.groupby(["seed", "kind"]).size().unstack(fill_value=0)
    expected_counts = {"HIO": 5, "SHIO": 3, "Control": 8}
    for kind, expected in expected_counts.items():
        if kind not in counts or not (counts[kind] == expected).all():
            raise ValueError(f"Expected {expected} {kind} rows per seed.")


def build_per_run(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, kind), group in data.groupby(["seed", "kind"]):
        image_delay = pd.to_numeric(
            group["first_successful_image_delay_sec"], errors="coerce"
        )
        delivery_delay = pd.to_numeric(
            group["first_useful_downlink_delay_sec"], errors="coerce"
        )
        packet_latency = pd.to_numeric(
            group["first_verified_useful_capture_to_downlink_sec"],
            errors="coerce",
        )
        presentations = pd.to_numeric(
            group["presentations_through_first_success_selection"],
            errors="coerce",
        )
        imaged = image_delay.notna()
        delivered = delivery_delay.notna()
        rows.append(
            {
                "seed": int(seed),
                "kind": kind,
                "target_count": int(len(group)),
                "successfully_imaged_pct": float(100.0 * imaged.mean()),
                "usefully_delivered_pct": float(100.0 * delivered.mean()),
                "median_image_delay_min": float(image_delay.median() / 60.0),
                "median_useful_downlink_delay_min": float(
                    delivery_delay.median() / 60.0
                ),
                "median_packet_capture_to_downlink_min": float(
                    packet_latency.median() / 60.0
                ),
                "selected_on_first_presentation_pct": float(
                    100.0 * group["selected_on_first_presentation"].mean()
                ),
                "successful_on_first_presentation_pct": float(
                    100.0
                    * group["successful_image_selected_on_first_presentation"].mean()
                ),
                "median_presentations_through_success": float(
                    presentations[imaged].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        column
        for column in per_run.columns
        if column not in {"seed", "kind", "target_count"}
    ]
    for kind in CLASS_ORDER:
        group = per_run[per_run["kind"] == kind]
        row: dict[str, float | int | str] = {
            "kind": kind,
            "run_count": int(group["seed"].nunique()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_ci95"] = ci95(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def build_stage_per_run(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, kind), group in data.groupby(["seed", "kind"]):
        rows.append(
            {
                "seed": int(seed),
                "kind": kind,
                "first_eligible_visible_delay_min": float(
                    pd.to_numeric(
                        group["first_eligible_visible_delay_sec"], errors="coerce"
                    ).median()
                    / 60.0
                ),
                "first_successful_image_delay_min": float(
                    pd.to_numeric(
                        group["first_successful_image_delay_sec"],
                        errors="coerce",
                    ).median()
                    / 60.0
                ),
                "first_useful_downlink_delay_min": float(
                    pd.to_numeric(
                        group["first_useful_downlink_delay_sec"],
                        errors="coerce",
                    ).median()
                    / 60.0
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_stages(stages: pd.DataFrame, output_dir: Path) -> None:
    specs = (
        (
            "first_eligible_visible_delay_min",
            "First eligible\nvisible access",
        ),
        ("first_successful_image_delay_min", "First successful\nimage"),
        ("first_useful_downlink_delay_min", "First useful\ndownlink"),
    )
    fig, ax = plt.subplots(figsize=(7.0, 3.55), constrained_layout=True)
    x = np.arange(len(specs), dtype=float)
    width = 0.22
    offsets = (-width, 0.0, width)
    for offset, kind in zip(offsets, CLASS_ORDER, strict=True):
        means = []
        intervals = []
        for metric, _ in specs:
            values = stages.loc[stages["kind"] == kind, metric]
            means.append(float(values.mean()))
            intervals.append(ci95(values))
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=intervals,
            capsize=2.3,
            color=CLASS_COLORS[kind],
            edgecolor="0.2",
            linewidth=0.6,
            label=kind,
            zorder=3,
        )
    ax.set_xticks(x, [label for _, label in specs])
    ax.set_ylim(bottom=0.0)
    finish_axis(ax, ylabel="Latency after priority injection [min]")
    ax.legend(frameon=True, edgecolor="0.25", loc="upper left")
    save_figure(fig, output_dir, "priority_response_three_stage_exact")


def plot_first_presentation(per_run: pd.DataFrame, output_dir: Path) -> None:
    specs = (
        (
            "selected_on_first_presentation_pct",
            "Selected at first\npresentation [%]",
        ),
        (
            "successful_on_first_presentation_pct",
            "Successful image selected at\nfirst presentation [%]",
        ),
        (
            "median_presentations_through_success",
            "Candidate presentations through\nsuccessful selection",
        ),
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.95),
        constrained_layout=True,
    )
    x = np.arange(len(CLASS_ORDER), dtype=float)
    for ax, (metric, ylabel) in zip(axes, specs, strict=True):
        means = []
        intervals = []
        for kind in CLASS_ORDER:
            values = per_run.loc[per_run["kind"] == kind, metric]
            means.append(float(values.mean()))
            intervals.append(ci95(values))
        ax.bar(
            x,
            means,
            yerr=intervals,
            capsize=2.3,
            color=[CLASS_COLORS[kind] for kind in CLASS_ORDER],
            edgecolor="0.2",
            linewidth=0.6,
            zorder=3,
        )
        ax.set_xticks(x, ["HIO", "SHIO", "Random\ncontrol"])
        ax.set_ylim(bottom=0.0)
        finish_axis(ax, ylabel=ylabel)
    save_figure(fig, output_dir, "priority_first_presentation_response_exact")


def plot_delivery(per_run: pd.DataFrame, output_dir: Path) -> None:
    specs = (
        ("usefully_delivered_pct", "Tracked targets usefully\ndelivered [%]"),
        (
            "median_packet_capture_to_downlink_min",
            "Median packet-matched capture-to-\ndownlink latency [min]",
        ),
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.15),
        constrained_layout=True,
    )
    x = np.arange(len(CLASS_ORDER), dtype=float)
    for ax, (metric, ylabel) in zip(axes, specs, strict=True):
        means = []
        intervals = []
        for kind in CLASS_ORDER:
            values = per_run.loc[per_run["kind"] == kind, metric]
            means.append(float(values.mean()))
            intervals.append(ci95(values))
        ax.bar(
            x,
            means,
            yerr=intervals,
            capsize=2.3,
            color=[CLASS_COLORS[kind] for kind in CLASS_ORDER],
            edgecolor="0.2",
            linewidth=0.6,
            zorder=3,
        )
        ax.set_xticks(x, ["HIO", "SHIO", "Random control"])
        ax.set_ylim(bottom=0.0)
        finish_axis(ax, ylabel=ylabel)
    save_figure(fig, output_dir, "priority_exact_delivery_response")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_root / "priority_response_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    data = load_target_outputs(args.input_root)
    validate(data, parse_seed_range(args.expected_seeds))
    per_run = build_per_run(data)
    summary = build_summary(per_run)
    stages = build_stage_per_run(data)

    data.to_csv(output_dir / "priority_response_targets_combined.csv", index=False)
    per_run.to_csv(output_dir / "priority_response_per_run.csv", index=False)
    summary.to_csv(output_dir / "priority_response_summary_by_class.csv", index=False)
    stages.to_csv(output_dir / "priority_response_stages_per_run.csv", index=False)
    plot_stages(stages, output_dir)
    plot_first_presentation(per_run, output_dir)
    plot_delivery(per_run, output_dir)
    print(f"Wrote exact priority-response analysis to {output_dir}")


if __name__ == "__main__":
    main()
