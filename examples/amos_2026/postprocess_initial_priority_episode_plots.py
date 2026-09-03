#!/usr/bin/env python3
"""Create per-episode plots from a completed initial-priority campaign.

The Monte Carlo campaign stores detailed CSV and NumPy outputs for every seed
but intentionally skips plotting during the expensive dynamics run.  This
postprocessor reads those saved products and creates publication-readable PDF
and PNG diagnostics without rerunning Basilisk or policy inference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CASE_LABELS = {
    "ground_confirmation": "Ground-confirmation cooldown",
    "one_orbit": "One-orbit cooldown",
    "single": "Single episode",
}
ACTION_ORDER = ("Charge", "Downlink", "Desat", "Imaging")
ACTION_COLORS = {
    "Charge": "#56B4E9",
    "Downlink": "#CC79A7",
    "Desat": "#D55E00",
    "Imaging": "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Completed campaign root containing both cooldown cases.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to <input-root>/per_seed_plots.",
    )
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        help="Process one evaluation directory instead of a full campaign.",
    )
    parser.add_argument("--case", default="single")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def parse_bool(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(bool)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def numeric_column(
    frame: pd.DataFrame, column: str, default: float = 0.0
) -> pd.Series:
    """Return a numeric Series aligned with ``frame`` even if a column is absent."""
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def seed_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.search(r"seed[_-]?(\d+)", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not infer a seed from {path}")


def locate_evaluation(
    input_root: Path, case: str, seed: int, recorded_path: str | None
) -> Path:
    if recorded_path:
        recorded = Path(recorded_path)
        if recorded.is_dir():
            return recorded.resolve()

    seed_dir = input_root / case / f"seed_{seed:03d}"
    candidates = sorted(
        path
        for path in seed_dir.iterdir()
        if path.is_dir()
        and (path / "steps.csv").is_file()
        and (path / "images.csv").is_file()
        and (path / "verified_deliveries.csv").is_file()
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one completed evaluation under {seed_dir}, "
            f"found {len(candidates)}"
        )
    return candidates[0].resolve()


def discover_evaluations(
    args: argparse.Namespace,
) -> tuple[Path, list[dict[str, object]]]:
    if args.evaluation_dir is not None:
        evaluation_dir = args.evaluation_dir.expanduser().resolve()
        if not evaluation_dir.is_dir():
            raise FileNotFoundError(evaluation_dir)
        seed = args.seed if args.seed is not None else seed_from_path(evaluation_dir)
        input_root = (
            args.input_root.expanduser().resolve()
            if args.input_root is not None
            else evaluation_dir
        )
        return input_root, [
            {"case": args.case, "seed": seed, "evaluation_dir": evaluation_dir}
        ]

    if args.input_root is None:
        raise ValueError("Pass --input-root or --evaluation-dir")
    input_root = args.input_root.expanduser().resolve()
    audit_path = (
        input_root
        / "analysis_initial_priority_allocation"
        / "campaign_audit.csv"
    )
    if not audit_path.is_file():
        raise FileNotFoundError(f"Campaign audit not found: {audit_path}")
    audit = pd.read_csv(audit_path)
    if "valid" in audit and not parse_bool(audit["valid"]).all():
        raise ValueError("Campaign audit contains invalid episodes")

    evaluations = []
    for row in audit.sort_values(["case", "seed"]).itertuples(index=False):
        case = str(row.case)
        seed = int(row.seed)
        evaluations.append(
            {
                "case": case,
                "seed": seed,
                "evaluation_dir": locate_evaluation(
                    input_root,
                    case,
                    seed,
                    str(getattr(row, "evaluation_dir", "")) or None,
                ),
            }
        )
    if len(evaluations) != args.expected_episodes:
        raise ValueError(
            f"Expected {args.expected_episodes} episodes, found {len(evaluations)}"
        )
    return input_root, evaluations


def required_table(evaluation_dir: Path, name: str) -> pd.DataFrame:
    path = evaluation_dir / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def optional_table(evaluation_dir: Path, name: str) -> pd.DataFrame:
    path = evaluation_dir / name
    return pd.read_csv(path, low_memory=False) if path.is_file() else pd.DataFrame()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=250)
    plt.close(fig)
    return paths


def episode_title(case: str, seed: int) -> str:
    return f"{CASE_LABELS.get(case, case.replace('_', ' ').title())}, seed {seed}"


def cumulative_series(times: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    finite = pd.to_numeric(times, errors="coerce").dropna().sort_values().to_numpy()
    if finite.size == 0:
        return np.array([0.0]), np.array([0])
    return np.r_[0.0, finite], np.arange(finite.size + 1)


def cumulative_axis_limit(*values: np.ndarray) -> float:
    maximum = max((float(np.max(value)) for value in values if value.size), default=0)
    return 300.0 if maximum <= 300.0 else 100.0 * math.ceil(maximum / 100.0)


def shade_windows(ax: plt.Axes, windows: pd.DataFrame) -> None:
    if windows.empty:
        return
    first = True
    for row in windows.itertuples(index=False):
        start = float(getattr(row, "window_open_sec"))
        stop = float(getattr(row, "window_close_sec"))
        ax.axvspan(
            start,
            stop,
            color="#55A868",
            alpha=0.10,
            linewidth=0,
            label="Ground-station access" if first else None,
            zorder=0,
        )
        first = False


def plot_action_distribution(
    steps: pd.DataFrame, title: str, output_dir: Path
) -> list[Path]:
    counts = steps["action_category"].value_counts()
    values = np.array([int(counts.get(action, 0)) for action in ACTION_ORDER])
    total = max(int(values.sum()), 1)
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    bars = ax.bar(
        np.arange(len(ACTION_ORDER)),
        values,
        color=[ACTION_COLORS[action] for action in ACTION_ORDER],
        edgecolor="0.2",
        linewidth=0.7,
    )
    for bar, count in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n({100 * count / total:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(np.arange(len(ACTION_ORDER)), ACTION_ORDER)
    ax.set_ylabel("Action count")
    ax.set_title(title)
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.margins(y=0.15)
    fig.tight_layout()
    return save_figure(fig, output_dir, "action_distribution")


def plot_resources_and_service(
    steps: pd.DataFrame,
    images: pd.DataFrame,
    deliveries: pd.DataFrame,
    windows: pd.DataFrame,
    title: str,
    output_dir: Path,
) -> list[Path]:
    steps = steps.sort_values("t_cmd")
    t = pd.to_numeric(steps["t_after"], errors="coerce").to_numpy(float)
    battery = pd.to_numeric(steps["battery_frac_after"], errors="coerce").to_numpy(float)
    storage = pd.to_numeric(steps["storage_frac_after"], errors="coerce").to_numpy(float)

    successful_images = images[numeric_column(images, "acq_success").eq(1)]
    image_time_column = (
        "t_acq" if "t_acq" in successful_images else "action_end_time"
    )
    image_times, image_counts = cumulative_series(successful_images[image_time_column])

    useful = deliveries[
        parse_bool(deliveries["useful_delivery"])
        if "useful_delivery" in deliveries
        else np.ones(len(deliveries), dtype=bool)
    ]
    delivery_times, delivery_counts = cumulative_series(useful["downlink_time"])

    fig, ax = plt.subplots(figsize=(9.1, 4.2))
    shade_windows(ax, windows)
    ax.plot(t, battery, color="#0072B2", linewidth=1.3, label="Battery fraction")
    ax.plot(t, storage, color="#E69F00", linewidth=1.3, label="Storage fraction")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Resource fraction")
    ax.set_ylim(0, 1.02)
    if t.size:
        ax.set_xlim(0, float(np.nanmax(t)))
    ax.grid(color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)

    for category, ymin, ymax in (
        ("Charge", 0.005, 0.020),
        ("Downlink", 0.025, 0.040),
        ("Desat", 0.045, 0.060),
    ):
        rows = steps[steps["action_category"].eq(category)]
        for row in rows.itertuples(index=False):
            ax.axvspan(
                float(row.t_cmd),
                float(row.t_after),
                ymin=ymin,
                ymax=ymax,
                color=ACTION_COLORS[category],
                alpha=0.9,
                linewidth=0,
            )

    count_ax = ax.twinx()
    count_ax.step(
        image_times,
        image_counts,
        where="post",
        color="#009E73",
        linewidth=1.4,
        label="Successful images",
    )
    count_ax.step(
        delivery_times,
        delivery_counts,
        where="post",
        color="#D55E00",
        linewidth=1.4,
        label="Useful deliveries",
    )
    count_ax.set_ylabel("Cumulative count")
    count_ax.set_ylim(0, cumulative_axis_limit(image_counts, delivery_counts))
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = count_ax.get_legend_handles_labels()
    by_label = dict(zip(labels1 + labels2, lines1 + lines2))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", ncol=2)
    ax.set_title(title)
    fig.tight_layout()
    return save_figure(fig, output_dir, "resources_and_cumulative_service")


def plot_target_availability(
    steps: pd.DataFrame, title: str, output_dir: Path
) -> list[Path]:
    columns = (
        ("eligible_target_count", "Eligible", "#0072B2"),
        ("imageable_eligible_count", "Imageable now", "#009E73"),
        ("pending_verification_target_count", "Pending verification", "#CC79A7"),
        ("cooldown_target_count", "Cooldown", "#D55E00"),
    )
    available = [item for item in columns if item[0] in steps]
    if not available:
        return []
    fig, ax = plt.subplots(figsize=(9.1, 3.9))
    t = pd.to_numeric(steps["t_cmd"], errors="coerce")
    for column, label, color in available:
        ax.step(
            t,
            pd.to_numeric(steps[column], errors="coerce"),
            where="post",
            label=label,
            color=color,
            linewidth=1.25,
        )
    desat = steps[steps["action_category"].eq("Desat")]
    for number, row in enumerate(desat.itertuples(index=False)):
        ax.axvspan(
            float(row.t_cmd),
            float(row.t_after),
            color=ACTION_COLORS["Desat"],
            alpha=0.12,
            linewidth=0,
            label="Desat action" if number == 0 else None,
        )
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Target count")
    ax.set_xlim(0, float(pd.to_numeric(steps["t_after"]).max()))
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.grid(color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    return save_figure(fig, output_dir, "target_availability_and_desat")


def plot_pointing_history(
    images: pd.DataFrame, title: str, output_dir: Path
) -> list[Path]:
    images = images.sort_values("t_cmd")
    success = numeric_column(images, "acq_success").eq(1)
    fig, ax = plt.subplots(figsize=(9.1, 3.9))
    ax.scatter(
        pd.to_numeric(images.loc[success, "t_cmd"], errors="coerce"),
        pd.to_numeric(images.loc[success, "azimuth_deg"], errors="coerce"),
        s=11,
        color="#0072B2",
        alpha=0.72,
        label="Azimuth, successful",
    )
    if (~success).any():
        ax.scatter(
            pd.to_numeric(images.loc[~success, "t_cmd"], errors="coerce"),
            pd.to_numeric(images.loc[~success, "azimuth_deg"], errors="coerce"),
            s=13,
            facecolors="none",
            edgecolors="#0072B2",
            alpha=0.65,
            label="Azimuth, unsuccessful",
        )
    elevation_ax = ax.twinx()
    elevation_artist = elevation_ax.scatter(
        pd.to_numeric(images["t_cmd"], errors="coerce"),
        pd.to_numeric(images["elevation_local_deg"], errors="coerce"),
        s=10,
        marker="x",
        color="#009E73",
        alpha=0.65,
        label="Elevation",
    )
    ax.set_xlabel("Imaging-command time [s]")
    ax.set_ylabel("Azimuth [deg]", color="#0072B2")
    elevation_ax.set_ylabel("Elevation [deg]", color="#009E73")
    ax.tick_params(axis="y", labelcolor="#0072B2")
    elevation_ax.tick_params(axis="y", labelcolor="#009E73")
    if not images.empty:
        ax.set_xlim(0, float(pd.to_numeric(images["t_cmd"]).max()))
    ax.grid(color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(elevation_artist)
    labels.append("Elevation")
    ax.legend(handles, labels, loc="upper right", ncol=2)
    ax.set_title(title)
    fig.tight_layout()
    return save_figure(fig, output_dir, "pointing_history")


def process_episode(
    evaluation_dir: Path,
    output_dir: Path,
    case: str,
    seed: int,
    overwrite: bool,
) -> dict[str, object]:
    expected = tuple(
        output_dir / f"{stem}.{extension}"
        for stem in (
            "action_distribution",
            "resources_and_cumulative_service",
            "pointing_history",
        )
        for extension in ("pdf", "png")
    )
    if not overwrite and all(path.is_file() for path in expected):
        return {
            "case": case,
            "seed": seed,
            "evaluation_dir": str(evaluation_dir),
            "output_dir": str(output_dir),
            "status": "already_complete",
        }

    steps = required_table(evaluation_dir, "steps.csv")
    images = required_table(evaluation_dir, "images.csv")
    deliveries = required_table(evaluation_dir, "verified_deliveries.csv")
    windows = optional_table(evaluation_dir, "ground_station_windows.csv")
    title = episode_title(case, seed)
    paths = []
    paths += plot_action_distribution(steps, title, output_dir)
    paths += plot_resources_and_service(
        steps, images, deliveries, windows, title, output_dir
    )
    paths += plot_target_availability(steps, title, output_dir)
    paths += plot_pointing_history(images, title, output_dir)
    return {
        "case": case,
        "seed": seed,
        "evaluation_dir": str(evaluation_dir),
        "output_dir": str(output_dir),
        "status": "created",
        "plot_file_count": len(paths),
        "step_count": len(steps),
        "image_command_count": len(images),
        "useful_delivery_count": int(
            parse_bool(deliveries["useful_delivery"]).sum()
            if "useful_delivery" in deliveries
            else len(deliveries)
        ),
    }


def main() -> int:
    args = parse_args()
    configure_style()
    input_root, evaluations = discover_evaluations(args)
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else input_root / "per_seed_plots"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    inventory = []
    for episode in evaluations:
        case = str(episode["case"])
        seed = int(episode["seed"])
        evaluation_dir = Path(episode["evaluation_dir"])
        output_dir = output_root / case / f"seed_{seed:03d}"
        inventory.append(
            process_episode(
                evaluation_dir,
                output_dir,
                case,
                seed,
                args.overwrite,
            )
        )
        print(f"[{len(inventory):03d}/{len(evaluations):03d}] {case} seed {seed}")

    inventory_frame = pd.DataFrame(inventory).sort_values(["case", "seed"])
    inventory_frame.to_csv(output_root / "per_seed_plot_inventory.csv", index=False)
    summary = {
        "schema_version": 1,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "episode_count": len(inventory_frame),
        "created_count": int(inventory_frame["status"].eq("created").sum()),
        "already_complete_count": int(
            inventory_frame["status"].eq("already_complete").sum()
        ),
    }
    (output_root / "postprocess_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
