#!/usr/bin/env python3
"""Mixed-regime diagnostics for AMOS 2026 GAT MC image selections.

This script focuses on target-regime behavior in the mixed LEO/MEO/GEO Monte
Carlo campaigns. It consumes the per-run ``images.csv`` files produced by the
evaluator and writes paper-style PNG/PDF figures plus summary CSVs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator


POLICY_ORDER = [
    "00d100i",
    "10d90i",
    "20d80i",
    "30d70i",
    "40d60i",
    "50d50i",
    "75d25i",
    "100d00i",
]
REGIME_ORDER = ["LEO", "MEO", "GEO"]
REGIME_COLORS = {
    "LEO": "#2C7BB6",
    "MEO": "#FDAE61",
    "GEO": "#D7191C",
}
CONDITION_COLORS = {
    "umbra": "#4B0082",
    "sunlit": "#E6AB02",
}
THIRD_LABELS = ["0-15 ks", "15-30 ks", "30-45 ks"]
GRID_ALPHA = 0.24
DPI = 320


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create mixed LEO/MEO/GEO diagnostics from AMOS 2026 MC images.csv files."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <input-root>/analysis_mixed_regime_plots.",
    )
    parser.add_argument(
        "--expected-seeds",
        default="0:100",
        help="Expected Python-style seed range, e.g. 0:100.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Generate plots even if some completed runs have not been transferred locally.",
    )
    return parser.parse_args()


def policy_alpha(policy_tag: str) -> float:
    match = re.fullmatch(r"(\d+)d(\d+)i", str(policy_tag))
    if not match:
        raise ValueError(f"Cannot parse policy tag {policy_tag}")
    d_weight = float(match.group(1))
    i_weight = float(match.group(2))
    return d_weight / (d_weight + i_weight)


def parse_expected_seeds(spec: str) -> list[int]:
    start_text, stop_text = spec.split(":", 1)
    start, stop = int(start_text), int(stop_text)
    if stop <= start:
        raise ValueError("--expected-seeds stop must be greater than start")
    return list(range(start, stop))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.labelsize": 15,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 140,
            "savefig.dpi": DPI,
            "axes.linewidth": 1.05,
        }
    )


def finish_axis(ax: plt.Axes, xlabel: str | None = None) -> None:
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.minorticks_on()
    ax.grid(True, alpha=GRID_ALPHA)
    ax.grid(True, which="minor", alpha=GRID_ALPHA * 0.55)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    for spine in ax.spines.values():
        spine.set_linewidth(1.05)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=DPI, pad_inches=0.06)
    plt.close(fig)


def status_paths_under_root(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("seeds_*/*/seed_*/mc_status.json"))


def metrics_files_for_seed(seed_dir: Path) -> list[Path]:
    return sorted(seed_dir.glob("metrics_*.json")) + sorted(
        seed_dir.glob("*/metrics_*.json")
    )


def images_files_for_seed(seed_dir: Path) -> list[Path]:
    return sorted(seed_dir.glob("images.csv")) + sorted(seed_dir.glob("*/images.csv"))


def status_record(status_path: Path) -> dict:
    seed_dir = status_path.parent
    policy_tag = status_path.parents[1].name
    seed = None
    try:
        seed = int(status_path.parent.name.removeprefix("seed_"))
    except ValueError:
        pass
    status_readable = True
    payload: dict = {}
    try:
        payload = json.loads(status_path.read_text())
    except json.JSONDecodeError:
        status_readable = False
    return {
        "policy_tag": payload.get("policy_tag", policy_tag),
        "seed": payload.get("seed", seed),
        "state": payload.get("state", "unreadable_status"),
        "status_readable": status_readable,
        "status_path": str(status_path),
        "metrics_files": len(metrics_files_for_seed(seed_dir)),
        "images_files": len(images_files_for_seed(seed_dir)),
    }


def load_status_table(input_root: Path) -> pd.DataFrame:
    rows = [status_record(path) for path in status_paths_under_root(input_root)]
    if not rows:
        return pd.DataFrame(
            columns=[
                "policy_tag",
                "seed",
                "state",
                "status_readable",
                "status_path",
                "metrics_files",
                "images_files",
            ]
        )
    df = pd.DataFrame(rows)
    df = df[df["policy_tag"].isin(POLICY_ORDER)].copy()
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    return df


def completeness_tables(
    status_df: pd.DataFrame, expected_seeds: Iterable[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = {(policy, seed) for policy in POLICY_ORDER for seed in expected_seeds}
    observed = {
        (str(row.policy_tag), int(row.seed))
        for row in status_df.itertuples()
        if pd.notna(row.seed)
    }
    missing_status = sorted(expected - observed)

    rows = []
    for policy in POLICY_ORDER:
        subset = status_df[status_df["policy_tag"] == policy]
        completed = subset[subset["state"] == "completed"]
        rows.append(
            {
                "policy_tag": policy,
                "alpha": policy_alpha(policy),
                "expected_runs": len(list(expected_seeds)),
                "status_files": int(len(subset)),
                "completed_status_files": int(len(completed)),
                "metrics_payloads": int((subset["metrics_files"] > 0).sum()),
                "images_payloads": int((subset["images_files"] > 0).sum()),
                "missing_status_files": int(
                    sum(1 for item in missing_status if item[0] == policy)
                ),
                "unreadable_status_files": int((subset["state"] == "unreadable_status").sum()),
                "missing_metrics_payloads": int(
                    ((subset["state"] != "completed") | (subset["metrics_files"] <= 0)).sum()
                ),
                "missing_images_payloads": int(
                    ((subset["state"] != "completed") | (subset["images_files"] <= 0)).sum()
                ),
            }
        )
    completeness = pd.DataFrame(rows)

    missing_payload_rows = []
    for row in status_df.itertuples():
        if row.state != "completed" or row.metrics_files <= 0 or row.images_files <= 0:
            missing_payload_rows.append(
                {
                    "policy_tag": row.policy_tag,
                    "seed": int(row.seed),
                    "state": row.state,
                    "missing_metrics": row.metrics_files <= 0,
                    "missing_images": row.images_files <= 0,
                    "status_path": row.status_path,
                }
            )
    for policy, seed in missing_status:
        missing_payload_rows.append(
            {
                "policy_tag": policy,
                "seed": seed,
                "state": "missing_status",
                "missing_metrics": True,
                "missing_images": True,
                "status_path": "",
            }
        )
    return completeness, pd.DataFrame(missing_payload_rows)


def load_image_events(input_root: Path) -> pd.DataFrame:
    frames = []
    usecols = [
        "t_cmd",
        "target_id",
        "azimuth_deg",
        "elevation_local_deg",
        "range_m",
        "target_shadow_cmd",
        "target_priority_cmd",
        "priority_event_kind",
        "action_id",
        "candidate_slot",
        "sat_shadow_cmd",
        "phase_state",
        "acq_success",
        "target_alt_km",
        "target_regime",
        "target_shadow_acq",
    ]
    for images_path in sorted(input_root.glob("seeds_*/*/seed_*/*/images.csv")):
        try:
            policy_tag = images_path.parents[2].name
            seed_text = images_path.parents[1].name.removeprefix("seed_")
            seed = int(seed_text)
        except (IndexError, ValueError):
            continue
        if policy_tag not in POLICY_ORDER:
            continue
        df = pd.read_csv(images_path, usecols=lambda col: col in usecols)
        if df.empty:
            continue
        df["policy_tag"] = policy_tag
        df["alpha"] = policy_alpha(policy_tag)
        df["seed"] = seed
        df["run_dir"] = str(images_path.parent)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    events = pd.concat(frames, ignore_index=True)
    for col in (
        "t_cmd",
        "target_alt_km",
        "acq_success",
        "sat_shadow_cmd",
        "target_priority_cmd",
        "range_m",
        "elevation_local_deg",
        "target_shadow_cmd",
        "target_shadow_acq",
    ):
        if col in events:
            events[col] = pd.to_numeric(events[col], errors="coerce")
    events["policy_tag"] = pd.Categorical(
        events["policy_tag"], categories=POLICY_ORDER, ordered=True
    )
    events["target_regime"] = pd.Categorical(
        events["target_regime"], categories=REGIME_ORDER, ordered=True
    )
    events["episode_third"] = pd.cut(
        events["t_cmd"],
        bins=[-np.inf, 15_000.0, 30_000.0, np.inf],
        labels=THIRD_LABELS,
    )
    phase = events.get("phase_state", pd.Series(index=events.index, dtype=object)).astype(str)
    fallback = np.where(
        events.get("sat_shadow_cmd", pd.Series(np.nan, index=events.index)) < 0.5,
        "umbra",
        "sunlit",
    )
    events["scanner_condition"] = np.where(
        phase.isin(["umbra", "sunlit"]),
        phase,
        fallback,
    )
    return events.sort_values(["policy_tag", "seed", "t_cmd"]).reset_index(drop=True)


def fraction_table(
    events: pd.DataFrame,
    *,
    group_cols: list[str],
    category_col: str,
    categories: list[str],
) -> pd.DataFrame:
    counts = (
        events.groupby(group_cols + [category_col], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby(group_cols, observed=True)["count"].transform("sum")
    counts["fraction"] = counts["count"] / totals.replace(0, np.nan)
    counts[category_col] = pd.Categorical(
        counts[category_col], categories=categories, ordered=True
    )
    return counts.sort_values(group_cols + [category_col]).reset_index(drop=True)


def per_run_means(
    events: pd.DataFrame, group_cols: list[str], metric: str
) -> pd.DataFrame:
    run = (
        events.groupby(["policy_tag", "alpha", "seed"] + group_cols, observed=True)[metric]
        .mean()
        .rename(metric)
        .reset_index()
    )
    return (
        run.groupby(["policy_tag", "alpha"] + group_cols, observed=True)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def plot_stacked_regime_bars(
    table: pd.DataFrame,
    out_dir: Path,
    *,
    stem: str,
    title: str,
    axes: list[plt.Axes] | None = None,
    facet_col: str | None = None,
    facet_values: list[str] | None = None,
) -> None:
    own_fig = axes is None
    if own_fig:
        fig, axes_arr = plt.subplots(figsize=(8.4, 5.25))
        axes = [axes_arr]
        facet_values = [""]
    else:
        fig = axes[0].figure
        facet_values = facet_values or [""]

    for ax, facet in zip(axes, facet_values or [""], strict=True):
        data = table.copy()
        if facet_col is not None:
            data = data[data[facet_col].astype(str) == str(facet)]
            ax.set_title(str(facet))
        x = np.array([policy_alpha(policy) for policy in POLICY_ORDER], dtype=float)
        bottom = np.zeros(len(POLICY_ORDER))
        for regime in REGIME_ORDER:
            values = []
            for policy in POLICY_ORDER:
                row = data[
                    (data["policy_tag"].astype(str) == policy)
                    & (data["target_regime"].astype(str) == regime)
                ]
                values.append(float(row["fraction"].sum()) if not row.empty else 0.0)
            values_arr = np.array(values)
            ax.bar(
                x,
                values_arr,
                bottom=bottom,
                width=0.075,
                color=REGIME_COLORS[regime],
                edgecolor="0.15",
                linewidth=0.35,
                label=regime,
                alpha=0.92,
            )
            bottom += values_arr
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylabel("Selected target fraction")
        finish_axis(ax, r"Downlink reward weight $\alpha$" if own_fig else None)

    if own_fig:
        axes[0].set_title(title)
        axes[0].legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.255),
            ncol=3,
            framealpha=0.92,
        )
        fig.subplots_adjust(left=0.12, right=0.985, top=0.9, bottom=0.245)
        save_figure(fig, out_dir, stem)


def plot_regime_selection_fraction(events: pd.DataFrame, out_dir: Path) -> None:
    table = fraction_table(
        events,
        group_cols=["policy_tag", "alpha"],
        category_col="target_regime",
        categories=REGIME_ORDER,
    )
    table.to_csv(out_dir / "selected_regime_fraction_by_policy.csv", index=False)
    plot_stacked_regime_bars(
        table,
        out_dir,
        stem="mixed_selected_regime_fraction_by_alpha",
        title="Selected target regime mix",
    )


def plot_regime_success_rate(events: pd.DataFrame, out_dir: Path) -> None:
    stats = per_run_means(events.dropna(subset=["target_regime"]), ["target_regime"], "acq_success")
    stats.to_csv(out_dir / "regime_acq_success_by_policy.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.3, 4.9), constrained_layout=True)
    for regime in REGIME_ORDER:
        data = stats[stats["target_regime"].astype(str) == regime].copy()
        data = data.sort_values("alpha")
        if data.empty:
            continue
        x = data["alpha"].to_numpy(dtype=float)
        y = data["mean"].to_numpy(dtype=float)
        yerr = data["std"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, color=REGIME_COLORS[regime], linewidth=1.5, alpha=0.72)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="0.25", capsize=3, alpha=0.45)
        ax.scatter(
            x,
            y,
            color=REGIME_COLORS[regime],
            edgecolor="k",
            linewidth=0.7,
            s=78,
            label=regime,
            zorder=3,
        )
    ax.set_ylabel("Acquisition success fraction")
    ax.set_ylim(0.0, 1.03)
    ax.legend(loc="lower center", ncol=3, framealpha=0.92)
    finish_axis(ax, r"Downlink reward weight $\alpha$")
    save_figure(fig, out_dir, "mixed_regime_acq_success_by_alpha")


def plot_time_bin_regime_fraction(events: pd.DataFrame, out_dir: Path) -> None:
    table = fraction_table(
        events.dropna(subset=["episode_third"]),
        group_cols=["policy_tag", "alpha", "episode_third"],
        category_col="target_regime",
        categories=REGIME_ORDER,
    )
    table.to_csv(out_dir / "selected_regime_fraction_by_policy_and_episode_third.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.45), sharey=True)
    plot_stacked_regime_bars(
        table,
        out_dir,
        stem="_unused",
        title="",
        axes=list(axes),
        facet_col="episode_third",
        facet_values=THIRD_LABELS,
    )
    for ax in axes[1:]:
        ax.set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.supxlabel(r"Downlink reward weight $\alpha$", y=0.075)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        framealpha=0.92,
    )
    fig.suptitle("Selected target regime by episode third", y=0.965)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.24, wspace=0.04)
    save_figure(fig, out_dir, "mixed_selected_regime_fraction_by_episode_third")


def plot_time_bin_mean_altitude(events: pd.DataFrame, out_dir: Path) -> None:
    stats = per_run_means(
        events.dropna(subset=["episode_third", "target_alt_km"]),
        ["episode_third"],
        "target_alt_km",
    )
    stats.to_csv(out_dir / "mean_target_altitude_by_policy_and_episode_third.csv", index=False)
    colors = dict(zip(THIRD_LABELS, ["#1B9E77", "#7570B3", "#E7298A"], strict=True))
    fig, ax = plt.subplots(figsize=(8.3, 4.9), constrained_layout=True)
    for label in THIRD_LABELS:
        data = stats[stats["episode_third"].astype(str) == label].sort_values("alpha")
        if data.empty:
            continue
        x = data["alpha"].to_numpy(dtype=float)
        y = data["mean"].to_numpy(dtype=float)
        yerr = data["std"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, color=colors[label], linewidth=1.5, alpha=0.72)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="0.25", capsize=3, alpha=0.42)
        ax.scatter(
            x,
            y,
            color=colors[label],
            edgecolor="k",
            linewidth=0.7,
            s=78,
            label=label,
            zorder=3,
        )
    ax.set_ylabel("Mean selected target altitude [km]")
    ax.legend(loc="upper center", ncol=3, framealpha=0.92)
    finish_axis(ax, r"Downlink reward weight $\alpha$")
    save_figure(fig, out_dir, "mixed_mean_target_altitude_by_episode_third")


def plot_scanner_condition_regime_fraction(events: pd.DataFrame, out_dir: Path) -> None:
    condition_events = events[events["scanner_condition"].isin(["umbra", "sunlit"])].copy()
    table = fraction_table(
        condition_events,
        group_cols=["policy_tag", "alpha", "scanner_condition"],
        category_col="target_regime",
        categories=REGIME_ORDER,
    )
    table.to_csv(out_dir / "selected_regime_fraction_by_policy_and_scanner_condition.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.55), sharey=True)
    plot_stacked_regime_bars(
        table,
        out_dir,
        stem="_unused",
        title="",
        axes=list(axes),
        facet_col="scanner_condition",
        facet_values=["umbra", "sunlit"],
    )
    axes[0].set_title("Scanner in umbra")
    axes[1].set_title("Scanner sunlit")
    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.supxlabel(r"Downlink reward weight $\alpha$", y=0.075)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        framealpha=0.92,
    )
    fig.suptitle("Selected target regime by scanner lighting", y=0.965)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.84, bottom=0.24, wspace=0.04)
    save_figure(fig, out_dir, "mixed_selected_regime_fraction_by_scanner_lighting")


def plot_scanner_condition_mean_altitude(events: pd.DataFrame, out_dir: Path) -> None:
    condition_events = events[events["scanner_condition"].isin(["umbra", "sunlit"])].copy()
    stats = per_run_means(
        condition_events.dropna(subset=["target_alt_km"]),
        ["scanner_condition"],
        "target_alt_km",
    )
    stats.to_csv(out_dir / "mean_target_altitude_by_policy_and_scanner_condition.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.3, 4.9), constrained_layout=True)
    labels = [("umbra", "Scanner in umbra"), ("sunlit", "Scanner sunlit")]
    for key, label in labels:
        data = stats[stats["scanner_condition"].astype(str) == key].sort_values("alpha")
        if data.empty:
            continue
        x = data["alpha"].to_numpy(dtype=float)
        y = data["mean"].to_numpy(dtype=float)
        yerr = data["std"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, color=CONDITION_COLORS[key], linewidth=1.5, alpha=0.72)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="0.25", capsize=3, alpha=0.42)
        ax.scatter(
            x,
            y,
            color=CONDITION_COLORS[key],
            edgecolor="k",
            linewidth=0.7,
            s=82,
            label=label,
            zorder=3,
        )
    ax.set_ylabel("Mean selected target altitude [km]")
    ax.legend(loc="upper center", ncol=2, framealpha=0.92)
    finish_axis(ax, r"Downlink reward weight $\alpha$")
    save_figure(fig, out_dir, "mixed_mean_target_altitude_by_scanner_lighting")


def write_manifest(
    out_dir: Path,
    input_root: Path,
    completeness: pd.DataFrame,
    events: pd.DataFrame,
    allow_incomplete: bool,
) -> None:
    pngs = sorted(p.name for p in out_dir.glob("*.png"))
    missing_metrics = int(completeness["missing_metrics_payloads"].sum())
    missing_images = int(completeness["missing_images_payloads"].sum())
    lines = [
        "# AMOS 2026 Mixed-Regime Figure Manifest",
        "",
        f"Input root: `{input_root}`",
        f"Image-command rows used: `{len(events)}`",
        f"Incomplete transfer allowed: `{allow_incomplete}`",
        f"Missing completed metrics payloads: `{missing_metrics}`",
        f"Missing completed images payloads: `{missing_images}`",
        "",
        "Figures are saved as `.png` and `.pdf`.",
        "",
    ]
    for name in pngs:
        lines.append(f"- `{name}`")
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Regime behavior is based on commanded imaging selections in `images.csv`.",
            "- Acquisition success is the per-command acquisition flag, not confirmed downlink delivery.",
            "- Scanner lighting uses `phase_state` when available, with `sat_shadow_cmd < 0.5` as a fallback classifier.",
            "- Episode thirds are fixed windows: 0-15 ks, 15-30 ks, and 30-45 ks.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    setup_style()
    input_root = args.input_root.expanduser().resolve()
    out_dir = (args.output_dir or input_root / "analysis_mixed_regime_plots").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_seeds = parse_expected_seeds(args.expected_seeds)

    status_df = load_status_table(input_root)
    completeness, missing_payloads = completeness_tables(status_df, expected_seeds)
    completeness.to_csv(out_dir / "mixed_transfer_completeness_by_policy.csv", index=False)
    missing_payloads.to_csv(out_dir / "mixed_missing_payloads.csv", index=False)

    missing_metrics = int(completeness["missing_metrics_payloads"].sum())
    missing_images = int(completeness["missing_images_payloads"].sum())
    if (missing_metrics or missing_images) and not args.allow_incomplete:
        print(completeness.to_string(index=False))
        print()
        print(
            "Refusing to generate final mixed-regime plots because completed run payloads "
            "are missing locally. Re-run with --allow-incomplete only for a partial smoke test."
        )
        print(f"Missing completed metrics payloads: {missing_metrics}")
        print(f"Missing completed images payloads: {missing_images}")
        print(f"Completeness tables written to: {out_dir}")
        return 2

    events = load_image_events(input_root)
    if events.empty:
        print(f"No image-event CSV rows found below {input_root}")
        return 1
    events.to_csv(out_dir / "mixed_image_selection_events.csv", index=False)

    plot_regime_selection_fraction(events, out_dir)
    plot_regime_success_rate(events, out_dir)
    plot_time_bin_regime_fraction(events, out_dir)
    plot_time_bin_mean_altitude(events, out_dir)
    plot_scanner_condition_regime_fraction(events, out_dir)
    plot_scanner_condition_mean_altitude(events, out_dir)
    write_manifest(out_dir, input_root, completeness, events, args.allow_incomplete)

    print(completeness.to_string(index=False))
    print()
    print(f"Mixed-regime plots written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
