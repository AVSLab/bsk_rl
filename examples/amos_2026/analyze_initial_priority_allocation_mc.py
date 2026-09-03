#!/usr/bin/env python3
"""Analyze the paired AMOS initial-priority allocation campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


CASE_ORDER = ("ground_confirmation", "one_orbit")
CASE_LABELS = {
    "ground_confirmation": "Ground-confirmation cooldown",
    "one_orbit": "One-orbit cooldown",
}
CLASS_ORDER = (
    "Normal: lower third",
    "Normal: middle third",
    "Normal: upper third",
    "HIO",
    "SHIO",
)
CLASS_LABELS = (
    "Normal\nlower third",
    "Normal\nmiddle third",
    "Normal\nupper third",
    "HIO",
    "SHIO",
)
CLASS_COLORS = {
    "Normal: lower third": "#9ecae1",
    "Normal: middle third": "#4292c6",
    "Normal: upper third": "#08519c",
    "HIO": "#8c6bb1",
    "SHIO": "#54278f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=50)
    parser.add_argument("--n-targets", type=int, default=200)
    parser.add_argument("--hio-count", type=int, default=20)
    parser.add_argument("--shio-count", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <input-root>/analysis_initial_priority_allocation.",
    )
    return parser.parse_args()


def parse_bool(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(bool)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def ci95(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return math.nan
    return float(stats.t.ppf(0.975, len(values) - 1) * values.sem())


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
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


def metrics_payload(evaluation_dir: Path) -> dict:
    files = sorted(evaluation_dir.glob("metrics_*.json"))
    if len(files) != 1:
        raise ValueError(
            f"Expected one metrics JSON in {evaluation_dir}, found {len(files)}"
        )
    return json.loads(files[0].read_text())


def assign_normal_tertiles(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["allocation_class"] = frame["response_class"]
    normal = frame[frame["response_class"].eq("CONTROL")].sort_values(
        ["initial_priority", "target_id"], kind="stable"
    )
    if len(normal) < 3:
        raise ValueError("At least three normal targets are required for tertiles")
    tertile_labels = CLASS_ORDER[:3]
    for label, indices in zip(tertile_labels, np.array_split(normal.index, 3)):
        frame.loc[indices, "allocation_class"] = label
    frame.loc[frame["response_class"].eq("HIO"), "allocation_class"] = "HIO"
    frame.loc[frame["response_class"].eq("SHIO"), "allocation_class"] = "SHIO"
    return frame


def load_campaign(
    input_root: Path,
    expected_seed_count: int,
    n_targets: int,
    hio_count: int,
    shio_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_frames: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    expected_seeds = set(range(expected_seed_count))

    for case in CASE_ORDER:
        case_root = input_root / case
        observed_seeds: set[int] = set()
        for status_path in sorted(case_root.glob("seed_*/mc_status.json")):
            status = json.loads(status_path.read_text())
            seed = int(status["seed"])
            observed_seeds.add(seed)
            evaluation_dir_raw = status.get("evaluation_dir")
            evaluation_dir = (
                Path(evaluation_dir_raw) if evaluation_dir_raw else None
            )
            complete = status.get("state") == "completed"
            if not complete or evaluation_dir is None or not evaluation_dir.is_dir():
                audit_rows.append(
                    {
                        "case": case,
                        "seed": seed,
                        "status": status.get("state", "missing"),
                        "evaluation_dir": evaluation_dir_raw,
                        "valid": False,
                    }
                )
                continue

            response_path = evaluation_dir / "priority_response_targets.csv"
            catalog_path = evaluation_dir / "target_catalog.csv"
            delivery_path = evaluation_dir / "verified_deliveries.csv"
            step_path = evaluation_dir / "steps.csv"
            image_path = evaluation_dir / "images.csv"
            required = (
                response_path,
                catalog_path,
                delivery_path,
                step_path,
                image_path,
            )
            missing = [str(path.name) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Missing outputs for {case} seed {seed}: {missing}"
                )

            response = pd.read_csv(response_path)
            catalog = pd.read_csv(catalog_path)[
                ["target_id", "initial_priority", "priority_event_kind"]
            ]
            if (
                len(response) != n_targets
                or response["target_id"].nunique() != n_targets
            ):
                raise ValueError(
                    f"Expected {n_targets} response rows for {case} seed {seed}"
                )
            counts = response["response_class"].value_counts().to_dict()
            expected_counts = {
                "HIO": hio_count,
                "SHIO": shio_count,
                "CONTROL": n_targets - hio_count - shio_count,
            }
            if counts != expected_counts:
                raise ValueError(
                    f"Unexpected class counts for {case} seed {seed}: {counts}"
                )
            event_times = pd.to_numeric(
                response["priority_event_time_sec"], errors="raise"
            )
            if not np.allclose(event_times, 0.0):
                raise ValueError(
                    f"Priority assignment was not at t=0 for {case} seed {seed}"
                )

            payload = metrics_payload(evaluation_dir)
            data = payload.get("data", {})
            applied_time = float(
                data.get("dynamic_priority_event_applied_time_sec", math.nan)
            )
            if not math.isclose(applied_time, 0.0, abs_tol=1e-9):
                raise ValueError(
                    f"Priority event first became active at {applied_time} s for "
                    f"{case} seed {seed}; expected 0 s"
                )

            merged = response.merge(
                catalog,
                on="target_id",
                how="left",
                validate="one_to_one",
                suffixes=("", "_catalog"),
            )
            deliveries = pd.read_csv(delivery_path)
            if deliveries.empty:
                useful_counts = pd.Series(dtype=int)
                useful_values = pd.Series(dtype=float)
            else:
                useful = deliveries[parse_bool(deliveries["useful_delivery"])].copy()
                useful["delivered_priority_value"] = (
                    pd.to_numeric(useful["target_priority"], errors="coerce")
                    * pd.to_numeric(useful["quality_value"], errors="coerce").fillna(1.0)
                )
                useful_counts = useful.groupby("target_id").size()
                useful_values = useful.groupby("target_id")[
                    "delivered_priority_value"
                ].sum()
            merged["useful_delivery_count"] = (
                merged["target_id"].map(useful_counts).fillna(0).astype(int)
            )
            merged["delivered_priority_value"] = (
                merged["target_id"].map(useful_values).fillna(0.0)
            )
            merged["case"] = case
            merged["seed"] = seed
            merged["evaluation_dir"] = str(evaluation_dir)
            merged = assign_normal_tertiles(merged)
            target_frames.append(merged)
            audit_rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "status": "completed",
                    "evaluation_dir": str(evaluation_dir),
                    "valid": True,
                    "step_count": len(pd.read_csv(step_path, usecols=["t_cmd"])),
                    "image_command_count": len(
                        pd.read_csv(image_path, usecols=["t_cmd"])
                    ),
                }
            )

        if observed_seeds != expected_seeds:
            raise ValueError(
                f"{case} seed mismatch: missing={sorted(expected_seeds-observed_seeds)}, "
                f"extra={sorted(observed_seeds-expected_seeds)}"
            )

    targets = pd.concat(target_frames, ignore_index=True)
    audit = pd.DataFrame(audit_rows).sort_values(["case", "seed"])
    if not audit["valid"].all():
        bad = audit[~audit["valid"]][["case", "seed", "status"]]
        raise ValueError(f"Incomplete runs remain:\n{bad.to_string(index=False)}")
    return targets, audit


def build_seed_class_summary(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, seed, allocation_class), group in targets.groupby(
        ["case", "seed", "allocation_class"], sort=False
    ):
        successful = pd.to_numeric(
            group["successful_image_count_after_event"], errors="raise"
        )
        deliveries = pd.to_numeric(group["useful_delivery_count"], errors="raise")
        rows.append(
            {
                "case": case,
                "seed": int(seed),
                "allocation_class": allocation_class,
                "target_count": len(group),
                "mean_successful_images_per_target": successful.mean(),
                "mean_useful_deliveries_per_target": deliveries.mean(),
                "successful_image_coverage_pct": 100.0 * successful.gt(0).mean(),
                "repeated_image_coverage_pct": 100.0 * successful.ge(2).mean(),
                "useful_delivery_coverage_pct": 100.0 * deliveries.gt(0).mean(),
                "mean_delivered_priority_value_per_target": pd.to_numeric(
                    group["delivered_priority_value"], errors="raise"
                ).mean(),
                "mean_candidate_presentations_per_target": pd.to_numeric(
                    group["candidate_presentation_count"], errors="raise"
                ).mean(),
                "mean_eligible_visible_accesses_per_target": pd.to_numeric(
                    group["eligible_visible_access_count"], errors="raise"
                ).mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary["allocation_class"] = pd.Categorical(
        summary["allocation_class"], categories=CLASS_ORDER, ordered=True
    )
    return summary.sort_values(["case", "seed", "allocation_class"])


def paired_bootstrap_ci(differences: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(differences, errors="raise").to_numpy(float)
    rng = np.random.default_rng(20260903)
    indices = rng.integers(0, len(values), size=(50_000, len(values)))
    boot = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(boot, [0.025, 0.975]))


def hio_shio_statistics(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "mean_successful_images_per_target",
        "mean_useful_deliveries_per_target",
        "successful_image_coverage_pct",
        "repeated_image_coverage_pct",
        "useful_delivery_coverage_pct",
    )
    rows = []
    for case in CASE_ORDER:
        case_data = summary[summary["case"].eq(case)]
        for metric in metrics:
            wide = case_data.pivot(
                index="seed", columns="allocation_class", values=metric
            )
            differences = wide["SHIO"] - wide["HIO"]
            low, high = paired_bootstrap_ci(differences)
            if np.allclose(differences, 0.0):
                wilcoxon_p = 1.0
            else:
                wilcoxon_p = float(
                    stats.wilcoxon(differences, zero_method="pratt").pvalue
                )
            rows.append(
                {
                    "case": case,
                    "metric": metric,
                    "seed_count": len(wide),
                    "hio_mean": wide["HIO"].mean(),
                    "shio_mean": wide["SHIO"].mean(),
                    "shio_minus_hio_mean": differences.mean(),
                    "paired_bootstrap_ci95_low": low,
                    "paired_bootstrap_ci95_high": high,
                    "paired_t_pvalue": stats.ttest_rel(
                        wide["SHIO"], wide["HIO"]
                    ).pvalue,
                    "wilcoxon_pvalue": wilcoxon_p,
                }
            )
    return pd.DataFrame(rows)


def plot_allocation(summary: pd.DataFrame, output_dir: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)
    rng = np.random.default_rng(20260903)
    x_positions = np.arange(len(CLASS_ORDER))
    for panel_index, (ax, case) in enumerate(zip(axes, CASE_ORDER)):
        case_data = summary[summary["case"].eq(case)]
        for x, allocation_class in zip(x_positions, CLASS_ORDER):
            values = case_data[
                case_data["allocation_class"].eq(allocation_class)
            ]["mean_successful_images_per_target"]
            jitter = rng.uniform(-0.10, 0.10, len(values))
            color = CLASS_COLORS[allocation_class]
            ax.scatter(
                x + jitter,
                values,
                s=12,
                color=color,
                alpha=0.22,
                edgecolors="none",
                zorder=2,
            )
            ax.errorbar(
                x,
                values.mean(),
                yerr=ci95(values),
                fmt="o",
                markersize=7,
                markerfacecolor=color,
                markeredgecolor="0.15",
                markeredgewidth=0.7,
                ecolor="0.15",
                elinewidth=1.1,
                capsize=3,
                zorder=4,
            )
        ax.set_title(f"({chr(97 + panel_index)}) {CASE_LABELS[case]}")
        ax.set_xticks(x_positions, CLASS_LABELS)
        ax.set_xlabel("Initial target-priority class")
        ax.grid(axis="y", color="0.87", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Successful images per target")
    fig.tight_layout(w_pad=1.1)
    save_figure(fig, output_dir, "initial_priority_capture_allocation")


def plot_service_metrics(summary: pd.DataFrame, output_dir: Path) -> None:
    configure_style()
    metrics = (
        ("successful_image_coverage_pct", "Imaged at least once [%]"),
        ("repeated_image_coverage_pct", "Imaged at least twice [%]"),
        ("mean_useful_deliveries_per_target", "Useful deliveries per target"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.8))
    x_positions = np.arange(len(CLASS_ORDER))
    for row, case in enumerate(CASE_ORDER):
        case_data = summary[summary["case"].eq(case)]
        for column, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, column]
            means = []
            errors = []
            for allocation_class in CLASS_ORDER:
                values = case_data[
                    case_data["allocation_class"].eq(allocation_class)
                ][metric]
                means.append(values.mean())
                errors.append(ci95(values))
            ax.bar(
                x_positions,
                means,
                yerr=errors,
                color=[CLASS_COLORS[value] for value in CLASS_ORDER],
                edgecolor="0.2",
                linewidth=0.6,
                capsize=2.5,
                zorder=2,
            )
            ax.set_xticks(x_positions, CLASS_LABELS, rotation=18, ha="right")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", color="0.87", linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            if column == 1:
                ax.set_title(CASE_LABELS[case])
    fig.tight_layout(h_pad=1.4, w_pad=1.0)
    save_figure(fig, output_dir, "initial_priority_service_metrics")


def plot_hio_shio_differences(summary: pd.DataFrame, output_dir: Path) -> None:
    configure_style()
    metrics = (
        ("mean_successful_images_per_target", "Successful images per target"),
        ("mean_useful_deliveries_per_target", "Useful deliveries per target"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.4))
    rng = np.random.default_rng(20260903)
    for ax, (metric, ylabel) in zip(axes, metrics):
        differences = []
        for case in CASE_ORDER:
            wide = summary[summary["case"].eq(case)].pivot(
                index="seed", columns="allocation_class", values=metric
            )
            differences.append((wide["SHIO"] - wide["HIO"]).to_numpy())
        box = ax.boxplot(
            differences,
            positions=[0, 1],
            widths=0.46,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 1.4},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(CLASS_COLORS["SHIO"])
            patch.set_alpha(0.72)
        for x, values in enumerate(differences):
            jitter = rng.uniform(-0.12, 0.12, len(values))
            ax.scatter(
                x + jitter,
                values,
                s=12,
                color="#2f2f2f",
                alpha=0.35,
                edgecolors="none",
                zorder=3,
            )
            ax.scatter(
                x,
                np.mean(values),
                marker="D",
                s=42,
                color="white",
                edgecolor="0.15",
                linewidth=0.8,
                zorder=4,
            )
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9)
        ax.set_xticks([0, 1], ["Ground\nconfirmation", "One-orbit\ncooldown"])
        ax.set_ylabel(f"SHIO minus HIO\n{ylabel.lower()}")
        ax.grid(axis="y", color="0.87", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1.2)
    save_figure(fig, output_dir, "initial_priority_hio_shio_differences")


def write_summary(
    statistics: pd.DataFrame, summary: pd.DataFrame, output_dir: Path
) -> None:
    lines = [
        "# Initial-priority allocation Monte Carlo summary",
        "",
        "The campaign contains 50 paired seeds for each cooldown case. Each seed "
        "uses 20 HIOs (10% of the catalog) at 5 times the realized initial-priority "
        "maximum, 20 SHIOs (10%) at 10 times the maximum, and 160 non-promoted "
        "targets divided by their within-seed initial-priority ranks.",
        "",
        "## SHIO versus HIO paired comparisons",
        "",
        "| Cooldown | Metric | HIO mean | SHIO mean | SHIO-HIO | 95% bootstrap CI | Paired t p | Wilcoxon p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in statistics.itertuples(index=False):
        lines.append(
            f"| {CASE_LABELS[row.case]} | {row.metric.replace('_', ' ')} | "
            f"{row.hio_mean:.3f} | {row.shio_mean:.3f} | "
            f"{row.shio_minus_hio_mean:.3f} | "
            f"[{row.paired_bootstrap_ci95_low:.3f}, "
            f"{row.paired_bootstrap_ci95_high:.3f}] | "
            f"{row.paired_t_pvalue:.4g} | {row.wilcoxon_pvalue:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Mean allocation by class",
            "",
            "Values below are means across the 50 per-seed class means.",
            "",
            "| Cooldown | Class | Successful images/target | Useful deliveries/target | Imaged >=1 [%] | Imaged >=2 [%] |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    grouped = summary.groupby(["case", "allocation_class"], observed=True).mean(
        numeric_only=True
    )
    for case in CASE_ORDER:
        for allocation_class in CLASS_ORDER:
            row = grouped.loc[(case, allocation_class)]
            lines.append(
                f"| {CASE_LABELS[case]} | {allocation_class} | "
                f"{row.mean_successful_images_per_target:.3f} | "
                f"{row.mean_useful_deliveries_per_target:.3f} | "
                f"{row.successful_image_coverage_pct:.2f} | "
                f"{row.repeated_image_coverage_pct:.2f} |"
            )
    (output_dir / "STATISTICAL_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else input_root / "analysis_initial_priority_allocation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.hio_count + args.shio_count >= args.n_targets:
        raise ValueError("HIO and SHIO groups must leave normal control targets")
    targets, audit = load_campaign(
        input_root,
        args.expected_seeds,
        args.n_targets,
        args.hio_count,
        args.shio_count,
    )
    summary = build_seed_class_summary(targets)
    statistics = hio_shio_statistics(summary)

    targets.to_csv(output_dir / "target_allocation_combined.csv", index=False)
    summary.to_csv(output_dir / "seed_class_summary.csv", index=False)
    statistics.to_csv(output_dir / "paired_hio_shio_statistics.csv", index=False)
    audit.to_csv(output_dir / "campaign_audit.csv", index=False)
    plot_allocation(summary, output_dir)
    plot_service_metrics(summary, output_dir)
    plot_hio_shio_differences(summary, output_dir)
    write_summary(statistics, summary, output_dir)

    print(f"Validated {len(audit)} episodes and {len(targets)} target rows.")
    print(f"Analysis written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
