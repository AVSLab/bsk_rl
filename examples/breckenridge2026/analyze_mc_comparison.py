#!/usr/bin/env python3
"""Analyze the Breckenridge four-cell Monte Carlo comparison.

This keeps the paper's aggregation conventions (mean +/- sample standard
deviation and reward-derived useful downlinks) while understanding the newer
training-environment/evaluation-environment directory layout.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Iterable


NEW_METRICS = (
    "total_reward",
    "illuminated_images",
    "useful_downlinks_paper_estimate",
    "useful_downlinks_direct",
    "target_imaging_count",
    "downlink_action_count",
    "charge_action_count",
    "desat_action_count",
    "acq_success_rate",
    "avg_acquisition_time_sec",
    "frac_all_LEO",
    "frac_all_MEO",
    "frac_all_GEO",
    "umbra_smart_fraction",
)

PAPER_COMPARISON_METRICS = {
    "total_reward": "total_reward",
    "illuminated_images": "illuminated_images",
    "useful_downlinks_paper_estimate": "useful_downlinks_est",
    "target_imaging_count": "target_imaging_count",
    "downlink_action_count": "downlink_action_count",
}

HISTORICAL_METRICS = (
    "total_reward",
    "illuminated_images",
    "useful_downlinks_est",
    "target_imaging_count",
    "downlink_action_count",
    "acq_success_rate",
    "avg_acquisition_time_sec",
    "frac_all_LEO",
    "frac_all_MEO",
    "frac_all_GEO",
)


def finite_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def zero_compatible_count(value) -> float:
    """The historical evaluator serialized legitimate zero counts as null."""
    number = finite_number(value)
    return 0.0 if number is None else number


def mean_std(values: Iterable[float | None]) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    return statistics.mean(clean), statistics.stdev(clean) if len(clean) > 1 else None


def alpha_from_name(name: str) -> float:
    match = re.search(r"(\d{1,3})d(\d{1,3})i", name)
    if not match:
        raise ValueError(f"Cannot infer alpha from policy name {name!r}")
    return int(match.group(1)) / 100.0


def useful_downlinks_estimate(reward: float, images: float, alpha: float) -> float | None:
    # This is the exact alpha > 0 formula used by mc_overall_from_json.py.
    if alpha <= 0.0:
        return None
    return (reward - (1.0 - alpha) * images) / alpha


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_from_path(path: Path, input_root: Path) -> str:
    for parent in path.parents:
        if parent == input_root:
            break
        if "__" in parent.name and parent.name.endswith("_eval"):
            return parent.name
    raise ValueError(f"Cannot identify Monte Carlo cell for {path}")


def load_new_row(path: Path, input_root: Path) -> dict:
    payload = json.loads(path.read_text())
    meta = payload.get("meta", {}) or {}
    data = payload.get("data", {}) or {}
    summary = payload.get("summary", {}) or {}
    regime = summary.get("regime_metrics", {}) or {}
    frac_all = regime.get("frac_target_regime_all", {}) or {}

    cell = cell_from_path(path, input_root)
    policy_name = str(meta.get("policy_name", ""))
    alpha = alpha_from_name(policy_name)
    reward = finite_number(data.get("cumulativeRewardSS1"))
    images = finite_number(data.get("illuminated_images"))
    if reward is None or images is None:
        raise ValueError(f"Missing reward or image metric in {path}")

    return {
        "cell": cell,
        "training_environment": cell.split("_trained__", 1)[0],
        "evaluation_environment": cell.split("__", 1)[1].removesuffix("_eval"),
        "seed": int(meta["seed"]),
        "policy_name": policy_name,
        "alpha": alpha,
        "total_reward": reward,
        "illuminated_images": images,
        "useful_downlinks_paper_estimate": useful_downlinks_estimate(
            reward, images, alpha
        ),
        "useful_downlinks_direct": finite_number(
            data.get("useful_images_downlinked")
        ),
        "target_imaging_count": zero_compatible_count(
            summary.get("target_imaging_count")
        ),
        "downlink_action_count": zero_compatible_count(
            summary.get("downlink_action_count")
        ),
        "charge_action_count": zero_compatible_count(
            summary.get("charge_action_count")
        ),
        "desat_action_count": zero_compatible_count(
            summary.get("desat_action_count")
        ),
        "acq_success_rate": finite_number(summary.get("acq_success_rate")),
        "avg_acquisition_time_sec": finite_number(
            summary.get("avg_acquisition_time_sec")
        ),
        "frac_all_LEO": finite_number(frac_all.get("LEO")) or 0.0,
        "frac_all_MEO": finite_number(frac_all.get("MEO")) or 0.0,
        "frac_all_GEO": finite_number(frac_all.get("GEO")) or 0.0,
        "umbra_smart_fraction": finite_number(data.get("umbra_smart_fraction")),
        "metrics_path": str(path),
    }


def load_new_campaigns(input_root: Path) -> list[dict]:
    rows = [
        load_new_row(path, input_root)
        for path in sorted(input_root.rglob("metrics_*.json"))
    ]
    identities = [(row["cell"], row["seed"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate cell/seed metrics found below input root")
    return rows


def summarize_new(rows: list[dict]) -> list[dict]:
    summaries = []
    for cell in sorted({row["cell"] for row in rows}):
        selected = [row for row in rows if row["cell"] == cell]
        seeds = {row["seed"] for row in selected}
        summary = {
            "cell": cell,
            "training_environment": selected[0]["training_environment"],
            "evaluation_environment": selected[0]["evaluation_environment"],
            "alpha": selected[0]["alpha"],
            "N": len(selected),
            "seeds_complete_0_to_99": seeds == set(range(100)),
        }
        for metric in NEW_METRICS:
            values = [row.get(metric) for row in selected]
            summary[f"{metric}_N"] = sum(value is not None for value in values)
            mean, std = mean_std(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        summaries.append(summary)
    return summaries


def paired_rows(
    left: dict[int, dict],
    right: dict[int, dict],
    metrics: Iterable[str],
    comparison: str,
) -> list[dict]:
    seeds = sorted(set(left) & set(right))
    output = []
    for metric in metrics:
        deltas = [
            float(right[seed][metric]) - float(left[seed][metric])
            for seed in seeds
            if left[seed].get(metric) is not None and right[seed].get(metric) is not None
        ]
        mean, std = mean_std(deltas)
        half_width = None
        if mean is not None and std is not None:
            half_width = 1.984 * std / math.sqrt(len(deltas))
        output.append(
            {
                "comparison": comparison,
                "metric": metric,
                "N": len(deltas),
                "mean_delta_right_minus_left": mean,
                "std_delta": std,
                "ci95_low": mean - half_width if half_width is not None else None,
                "ci95_high": mean + half_width if half_width is not None else None,
                "max_absolute_delta": max(map(abs, deltas)) if deltas else None,
                "exact_match_count": sum(abs(value) < 1e-9 for value in deltas),
            }
        )
    return output


def training_effect(rows: list[dict]) -> list[dict]:
    output = []
    by_cell = {
        cell: {row["seed"]: row for row in rows if row["cell"] == cell}
        for cell in {row["cell"] for row in rows}
    }
    for environment in ("leo", "mixed"):
        leo_cell = f"leo_trained__{environment}_eval"
        mixed_cell = f"mixed_trained__{environment}_eval"
        if leo_cell in by_cell and mixed_cell in by_cell:
            output.extend(
                paired_rows(
                    by_cell[leo_cell],
                    by_cell[mixed_cell],
                    PAPER_COMPARISON_METRICS,
                    f"mixed-trained minus LEO-trained in {environment} evaluation",
                )
            )
    return output


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def paper_alpha01_comparison(rows: list[dict], reference_path: Path) -> list[dict]:
    actual = {
        row["seed"]: row
        for row in rows
        if row["cell"] == "leo_trained__mixed_eval"
    }
    reference_rows = [
        row
        for row in read_csv(reference_path)
        if str(row.get("env", "")).upper() == "MIXED"
        and finite_number(row.get("alpha")) == 0.1
    ]
    reference = {
        int(float(row["seed"])): {
            actual_name: finite_number(row.get(reference_name))
            for actual_name, reference_name in PAPER_COMPARISON_METRICS.items()
        }
        for row in reference_rows
    }
    return paired_rows(
        reference,
        actual,
        PAPER_COMPARISON_METRICS,
        "cluster LEO-trained mixed rerun minus paper alpha=0.1 mixed snapshot",
    )


def summarize_historical(rows: list[dict], group_fields: tuple[str, ...]) -> list[dict]:
    groups = sorted({tuple(row[field] for field in group_fields) for row in rows})
    output = []
    for identity in groups:
        selected = [
            row
            for row in rows
            if tuple(row[field] for field in group_fields) == identity
        ]
        summary = dict(zip(group_fields, identity))
        summary["N"] = len(selected)
        for metric in HISTORICAL_METRICS:
            values = [finite_number(row.get(metric)) for row in selected]
            summary[f"{metric}_N"] = sum(value is not None for value in values)
            mean, std = mean_std(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        output.append(summary)
    return output


def formatted_historical_table(summary: list[dict]) -> list[dict]:
    def pm(row: dict, metric: str, decimals: int = 2) -> str:
        mean = row.get(f"{metric}_mean")
        std = row.get(f"{metric}_std")
        if mean is None or std is None:
            return ""
        return f"{float(mean):.{decimals}f} +/- {float(std):.{decimals}f}"

    return [
        {
            "alpha": f"{float(row['alpha']):.1f}",
            "Env": str(row["env"]),
            "N": int(row["N"]),
            "Total reward": pm(row, "total_reward"),
            "Illuminated images": pm(row, "illuminated_images"),
            "Useful downlinks (est)": pm(row, "useful_downlinks_est"),
            "Imaging actions": pm(row, "target_imaging_count"),
            "Downlink actions": pm(row, "downlink_action_count"),
            "Acq success rate": pm(row, "acq_success_rate", 3),
            "Mean dt_acq [s]": pm(row, "avg_acquisition_time_sec"),
        }
        for row in sorted(summary, key=lambda item: float(item["alpha"]))
    ]


def write_markdown_table(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row[field]) for field in fields) + " |"
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n")


def write_latex_table(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])

    def escaped(value) -> str:
        text = str(value)
        for source, replacement in (
            ("\\", r"\textbackslash{}"),
            ("_", r"\_"),
            ("%", r"\%"),
            ("&", r"\&"),
            ("#", r"\#"),
        ):
            text = text.replace(source, replacement)
        return text

    lines = [
        r"\begin{tabular}{c c c c c c c c c c}",
        r"\toprule",
        " & ".join(escaped(field) for field in fields) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        values = [
            escaped(row[field]).replace("+/-", r"$\pm$") for field in fields
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def compare_historical_summaries(
    early: list[dict], latest: list[dict]
) -> list[dict]:
    early_by_alpha = {float(row["alpha"]): row for row in early}
    latest_by_alpha = {float(row["alpha"]): row for row in latest}
    output = []
    for alpha in sorted(set(early_by_alpha) | set(latest_by_alpha)):
        early_row = early_by_alpha.get(alpha)
        latest_row = latest_by_alpha.get(alpha)
        for metric in HISTORICAL_METRICS:
            early_mean = early_row.get(f"{metric}_mean") if early_row else None
            early_std = early_row.get(f"{metric}_std") if early_row else None
            latest_mean = latest_row.get(f"{metric}_mean") if latest_row else None
            latest_std = latest_row.get(f"{metric}_std") if latest_row else None
            output.append(
                {
                    "alpha": alpha,
                    "metric": metric,
                    "jan15_N": early_row.get("N") if early_row else None,
                    "jan16_N": latest_row.get("N") if latest_row else None,
                    "jan15_mean": early_mean,
                    "jan16_mean": latest_mean,
                    "mean_delta_jan16_minus_jan15": (
                        float(latest_mean) - float(early_mean)
                        if early_mean is not None and latest_mean is not None
                        else None
                    ),
                    "jan15_std": early_std,
                    "jan16_std": latest_std,
                    "std_delta_jan16_minus_jan15": (
                        float(latest_std) - float(early_std)
                        if early_std is not None and latest_std is not None
                        else None
                    ),
                }
            )
    return output


def compare_formatted_tables(
    generated: list[dict], saved: list[dict]
) -> tuple[int, list[dict]]:
    """Compare tables after normalizing the saved Unicode plus/minus sign."""

    def normalized(value) -> str:
        return str(value).replace("\N{PLUS-MINUS SIGN}", "+/-").replace(" ", "")

    saved_by_key = {
        (normalized(row["alpha"]), normalized(row["Env"])): row for row in saved
    }
    mismatches = []
    comparison_count = 0
    for row in generated:
        key = (normalized(row["alpha"]), normalized(row["Env"]))
        saved_row = saved_by_key.get(key)
        for field, generated_value in row.items():
            comparison_count += 1
            saved_value = saved_row.get(field) if saved_row else None
            if normalized(generated_value) != normalized(saved_value):
                mismatches.append(
                    {
                        "alpha": row["alpha"],
                        "Env": row["Env"],
                        "field": field,
                        "generated": generated_value,
                        "saved": saved_value,
                    }
                )
    return comparison_count, mismatches


def recompute_paper_snapshots(
    repo_root: Path, output_dir: Path, analysis_tag: str
) -> dict:
    source_info = {}
    robustness = repo_root / "examples" / "per_seed_metrics_from_json.csv"
    if robustness.is_file():
        rows = read_csv(robustness)
        write_csv(
            output_dir / "original_paper_robustness_recomputed.csv",
            summarize_historical(rows, ("env",)),
        )
        source_info["robustness"] = {
            "path": str(robustness),
            "sha256": sha256_file(robustness),
            "rows": len(rows),
        }

    early = (
        repo_root
        / "examples"
        / "results"
        / "per_seed_metrics_allPolicies_20260115_201523.csv"
    )
    late = (
        repo_root
        / "examples"
        / "results"
        / "per_seed_metrics_allPolicies_20260116_150922.csv"
    )
    latest_saved = (
        repo_root
        / "examples"
        / "results"
        / "overall_summary_by_alpha_allPolicies_20260116_150922.csv"
    )
    if early.is_file() and late.is_file() and latest_saved.is_file():
        early_rows = read_csv(early)
        late_rows = read_csv(late)
        manuscript_rows = list(early_rows)
        manuscript_rows.extend(
            row
            for row in late_rows
            if finite_number(row.get("alpha")) in (0.8, 0.9)
        )
        unique = {}
        for row in manuscript_rows:
            key = (
                round(float(row["alpha"]), 2),
                str(row["env"]),
                int(float(row["seed"])),
            )
            unique.setdefault(key, row)
        write_csv(
            output_dir / "original_paper_alpha_sweep_recomputed.csv",
            summarize_historical(list(unique.values()), ("alpha", "env")),
        )
        manuscript_summary = summarize_historical(
            list(unique.values()), ("alpha", "env")
        )
        write_csv(
            output_dir / "manuscript_alpha_sweep_recomputed.csv",
            manuscript_summary,
        )

        latest_summary = summarize_historical(late_rows, ("alpha", "env"))
        numeric_path = output_dir / f"gnc_alpha_sweep_{analysis_tag}_numeric.csv"
        table_path = output_dir / f"gnc_alpha_sweep_{analysis_tag}_table.csv"
        markdown_path = output_dir / f"gnc_alpha_sweep_{analysis_tag}_table.md"
        latex_path = output_dir / f"gnc_alpha_sweep_{analysis_tag}_table.tex"
        write_csv(numeric_path, latest_summary)
        formatted = formatted_historical_table(latest_summary)
        write_csv(table_path, formatted)
        write_markdown_table(markdown_path, formatted)
        write_latex_table(latex_path, formatted)
        comparison_count, mismatches = compare_formatted_tables(
            formatted, read_csv(latest_saved)
        )

        early_summary = summarize_historical(early_rows, ("alpha", "env"))
        differences = compare_historical_summaries(early_summary, latest_summary)
        difference_path = output_dir / "alpha_sweep_jan15_vs_jan16.csv"
        write_csv(difference_path, differences)

        source_info["alpha_sweep"] = {
            "early_path": str(early),
            "early_sha256": sha256_file(early),
            "latest_complete_path": str(late),
            "late_sha256": sha256_file(late),
            "latest_saved_summary_path": str(latest_saved),
            "latest_saved_summary_sha256": sha256_file(latest_saved),
            "manuscript_snapshot_rows": len(unique),
            "latest_complete_rows": len(late_rows),
            "latest_complete_all_alphas_have_100_seeds": all(
                int(row["N"]) == 100 for row in latest_summary
            ),
            "latest_displayed_cells_compared": comparison_count,
            "latest_displayed_cells_match_saved_summary": not mismatches,
            "latest_displayed_cell_mismatches": mismatches,
            "latest_outputs": [
                str(numeric_path),
                str(table_path),
                str(markdown_path),
                str(latex_path),
                str(difference_path),
            ],
        }
    return source_info


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default=str(Path.home() / "rllib_results" / "breckenridge2026_mc"),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--analysis-tag",
        default=datetime.now().strftime("%Y%m%d"),
        help="Date or label included in regenerated historical table filenames.",
    )
    parser.add_argument(
        "--paper-alpha01-reference",
        default=str(
            repo_root
            / "policies"
            / "breckenridge2026_leo_trained_10d90i"
            / "reference"
            / "paper_mixed_per_seed.csv"
        ),
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_root / "analysis"
    )
    reference_path = Path(args.paper_alpha01_reference).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_new_campaigns(input_root)
    summaries = summarize_new(rows)
    effects = training_effect(rows)
    paper_comparison = paper_alpha01_comparison(rows, reference_path)

    write_csv(output_dir / "new_mc_per_seed.csv", rows)
    write_csv(output_dir / "new_mc_summary.csv", summaries)
    write_csv(output_dir / "new_mc_training_effect_paired.csv", effects)
    write_csv(
        output_dir / "leo_mixed_rerun_vs_paper_alpha01.csv", paper_comparison
    )
    historical_sources = recompute_paper_snapshots(
        repo_root, output_dir, args.analysis_tag
    )

    expected_cells = {
        "leo_trained__leo_eval",
        "leo_trained__mixed_eval",
        "mixed_trained__leo_eval",
        "mixed_trained__mixed_eval",
    }
    cells = {row["cell"] for row in rows}
    report = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "metric_json_count": len(rows),
        "cells": sorted(cells),
        "all_four_cells_present": cells == expected_cells,
        "all_cells_have_seeds_0_to_99": all(
            summary["seeds_complete_0_to_99"] for summary in summaries
        ),
        "paper_conventions": {
            "dispersion": "sample standard deviation (ddof=1)",
            "useful_downlinks": (
                "(reward - (1-alpha)*illuminated_images) / alpha"
            ),
        },
        "paper_alpha01_reference": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
        },
        "historical_sources": historical_sources,
        "outputs": sorted(path.name for path in output_dir.glob("*.csv")),
    }
    report_path = output_dir / "analysis_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"Loaded metrics: {len(rows)}")
    print(f"Cells: {', '.join(sorted(cells))}")
    complete = (
        report["all_four_cells_present"]
        and report["all_cells_have_seeds_0_to_99"]
    )
    print(f"All four cells complete: {complete}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
