#!/usr/bin/env python3
"""Strictly validate a completed Breckenridge Monte Carlo campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


REQUIRED_METRICS = {
    "cumulative_reward": ("data", "cumulativeRewardSS1"),
    "illuminated_images": ("data", "illuminated_images"),
    "useful_images_downlinked": ("data", "useful_images_downlinked"),
    "total_images_downlinked": ("data", "total_images_downlinked"),
    "episode_end_time_sec": ("data", "episode_end_time_sec"),
    "target_imaging_count": ("summary", "target_imaging_count"),
    "downlink_action_count": ("summary", "downlink_action_count"),
    "charge_action_count": ("summary", "charge_action_count"),
    "desat_action_count": ("summary", "desat_action_count"),
}

ACTION_COUNT_METRICS = {
    "target_imaging_count",
    "downlink_action_count",
    "charge_action_count",
    "desat_action_count",
}

PAPER_METRIC_MAP = {
    "cumulative_reward": "total_reward",
    "illuminated_images": "illuminated_images",
    "useful_images_downlinked": "useful_downlinks_est",
    "target_imaging_count": "target_imaging_count",
    "downlink_action_count": "downlink_action_count",
}


def paper_comparable_metric(actual: dict[str, float], metric: str) -> float | None:
    if metric != "useful_images_downlinked":
        return actual.get(metric)
    reward = actual.get("cumulative_reward")
    images = actual.get("illuminated_images")
    if reward is None or images is None:
        return None
    # The paper analysis inferred useful downlinks from the alpha=0.1 reward.
    return (reward - 0.9 * images) / 0.1


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def finite_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def integer(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close_enough(actual, expected, tolerance: float = 1e-9) -> bool:
    actual_number = finite_number(actual)
    expected_number = finite_number(expected)
    return (
        actual_number is not None
        and expected_number is not None
        and math.isclose(actual_number, expected_number, abs_tol=tolerance, rel_tol=0.0)
    )


def metric_number(payload: dict, output_name: str, section: str, source_name: str):
    raw_value = payload.get(section, {}).get(source_name)
    # Older evaluator output encoded a legitimate action count of zero as null.
    if raw_value is None and output_name in ACTION_COUNT_METRICS:
        return 0.0
    return finite_number(raw_value)


def paper_reference(repo_root: Path) -> dict[int, dict[str, str]]:
    path = (
        repo_root
        / "policies"
        / "breckenridge2026_leo_trained_10d90i"
        / "reference"
        / "paper_mixed_per_seed.csv"
    )
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row.get("env", "").upper() == "MIXED"
        and close_enough(row.get("alpha"), 0.1)
        and "oct14" in row.get("policy_name", "").lower()
    ]
    return {int(float(row["seed"])): row for row in selected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--paper-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    errors: list[str] = []
    warnings: list[str] = []
    rows: dict[tuple[str, int], dict[str, float]] = {}

    manifest_paths = sorted(input_root.glob("*manifest.json"))
    if len(manifest_paths) != 1:
        raise SystemExit(
            f"Expected exactly one manifest in {input_root}; found {len(manifest_paths)}"
        )
    manifest_path = manifest_paths[0]
    manifest = load_json(manifest_path)
    expected_seeds = {int(seed) for seed in manifest.get("seeds", [])}
    expected_cells = manifest.get("cells", {})
    policies = manifest.get("policies", {})
    evaluation = manifest.get("evaluation", {})
    manifest_commit = manifest.get("repository", {}).get("commit")

    if expected_seeds != set(range(100)):
        errors.append(f"Manifest seeds are not exactly 0-99: {sorted(expected_seeds)}")
    if not expected_cells:
        errors.append("Manifest contains no cells")
    if not manifest_commit:
        warnings.append("Older manifest does not record the repository commit")

    for policy_key, policy in policies.items():
        checkpoint = Path(policy.get("checkpoint", ""))
        state_path = (
            checkpoint
            / "learner_group"
            / "learner"
            / "rl_module"
            / "inspector"
            / "module_state.pt"
        )
        if not state_path.is_file():
            errors.append(f"{policy_key}: checkpoint weights missing: {state_path}")
            continue
        expected_hash = policy.get("module_state_sha256")
        if expected_hash and sha256_file(state_path) != expected_hash:
            errors.append(f"{policy_key}: checkpoint SHA-256 does not match manifest")
        elif not expected_hash:
            warnings.append(f"{policy_key}: older manifest has no checkpoint SHA-256")

    for cell, cell_config in expected_cells.items():
        policy = policies[cell_config["policy"]]
        expected_env = cell_config["evaluation_environment"]
        for seed in sorted(expected_seeds):
            seed_dir = input_root / cell / f"seed_{seed:03d}"
            status_path = seed_dir / "mc_status.json"
            if not status_path.is_file():
                errors.append(f"{cell} seed {seed}: missing mc_status.json")
                continue
            try:
                status = load_json(status_path)
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"{cell} seed {seed}: unreadable status: {error}")
                continue
            if status.get("state") != "completed":
                errors.append(
                    f"{cell} seed {seed}: state is {status.get('state')!r}, not completed"
                )
                continue
            status_manifest = status.get("manifest")
            if (
                not isinstance(status_manifest, str)
                or Path(status_manifest).resolve() != manifest_path
            ):
                errors.append(f"{cell} seed {seed}: status manifest mismatch")
            if status.get("cell") != cell or integer(status.get("seed")) != seed:
                errors.append(f"{cell} seed {seed}: status identity mismatch")
            if manifest_commit and status.get("git_commit") != manifest_commit:
                errors.append(f"{cell} seed {seed}: Git commit does not match manifest")
            if status.get("metrics_file_count") != 1:
                errors.append(f"{cell} seed {seed}: metrics_file_count is not 1")

            metrics_value = status.get("metrics_path")
            if not isinstance(metrics_value, str) or not metrics_value:
                errors.append(f"{cell} seed {seed}: metrics path is missing")
                continue
            metrics_path = Path(metrics_value)
            if not metrics_path.is_file():
                errors.append(f"{cell} seed {seed}: metrics file missing: {metrics_path}")
                continue
            try:
                payload = load_json(metrics_path)
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"{cell} seed {seed}: unreadable metrics: {error}")
                continue

            meta = payload.get("meta", {})
            if integer(meta.get("seed")) != seed:
                errors.append(f"{cell} seed {seed}: metrics seed mismatch")
            if meta.get("target_env") != expected_env:
                errors.append(f"{cell} seed {seed}: target environment mismatch")
            if meta.get("policy_name") != policy.get("name"):
                errors.append(f"{cell} seed {seed}: policy name mismatch")
            meta_policy_path = meta.get("policy_path")
            expected_policy_path = Path(policy.get("checkpoint", "")).resolve()
            if (
                not isinstance(meta_policy_path, str)
                or Path(meta_policy_path).resolve() != expected_policy_path
            ):
                errors.append(f"{cell} seed {seed}: policy checkpoint path mismatch")
            if integer(meta.get("n_targets")) != integer(evaluation.get("n_targets")):
                errors.append(f"{cell} seed {seed}: target count mismatch")
            if integer(meta.get("n_targets_ahead")) != integer(
                evaluation.get("n_targets_ahead")
            ):
                errors.append(f"{cell} seed {seed}: candidate-target count mismatch")
            if not close_enough(
                meta.get("total_time_sec"), evaluation.get("total_time_sec")
            ):
                errors.append(f"{cell} seed {seed}: episode horizon mismatch")
            duration_keys = {
                "image": "image_duration_sec",
                "charge": "charge_duration_sec",
                "downlink": "downlink_duration_sec",
                "desat": "desat_duration_sec",
            }
            for action, meta_key in duration_keys.items():
                if not close_enough(
                    meta.get(meta_key),
                    evaluation.get("action_durations_sec", {}).get(action),
                ):
                    errors.append(f"{cell} seed {seed}: {action} duration mismatch")
            expected_mix = evaluation.get("mixed_weights") if expected_env == "mixed" else None
            if meta.get("mix_weights") != expected_mix:
                errors.append(f"{cell} seed {seed}: target mix mismatch")
            if meta.get("shield_enabled") is not True:
                errors.append(f"{cell} seed {seed}: safety shield was not enabled")

            metric_row: dict[str, float] = {}
            for output_name, (section, source_name) in REQUIRED_METRICS.items():
                value = metric_number(payload, output_name, section, source_name)
                if value is None:
                    errors.append(f"{cell} seed {seed}: invalid metric {output_name}")
                else:
                    metric_row[output_name] = value
            if not close_enough(
                metric_row.get("episode_end_time_sec"), evaluation.get("total_time_sec")
            ):
                errors.append(f"{cell} seed {seed}: episode ended at the wrong time")
            if not close_enough(
                payload.get("data", {}).get("mean_target_priority"),
                evaluation.get("uniform_target_priority"),
            ):
                errors.append(f"{cell} seed {seed}: target priority is not uniformly 1")
            rows[(cell, seed)] = metric_row

    actual_statuses = {
        (path.parents[1].name, int(path.parent.name.removeprefix("seed_")))
        for path in input_root.glob("*/seed_*/mc_status.json")
        if path.parent.name.removeprefix("seed_").isdigit()
    }
    expected_statuses = {
        (cell, seed) for cell in expected_cells for seed in expected_seeds
    }
    for cell, seed in sorted(actual_statuses - expected_statuses):
        warnings.append(f"Unexpected status output: {cell} seed {seed}")

    paper_result = None
    paper_cell = "leo_trained__mixed_eval"
    if paper_cell in expected_cells:
        reference = paper_reference(repo_root)
        mismatches = []
        mismatch_count_by_metric = {metric: 0 for metric in PAPER_METRIC_MAP}
        max_absolute_difference = {metric: 0.0 for metric in PAPER_METRIC_MAP}
        if not reference:
            warnings.append(
                "No archived paper per-seed result table is bundled with this public "
                "snapshot; skipping exact paper-output comparison."
            )
        elif set(reference) != set(range(100)):
            warnings.append(
                f"Paper reference contains {len(reference)} alpha-0.1 mixed seeds, not 100"
            )
        else:
            for seed in range(100):
                actual = rows.get((paper_cell, seed), {})
                expected = reference[seed]
                for actual_name, paper_name in PAPER_METRIC_MAP.items():
                    actual_value = paper_comparable_metric(actual, actual_name)
                    expected_value = finite_number(expected.get(paper_name))
                    if actual_value is None or expected_value is None:
                        continue
                    difference = abs(actual_value - expected_value)
                    max_absolute_difference[actual_name] = max(
                        max_absolute_difference[actual_name], difference
                    )
                    if difference > args.paper_tolerance:
                        mismatch_count_by_metric[actual_name] += 1
                        mismatches.append(
                            {
                                "seed": seed,
                                "metric": actual_name,
                                "actual": actual_value,
                                "paper": expected_value,
                                "absolute_difference": difference,
                            }
                        )
            if mismatches:
                warnings.append(
                    "Paper baseline exact-reproduction comparison has "
                    f"{len(mismatches)} metric mismatches"
                )
        paper_result = {
            "reference_seed_count": len(reference),
            "tolerance": args.paper_tolerance,
            "mismatch_count": len(mismatches),
            "exact_reproduction": not mismatches,
            "mismatched_seed_count": len({item["seed"] for item in mismatches}),
            "mismatch_count_by_metric": mismatch_count_by_metric,
            "max_absolute_difference": max_absolute_difference,
            "first_mismatches": mismatches[:20],
        }
        if set(reference) == set(range(100)):
            paper_result["aggregate_comparison"] = {}
            for actual_name, paper_name in PAPER_METRIC_MAP.items():
                actual_values = [
                    rows[(paper_cell, seed)].get(actual_name) for seed in range(100)
                ]
                expected_values = [
                    finite_number(reference[seed].get(paper_name)) for seed in range(100)
                ]
                actual_values = [value for value in actual_values if value is not None]
                expected_values = [value for value in expected_values if value is not None]
                paper_result["aggregate_comparison"][actual_name] = {
                    "actual_mean": statistics.mean(actual_values),
                    "actual_std": statistics.stdev(actual_values),
                    "paper_mean": statistics.mean(expected_values),
                    "paper_std": statistics.stdev(expected_values),
                }

    report = {
        "passed": not errors,
        "input_root": str(input_root),
        "manifest": str(manifest_path),
        "study": manifest.get("study"),
        "expected_cells": sorted(expected_cells),
        "expected_seed_rows": len(expected_cells) * len(expected_seeds),
        "validated_seed_rows": len(rows),
        "paper_baseline_comparison": paper_result,
        "errors": errors,
        "warnings": warnings,
    }
    report_path = input_root / "breckenridge2026_audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"Audit: {'PASS' if report['passed'] else 'FAIL'}")
    print(
        f"Validated seed rows: {report['validated_seed_rows']} / "
        f"{report['expected_seed_rows']}"
    )
    if paper_result is not None:
        print(
            "Paper mixed-baseline mismatches: "
            f"{paper_result['mismatch_count']} at tolerance {args.paper_tolerance:g}"
        )
        print(f"Mismatched seeds: {paper_result['mismatched_seed_count']} / 100")
        print(
            "Mismatches by metric: "
            + ", ".join(
                f"{name}={count}"
                for name, count in paper_result["mismatch_count_by_metric"].items()
            )
        )
    print(f"Errors: {len(errors)}; warnings: {len(warnings)}")
    print(f"Report: {report_path}")
    if errors:
        for error in errors[:30]:
            print(f"ERROR: {error}", file=sys.stderr)
        if len(errors) > 30:
            print(f"... {len(errors) - 30} additional errors in report", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
