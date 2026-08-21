#!/usr/bin/env python3
"""Audit training progress without treating PPO iteration as physical time."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


@dataclass
class RunDiagnostic:
    run: str
    iterations: int
    environment_steps: int
    wall_clock_h: float
    median_steps_per_iteration: float
    median_samples_per_second: float
    first_episode_return: float | None
    last_episode_return: float | None
    return_slope_per_10000_steps: float | None
    last_episode_target_count: float | None
    last_successful_observation_fraction: float | None
    last_illuminated_observation_fraction: float | None
    metric_keys: str


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _metric(row: dict[str, Any], name: str) -> float | None:
    """Find a scalar despite RLlib version-specific metric prefixes."""

    candidates = [
        (key, _finite(value))
        for key, value in row.items()
        if key == name or key.endswith("/" + name) or key.endswith("_" + name)
    ]
    candidates = [(key, value) for key, value in candidates if value is not None]
    if not candidates:
        return None
    # Prefer the least-prefixed spelling for reproducibility across Ray versions.
    return sorted(candidates, key=lambda item: (len(item[0]), item[0]))[0][1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = path.read_text().splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            # A live trainer may be in the middle of appending its final line.
            if line_number == len(lines):
                continue
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def diagnose_rows(run: str, rows: list[dict[str, Any]]) -> RunDiagnostic:
    if not rows:
        raise ValueError(f"{run} has no training metric rows")
    steps = np.asarray([_metric(row, "environment_steps") or 0.0 for row in rows])
    returns = np.asarray(
        [
            np.nan
            if _metric(row, "episode_return_mean") is None
            else _metric(row, "episode_return_mean")
            for row in rows
        ],
        dtype=float,
    )
    valid = np.isfinite(steps) & np.isfinite(returns)
    slope = None
    if valid.sum() >= 3 and np.ptp(steps[valid]) > 0.0:
        slope = float(np.polyfit(steps[valid], returns[valid], 1)[0] * 10_000.0)
    finite_returns = returns[np.isfinite(returns)]
    deltas = np.diff(steps)
    positive_deltas = deltas[deltas > 0.0]
    throughput = [
        value
        for row in rows
        if (value := _metric(row, "samples_per_second")) is not None
    ]
    keys = sorted({key for row in rows for key in row})
    last = rows[-1]
    return RunDiagnostic(
        run=run,
        iterations=int(_metric(last, "training_iteration") or len(rows)),
        environment_steps=int(steps[-1]),
        wall_clock_h=float(_metric(last, "wall_clock_h") or 0.0),
        median_steps_per_iteration=(
            float(median(positive_deltas)) if positive_deltas.size else 0.0
        ),
        median_samples_per_second=float(median(throughput)) if throughput else 0.0,
        first_episode_return=(
            float(finite_returns[0]) if finite_returns.size else None
        ),
        last_episode_return=(
            float(finite_returns[-1]) if finite_returns.size else None
        ),
        return_slope_per_10000_steps=slope,
        last_episode_target_count=_metric(last, "episode_target_count"),
        last_successful_observation_fraction=_metric(
            last, "successful_observation_fraction"
        ),
        last_illuminated_observation_fraction=_metric(
            last, "illuminated_observation_fraction"
        ),
        metric_keys=";".join(keys),
    )


def diagnose_root(input_root: Path) -> list[RunDiagnostic]:
    paths = sorted(input_root.glob("training/*/training_metrics.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no training_metrics.jsonl files under {input_root}")
    return [
        diagnose_rows(path.parent.name, load_jsonl(path))
        for path in paths
        if path.stat().st_size > 0
    ]


def write_outputs(diagnostics: list[RunDiagnostic], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_diagnostic_summary.csv"
    rows = [asdict(item) for item in diagnostics]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Training diagnostic",
        "",
        "This audit uses environment transitions and wall time. PPO iteration is not a "
        "comparable unit when batch sizes differ.",
        "",
        "| Run | Iter. | Env. steps | Wall h | Steps/iter. | Samples/s | "
        "Return first→last | Return slope / 10k steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in diagnostics:
        first = (
            "NA"
            if item.first_episode_return is None
            else f"{item.first_episode_return:.3f}"
        )
        last = (
            "NA"
            if item.last_episode_return is None
            else f"{item.last_episode_return:.3f}"
        )
        slope = (
            "NA"
            if item.return_slope_per_10000_steps is None
            else f"{item.return_slope_per_10000_steps:.3f}"
        )
        report.append(
            f"| {item.run} | {item.iterations} | {item.environment_steps} | "
            f"{item.wall_clock_h:.2f} | {item.median_steps_per_iteration:.0f} | "
            f"{item.median_samples_per_second:.3f} | {first}→{last} | {slope} |"
        )
    report.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Raw return is catalog-size-confounded in variable-N training. Prefer "
            "successful and illuminated observation fractions on held-out fixed-N episodes.",
            "- A raw-return slope is descriptive, not evidence of policy superiority.",
            "- Compare W&B curves against `prospectus_rfi/environment_steps` or "
            "`prospectus_rfi/wall_clock_h`, not generic `Step`.",
            "- The final claim must come from paired Monte Carlo evaluation of saved checkpoints.",
            "",
        ]
    )
    (output_dir / "TRAINING_DIAGNOSTIC.md").write_text("\n".join(report))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_root / "analysis" / "training_diagnostic"
    diagnostics = diagnose_root(args.input_root)
    write_outputs(diagnostics, output_dir)
    print(f"runs={len(diagnostics)}")
    print(f"csv={output_dir / 'training_diagnostic_summary.csv'}")
    print(f"report={output_dir / 'TRAINING_DIAGNOSTIC.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
