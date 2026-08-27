#!/usr/bin/env python3
"""Validate all per-episode results and select one checkpoint per policy/K run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from examples.prospectus_rfi.validation_campaign import (
    completed_task,
    read_manifest,
    task_output_relative,
)


def atomic_csv(frame: pd.DataFrame, output: Path) -> None:
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)


def atomic_json(payload: dict, output: Path) -> None:
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    root = Path(manifest["root"])
    tasks = manifest["tasks"]
    missing = [task["task_id"] for task in tasks if not completed_task(task, root)]
    if missing:
        raise SystemExit(
            f"incomplete validation: {len(missing)} missing task ids: "
            + ",".join(map(str, missing))
        )

    frames = []
    for task in tasks:
        row = pd.read_csv(root / task_output_relative(task))
        if len(row) != 1:
            raise ValueError(f"task {task['task_id']} must have exactly one result row")
        record = row.iloc[0]
        expected = {
            "method": task["architecture"],
            "candidate_count": task["candidate_count"],
            "catalog_size": task["catalog_size"],
            "scenario_seed": task["seed"],
            "checkpoint": task["checkpoint_name"],
            "validation_task_id": task["task_id"],
        }
        for name, value in expected.items():
            if record[name] != value:
                raise ValueError(
                    f"task {task['task_id']} has {name}={record[name]!r}; expected {value!r}"
                )
        frames.append(row)
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated("validation_task_id").any():
        raise ValueError("duplicate validation task rows")

    validation_root = root / "validation"
    validation_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(frame, validation_root / "validation_combined.csv")
    scores = (
        frame.groupby(["method", "candidate_count", "checkpoint"], as_index=False)
        .agg(
            physical_validation_score=("physical_validation_score", "mean"),
            episodes=("physical_validation_score", "size"),
        )
        .sort_values(
            ["method", "candidate_count", "physical_validation_score", "checkpoint"],
            ascending=[True, True, False, True],
        )
    )
    atomic_csv(scores, validation_root / "checkpoint_scores.csv")

    selections = []
    for (method, candidate_count), group in frame.groupby(["method", "candidate_count"]):
        score_group = scores[
            (scores["method"] == method)
            & (scores["candidate_count"] == candidate_count)
        ]
        winner = score_group.iloc[0]
        checkpoint_name = str(winner["checkpoint"])
        source_task = next(
            task
            for task in tasks
            if task["architecture"] == method
            and task["candidate_count"] == int(candidate_count)
            and task["checkpoint_name"] == checkpoint_name
        )
        run_dir = Path(source_task["run_dir"])
        atomic_csv(
            group.drop(columns=["validation_task_id"]), run_dir / "validation_metrics.csv"
        )
        atomic_csv(score_group, run_dir / "validation_checkpoint_scores.csv")
        link = run_dir / "checkpoints" / "best_validation"
        if link.is_symlink() or link.exists():
            if link.is_dir() and not link.is_symlink():
                raise RuntimeError(f"refusing to replace non-symlink {link}")
            link.unlink()
        os.symlink(checkpoint_name, link, target_is_directory=True)
        payload = {
            "checkpoint": checkpoint_name,
            "mean_physical_validation_score": float(winner["physical_validation_score"]),
            "episodes": int(winner["episodes"]),
            "selection_rule": "maximum mean predeclared physical validation score",
            "catalog_sizes": manifest["catalog_sizes"],
            "seeds": manifest["seeds"],
            "manifest": str(args.manifest.resolve()),
        }
        atomic_json(payload, run_dir / "best_validation.json")
        selections.append({"method": method, "candidate_count": int(candidate_count), **payload})
    atomic_json({"selections": selections}, validation_root / "selection.json")
    print(f"PASS completed_tasks={len(tasks)} selections={len(selections)}")


if __name__ == "__main__":
    main()
