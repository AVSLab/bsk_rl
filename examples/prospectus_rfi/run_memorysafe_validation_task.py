#!/usr/bin/env python3
"""Run exactly one held-out checkpoint-validation episode and commit it atomically."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd

from examples.prospectus_rfi.config import git_metadata, load_study_config
from examples.prospectus_rfi.evaluate import load_policy, run_episode
from examples.prospectus_rfi.validation_campaign import read_manifest, task_output_relative


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
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    root = Path(manifest["root"])
    tasks = manifest["tasks"]
    if not 0 <= args.task_id < len(tasks):
        raise SystemExit(f"task id out of range: {args.task_id}")
    task = tasks[args.task_id]
    output = root / task_output_relative(task)
    metadata = output.with_suffix(".metadata.json")
    if output.is_file() and metadata.is_file():
        print(f"already complete: task={args.task_id} output={output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    architecture_config = (
        "mlp_selected.yaml" if task["architecture"] == "mlp" else "attention_selected.yaml"
    )
    study = load_study_config(
        Path(__file__).parent / "configs" / architecture_config,
        Path(manifest["base_config"]),
    )
    study = replace(
        study,
        environment=replace(study.environment, candidate_count=int(task["candidate_count"])),
    )
    checkpoint = Path(task["checkpoint"])
    policy, checkpoint_metadata = load_policy(checkpoint)
    row = run_episode(
        study,
        method=task["architecture"],
        seed=int(task["seed"]),
        catalog_size=int(task["catalog_size"]),
        learned_policy=policy,
        shield=True,
    )
    row["checkpoint"] = task["checkpoint_name"]
    row["validation_task_id"] = args.task_id
    atomic_csv(pd.DataFrame([row]), output)
    atomic_json(
        {
            "task": task,
            "checkpoint": checkpoint_metadata,
            "manifest": str(args.manifest.resolve()),
            "git": git_metadata(Path.cwd()),
            "study_config": study.to_dict(),
        },
        metadata,
    )
    print(f"PASS task={args.task_id} output={output}")


if __name__ == "__main__":
    main()
