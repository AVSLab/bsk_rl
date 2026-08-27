"""Restartable, per-episode validation campaign helpers.

The original validation launcher evaluated every retained checkpoint serially in
one Slurm allocation.  This module instead describes each held-out episode as
an immutable task so that a scheduler time limit cannot discard prior work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ARCHITECTURES = ("mlp", "attention")
CANDIDATE_COUNTS = (5, 10, 20)


def checkpoint_directories(run_dir: Path) -> list[Path]:
    """Return retained checkpoint directories in stable selection order."""

    root = run_dir / "checkpoints"
    checkpoints = sorted(path for path in root.glob("iteration_*") if path.is_dir())
    final = root / "final"
    if final.is_dir():
        checkpoints.append(final)
    if not checkpoints:
        raise FileNotFoundError(f"no retained checkpoints under {root}")
    return checkpoints


def task_output_relative(task: dict[str, Any]) -> Path:
    return Path("validation") / "raw" / task["output_name"]


def build_tasks(root: Path, *, catalog_sizes: tuple[int, ...], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    """Build a deterministic task list from the retained training checkpoints."""

    tasks: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        for candidate_count in CANDIDATE_COUNTS:
            run_dir = root / "training" / f"{architecture}_k{candidate_count}_seed10001"
            for checkpoint in checkpoint_directories(run_dir):
                for catalog_size in catalog_sizes:
                    for seed in seeds:
                        task_id = len(tasks)
                        stem = (
                            f"{architecture}_k{candidate_count}_{checkpoint.name}"
                            f"_n{catalog_size}_seed{seed}"
                        )
                        tasks.append(
                            {
                                "task_id": task_id,
                                "architecture": architecture,
                                "candidate_count": candidate_count,
                                "checkpoint": str(checkpoint.resolve()),
                                "checkpoint_name": checkpoint.name,
                                "run_dir": str(run_dir.resolve()),
                                "catalog_size": catalog_size,
                                "seed": seed,
                                "output_name": f"{stem}.csv",
                            }
                        )
    return tasks


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def completed_task(task: dict[str, Any], root: Path) -> bool:
    output = root / task_output_relative(task)
    return output.is_file() and output.with_suffix(".metadata.json").is_file()
