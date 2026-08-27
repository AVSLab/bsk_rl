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


def _iteration_number(path: Path) -> int:
    try:
        return int(path.name.removeprefix("iteration_"))
    except ValueError as error:
        raise ValueError(f"invalid checkpoint directory name: {path.name}") from error


def checkpoint_directories(run_dir: Path, *, iteration_limit: int = 5) -> list[Path]:
    """Return the configured tail of iteration checkpoints plus ``final``."""

    root = run_dir / "checkpoints"
    checkpoints = sorted(
        (path for path in root.glob("iteration_*") if path.is_dir()),
        key=_iteration_number,
    )[-iteration_limit:]
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


def slurm_array_expression(task_ids: list[int]) -> str:
    """Compact sorted task IDs into Slurm's range syntax."""

    if not task_ids:
        return ""
    values = sorted(set(task_ids))
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def write_task_shards(task_ids: list[int], directory: Path, *, shard_size: int) -> list[Path]:
    """Write local-array-index to validation-task-ID maps in bounded shards."""

    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for shard_index, offset in enumerate(range(0, len(task_ids), shard_size)):
        values = task_ids[offset : offset + shard_size]
        path = directory / f"shard_{shard_index:03d}.txt"
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text("".join(f"{value}\n" for value in values))
        temporary.replace(path)
        paths.append(path)
    return paths
