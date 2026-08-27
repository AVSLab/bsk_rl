#!/usr/bin/env python3
"""Create or inspect the immutable task manifest for checkpoint validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from examples.prospectus_rfi.config import load_study_config
from examples.prospectus_rfi.validation_campaign import (
    build_tasks,
    completed_task,
    read_manifest,
    slurm_array_expression,
    write_task_shards,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--print-missing", action="store_true")
    parser.add_argument("--print-array-expression", action="store_true")
    parser.add_argument("--print-missing-count", action="store_true")
    parser.add_argument("--write-shards", type=Path)
    parser.add_argument("--shard-size", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest = args.manifest or root / "validation" / "manifest.json"
    architecture_config = Path(__file__).parent / "configs" / "mlp_selected.yaml"
    study = load_study_config(architecture_config, args.base_config.resolve())
    expected_payload = {
        "schema_version": 2,
        "root": str(root),
        "base_config": str(args.base_config.resolve()),
        "catalog_sizes": list(study.validation.catalog_sizes),
        "seeds": list(study.validation.seeds),
        "checkpoint_policy": "last_5_iteration_checkpoints_plus_final",
        "tasks": build_tasks(
            root,
            catalog_sizes=tuple(study.validation.catalog_sizes),
            seeds=tuple(study.validation.seeds),
        ),
    }
    if manifest.exists():
        payload = read_manifest(manifest)
        if Path(payload["root"]).resolve() != root:
            raise SystemExit(f"manifest root differs from requested root: {manifest}")
        if payload.get("schema_version") != 2:
            completed = [task for task in payload["tasks"] if completed_task(task, root)]
            if completed:
                raise SystemExit(
                    "cannot replace an older validation manifest after task outputs exist"
                )
            payload = expected_payload
            write_manifest(manifest, payload)
        elif payload != expected_payload:
            raise SystemExit(
                f"existing validation manifest does not match current checkpoints: {manifest}"
            )
    else:
        payload = expected_payload
        write_manifest(manifest, payload)

    missing = [task["task_id"] for task in payload["tasks"] if not completed_task(task, root)]
    outputs_requested = sum(
        (
            args.print_missing,
            args.print_array_expression,
            args.print_missing_count,
            args.write_shards is not None,
        )
    )
    if outputs_requested > 1:
        raise SystemExit("select at most one --print-* output mode")
    if args.print_missing:
        print(",".join(str(task_id) for task_id in missing))
    elif args.print_array_expression:
        print(slurm_array_expression(missing))
    elif args.print_missing_count:
        print(len(missing))
    elif args.write_shards is not None:
        for path in write_task_shards(missing, args.write_shards, shard_size=args.shard_size):
            line_count = len(path.read_text().splitlines())
            print(f"{path.resolve()}\t{line_count}")
    else:
        print(f"MANIFEST={manifest}")
        print(f"TASKS={len(payload['tasks'])}")
        print(f"MISSING={len(missing)}")


if __name__ == "__main__":
    main()
