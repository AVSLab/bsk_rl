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
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--print-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest = args.manifest or root / "validation" / "manifest.json"
    if manifest.exists():
        payload = read_manifest(manifest)
        if Path(payload["root"]).resolve() != root:
            raise SystemExit(f"manifest root differs from requested root: {manifest}")
    else:
        study = load_study_config(
            Path(__file__).parent / "configs" / "mlp_selected.yaml",
            args.base_config.resolve(),
        )
        tasks = build_tasks(
            root,
            catalog_sizes=tuple(study.validation.catalog_sizes),
            seeds=tuple(study.validation.seeds),
        )
        payload = {
            "schema_version": 1,
            "root": str(root),
            "base_config": str(args.base_config.resolve()),
            "catalog_sizes": list(study.validation.catalog_sizes),
            "seeds": list(study.validation.seeds),
            "tasks": tasks,
        }
        write_manifest(manifest, payload)

    missing = [task["task_id"] for task in payload["tasks"] if not completed_task(task, root)]
    if args.print_missing:
        print(",".join(str(task_id) for task_id in missing))
    else:
        print(f"MANIFEST={manifest}")
        print(f"TASKS={len(payload['tasks'])}")
        print(f"MISSING={len(missing)}")


if __name__ == "__main__":
    main()
