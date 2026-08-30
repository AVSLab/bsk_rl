#!/usr/bin/env python3
"""Evaluate retained checkpoints on held-out seeds and select the best one."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from examples.prospectus_rfi.config import load_study_config
from examples.prospectus_rfi.evaluate import load_policy, run_episode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("mlp", "attention"), required=True)
    parser.add_argument(
        "--candidate-count", type=int, choices=(5, 10, 20), required=True
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--include-final", action="store_true")
    parser.add_argument("--no-wheel-guard", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    architecture_file = (
        "mlp_selected.yaml" if args.architecture == "mlp" else "attention_selected.yaml"
    )
    study = load_study_config(
        root / "configs" / architecture_file,
        args.base_config or root / "configs" / "base.yaml",
    )
    study = replace(
        study,
        environment=replace(study.environment, candidate_count=args.candidate_count),
    )
    checkpoint_root = args.run_dir.resolve() / "checkpoints"
    checkpoints = sorted(checkpoint_root.glob("iteration_*"))
    if args.include_final and (checkpoint_root / "final").exists():
        checkpoints.append(checkpoint_root / "final")
    if not checkpoints:
        raise SystemExit(f"no retained checkpoints found under {checkpoint_root}")

    rows = []
    for checkpoint in checkpoints:
        policy, _ = load_policy(checkpoint)
        for catalog_size in study.validation.catalog_sizes:
            for seed in study.validation.seeds:
                metrics = run_episode(
                    study,
                    method=args.architecture,
                    seed=seed,
                    catalog_size=catalog_size,
                    learned_policy=policy,
                    shield=True,
                    wheel_guard=not args.no_wheel_guard,
                )
                metrics["checkpoint"] = checkpoint.name
                rows.append(metrics)
    frame = pd.DataFrame(rows)
    output = args.run_dir.resolve() / "validation_metrics.csv"
    frame.to_csv(output, index=False)
    checkpoint_scores = (
        frame.groupby("checkpoint", as_index=False)["physical_validation_score"]
        .mean()
        .sort_values("physical_validation_score", ascending=False)
    )
    checkpoint_scores.to_csv(
        args.run_dir.resolve() / "validation_checkpoint_scores.csv", index=False
    )
    winner = str(checkpoint_scores.iloc[0]["checkpoint"])
    best_link = checkpoint_root / "best_validation"
    if best_link.is_symlink() or best_link.exists():
        if best_link.is_dir() and not best_link.is_symlink():
            raise RuntimeError(
                f"refusing to replace non-symlink directory {best_link}; move it first"
            )
        best_link.unlink()
    os.symlink(winner, best_link, target_is_directory=True)
    with (args.run_dir.resolve() / "best_validation.json").open("w") as stream:
        json.dump(
            {
                "checkpoint": winner,
                "mean_physical_validation_score": float(
                    checkpoint_scores.iloc[0]["physical_validation_score"]
                ),
                "wheel_guard_enabled": not args.no_wheel_guard,
                "selection_rule": (
                    "maximum mean predeclared physical validation score across held-out "
                    f"seeds {list(study.validation.seeds)} and catalog sizes "
                    f"{list(study.validation.catalog_sizes)}"
                ),
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
