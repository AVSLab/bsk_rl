#!/usr/bin/env python3
"""Generate equal-budget tuning tables and select by held-out physical score."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ARCHITECTURES = {
    "mlp": "fixed_input_monolithic_mlp",
    "attention": "target_set_attention",
}


def generate_table(architecture: str, trials: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    mlp_widths = ("1024-1024", "2048-1024", "2048-2048", "2048-2048-1024")
    attention_widths = ("128", "256", "256-256", "512-256")
    for index in range(trials):
        is_attention = architecture == "attention"
        embedding = int(rng.choice([64, 128, 256])) if is_attention else 128
        compatible_heads = [head for head in (1, 2, 4, 8) if embedding % head == 0]
        row = {
            "trial_index": index,
            "architecture": ARCHITECTURES[architecture],
            "candidate_count_for_tuning": 10,
            "learning_rate": float(10 ** rng.uniform(-5.5, -3.5)),
            "train_batch_size": int(rng.choice([2100, 4200, 8400])),
            "minibatch_size": int(rng.choice([64, 128, 256])),
            "ppo_epochs": int(rng.choice([5, 10, 20, 30])),
            "clip_parameter": float(rng.choice([0.05, 0.10, 0.15, 0.20])),
            "entropy_coefficient": float(rng.choice([0.0, 0.005, 0.01, 0.025])),
            "value_function_coefficient": float(rng.choice([0.5, 1.0, 2.0])),
            "gradient_clip": float(rng.choice([0.25, 0.5, 1.0])),
            "gamma": float(rng.choice([0.99, 0.995, 0.999, 0.9997])),
            "gae_lambda": float(rng.choice([0.85, 0.90, 0.95, 0.98])),
            "continuous_time_discount": bool(rng.choice([True, False])),
            "reward_time": str(rng.choice(["step_start", "step_end"])),
            "hidden_widths": str(
                rng.choice(attention_widths if is_attention else mlp_widths)
            ),
            "embedding_dim": embedding,
            "attention_heads": int(rng.choice(compatible_heads)) if is_attention else 2,
            "attention_blocks": int(rng.choice([1, 2, 3])) if is_attention else 1,
            "feed_forward_width": (
                int(rng.choice([64, 128, 256, 512])) if is_attention else 128
            ),
            "status": "pending",
            "wall_hours": 8.0,
        }
        # PPO requires at least one minibatch.
        row["minibatch_size"] = min(row["minibatch_size"], row["train_batch_size"])
        rows.append(row)
    return pd.DataFrame(rows)


def select_configuration(table_path: Path, validation_path: Path, output_dir: Path):
    table = pd.read_csv(table_path)
    validation = pd.read_csv(validation_path)
    required = {"trial_index", "physical_validation_score"}
    if not required.issubset(validation.columns):
        raise ValueError(f"validation table must contain {sorted(required)}")
    scores = validation.groupby("trial_index", as_index=False).agg(
        physical_validation_score=("physical_validation_score", "mean"),
        validation_score_std=("physical_validation_score", "std"),
        validation_episode_count=("physical_validation_score", "size"),
    )
    merged = table.merge(scores, on="trial_index", how="left")
    completed = merged.dropna(subset=["physical_validation_score"])
    if completed.empty:
        raise ValueError("no tuning trial has held-out validation results")
    completed = completed.sort_values(
        ["physical_validation_score", "validation_score_std", "trial_index"],
        ascending=[False, True, True],
    )
    selected = completed.iloc[0]
    merged["status"] = np.where(
        merged["trial_index"] == selected["trial_index"], "selected", "rejected"
    )
    merged.loc[merged["physical_validation_score"].isna(), "status"] = "incomplete"
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "tuning_table_with_selection.csv", index=False)
    merged[merged["status"] == "rejected"].to_csv(
        output_dir / "rejected_configurations.csv", index=False
    )

    architecture_payload = {
        "architecture": {
            "name": selected["architecture"],
            "hidden_widths": [
                int(value) for value in selected["hidden_widths"].split("-")
            ],
            "embedding_dim": int(selected["embedding_dim"]),
            "attention_heads": int(selected["attention_heads"]),
            "attention_blocks": int(selected["attention_blocks"]),
            "feed_forward_width": int(selected["feed_forward_width"]),
            "activation": "relu",
            "layer_norm": bool(selected["architecture"] == "target_set_attention"),
            "dropout": 0.1
            if selected["architecture"] == "target_set_attention"
            else 0.0,
            "separate_value_network": True,
        },
        "ppo": {
            key: (
                bool(selected[key])
                if key == "continuous_time_discount"
                else selected[key].item()
                if hasattr(selected[key], "item")
                else selected[key]
            )
            for key in (
                "learning_rate",
                "train_batch_size",
                "minibatch_size",
                "ppo_epochs",
                "clip_parameter",
                "entropy_coefficient",
                "value_function_coefficient",
                "gradient_clip",
                "gamma",
                "gae_lambda",
                "continuous_time_discount",
                "reward_time",
            )
        },
        "selection": {
            "trial_index": int(selected["trial_index"]),
            "physical_validation_score": float(selected["physical_validation_score"]),
            "rule": (
                "maximum mean held-out physical_validation_score; then lower validation "
                "standard deviation; then lower trial index"
            ),
        },
    }
    with (output_dir / "selected_from_tuning.yaml").open("w") as stream:
        yaml.safe_dump(architecture_payload, stream, sort_keys=False)
    with (output_dir / "selection_rule.json").open("w") as stream:
        json.dump(architecture_payload["selection"], stream, indent=2)


def collect_validation(run_root: Path, architecture: str, output: Path) -> None:
    rows = []
    pattern = re.compile(rf"^{architecture}_k10_seed\d+_tune(\d+)$")
    for path in sorted((run_root / "training").glob(f"{architecture}_k10_*_tune*")):
        match = pattern.match(path.name)
        validation_path = path / "validation_metrics.csv"
        if match is None or not validation_path.exists():
            continue
        frame = pd.read_csv(validation_path)
        best_path = path / "best_validation.json"
        if not best_path.exists():
            continue
        with best_path.open() as stream:
            best = json.load(stream)["checkpoint"]
        frame = frame[frame["checkpoint"] == best].copy()
        frame["trial_index"] = int(match.group(1))
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(
            f"no validated {architecture} tuning runs under {run_root}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(output, index=False)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    generate.add_argument("--trials", type=int, default=12)
    generate.add_argument("--seed", type=int, default=202508)
    generate.add_argument("--output", type=Path, required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--table", type=Path, required=True)
    select.add_argument("--validation", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--run-root", type=Path, required=True)
    collect.add_argument("--architecture", choices=("mlp", "attention"), required=True)
    collect.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "generate":
        table = generate_table(args.architecture, args.trials, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    elif args.command == "select":
        select_configuration(args.table, args.validation, args.output_dir)
    else:
        collect_validation(args.run_root, args.architecture, args.output)


if __name__ == "__main__":
    main()
