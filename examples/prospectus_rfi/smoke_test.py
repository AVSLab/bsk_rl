#!/usr/bin/env python3
"""Complete N=100 and N=400 episodes with both policies and the heuristic."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch

from examples.prospectus_rfi.config import load_study_config
from examples.prospectus_rfi.evaluate import run_episode
from examples.prospectus_rfi.models import build_actor_critic, layout_from_environment


def random_policy(study):
    torch.manual_seed(2025)
    model = build_actor_critic(
        study.architecture, layout_from_environment(study.environment)
    ).eval()

    def policy(observation):
        tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            logits, _ = model(tensor)
        return int(torch.argmax(logits[0]).item())

    return policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-count", type=int, choices=(5, 10, 20), default=10)
    parser.add_argument(
        "--output", type=Path, default=Path("results/prospectus_rfi/smoke_test.csv")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    rows = []
    for method, filename in (
        ("mlp", "mlp_selected.yaml"),
        ("attention", "attention_selected.yaml"),
        ("heuristic_historical", "mlp_selected.yaml"),
    ):
        study = load_study_config(
            root / "configs" / filename, root / "configs" / "base.yaml"
        )
        study = replace(
            study,
            environment=replace(
                study.environment, candidate_count=args.candidate_count
            ),
        )
        policy = None if method.startswith("heuristic") else random_policy(study)
        for catalog_size in (100, 400):
            metrics = run_episode(
                study,
                method=method,
                seed=808_000 + catalog_size,
                catalog_size=catalog_size,
                learned_policy=policy,
                shield=True,
            )
            if metrics["episode_duration_s"] != study.environment.episode_duration_s:
                raise RuntimeError(
                    f"{method} N={catalog_size} ended at "
                    f"{metrics['episode_duration_s']} s instead of 45,000 s"
                )
            rows.append(metrics)
            print(
                f"PASS {method} K={args.candidate_count} N={catalog_size}: "
                f"{metrics['decision_count']} decisions, "
                f"{metrics['successful_observations']} observations",
                flush=True,
            )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
