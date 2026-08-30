#!/usr/bin/env python3
"""Paired Monte Carlo evaluation for learned and heuristic policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

if __package__ in {None, ""}:  # Allow direct execution from the repository root.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

import bsk_rl  # noqa: F401 - registers Gym environments
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.rl_module import RLModule

from examples.prospectus_rfi.acquisition_timeline import append_trajectory_snapshot
from examples.prospectus_rfi.config import git_metadata, load_study_config
from examples.prospectus_rfi.environment import (
    STUDY_MASKED_OBSERVATION_CONTRACT,
    make_environment_args,
)
from examples.prospectus_rfi.metrics import episode_metrics, physical_validation_score
from examples.prospectus_rfi.models import ObservationLayout


def find_inspector_module(checkpoint: Path) -> Path:
    checkpoint = checkpoint.resolve()
    if checkpoint.name == "inspector" and (checkpoint / "module_state.pt").exists():
        return checkpoint
    candidates = list(checkpoint.rglob("rl_module/inspector/module_state.pt"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one inspector module under {checkpoint}, found {len(candidates)}"
        )
    return candidates[0].parent


def load_policy(checkpoint: Path) -> tuple[Callable[[np.ndarray], int], dict]:
    module_path = find_inspector_module(checkpoint)
    module = RLModule.from_checkpoint(module_path)
    module.eval()
    parameter_total = sum(parameter.numel() for parameter in module.parameters())
    module_config = getattr(module, "config", None)
    observation_space = getattr(module_config, "observation_space", None)
    action_space = getattr(module_config, "action_space", None)
    observation_shape = list(getattr(observation_space, "shape", ()))
    action_count = getattr(action_space, "n", None)
    model_config = dict(getattr(module_config, "model_config_dict", {}) or {})

    def policy(observation: np.ndarray) -> int:
        tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            outputs = module.forward_inference({Columns.OBS: tensor})
        logits = outputs[Columns.ACTION_DIST_INPUTS][0]
        return int(torch.argmax(logits).item())

    return policy, {
        "checkpoint": str(checkpoint.resolve()),
        "module_path": str(module_path),
        "module_class": type(module).__name__,
        "trainable_parameters": int(parameter_total),
        "observation_shape": observation_shape,
        "action_count": None if action_count is None else int(action_count),
        "hidden_widths": model_config.get("fcnet_hiddens"),
        "activation": model_config.get("fcnet_activation"),
        "value_function_shares_layers": model_config.get("vf_share_layers"),
    }


def matched_angle_action(observation: np.ndarray, layout: ObservationLayout) -> int:
    rows = observation[layout.global_features :].reshape(
        layout.target_capacity, layout.target_features
    )
    valid = rows[:, layout.target_mask_index] > 0.5
    if not np.any(valid):
        return layout.target_capacity  # charge safely when every target is exhausted
    # AMOS row: elevation, rel-pos-H[3], pointing angle, distance, shadow, mask.
    pointing_angle_index = 4
    scores = np.where(valid, np.abs(rows[:, pointing_angle_index]), np.inf)
    return int(np.argmin(scores))


def apply_resource_shield(
    base_env,
    action: int,
    candidate_count: int,
    *,
    include_wheel_guard: bool = True,
):
    """Historical battery/storage shield, optionally with a wheel-speed guard."""

    scanner = base_env.satellites[0]
    battery = float(scanner.dynamics.battery_charge_fraction)
    storage = float(scanner.dynamics.storage_level_fraction)
    wheel = float(np.max(np.abs(scanner.dynamics.wheel_speeds_fraction)))
    if battery <= 0.20:
        return candidate_count, "battery"
    if storage >= 0.99:
        return candidate_count + 1, "storage"
    if include_wheel_guard and wheel >= 0.90:
        return candidate_count + 2, "reaction_wheel"
    return int(action), None


def scenario_fingerprint(base_env) -> str:
    values = []
    for target in base_env.scenario.target_spacecrafts:
        values.extend(np.asarray(target.target_spacecraft.dynamics.r_BN_N, dtype=float))
        values.extend(np.asarray(target.target_spacecraft.dynamics.v_BN_N, dtype=float))
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def run_episode(
    study,
    *,
    method: str,
    seed: int,
    catalog_size: int,
    learned_policy: Callable[[np.ndarray], int] | None,
    shield: bool,
    wheel_guard: bool = True,
    observation_contract: str = STUDY_MASKED_OBSERVATION_CONTRACT,
    trajectory_rows: list[dict] | None = None,
) -> dict[str, float | int | str | bool]:
    historical_modes = {
        "heuristic_historical": "angle",
        "heuristic_distance_historical": "distance",
    }
    historical_mode = historical_modes.get(method)
    env_args = make_environment_args(
        study.environment,
        fixed_catalog_size=catalog_size,
        historical_heuristic_mode=historical_mode,
        observation_contract=observation_contract,
    )
    env_args["log_level"] = "ERROR"
    environment = gym.make(
        "ConstellationTasking-v1",
        disable_env_checker=True,
        **env_args,
    )
    observation, _ = environment.reset(seed=seed)
    base = environment.unwrapped
    fingerprint = scenario_fingerprint(base)
    layout = ObservationLayout(target_capacity=study.environment.candidate_count)
    base.study_constraint_interventions = 0
    if trajectory_rows is not None:
        append_trajectory_snapshot(trajectory_rows, base)
    intervention_reasons = {"battery": 0, "storage": 0, "reaction_wheel": 0}
    inference_times_ns: list[int] = []
    steps = 0
    try:
        while "SS1" in observation:
            scanner_observation = np.asarray(observation["SS1"], dtype=np.float32)
            start_ns = time.perf_counter_ns()
            if method == "heuristic_matched":
                requested_action = matched_angle_action(scanner_observation, layout)
            elif method in historical_modes:
                # ImageRSO replaces any valid image slot with the full-catalog
                # angle/distance choice when its historical mode is active.
                requested_action = 0
            else:
                if learned_policy is None:
                    raise ValueError(
                        "a checkpoint policy is required for a learned method"
                    )
                requested_action = learned_policy(scanner_observation)
            inference_times_ns.append(time.perf_counter_ns() - start_ns)

            executed_action = requested_action
            if shield:
                executed_action, reason = apply_resource_shield(
                    base,
                    requested_action,
                    study.environment.candidate_count,
                    include_wheel_guard=wheel_guard,
                )
                if reason is not None and executed_action != requested_action:
                    base.study_constraint_interventions += 1
                    intervention_reasons[reason] += 1

            actions = {agent: 0 for agent in observation if agent != "SS1"}
            actions["SS1"] = int(executed_action)
            observation, _, terminated, truncated, _ = environment.step(actions)
            steps += 1
            if trajectory_rows is not None:
                append_trajectory_snapshot(trajectory_rows, base)
            if bool(terminated.get("SS1", False)) or bool(truncated.get("SS1", False)):
                break
            if steps > 1000:
                raise RuntimeError("episode exceeded conservative 1000-decision limit")

        metrics = episode_metrics(base)
        metrics.update(
            {
                "method": method,
                "scenario_seed": seed,
                "scenario_fingerprint": fingerprint,
                "catalog_size": catalog_size,
                "candidate_count": study.environment.candidate_count,
                "shield_enabled": shield,
                "wheel_guard_enabled": wheel_guard,
                "observation_contract": observation_contract,
                "battery_shield_interventions": intervention_reasons["battery"],
                "storage_shield_interventions": intervention_reasons["storage"],
                "wheel_shield_interventions": intervention_reasons["reaction_wheel"],
                "decision_count": steps,
                "mean_inference_ms": (
                    float(np.mean(inference_times_ns)) / 1.0e6
                    if inference_times_ns
                    else 0.0
                ),
                "median_inference_ms": (
                    float(np.median(inference_times_ns)) / 1.0e6
                    if inference_times_ns
                    else 0.0
                ),
            }
        )
        metrics["physical_validation_score"] = physical_validation_score(
            metrics, study.validation
        )
        if trajectory_rows is not None:
            for row in trajectory_rows:
                row.update(
                    {
                        "method": method,
                        "scenario_seed": seed,
                        "scenario_fingerprint": fingerprint,
                        "catalog_size": catalog_size,
                        "candidate_count": study.environment.candidate_count,
                    }
                )
        return metrics
    finally:
        environment.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=(
            "mlp",
            "attention",
            "heuristic_historical",
            "heuristic_distance_historical",
            "heuristic_matched",
        ),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--candidate-count", type=int, choices=(5, 10, 20), required=True
    )
    parser.add_argument(
        "--catalog-sizes", nargs="+", type=int, default=[100, 200, 300, 400]
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=700_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-config",
        type=Path,
        help=(
            "Common physical-study configuration. Defaults to configs/base.yaml; "
            "the memory-safe campaign must pass base_memorysafe_100_200.yaml."
        ),
    )
    parser.add_argument("--no-shield", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.method in {"mlp", "attention"} and args.checkpoint is None:
        raise SystemExit("--checkpoint is required for a learned policy")
    root = Path(__file__).resolve().parent
    architecture = (
        "mlp_selected.yaml" if args.method == "mlp" else "attention_selected.yaml"
    )
    if args.method.startswith("heuristic"):
        architecture = "mlp_selected.yaml"  # physical config only
    base_config = (args.base_config or root / "configs" / "base.yaml").resolve()
    study = load_study_config(
        root / "configs" / architecture, base_config
    )
    study = replace(
        study,
        environment=replace(study.environment, candidate_count=args.candidate_count),
    )
    learned_policy = None
    checkpoint_metadata = None
    if args.checkpoint is not None:
        learned_policy, checkpoint_metadata = load_policy(args.checkpoint)

    rows = []
    for catalog_size in args.catalog_sizes:
        for episode_index in range(args.episodes):
            seed = args.seed_start + episode_index
            print(
                f"evaluate method={args.method} K={args.candidate_count} "
                f"N={catalog_size} episode={episode_index+1}/{args.episodes} seed={seed}",
                flush=True,
            )
            rows.append(
                run_episode(
                    study,
                    method=args.method,
                    seed=seed,
                    catalog_size=catalog_size,
                    learned_policy=learned_policy,
                    shield=not args.no_shield,
                )
            )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    try:
        frame.to_parquet(output.with_suffix(".parquet"), index=False)
    except (ImportError, ModuleNotFoundError):
        pass
    metadata = {
        "method": args.method,
        "candidate_count": args.candidate_count,
        "catalog_sizes": args.catalog_sizes,
        "episodes_per_catalog_size": args.episodes,
        "scenario_seed_rule": "seed_start + episode_index; reused at every N and method",
        "seed_start": args.seed_start,
        "shield_enabled": not args.no_shield,
        "information_scope": (
            "full eligible catalog"
            if args.method in {
                "heuristic_historical",
                "heuristic_distance_historical",
            }
            else f"same {args.candidate_count}-candidate list as learned policies"
        ),
        "checkpoint": checkpoint_metadata,
        "base_config": str(base_config),
        "base_config_sha256": hashlib.sha256(base_config.read_bytes()).hexdigest(),
        "git": git_metadata(Path.cwd()),
        "study_config": study.to_dict(),
    }
    with output.with_suffix(".metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
