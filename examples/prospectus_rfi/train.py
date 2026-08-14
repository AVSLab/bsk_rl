#!/usr/bin/env python3
"""Train one restartable PPO policy for a bounded wall-clock budget."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import signal
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # Allow ``python examples/.../train.py``.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import ray
import torch
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.logger import UnifiedLogger

from bsk_rl.utils.rllib.callbacks import WrappedEpisodeDataCallbacks
from bsk_rl.utils.rllib.discounting import TimeDiscountedGAEPPOTorchLearner
from bsk_rl.utils.utils import get_available_cores

try:
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
except (ImportError, ModuleNotFoundError):  # Ray < 2.35 compatibility
    from ray.rllib.core.rl_module.marl_module import (
        MultiAgentRLModuleSpec as MultiRLModuleSpec,
    )
    from ray.rllib.core.rl_module.rl_module import (
        SingleAgentRLModuleSpec as RLModuleSpec,
    )

from examples.prospectus_rfi.config import git_metadata, load_study_config
from examples.prospectus_rfi.environment import (
    environment_contract,
    make_environment_args,
)
from examples.prospectus_rfi.metrics import episode_metrics
from examples.prospectus_rfi.models import (
    FixedInputMonolithicPPOModule,
    TargetSetAttentionPPOModule,
    build_actor_critic,
    layout_from_environment,
    model_metadata,
)


STOP_REQUESTED = False


def _request_stop(signum, frame) -> None:  # pragma: no cover - scheduler behavior
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _module_class(architecture_name: str):
    if architecture_name == "fixed_input_monolithic_mlp":
        return FixedInputMonolithicPPOModule
    if architecture_name == "target_set_attention":
        return TargetSetAttentionPPOModule
    raise ValueError(f"unknown architecture {architecture_name}")


def _module_model_config(study) -> dict[str, Any]:
    architecture = study.architecture
    layout = layout_from_environment(study.environment)
    return {
        "global_features": layout.global_features,
        "target_features": layout.target_features,
        "target_capacity": layout.target_capacity,
        "target_mask_index": layout.target_mask_index,
        "non_target_actions": layout.non_target_actions,
        "hidden_widths": list(architecture.hidden_widths),
        "activation": architecture.activation,
        "layer_norm": architecture.layer_norm,
        "embedding_dim": architecture.embedding_dim,
        "attention_heads": architecture.attention_heads,
        "attention_blocks": architecture.attention_blocks,
        "feed_forward_width": architecture.feed_forward_width,
        "dropout": architecture.dropout,
    }


def _training_args(study) -> dict[str, Any]:
    ppo = study.ppo
    return {
        "lr": ppo.learning_rate,
        "gamma": ppo.gamma,
        "lambda_": ppo.gae_lambda,
        "train_batch_size": ppo.train_batch_size,
        "sgd_minibatch_size": ppo.minibatch_size,
        "num_sgd_iter": ppo.ppo_epochs,
        "clip_param": ppo.clip_parameter,
        "entropy_coeff": ppo.entropy_coefficient,
        "vf_loss_coeff": ppo.value_function_coefficient,
        "grad_clip": ppo.gradient_clip,
        "use_kl_loss": False,
    }


def build_ppo_config(study, seed: int, n_env_runners: int, temp_dir: Path):
    env_args = make_environment_args(
        study.environment,
        episode_data_callback=episode_metrics,
    )

    def policy_mapping_fn(agent_id, *args, **kwargs):
        del args, kwargs
        return "rso" if "target" in agent_id else "inspector"

    inspector_spec = RLModuleSpec(
        module_class=_module_class(study.architecture.name),
        model_config_dict=_module_model_config(study),
    )
    rso_spec = RLModuleSpec(
        model_config_dict={
            "use_lstm": False,
            "fcnet_hiddens": [2, 2],
            "vf_share_layers": False,
        }
    )
    training_args = _training_args(study)
    config = (
        PPOConfig()
        .training(**training_args)
        .env_runners(
            num_env_runners=n_env_runners,
            sample_timeout_s=50_000.0,
        )
        .environment(env="ConstellationTasking-RLlib", env_config=env_args)
        .callbacks(WrappedEpisodeDataCallbacks)
        .reporting(
            metrics_num_episodes_for_smoothing=1,
            metrics_episode_collection_timeout_s=180,
        )
        .checkpointing(export_native_model_files=True)
        .framework(framework="torch")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .resources(num_gpus=int(os.environ.get("BSK_RL_NUM_GPUS", "0")))
        .debugging(seed=seed)
        .multi_agent(
            policies={"inspector", "rso"},
            policy_mapping_fn=policy_mapping_fn,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                module_specs={"inspector": inspector_spec, "rso": rso_spec}
            )
        )
    )
    if study.ppo.continuous_time_discount:
        config.training(
            **training_args,
            learner_connector=lambda obs_space, act_space: (),
            learner_class=TimeDiscountedGAEPPOTorchLearner,
            learner_config_dict={"reward_time": study.ppo.reward_time},
        )
    config.logger_config = {
        "type": UnifiedLogger,
        "logdir": str(temp_dir.parent / "ray_logs"),
    }
    return config


def _scalar_metrics(result: dict[str, Any], elapsed_s: float, throughput: float):
    runners = result.get("env_runners", {})
    row: dict[str, Any] = {
        "training_iteration": result.get("training_iteration"),
        "environment_steps": result.get("num_env_steps_sampled_lifetime", 0),
        "wall_clock_s": elapsed_s,
        "wall_clock_h": elapsed_s / 3600.0,
        "samples_per_second": throughput,
        "episode_return_mean": runners.get("episode_return_mean"),
        "episode_len_mean": runners.get("episode_len_mean"),
    }
    # Metrics emitted by EpisodeDataWrapper appear as scalars or Stats objects.
    for key, value in runners.items():
        if key in row:
            continue
        if isinstance(value, (int, float, np.number, bool)):
            row[key] = float(value)
        elif hasattr(value, "peek"):
            try:
                peeked = value.peek()
                if isinstance(peeked, (int, float, np.number, bool)):
                    row[key] = float(peeked)
            except Exception:
                pass
    return row


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    # Maintain a stable core schema; the raw JSONL retains every extra metric.
    fields = [
        "training_iteration",
        "environment_steps",
        "wall_clock_s",
        "wall_clock_h",
        "samples_per_second",
        "episode_return_mean",
        "episode_len_mean",
    ]
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _replace_checkpoint(algorithm: PPO, path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    algorithm.save_checkpoint(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=("mlp", "attention"),
        required=True,
    )
    parser.add_argument("--architecture-config", type=Path)
    parser.add_argument(
        "--candidate-count", type=int, choices=(5, 10, 20), required=True
    )
    parser.add_argument("--seed", type=int, default=10_001)
    parser.add_argument("--wall-hours", type=float, default=48.0)
    parser.add_argument("--n-env-runners", type=int)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/prospectus_rfi")
    )
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tuning-table", type=Path)
    parser.add_argument("--trial-index", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    architecture_file = args.architecture_config or root / "configs" / (
        "mlp_selected.yaml" if args.architecture == "mlp" else "attention_selected.yaml"
    )
    study = load_study_config(
        architecture_file,
        root / "configs" / "base.yaml",
    )
    study = replace(
        study,
        environment=replace(study.environment, candidate_count=args.candidate_count),
    )
    if (args.tuning_table is None) != (args.trial_index is None):
        raise SystemExit("--tuning-table and --trial-index must be supplied together")
    if args.tuning_table is not None:
        table = pd.read_csv(args.tuning_table)
        matches = table[table["trial_index"] == args.trial_index]
        if len(matches) != 1:
            raise SystemExit(
                f"trial_index {args.trial_index} is not unique in the table"
            )
        row = matches.iloc[0]
        if row["architecture"] != study.architecture.name:
            raise SystemExit("tuning row architecture does not match --architecture")
        study = replace(
            study,
            ppo=replace(
                study.ppo,
                learning_rate=float(row["learning_rate"]),
                train_batch_size=int(row["train_batch_size"]),
                minibatch_size=int(row["minibatch_size"]),
                ppo_epochs=int(row["ppo_epochs"]),
                clip_parameter=float(row["clip_parameter"]),
                entropy_coefficient=float(row["entropy_coefficient"]),
                value_function_coefficient=float(row["value_function_coefficient"]),
                gradient_clip=float(row["gradient_clip"]),
                gamma=float(row["gamma"]),
                gae_lambda=float(row["gae_lambda"]),
                continuous_time_discount=str(row["continuous_time_discount"]).lower()
                in {"true", "1", "yes"},
                reward_time=str(row["reward_time"]),
            ),
            architecture=replace(
                study.architecture,
                hidden_widths=tuple(
                    int(value) for value in str(row["hidden_widths"]).split("-")
                ),
                embedding_dim=int(
                    row.get("embedding_dim", study.architecture.embedding_dim)
                ),
                attention_heads=int(
                    row.get("attention_heads", study.architecture.attention_heads)
                ),
                attention_blocks=int(
                    row.get("attention_blocks", study.architecture.attention_blocks)
                ),
                feed_forward_width=int(
                    row.get("feed_forward_width", study.architecture.feed_forward_width)
                ),
            ),
        )
    if args.smoke:
        study = replace(
            study,
            environment=replace(
                study.environment,
                catalog_min=100,
                catalog_max=100,
            ),
            ppo=replace(
                study.ppo,
                train_batch_size=128,
                minibatch_size=64,
                ppo_epochs=1,
            ),
        )
    study.validate()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    architecture_slug = (
        "mlp"
        if study.architecture.name == "fixed_input_monolithic_mlp"
        else "attention"
    )
    run_name = f"{architecture_slug}_k{args.candidate_count}_seed{args.seed}"
    if args.trial_index is not None:
        run_name += f"_tune{args.trial_index:02d}"
    run_dir = args.output_root.resolve() / "training" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = (
        args.temp_dir.resolve()
        if args.temp_dir
        else Path(os.environ.get("SLURM_TMPDIR", f"/tmp/{run_name}"))
    )
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RAY_TMPDIR"] = str(temp_dir)
    os.environ["TMPDIR"] = str(temp_dir)

    n_env_runners = args.n_env_runners
    if n_env_runners is None:
        allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", get_available_cores()))
        n_env_runners = max(1, allocated - 4)
    wall_hours = min(args.wall_hours, 0.05) if args.smoke else args.wall_hours
    wall_budget_s = wall_hours * 3600.0

    pure_model = build_actor_critic(
        study.architecture, layout_from_environment(study.environment)
    )
    metadata = {
        "run_name": run_name,
        "seed": args.seed,
        "candidate_count": args.candidate_count,
        "catalog_size_training_distribution": (
            "discrete_uniform_inclusive"
            f"[{study.environment.catalog_min},{study.environment.catalog_max}]"
        ),
        "requested_wall_hours": args.wall_hours,
        "effective_wall_hours": wall_hours,
        "n_env_runners": n_env_runners,
        "study_config": study.to_dict(),
        "environment_contract": environment_contract(study.environment),
        "model": model_metadata(pure_model),
        "git": git_metadata(Path.cwd()),
        "checkpoint_format": "RLlib 2.x Algorithm checkpoint with inspector module_state.pt",
    }
    with (run_dir / "metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    del pure_model

    ray.init(
        ignore_reinit_error=True,
        num_cpus=max(n_env_runners + 4, 1),
        object_store_memory=int(
            os.environ.get("BSK_RL_OBJECT_STORE_MEMORY", "2000000000")
        ),
        _temp_dir=str(temp_dir),
    )
    algorithm = PPO(build_ppo_config(study, args.seed, n_env_runners, temp_dir))
    if args.resume:
        algorithm.restore(str(args.resume.resolve()))

    csv_path = run_dir / "training_metrics.csv"
    jsonl_path = run_dir / "training_metrics.jsonl"
    start = time.monotonic()
    previous_time = start
    previous_steps = int(getattr(algorithm, "iteration", 0))
    iteration = 0
    try:
        while True:
            result = algorithm.train()
            iteration += 1
            now = time.monotonic()
            steps = int(result.get("num_env_steps_sampled_lifetime", 0))
            throughput = (steps - previous_steps) / max(now - previous_time, 1e-9)
            row = _scalar_metrics(result, now - start, throughput)
            _append_row(csv_path, row)
            with jsonl_path.open("a") as stream:
                stream.write(json.dumps(row, default=str, sort_keys=True) + "\n")
            previous_time, previous_steps = now, steps

            if iteration % study.compute.checkpoint_interval_iterations == 0:
                _replace_checkpoint(
                    algorithm, run_dir / "checkpoints" / f"iteration_{iteration:06d}"
                )
            print(
                f"[{run_name}] iteration={iteration} steps={steps} "
                f"wall_h={(now-start)/3600:.3f} samples_s={throughput:.3f}",
                flush=True,
            )
            if STOP_REQUESTED or now - start >= wall_budget_s:
                break
            if args.max_iterations is not None and iteration >= args.max_iterations:
                break
    finally:
        _replace_checkpoint(algorithm, run_dir / "checkpoints" / "final")
        status = {
            "completed_at_unix_s": time.time(),
            "wall_clock_s": time.monotonic() - start,
            "environment_steps": previous_steps,
            "iterations": iteration,
            "stop_requested": STOP_REQUESTED,
        }
        with (run_dir / "status.json").open("w") as stream:
            json.dump(status, stream, indent=2, sort_keys=True)
        algorithm.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
