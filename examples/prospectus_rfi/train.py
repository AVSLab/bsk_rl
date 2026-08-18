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
from examples.prospectus_rfi.wandb_logging import (
    maybe_init_wandb,
    public_wandb_metadata,
    wandb_settings,
)


STOP_REQUESTED = False


def _configure_cpu_threads(thread_count: int) -> None:
    """Bound BLAS/PyTorch parallelism inside every Ray process."""

    if thread_count < 1:
        raise ValueError("thread_count must be positive")
    value = str(thread_count)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value
    torch.set_num_threads(thread_count)


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
    ray_log_dir = temp_dir / "ray_logs"
    ray_log_dir.mkdir(parents=True, exist_ok=True)
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
            # Target spacecraft have one passive drift action.  Their tiny module
            # remains available for action generation, but their transitions must
            # not consume learner memory or PPO updates.
            policies_to_train=["inspector"],
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
        "logdir": str(ray_log_dir),
    }
    return config


def _scalar_metrics(
    result: dict[str, Any],
    elapsed_s: float,
    throughput: float,
    iteration_wall_s: float,
):
    runners = result.get("env_runners", {})
    row: dict[str, Any] = {
        "training_iteration": result.get("training_iteration"),
        "environment_steps": result.get("num_env_steps_sampled_lifetime", 0),
        "wall_clock_s": elapsed_s,
        "wall_clock_h": elapsed_s / 3600.0,
        "samples_per_second": throughput,
        "iteration_wall_s": iteration_wall_s,
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
        "iteration_wall_s",
        "episode_return_mean",
        "episode_len_mean",
    ]
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _load_prior_progress(run_dir: Path) -> dict[str, Any]:
    """Recover cumulative counters needed for a restartable training segment."""

    progress: dict[str, Any] = {
        "wall_clock_s": 0.0,
        "environment_steps": 0,
        "training_iteration": 0,
        "segments": [],
    }
    status_path = run_dir / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text())
        progress["wall_clock_s"] = float(
            status.get("cumulative_wall_clock_s", status.get("wall_clock_s", 0.0))
        )
        progress["environment_steps"] = int(status.get("environment_steps", 0))
        progress["training_iteration"] = int(
            status.get("training_iteration", status.get("iterations", 0))
        )
        progress["segments"] = list(status.get("segments", []))

    metrics_path = run_dir / "training_metrics.csv"
    if metrics_path.is_file():
        metrics = pd.read_csv(metrics_path)
        if not metrics.empty:
            last = metrics.iloc[-1]
            progress["wall_clock_s"] = max(
                progress["wall_clock_s"], float(last.get("wall_clock_s", 0.0))
            )
            progress["environment_steps"] = max(
                progress["environment_steps"],
                int(last.get("environment_steps", 0)),
            )
            progress["training_iteration"] = max(
                progress["training_iteration"],
                int(last.get("training_iteration", 0)),
            )
    return progress


def _segment_budget_seconds(
    *,
    target_wall_hours: float,
    prior_wall_seconds: float,
    segment_wall_hours: float | None,
) -> float:
    """Return this allocation's usable training time without exceeding the target."""

    remaining = max(0.0, target_wall_hours * 3600.0 - prior_wall_seconds)
    if segment_wall_hours is None:
        return remaining
    if segment_wall_hours <= 0.0:
        raise ValueError("segment_wall_hours must be positive")
    return min(remaining, segment_wall_hours * 3600.0)


def _can_start_iteration(
    *,
    elapsed_s: float,
    budget_s: float,
    shutdown_buffer_s: float,
    previous_iteration_s: float | None,
) -> bool:
    """Return whether another PPO iteration fits inside the guarded budget."""

    if shutdown_buffer_s < 0.0:
        raise ValueError("shutdown_buffer_s must be nonnegative")
    expected_iteration_s = (
        0.0 if previous_iteration_s is None else 1.25 * previous_iteration_s
    )
    required_s = max(shutdown_buffer_s, expected_iteration_s)
    return elapsed_s + required_s < budget_s


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
    parser.add_argument("--base-config", type=Path)
    parser.add_argument(
        "--candidate-count", type=int, choices=(5, 10, 20), required=True
    )
    parser.add_argument("--seed", type=int, default=10_001)
    parser.add_argument("--wall-hours", type=float, default=48.0)
    parser.add_argument(
        "--segment-wall-hours",
        type=float,
        help="Maximum time in this allocation; --wall-hours remains cumulative.",
    )
    parser.add_argument("--segment-index", type=int)
    parser.add_argument("--n-env-runners", type=int)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=int(os.environ.get("BSK_RL_TORCH_THREADS", "1")),
    )
    parser.add_argument(
        "--shutdown-buffer-minutes",
        type=float,
        default=15.0,
        help="Do not begin a new PPO iteration inside this end-of-segment buffer.",
    )
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
        args.base_config or root / "configs" / "base.yaml",
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
    _configure_cpu_threads(args.torch_threads)
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
    output_root = args.output_root.resolve()
    run_dir = output_root / "training" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    prior_progress = _load_prior_progress(run_dir)
    has_prior_output = any(
        (run_dir / name).exists()
        for name in ("training_metrics.csv", "status.json", "checkpoints")
    )
    if args.resume is None and has_prior_output:
        raise SystemExit(
            f"{run_dir} already contains training output; pass --resume explicitly"
        )
    if args.resume is not None and not args.resume.resolve().exists():
        raise SystemExit(f"resume checkpoint does not exist: {args.resume.resolve()}")
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
    segment_wall_hours = (
        min(args.segment_wall_hours, 0.05)
        if args.smoke and args.segment_wall_hours is not None
        else args.segment_wall_hours
    )
    wall_budget_s = _segment_budget_seconds(
        target_wall_hours=wall_hours,
        prior_wall_seconds=float(prior_progress["wall_clock_s"]),
        segment_wall_hours=segment_wall_hours,
    )
    shutdown_buffer_s = 0.0 if args.smoke else args.shutdown_buffer_minutes * 60.0
    if shutdown_buffer_s < 0.0:
        raise SystemExit("--shutdown-buffer-minutes must be nonnegative")
    resolved_wandb = wandb_settings(
        run_name,
        output_root,
        tuning=args.trial_index is not None,
    )

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
        "segment_wall_hours": segment_wall_hours,
        "n_env_runners": n_env_runners,
        "torch_threads": args.torch_threads,
        "shutdown_buffer_minutes": args.shutdown_buffer_minutes,
        "policies_to_train": ["inspector"],
        "physical_target_spacecraft_per_runner": study.environment.catalog_max,
        "study_config": study.to_dict(),
        "environment_contract": environment_contract(study.environment),
        "model": model_metadata(pure_model),
        "git": git_metadata(Path.cwd()),
        "wandb": public_wandb_metadata(resolved_wandb),
        "checkpoint_format": "RLlib 2.x Algorithm checkpoint with inspector module_state.pt",
    }
    metadata_path = run_dir / "metadata.json"
    if metadata_path.is_file():
        existing_metadata = json.loads(metadata_path.read_text())
        for key in ("run_name", "seed", "candidate_count"):
            if existing_metadata.get(key) != metadata.get(key):
                raise SystemExit(
                    f"resume metadata mismatch for {key}: "
                    f"{existing_metadata.get(key)!r} != {metadata.get(key)!r}"
                )
        metadata = existing_metadata
        metadata["latest_git"] = git_metadata(Path.cwd())
    attempts = list(metadata.get("execution_attempts", []))
    attempts.append(
        {
            "segment_index": args.segment_index,
            "resume_checkpoint": (
                str(args.resume.resolve()) if args.resume is not None else None
            ),
            "prior_wall_clock_s": float(prior_progress["wall_clock_s"]),
            "segment_budget_s": wall_budget_s,
            "n_env_runners": n_env_runners,
            "torch_threads": args.torch_threads,
            "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "started_at_unix_s": time.time(),
        }
    )
    metadata["execution_attempts"] = attempts
    with metadata_path.open("w") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    del pure_model

    if wall_budget_s <= 0.0:
        print(
            f"[{run_name}] cumulative target of {wall_hours:.3f} h already reached; "
            "no additional training required",
            flush=True,
        )
        return

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        num_cpus=max(n_env_runners + 4, 1),
        object_store_memory=int(
            os.environ.get("BSK_RL_OBJECT_STORE_MEMORY", "2000000000")
        ),
        _temp_dir=str(temp_dir),
    )
    algorithm = PPO(build_ppo_config(study, args.seed, n_env_runners, temp_dir))
    if args.resume:
        algorithm.restore(str(args.resume.resolve()))
    wandb_logger = maybe_init_wandb(resolved_wandb, metadata)
    if wandb_logger is not None:
        wandb_run = {
            **public_wandb_metadata(resolved_wandb),
            "url": getattr(wandb_logger.run, "url", None),
        }
        with (run_dir / "wandb_run.json").open("w") as stream:
            json.dump(wandb_run, stream, indent=2, sort_keys=True)

    csv_path = run_dir / "training_metrics.csv"
    jsonl_path = run_dir / "training_metrics.jsonl"
    start = time.monotonic()
    previous_time = start
    prior_wall_s = float(prior_progress["wall_clock_s"])
    previous_steps = int(prior_progress["environment_steps"])
    iteration = int(prior_progress["training_iteration"])
    previous_iteration_wall_s: float | None = None
    completed_normally = False
    try:
        while True:
            if STOP_REQUESTED:
                print(f"[{run_name}] scheduler stop requested", flush=True)
                break
            segment_elapsed_s = time.monotonic() - start
            if not _can_start_iteration(
                elapsed_s=segment_elapsed_s,
                budget_s=wall_budget_s,
                shutdown_buffer_s=shutdown_buffer_s,
                previous_iteration_s=previous_iteration_wall_s,
            ):
                print(
                    f"[{run_name}] stopping before another PPO iteration: "
                    f"segment_elapsed_s={segment_elapsed_s:.1f} "
                    f"segment_budget_s={wall_budget_s:.1f} "
                    f"previous_iteration_s={previous_iteration_wall_s}",
                    flush=True,
                )
                break
            iteration_start = time.monotonic()
            result = algorithm.train()
            iteration = int(result.get("training_iteration", iteration + 1))
            now = time.monotonic()
            previous_iteration_wall_s = now - iteration_start
            steps = int(result.get("num_env_steps_sampled_lifetime", 0))
            throughput = (steps - previous_steps) / max(now - previous_time, 1e-9)
            cumulative_elapsed_s = prior_wall_s + now - start
            row = _scalar_metrics(
                result,
                cumulative_elapsed_s,
                throughput,
                previous_iteration_wall_s,
            )
            _append_row(csv_path, row)
            with jsonl_path.open("a") as stream:
                stream.write(json.dumps(row, default=str, sort_keys=True) + "\n")
            if wandb_logger is not None:
                wandb_logger.log({**result, "prospectus_rfi": row})
            previous_time, previous_steps = now, steps

            if iteration % study.compute.checkpoint_interval_iterations == 0:
                _replace_checkpoint(
                    algorithm, run_dir / "checkpoints" / f"iteration_{iteration:06d}"
                )
            print(
                f"[{run_name}] iteration={iteration} steps={steps} "
                f"wall_h={cumulative_elapsed_s/3600:.3f} "
                f"segment_h={(now-start)/3600:.3f} "
                f"iteration_s={previous_iteration_wall_s:.1f} "
                f"samples_s={throughput:.3f}",
                flush=True,
            )
            if STOP_REQUESTED or now - start >= wall_budget_s:
                break
            if args.max_iterations is not None and iteration >= args.max_iterations:
                break
        completed_normally = True
    finally:
        _replace_checkpoint(algorithm, run_dir / "checkpoints" / "final")
        segment_elapsed_s = time.monotonic() - start
        cumulative_wall_s = prior_wall_s + segment_elapsed_s
        segments = list(prior_progress["segments"])
        segments.append(
            {
                "segment_index": args.segment_index,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "resume_checkpoint": (
                    str(args.resume.resolve()) if args.resume is not None else None
                ),
                "wall_clock_s": segment_elapsed_s,
                "cumulative_wall_clock_s": cumulative_wall_s,
                "training_iteration": iteration,
                "environment_steps": previous_steps,
                "stop_requested": STOP_REQUESTED,
            }
        )
        status = {
            "completed_at_unix_s": time.time(),
            "state": (
                "failed"
                if not completed_normally
                else (
                    "target_reached"
                    if cumulative_wall_s >= wall_hours * 3600.0
                    else "segment_completed"
                )
            ),
            "wall_clock_s": cumulative_wall_s,
            "cumulative_wall_clock_s": cumulative_wall_s,
            "segment_wall_clock_s": segment_elapsed_s,
            "environment_steps": previous_steps,
            "iterations": iteration,
            "training_iteration": iteration,
            "stop_requested": STOP_REQUESTED,
            "segments": segments,
        }
        with (run_dir / "status.json").open("w") as stream:
            json.dump(status, stream, indent=2, sort_keys=True)
        if wandb_logger is not None:
            wandb_logger.finish()
        algorithm.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
