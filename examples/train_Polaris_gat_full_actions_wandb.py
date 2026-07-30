#!/usr/bin/env python3
"""AMOS 2026 Polaris GAT trainer with full image/charge/downlink/desat actions.

Local IDE use:
    Press Run/Play on this file. The default local settings intentionally keep
    this lightweight enough for a startup check.

Local terminal equivalent:
    /Users/dahu1128/Repositories/bsk_rl/.venv/bin/python \
        /Users/dahu1128/Repositories/bsk_rl/examples/train_Polaris_gat_full_actions_wandb.py

Cluster use:
    Submit one of the reward-split Slurm wrappers, not the Python file directly:
        sbatch examples/amos_2026/sbatch_train_polaris_gat_full_actions_20d80i_48h.sh

Cluster comments:
    # Put the W&B key at /projects/$USER/bsk_rl/examples/wandb_key.txt, or set
    # BSK_RL_WANDB_KEY_PATH=/path/to/wandb_key.txt in the sbatch script.
    # The sbatch scripts also keep Ray's temp path short enough for AF_UNIX.

This script exposes charge, downlink, desat, and target image actions. The GAT
module expects non-imaging logits first, so the environment action order is:
charge, downlink, desat, then the repeated target-image actions.

This is observation layout v9 for the GAT full-action policy: one satellite
state block first, followed by repeated target-candidate chunks.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

# Keep imports such as `from sim_config import SimConfig` working from repo-root,
# from `examples/`, and from a Slurm working directory.
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import numpy as np
import torch
import yaml
from sim_config import SimConfig

_DEFAULT_TORCH_THREADS = "11" if os.environ.get("SLURM_JOB_ID") else "1"
_TORCH_THREADS = int(os.environ.get("BSK_RL_TORCH_THREADS", _DEFAULT_TORCH_THREADS))
torch.set_num_threads(_TORCH_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", str(_TORCH_THREADS))

import ray
from Basilisk.utilities import macros, orbitalMotion
from bsk_rl import act, data, obs, scene, sats
from bsk_rl.sim import dyn, fsw, world
from bsk_rl.utils.rllib.callbacks import WrappedEpisodeDataCallbacks
from bsk_rl.utils.rllib.discounting import TimeDiscountedGAEPPOTorchLearner
from bsk_rl.utils.utils import build_job_array, get_available_cores, sanitize_np
from pettingzoo.utils import BaseParallelWrapper
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.logger import UnifiedLogger
from gat_module_complete import GATModule
from wandb_config import WandbLogger

try:
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
except (ImportError, ModuleNotFoundError):  # Older versions of RLlib
    from ray.rllib.core.rl_module.marl_module import (
        MultiAgentRLModuleSpec as MultiRLModuleSpec,
    )
    from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec as RLModuleSpec


# Observation bookkeeping for the GAT module. The observation is:
# [satellite block][target_0 block][target_1 block]...
OBS_SAT_DIM = 14  # storage, battery, wheels(3), s_hat_H(3), eclipse(2), GS windows(4)
TARGET_FEATURES_PER_TARGET = 11  # priority, elevation, rel-pos-H(3), rel-vel-H(3), angle, distance, shadow
PRIORITY_NORM = 2.0  # Most baseline priorities land near [0, 2]; boosted priorities may intentionally exceed 1.
REL_VEL_NORM_MPS = 16_000.0  # [m/s], about twice circular LEO speed for worst-case retrograde encounters.
SUN_VECTOR_NORM = 1.0  # s_hat_H is already a dimensionless unit vector.
NON_IMAGING_ACTIONS = 3  # charge/downlink/desat first; image actions start at action 3
PPO_MIN_TRAIN_BATCH_SIZE = 128  # RLlib's default SGD minibatch size is 128.


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    return int(default if raw_value is None else raw_value)


def _env_optional_int(name: str, default: int | None = None) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return int(raw_value)


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    return float(default if raw_value is None else raw_value)


def _env_optional_float(name: str, default: float | None = None) -> float | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return float(raw_value)


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def _env_mix_weights(
    name: str, default: dict[str, float] | None = None
) -> dict[str, float] | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    parsed = json.loads(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object keyed by LEO/MEO/GEO.")
    return {str(key).upper(): float(value) for key, value in parsed.items()}


def _fixed_regime_counts(
    n_targets: int, mix_weights: dict[str, float]
) -> dict[str, int]:
    """Apportion a catalog into deterministic LEO/MEO/GEO counts."""
    regimes = ("LEO", "MEO", "GEO")
    weights = np.array([mix_weights.get(regime, 0.0) for regime in regimes])
    if n_targets <= 0:
        raise ValueError("n_targets must be positive.")
    if np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError(
            "mix_weights must contain nonnegative values with a positive sum."
        )

    raw_counts = n_targets * weights / weights.sum()
    counts = np.floor(raw_counts).astype(int)
    remainder = n_targets - int(counts.sum())
    fractional_order = sorted(
        range(len(regimes)),
        key=lambda index: (-(raw_counts[index] - counts[index]), index),
    )
    for index in fractional_order[:remainder]:
        counts[index] += 1
    return dict(zip(regimes, counts.tolist()))


def _cluster_scratch_root() -> Path:
    user = os.environ.get("USER", "dahu1128")
    scratch_root = Path(
        os.environ.get("BSK_RL_SCRATCH", f"/scratch/alpine/{user}")  # cluster default
    ).expanduser()
    if scratch_root.exists() or os.environ.get("SLURM_JOB_ID"):
        return scratch_root
    return Path("~/rllib_results/may_results").expanduser()  # local default


def _default_output_root() -> Path:
    explicit = os.environ.get("BSK_RL_OUTPUT_DIR")
    if explicit is not None:
        return Path(explicit).expanduser()  # cluster sbatch sets /scratch/alpine/$USER/rllib_results
    scratch_root = _cluster_scratch_root()
    if os.environ.get("SLURM_JOB_ID"):
        return scratch_root / "rllib_results"  # cluster fallback if BSK_RL_OUTPUT_DIR is not set
    return scratch_root  # local: ~/rllib_results/may_results/may6rllib_results


def _default_ray_tmpdir() -> Path:
    explicit = os.environ.get("BSK_RL_RAY_TMPDIR")
    if explicit is not None:
        return Path(explicit).expanduser()  # cluster sbatch sets /tmp/bskray_${SLURM_JOB_ID}_...

    if os.environ.get("SLURM_JOB_ID") and os.environ.get("TMPDIR"):
        return Path(os.environ["TMPDIR"]).expanduser()  # cluster fallback if TMPDIR is already set

    # Ray appends long socket paths below this directory, so keep it short.
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    if os.environ.get("SLURM_JOB_ID"):
        return Path(f"/tmp/bskray_{job_id}_{array_id}")  # cluster: avoid AF_UNIX path length errors
    # Local also needs a short path on macOS; Ray adds
    # session_.../sockets/plasma_store below this and otherwise exceeds AF_UNIX.
    return Path(f"/tmp/bskrl_{array_id}")  # local; outputs still go to ~/rllib_results/...


def _timestamp_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _checkpoint_iteration(path: Path) -> int | None:
    if not path.name.startswith("checkpoint_"):
        return None
    suffix = path.name.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _numeric_checkpoints(run_directory: Path) -> list[tuple[int, Path]]:
    checkpoints = []
    for path in run_directory.glob("checkpoint_*"):
        if not path.is_dir():
            continue
        iteration = _checkpoint_iteration(path)
        if iteration is not None:
            checkpoints.append((iteration, path))
    return sorted(checkpoints, key=lambda item: item[0])


def _latest_numeric_checkpoint(run_directory: Path) -> tuple[int, Path]:
    checkpoints = _numeric_checkpoints(run_directory)
    if not checkpoints:
        raise ValueError(f"No numeric checkpoints found in {run_directory}")
    return checkpoints[-1]


def _resolve_continuation_source(source: str) -> dict[str, Any]:
    """Resolve a user-provided run/checkpoint path to a concrete checkpoint."""
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Continuation source does not exist: {source_path}")

    if source_path.is_dir() and (
        source_path.name.startswith("checkpoint_")
        or (source_path / "learner_group").exists()
    ):
        run_directory = source_path.parent
        checkpoint_path = source_path
    elif source_path.is_dir() and _numeric_checkpoints(source_path):
        run_directory = source_path
        _, checkpoint_path = _latest_numeric_checkpoint(run_directory)
    elif source_path.is_dir():
        candidate_run_dirs = [
            child
            for child in source_path.iterdir()
            if child.is_dir() and _numeric_checkpoints(child)
        ]
        if len(candidate_run_dirs) != 1:
            raise ValueError(
                "BSK_RL_CONTINUE_FROM must point to a run directory, checkpoint "
                f"directory, or output directory with exactly one run. Found "
                f"{len(candidate_run_dirs)} candidate runs in {source_path}."
            )
        run_directory = candidate_run_dirs[0]
        _, checkpoint_path = _latest_numeric_checkpoint(run_directory)
    else:
        raise ValueError(f"Continuation source must be a directory: {source_path}")

    if checkpoint_path.name == "checkpoint_best":
        numeric_checkpoints = _numeric_checkpoints(run_directory)
        start_iteration = numeric_checkpoints[-1][0] + 1 if numeric_checkpoints else 0
    else:
        checkpoint_iter = _checkpoint_iteration(checkpoint_path)
        start_iteration = 0 if checkpoint_iter is None else checkpoint_iter + 1

    return {
        "source_path": source_path,
        "source_run_directory": run_directory,
        "source_checkpoint": checkpoint_path,
        "source_checkpoint_relative": checkpoint_path.relative_to(run_directory),
        "start_iteration": start_iteration,
    }


def _copy_continuation_run(
    *,
    continuation: dict[str, Any],
    destination_run_directory: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy an existing run directory and return the checkpoint inside the copy."""
    source_run_directory = Path(continuation["source_run_directory"])
    destination_run_directory = Path(destination_run_directory)

    if destination_run_directory.exists():
        if not overwrite:
            raise FileExistsError(
                f"Continuation destination already exists: {destination_run_directory}. "
                "Set BSK_RL_CONTINUE_OVERWRITE_COPY=1 only if this copied run can be replaced."
            )
        shutil.rmtree(destination_run_directory)

    destination_run_directory.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_run_directory, destination_run_directory, symlinks=True)
    copied_checkpoint = (
        destination_run_directory / continuation["source_checkpoint_relative"]
    )
    if not copied_checkpoint.exists():
        raise FileNotFoundError(
            f"Copied checkpoint was not found after copy: {copied_checkpoint}"
        )

    return {
        **continuation,
        "copied_run_directory": destination_run_directory,
        "copied_checkpoint": copied_checkpoint,
    }


def _wandb_key_path() -> Path:
    explicit = os.environ.get("BSK_RL_WANDB_KEY_PATH")
    if explicit:
        return Path(explicit).expanduser()  # cluster sbatch sets /projects/$USER/bsk_rl/examples/wandb_key.txt
    local_key = EXAMPLES_DIR / "wandb_key.txt"  # local: /Users/dahu1128/Repositories/bsk_rl/examples/wandb_key.txt
    if local_key.exists():
        return local_key
    return Path.cwd() / "wandb_key.txt"  # fallback if running from a different working directory


def _maybe_init_wandb(run_name: str, config: dict[str, Any]):
    if not _env_bool("BSK_RL_USE_WANDB", True):
        print("W&B disabled via BSK_RL_USE_WANDB=0")
        return None

    # Local runs are forgiving by default, so pressing play still works if W&B is
    # not installed. The W&B sbatch scripts set BSK_RL_REQUIRE_WANDB=1 so cluster
    # jobs fail immediately instead of silently training without live logging.
    require_wandb = _env_bool("BSK_RL_REQUIRE_WANDB", False)
    key_path = _wandb_key_path()
    if not key_path.exists():
        message = f"W&B key file not found at {key_path}"
        if require_wandb:
            raise FileNotFoundError(message)
        print(f"W&B disabled: {message}")
        return None

    try:
        return WandbLogger(
            project_name=os.environ.get("BSK_RL_WANDB_PROJECT", "amos2026-bsk-rl"),
            run_name=run_name,
            group=os.environ.get(
                "BSK_RL_WANDB_GROUP", "polaris-gat-full-actions-obs-v9"
            ),
            key_path=key_path,
            config=config,
        )
    except Exception as exc:
        if require_wandb:
            raise
        print(f"W&B disabled after initialization failure: {exc}")
        return None


def gat_model_config(n_targets_ahead: int) -> dict[str, Any]:
    """Return the GAT settings for full AMOS actions plus target image actions."""
    return {
        "n_targets": int(n_targets_ahead),  # candidate targets shown to policy, not total catalog targets
        "obs_sat": OBS_SAT_DIM,  # number of non-target/global observation values at the front
        "non_imaging_actions": NON_IMAGING_ACTIONS,
        "width_f": 256,
        "depth_f": 2,
        "width_g": 128,
        "depth_g": 4,
        "tgt_encoded_dim": 128,
        "attention_depth": 1,
        "num_heads": 2,
        "attention_dim": 128,
        "width_f_sat": 256,
        "depth_f_sat": 2,
        "width_g_sat": 128,
        "depth_g_sat": 4,
        "sat_attention_dim": 128,
        "sat_attention_heads": 2,
        "sat_encoded_dim": 128,
        "critic_tgt_encoded_dim": 128,
        "critic_width_f": 256,
        "critic_depth_f": 2,
        "critic_width_g": 64,
        "critic_depth_g": 3,
        "dropout": 0.1,
        "post_self_attention": True,
    }


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _curriculum_alpha(
    difficulty: float,
    *,
    start_alpha: float,
    end_alpha: float,
    power: float = 1.0,
) -> float:
    """Map a normalized curriculum difficulty to the downlink reward weight."""
    difficulty = _clamp01(difficulty)
    power = max(float(power), 1e-12)
    return float(start_alpha) + (float(end_alpha) - float(start_alpha)) * (
        difficulty**power
    )


def _unwrap_bsk_env(env: Any) -> Any | None:
    """Find the underlying bsk_rl env below RLlib/PettingZoo/Gym wrappers."""
    seen: set[int] = set()
    stack = [env]
    while stack:
        candidate = stack.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if hasattr(candidate, "satellites") and hasattr(candidate, "rewarder"):
            return candidate
        for attr in ("par_env", "env", "unwrapped"):
            try:
                child = getattr(candidate, attr)
            except Exception:
                child = None
            if child is not None and child is not candidate:
                stack.append(child)
        for attr in ("envs", "vector_env"):
            try:
                children = getattr(candidate, attr)
            except Exception:
                children = None
            if isinstance(children, (list, tuple)):
                stack.extend(children)
            elif children is not None and children is not candidate:
                stack.append(children)
    return None


def _apply_reward_alpha_to_env(env: Any, alpha: float) -> bool:
    """Apply alpha to the scanner reward split without changing the action space."""
    bsk_env = _unwrap_bsk_env(env)
    if bsk_env is None:
        return False

    alpha = _clamp01(alpha)
    imaging_bonus = 1.0 - alpha
    setattr(bsk_env, "curriculum_alpha", alpha)
    setattr(bsk_env, "curriculum_imaging_bonus", imaging_bonus)
    if hasattr(bsk_env, "rewarder"):
        setattr(bsk_env.rewarder, "alpha", alpha)
        setattr(bsk_env.rewarder, "curriculum_alpha", alpha)

    satellites = getattr(bsk_env, "satellites", [])
    if not satellites:
        return True
    scanner = satellites[0]

    for args_name in ("sat_args_generator", "sat_args"):
        args = getattr(scanner, args_name, None)
        if isinstance(args, dict):
            args["downlink_bonus"] = alpha
            args["imaging_bonus"] = imaging_bonus

    dynamics = getattr(scanner, "dynamics", None)
    if dynamics is not None:
        dynamics.downlink_bonus = alpha
        dynamics.imaging_bonus = imaging_bonus

    return True


class WrapperCurriculumLearning(BaseParallelWrapper):
    """PettingZoo wrapper that ramps the AMOS reward alpha during training.

    The training reward split lives on the scanner dynamics as
    ``downlink_bonus`` and ``imaging_bonus``. This wrapper keeps those values in
    sync with a curriculum difficulty, while preserving the existing wrapped
    BSK-RL/PettingZoo observation and action spaces.
    """

    def __init__(
        self,
        env,
        difficulty: float = 0.0,
        difficulty_function: Callable[[float], float] | None = None,
        local_step_increment: float = 0.0,
    ):
        super().__init__(env)
        self.difficulty = _clamp01(difficulty)
        self.difficulty_function = difficulty_function or (lambda x: x)
        self.local_step_increment = max(0.0, float(local_step_increment))
        self.current_alpha = float(self.difficulty_function(self.difficulty))

    def _apply(self) -> None:
        self.current_alpha = _clamp01(self.difficulty_function(self.difficulty))
        _apply_reward_alpha_to_env(self.env, self.current_alpha)

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        # Dynamics are reconstructed during reset, so apply again afterwards.
        self._apply()
        return result

    def step(self, actions):
        self._apply()
        result = self.env.step(actions)
        if self.local_step_increment > 0.0:
            self.difficulty = _clamp01(self.difficulty + self.local_step_increment)
        return result

    def set_task(self, difficulty: float) -> None:
        self.difficulty = _clamp01(difficulty)
        self._apply()

    def get_task(self) -> float:
        return self.difficulty


def _make_alpha_curriculum_wrapper(
    *,
    start_alpha: float,
    end_alpha: float,
    ramp_steps_per_env: int,
    power: float,
):
    local_step_increment = 1.0 / max(1, int(ramp_steps_per_env))

    def wrap(env):
        return WrapperCurriculumLearning(
            env,
            difficulty=0.0,
            difficulty_function=lambda difficulty: _curriculum_alpha(
                difficulty,
                start_alpha=start_alpha,
                end_alpha=end_alpha,
                power=power,
            ),
            local_step_increment=local_step_increment,
        )

    return wrap


def _peek_curriculum_steps(metrics_logger) -> float:
    """Read the best available sampled-step counter from an RLlib MetricsLogger."""
    if metrics_logger is None:
        return 0.0
    keys = (
        ("env_runners", "num_module_steps_sampled_lifetime", "inspector"),
        ("env_runners", "num_agent_steps_sampled_lifetime", "SS1"),
        ("num_module_steps_sampled_lifetime", "inspector"),
        ("num_agent_steps_sampled_lifetime", "SS1"),
        "num_env_steps_sampled_lifetime",
    )
    for key in keys:
        try:
            value = metrics_logger.peek(key, default=None)
        except Exception:
            continue
        try:
            if value is not None and np.isfinite(float(value)):
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _curriculum_steps_from_results(results: dict[str, Any]) -> float:
    env_runner_results = results.get("env_runners", {})
    for source, key in (
        (env_runner_results.get("num_module_steps_sampled_lifetime", {}), "inspector"),
        (env_runner_results.get("num_agent_steps_sampled_lifetime", {}), "SS1"),
    ):
        if isinstance(source, dict) and key in source:
            try:
                return float(source[key])
            except (TypeError, ValueError):
                pass
    for key in ("num_env_steps_sampled_lifetime", "num_env_steps_sampled"):
        if key in results:
            try:
                return float(results[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def initialize_alpha_curriculum_callbacks(
    *,
    ramp_steps: int,
    start_alpha: float,
    end_alpha: float,
    power: float = 1.0,
):
    class CLCallbacks(WrappedEpisodeDataCallbacks):
        def _difficulty_from_metrics(self, metrics_logger) -> float:
            return _clamp01(_peek_curriculum_steps(metrics_logger) / max(1, ramp_steps))

        def _set_env_task(self, env, env_index: int, difficulty: float) -> None:
            candidates = []
            if env is not None:
                candidates.append(env)
                try:
                    envs = getattr(env, "envs", None)
                except Exception:
                    envs = None
                if isinstance(envs, (list, tuple)) and 0 <= env_index < len(envs):
                    candidates.insert(0, envs[env_index])
            for candidate in candidates:
                task_env = self._find_task_env(candidate)
                if task_env is not None:
                    task_env.set_task(difficulty)
                    return
                alpha = _curriculum_alpha(
                    difficulty,
                    start_alpha=start_alpha,
                    end_alpha=end_alpha,
                    power=power,
                )
                if _apply_reward_alpha_to_env(candidate, alpha):
                    return

        def _find_task_env(self, env) -> Any | None:
            seen: set[int] = set()
            stack = [env]
            while stack:
                candidate = stack.pop()
                if candidate is None or id(candidate) in seen:
                    continue
                seen.add(id(candidate))
                if hasattr(candidate, "set_task") and hasattr(candidate, "get_task"):
                    return candidate
                for attr in ("par_env", "env", "unwrapped"):
                    try:
                        child = getattr(candidate, attr, None)
                    except Exception:
                        child = None
                    if child is not None and child is not candidate:
                        stack.append(child)
                try:
                    envs = getattr(candidate, "envs", None)
                except Exception:
                    envs = None
                if isinstance(envs, (list, tuple)):
                    stack.extend(envs)
            return None

        def on_environment_created(self, *, env=None, metrics_logger=None, **kwargs):
            super().on_environment_created(
                env=env, metrics_logger=metrics_logger, **kwargs
            )
            self._set_env_task(env, 0, 0.0)

        def on_episode_start(
            self,
            *,
            metrics_logger=None,
            env=None,
            env_index,
            **kwargs,
        ) -> None:
            difficulty = self._difficulty_from_metrics(metrics_logger)
            self._set_env_task(env, env_index, difficulty)
            alpha = _curriculum_alpha(
                difficulty,
                start_alpha=start_alpha,
                end_alpha=end_alpha,
                power=power,
            )
            if metrics_logger is not None:
                metrics_logger.log_value(
                    "curriculum_alpha", alpha, clear_on_reduce=True
                )
                metrics_logger.log_value(
                    "curriculum_difficulty", difficulty, clear_on_reduce=True
                )

    return CLCallbacks


def _push_curriculum_alpha_to_envs(ppo: PPO, alpha: float) -> None:
    """Best-effort alpha update for already-created env runners."""

    def set_on_runner(env_runner):
        if hasattr(env_runner, "foreach_env"):
            return env_runner.foreach_env(
                lambda env: _apply_reward_alpha_to_env(env, alpha)
            )
        env = getattr(env_runner, "env", None)
        if env is not None:
            return _apply_reward_alpha_to_env(env, alpha)
        return False

    try:
        ppo.env_runner_group.foreach_worker(
            set_on_runner,
            local_env_runner=True,
            healthy_only=True,
            timeout_seconds=30,
        )
    except Exception as exc:
        print(f"[curriculum] warning: failed to push alpha to env runners: {exc}")


def train_model(
    model_name: str,
    output_directory: Path,
    inspector_rl_module_spec: RLModuleSpec,
    env_args: dict[str, Any],
    training_args: dict[str, Any],
    n_envs: int = 1,
    checkpoint_frequency: int = 1,
    checkpoints_to_keep: int = 2,
    reload_frequency: int = 500_000,
    total_timesteps: int | None = 1_000_000,
    temp_dir: str = "/tmp",
    wandb_logger=None,
    curriculum_config: dict[str, Any] | None = None,
    restore_checkpoint_dir: Path | None = None,
    start_iteration: int = 0,
    timeout: float | None = None,
) -> None:
    os.environ["RAY_TMPDIR"] = os.environ["TMPDIR"] = temp_dir
    output_directory = Path(output_directory)
    run_directory = output_directory / model_name
    run_directory.mkdir(exist_ok=True, parents=True)

    def policy_mapping_fn(agent_id, *args, **kwargs):
        if "target" in agent_id:
            return "rso"
        return "inspector"

    ray.init(
        ignore_reinit_error=True,
        num_cpus=get_available_cores(),
        object_store_memory=_env_int("BSK_RL_OBJECT_STORE_MEMORY", 2_000_000_000),
        _temp_dir=temp_dir,
    )

    curriculum_config = curriculum_config or {}
    curriculum_enabled = bool(curriculum_config.get("enabled", False))
    callback_class = WrappedEpisodeDataCallbacks
    if curriculum_enabled:
        callback_class = initialize_alpha_curriculum_callbacks(
            ramp_steps=int(curriculum_config["ramp_steps"]),
            start_alpha=float(curriculum_config["start_alpha"]),
            end_alpha=float(curriculum_config["end_alpha"]),
            power=float(curriculum_config.get("power", 1.0)),
        )

    config = (
        PPOConfig()
        .training(**training_args)
        .env_runners(
            num_env_runners=n_envs,
            sample_timeout_s=50000.0,
        )
        .environment(
            env="ConstellationTasking-RLlib",
            env_config=env_args,
        )
        .callbacks(callback_class)
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
        .resources(num_gpus=0)
        .multi_agent(
            policies={"inspector", "rso"},
            policy_mapping_fn=policy_mapping_fn,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                module_specs={
                    "inspector": inspector_rl_module_spec,
                    # Target spacecraft only drift. Keep their policy tiny so the
                    # training signal and parameter count are dominated by SS1.
                    "rso": RLModuleSpec(
                        model_config_dict={
                            "use_lstm": False,
                            "fcnet_hiddens": [2, 2],
                            "vf_share_layers": False,
                        },
                    ),
                }
            ),
        )
    )
    config.training(
        **training_args,
        learner_connector=lambda obs_space, act_space: (),
        learner_class=TimeDiscountedGAEPPOTorchLearner,
        learner_config_dict=dict(reward_time="step_start"),
    )
    config.logger_config = dict(type=UnifiedLogger, logdir=run_directory)

    iteration = int(start_iteration)
    step = 0
    current_best_return = -np.inf
    tic = time.monotonic()

    try:
        ppo = PPO(config)
        if restore_checkpoint_dir is not None:
            restore_checkpoint_dir = Path(restore_checkpoint_dir)
            print(
                f"[continue] restoring PPO state from {restore_checkpoint_dir}",
                flush=True,
            )
            ppo.restore(str(restore_checkpoint_dir))

        while True:
            prev_step = step
            print(
                f"[train] starting iteration {iteration} at sampled_steps={step}",
                flush=True,
            )
            results = ppo.train()
            step = results["num_env_steps_sampled_lifetime"]
            step_return = results["env_runners"].get("episode_return_mean", -np.inf)
            if curriculum_enabled:
                curriculum_steps = _curriculum_steps_from_results(results)
                curriculum_difficulty = _clamp01(
                    curriculum_steps / max(1, int(curriculum_config["ramp_steps"]))
                )
                curriculum_alpha = _curriculum_alpha(
                    curriculum_difficulty,
                    start_alpha=float(curriculum_config["start_alpha"]),
                    end_alpha=float(curriculum_config["end_alpha"]),
                    power=float(curriculum_config.get("power", 1.0)),
                )
                _push_curriculum_alpha_to_envs(ppo, curriculum_alpha)
                results["curriculum_alpha"] = curriculum_alpha
                results["curriculum_difficulty"] = curriculum_difficulty
                results["curriculum_sampled_steps"] = curriculum_steps
                print(
                    "[curriculum] "
                    f"sampled_steps={curriculum_steps:.0f}, "
                    f"difficulty={curriculum_difficulty:.4f}, "
                    f"alpha={curriculum_alpha:.4f}",
                    flush=True,
                )
            print(
                "[train] finished iteration "
                f"{iteration}: sampled_steps={step}, episode_return_mean={step_return}",
                flush=True,
            )

            if wandb_logger is not None:
                wandb_logger.log(results)

            if step_return > current_best_return:
                checkpoint_path = run_directory / "checkpoint_best"
                try:
                    shutil.rmtree(checkpoint_path)
                except FileNotFoundError:
                    pass
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                ppo.save_checkpoint(checkpoint_path)
                with open(
                    checkpoint_path / f"iteration_{str(iteration).zfill(6)}.txt", "w"
                ) as file:
                    file.write(f"iter: {iteration}\n")
                current_best_return = step_return

            checkpoint_path = run_directory / f"checkpoint_{str(iteration).zfill(6)}"
            if iteration % checkpoint_frequency == 0:
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                ppo.save_checkpoint(checkpoint_path)

            if total_timesteps is not None and step > total_timesteps:
                break

            if timeout is not None and time.monotonic() - tic >= timeout:
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                ppo.save_checkpoint(checkpoint_path)
                print(
                    "[train] timeout reached; saved final checkpoint at "
                    f"{checkpoint_path}",
                    flush=True,
                )
                break

            if step % reload_frequency < prev_step % reload_frequency:
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                ppo.save_checkpoint(checkpoint_path)
                ray.shutdown()
                ray.init(
                    ignore_reinit_error=True,
                    num_cpus=get_available_cores(),
                    object_store_memory=_env_int(
                        "BSK_RL_OBJECT_STORE_MEMORY", 3_000_000_000
                    ),
                    _temp_dir=temp_dir,
                )
                ppo = PPO.from_checkpoint(checkpoint_path)

            if iteration > checkpoints_to_keep * checkpoint_frequency - 1:
                for i in range(checkpoint_frequency):
                    remove_dir = (
                        run_directory
                        / f"checkpoint_{str(iteration - checkpoints_to_keep * checkpoint_frequency - i).zfill(6)}"
                    )
                    try:
                        shutil.rmtree(remove_dir)
                    except FileNotFoundError:
                        pass

            iteration += 1
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()


def _finite_values(values):
    finite = []
    for value in values:
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            finite.append(value)
    return finite


def _safe_mean(values, default=-1.0):
    values = _finite_values(values)
    return float(np.mean(values)) if values else default


def _safe_median(values, default=-1.0):
    values = _finite_values(values)
    return float(np.median(values)) if values else default


def _safe_std(values, default=-1.0):
    values = _finite_values(values)
    return float(np.std(values)) if values else default


def _safe_max(values, default=-1.0):
    values = _finite_values(values)
    return float(np.max(values)) if values else default


def _safe_min(values, default=-1.0):
    values = _finite_values(values)
    return float(np.min(values)) if values else default


def _reward_split_tag(alpha: float) -> str:
    """Return the existing reward split style, e.g. 20d80i or 00d100i."""
    downlink_pct = int(round(100.0 * float(alpha)))
    imaging_pct = int(round(100.0 * (1.0 - float(alpha))))
    return f"{downlink_pct:02d}d{imaging_pct:02d}i"


def _alpha_tag(alpha: float) -> str:
    """Return a filesystem-friendly alpha tag, e.g. alpha0p2."""
    return f"alpha{float(alpha):g}".replace(".", "p")


def env_metrics_callback(env):
    """Episode-level metrics useful for the full-action GAT experiment."""
    data = {}
    reward_data = env.rewarder.data
    episode_duration = float(env.simulator.sim_time)
    ss1_action_specs = getattr(env.satellites[0].action_builder, "action_spec", [])
    ss1_actions = next(
        (
            spec
            for spec in ss1_action_specs
            if hasattr(spec, "imaging_attempt_records")
            or hasattr(spec, "chosen_target_ids")
        ),
        None,
    )

    num_imaged = len(getattr(reward_data, "imaged", []))
    data["num_unique_targets_imaged"] = num_imaged
    data["episode_duration_sec"] = episode_duration
    data["number of alive cases"] = env.satellites[0].dynamics.is_alive()
    data["battery_valid"] = env.satellites[0].dynamics.battery_valid()
    data["rw_valid"] = env.satellites[0].dynamics.rw_speeds_valid()
    data["cumulativeRewardSS1"] = env.rewarder.cum_reward["SS1"]
    data["illuminated_images"] = len(getattr(env.rewarder, "imaged_illuminated", []))
    data["curriculum_alpha"] = float(
        getattr(
            env,
            "curriculum_alpha",
            getattr(env.satellites[0].dynamics, "downlink_bonus", -1.0),
        )
    )
    data["curriculum_imaging_bonus"] = float(
        getattr(
            env,
            "curriculum_imaging_bonus",
            getattr(env.satellites[0].dynamics, "imaging_bonus", -1.0),
        )
    )

    target_priority_by_id = {}
    try:
        target_priority_by_id = {
            int(target.id): float(target.priority)
            for target in env.scenario.target_spacecrafts
        }
    except Exception:
        target_priority_by_id = {}

    all_priorities = list(target_priority_by_id.values())
    data["target_priority_sum"] = float(np.sum(all_priorities)) if all_priorities else -1.0
    data["target_priority_mean"] = _safe_mean(all_priorities)
    data["target_priority_std"] = _safe_std(all_priorities)
    data["target_priority_max"] = _safe_max(all_priorities)
    data["mean_target_priority"] = data["target_priority_mean"]
    data["std_target_priority"] = data["target_priority_std"]
    data["max_target_priority"] = data["target_priority_max"]

    scenario = getattr(env, "scenario", None)
    hio_ids = set(getattr(scenario, "hio_target_ids", []))
    shio_ids = set(getattr(scenario, "shio_target_ids", []))
    active_n_targets = int(getattr(scenario, "n_targets", n_targets))
    data["n_targets"] = active_n_targets
    event_enabled = bool(getattr(scenario, "dynamic_priority_event_enabled", False))
    event_time = getattr(scenario, "priority_event_time", None)
    if event_time is None and scenario is not None:
        event_time = getattr(scenario, "dynamic_priority_event_time_sec", None)
    if event_time is None and scenario is not None:
        event_time = (
            float(env.time_limit)
            * float(getattr(scenario, "dynamic_priority_event_fraction", 0.5))
        )
    event_time = float(event_time) if event_time is not None else -1.0
    event_applied_time = getattr(scenario, "priority_event_applied_time", None)
    event_targets = [
        target
        for target in getattr(scenario, "target_spacecrafts", [])
        if getattr(target, "priority_event_kind", "") in {"HIO", "SHIO"}
    ]
    hio_targets = [
        target for target in event_targets if getattr(target, "priority_event_kind", "") == "HIO"
    ]
    shio_targets = [
        target for target in event_targets if getattr(target, "priority_event_kind", "") == "SHIO"
    ]
    data["dynamic_priority_event_enabled"] = float(event_enabled)
    data["dynamic_priority_event_applied"] = float(
        bool(getattr(scenario, "priority_event_applied", False))
    )
    data["dynamic_priority_event_time_sec"] = event_time
    data["dynamic_priority_event_applied_time_sec"] = (
        float(event_applied_time) if event_applied_time is not None else -1.0
    )
    data["hio_target_count"] = len(hio_ids)
    data["shio_target_count"] = len(shio_ids)
    data["hio_candidate_access_count"] = int(
        sum(getattr(target, "priority_event_candidate_count", 0) for target in hio_targets)
    )
    data["shio_candidate_access_count"] = int(
        sum(getattr(target, "priority_event_candidate_count", 0) for target in shio_targets)
    )
    data["time_to_first_hio_candidate_access_sec"] = _safe_min(
        getattr(target, "priority_event_first_candidate_time", None)
        for target in hio_targets
    )
    data["time_to_first_shio_candidate_access_sec"] = _safe_min(
        getattr(target, "priority_event_first_candidate_time", None)
        for target in shio_targets
    )

    imaging_attempt_records = list(getattr(ss1_actions, "imaging_attempt_records", []))
    imaging_durations = [
        float(record["end_time"]) - float(record["start_time"])
        for record in imaging_attempt_records
        if record.get("start_time") is not None and record.get("end_time") is not None
    ]
    successful_durations = [
        float(record["end_time"]) - float(record["start_time"])
        for record in imaging_attempt_records
        if record.get("success")
        and record.get("start_time") is not None
        and record.get("end_time") is not None
    ]
    unsuccessful_durations = [
        float(record["end_time"]) - float(record["start_time"])
        for record in imaging_attempt_records
        if not record.get("success")
        and record.get("start_time") is not None
        and record.get("end_time") is not None
    ]
    imaging_slew_times = [
        record.get("slew_time_s") for record in imaging_attempt_records
    ]
    successful_slew_times = [
        record.get("slew_time_s")
        for record in imaging_attempt_records
        if record.get("success")
    ]
    unsuccessful_slew_times = [
        record.get("slew_time_s")
        for record in imaging_attempt_records
        if not record.get("success")
    ]
    imaging_success_flags = [bool(record.get("success")) for record in imaging_attempt_records]
    attempted_priorities = [
        record.get(
            "target_priority",
            target_priority_by_id.get(int(record["target_id"])),
        )
        for record in imaging_attempt_records
        if record.get("target_id") is not None
    ]
    successful_priorities = [
        record.get(
            "target_priority",
            target_priority_by_id.get(int(record["target_id"])),
        )
        for record in imaging_attempt_records
        if record.get("success") and record.get("target_id") is not None
    ]

    data["num_imaging_attempts"] = len(imaging_attempt_records)
    data["imaging_attempt_success_rate"] = _safe_mean(imaging_success_flags)
    data["actual_imaging_action_time_sec"] = float(np.sum(imaging_durations))
    data["actual_non_imaging_time_sec"] = episode_duration - data["actual_imaging_action_time_sec"]
    data["mean_imaging_action_duration_sec"] = _safe_mean(imaging_durations)
    data["median_imaging_action_duration_sec"] = _safe_median(imaging_durations)
    data["mean_successful_imaging_action_duration_sec"] = _safe_mean(successful_durations)
    data["median_successful_imaging_action_duration_sec"] = _safe_median(successful_durations)
    data["mean_unsuccessful_imaging_action_duration_sec"] = _safe_mean(unsuccessful_durations)
    data["median_unsuccessful_imaging_action_duration_sec"] = _safe_median(unsuccessful_durations)
    data["mean_imaging_slew_time_sec"] = _safe_mean(imaging_slew_times)
    data["median_imaging_slew_time_sec"] = _safe_median(imaging_slew_times)
    data["mean_successful_imaging_slew_time_sec"] = _safe_mean(successful_slew_times)
    data["median_successful_imaging_slew_time_sec"] = _safe_median(successful_slew_times)
    data["mean_unsuccessful_imaging_slew_time_sec"] = _safe_mean(unsuccessful_slew_times)
    data["median_unsuccessful_imaging_slew_time_sec"] = _safe_median(unsuccessful_slew_times)
    data["mean_attempted_target_priority"] = _safe_mean(attempted_priorities)
    data["mean_successful_capture_priority"] = _safe_mean(successful_priorities)
    chosen_pairs = [
        (int(target_id), float(command_time))
        for target_id, command_time in zip(
            getattr(ss1_actions, "chosen_target_ids", []),
            getattr(ss1_actions, "imaging_times", []),
        )
        if command_time is not None and float(command_time) >= event_time
    ]
    hio_command_times = [t for target_id, t in chosen_pairs if target_id in hio_ids]
    shio_command_times = [t for target_id, t in chosen_pairs if target_id in shio_ids]
    successful_event_records = [
        record
        for record in imaging_attempt_records
        if record.get("success")
        and record.get("target_id") is not None
        and float(record.get("first_capture_time") or record.get("end_time") or -1.0)
        >= event_time
    ]
    hio_success_times = [
        float(record.get("first_capture_time") or record.get("end_time"))
        for record in successful_event_records
        if int(record["target_id"]) in hio_ids
    ]
    shio_success_times = [
        float(record.get("first_capture_time") or record.get("end_time"))
        for record in successful_event_records
        if int(record["target_id"]) in shio_ids
    ]
    data["hio_command_count_after_event"] = len(hio_command_times)
    data["shio_command_count_after_event"] = len(shio_command_times)
    data["hio_successful_capture_count_after_event"] = len(hio_success_times)
    data["shio_successful_capture_count_after_event"] = len(shio_success_times)
    data["time_to_first_hio_command_sec"] = _safe_min(hio_command_times)
    data["time_to_first_shio_command_sec"] = _safe_min(shio_command_times)
    data["time_to_first_hio_success_sec"] = _safe_min(hio_success_times)
    data["time_to_first_shio_success_sec"] = _safe_min(shio_success_times)
    data["reimage_count"] = int(getattr(env.rewarder, "reimage_count", 0))
    cooldown_until_by_id = getattr(reward_data, "cooldown_until_by_id", {})
    pending_by_id = getattr(reward_data, "pending_image_records_by_id", {})
    active_cooldown_ids = {
        int(target_id)
        for target_id, cooldown_until in cooldown_until_by_id.items()
        if episode_duration < float(cooldown_until)
    }
    pending_target_ids = {
        int(target_id) for target_id, records in pending_by_id.items() if records
    }
    temporarily_ineligible_ids = set(active_cooldown_ids)
    if getattr(reward_data, "hide_pending_targets", True):
        temporarily_ineligible_ids.update(pending_target_ids)
    data["cooldown_target_count_legacy"] = len(cooldown_until_by_id)
    data["cooldown_target_count"] = len(active_cooldown_ids)
    data["pending_verification_target_count"] = len(pending_target_ids)
    data["temporarily_ineligible_target_count"] = len(temporarily_ineligible_ids)

    if getattr(ss1_actions, "chosen_target_priority", None):
        chosen_priority_mean = float(np.mean(ss1_actions.chosen_target_priority))
        chosen_priority_std = float(np.std(ss1_actions.chosen_target_priority))
        chosen_priority_max = float(np.max(ss1_actions.chosen_target_priority))
        data["std_chosen_target_priority"] = chosen_priority_std
        data["max_chosen_target_priority"] = chosen_priority_max
        data["mean_chosen_target_priority"] = chosen_priority_mean
    else:
        data["mean_chosen_target_priority"] = -1.0
        data["std_chosen_target_priority"] = -1.0
        data["max_chosen_target_priority"] = -1.0

    if getattr(ss1_actions, "chosen_target_illumination_status", None):
        illumination = ss1_actions.chosen_target_illumination_status
        data["mean_target_illumination_status"] = float(np.mean(illumination))
        data["num_target_above_illumination_threshold"] = sum(ill > 0.5 for ill in illumination)
        data["num_target_below_illumination_threshold"] = sum(ill <= 0.5 for ill in illumination)
    else:
        data["mean_target_illumination_status"] = -1.0

    if getattr(ss1_actions, "ever_visible", None):
        data["target_ever_visible_fraction"] = (
            len(ss1_actions.ever_visible) / max(1, active_n_targets)
        )
    else:
        data["target_ever_visible_fraction"] = -1.0

    data["total_downlinks"] = int(getattr(env.rewarder, "total_downlinks", 0))
    data["useful_downlinks"] = int(getattr(env.rewarder, "useful_downlinks", 0))
    data["failed_downlinks"] = int(getattr(env.rewarder, "failed_downlinks", 0))
    data["pending_images"] = int(getattr(env.rewarder, "pending_images", 0))

    return data


def sat_metrics_callback(env, satellite):
    data = {}
    if satellite.name == "SS1":
        data["RW_norm"] = np.linalg.norm(satellite.dynamics.wheel_speeds)
        data["RW1"] = satellite.dynamics.wheel_speeds[0]
        data["RW2"] = satellite.dynamics.wheel_speeds[1]
        data["RW3"] = satellite.dynamics.wheel_speeds[2]
        data["battery_charge_fraction"] = satellite.dynamics.battery_charge_fraction
        data["storage_level_fraction"] = satellite.dynamics.storage_level_fraction
    else:
        data["RW_norm"] = 0.0
        data["RW1"] = 0.0
        data["RW2"] = 0.0
        data["RW3"] = 0.0
        data["battery_charge_fraction"] = 0.0
        data["storage_level_fraction"] = 0.0
    return data


if __name__ == "__main__":
    # Local defaults are chosen so you can press Run/Play in the IDE and quickly
    # see that Ray, Basilisk, the GAT module, and optional W&B setup all start.
    # The cluster sbatch files override these through BSK_RL_* environment vars.
    sim_cfg = SimConfig(
        n_targets=_env_int("BSK_RL_N_TARGETS", 100),  # local: 100 catalog targets; cluster: override if sweeping scale
        n_targets_ahead=_env_int("BSK_RL_N_TARGETS_AHEAD", 10),  # local/cluster: GNN action count = 10 candidates
        imaging_duration=_env_float("BSK_RL_IMAGING_DURATION", 300.0),  # max action duration before fast hold gate stops it
        extra_time_factor=_env_float("BSK_RL_EXTRA_TIME_FACTOR", 1.5),  # episode length multiplier
        obs_v=9.0,  # v9: sat block first, then priority/rel-vel-H target chunks
        just_imaging=False,
        verify_image_quality_on_downlink=True,
        hide_pending_targets=True,
        dynamic_priority_event_enabled=_env_bool("BSK_RL_DYNAMIC_PRIORITY_EVENT", True),
        dynamic_priority_event_time_sec=_env_optional_float(
            "BSK_RL_DYNAMIC_PRIORITY_EVENT_TIME_SEC"
        ),
        dynamic_priority_event_fraction=_env_float(
            "BSK_RL_DYNAMIC_PRIORITY_EVENT_FRACTION", 0.5
        ),
        hio_count=_env_int("BSK_RL_HIO_COUNT", 5),
        hio_priority=_env_float("BSK_RL_HIO_PRIORITY", 5.0),
        shio_count=_env_int("BSK_RL_SHIO_COUNT", 3),
        shio_priority=_env_float("BSK_RL_SHIO_PRIORITY", 10.0),
        dynamic_priority_event_seed=_env_optional_int(
            "BSK_RL_DYNAMIC_PRIORITY_EVENT_SEED"
        ),
    )

    n_targets = sim_cfg.n_targets
    n_targets_ahead = sim_cfg.n_targets_ahead
    imaging_duration = sim_cfg.imaging_duration
    total_time = sim_cfg.total_time
    randomize_n_targets = _env_bool("BSK_RL_RANDOMIZE_N_TARGETS", False)
    n_targets_min = _env_int("BSK_RL_N_TARGETS_MIN", n_targets)
    n_targets_max = _env_int("BSK_RL_N_TARGETS_MAX", n_targets)
    if randomize_n_targets:
        if n_targets_min < sim_cfg.hio_count + sim_cfg.shio_count:
            raise ValueError("BSK_RL_N_TARGETS_MIN must cover HIO + SHIO targets.")
        if n_targets_min > n_targets_max:
            raise ValueError("BSK_RL_N_TARGETS_MIN cannot exceed BSK_RL_N_TARGETS_MAX.")
        n_targets = n_targets_max

    class RandomCountSatellites(scene.RandomSatellites):
        def __init__(
            self,
            *args,
            n_targets: int,
            n_targets_min: int,
            n_targets_max: int,
            **kwargs,
        ):
            self.n_targets_min = int(n_targets_min)
            self.n_targets_max = int(n_targets_max)
            super().__init__(*args, n_targets=self.n_targets_max, **kwargs)

        def reset_overwrite_previous(self) -> None:
            super().reset_overwrite_previous()
            self.n_targets = int(
                np.random.randint(self.n_targets_min, self.n_targets_max + 1)
            )

    def make_rso_scenario():
        scenario_type = RandomCountSatellites if randomize_n_targets else scene.RandomSatellites
        scenario_kwargs = {}
        if randomize_n_targets:
            scenario_kwargs.update(
                n_targets_min=n_targets_min,
                n_targets_max=n_targets_max,
            )
        return scenario_type(
            "SS1",
            n_targets=n_targets,
            priority_mode=sim_cfg.priority_mode,
            priority_sum=sim_cfg.priority_sum,
            rescale_priorities_to_sum=sim_cfg.rescale_priorities_to_sum,
            priority_constant=sim_cfg.priority_constant,
            priority_uniform_low=sim_cfg.priority_uniform_low,
            priority_uniform_high=sim_cfg.priority_uniform_high,
            priority_gaussian_mean=sim_cfg.priority_gaussian_mean,
            priority_gaussian_std=sim_cfg.priority_gaussian_std,
            priority_min=sim_cfg.priority_min,
            priority_max=sim_cfg.priority_max,
            dynamic_priority_event_enabled=sim_cfg.dynamic_priority_event_enabled,
            dynamic_priority_event_time_sec=sim_cfg.dynamic_priority_event_time_sec,
            dynamic_priority_event_fraction=sim_cfg.dynamic_priority_event_fraction,
            hio_count=sim_cfg.hio_count,
            hio_priority=sim_cfg.hio_priority,
            shio_count=sim_cfg.shio_count,
            shio_priority=sim_cfg.shio_priority,
            dynamic_priority_event_seed=sim_cfg.dynamic_priority_event_seed,
            **scenario_kwargs,
        )

    def make_rso_rewarder():
        return data.RSOTargetImageReward(
            reimage_cooldown_orbits=sim_cfg.reimage_cooldown_orbits,
            verify_image_quality_on_downlink=sim_cfg.verify_image_quality_on_downlink,
            hide_pending_targets=sim_cfg.hide_pending_targets,
            image_quality_threshold=sim_cfg.image_quality_threshold,
        )

    def make_downlink_action(duration: float):
        return act.Downlink(
            duration=duration,
            variable_duration_downlink=sim_cfg.variable_duration_downlink,
            empty_storage_threshold_bits=sim_cfg.downlink_empty_threshold_bits,
        )

    class MyScanningSatellite(sats.AccessSatellite):
        observation_spec = [
            obs.SatProperties(
                dict(prop="storage_level_fraction"),
                dict(prop="battery_charge_fraction"),
                dict(prop="wheel_speeds_fraction"),
                dict(prop="s_hat_H", fn=obs.s_hat_H, norm=SUN_VECTOR_NORM),
            ),
            obs.Eclipse(norm=5700),
            obs.OpportunityProperties(
                dict(prop="opportunity_open", norm=5700.0),
                dict(prop="opportunity_close", norm=5700.0),
                type="ground_station",
                n_ahead_observe=2,
            ),
            # Target candidate blocks. The GAT applies the same target encoder to
            # each chunk, with satellite context available through the first block.
            obs.PolarisScTargetProperties(
                dict(prop="priority", norm=PRIORITY_NORM),
                dict(prop="target_elevation_angle", norm=90.0),
                dict(prop="rel_pos_vector_r_BR_H", norm=15960 * 1000),
                dict(prop="rel_vel_vector_v_BR_H", norm=REL_VEL_NORM_MPS),
                dict(prop="angle_to_target", norm=90.0),
                dict(prop="target_distance", norm=15960 * 1000),
                dict(prop="target_shadowFactor", norm=1.0),
                n_ahead_observe=n_targets_ahead,
            ),
        ]
        action_spec = [
            # GATModule expects all non-imaging logits first, followed by image logits.
            act.Charge(duration=300.0),
            make_downlink_action(300.0),
            act.Desat(duration=150.0),
            act.ImageRSO(
                n_ahead_image=n_targets_ahead,
                duration=imaging_duration,
                variable_duration_imaging=sim_cfg.variable_duration_imaging,
                min_pointing_hold_s=sim_cfg.min_pointing_hold_s,
                hold_mode=sim_cfg.hold_mode,
                require_illumination_during_hold=sim_cfg.require_illumination_during_hold,
                hold_illumination_threshold=sim_cfg.hold_illumination_threshold,
            ),
        ]
        dyn_type = dyn.ImagingSCDynModel
        fsw_type = fsw.ImagingSCFSWModel

    sat_args = {}
    sat_args["imageAttErrorRequirement"] = 0.0025
    sat_args["imageRateErrorRequirement"] = 0.01

    # Match the restricted-resource full-action BigNetwork setup.
    image_bits = 8e6 / 2
    image_storage_capacity = _env_float("BSK_RL_IMAGE_STORAGE_CAPACITY_IMAGES", 50.0)
    baseline_battery_ws = 500 * 3600
    battery_life_multiplier = _env_float("BSK_RL_BATTERY_LIFE_MULTIPLIER", 1.0)
    sat_args["dataStorageCapacity"] = image_storage_capacity * image_bits
    sat_args["storageInit"] = lambda: 0.0
    sat_args["instrumentBaudRate"] = 0.5 * 8e6
    sat_args["transmitterBaudRate"] = -0.5 * 8e6

    sat_args["batteryStorageCapacity"] = battery_life_multiplier * baseline_battery_ws
    sat_args["storedCharge_Init"] = lambda: np.random.uniform(0.8, 1.0) * battery_life_multiplier * baseline_battery_ws
    sat_args["basePowerDraw"] = -10.0
    sat_args["instrumentPowerDraw"] = -30.0
    sat_args["transmitterPowerDraw"] = -25.0
    sat_args["thrusterPowerDraw"] = -80.0
    sat_args["panelArea"] = 1.0

    sat_args["disturbance_vector"] = lambda: np.random.normal(scale=0.000, size=3)
    sat_args["maxWheelSpeed"] = 6000.0
    sat_args["wheelSpeeds"] = lambda: np.random.uniform(-500, 500, 3)
    sat_args["desatAttitude"] = "sun"

    curriculum_enabled = _env_bool("BSK_RL_ALPHA_CURRICULUM", False)
    downlink_bonus = _env_float("BSK_RL_DOWNLINK_BONUS", 0.2)
    curriculum_start_alpha = _env_float(
        "BSK_RL_ALPHA_CURRICULUM_START", downlink_bonus
    )
    curriculum_end_alpha = _env_float("BSK_RL_ALPHA_CURRICULUM_END", downlink_bonus)
    curriculum_power = _env_float("BSK_RL_ALPHA_CURRICULUM_POWER", 1.0)
    if curriculum_enabled:
        downlink_bonus = curriculum_start_alpha
    if not 0.0 <= downlink_bonus <= 1.0:
        raise ValueError("BSK_RL_DOWNLINK_BONUS must be in [0, 1].")
    if not 0.0 <= curriculum_start_alpha <= 1.0:
        raise ValueError("BSK_RL_ALPHA_CURRICULUM_START must be in [0, 1].")
    if not 0.0 <= curriculum_end_alpha <= 1.0:
        raise ValueError("BSK_RL_ALPHA_CURRICULUM_END must be in [0, 1].")
    imaging_bonus = 1.0 - downlink_bonus
    reward_split_tag = os.environ.get(
        "BSK_RL_REWARD_SPLIT_TAG", _reward_split_tag(downlink_bonus)
    )
    alpha_tag = os.environ.get("BSK_RL_ALPHA_TAG", _alpha_tag(downlink_bonus))

    sat_args["downlink_bonus"] = downlink_bonus
    sat_args["imaging_bonus"] = imaging_bonus
    sat_args["eclipse_threshold_for_imaging"] = 0.5
    sat_args["eclipse_threshold_for_reward"] = 0.5
    sat_args["empty_downlink_penalty"] = -1

    class MyTargetSatellite(sats.Satellite):
        observation_spec = [obs.Time()]
        action_spec = [act.Drift(duration=total_time)]
        dyn_type = dyn.BasicTargetDynamicsModel
        fsw_type = fsw.BasicTargetFSWModel

    R_E = 6371e3
    D2R = macros.D2R
    DEFAULT_ALT_BOUNDS = {
        "LEO": (400e3, 2000e3),
        "MEO": (2000e3, 35000e3),
        "GEO": (35786e3 - 300e3, 35786e3 + 300e3),
    }

    def _sample_for_regime(
        regime: str,
        altitude_bounds: dict[str, tuple[float, float]],
        min_perigee_alt: float,
    ) -> orbitalMotion.ClassicElements:
        oe = orbitalMotion.ClassicElements()
        h_min, h_max = altitude_bounds[regime]
        h = np.random.uniform(h_min, h_max)
        a = R_E + h

        if regime == "LEO":
            e = np.random.uniform(0.0, 0.02)
            while a * (1 - e) < (R_E + min_perigee_alt):
                e = np.random.uniform(0.0, 0.02)
            i_deg = np.random.uniform(0.0, 180.0)
        elif regime == "MEO":
            e = np.random.uniform(0.0, 0.10)
            while a * (1 - e) < (R_E + min_perigee_alt):
                e = np.random.uniform(0.0, 0.10)
            i_deg = np.random.uniform(0.0, 120.0)
        elif regime == "GEO":
            e = np.random.uniform(0.0, 0.0015)
            i_deg = np.random.uniform(0.0, 15.0)
            if a * (1 - e) < (R_E + min_perigee_alt):
                e = 0.0
        else:
            raise ValueError(f"Unknown orbit regime '{regime}'")

        oe.a = a
        oe.e = e
        oe.i = i_deg * D2R
        oe.Omega = np.random.uniform(0.0, 360.0) * D2R
        oe.omega = np.random.uniform(0.0, 360.0) * D2R
        oe.f = np.random.uniform(0.0, 360.0) * D2R
        return oe

    def custom_oe_randomizer(
        regime: str = "LEO",
        mix_weights: dict[str, float] | None = None,
        altitude_bounds: dict[str, tuple[float, float]] | None = None,
        min_perigee_alt: float = 400e3,
    ) -> orbitalMotion.ClassicElements:
        if altitude_bounds is None:
            altitude_bounds = DEFAULT_ALT_BOUNDS
        if regime.lower() == "mixed":
            regimes = ["LEO", "MEO", "GEO"]
            if mix_weights is None:
                probs = np.array([0.6, 0.3, 0.1])
            else:
                probs = np.array([mix_weights.get(r, 0.0) for r in regimes], dtype=float)
                if probs.sum() <= 0:
                    raise ValueError("mix_weights must include positive weights.")
                probs = probs / probs.sum()
            regime = np.random.choice(regimes, p=probs)
        return _sample_for_regime(regime.upper(), altitude_bounds, min_perigee_alt)

    def _random_simplex_mix_weights() -> dict[str, float]:
        leo = float(np.random.uniform(0.0, 1.0))
        meo = float(np.random.uniform(0.0, 1.0 - leo))
        geo = float(1.0 - leo - meo)
        return {"LEO": leo, "MEO": meo, "GEO": geo}

    def _make_target_oe_randomizer(
        *,
        target_env: str,
        fixed_mix_weights: dict[str, float] | None,
        randomize_mix_weights: bool,
        exact_mix_counts: bool,
        targets_per_catalog: int,
    ) -> Callable[[], orbitalMotion.ClassicElements]:
        target_env = target_env.lower()
        if target_env not in {"leo", "mixed"}:
            raise ValueError("BSK_RL_TARGET_ENV must be 'leo' or 'mixed'.")
        if target_env == "leo":
            return lambda: custom_oe_randomizer(regime="LEO")
        if exact_mix_counts and randomize_mix_weights:
            raise ValueError(
                "BSK_RL_EXACT_MIX_COUNTS and BSK_RL_RANDOMIZE_MIX_WEIGHTS "
                "cannot both be enabled."
            )

        state = {
            "remaining": 0,
            "mix_weights": fixed_mix_weights or {"LEO": 0.6, "MEO": 0.3, "GEO": 0.1},
            "regime_queue": [],
        }

        def randomizer() -> orbitalMotion.ClassicElements:
            if exact_mix_counts:
                if not state["regime_queue"]:
                    counts = _fixed_regime_counts(
                        targets_per_catalog, state["mix_weights"]
                    )
                    state["regime_queue"] = [
                        regime
                        for regime in ("LEO", "MEO", "GEO")
                        for _ in range(counts[regime])
                    ]
                    np.random.shuffle(state["regime_queue"])
                regime = state["regime_queue"].pop()
                return custom_oe_randomizer(regime=regime)

            if state["remaining"] <= 0:
                if randomize_mix_weights:
                    state["mix_weights"] = _random_simplex_mix_weights()
                state["remaining"] = max(1, targets_per_catalog)
            state["remaining"] -= 1
            return custom_oe_randomizer(
                regime="mixed",
                mix_weights=state["mix_weights"],
            )

        return randomizer

    target_env = os.environ.get("BSK_RL_TARGET_ENV", "LEO").strip().lower()
    randomize_mix_weights = _env_bool("BSK_RL_RANDOMIZE_MIX_WEIGHTS", False)
    fixed_mix_weights = _env_mix_weights("BSK_RL_MIX_WEIGHTS")
    exact_mix_counts = _env_bool("BSK_RL_EXACT_MIX_COUNTS", False)
    target_oe_randomizer = _make_target_oe_randomizer(
        target_env=target_env,
        fixed_mix_weights=fixed_mix_weights,
        randomize_mix_weights=randomize_mix_weights,
        exact_mix_counts=exact_mix_counts,
        targets_per_catalog=n_targets,
    )

    # Keep target satellites passive/alive in this training entrypoint. The
    # scanner is the only learned agent, and killing all target sats at t=0 adds
    # a lot of Ray log noise without changing the target-selection objective.
    target_args = dict(
        oe=target_oe_randomizer,
        batteryStorageCapacity=1.0,
        storedCharge_Init=1.0,
        basePowerDraw=0.0,
    )

    sat = MyScanningSatellite(name="SS1", sat_args=sat_args)
    targets = [
        MyTargetSatellite(name=f"target_{i}", sat_args=target_args)
        for i in range(n_targets)
    ]
    all_sat = [sat] + targets

    default_job_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    job_index = _env_int(
        "BSK_RL_JOB_INDEX",
        _env_int("SLURM_ARRAY_TASK_ID", default_job_index),
    )
    on_cluster = bool(os.environ.get("SLURM_JOB_ID"))
    default_n_envs = (
        get_available_cores() - 4 if on_cluster else get_available_cores() - 6
    )  # local: cores minus 6 like copy_updated_train_Polaris.py; cluster: cores minus 4
    default_batch_multiplier = 150 if on_cluster else 32  # local: 32; cluster: 150 unless sbatch overrides
    default_total_timesteps = 20_000_000 if on_cluster else 10_000  # local startup check; cluster full train
    default_checkpoint_frequency = 3 if on_cluster else 1  # local: checkpoint every iter; cluster: less often
    n_envs = max(1, _env_int("BSK_RL_NUM_ENVS", default_n_envs))
    batch_multiplier = _env_int("BSK_RL_BATCH_MULTIPLIER", default_batch_multiplier)
    batch_size = max(
        PPO_MIN_TRAIN_BATCH_SIZE,
        int(batch_multiplier * n_envs),
    )  # keep train_batch_size >= RLlib's default sgd_minibatch_size
    total_timesteps = _env_int("BSK_RL_TOTAL_TIMESTEPS", default_total_timesteps)
    continuation_source = os.environ.get("BSK_RL_CONTINUE_FROM", "").strip()
    continuation = (
        _resolve_continuation_source(continuation_source)
        if continuation_source
        else None
    )
    train_timeout = _env_optional_float("BSK_RL_TRAIN_TIMEOUT_SEC")
    disable_timestep_limit = _env_bool(
        "BSK_RL_DISABLE_TIMESTEP_LIMIT",
        bool(continuation is not None and train_timeout is not None),
    )
    train_step_limit = None if disable_timestep_limit else total_timesteps
    checkpoint_frequency = _env_int(
        "BSK_RL_CHECKPOINT_FREQUENCY", default_checkpoint_frequency
    )
    curriculum_ramp_steps = _env_int(
        "BSK_RL_ALPHA_CURRICULUM_RAMP_STEPS", total_timesteps
    )
    # With one env per RLlib env runner, this local increment lets each worker
    # finish its own ramp at about the same time as the global callback ramp.
    curriculum_ramp_steps_per_env = _env_int(
        "BSK_RL_ALPHA_CURRICULUM_RAMP_STEPS_PER_ENV",
        max(1, int(np.ceil(curriculum_ramp_steps / max(1, n_envs)))),
    )
    curriculum_config = {
        "enabled": curriculum_enabled,
        "start_alpha": curriculum_start_alpha,
        "end_alpha": curriculum_end_alpha,
        "power": curriculum_power,
        "ramp_steps": curriculum_ramp_steps,
        "ramp_steps_per_env": curriculum_ramp_steps_per_env,
    }

    run_tag = (
        f"amos2026_{target_env.upper()}_GAT_fullActions_{reward_split_tag}_{batch_size}batch_"
        "restrictedResources_obs-v9_hold10s_reimage2orb_prioritySum100"
    )
    if target_env == "mixed" and randomize_mix_weights:
        run_tag = run_tag.replace(
            "_restrictedResources_", "_randomMixLEOMEOFirst_restrictedResources_"
        )
    if randomize_n_targets:
        run_tag = run_tag.replace(
            "_restrictedResources_",
            f"_random{n_targets_min}to{n_targets_max}targets_restrictedResources_",
        )
    if continuation is None:
        model_name = f"{run_tag}.out_{job_index}"
        output_dir = _default_output_root() / f"{run_tag}_{time.time()}"  # local: ~/rllib_results/...; cluster: /scratch/alpine/$USER/rllib_results/...
    else:
        continue_suffix = os.environ.get(
            "BSK_RL_CONTINUE_SUFFIX", f"continue_{_timestamp_tag()}"
        )
        source_run_name = Path(continuation["source_run_directory"]).name
        model_name = os.environ.get(
            "BSK_RL_CONTINUE_MODEL_NAME", f"{source_run_name}_{continue_suffix}"
        )
        explicit_continue_output = os.environ.get("BSK_RL_CONTINUE_OUTPUT_DIR")
        if explicit_continue_output:
            output_dir = Path(explicit_continue_output).expanduser()
        else:
            output_dir = _default_output_root() / f"{run_tag}_{continue_suffix}"
    ray_tmpdir = _default_ray_tmpdir()  # local: /tmp/bskrl_0; cluster: /tmp/bskray_${SLURM_JOB_ID}_...
    output_dir.mkdir(parents=True, exist_ok=True)
    ray_tmpdir.mkdir(parents=True, exist_ok=True)
    run_directory = output_dir / model_name

    inspector_model_config = gat_model_config(n_targets_ahead)  # n_targets_ahead must match ImageRSO/observation count
    inspector_rl_module_spec = RLModuleSpec(
        module_class=GATModule,
        model_config_dict=inspector_model_config,
    )

    base_lr = 1e-5 #0.00033003435881682255  # current GNN hyperparameter starting point; not yet AMOS-tuned
    env_args = dict(
        satellites=[all_sat],
        scenario=[make_rso_scenario()],
        rewarder=[make_rso_rewarder()],
        world_type=[world.GroundStationWorldModel],  # ImagingSCDynModel expects this AMOS world type.
        time_limit=[total_time],
        failure_penalty=[-100.0],
        terminate_on_time_limit=[False],
        generate_obs_retasking_only=[False],
        episode_data_callback=[env_metrics_callback],
        satellite_data_callback=[sat_metrics_callback],
    )
    if curriculum_enabled:
        env_args["env_wrapper"] = [
            _make_alpha_curriculum_wrapper(
                start_alpha=curriculum_start_alpha,
                end_alpha=curriculum_end_alpha,
                ramp_steps_per_env=curriculum_ramp_steps_per_env,
                power=curriculum_power,
            )
        ]

    jobs = build_job_array(
        training_args=dict(
            lr=[[[0, base_lr], [40000, base_lr / 16.749479444886223]]],
            gamma=[0.9915045428565076],
            train_batch_size=[batch_size],
            num_sgd_iter=[30],
            lambda_=[0.8713548569911232],
            use_kl_loss=[False],
            clip_param=[0.14701727973480344],
            grad_clip=[0.3104924935285628],
            entropy_coeff=[0.023694512589767867],
        ),
        env_args=env_args,
    )

    print(f"n_envs={n_envs}; batch_size={batch_size}; torch_threads={_TORCH_THREADS}")
    print(f"Tensorboard logging: tensorboard --logdir {output_dir}")
    print(f"Ray temp dir: {ray_tmpdir}")
    print(f"Total timesteps: {train_step_limit if train_step_limit is not None else 'disabled'}")
    if train_timeout is not None:
        print(f"Train timeout: {train_timeout} seconds")
    if continuation is not None:
        print(f"Continuation source run: {continuation['source_run_directory']}")
        print(f"Continuation source checkpoint: {continuation['source_checkpoint']}")
        print(f"Continuation copied run: {run_directory}")
    print(f"Running job {job_index}: {job_index + 1} of {len(jobs)}")

    job_args = jobs[job_index]
    continuation_copy = None
    dry_run = _env_bool("BSK_RL_DRY_RUN", False)
    if continuation is not None and not dry_run:
        continuation_copy = _copy_continuation_run(
            continuation=continuation,
            destination_run_directory=run_directory,
            overwrite=_env_bool("BSK_RL_CONTINUE_OVERWRITE_COPY", False),
        )
        print(
            "[continue] copied original run to "
            f"{continuation_copy['copied_run_directory']}",
            flush=True,
        )
        print(
            "[continue] will restore from copied checkpoint "
            f"{continuation_copy['copied_checkpoint']}",
            flush=True,
        )

    run_cfg = {
        "sim": asdict(sim_cfg),
        "observation_layout": {
            "obs_sat_dim": OBS_SAT_DIM,
            "target_features_per_target": TARGET_FEATURES_PER_TARGET,
            "n_targets_ahead": n_targets_ahead,
            "non_imaging_actions": NON_IMAGING_ACTIONS,
            "priority_norm": PRIORITY_NORM,
            "relative_velocity_norm_mps": REL_VEL_NORM_MPS,
            "sun_vector_norm": SUN_VECTOR_NORM,
            "satellite_chunk_order": [
                "storage_level_fraction",
                "battery_charge_fraction",
                "wheel_speeds_fraction[0:3]",
                "s_hat_H[0:3]",
                "next_eclipse_start",
                "next_eclipse_end",
                "ground_station_0_open",
                "ground_station_0_close",
                "ground_station_1_open",
                "ground_station_1_close",
            ],
            "target_chunk_order": [
                "priority",
                "target_elevation_angle",
                "rel_pos_vector_r_BR_H[0:3]",
                "rel_vel_vector_v_BR_H[0:3]",
                "angle_to_target",
                "target_distance",
                "target_shadowFactor",
            ],
        },
        "action_layout": {
            "non_imaging_actions": NON_IMAGING_ACTIONS,
            "flat_action_order": [
                "0: charge",
                "1: downlink",
                "2: desat",
                "3..: image target candidate",
            ],
        },
        "reward_split": {
            "downlink_bonus": downlink_bonus,
            "imaging_bonus": imaging_bonus,
            "tag": reward_split_tag,
            "alpha_tag": alpha_tag,
            "curriculum_enabled": curriculum_enabled,
        },
        "target_regime": {
            "target_env": target_env,
            "randomize_mix_weights": randomize_mix_weights,
            "fixed_mix_weights": fixed_mix_weights,
            "exact_mix_counts": exact_mix_counts,
            "exact_regime_counts": (
                _fixed_regime_counts(
                    n_targets,
                    fixed_mix_weights
                    or {"LEO": 0.6, "MEO": 0.3, "GEO": 0.1},
                )
                if target_env == "mixed" and exact_mix_counts
                else None
            ),
            "random_mix_sampling": (
                "LEO=x~Uniform(0,1); "
                "MEO=y~Uniform(0,1-x); "
                "GEO=1-x-y"
                if target_env == "mixed" and randomize_mix_weights
                else None
            ),
        },
        "target_count": {
            "capacity_n_targets": n_targets,
            "randomize_n_targets": randomize_n_targets,
            "n_targets_min": n_targets_min if randomize_n_targets else n_targets,
            "n_targets_max": n_targets_max if randomize_n_targets else n_targets,
        },
        "gat_model_config": inspector_model_config,
        "job_args": sanitize_np(
            {
                **job_args,
                "env_args": {
                    **job_args["env_args"],
                    **(
                        {"env_wrapper": "WrapperCurriculumLearning"}
                        if "env_wrapper" in job_args["env_args"]
                        else {}
                    ),
                },
            }
        ),
        "curriculum": curriculum_config,
        "cluster": {
            "on_cluster": on_cluster,
            "job_index": job_index,
            "n_envs": n_envs,
            "batch_multiplier": batch_multiplier,
            "batch_size": batch_size,
            "total_timesteps": train_step_limit,
            "configured_total_timesteps": total_timesteps,
            "timestep_limit_disabled": disable_timestep_limit,
            "train_timeout_sec": train_timeout,
            "battery_life_multiplier": battery_life_multiplier,
            "image_storage_capacity_images": image_storage_capacity,
            "ray_tmpdir": str(ray_tmpdir),
            "torch_threads": _TORCH_THREADS,
        },
        "wandb": {
            "enabled": _env_bool("BSK_RL_USE_WANDB", True),
            "key_path": str(_wandb_key_path()),
            "project": os.environ.get("BSK_RL_WANDB_PROJECT", "amos2026-bsk-rl"),
            "group": os.environ.get(
                "BSK_RL_WANDB_GROUP", "polaris-gat-full-actions-obs-v9"
            ),
        },
        "continuation": None
        if continuation is None
        else {
            "source_path": str(continuation["source_path"]),
            "source_run_directory": str(continuation["source_run_directory"]),
            "source_checkpoint": str(continuation["source_checkpoint"]),
            "copied_run_directory": (
                str(continuation_copy["copied_run_directory"])
                if continuation_copy is not None
                else str(run_directory)
            ),
            "copied_checkpoint": (
                str(continuation_copy["copied_checkpoint"])
                if continuation_copy is not None
                else None
            ),
            "start_iteration": continuation["start_iteration"],
        },
    }
    with open(output_dir / f"{model_name}_config.yaml", "w") as file:
        yaml.dump(run_cfg, file)

    if dry_run:
        print("Dry run requested via BSK_RL_DRY_RUN=1; configuration written.")
        raise SystemExit(0)

    wandb_logger = _maybe_init_wandb(model_name, run_cfg)  # local: optional; cluster W&B sbatch: required

    train_model(
        model_name=model_name,
        output_directory=output_dir,
        inspector_rl_module_spec=inspector_rl_module_spec,
        checkpoint_frequency=checkpoint_frequency,
        checkpoints_to_keep=3,
        total_timesteps=train_step_limit,
        reload_frequency=500_000,
        n_envs=n_envs,
        temp_dir=str(ray_tmpdir),
        wandb_logger=wandb_logger,
        curriculum_config=curriculum_config,
        restore_checkpoint_dir=(
            continuation_copy["copied_checkpoint"]
            if continuation_copy is not None
            else None
        ),
        start_iteration=(
            int(continuation["start_iteration"]) if continuation is not None else 0
        ),
        timeout=train_timeout,
        **job_args,
    )
