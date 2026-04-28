from pathlib import Path
from typing import Callable
import glob
import os
import time

import numpy as np
import torch
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.numpy import convert_to_numpy, softmax

# --- FLAG TO QUICKLY CHANGE ---
# Set to "latest", "smallest", or "best" to control which policy is loaded.
CHECKPOINT_SELECTION_MODE = "best"


def find_latest_checkpoint(checkpoint_path_dir: Path, mode: str = "latest") -> Path:
    """Find a checkpoint in a directory based on the requested selection mode."""
    if mode not in ["latest", "smallest", "best"]:
        raise ValueError(
            f"Invalid mode '{mode}'. Choose from 'latest', 'smallest', or 'best'."
        )

    glob_pattern = os.path.join(str(checkpoint_path_dir), "**", "checkpoint_*")
    all_checkpoints = glob.glob(glob_pattern, recursive=True)

    if not all_checkpoints:
        raise ValueError(f"No checkpoint directories found in {checkpoint_path_dir}")

    numeric_checkpoints = []
    for path_str in all_checkpoints:
        folder_name = os.path.basename(path_str)
        try:
            checkpoint_num = int(folder_name.split("_")[-1])
            numeric_checkpoints.append((checkpoint_num, path_str))
        except ValueError:
            continue

    if not numeric_checkpoints:
        raise ValueError(f"No numeric checkpoints found in {checkpoint_path_dir}")

    numeric_checkpoints.sort(key=lambda item: item[0])

    if mode == "best":
        for path in all_checkpoints:
            if os.path.basename(path) == "checkpoint_best":
                return Path(path)
        print("'best' didn't exist... choosing smallest (renamed from best)")
        return Path(numeric_checkpoints[0][1])

    if mode == "smallest":
        selected_path = numeric_checkpoints[0][1]
    else:
        selected_path = numeric_checkpoints[-1][1]

    return Path(selected_path)


class LoadedPolicy:
    """Callable policy wrapper with lightweight timing and model metadata helpers."""

    def __init__(self, policy_path_general: Path, policy_mode: str) -> None:
        self.policy_root = Path(policy_path_general)
        self.policy_mode = policy_mode
        self.checkpoint_path = find_latest_checkpoint(
            self.policy_root,
            mode=policy_mode,
        )
        print(f"✅ Loading policy from '{policy_mode}' checkpoint: {self.checkpoint_path}")

        self.rl_module = RLModule.from_checkpoint(
            self.checkpoint_path / "learner_group" / "learner" / "rl_module" / "inspector",
        )
        self.inference_times_ms: list[float] = []

    def __call__(self, obs: list[float], deterministic: bool = True) -> int:
        """Run one policy inference and return the chosen discrete action."""
        start = time.perf_counter()
        obs = np.array(obs, dtype=np.float32)
        input_dict = {Columns.OBS: torch.from_numpy(obs).unsqueeze(0)}

        with torch.inference_mode():
            rl_module_out = self.rl_module.forward_inference(input_dict)
            logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])
            if deterministic:
                action = np.argmax(logits[0])
            else:
                action = np.random.choice(len(logits[0]), p=softmax(logits[0]))

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.inference_times_ms.append(float(elapsed_ms))
        return int(action)

    def reset_timing(self) -> None:
        self.inference_times_ms.clear()

    def timing_summary(self) -> dict:
        if not self.inference_times_ms:
            return {
                "count": 0,
                "mean_ms": None,
                "std_ms": None,
                "sample_std_ms": None,
                "min_ms": None,
                "max_ms": None,
                "median_ms": None,
                "p95_ms": None,
            }

        times = np.asarray(self.inference_times_ms, dtype=float)
        return {
            "count": int(times.size),
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times, ddof=0)),
            "sample_std_ms": float(np.std(times, ddof=1)) if times.size > 1 else 0.0,
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
        }

    def model_summary(self) -> dict:
        summary = {
            "policy_root": str(self.policy_root),
            "policy_mode": self.policy_mode,
            "checkpoint_path": str(self.checkpoint_path),
            "module_type": type(self.rl_module).__name__,
            "module_id": DEFAULT_MODULE_ID,
        }

        if not hasattr(self.rl_module, "named_parameters"):
            summary["total_parameters"] = None
            summary["trainable_parameters"] = None
            summary["parameter_tensors"] = []
            return summary

        parameter_tensors = []
        total_parameters = 0
        trainable_parameters = 0
        for name, parameter in self.rl_module.named_parameters():
            count = int(parameter.numel())
            total_parameters += count
            if parameter.requires_grad:
                trainable_parameters += count
            parameter_tensors.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "numel": count,
                    "requires_grad": bool(parameter.requires_grad),
                }
            )

        summary["total_parameters"] = total_parameters
        summary["trainable_parameters"] = trainable_parameters
        summary["parameter_tensors"] = parameter_tensors
        summary["module_repr"] = str(self.rl_module)
        return summary


def load_policy(policy_path_general: Path, policy_mode: str = CHECKPOINT_SELECTION_MODE) -> Callable:
    """Load a PyTorch policy from a saved RLlib model checkpoint."""
    return LoadedPolicy(policy_path_general, policy_mode=policy_mode)
