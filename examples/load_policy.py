from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core import DEFAULT_MODULE_ID
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.utils.numpy import convert_to_numpy, softmax
import numpy as np
import glob
from pathlib import Path
from typing import Callable
import os
import time

# --- FLAG TO QUICKLY CHANGE ---
# Set to 'latest', 'smallest', or 'best' to control which policy is loaded.
CHECKPOINT_SELECTION_MODE = 'best'

def find_latest_checkpoint(checkpoint_path_dir: Path, mode: str = 'latest') -> Path:
    """
    Find a specific checkpoint in a directory based on the given mode.
    The function name is kept for consistency with the original structure.

    Args:
        checkpoint_path_dir: The path to the directory containing checkpoints.
        mode: The selection mode. Can be 'latest', 'smallest', or 'best'.

    Returns:
        The path to the selected checkpoint directory.

    Raises:
        ValueError: If no suitable checkpoints are found.
    """
    if mode not in ['latest', 'smallest', 'best']:
        raise ValueError(f"Invalid mode '{mode}'. Choose from 'latest', 'smallest', or 'best'.")

    # Use os.path.join for robust path construction and glob recursively
    glob_pattern = os.path.join(str(checkpoint_path_dir), "**", "checkpoint_*")
    all_checkpoints = glob.glob(glob_pattern, recursive=True)

    if not all_checkpoints:
        raise ValueError(f"No checkpoint directories found in {checkpoint_path_dir}")


    # For 'latest' or 'smallest', parse only the numeric checkpoints
    numeric_checkpoints = []
    for path_str in all_checkpoints:
        folder_name = os.path.basename(path_str)
        try:
            # Safely attempt to convert the last part of the folder name to an integer
            checkpoint_num = int(folder_name.split('_')[-1])
            numeric_checkpoints.append((checkpoint_num, path_str))
        except ValueError:
            # This line is the fix: it gracefully skips non-numeric folders like 'checkpoint_best'
            continue

    if not numeric_checkpoints:
        raise ValueError(f"No numeric checkpoints found in {checkpoint_path_dir}")

    # Sort checkpoints by their number
    numeric_checkpoints.sort(key=lambda item: item[0])

    # Handle 'best' mode first
    if mode == 'best':
        for path in all_checkpoints:
            if os.path.basename(path) == 'checkpoint_best':
                return Path(path)
        print("'best' didn't exist... choosing smallest (renamed from best)")
        return Path(numeric_checkpoints[0][1])
        # raise ValueError(f"Mode was 'best' but 'checkpoint_best' not found in {checkpoint_path_dir}")


    # Return the correct path based on the mode
    if mode == 'smallest':
        selected_path = numeric_checkpoints[0][1]
    else:  # mode == 'latest'
        selected_path = numeric_checkpoints[-1][1]

    return Path(selected_path)

def _extract_linear_layers(named_parameters: dict[str, torch.Tensor], prefix: str) -> list[list[int]]:
    layers = []
    layer_idx = 0
    while True:
        weight_key = f"{prefix}.{layer_idx}.weight"
        if weight_key not in named_parameters:
            break
        weight = named_parameters[weight_key]
        layers.append([int(weight.shape[1]), int(weight.shape[0])])
        layer_idx += 2
    return layers


def summarize_rl_module(rl_module: RLModule) -> dict:
    named_parameters = dict(rl_module.named_parameters())
    total_params = sum(int(param.numel()) for param in named_parameters.values())
    trainable_params = sum(
        int(param.numel()) for param in named_parameters.values() if param.requires_grad
    )

    actor_layers = _extract_linear_layers(
        named_parameters,
        "encoder.actor_encoder.net.mlp",
    )
    critic_layers = _extract_linear_layers(
        named_parameters,
        "encoder.critic_encoder.net.mlp",
    )
    policy_head_layers = _extract_linear_layers(named_parameters, "pi.net.mlp")
    value_head_layers = _extract_linear_layers(named_parameters, "vf.net.mlp")

    return {
        "module_class": type(rl_module).__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "actor_encoder_layers": actor_layers,
        "critic_encoder_layers": critic_layers,
        "policy_head_layers": policy_head_layers,
        "value_head_layers": value_head_layers,
        "parameter_shapes": {
            name: list(param.shape) for name, param in named_parameters.items()
        },
        "module_repr": str(rl_module),
    }


def load_policy(
    policy_path_general: Path,
    policy_mode,
    profile_inference: bool = False,
) -> Callable:
    """Load a PyTorch policy from a saved model.

    Args:
        policy_path_general: The path to the saved model.
    Returns:
        A function that takes observations and returns actions.
    """
    # The selection mode is passed here from the global flag
    path_checkpoint = find_latest_checkpoint(policy_path_general, mode=policy_mode)
    print(f"✅ Loading policy from '{policy_mode}' checkpoint: {path_checkpoint}")

    rl_module = RLModule.from_checkpoint(
        path_checkpoint / "learner_group" / "learner" / "rl_module" / "inspector",
    )
    model_summary = summarize_rl_module(rl_module)
    inference_times_ns: list[int] = []

    def policy(
        obs: list[float],
        deterministic: bool = True,
    ) -> int:
        """Policy function that takes observations and returns actions.

        Args:
            obs: A list of observations.
            deterministic: If True, use argmax for action selection; otherwise, sample from the action distribution.
        Returns:
            An integer representing the selected action.
        """
        start_ns = time.perf_counter_ns() if profile_inference else None
        obs = np.asarray(obs, dtype=np.float32)
        input_dict = {Columns.OBS: torch.from_numpy(obs).unsqueeze(0)}

        rl_module_out = rl_module.forward_inference(input_dict)
        logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])
        if deterministic:
            action = np.argmax(logits[0])  # Use argmax for deterministic action
        else:
            action = np.random.choice(len(logits[0]), p=softmax(logits[0]))

        if start_ns is not None:
            inference_times_ns.append(time.perf_counter_ns() - start_ns)

        return action

    policy.inference_times_ns = inference_times_ns
    policy.checkpoint_path = path_checkpoint
    policy.model_summary = model_summary
    policy.profile_inference = profile_inference

    return policy
