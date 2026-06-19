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
    checkpoint_path_dir = Path(checkpoint_path_dir).expanduser()
    direct_module = (
        checkpoint_path_dir
        / "learner_group"
        / "learner"
        / "rl_module"
        / "inspector"
    )
    if direct_module.is_dir():
        return checkpoint_path_dir

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


def load_policy(policy_path_general: Path, policy_mode) -> Callable:
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
        obs = np.array(obs, dtype=np.float32)
        input_dict = {Columns.OBS: torch.from_numpy(obs).unsqueeze(0)}

        rl_module_out = rl_module.forward_inference(input_dict)
        logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])
        if deterministic:
            action = np.argmax(logits[0])  # Use argmax for deterministic action
        else:
            action = np.random.choice(len(logits[0]), p=softmax(logits[0]))

        return action

    return policy
