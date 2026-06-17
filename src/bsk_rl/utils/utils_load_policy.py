from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core import DEFAULT_MODULE_ID
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.utils.numpy import convert_to_numpy, softmax
import numpy as np
import glob
from pathlib import Path
from typing import Callable


def find_latest_checkpoint(checkpoint_path_dir: Path) -> Path:
    """Find the latest checkpoint in a directory.

    Args:
        checkpoint_path_dir_path: The path to the directory containing checkpoints.

    Returns:
        The path to the latest checkpoint directory.

    Raises:
        ValueError: If no checkpoints are found in the directory.
    """

    checkpoints = glob.glob(
        str(checkpoint_path_dir) + "/**/checkpoint_*", recursive=True
    )
    if len(checkpoints) == 0:
        raise ValueError("No model to re-load and continue training")
    checkpoint_number_str = ""
    checkpoint_number_int = 0
    for checkpoint_number in checkpoints:
        checkpoint_folder_i = checkpoint_number.split("/")[-1]
        checkpoint_number_i = int(checkpoint_folder_i.split("_")[-1])
        if checkpoint_number_i > checkpoint_number_int:
            checkpoint_number_int = checkpoint_number_i
            checkpoint_number_str = checkpoint_folder_i.split("_")[-1]

    latest_checkpoint_dir = glob.glob(
        str(checkpoint_path_dir) + f"/**/checkpoint_{checkpoint_number_str}",
        recursive=True,
    )

    return Path(latest_checkpoint_dir[0])


def utils_load_policy(policy_path_general: Path) -> Callable:
    """Load a PyTorch policy from a saved model.

    Args:
        policy_path_general: The path to the saved model.
    Returns:
        A function that takes observations and returns actions.
    """

    path_checkpoint = find_latest_checkpoint(policy_path_general)
    print(f"Attempting to load policy from: {path_checkpoint}")

    rl_module = RLModule.from_checkpoint(
        path_checkpoint / "learner_group" / "learner" / "rl_module" / "inspector" ,
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