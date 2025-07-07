"""A collection of utilities at ``bsk_rl.utils.rllib``.

* :ref:`bsk_rl.utils.rllib.discounting` - For semi-MDP discounting with GAE.
* :ref:`bsk_rl.utils.rllib.callbacks` - For logging data at the end of each episode.

Two environments are added to the ``ray.tune.registry`` with this import. They are
``"SatelliteTasking-RLlib"`` and ``"ConstellationTasking-RLlib"``. These environments
are wrapped with the :func:`unpack_config` function to make them compatible with RLlib's
API, and they are wrapped with the :class:`EpisodeDataWrapper` to allow for data logging
at the end of each episode during training. These environments can be selected by name
when setting ``config.environment(env="SatelliteTasking-RLlib")``. Callback functions
that are arguments to :class:`EpisodeDataWrapper` can be set in the ``env_config`` dictionary.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import unsquash_action
from ray.tune.registry import register_env

from bsk_rl import ConstellationTasking, GeneralSatelliteTasking, SatelliteTasking
from bsk_rl.utils.rllib.callbacks import EpisodeDataParallelWrapper, EpisodeDataWrapper


def load_torch_mlp_policy(
    policy_path: Path, policy_name: str = "default_agent"
) -> Callable:
    """Load a PyTorch policy from a saved model.

    Args:
        policy_path: Path to the directory containing the policy checkpoint.
        policy_name: Name of the policy to load from the checkpoint.
    """
    rl_module = RLModule.from_checkpoint(
        Path(policy_path) / "learner_group" / "learner" / "rl_module" / policy_name
    )
    cat = rl_module.config.get_catalog()
    action_dist_cls = cat.get_action_dist_cls(framework="torch")

    def policy(obs: list[float], deterministic: bool = True) -> np.ndarray:
        """Policy function that takes observations and returns actions.

        Args:
            obs: Observation vector.
            deterministic: If True, use loc for action selection; otherwise, sample from the action distribution.
        """
        obs = np.array(obs, dtype=np.float32)
        action_dist_params = rl_module.forward_inference(
            dict(obs=torch.tensor(obs)[None, :])
        )
        action_dist = action_dist_cls.from_logits(
            action_dist_params[Columns.ACTION_DIST_INPUTS]
        )

        if deterministic:
            action_squashed = action_dist.to_deterministic().sample()
        else:
            action_squashed = action_dist.sample()

        action_squashed = convert_to_numpy(action_squashed[0])
        action = unsquash_action(action_squashed, rl_module.config.action_space)

        return action

    return policy


def unpack_config(env):
    """Create a wrapped version of an env class that unpacks env_config from Ray into kwargs.

    Necessary when setting

    .. code-block:: python

        config.environment(
            env=unpack_config(SatelliteTasking),
            env_config=env_args
        )

    which generates environments that look like

    .. code-block:: python

        SatelliteTasking(**env_args)

    since RLlib expects the environment to take a dictionary called ``kwargs`` instead
    of the actual arguments.

    """

    class UnpackedEnv(env):
        def __init__(self, env_config):
            super().__init__(**env_config)

    UnpackedEnv.__name__ = f"{env.__name__}_Unpacked"

    return UnpackedEnv


def _satellite_tasking_env_creator(env_config):
    if "episode_data_callback" in env_config:
        episode_data_callback = env_config.pop("episode_data_callback")
    else:
        episode_data_callback = None
    if "satellite_data_callback" in env_config:
        satellite_data_callback = env_config.pop("satellite_data_callback")
    else:
        satellite_data_callback = None

    return EpisodeDataWrapper(
        SatelliteTasking(**env_config),
        episode_data_callback=episode_data_callback,
        satellite_data_callback=satellite_data_callback,
    )


register_env("SatelliteTasking-RLlib", _satellite_tasking_env_creator)


def _constellation_tasking_env_creator(env_config):
    if "episode_data_callback" in env_config:
        episode_data_callback = env_config.pop("episode_data_callback")
    else:
        episode_data_callback = None
    if "satellite_data_callback" in env_config:
        satellite_data_callback = env_config.pop("satellite_data_callback")
    else:
        satellite_data_callback = None

    return ParallelPettingZooEnv(
        EpisodeDataParallelWrapper(
            ConstellationTasking(**env_config),
            episode_data_callback=episode_data_callback,
            satellite_data_callback=satellite_data_callback,
        )
    )


register_env("ConstellationTasking-RLlib", _constellation_tasking_env_creator)


__doc_title__ = "RLlib Utilities"
__all__ = ["unpack_config", "load_torch_mlp_policy"]
