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

import json
from pathlib import Path

import numpy as np
import torch
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.base import ACTOR, ENCODER_OUT
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from scipy.special import softmax

from bsk_rl import ConstellationTasking, GeneralSatelliteTasking, SatelliteTasking
from bsk_rl.utils.rllib.callbacks import EpisodeDataParallelWrapper, EpisodeDataWrapper


def load_torch_mlp_policy(policy_path: str, env: GeneralSatelliteTasking):
    """Load a PyTorch policy from a saved model.

    Args:
        policy_path: The path to the saved model.
        env: The environment to load the policy for.
    """
    policy_path = Path(policy_path)
    state_dict = torch.load(policy_path / "module_state_dir" / "module_state.pt")
    with open(policy_path / "rl_module_metadata.json") as f:
        module_config = json.load(f)["module_spec_dict"]["module_config"]
        model_config_dict = module_config["model_config_dict"]

    cat = PPOCatalog(
        env.satellites[0].observation_space,  # TODO do this by agent ID
        env.satellites[0].action_space,
        model_config_dict,
    )
    encoder = cat.build_actor_critic_encoder("torch")
    pi_head = cat.build_pi_head("torch")

    encoder_state_dict = {
        ".".join(k.split(".")[1:]): torch.from_numpy(v)
        for k, v in state_dict.items()
        if k.split(".")[0] == "encoder"
    }
    encoder.load_state_dict(encoder_state_dict)
    pi_state_dict = {
        ".".join(k.split(".")[1:]): torch.from_numpy(v)
        for k, v in state_dict.items()
        if k.split(".")[0] == "pi"
    }
    pi_head.load_state_dict(pi_state_dict)

    def policy(obs, deterministic=True):
        action_logits = pi_head(
            encoder(dict(obs=torch.from_numpy(obs[None, :])))[ENCODER_OUT][ACTOR]
        )
        if deterministic:
            return action_logits.argmax().item()
        else:
            return np.random.choice(
                np.arange(0, len(action_logits[0])),
                p=softmax(action_logits.detach())[0, :],
            )

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
