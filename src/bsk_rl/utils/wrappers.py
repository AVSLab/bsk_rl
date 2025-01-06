import logging

import numpy as np
from gymnasium import Wrapper
from pettingzoo.utils import BaseParallelWrapper

logger = logging.getLogger(__name__)


def sanitize_nan(value, replace_with=0, warn=True):
    """Replace NaN values with a given value."""
    recast_to_list = False
    if isinstance(value, list):
        recast_to_list = True
        value = np.array(value)

    if isinstance(value, np.ndarray):
        if warn and np.isnan(value).any():
            logger.warning(
                f"Replacing NaN values in array with {replace_with}. Array: {value}"
            )
        value[np.isnan(value)] = replace_with
    elif isinstance(value, dict):
        for key, val in value.items():
            value[key] = sanitize_nan(val, replace_with)
    elif isinstance(value, tuple):
        value = tuple(sanitize_nan(val, replace_with) for val in value)

    if recast_to_list:
        value = value.tolist()

    return value


class SanitizeNanBaseWrapper:
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = sanitize_nan(observation)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = sanitize_nan(observation)
        return observation, reward, terminated, truncated, info


class SanitizeNanWrapper(SanitizeNanBaseWrapper, Wrapper):
    def __init__(self, env):
        SanitizeNanBaseWrapper.__init__(self)
        Wrapper.__init__(self, env)


class SanitizeNanParallelWrapper(SanitizeNanBaseWrapper, BaseParallelWrapper):
    def __init__(self, env):
        SanitizeNanBaseWrapper.__init__(self)
        BaseParallelWrapper.__init__(self, env)
