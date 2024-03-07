import gymnasium as gym
from gymnasium.envs.registration import register

from bsk_rl._check_bsk_version import _check_bsk_version
from bsk_rl.env.gym_env import (
    GeneralSatelliteTasking,
    MultiagentSatelliteTasking,
    SingleSatelliteTasking,
)

register(
    id="GeneralSatelliteTasking-v1",
    entry_point="bsk_rl.env.gym_env:GeneralSatelliteTasking",
)

register(
    id="SingleSatelliteTasking-v1",
    entry_point="bsk_rl.env.gym_env:SingleSatelliteTasking",
)

register(
    id="MultiagentSatelliteTasking-v1",
    entry_point="bsk_rl.env.gym_env:MultiagentSatelliteTasking",
)

_check_bsk_version()
