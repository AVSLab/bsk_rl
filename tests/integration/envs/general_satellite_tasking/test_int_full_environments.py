import gymnasium as gym
import pytest

from bsk_rl.envs.general_satellite_tasking.scenario import data
from bsk_rl.envs.general_satellite_tasking.scenario import satellites as sats
from bsk_rl.envs.general_satellite_tasking.scenario.environment_features import (
    StaticTargets,
)
from bsk_rl.envs.general_satellite_tasking.simulation import environment
from bsk_rl.envs.general_satellite_tasking.utils.orbital import random_orbit

multi_env = gym.make(
    "GeneralSatelliteTasking-v1",
    satellites=[
        sats.FullFeaturedSatellite(
            "Sentinel-2A",
            sat_args=sats.FullFeaturedSatellite.default_sat_args(oe=random_orbit),
            imageAttErrorRequirement=0.01,
            imageRateErrorRequirement=0.01,
        ),
        sats.FullFeaturedSatellite(
            "Sentinel-2B",
            sat_args=sats.FullFeaturedSatellite.default_sat_args(oe=random_orbit),
            imageAttErrorRequirement=0.01,
            imageRateErrorRequirement=0.01,
        ),
    ],
    env_type=environment.GroundStationEnvModel,
    env_args=None,
    env_features=StaticTargets(n_targets=1000),
    data_manager=data.UniqueImagingManager(),
    sim_rate=0.5,
    max_step_duration=1e9,
    time_limit=5700.0,
    disable_env_checker=True,
)


@pytest.mark.parametrize("env", [multi_env])
def test_reproducibility(env):
    actions = [env.action_space.sample() for _ in range(1000)]
    reward_sum_1 = 0
    observation_1, info = env.reset(seed=0)
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum_1 += reward
        if truncated or terminated:
            break

    truncated = False
    terminated = False
    reward_sum_2 = 0
    observation_2, info = env.reset(seed=0)
    for o1, o2 in zip(observation_1, observation_2):
        assert abs(o1 - o2 < 1e-6).all()

    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum_2 += reward
        if truncated or terminated:
            break

    assert reward_sum_2 == reward_sum_1
