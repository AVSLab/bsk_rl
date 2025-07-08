from warnings import warn

import gymnasium as gym
import numpy as np
import pytest
from pettingzoo.test.parallel_test import parallel_api_test

from bsk_rl import ConstellationTasking, act, data, obs, sats, scene
from bsk_rl.utils.orbital import random_orbit


class FullFeaturedSatellite(sats.ImagingSatellite):
    observation_spec = [
        obs.SatProperties(dict(prop="r_BN_P", module="dynamics", norm=6e6)),
        obs.Time(),
    ]
    action_spec = [act.Image(n_ahead_image=10)]


multi_env = gym.make(
    "GeneralSatelliteTasking-v1",
    satellites=[
        FullFeaturedSatellite(
            "Sentinel-2A",
            sat_args=FullFeaturedSatellite.default_sat_args(
                oe=random_orbit,
                imageAttErrorRequirement=0.01,
                imageRateErrorRequirement=0.01,
            ),
        ),
        FullFeaturedSatellite(
            "Sentinel-2B",
            sat_args=FullFeaturedSatellite.default_sat_args(
                oe=random_orbit,
                imageAttErrorRequirement=0.01,
                imageRateErrorRequirement=0.01,
            ),
        ),
    ],
    scenario=scene.UniformTargets(n_targets=1000),
    rewarder=data.UniqueImageReward(),
    sim_rate=0.5,
    max_step_duration=1e9,
    time_limit=5700.0,
    disable_env_checker=True,
)

parallel_env = ConstellationTasking(
    satellites=[
        FullFeaturedSatellite(
            "Sentinel-2A",
            sat_args=FullFeaturedSatellite.default_sat_args(
                oe=random_orbit,
                imageAttErrorRequirement=0.01,
                imageRateErrorRequirement=0.01,
            ),
        ),
        FullFeaturedSatellite(
            "Sentinel-2B",
            sat_args=FullFeaturedSatellite.default_sat_args(
                oe=random_orbit,
                imageAttErrorRequirement=0.01,
                imageRateErrorRequirement=0.01,
            ),
        ),
    ],
    scenario=scene.UniformTargets(n_targets=1000),
    rewarder=data.UniqueImageReward(),
    sim_rate=0.5,
    max_step_duration=1e9,
    time_limit=5700.0,
)

parallel_meta_env = ConstellationTasking(
    satellites=[
        FullFeaturedSatellite(
            "Sentinel-2A",
            sat_args=FullFeaturedSatellite.default_sat_args(
                oe=random_orbit,
                imageAttErrorRequirement=0.01,
                imageRateErrorRequirement=0.01,
            ),
        ),
        FullFeaturedSatellite(
            "Sentinel-2B",
            sat_args=FullFeaturedSatellite.default_sat_args(
                oe=random_orbit,
                imageAttErrorRequirement=0.01,
                imageRateErrorRequirement=0.01,
            ),
        ),
        FullFeaturedSatellite(
            "Sentinel-2C",
            sat_args=FullFeaturedSatellite.default_sat_args(
                oe=random_orbit,
                imageAttErrorRequirement=0.01,
                imageRateErrorRequirement=0.01,
            ),
        ),
    ],
    scenario=scene.UniformTargets(n_targets=1000),
    rewarder=data.UniqueImageReward(),
    sim_rate=0.5,
    max_step_duration=1e9,
    time_limit=5700.0,
    meta_agent_groupings={
        "DoubleSat": ["Sentinel-2A", "Sentinel-2C"],
        "SingleSat": ["Sentinel-2B"],
    },
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
        np.testing.assert_allclose(o1, o2)

    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum_2 += reward
        if truncated or terminated:
            break

    assert reward_sum_2 == reward_sum_1


@pytest.mark.repeat(5)
@pytest.mark.parametrize("env", [parallel_env, parallel_meta_env])
def test_parallel_api(env):
    with pytest.warns(UserWarning):
        # expect an erroneous warning about the info dict due to our additional info
        try:
            parallel_api_test(env)
        except AssertionError as e:
            if "agent cannot be revived once dead" in str(e):
                warn(f"'{e}' is a known issue (#59)")
            else:
                raise (e)
