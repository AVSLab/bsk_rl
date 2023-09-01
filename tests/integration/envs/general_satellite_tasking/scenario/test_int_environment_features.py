import gymnasium as gym

from bsk_rl.envs.general_satellite_tasking.scenario import data
from bsk_rl.envs.general_satellite_tasking.scenario import sat_actions as sa
from bsk_rl.envs.general_satellite_tasking.scenario import sat_observations as so
from bsk_rl.envs.general_satellite_tasking.scenario.environment_features import (
    CityTargets,
    StaticTargets,
)
from bsk_rl.envs.general_satellite_tasking.simulation import dynamics, environment, fsw
from bsk_rl.envs.general_satellite_tasking.utils.orbital import random_orbit


def make_env(env_features):
    class ImageSat(
        sa.ImagingActions,
        so.TimeState,
    ):
        dyn_type = dynamics.GroundStationDynModel
        fsw_type = fsw.ImagingFSWModel

    env = gym.make(
        "SingleSatelliteTasking-v1",
        satellites=ImageSat(
            "EO-1",
            n_ahead_act=10,
            sat_args=ImageSat.default_sat_args(
                oe=random_orbit,
                imageAttErrorRequirement=0.05,
                imageRateErrorRequirement=0.05,
            ),
        ),
        env_type=environment.GroundStationEnvModel,
        env_args=environment.GroundStationEnvModel.default_env_args(),
        env_features=env_features,
        data_manager=data.UniqueImagingManager(),
        sim_rate=1.0,
        time_limit=5700.0,
        max_step_duration=1e9,
        disable_env_checker=True,
    )
    return env


class TestStaticTargets:
    def test_priority_dist(self):
        env = make_env(StaticTargets(n_targets=1000, priority_distribution=lambda: 1))
        env.reset()
        for i in range(10):
            observation, reward, terminated, truncated, info = env.step(i)
            assert reward in [0.0, 1.0]


class TestCityTargets:
    def test_city_distribution(self):
        # Just a smoke test
        env = make_env(
            CityTargets(n_targets=1000, n_select_from=10000, location_offset=1e3)
        )
        env.reset()
        reward_sum = 0.0
        for i in range(10):
            observation, reward, terminated, truncated, info = env.step(i)
            reward_sum += reward
        assert reward_sum > 0
