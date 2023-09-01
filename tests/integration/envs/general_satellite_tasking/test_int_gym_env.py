import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bsk_rl.envs.general_satellite_tasking.scenario import data
from bsk_rl.envs.general_satellite_tasking.scenario import satellites as sats
from bsk_rl.envs.general_satellite_tasking.scenario.environment_features import (
    StaticTargets,
)
from bsk_rl.envs.general_satellite_tasking.simulation import environment
from bsk_rl.envs.general_satellite_tasking.utils.orbital import random_orbit


class TestSingleSatelliteTasking:
    env = gym.make(
        "SingleSatelliteTasking-v1",
        satellites=sats.DoNothingSatellite(
            "Sputnik",
            sat_args=sats.DoNothingSatellite.default_sat_args(oe=random_orbit),
        ),
        env_type=environment.BasicEnvironmentModel,
        env_args=environment.BasicEnvironmentModel.default_env_args(),
        env_features=StaticTargets(n_targets=0),
        data_manager=data.NoDataManager(),
        sim_rate=1.0,
        max_step_duration=10.0,
        time_limit=100.0,
        disable_env_checker=True,
    )

    def test_reset(self):
        observation, info = self.env.reset()
        assert (observation == np.array([0.0])).all()

    def test_action_space(self):
        assert self.env.action_space == spaces.Discrete(1)

    def test_observation_space(self):
        assert self.env.observation_space == spaces.Box(-1e16, 1e16, (1,))

    def test_step(self):
        observation, reward, terminated, truncated, info = self.env.step(0)
        assert (observation == np.array([0.1])).all()

    def test_truncate(self):
        terminated = truncated = False
        while not (terminated or truncated):
            observation, reward, terminated, truncated, info = self.env.step(0)
        assert truncated
        assert self.env.simulator.sim_time == 100.0

    def test_repeatable(self):
        self.env.reset(seed=0)
        env_args_old = self.env.env_args
        sat_args_old = self.env.satellite.sat_args
        self.env.reset(seed=0)
        assert self.env.env_args == env_args_old
        for val, val_old in zip(
            self.env.satellite.sat_args.values(), sat_args_old.values()
        ):
            if (
                isinstance(val, (np.ndarray, list, type(None)))
                or np.issubdtype(type(val), np.integer)
                or np.issubdtype(type(val), float)
            ):
                assert np.all(val == val_old)


class TestSingleSatelliteDeath:
    env = gym.make(
        "SingleSatelliteTasking-v1",
        satellites=sats.DoNothingSatellite(
            "Skydiver",
            sat_args=sats.DoNothingSatellite.default_sat_args(
                rN=[0, 0, 7e6], vN=[0, 0, -100.0]
            ),
        ),
        env_type=environment.BasicEnvironmentModel,
        env_args=environment.BasicEnvironmentModel.default_env_args(),
        env_features=StaticTargets(n_targets=0),
        data_manager=data.NoDataManager(),
        sim_rate=1.0,
        time_limit=1000.0,
        failure_penalty=-1000,
        disable_env_checker=True,
    )

    def test_fail(self):
        observation, info = self.env.reset()
        observation, reward, terminated, truncated, info = self.env.step(0)
        assert terminated
        assert reward == -1000

    def test_close(self):
        self.env.close()


class TestGeneralSatelliteTasking:
    env = gym.make(
        "GeneralSatelliteTasking-v1",
        satellites=[
            sats.DoNothingSatellite(
                "Sentinel-2A",
                sat_args=sats.DoNothingSatellite.default_sat_args(oe=random_orbit),
            ),
            sats.DoNothingSatellite(
                "Sentinel-2B",
                sat_args=sats.DoNothingSatellite.default_sat_args(oe=random_orbit),
            ),
        ],
        env_type=environment.BasicEnvironmentModel,
        env_args=None,
        env_features=StaticTargets(n_targets=0),
        data_manager=data.NoDataManager(),
        sim_rate=1.0,
        max_step_duration=10.0,
        time_limit=100.0,
        disable_env_checker=True,
    )

    def test_reset(self):
        observation, info = self.env.reset()
        assert (observation == np.array([0.0])).all()

    def test_action_space(self):
        assert self.env.action_space == spaces.Tuple(
            (spaces.Discrete(1), spaces.Discrete(1))
        )

    def test_observation_space(self):
        assert self.env.observation_space == spaces.Tuple(
            (spaces.Box(-1e16, 1e16, (1,)), spaces.Box(-1e16, 1e16, (1,)))
        )

    def test_step(self):
        observation, reward, terminated, truncated, info = self.env.step([0, 0])
        assert (observation == np.array([0.1])).all()

    def test_truncate(self):
        terminated = truncated = False
        while not (terminated or truncated):
            observation, reward, terminated, truncated, info = self.env.step([0, 0])
        assert truncated
        assert self.env.simulator.sim_time == 100.0
