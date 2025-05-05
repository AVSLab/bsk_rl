import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bsk_rl import act, data, obs, sats
from bsk_rl.scene import UniformTargets
from bsk_rl.utils.orbital import random_orbit


class DoNothingSatellite(sats.ImagingSatellite):
    observation_spec = [obs.Time()]
    action_spec = [act.Drift()]


class TestSatelliteTasking:
    env = gym.make(
        "SatelliteTasking-v1",
        satellite=DoNothingSatellite(
            "Sputnik",
            sat_args=DoNothingSatellite.default_sat_args(oe=random_orbit),
        ),
        scenario=UniformTargets(n_targets=0),
        rewarder=data.NoReward(),
        sim_rate=1.0,
        max_step_duration=10.0,
        time_limit=100.0,
        disable_env_checker=True,
        dtype=np.float32,
    )

    def test_reset(self):
        observation, info = self.env.reset()
        assert (observation == np.array([0.0])).all()

    def test_action_space(self):
        assert self.env.action_space == spaces.Discrete(1)

    def test_observation_space(self):
        assert self.env.observation_space == spaces.Box(
            -1e16, 1e16, (1,), dtype=np.float32
        )

    def test_step(self):
        observation, reward, terminated, truncated, info = self.env.step(0)
        assert (observation == np.array([0.1])).all()

    def test_truncate(self):
        terminated = truncated = False
        while not (terminated or truncated):
            observation, reward, terminated, truncated, info = self.env.step(0)
        assert truncated
        assert self.env.unwrapped.simulator.sim_time == 100.0

    def test_repeatable(self):
        self.env.reset(seed=0)
        world_args_old = self.env.unwrapped.world_args
        sat_args_old = self.env.unwrapped.satellite.sat_args
        self.env.reset(seed=0)
        assert self.env.unwrapped.world_args == world_args_old
        for val, val_old in zip(
            self.env.unwrapped.satellite.sat_args.values(), sat_args_old.values()
        ):
            if (
                isinstance(val, (np.ndarray, list, type(None)))
                or np.issubdtype(type(val), np.integer)
                or np.issubdtype(type(val), float)
            ):
                assert np.all(val == val_old)


class TestSingleSatelliteDeath:
    env = gym.make(
        "SatelliteTasking-v1",
        satellite=DoNothingSatellite(
            "Skydiver",
            sat_args=DoNothingSatellite.default_sat_args(
                rN=[0, 0, 7e6], vN=[0, 0, -100.0], oe=None
            ),
        ),
        scenario=UniformTargets(n_targets=0),
        rewarder=data.NoReward(),
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
            DoNothingSatellite(
                "Sentinel-2A",
                sat_args=DoNothingSatellite.default_sat_args(oe=random_orbit),
            ),
            DoNothingSatellite(
                "Sentinel-2B",
                sat_args=DoNothingSatellite.default_sat_args(oe=random_orbit),
            ),
        ],
        scenario=UniformTargets(n_targets=0),
        rewarder=data.NoReward(),
        sim_rate=1.0,
        max_step_duration=10.0,
        time_limit=100.0,
        disable_env_checker=True,
        dtype=np.float32,
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
            (
                spaces.Box(-1e16, 1e16, (1,), dtype=np.float32),
                spaces.Box(-1e16, 1e16, (1,), dtype=np.float32),
            )
        )

    def test_step(self):
        observation, reward, terminated, truncated, info = self.env.step([0, 0])
        assert (observation == np.array([0.1])).all()

    def test_truncate(self):
        terminated = truncated = False
        while not (terminated or truncated):
            observation, reward, terminated, truncated, info = self.env.step([0, 0])
        assert truncated
        assert self.env.unwrapped.simulator.sim_time == 100.0
