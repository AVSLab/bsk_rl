import gymnasium as gym
import numpy as np
from pytest import approx

from bsk_rl.envs.general_satellite_tasking.scenario import data
from bsk_rl.envs.general_satellite_tasking.scenario import sat_actions as sa
from bsk_rl.envs.general_satellite_tasking.scenario import sat_observations as so
from bsk_rl.envs.general_satellite_tasking.scenario.environment_features import (
    StaticTargets,
)
from bsk_rl.envs.general_satellite_tasking.simulation import dynamics, environment, fsw
from bsk_rl.envs.general_satellite_tasking.utils.orbital import random_orbit


class TestImagingSatellite:
    class ImageSat(
        sa.ImagingActions.configure(),
        so.TimeState,
    ):
        dyn_type = dynamics.ImagingDynModel
        fsw_type = fsw.ImagingFSWModel

    env = gym.make(
        "SingleSatelliteTasking-v1",
        satellites=ImageSat(
            "EO-1",
            n_ahead_act=10,
            initial_generation_duration=1000.0,
            generation_duration=100.0,
            sat_args=ImageSat.default_sat_args(
                oe=random_orbit,
                imageAttErrorRequirement=0.05,
                imageRateErrorRequirement=0.05,
            ),
        ),
        env_type=environment.BasicEnvironmentModel,
        env_args=environment.BasicEnvironmentModel.default_env_args(),
        env_features=StaticTargets(n_targets=5000),
        data_manager=data.NoDataManager(),
        sim_rate=1.0,
        time_limit=2000.0,
        max_step_duration=500.0,
        disable_env_checker=True,
    )

    def test_generate_more_windows(self):
        self.env.reset()
        dts = []
        while self.env.simulator.sim_time < 1500.0:
            observation, reward, terminated, truncated, info = self.env.step(1)
            dts.append(info["d_ts"])
        assert True  # Ensure stepping past initial generation duration works
        assert (np.array(dts) < 500.0).any()  # Ensure variable interval works

    def test_fixed_interval(self):
        self.env.satellite.variable_interval = False
        self.env.reset()
        dts = []
        while self.env.simulator.sim_time < 1500.0:
            observation, reward, terminated, truncated, info = self.env.step(1)
            dts.append(info["d_ts"])
        assert True  # Ensure stepping past initial generation duration works
        for dt in dts:  # Ensure fixed interval works
            assert np.array(dts) == approx(500.0)
