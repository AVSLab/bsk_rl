import gymnasium as gym
import numpy as np
from pytest import approx

from bsk_rl.env.scenario import actions as act
from bsk_rl.env.scenario import data
from bsk_rl.env.scenario import observations as obs
from bsk_rl.env.scenario.environment_features import StaticTargets
from bsk_rl.env.simulation import dynamics, fsw
from bsk_rl.utils.orbital import random_orbit


class TestImagingSatellite:
    class ImageSat(
        act.ImagingActions.configure(),
    ):
        dyn_type = dynamics.ImagingDynModel
        fsw_type = fsw.ImagingFSWModel
        observation_spec = [obs.Time()]

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
