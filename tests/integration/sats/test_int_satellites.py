import gymnasium as gym
import numpy as np
import pytest
from pytest import approx

from bsk_rl import act, data, obs, sats
from bsk_rl.scene import UniformTargets
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import random_orbit


class TestImagingSatellite:
    class ImageSat(sats.ImagingSatellite):
        dyn_type = dyn.ImagingDynModel
        fsw_type = fsw.ImagingFSWModel
        observation_spec = [obs.Time()]
        action_spec = [act.Image(n_ahead_image=10)]

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=ImageSat(
            "EO-1",
            initial_generation_duration=1000.0,
            generation_duration=100.0,
            sat_args=ImageSat.default_sat_args(
                oe=random_orbit,
                imageAttErrorRequirement=0.05,
                imageRateErrorRequirement=0.05,
            ),
        ),
        scenario=UniformTargets(n_targets=5000),
        rewarder=data.UniqueImageReward(),
        sim_rate=1.0,
        time_limit=2000.0,
        max_step_duration=500.0,
        disable_env_checker=True,
    )

    @pytest.mark.skip(reason="Causes tests to hang sometimes")
    def test_generate_more_windows(self):
        self.env.reset()
        dts = []
        while self.env.unwrapped.simulator.sim_time < 1500.0:
            observation, reward, terminated, truncated, info = self.env.step(1)
            dts.append(info["d_ts"])
        assert True  # Ensure stepping past initial generation duration works
        assert (np.array(dts) < 500.0).any()  # Ensure variable interval works

    def test_fixed_interval(self):
        self.env.unwrapped.satellite.variable_interval = False
        self.env.reset()
        dts = []
        while self.env.unwrapped.simulator.sim_time < 1500.0:
            observation, reward, terminated, truncated, info = self.env.step(1)
            dts.append(info["d_ts"])
        assert True  # Ensure stepping past initial generation duration works
        for dt in dts:  # Ensure fixed interval works
            assert np.array(dts) == approx(500.0)
