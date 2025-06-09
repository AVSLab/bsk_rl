import gymnasium as gym
import numpy as np
from pytest import approx

from bsk_rl import act, obs, sats
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import random_orbit


class TestImpulsiveThrust:
    class ThrustSat(sats.Satellite):
        dyn_type = dyn.BasicDynamicsModel
        fsw_type = fsw.MagicOrbitalManeuverFSWModel
        observation_spec = [obs.SatProperties(dict(prop="v_BN_N"))]
        action_spec = [act.ImpulsiveThrust()]

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=ThrustSat(
            "ThrusterSat",
            sat_args=dict(dv_available_init=1000.0),
        ),
        sim_rate=0.01,
        time_limit=10000.0,
        disable_env_checker=True,
    )

    def test_thrust(self):
        observation_before, _ = self.env.reset()
        observation_after, _, _, _, _ = self.env.step([100.0, 0.0, 0.0, 0.02])
        diff = observation_after - observation_before
        assert np.allclose(diff, np.array([100.0, 0.0, 0.0]), atol=0.5)

        time_before = self.env.unwrapped.simulator.sim_time
        self.env.step([0.0, 0.0, 0.0, 5.0])
        time_after = self.env.unwrapped.simulator.sim_time
        assert time_after - time_before == approx(5.0)


class TestHillImpulsiveThrust:
    class ThrustSat(sats.Satellite):
        dyn_type = dyn.BasicDynamicsModel
        fsw_type = fsw.MagicOrbitalManeuverFSWModel
        observation_spec = [obs.SatProperties(dict(prop="v_BN_N"))]
        action_spec = [act.ImpulsiveThrustHill(chief_name="ThrusterSat")]

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=ThrustSat(
            "ThrusterSat",
            sat_args=dict(
                dv_available_init=1000.0,
                oe=random_orbit(i=0.0, e=0.0, Omega=0.0, omega=0.0, f=90.0),
            ),
        ),
        sim_rate=0.01,
        time_limit=10000.0,
        disable_env_checker=True,
    )

    def test_thrust(self):
        observation_before, _ = self.env.reset()
        observation_after, _, _, _, _ = self.env.step([100.0, 0.0, 0.0, 0.02])
        diff = observation_after - observation_before
        assert np.allclose(diff, np.array([0.0, 100.0, 0.0]), atol=0.5)

        time_before = self.env.unwrapped.simulator.sim_time
        self.env.step([0.0, 0.0, 0.0, 5.0])
        time_after = self.env.unwrapped.simulator.sim_time
        assert time_after - time_before == approx(5.0)
