import gymnasium as gym
import numpy as np
from pytest import approx

from bsk_rl import act, obs, sats
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import random_orbit


class TestRelProperties:
    class BasicSat(sats.Satellite):
        dyn_type = dyn.BasicDynamicsModel
        fsw_type = fsw.BasicFSWModel
        observation_spec = [obs.Time()]
        action_spec = [act.Drift()]

    class RelPropertiesSat(sats.Satellite):
        dyn_type = dyn.BasicDynamicsModel
        fsw_type = fsw.BasicFSWModel
        observation_spec = [
            obs.RelativeProperties(dict(prop="r_DC_N"), chief_name="Chief")
        ]
        action_spec = [act.Drift()]

    env = gym.make(
        "ConstellationTasking-v1",
        satellites=[
            RelPropertiesSat(
                "Deputy",
                sat_args=dict(
                    oe=random_orbit(i=1.0, a=7000.0, e=0.0, Omega=0.0, omega=0.0, f=0.0)
                ),
            ),
            BasicSat(
                "Chief",
                sat_args=dict(
                    oe=random_orbit(i=1.0, a=7100.0, e=0.0, Omega=0.0, omega=0.0, f=0.0)
                ),
            ),
        ],
        sim_rate=1.0,
        max_step_duration=10.0,
        time_limit=100.0,
        disable_env_checker=True,
    )

    def test_relative_observation(self):
        observation, info = self.env.reset()
        assert np.linalg.norm(observation["Deputy"]) == approx(100.0 * 1e3)
