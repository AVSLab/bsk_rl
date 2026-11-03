from typing import ClassVar

import gymnasium as gym
import pytest

from bsk_rl import act, obs, sats
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.functional import aliveness_checker
from bsk_rl.utils.orbital import random_orbit


class TestContinuousAlivenessChecker:
    @pytest.mark.parametrize(
        "continuous,check_rate,expected_time",
        [
            (False, None, 100.0),
            (True, None, 5.0),
            (True, 0.5, 5.0),
            (True, 2.0, 6.0),
            (True, 2.5, 6.0),
            (True, 3.0, 6.0),
        ],
    )
    def test_step_vs_continuous(
        self, continuous, check_rate, expected_time, caplog
    ):
        class BoomFSW(fsw.FSWModel):
            @aliveness_checker(continuous=continuous, check_rate=check_rate)
            def time_valid(self):
                return self.simulator.sim_time < 5.0

        class BoomSat(sats.Satellite):
            fsw_type = BoomFSW
            dyn_type = dyn.DynamicsModel
            observation_spec: ClassVar[list[obs.Observation]] = [obs.Time()]
            action_spec: ClassVar[list[act.Action]] = [act.Drift()]

        env = gym.make(
            "SatelliteTasking-v1",
            satellite=BoomSat("Boom", sat_args=dict(oe=random_orbit)),
            sim_rate=1.0,
            time_limit=100.0,
            max_step_duration=100.0,
            disable_env_checker=True,
        )

        env.reset(seed=0)
        _, _, terminated, _, _ = env.step(0)

        assert env.unwrapped.simulator.sim_time == pytest.approx(expected_time)
        assert not env.unwrapped.satellite.is_alive()
        assert env.unwrapped.satellite.time_of_death == pytest.approx(expected_time)
        assert terminated
        assert caplog.messages.count("failed time_valid check") == 1
        env.close()
