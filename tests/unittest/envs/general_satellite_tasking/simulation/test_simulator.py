from unittest.mock import MagicMock, patch

import pytest

from bsk_rl.envs.general_satellite_tasking.simulation.simulator import Simulator


@patch("Basilisk.utilities.SimulationBaseClass.SimBaseClass.__init__")
class TestSimulator:
    dyn = MagicMock()
    fsw = MagicMock()

    class MockEnv(MagicMock):
        def __init__(self, sim, rate, **kwargs):
            super().__init__(**kwargs)
            self.sim = sim
            self.rate = rate

    def mock_sim(self, **kwargs):
        Simulator.InitializeSimulation = MagicMock()
        Simulator.ConfigureStopTime = MagicMock()
        Simulator.ExecuteSimulation = MagicMock()

        class MockSat(MagicMock):
            @property
            def id(self):
                return "sat_1"

        sat = MockSat(
            set_dynamics=MagicMock(return_value=self.dyn),
            set_fsw=MagicMock(return_value=self.fsw),
        )
        sim = Simulator([sat], env_type=self.MockEnv, env_args={}, **kwargs)
        sim.TotalSim = MagicMock(CurrentNanos=1000000000)

        return sim

    def test_init(self, simbase_init):
        sim = self.mock_sim()
        assert sim.dynamics_list == {"sat_1": self.dyn}
        assert sim.fsw_list == {"sat_1": self.fsw}

    def test_sim_time_ns(self, simbase_init):
        sim = self.mock_sim()
        assert sim.sim_time_ns == 1000000000

    def test_sim_time(self, simbase_init):
        sim = self.mock_sim()
        assert sim.sim_time == 1.0

    def test_set_environment(self, simbase_init):
        sim = self.mock_sim()
        assert sim.environment.sim == sim
        assert sim.environment.rate == sim.sim_rate

    def test_delete_event(self, simbase_init):
        sim = self.mock_sim()
        event = MagicMock()
        sim.eventMap = {"event": event, "other": MagicMock()}
        sim.eventList = [MagicMock(), event, MagicMock()]
        sim.delete_event("event")
        assert "event" not in sim.eventMap
        assert event not in sim.eventList

    @pytest.mark.parametrize(
        "start_time,step_duration,time_limit,stop_time",
        [(0, 100, 50, 50), (0, 100, 200, 100), (10, 10, 50, 20)],
    )
    def test_run(self, simbase_init, start_time, step_duration, time_limit, stop_time):
        sim = self.mock_sim(max_step_duration=step_duration, time_limit=time_limit)
        sim.TotalSim.CurrentNanos = start_time * 1000000000
        sim.run()
        sim.ConfigureStopTime.assert_called_with(stop_time * 1000000000)
