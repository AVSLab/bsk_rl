from functools import partial
from unittest.mock import MagicMock, patch
from weakref import proxy

import numpy as np
import pytest
from Basilisk.utilities import macros

from bsk_rl import sats
from bsk_rl.sim import Simulator
from bsk_rl.sim.fsw import Task
from bsk_rl.utils import functional


class MockDynType:
    def with_defaults(self):
        pass


MockDynType.with_defaults.defaults = {"a": 1, "d": np.array([4, 5])}


class MockFSWType:
    def with_defaults(self):
        pass

    some_task = Task


MockFSWType.with_defaults.defaults = {"b": 2}


def with_defaults():
    pass


MockFSWType.some_task.with_defaults = with_defaults
MockFSWType.some_task.with_defaults.defaults = {"c": 3}


@patch.multiple(sats.Satellite, __abstractmethods__=set())
@patch("bsk_rl.sats.Satellite.observation_spec", MagicMock())
@patch("bsk_rl.sats.Satellite.action_spec", [MagicMock()])
# Mock to avoid picking up the base models
@patch("bsk_rl.sats.Satellite.get_dyn_type", MagicMock(return_value=MockDynType))
@patch("bsk_rl.sats.Satellite.get_fsw_type", MagicMock(return_value=MockFSWType))
class TestSatellite:
    sats.Satellite.dyn_type = MockDynType
    sats.Satellite.fsw_type = MockFSWType
    sats.Satellite.logger = MagicMock()

    def test_default_sat_args(self):
        np.testing.assert_equal(
            sats.Satellite.default_sat_args(),
            {
                "a": 1,
                "b": 2,
                "c": 3,
                "d": np.array([4, 5]),
            },
        )

    @pytest.mark.parametrize(
        "overwrite,error", [({"c": 4}, False), ({"not_c": 4}, True)]
    )
    def test_default_sat_args_overwrote(self, overwrite, error):
        if not error:
            np.testing.assert_equal(
                sats.Satellite.default_sat_args(**overwrite),
                {
                    "a": 1,
                    "b": 2,
                    "c": 4,
                    "d": np.array([4, 5]),
                },
            )
        else:
            with pytest.raises(KeyError):
                sats.Satellite.default_sat_args(**overwrite)

    def test_init_default(self):
        sat = sats.Satellite(name="TestSat", sat_args=None)
        np.testing.assert_equal(
            sat.sat_args_generator, {"a": 1, "b": 2, "c": 3, "d": np.array([4, 5])}
        )

    def test_generate_sat_args(self):
        sat = sats.Satellite(
            name="TestSat",
            sat_args={"a": 4, "b": lambda: 5},
        )
        sat.generate_sat_args(a=10, d=np.array([4, 6]))
        np.testing.assert_equal(
            sat.sat_args, {"a": 10, "b": 5, "c": 3, "d": np.array([4, 6])}
        )

    # @patch("bsk_rl.env.utils.orbital.TrajectorySimulator")
    # def test_reset_pre_sim_init(self, trajsim_patch):
    #     sat = sats.Satellite(name="TestSat", sat_args=None)
    #     sat.data_store = MagicMock(is_fresh=True)
    #     sat._generate_sat_args = MagicMock()
    #     sat.sat_args = {"utc_init": 0, "rN": 0, "vN": 0, "oe": 0, "mu": 0}
    #     sat.reset_pre_sim_init()
    #     trajsim_patch.assert_called_once()
    #     sat._generate_sat_args.assert_called_once()
    #     assert sat.info == []
    #     assert sat._timed_terminal_event_name is None

    @pytest.mark.parametrize(
        "dyn_state,fsw_state,sat_past_is_alive",
        [
            (a, b, c)
            for a in [True, False]
            for b in [True, False]
            for c in [True, False]
        ],
    )
    def test_is_alive(self, dyn_state, fsw_state, sat_past_is_alive):
        sat = sats.Satellite(name="TestSat", sat_args={})
        if sat_past_is_alive:
            sat.time_of_death = None
        else:
            sat.time_of_death = 5.0
        sat.simulator = MagicMock(sim_time=10.0)
        sat._is_alive = sat_past_is_alive
        sat.dynamics = MagicMock(is_alive=MagicMock(return_value=dyn_state))
        sat.fsw = MagicMock(is_alive=MagicMock(return_value=fsw_state))
        assert sat.is_alive() == (dyn_state and fsw_state and sat_past_is_alive)
        if not sat.is_alive():
            if sat_past_is_alive:
                assert sat.time_of_death == 10.0
            else:
                assert sat.time_of_death == 5.0
        else:
            assert sat.time_of_death is None

    def test_satellite_command(self):
        sat1 = sats.Satellite(name="TestSat_1", sat_args={})
        sat2 = sats.Satellite(name="TestSat_2", sat_args={})
        self.satellites = [sat1, sat2]
        self.get_satellite = partial(Simulator.get_satellite, self)
        assert sat1 == eval(sat1._satellite_command)
        assert sat1 != eval(sat2._satellite_command)
        assert sat2 == eval(sat2._satellite_command)

    def test_update_timed_terminal_event(self):
        pass  # Probably better with integration testing

    def test_setup_aliveness_events(self):
        sat = sats.Satellite(name="TestSat", sat_args={})
        sat._is_alive = True
        sat.time_of_death = None
        sat.simulator = MagicMock(sim_rate=1.0, sim_time=12.0)

        class Dyn:
            def __init__(self, satellite):
                self.satellite = satellite
                self.fail_continuous = False

            @property
            def should_not_run(self):
                raise RuntimeError("property should not be accessed")

            @functional.aliveness_checker
            def step_valid(self):
                return True

            @functional.aliveness_checker(continuous=True, check_rate=0.25)
            def continuous_valid(self):
                return not self.fail_continuous

        class Fsw:
            def __init__(self, satellite):
                self.satellite = satellite

            @functional.aliveness_checker(continuous=True)
            def fsw_valid(self):
                return True

        dyn = Dyn(sat)
        fsw = Fsw(sat)
        sat.dynamics = proxy(dyn)
        sat.fsw = proxy(fsw)
        sat.setup_aliveness_events()

        assert sat.simulator.createNewEvent.call_count == 2
        event_names = [
            call.args[0] for call in sat.simulator.createNewEvent.call_args_list
        ]
        assert any("continuous_valid" in name and "Dyn" in name for name in event_names)
        assert any("fsw_valid" in name and "Fsw" in name for name in event_names)
        assert not any("ProxyType" in name for name in event_names)

        dyn_call = sat.simulator.createNewEvent.call_args_list[0]
        assert "continuous_valid" in dyn_call.args[0]
        assert dyn_call.args[1] == macros.sec2nano(0.25)
        assert dyn_call.kwargs["terminal"] is True
        assert dyn_call.kwargs["exactRateMatch"] is False

        assert dyn_call.kwargs["conditionFunction"](sat.simulator) is False
        sat.dynamics.fail_continuous = True
        assert dyn_call.kwargs["conditionFunction"](sat.simulator) is True
        dyn_call.kwargs["actionFunction"](sat.simulator)
        assert sat._is_alive is False
        assert sat.time_of_death == 12.0

    def test_disable_timed_event(self):
        sat = sats.Satellite(name="TestSat", sat_args={})
        sat.simulator = MagicMock(eventMap={"some_event": 1})
        sat._timed_terminal_event_name = "some_event"
        sat.disable_timed_terminal_event()
        sat.simulator.delete_event.assert_called_with("some_event")

    def test_disable_timed_event_no_event(self):
        sat = sats.Satellite(name="TestSat", sat_args={})
        sat.simulator = MagicMock(eventMap={"some_event": 1})
        sat._timed_terminal_event_name = None
        sat.disable_timed_terminal_event()
        assert not sat.simulator.delete_event.called

    def test_proxy_setters(self):
        # Must be last test or others break
        mock_sim = MagicMock()
        mock_dyn = MagicMock()
        mock_fsw = MagicMock()

        sat = sats.Satellite(name="TestSat", sat_args=None)
        sat.dyn_type = MagicMock()
        sat.dyn_type.return_value = mock_dyn
        sat.fsw_type = MagicMock()
        sat.fsw_type.return_value = mock_fsw
        sat.generate_sat_args()
        sat.set_simulator(mock_sim)
        assert mock_dyn == sat.set_dynamics(1.0)
        assert mock_fsw == sat.set_fsw(1.0)
        # Should be proxies, not the actual object
        assert sat.simulator is not mock_sim
        assert sat.simulator == mock_sim
        assert sat.fsw is not mock_fsw
        assert sat.fsw == mock_fsw
        assert sat.dynamics is not mock_dyn
        assert sat.dynamics == mock_dyn
