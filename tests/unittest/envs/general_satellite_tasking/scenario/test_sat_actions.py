from unittest.mock import MagicMock, call, patch

import pytest
from gymnasium import spaces

from bsk_rl.envs.general_satellite_tasking.scenario import sat_actions as sa
from bsk_rl.envs.general_satellite_tasking.scenario.environment_features import Target


@patch.multiple(sa.DiscreteSatAction, __abstractmethods__=set())
@patch("bsk_rl.envs.general_satellite_tasking.scenario.satellites.Satellite.__init__")
class TestDiscreteSatAction:
    def test_init(self, sat_init):
        sa.DiscreteSatAction()
        sat_init.assert_called_once()

    mock_action = MagicMock(__name__="some_action")

    @pytest.mark.parametrize(
        "kwargs,expected_map,expected_list",
        [
            (dict(act_fn=mock_action), {"0": "some_action"}, [mock_action]),
            (
                dict(act_fn=mock_action, act_name="new_name"),
                {"0": "new_name"},
                [mock_action],
            ),
        ],
    )
    def test_add_single_action(self, sat_init, kwargs, expected_map, expected_list):
        sat = sa.DiscreteSatAction()
        sat.add_action(**kwargs)
        assert sat.action_map == expected_map
        assert sat.action_list == expected_list

    @pytest.mark.parametrize("n_actions", [1, 3])
    def test_add_multiple_actions(self, sat_init, n_actions):
        sat = sa.DiscreteSatAction()
        sat.some_action = MagicMock(
            __name__="some_action", side_effect=lambda x, prev_action_key=None: x
        )
        sat.add_action(sat.some_action, n_actions=n_actions)
        assert sat.action_map == {f"0-{n_actions-1}": "some_action"}
        assert [act() for act in sat.action_list] == list(range(n_actions))
        sat.some_action.assert_has_calls(
            [call(i, prev_action_key=None) for i in range(n_actions)]
        )

    @patch(
        "bsk_rl.envs.general_satellite_tasking.scenario.satellites.Satellite._disable_timed_terminal_event"
    )
    def test_set_action(self, sat_init, disable_timed):
        sat = sa.DiscreteSatAction()
        sat.action_list = [MagicMock(return_value="act_key")]
        sat.set_action(0)
        disable_timed.assert_called_once()
        sat.action_list[0].assert_called_once()
        assert sat.prev_action_key == "act_key"

    def test_action_space(self, sat_init):
        sat = sa.DiscreteSatAction()
        sat.action_list = [0, 1, 2]
        assert sat.action_space == spaces.Discrete(3)


@patch.multiple(sa.DiscreteSatAction, __abstractmethods__=set())
@patch("bsk_rl.envs.general_satellite_tasking.scenario.satellites.Satellite.__init__")
class TestFSWAction:
    def test_init(self, sat_init):
        FSWAct = sa.fsw_action_gen("cool_action")
        sat = FSWAct(action_duration=10.0)
        sat_init.assert_called_once()
        assert sat.cool_action_duration == 10.0

    def make_action_sat(self):
        FSWAct = sa.fsw_action_gen("cool_action")
        sat = FSWAct()
        sat.fsw = MagicMock(cool_action=MagicMock())
        sat.log_info = MagicMock()
        sat._disable_timed_terminal_event = MagicMock()
        sat._update_timed_terminal_event = MagicMock()
        sat.simulator = MagicMock(sim_time=0.0)
        return sat

    def test_act(self, sat_init):
        sat = self.make_action_sat()
        assert sat.action_list[0].__name__ == "act_cool_action"
        sat.set_action(0)
        assert "cool_action" == sat.prev_action_key
        sat.log_info.assert_called_once_with("cool_action tasked for 60.0 seconds")
        sat.fsw.cool_action.assert_called_once()

    def test_retask(self, sat_init):
        sat = self.make_action_sat()
        sat.set_action(0)
        sat.set_action(0)
        sat.fsw.cool_action.assert_called_once()


@patch.multiple(sa.ImagingActions, __abstractmethods__=set())
@patch(
    "bsk_rl.envs.general_satellite_tasking.scenario.satellites.ImagingSatellite.__init__"
)
class TestImagingActions:
    def test_init(self, sat_init):
        sat = sa.ImagingActions(n_ahead_act=3)
        sat_init.assert_called_once()
        assert sat.action_map == {"0-2": "image"}

    class MockTarget(MagicMock, Target):
        @property
        def id(self):
            return "target_1"

    @pytest.mark.parametrize("target", [1, "target_1", MockTarget()])
    def test_image(self, sat_init, target):
        sat = sa.ImagingActions()
        sat.log_info = MagicMock()
        sat.parse_target_selection = MagicMock(return_value=self.MockTarget())
        sat.task_target_for_imaging = MagicMock()
        assert "target_1" == sat.image(target)
        sat.task_target_for_imaging.assert_called_once_with(
            sat.parse_target_selection()
        )

    @pytest.mark.parametrize("target", [1, "target_1", MockTarget()])
    def test_image_retask(self, sat_init, target):
        sat = sa.ImagingActions()
        sat.log_info = MagicMock()
        sat.parse_target_selection = MagicMock(return_value=self.MockTarget())
        sat.task_target_for_imaging = MagicMock()
        sat.image(target, prev_action_key="target_1")
        sat.task_target_for_imaging.assert_not_called()

    @patch(
        "bsk_rl.envs.general_satellite_tasking.scenario.sat_actions.DiscreteSatAction.set_action"
    )
    @pytest.mark.parametrize("target", [1, "target_1", MockTarget()])
    def test_set_action(self, sat_init, discrete_set, target):
        sat = sa.ImagingActions()
        sat._disable_image_event = MagicMock()
        sat.image = MagicMock()
        sat.set_action(target)
        sat._disable_image_event.assert_called()
        if isinstance(target, int):
            discrete_set.assert_called_once()
        elif isinstance(target, (Target, str)):
            sat.image.assert_called_once()


@patch.multiple(sa.ChargingAction, __abstractmethods__=set())
@patch.multiple(sa.DriftAction, __abstractmethods__=set())
@patch.multiple(sa.DesatAction, __abstractmethods__=set())
@patch.multiple(sa.DownlinkAction, __abstractmethods__=set())
@patch.multiple(sa.ImagingActions, __abstractmethods__=set())
@patch(
    "bsk_rl.envs.general_satellite_tasking.scenario.satellites.ImagingSatellite.__init__"
)
def test_combination(sat_init):
    class ComboAct(
        sa.ImagingActions.configure(n_ahead_act=3),
        sa.DownlinkAction,
        sa.DesatAction,
        sa.DriftAction,
        sa.ChargingAction,
    ):
        pass

    sat = ComboAct()
    assert sat.action_map == {
        "0": "action_charge",
        "1": "action_drift",
        "2": "action_desat",
        "3": "action_downlink",
        "4-6": "image",
    }
    assert len(sat.action_list) == 7
    sat_init.assert_called_once()
