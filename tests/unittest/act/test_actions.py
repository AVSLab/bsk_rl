from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from gymnasium import spaces

from bsk_rl import act
from bsk_rl.act.actions import ActionBuilder
from bsk_rl.act.continuous_actions import ContinuousActionBuilder
from bsk_rl.act.discrete_actions import DiscreteActionBuilder


@patch.multiple(ActionBuilder, __abstractmethods__=set())
class TestActionBuilder:
    def test_init(self):
        action_spec = [MagicMock() for _ in range(3)]
        satellite = MagicMock(action_spec=action_spec)
        ab = ActionBuilder(satellite)
        for a in ab.action_spec:
            a.link_satellite.assert_called_once()

    def test_reset_post_sim_init(self):
        ab = ActionBuilder(MagicMock(action_spec=[MagicMock() for _ in range(3)]))
        ab.reset_post_sim_init()
        for a in ab.action_spec:
            a.link_simulator.assert_called_once()
            a.reset_post_sim_init.assert_called_once()


class TestDiscreteActionBuilder:
    def test_action_space(self):
        satellite = MagicMock(
            action_spec=[MagicMock(n_actions=1), MagicMock(n_actions=2)]
        )
        ab = DiscreteActionBuilder(satellite)
        assert ab.action_space == spaces.Discrete(3)

    def test_action_description(self):
        satellite = MagicMock(
            action_spec=[
                MagicMock(n_actions=1),
                MagicMock(n_actions=2),
            ]
        )
        satellite.action_spec[0].name = "foo"
        satellite.action_spec[1].name = "bar"
        ab = DiscreteActionBuilder(satellite)
        assert ab.action_description == ["foo", "bar_0", "bar_1"]

    def test_set_action(self):
        satellite = MagicMock(
            action_spec=[
                MagicMock(n_actions=1, set_action=MagicMock(return_value="foo")),
                MagicMock(n_actions=2, set_action=MagicMock(return_value="bar")),
                MagicMock(n_actions=1, set_action=MagicMock(return_value="baz")),
            ]
        )
        ab = DiscreteActionBuilder(satellite)
        ab.set_action(0)
        assert ab.action_spec[0].set_action.call_args == call(0, prev_action_key=None)
        ab.set_action(1)
        assert ab.action_spec[1].set_action.call_args == call(0, prev_action_key="foo")
        ab.set_action(2)
        assert ab.action_spec[1].set_action.call_args == call(1, prev_action_key="bar")
        ab.set_action(3)
        assert ab.action_spec[2].set_action.call_args == call(0, prev_action_key="bar")

    def test_set_action_override(self):
        satellite = MagicMock(
            action_spec=[
                MagicMock(n_actions=1, set_action_override=None),
                MagicMock(n_actions=2, set_action_override=MagicMock()),
            ]
        )
        ab = DiscreteActionBuilder(satellite)
        ab.set_action("foo")
        assert ab.action_spec[1].set_action_override.call_args == call(
            "foo", prev_action_key=None
        )


class TestDiscreteFSWAction:
    def test_set_action(self):
        fswact = act.DiscreteFSWAction("action_fsw")
        fswact.satellite = MagicMock()
        fswact.simulator = MagicMock()
        fswact.set_action(0)
        fswact.satellite.fsw.action_fsw.assert_called_once()

    def test_set_action_again(self):
        fswact = act.DiscreteFSWAction("action_fsw")
        fswact.satellite = MagicMock()
        fswact.simulator = MagicMock()
        fswact.set_action(0, prev_action_key="action_fsw")
        fswact.satellite.fsw.action_fsw.assert_not_called()

    def test_set_action_reset(self):
        fswact = act.DiscreteFSWAction("action_fsw", reset_task=True)
        fswact.satellite = MagicMock()
        fswact.simulator = MagicMock()
        fswact.set_action(0, prev_action_key="action_fsw")
        fswact.satellite.fsw.action_fsw.assert_called_once()


class TestImage:
    target = MagicMock()
    target.id = "target_1"

    def test_image(self):
        image = act.Image(n_ahead_image=10)
        image.satellite = MagicMock()
        image.satellite.parse_target_selection.return_value = self.target
        out = image.image(5, None)
        image.satellite.task_target_for_imaging.assert_called_once_with(
            self.target, max_duration=None
        )
        assert out == "target_1"

    def test_image_retask(self):
        image = act.Image(n_ahead_image=10)
        image.satellite = MagicMock()
        image.satellite.parse_target_selection.return_value = self.target
        out = image.image(5, "target_1")
        image.satellite.enable_target_window.assert_called_once_with(
            self.target, max_duration=None
        )
        assert out == "target_1"

    def test_image_max_duration(self):
        image = act.Image(n_ahead_image=10, max_duration=100.0)
        image.satellite = MagicMock()
        image.satellite.parse_target_selection.return_value = self.target
        out = image.image(5, None)
        image.satellite.task_target_for_imaging.assert_called_once_with(
            self.target, max_duration=100.0
        )
        assert out == "target_1"

    def test_set_action(self):
        image = act.Image(n_ahead_image=10)
        image.satellite = MagicMock()
        image.image = MagicMock()
        image.set_action(5)
        image.image.assert_called_once_with(5, None)

    def test_set_action_override(self):
        image = act.Image(n_ahead_image=10)
        image.satellite = MagicMock()
        image.image = MagicMock()
        image.set_action_override("image")
        image.image.assert_called_once_with("image", None)


class TestContinuousActionBuilder:
    def test_action_space(self):
        action_space = spaces.Box(low=0, high=1, shape=(3,))
        action = MagicMock(space=action_space)
        satellite = MagicMock(action_spec=[action])
        ab = ContinuousActionBuilder(satellite)
        assert ab.action_space == action_space
        assert ab._action == action

    def test_too_many_actions(self):
        satellite = MagicMock(action_spec=[MagicMock(), MagicMock()])
        with pytest.raises(AssertionError):
            ContinuousActionBuilder(satellite)

    def test_action(self):
        action = MagicMock()
        satellite = MagicMock(action_spec=[action])
        ab = ContinuousActionBuilder(satellite)
        act = MagicMock()
        ab.set_action(act)
        ab._action.set_action.assert_called_once_with(act)


class TestImpulsiveThrust:
    @pytest.mark.parametrize("requested,actual", [(0.5, 0.5), (2.0, 1.0)])
    def test_max_dv(self, requested, actual):
        impulsive_thrust = act.ImpulsiveThrust(max_dv=1.0)
        satellite = MagicMock()
        impulsive_thrust.satellite = satellite
        satellite.simulator.sim_rate = 5.0

        impulsive_thrust.set_action(np.array([requested, 0.0, 0.0, 100]))
        assert satellite.fsw.action_impulsive_thrust.call_args[0][0][0] == actual

    def test_fsw_action(self):
        impulsive_thrust = act.ImpulsiveThrust(fsw_action="some_action")
        satellite = MagicMock()
        impulsive_thrust.satellite = satellite
        satellite.simulator.sim_rate = 5.0

        impulsive_thrust.set_action(np.array([0.5, 0.0, 0.0, 100]))
        satellite.fsw.some_action.assert_called_once()

    @pytest.mark.parametrize(
        "start_time, duration_request, actual_end",
        [(10, 8, 20), (10, 1000, 110), (10, 50, 60)],
    )
    def test_duration(self, start_time, duration_request, actual_end):
        impulsive_thrust = act.ImpulsiveThrust(max_drift_duration=100)
        satellite = MagicMock()
        satellite.simulator.sim_rate = 5.0
        satellite.simulator.sim_time = start_time
        impulsive_thrust.satellite = satellite

        impulsive_thrust.set_action(np.array([0.0, 0.0, 0.0, duration_request]))

        satellite.update_timed_terminal_event.assert_called_once_with(actual_end)
