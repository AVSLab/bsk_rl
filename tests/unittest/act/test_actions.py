from unittest.mock import MagicMock, call, patch

from gymnasium import spaces

from bsk_rl import act
from bsk_rl.act.actions import ActionBuilder
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
        image.satellite.task_target_for_imaging.assert_called_once_with(self.target)
        assert out == "target_1"

    def test_image_retask(self):
        image = act.Image(n_ahead_image=10)
        image.satellite = MagicMock()
        image.satellite.parse_target_selection.return_value = self.target
        out = image.image(5, "target_1")
        image.satellite.enable_target_window.assert_called_once_with(self.target)
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
