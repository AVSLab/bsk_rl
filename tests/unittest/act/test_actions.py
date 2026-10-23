from types import SimpleNamespace
from unittest.mock import MagicMock, call, create_autospec, patch

import numpy as np
import pytest
from gymnasium import spaces

from bsk_rl import act
from bsk_rl.act.actions import ActionBuilder
from bsk_rl.act.continuous_actions import ContinuousActionBuilder
from bsk_rl.act.discrete_actions import DiscreteActionBuilder


@pytest.fixture
def imaging_satellite():
    satellite = MagicMock()
    satellite.fsw.insControl = SimpleNamespace(
        timeToleranceLower=0, timeToleranceUpper=0, imagingTime=0
    )
    return satellite


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
        for action in satellite.action_spec:
            action.hybrid = False
        ab = DiscreteActionBuilder(satellite)
        assert ab.action_space == spaces.Discrete(3)

    def test_hybrid_action_space(self):
        satellite = MagicMock(
            action_spec=[MagicMock(n_actions=1), MagicMock(n_actions=2)]
        )
        for action in satellite.action_spec:
            action.hybrid = True
        ab = DiscreteActionBuilder(satellite)
        assert ab.action_space == spaces.Tuple(
            (spaces.Discrete(3), spaces.Box(low=0, high=np.inf, shape=(1,)))
        )

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
        assert ab.action_spec[0].set_action.call_args == call(
            (0, None), prev_action_key=None
        )
        ab.set_action(1)
        assert ab.action_spec[1].set_action.call_args == call(
            (0, None), prev_action_key="foo"
        )
        ab.set_action(2)
        assert ab.action_spec[1].set_action.call_args == call(
            (1, None), prev_action_key="bar"
        )
        ab.set_action(3)
        assert ab.action_spec[2].set_action.call_args == call(
            (0, None), prev_action_key="bar"
        )

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
            ("foo", None), prev_action_key=None
        )


class TestDiscreteFSWAction:
    def test_set_action(self):
        fswact = act.DiscreteFSWAction("action_fsw")
        fswact.satellite = MagicMock()
        fswact.simulator = SimpleNamespace(sim_rate=1.0, sim_time=0.0)
        fswact.set_action(0)
        fswact.satellite.fsw.action_fsw.assert_called_once()

    def test_set_action_again(self):
        fswact = act.DiscreteFSWAction("action_fsw")
        fswact.satellite = MagicMock()
        fswact.simulator = SimpleNamespace(sim_rate=1.0, sim_time=0.0)
        fswact.set_action(0, prev_action_key="action_fsw")
        fswact.satellite.fsw.action_fsw.assert_not_called()

    def test_set_action_reset(self):
        fswact = act.DiscreteFSWAction("action_fsw", reset_task=True)
        fswact.satellite = MagicMock()
        fswact.simulator = SimpleNamespace(sim_rate=1.0, sim_time=0.0)
        fswact.set_action(0, prev_action_key="action_fsw")
        fswact.satellite.fsw.action_fsw.assert_called_once()


@pytest.mark.parametrize("kind", ["fsw", "broadcast"])
@pytest.mark.parametrize(
    "requested,cap,sim_rate,expected_duration",
    [
        (0.0, 100.0, 1.0, 1.0),
        (0.25, 100.0, 1.0, 1.0),
        (5.0, 100.0, 1.0, 5.0),
        (20.0, 10.0, 1.0, 10.0),
        (0.1, 0.25, 1.0, 1.0),
        (None, 0.25, 1.0, 1.0),
        (None, 10.0, 1.0, 10.0),
        (0.25, 100.0, 0.5, 0.5),
    ],
)
def test_hybrid_mode_action_duration(kind, requested, cap, sim_rate, expected_duration):
    if kind == "fsw":
        action = act.DiscreteFSWAction("action_fsw", duration=cap, hybrid=True)
    else:
        action = act.Broadcast(duration=cap, hybrid=True)
    action.satellite = MagicMock()
    action.simulator = SimpleNamespace(sim_rate=sim_rate, sim_time=10.0)

    action.set_action((0, requested))

    action.satellite.update_timed_terminal_event.assert_called_once()
    assert action.satellite.update_timed_terminal_event.call_args.args[
        0
    ] == pytest.approx(10.0 + expected_duration)
    if kind == "fsw":
        action.satellite.logger.info.assert_called_once_with(
            f"{action.name} tasked for {expected_duration} seconds"
        )


@pytest.fixture
def make_image(imaging_satellite):
    def make(*, hybrid=True, max_duration=None, sim_rate=1.0):
        image = act.Image(n_ahead_image=10, max_duration=max_duration, hybrid=hybrid)
        image.satellite = imaging_satellite
        image.simulator = SimpleNamespace(sim_rate=sim_rate, sim_time=100.0)
        image.satellite.parse_target_selection.return_value = SimpleNamespace(
            id="target_1"
        )
        return image

    return make


class TestImage:
    # image() is shared by discrete and hybrid actions; only the entry points
    # below need to vary the hybrid flag.
    def test_image(self, make_image):
        image = make_image()
        target = image.satellite.parse_target_selection.return_value

        assert image.image(5) == target.id
        image.satellite.task_target_for_imaging.assert_called_once_with(
            target, max_duration=None
        )

    @pytest.mark.parametrize("hybrid", [False, True])
    def test_image_legacy_satellite_override(self, make_image, hybrid):
        image = make_image(hybrid=hybrid)
        target = image.satellite.parse_target_selection.return_value
        image.satellite.task_target_for_imaging = create_autospec(
            lambda target, max_duration=None: None
        )

        assert image.set_action(5) == target.id
        image.satellite.task_target_for_imaging.assert_called_once_with(
            target, max_duration=None
        )

    def test_image_retask(self, make_image):
        image = make_image()
        target = image.satellite.parse_target_selection.return_value

        assert image.image(5, target.id) == target.id
        image.satellite.enable_target_window.assert_called_once_with(
            target, max_duration=None
        )
        image.satellite.task_target_for_imaging.assert_not_called()

    def test_image_max_duration(self, make_image):
        image = make_image(max_duration=100.0)
        target = image.satellite.parse_target_selection.return_value

        assert image.image(5) == target.id
        image.satellite.task_target_for_imaging.assert_called_once_with(
            target, max_duration=100.0
        )

    @pytest.mark.parametrize("imaging_time", [0.0, 20.0])
    def test_image_forwards_requested_time(self, make_image, imaging_time):
        image = make_image()
        target = image.satellite.parse_target_selection.return_value

        assert image.image(5, imaging_time=imaging_time) == target.id
        image.satellite.task_target_for_imaging.assert_called_once_with(
            target, max_duration=None, imaging_time=imaging_time
        )

    @pytest.mark.parametrize(
        "imaging_time,upper,cap,sim_rate,expected_duration,prev_action_key",
        [
            (20.0, 3.0, None, 1.0, 23.0, None),
            (20.0, 3.0, 10.0, 1.0, 10.0, None),
            (20.0, 3.0, 100.0, 1.0, 23.0, None),
            (0.0, 0.25, None, 1.0, 1.0, None),
            (20.0, 3.0, 0.25, 1.0, 1.0, None),
            (0.0, 0.25, None, 0.5, 0.5, None),
            (20.0, 0.0, None, 1.0, None, None),
            (20.0, 0.0, 100.0, 1.0, 100.0, None),
            (None, 3.0, None, 1.0, None, None),
            (None, 3.0, 100.0, 1.0, 100.0, None),
            # The deadline calculation is shared. Retasking needs representative
            # timed and untimed cases to check the separate forwarding path.
            (0.0, 0.25, None, 1.0, 1.0, "target_1"),
            (None, 3.0, 100.0, 1.0, 100.0, "target_1"),
        ],
    )
    def test_imaging_deadline(
        self,
        make_image,
        imaging_time,
        upper,
        cap,
        sim_rate,
        expected_duration,
        prev_action_key,
    ):
        image = make_image(max_duration=cap, sim_rate=sim_rate)
        target = image.satellite.parse_target_selection.return_value
        image.satellite.fsw.insControl.timeToleranceUpper = int(upper * 1e9)

        assert (
            image.image(5, prev_action_key=prev_action_key, imaging_time=imaging_time)
            == target.id
        )

        if prev_action_key is None:
            kwargs = {} if imaging_time is None else {"imaging_time": imaging_time}
            image.satellite.task_target_for_imaging.assert_called_once_with(
                target, max_duration=expected_duration, **kwargs
            )
            image.satellite.enable_target_window.assert_not_called()
        else:
            image.satellite.enable_target_window.assert_called_once_with(
                target, max_duration=expected_duration
            )
            image.satellite.task_target_for_imaging.assert_not_called()
            if imaging_time is None:
                image.satellite.update_imaging_time.assert_not_called()
            else:
                image.satellite.update_imaging_time.assert_called_once_with(
                    imaging_time=imaging_time
                )

    @pytest.mark.parametrize("prev_action_key", [None, "target_1"])
    @pytest.mark.parametrize(
        "lower,upper,imaging_time,expect_warning",
        [
            (0, 0, None, False),
            (2, 0, None, True),
            (0, 3, None, True),
            (2, 3, None, True),
            (2, 3, 0.0, False),
            (2, 3, 20.0, False),
        ],
    )
    def test_missing_imaging_time_warning(
        self,
        make_image,
        recwarn,
        prev_action_key,
        lower,
        upper,
        imaging_time,
        expect_warning,
    ):
        image = make_image()
        target = image.satellite.parse_target_selection.return_value
        controller = image.satellite.fsw.insControl
        controller.timeToleranceLower = int(lower * 1e9)
        controller.timeToleranceUpper = int(upper * 1e9)
        controller.imagingTime = int(100 * 1e9)
        previous_timing = vars(controller).copy()

        assert (
            image.image(5, prev_action_key=prev_action_key, imaging_time=imaging_time)
            == target.id
        )

        assert not recwarn
        if expect_warning:
            image.satellite.logger.warning.assert_called_once()
            assert (
                "existing imagingTime and tolerances are retained"
                in (image.satellite.logger.warning.call_args.args[0])
            )
        else:
            image.satellite.logger.warning.assert_not_called()
        if imaging_time is None:
            assert vars(controller) == previous_timing
            image.satellite.update_imaging_time.assert_not_called()
            if prev_action_key is None:
                image.satellite.task_target_for_imaging.assert_called_once_with(
                    target, max_duration=None
                )

    @pytest.mark.parametrize("hybrid", [False, True])
    @pytest.mark.parametrize(
        "selection,method", [(5, "set_action"), ("image", "set_action_override")]
    )
    @pytest.mark.parametrize("imaging_time", [None, 20.0])
    def test_action_entry_points(
        self, make_image, hybrid, selection, method, imaging_time
    ):
        image = make_image(hybrid=hybrid)
        image.image = MagicMock()
        payload = selection if imaging_time is None else (selection, imaging_time)

        getattr(image, method)(payload)

        image.image.assert_called_once_with(
            selection, None, imaging_time=imaging_time if hybrid else None
        )


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
