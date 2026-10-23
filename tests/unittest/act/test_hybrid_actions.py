"""Hybrid parameters must survive normalization and mixed action dispatch."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from gymnasium import spaces

from bsk_rl import act
from bsk_rl.act.discrete_actions import DiscreteActionBuilder, clean_action
from bsk_rl.scene.targets import Target


@pytest.mark.parametrize(
    "parameter,expected",
    [
        (0, 0.0),
        (2.5, 2.5),
        (np.float32(2.5), 2.5),
        (np.array(2.5), 2.5),
        (np.array([2.5], dtype=np.float32), 2.5),
        ([2.5], 2.5),
        ((2.5,), 2.5),
        (None, None),
    ],
)
def test_clean_action_normalizes_parameter(parameter, expected):
    selected, value = clean_action((2, parameter), hybrid=True)

    assert selected == 2
    assert value == expected
    if value is not None:
        assert type(value) is float


@pytest.mark.parametrize("selection", [np.int64(2), "target_id", object()])
def test_clean_action_preserves_selection(selection):
    selected, value = clean_action((selection, 2.5), hybrid=True)

    assert selected is selection
    assert value == 2.5


@pytest.mark.parametrize("hybrid", [False, True])
def test_clean_action_without_parameter(hybrid):
    assert clean_action(2, hybrid=hybrid) == (2, None)


@pytest.mark.parametrize("parameter", [[], [1.0, 2.0], [[1.0]], np.ones((1, 1))])
def test_clean_action_rejects_non_scalar_parameters(parameter):
    with pytest.raises(ValueError, match="scalar or one-element"):
        clean_action((0, parameter), hybrid=True)


@pytest.mark.parametrize("action", [(), (0,), (0, 1.0, 2.0)])
def test_clean_action_rejects_malformed_tuples(action):
    with pytest.raises(ValueError):
        clean_action(action, hybrid=True)


def test_nonhybrid_action_discards_continuous_parameter():
    assert clean_action((2, np.array([3.0, 4.0])), hybrid=False) == (2, None)


@pytest.fixture
def mixed_builder():
    targets = [Target(f"target_{i}", [1, 0, 0], 1.0) for i in range(2)]
    simulator = SimpleNamespace(sim_time=30.0, sim_rate=1.0)
    satellite = SimpleNamespace(
        action_spec=[
            act.Charge(duration=12.0),
            act.Image(n_ahead_image=2, hybrid=True),
            act.Broadcast(duration=9.0, hybrid=True),
        ],
        simulator=simulator,
        fsw=SimpleNamespace(
            action_charge=Mock(),
            insControl=SimpleNamespace(timeToleranceLower=0, timeToleranceUpper=0),
        ),
        logger=Mock(),
        disable_timed_terminal_event=Mock(),
        update_timed_terminal_event=Mock(),
        task_target_for_imaging=Mock(),
        enable_target_window=Mock(),
        update_imaging_time=Mock(),
    )

    def select_target(selection):
        if isinstance(selection, (int, np.integer)):
            return targets[selection]
        return next(
            target
            for target in targets
            if selection is target or selection == target.id
        )

    satellite.parse_target_selection = select_target
    builder = DiscreteActionBuilder(satellite)
    builder.reset_post_sim_init()
    return builder, satellite, targets


def test_mixed_action_space_and_sample(mixed_builder):
    builder, satellite, _ = mixed_builder
    space = builder.action_space
    assert isinstance(space, spaces.Tuple)
    assert space[0].n == 4
    assert space[1].shape == (1,)
    space.seed(42)
    action = space.sample()
    assert space.contains(action)

    builder.set_action(action)

    satellite.disable_timed_terminal_event.assert_called_once()
    assert builder.prev_action_key is not None


@pytest.mark.parametrize("global_index,local_index", [(1, 0), (np.int64(2), 1)])
def test_mixed_builder_dispatches_global_image_index(
    mixed_builder, global_index, local_index
):
    builder, satellite, targets = mixed_builder

    builder.set_action((global_index, np.array([2.5], dtype=np.float32)))

    satellite.task_target_for_imaging.assert_called_once_with(
        targets[local_index], max_duration=None, imaging_time=2.5
    )
    satellite.fsw.action_charge.assert_not_called()
    assert not builder.action_spec[2].broadcast_pending
    assert builder.prev_action_key == targets[local_index].id


@pytest.mark.parametrize("as_object", [False, True])
def test_mixed_builder_forwards_target_override(mixed_builder, as_object):
    builder, satellite, targets = mixed_builder
    target = targets[1]

    builder.set_action((target if as_object else target.id, np.array([0.0])))

    satellite.task_target_for_imaging.assert_called_once_with(
        target, max_duration=None, imaging_time=0.0
    )
    assert builder.prev_action_key == target.id


def test_mixed_builder_retasks_same_target_without_restarting_fsw(mixed_builder):
    builder, satellite, targets = mixed_builder
    builder.set_action((1, 5.0))
    builder.set_action((1, np.array([0.0])))

    satellite.task_target_for_imaging.assert_called_once()
    satellite.enable_target_window.assert_called_once_with(
        targets[0], max_duration=None
    )
    satellite.update_imaging_time.assert_called_once_with(imaging_time=0.0)


def test_nonhybrid_override_discards_parameter_in_mixed_builder(mixed_builder):
    builder, satellite, targets = mixed_builder
    builder.action_spec[1].hybrid = False
    # Broadcast still makes the overall space hybrid.
    assert isinstance(builder.action_space, spaces.Tuple)

    builder.set_action((targets[0].id, np.array([2.5])))

    satellite.task_target_for_imaging.assert_called_once_with(
        targets[0], max_duration=None
    )


def test_mixed_builder_accepts_legacy_integer_for_hybrid_action(mixed_builder):
    builder, satellite, targets = mixed_builder

    builder.set_action(1)

    satellite.task_target_for_imaging.assert_called_once_with(
        targets[0], max_duration=None
    )


def test_mixed_builder_keeps_fixed_duration_for_nonhybrid_action(mixed_builder):
    builder, satellite, _ = mixed_builder

    builder.set_action((0, np.array([2.5])))

    satellite.fsw.action_charge.assert_called_once()
    satellite.update_timed_terminal_event.assert_called_once_with(
        42.0, info="for action_charge"
    )


def test_mixed_builder_dispatches_and_caps_broadcast(mixed_builder):
    builder, satellite, _ = mixed_builder

    builder.set_action((3, np.array([20.0])))

    assert builder.action_spec[2].broadcast_pending
    satellite.update_timed_terminal_event.assert_called_once_with(
        39.0, info="for broadcast"
    )


@pytest.mark.parametrize("index", [-1, np.int64(-1), 4, 100])
def test_mixed_builder_rejects_out_of_range_index_without_tasking(mixed_builder, index):
    builder, satellite, _ = mixed_builder

    with pytest.raises(ValueError, match="out of range"):
        builder.set_action((index, np.array([2.5])))

    satellite.fsw.action_charge.assert_not_called()
    satellite.task_target_for_imaging.assert_not_called()
    satellite.update_timed_terminal_event.assert_not_called()
    assert builder.prev_action_key is None


@pytest.mark.parametrize(
    "action_type",
    [act.Charge, act.Drift, act.NadirPoint, act.Desat, act.Downlink, act.Scan],
)
def test_mode_actions_accept_sampled_hybrid_duration(action_type):
    action = action_type(duration=10.0, hybrid=True)
    action.satellite = Mock()
    action.simulator = SimpleNamespace(sim_rate=0.5, sim_time=30.0)

    action.set_action((np.int64(0), np.array([2.5], dtype=np.float32)))

    getattr(action.satellite.fsw, action.fsw_action).assert_called_once()
    action.satellite.update_timed_terminal_event.assert_called_once_with(
        32.5, info=f"for {action.fsw_action}"
    )
