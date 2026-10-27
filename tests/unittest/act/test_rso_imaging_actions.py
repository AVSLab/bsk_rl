"""Sampled hold before acquisition and invalid candidate handling."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from bsk_rl.act import ImageRSO
from bsk_rl.scene import RSOTarget, RSOTargetCatalog, RSOTargets
from bsk_rl.utils.rso_imaging import CandidateSnapshot


def test_empty_candidates_never_task_another_target():
    action = ImageRSO(2)
    action.satellite = NS(
        rso_scenario=NS(revision=0, targets_by_id={}), simulator=NS(sim_time=0)
    )
    action.simulator = action.satellite.simulator
    action._task = Mock()
    with pytest.raises(ValueError, match="empty"):
        action.set_action(0)
    action._task.assert_not_called()


def test_stale_snapshot_warns_and_preserves_the_observed_target():
    action = ImageRSO(1)
    action.satellite = NS(
        rso_scenario=NS(revision=1, targets_by_id={"observed": NS(id="observed")}),
        simulator=NS(sim_time=10),
        _rso_candidate_snapshot=CandidateSnapshot(10, 0, (NS(id="observed"),)),
    )
    action.simulator = action.satellite.simulator
    action._task = Mock()
    with pytest.warns(UserWarning, match="original target ID"):
        action.set_action(0)
    action._task.assert_called_once_with(
        action.satellite.rso_scenario.targets_by_id["observed"]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_ahead_image=0),
        dict(n_ahead_image=2, max_duration=0),
        dict(n_ahead_image=1, min_pointing_hold_s=-1),
        dict(n_ahead_image=1, hold_mode="bad"),
    ],
)
def test_invalid_action_settings(kwargs):
    with pytest.raises(ValueError):
        ImageRSO(**kwargs)


def test_stale_ineligible_target_does_not_substitute_another_candidate():
    action = ImageRSO(1)
    target = NS(id="observed")
    action.satellite = NS(
        rso_scenario=NS(
            revision=1,
            targets_by_id={target.id: target},
            is_eligible=lambda target_id, time: False,
        ),
        simulator=NS(sim_time=10),
        _rso_candidate_snapshot=CandidateSnapshot(10, 0, (target,)),
    )
    action.simulator = action.satellite.simulator
    with (
        pytest.warns(UserWarning),
        pytest.raises(ValueError, match="pending or in cooldown"),
    ):
        action.set_action(0)


@pytest.mark.parametrize("block_pending_targets", [False, True])
@pytest.mark.parametrize("selection", ["candidate", "target_id"])
def test_pending_target_tasking_respects_scenario_setting(
    block_pending_targets, selection
):
    definition = RSOTarget("observed", "debris", (7e6, 0, 0), (0, 7500, 0))
    mission = RSOTargets(
        RSOTargetCatalog((definition,)),
        ["camera"],
        block_pending_targets=block_pending_targets,
    )
    mission.targets_by_id[definition.id] = definition
    mission.pending[definition.id] = {"camera:1"}
    action = ImageRSO(1, variable_duration_imaging=False)
    fsw = NS(action_image_rso=Mock())
    action.satellite = NS(
        rso_scenario=mission,
        simulator=NS(sim_time=10, eventMap={}),
        fsw=fsw,
        update_timed_terminal_event=Mock(),
        _rso_candidate_snapshot=CandidateSnapshot(10, 0, (definition,)),
    )
    action.simulator = action.satellite.simulator
    select = (
        (lambda: action.set_action(0))
        if selection == "candidate"
        else (lambda: action.set_action_override(definition.id))
    )
    if block_pending_targets:
        with pytest.raises(ValueError, match="pending or in cooldown"):
            select()
        fsw.action_image_rso.assert_not_called()
    else:
        assert select() == definition.id
        assert fsw.action_image_rso.call_args.args == (definition,)
        action.satellite.update_timed_terminal_event.assert_called_once()
        assert action.chosen_target_ids == [definition.id]
    assert mission.pending[definition.id] == {"camera:1"}
@pytest.mark.parametrize("parameter", [None, 0.0, np.array([2.5], dtype=np.float32)])
def test_nonhybrid_rso_accepts_tuple_without_forwarding_parameter(parameter):
    action = ImageRSO(1)
    target = NS(id="observed")
    action.satellite = NS(
        rso_scenario=NS(revision=0, targets_by_id={target.id: target}),
        _rso_candidate_snapshot=CandidateSnapshot(10, 0, (target,)),
    )
    action.simulator = NS(sim_time=10)
    action._task = Mock(return_value=target.id)

    assert action.set_action((0, parameter)) == target.id

    action._task.assert_called_once_with(target)
    assert action.hybrid is False


@pytest.mark.parametrize("as_object", [False, True])
def test_nonhybrid_rso_normalizes_explicit_target_tuple(as_object):
    action = ImageRSO(1)
    target = NS(id="observed")
    action.satellite = NS(rso_scenario=NS(targets_by_id={target.id: target}))
    action._task = Mock(return_value=target.id)

    assert (
        action.set_action_override(
            (target if as_object else target.id, np.array([2.5]))
        )
        == target.id
    )

    action._task.assert_called_once_with(target)
