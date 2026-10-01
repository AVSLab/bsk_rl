"""Sampled hold before acquisition and invalid candidate handling."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from bsk_rl.act import ImageRSO
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
