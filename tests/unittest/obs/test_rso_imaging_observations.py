"""Candidate snapshots are shared by observations and action selection."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
from Basilisk.architecture import messaging
from Basilisk.simulation import spacecraftLocation

from bsk_rl.act import ImageRSO
from bsk_rl.obs import RSOTargetProperties
from bsk_rl.obs.rso_imaging import target_properties
from bsk_rl.utils.rso_imaging import candidate_snapshot, illumination_factor


def observer():
    targets = {}
    for key, position, velocity in (
        ("83", [100, 7e6 + 200, 300], [-7390, 20, -30]),
        ("far/blue", [-400, 7e6 - 100, -200], [-7415, -25, 40]),
    ):
        targets[key] = NS(
            id=key,
            priority=2.0,
            target_spacecraft=NS(
                dynamics=NS(
                    r_BN_N=position,
                    v_BN_N=velocity,
                    eclipse_index=0,
                    world=NS(
                        eclipseObject=NS(
                            eclipseOutMsgs=[
                                NS(
                                    read=lambda: messaging.EclipseMsgPayload(
                                        illuminationFactor=0.8
                                    )
                                )
                            ]
                        )
                    ),
                )
            ),
        )
    scene = NS(targets_by_id=targets, revision=0, is_eligible=lambda *_: True)
    return NS(
        simulator=NS(sim_time=0),
        rso_scenario=scene,
        dynamics=NS(r_BN_N=[0, 7e6, 0], v_BN_N=[-7400, 0, 0], BN=np.eye(3)),
        fsw=NS(locPoint=NS(pHat_B=[0, 0, 1])),
    )


def test_observation_and_action_share_snapshot_and_explicit_padding():
    sat = observer()
    observation = RSOTargetProperties(dict(prop="valid"), n_ahead_observe=3)
    observation.link_satellite(sat)
    values = observation.get_obs()
    assert values["rso_targets_2"]["valid"] == 0
    snapshot = sat._rso_candidate_snapshot
    assert observation.candidate_ids() == tuple(
        None if target is None else target.id for target in snapshot.targets
    )
    assert sat._rso_candidate_snapshot is snapshot
    action = ImageRSO(3)
    action.satellite, action.simulator = sat, sat.simulator
    selected = []
    action._task = lambda target: selected.append(target.id)
    action.set_action(1)
    assert sat._rso_candidate_snapshot is snapshot
    assert selected == [snapshot.targets[1].id]
    with pytest.raises(ValueError, match="empty"):
        action.set_action(2)
    sat.rso_scenario.revision += 1
    assert candidate_snapshot(sat, 3) is not snapshot


def test_default_row_shape_and_normalization():
    sat = observer()
    observation = RSOTargetProperties(n_ahead_observe=10)
    observation.link_satellite(sat)
    rows = observation.get_obs()
    assert sum(np.size(value) for row in rows.values() for value in row.values()) == 110
    assert rows["rso_targets_0"]["priority"] == 1.0
    assert rows["rso_targets_0"]["target_illumination_factor"] == 0.8
    assert rows["rso_targets_9"]["target_distance"] == 0.0


@pytest.mark.parametrize("observe_count, action_count", [(5, 1), (1, 3)])
def test_observation_count_is_independent_of_action_count(observe_count, action_count):
    sat = observer()
    action = ImageRSO(action_count)
    action.satellite, action.simulator = sat, sat.simulator
    sat.action_builder = NS(action_spec=[action])
    observation = RSOTargetProperties(n_ahead_observe=observe_count)
    observation.link_satellite(sat)
    observation.reset_post_sim_init()
    assert len(observation.get_obs()) == observe_count
    snapshot = sat._rso_candidate_snapshot
    selected = []
    action._task = lambda target: selected.append(target.id)
    action.set_action(0)
    assert sat._rso_candidate_snapshot is snapshot
    assert selected == [observation.candidate_ids()[0]]


def test_observation_without_an_imaging_action():
    observation = RSOTargetProperties(n_ahead_observe=2)
    observation.link_satellite(observer())
    observation.reset_post_sim_init()
    assert len(observation.get_obs()) == 2


@pytest.mark.parametrize(
    "target_id, relative_hill, velocity_hill, relative_inertial",
    [
        ("83", [200, -100, 300], [20, -10, -30], [100, 200, 300]),
        ("far/blue", [-100, 400, -200], [-25, 15, 40], [-400, -100, -200]),
    ],
)
def test_relative_features_with_different_positions_and_velocities(
    target_id, relative_hill, velocity_hill, relative_inertial
):
    sat = observer()
    values = target_properties(sat, sat.rso_scenario.targets_by_id[target_id])
    # For this orbit the Hill axes are +Y, -X, +Z. These expectations are
    # independent of rv2HN and expose both axis order and velocity-sign errors.
    np.testing.assert_allclose(values["rel_pos_vector_r_BR_H"], relative_hill)
    np.testing.assert_allclose(values["rel_vel_vector_v_BR_H"], velocity_hill)
    distance = np.linalg.norm(relative_inertial)
    assert values["target_distance"] == pytest.approx(distance)
    assert values["target_elevation_angle"] == pytest.approx(
        np.degrees(np.arcsin(relative_inertial[1] / distance))
    )
    assert values["angle_to_target"] == pytest.approx(
        np.degrees(np.arccos(relative_inertial[2] / distance))
    )
    observation = RSOTargetProperties(n_ahead_observe=2)
    observation.link_satellite(sat)
    index = observation.candidate_ids().index(target_id)
    row = observation.get_obs()[f"rso_targets_{index}"]
    np.testing.assert_allclose(
        row["rel_pos_vector_r_BR_H"], np.array(relative_hill) / 15960e3
    )
    np.testing.assert_allclose(
        row["rel_vel_vector_v_BR_H"], np.array(velocity_hill) / 16000
    )


@pytest.mark.parametrize("fraction", [0.0, 0.8, 1.0])
def test_illumination_is_read_without_inversion(fraction):
    payload = messaging.EclipseMsgPayload(illuminationFactor=fraction)
    assert illumination_factor(payload) == fraction


def test_illumination_accessor_handles_equivalent_native_wrappers():
    # This native module exposes another wrapper for the same payload pointer.
    payload = spacecraftLocation.EclipseMsgPayload()
    messaging.EclipseMsgPayload.illuminationFactor.fset(payload, 0.8)
    assert illumination_factor(payload) == 0.8
