import numpy as np

from bsk_rl.comm.teammate_state import (
    TEAMMATE_SUMMARY_KEYS,
    TeammateStatus,
    pool_teammate_statuses,
)


def status(name, position, *, action="Imaging", created=90.0):
    return TeammateStatus(
        source_sensor=name,
        creation_time=created,
        position_N=position,
        velocity_N=(0.0, 7500.0, 0.0),
        battery_fraction=0.8,
        storage_fraction=0.2,
        wheel_speed_fraction=0.1,
        action=action,
        action_remaining_s=20.0,
        catalog_update_time=80.0,
        target_id=3,
    )


def pooled(statuses):
    return pool_teammate_statuses(
        statuses,
        receiver_position_N=(7000e3, 0.0, 0.0),
        receiver_velocity_N=(0.0, 7500.0, 0.0),
        sim_time=100.0,
        distance_norm_m=16000e3,
        speed_norm_m_s=12000.0,
        duration_norm_s=300.0,
        age_norm_s=600.0,
    )


def test_teammate_pool_is_permutation_invariant():
    peers = [
        status("peer_b", (7100e3, 10e3, 0.0)),
        status("peer_a", (6900e3, -20e3, 0.0), action="Downlink"),
        status("peer_c", (7000e3, 30e3, 0.0), action="Charge"),
    ]
    assert pooled(peers) == pooled([peers[2], peers[0], peers[1]])


def test_teammate_pool_has_fixed_size_for_variable_sensor_count():
    empty = pooled([])
    one = pooled([status("peer_a", (7100e3, 0.0, 0.0))])
    many = pooled(
        [
            status("peer_a", (7100e3, 0.0, 0.0)),
            status("peer_b", (6900e3, 0.0, 0.0)),
            status("peer_c", (7000e3, 20e3, 0.0)),
        ]
    )
    assert tuple(empty) == TEAMMATE_SUMMARY_KEYS
    assert len(empty) == len(one) == len(many) == 24
    assert np.allclose(list(empty.values()), 0.0)
