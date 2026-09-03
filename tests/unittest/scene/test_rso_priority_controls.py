from types import SimpleNamespace

import numpy as np
import pytest

from bsk_rl.scene.rso_targets import RandomSatellites


def make_scenario(control_count, control_seed=None):
    scenario = RandomSatellites(
        "SS1",
        n_targets=40,
        dynamic_priority_event_enabled=True,
        hio_count=5,
        shio_count=3,
        priority_control_count=control_count,
        priority_control_seed=control_seed,
        dynamic_priority_event_seed=1234,
    )
    scenario.target_spacecrafts = [
        SimpleNamespace(id=i, priority=float(i + 1) / 40.0) for i in range(40)
    ]
    scenario._select_dynamic_priority_targets()
    return scenario


def test_priority_controls_do_not_change_hio_or_shio_selection():
    baseline = make_scenario(control_count=0)
    with_controls = make_scenario(control_count=8)

    assert with_controls.hio_target_ids == baseline.hio_target_ids
    assert with_controls.shio_target_ids == baseline.shio_target_ids
    assert len(with_controls.priority_control_target_ids) == 8
    assert not (
        set(with_controls.priority_control_target_ids)
        & (set(with_controls.hio_target_ids) | set(with_controls.shio_target_ids))
    )


def test_controls_activate_without_receiving_priority_boost():
    scenario = make_scenario(control_count=8)
    original_priorities = {
        target.id: target.priority for target in scenario.target_spacecrafts
    }

    assert scenario.maybe_apply_dynamic_priority_event(
        sim_time=50.0,
        time_limit=100.0,
    )

    for target in scenario.target_spacecrafts:
        if target.id in scenario.hio_target_ids:
            assert target.priority == 5.0
            assert target.priority_event_active
        elif target.id in scenario.shio_target_ids:
            assert target.priority == 10.0
            assert target.priority_event_active
        elif target.id in scenario.priority_control_target_ids:
            assert target.priority == original_priorities[target.id]
            assert target.priority_event_active


def test_zero_time_priority_assignment_is_applied_at_zero():
    scenario = RandomSatellites(
        "SS1",
        n_targets=10,
        dynamic_priority_event_enabled=True,
        dynamic_priority_event_time_sec=0.0,
        hio_count=2,
        hio_priority=5.0,
        shio_count=2,
        shio_priority=10.0,
        priority_control_count=6,
        dynamic_priority_event_seed=1234,
    )
    scenario.target_spacecrafts = [
        SimpleNamespace(id=i, priority=1.0) for i in range(10)
    ]
    scenario._select_dynamic_priority_targets()

    assert scenario.maybe_apply_dynamic_priority_event(
        sim_time=0.0,
        time_limit=100.0,
    )
    assert scenario.priority_event_applied_time == 0.0
    assert all(
        target.priority_event_active for target in scenario.target_spacecrafts
    )


def test_priority_controls_can_use_an_independent_reproducible_seed():
    first = make_scenario(control_count=8, control_seed=20260729)
    second = make_scenario(control_count=8, control_seed=20260729)
    different = make_scenario(control_count=8, control_seed=20260730)

    assert first.hio_target_ids == second.hio_target_ids == different.hio_target_ids
    assert first.shio_target_ids == second.shio_target_ids == different.shio_target_ids
    assert first.priority_control_target_ids == second.priority_control_target_ids
    assert first.priority_control_target_ids != different.priority_control_target_ids


def test_event_priorities_can_scale_from_realized_initial_maximum():
    scenario = RandomSatellites(
        "SS1",
        n_targets=200,
        priority_mode="uniform",
        priority_sum=200.0,
        priority_uniform_low=0.0,
        priority_uniform_high=2.0,
        rescale_priorities_to_sum=True,
        dynamic_priority_event_enabled=True,
        hio_count=5,
        hio_priority_max_multiplier=5.0,
        shio_count=3,
        shio_priority_max_multiplier=10.0,
        priority_control_count=8,
        dynamic_priority_event_seed=1234,
    )
    np.random.seed(99)
    priorities = scenario._generate_priorities()
    scenario.realized_initial_priority_max = float(np.max(priorities))
    scenario.target_spacecrafts = [
        SimpleNamespace(id=i, priority=float(priority))
        for i, priority in enumerate(priorities)
    ]
    scenario._select_dynamic_priority_targets()

    assert float(np.sum(priorities)) == pytest.approx(200.0)
    assert scenario.effective_hio_priority == pytest.approx(
        5.0 * float(np.max(priorities))
    )
    assert scenario.effective_shio_priority == pytest.approx(
        10.0 * float(np.max(priorities))
    )

    assert scenario.maybe_apply_dynamic_priority_event(
        sim_time=50.0,
        time_limit=100.0,
    )
    for target in scenario.target_spacecrafts:
        if target.id in scenario.hio_target_ids:
            assert target.priority == pytest.approx(
                5.0 * scenario.realized_initial_priority_max
            )
        elif target.id in scenario.shio_target_ids:
            assert target.priority == pytest.approx(
                10.0 * scenario.realized_initial_priority_max
            )


@pytest.mark.parametrize(
    "field",
    ["hio_priority_max_multiplier", "shio_priority_max_multiplier"],
)
def test_event_priority_max_multiplier_must_be_positive(field):
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        RandomSatellites("SS1", n_targets=8, **{field: 0.0})
