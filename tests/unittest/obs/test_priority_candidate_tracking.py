from types import SimpleNamespace

from bsk_rl.obs.observations import (
    _record_dynamic_priority_candidate_access,
    _record_dynamic_priority_visible_access,
)


def make_target(target_id, kind, *, active=True):
    return SimpleNamespace(
        id=target_id,
        priority_event_kind=kind,
        priority_event_active=active,
        priority_event_candidate_count=0,
        priority_event_first_candidate_time=None,
        priority_event_last_candidate_log_time=None,
        priority_event_candidate_times=[],
        priority_event_candidate_slots=[],
        priority_event_visible_count=0,
        priority_event_first_visible_time=None,
        priority_event_last_visible_log_time=None,
        priority_event_visible_times=[],
    )


def test_candidate_tracking_records_time_and_slot_once_per_decision():
    hio = make_target(1, "HIO")
    control = make_target(2, "CONTROL")

    _record_dynamic_priority_candidate_access(
        [hio, control, hio],
        sim_time=120.0,
    )
    _record_dynamic_priority_candidate_access([hio], sim_time=120.0)
    _record_dynamic_priority_candidate_access([hio], sim_time=180.0)

    assert hio.priority_event_candidate_count == 2
    assert hio.priority_event_candidate_times == [120.0, 180.0]
    assert hio.priority_event_candidate_slots == [0, 0]
    assert hio.priority_event_first_candidate_time == 120.0
    assert control.priority_event_candidate_count == 1
    assert control.priority_event_candidate_slots == [1]


def test_candidate_tracking_ignores_inactive_and_untracked_targets():
    inactive = make_target(1, "HIO", active=False)
    ordinary = make_target(2, "")

    _record_dynamic_priority_candidate_access(
        [inactive, ordinary],
        sim_time=120.0,
    )

    assert inactive.priority_event_candidate_count == 0
    assert ordinary.priority_event_candidate_count == 0


def test_visible_tracking_records_each_target_once_per_decision():
    hio = make_target(1, "HIO")
    control = make_target(2, "CONTROL")

    _record_dynamic_priority_visible_access([hio, control, hio], sim_time=120.0)
    _record_dynamic_priority_visible_access([hio], sim_time=120.0)
    _record_dynamic_priority_visible_access([hio], sim_time=180.0)

    assert hio.priority_event_visible_count == 2
    assert hio.priority_event_visible_times == [120.0, 180.0]
    assert hio.priority_event_first_visible_time == 120.0
    assert control.priority_event_visible_count == 1


def test_visible_tracking_ignores_inactive_and_untracked_targets():
    inactive = make_target(1, "SHIO", active=False)
    ordinary = make_target(2, "")

    _record_dynamic_priority_visible_access(
        [inactive, ordinary],
        sim_time=120.0,
    )

    assert inactive.priority_event_visible_count == 0
    assert ordinary.priority_event_visible_count == 0
