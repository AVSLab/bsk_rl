"""Behavioral regressions for durable, receiver-local completion knowledge."""

from dataclasses import replace
import pytest
from bsk_rl.data.completion_catalog import CompletionCatalog, CompletionRecord
from bsk_rl.data.multiagent_rso_data import ImageProductRecord
from bsk_rl.data.multiagent_rso_reward import _TeamServiceAccounting


def fact(record_id="a", capture=10.0, **kw):
    return CompletionRecord(
        record_id, "sensor_0", 3, 0.0, capture, capture + 10, True, **kw
    )


def test_out_of_order_exposures_preserve_acquisition_and_ground_age():
    catalog = CompletionCatalog("sensor_1", [3], cooldown_s=100)
    newer = fact("new", 40)
    older_delivered = fact("old", 10, delivery_time=60)
    catalog.merge_record(newer, 50)
    catalog.merge_record(older_delivered, 70)
    catalog.merge_record(fact("old", 10), 80)
    assert catalog.target(3).latest_acquisition_time == 40
    assert catalog.target(3).latest_delivery_time == 60
    assert catalog.freshest_delivered_capture(3) == 10
    assert catalog.records["old"].delivery_time == 60
    assert catalog.received_at["old"] == 70
    assert catalog.target(3).cooldown_until == 140
    assert not catalog.target(3).pending_record_ids


def test_receipt_age_and_packet_expiry_do_not_erase_completed_service():
    catalog = CompletionCatalog("sensor_1", [3], cooldown_s=100)
    catalog.merge_record(fact(), 25)
    assert not catalog.is_eligible(3, 100)
    assert catalog.is_eligible(3, 110)
    assert catalog.target(3).latest_acquisition_time == 10
    catalog.set_request_epoch(3, 80)
    assert catalog.is_eligible(3, 100)
    assert catalog.target(3).latest_acquisition_time is None
    assert "a" in catalog.records
    with pytest.raises(ValueError):
        catalog.set_request_epoch(3, 70)


def test_remote_completion_does_not_remove_own_onboard_product():
    catalog = CompletionCatalog("sensor_1", [3], cooldown_s=100)
    own = ImageProductRecord(
        "own", "sensor_1", 3, 5, None, 1, "sensor_1", completion_time=15
    )
    catalog.record_capture(own)
    catalog.merge_record(fact(), 30)
    assert catalog.target(3).pending_record_ids == ("own",)
    catalog.record_delivery(own.delivered(40))
    assert catalog.target(3).pending_record_ids == ()
    assert not catalog.is_eligible(3, 40)


def test_invalid_or_conflicting_facts_cannot_replace_provenance():
    catalog = CompletionCatalog("sensor_1", [3])
    with pytest.raises(ValueError):
        catalog.merge_record(fact(), 19)
    catalog.merge_record(fact(), 20)
    with pytest.raises(ValueError, match="provenance"):
        catalog.merge_record(replace(fact(), source_sensor="sensor_2"), 21)
    assert catalog.records["a"] == fact()


def test_new_request_gets_credit_despite_previous_request_cooldown():
    ledger = _TeamServiceAccounting(cooldown_s=100, quality_threshold=0.5)
    first = ImageProductRecord("a", "sensor_0", 3, 10, None, 1, "sensor_0")
    second = replace(first, record_id="b", capture_time=30, request_epoch=25)
    assert ledger.register_acquisitions([first], {3: 8}) == {"sensor_0": 8}
    assert ledger.register_acquisitions([second], {3: 8}) == {"sensor_0": 8}


def test_capture_and_ground_upgrade_keep_distinct_receipt_timestamps():
    catalog = CompletionCatalog("sensor_1", [3])
    catalog.merge_record(fact(), 30)
    catalog.merge_record(fact(delivery_time=40), 50)
    catalog.merge_record(fact(), 60)
    assert catalog.received_at["a"] == 30
    assert catalog.version_received_at["a"] == {1: 30, 2: 50}
    assert catalog.records["a"].capture_time == 10
    assert catalog.records["a"].completion_time == 20
    assert catalog.records["a"].delivery_time == 40
