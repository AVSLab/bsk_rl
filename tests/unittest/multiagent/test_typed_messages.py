from bsk_rl.comm.typed_messages import (
    IntentStatusInbox,
    IntentStatusMessage,
    MessageDisposition,
    PerfectMetadataChannel,
)
from bsk_rl.data.multiagent_rso_data import LocalCatalogKnowledge


def message(sequence, *, created=10.0, expires=20.0, cooldown=100.0):
    return IntentStatusMessage(
        sender="sensor_0",
        sequence_number=sequence,
        target_id=2,
        action="Imaging",
        creation_time=created,
        expiry_time=expires,
        latest_acquisition_time=created,
        latest_delivery_time=None,
        cooldown_until=cooldown,
        lifecycle_status="pending_verification",
    )


def test_one_way_delivery_updates_only_receiver_catalog():
    cat_0 = LocalCatalogKnowledge("sensor_0", [2])
    cat_1 = LocalCatalogKnowledge("sensor_1", [2])
    inbox_0 = IntentStatusInbox("sensor_0", cat_0)
    inbox_1 = IntentStatusInbox("sensor_1", cat_1)
    channel = PerfectMetadataChannel({"sensor_0": inbox_0, "sensor_1": inbox_1})
    channel.send(message(0), ["sensor_1"])
    results = channel.deliver(10.0)

    assert results[0][1] is MessageDisposition.ACCEPTED
    assert cat_0.target(2).latest_acquisition_time is None
    assert cat_1.target(2).latest_acquisition_time == 10.0


def test_duplicate_stale_expired_and_out_of_order_are_deterministic():
    catalog = LocalCatalogKnowledge("sensor_1", [2])
    inbox = IntentStatusInbox("sensor_1", catalog)
    assert inbox.receive(message(2), 10.0) is MessageDisposition.ACCEPTED
    assert inbox.receive(message(2), 10.0) is MessageDisposition.DUPLICATE
    assert inbox.receive(message(1), 10.0) is MessageDisposition.STALE
    assert inbox.receive(message(3, expires=9.0), 10.0) is MessageDisposition.EXPIRED


def test_message_created_during_action_is_visible_at_next_decision_epoch():
    catalog = LocalCatalogKnowledge("sensor_1", [2])
    inbox = IntentStatusInbox("sensor_1", catalog)
    channel = PerfectMetadataChannel({"sensor_1": inbox})
    update = message(0, created=15.0, expires=30.0)
    channel.send(update, ["sensor_1"], available_time=15.0)

    # The receiver remains in its current action through t=14.
    assert channel.deliver(14.0) == []
    assert catalog.target(2).latest_acquisition_time is None
    # Communication completes at t=15; the receiver's next observation/decision sees it.
    assert channel.deliver(15.0)[0][1] is MessageDisposition.ACCEPTED
    assert catalog.target(2).latest_acquisition_time == 15.0


def test_received_pending_status_stops_filtering_after_message_expiry():
    catalog = LocalCatalogKnowledge("sensor_1", [2])
    inbox = IntentStatusInbox("sensor_1", catalog)
    assert (
        inbox.receive(message(0, expires=20.0, cooldown=None), 10.0)
        is MessageDisposition.ACCEPTED
    )
    assert not catalog.is_eligible(2, 20.0)
    assert catalog.is_eligible(2, 20.1)
    assert catalog.target(2).remote_pending_sources == ()


def test_status_only_message_does_not_mutate_catalog():
    catalog = LocalCatalogKnowledge("sensor_1", [2])
    inbox = IntentStatusInbox("sensor_1", catalog)
    update = IntentStatusMessage(
        sender="sensor_0",
        sequence_number=0,
        target_id=None,
        action="Charge",
        creation_time=10.0,
        expiry_time=20.0,
        sender_position_N=(7000e3, 0.0, 0.0),
        sender_velocity_N=(0.0, 7500.0, 0.0),
        sender_battery_fraction=0.9,
        sender_storage_fraction=0.1,
        sender_wheel_speed_fraction=0.2,
        sender_action_remaining_s=30.0,
        sender_catalog_update_time=None,
    )
    assert inbox.receive(update, 10.0) is MessageDisposition.ACCEPTED
    assert update.teammate_status() is not None
    assert catalog.target(2).last_update_time == float("-inf")


def test_remote_cooldown_changes_shared_but_not_private_eligibility():
    catalog = LocalCatalogKnowledge("sensor_1", [2])
    inbox = IntentStatusInbox("sensor_1", catalog)
    assert (
        inbox.receive(message(0, created=10.0, expires=20.0, cooldown=100.0), 10.0)
        is MessageDisposition.ACCEPTED
    )
    assert catalog.is_privately_eligible(2, 30.0)
    assert not catalog.is_eligible(2, 30.0)
