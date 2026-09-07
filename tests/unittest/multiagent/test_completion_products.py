"""Physical image ownership and hold-gated capture/delivery accounting."""

from types import SimpleNamespace as NS
from Basilisk.simulation.THRSimConfig import StringVector
import numpy as np
from bsk_rl.data.completion_reward import CompletionImageStore, CompletionImageReward
from bsk_rl.data.rso_targets_data import RSOTargetImageData
from bsk_rl.sats.roles import SpacecraftRole


def setup():
    target = type("Target", (), {"id": 3, "name": "target_3"})()
    names = StringVector()
    names.push_back(target.name)
    message = NS(storedData=[0.0], storedDataName=names)
    sensor = NS(
        name="sensor_0",
        role=SpacecraftRole.SENSING_AGENT,
        simulator=NS(sim_time=5),
        dynamics=NS(
            storageUnit=NS(storageUnitDataOutMsg=NS(read=lambda: message)),
            eclipse_threshold_for_imaging=0.5,
        ),
        _rso_pending_capture_metadata_by_name={},
        _active_image_rso_action=NS(_hold_target=target),
    )
    store = CompletionImageStore(
        sensor, initial_data=RSOTargetImageData(known=[target])
    )
    sensor.data_store = store
    return sensor, store, target, message


def test_raw_bits_at_teammate_boundary_are_not_a_completed_exposure():
    sensor, store, target, msg = setup()
    early = store.compare_log_states(np.array([0.0]), np.array([100.0]))
    assert not early.pending_image_records_by_id
    assert not store.catalog.records
    assert 3 in store.unresolved_captures
    sensor.simulator.sim_time = 15
    sensor._active_image_rso_action = None
    sensor._rso_pending_capture_metadata_by_name[target.name] = [
        dict(
            record_id="physical",
            capture_time=4,
            end_time=15,
            success=True,
            request_epoch=0,
            mean_hold_shadow_factor=1.0,
        )
    ]
    # The hold gate finishes without any additional instrument bits being written.
    resolved = store.compare_log_states(np.array([100.0]), np.array([100.0]))
    record = resolved.pending_image_records_by_id[3][0]
    assert record["capture_time"] == 4
    assert record["end_time"] == 15
    assert record["storage_delta_bits"] == 100
    assert not store.unresolved_captures


def test_interrupted_hold_creates_unqualified_owned_storage():
    sensor, store, target, msg = setup()
    store.compare_log_states(np.array([0.0]), np.array([100.0]))
    sensor.simulator.sim_time = 10
    sensor._active_image_rso_action = None
    failed = store.compare_log_states(np.array([100.0]), np.array([100.0]))
    record = failed.pending_image_records_by_id[3][0]
    reward = object.__new__(CompletionImageReward)
    product = reward._product_from_record(sensor, record)
    store.store_product(product)
    store.catalog.record_capture(product)
    assert product.quality == 0
    assert not store.catalog.records[product.record_id].qualified
    assert store.catalog.target(3).pending_record_ids == (product.record_id,)


def test_partial_downlink_does_not_verify_or_remove_the_product():
    sensor, store, target, msg = setup()
    reward = object.__new__(CompletionImageReward)
    reward.old_storage_by_sensor = {sensor.name: np.array([100.0])}
    reward.scenario = NS(target_spacecrafts=[target], satellites=[sensor])
    reward.quality_threshold = 0.5
    reward.per_sensor_metrics = {sensor.name: {"deliveries": 0}}
    product = reward._product_from_record(
        sensor,
        dict(
            record_id="physical",
            target_id=3,
            capture_time=0,
            end_time=5,
            mean_hold_shadow_factor=1.0,
            storage_delta_bits=100,
        ),
    )
    store.store_product(product)
    store.catalog.record_capture(product)
    store.data.mark_target_pending(target, {"record_id": "physical"})
    msg.storedData = [60.0]
    sensor.simulator.sim_time = 10
    assert reward._downlinked_products() == []
    assert store.products == (product,)
    assert store.catalog.records["physical"].delivery_time is None
    msg.storedData = [0.0]
    sensor.simulator.sim_time = 15
    delivered = reward._downlinked_products()
    assert delivered == [product.delivered(15)]
    assert not store.products
    assert store.catalog.records["physical"].delivery_time == 15
