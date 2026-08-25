import pytest
from types import SimpleNamespace

from bsk_rl.obs.observations import _eligible_targets_now

from bsk_rl.data.multiagent_rso_data import (
    ImageProductRecord,
    LocalCatalogKnowledge,
)
from bsk_rl.data.multiagent_rso_reward import (
    MultiSensorRSOTargetImageStore,
    _TeamServiceAccounting,
)


def product(sensor, record, target=3, capture=10.0, quality=1.0):
    return ImageProductRecord(
        record_id=record,
        source_sensor=sensor,
        target_id=target,
        capture_time=capture,
        delivery_time=None,
        quality=quality,
        storage_owner=sensor,
    )


def product_store(sensor_id):
    store = object.__new__(MultiSensorRSOTargetImageStore)
    store.satellite = SimpleNamespace(name=sensor_id)
    store._products = {}
    return store


def test_physical_stores_are_independent_and_enforce_downlink_ownership():
    store_0 = product_store("sensor_0")
    store_1 = product_store("sensor_1")
    item = product("sensor_0", "p0")
    store_0.store_product(item)

    assert store_0.products == (item,)
    assert store_1.products == ()
    with pytest.raises(KeyError):
        store_1.downlink_product("p0", 100.0)
    with pytest.raises(ValueError, match="owned"):
        store_1.store_product(item)
    assert store_0.downlink_product("p0", 100.0).delivery_time == 100.0


def test_local_catalogs_diverge_without_communication():
    catalog_0 = LocalCatalogKnowledge("sensor_0", [3])
    catalog_1 = LocalCatalogKnowledge("sensor_1", [3])
    item = product("sensor_0", "p0")
    catalog_0.record_capture(item)

    assert not catalog_0.is_eligible(3, 11.0)
    assert catalog_1.is_eligible(3, 11.0)


def test_global_ledger_does_not_leak_into_local_catalogs():
    catalog_0 = LocalCatalogKnowledge("sensor_0", [3])
    catalog_1 = LocalCatalogKnowledge("sensor_1", [3])
    item = product("sensor_0", "p0").delivered(20.0)
    ledger = _TeamServiceAccounting(cooldown_s=100.0, quality_threshold=0.5)
    assert ledger.register_deliveries([item], {3: 7.0}) == {"sensor_0": 7.0}
    assert catalog_0.is_eligible(3, 21.0)
    assert catalog_1.is_eligible(3, 21.0)


def test_candidate_eligibility_uses_local_catalog_not_team_truth():
    target = type("Target", (), {"id": 3})()
    local_catalog = LocalCatalogKnowledge("sensor_0", [3])
    item = product("sensor_0", "p0")
    local_catalog.record_capture(item)
    local_data = type(
        "LocalData",
        (),
        {"eligible_targets": lambda self, sim_time, known: list(known)},
    )()
    satellite = type(
        "Sensor",
        (),
        {
            "data_store": type(
                "Store", (), {"data": local_data, "catalog": local_catalog}
            )(),
            "simulator": type("Simulator", (), {"sim_time": 11.0})(),
        },
    )()

    # Global truth may contain service, but the observation-side filter consults only
    # the explicitly attached local catalog.
    team_ledger = _TeamServiceAccounting(cooldown_s=100.0, quality_threshold=0.5)
    team_ledger.register_deliveries([item.delivered(20.0)], {3: 7.0})
    assert _eligible_targets_now(satellite, [target]) == []


def test_simultaneous_service_splits_credit_without_changing_team_total():
    left = product("sensor_0", "a", capture=10.0).delivered(20.0)
    right = product("sensor_1", "b", capture=10.0).delivered(20.0)
    ledger = _TeamServiceAccounting(cooldown_s=100.0, quality_threshold=0.5)
    credit = ledger.register_deliveries([right, left], {3: 8.0})

    assert credit == {"sensor_0": 4.0, "sensor_1": 4.0}
    assert ledger.team_value == 8.0
    assert ledger.unique_service_count == 1


def test_successful_duplicate_is_logged_separately():
    ledger = _TeamServiceAccounting(cooldown_s=100.0, quality_threshold=0.5)
    first = product("sensor_0", "a", capture=10.0).delivered(20.0)
    second = product("sensor_1", "b", capture=50.0).delivered(60.0)
    ledger.register_deliveries([first], {3: 8.0})
    assert ledger.register_deliveries([second], {3: 8.0}) == {"sensor_1": 0.0}
    assert ledger.successful_duplicate_count == 1


def test_later_capture_inside_cooldown_is_logged_as_duplicate_attempt():
    ledger = _TeamServiceAccounting(cooldown_s=100.0, quality_threshold=0.5)
    first = product("sensor_0", "a", capture=10.0)
    second = product("sensor_1", "b", capture=50.0)
    assert not ledger.register_capture_attempt(first)
    ledger.register_acquisitions([first], {3: 8.0})
    assert ledger.register_capture_attempt(second)
    assert ledger.duplicate_attempt_count == 1
