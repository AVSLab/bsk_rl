"""Transmission duration, latency, loss, and independent record merge regressions."""

from dataclasses import fields, replace
from types import SimpleNamespace as NS
import numpy as np
import pytest
from bsk_rl.comm.completion_communication import CompletionCommunication
from bsk_rl.data.completion_catalog import CompletionCatalog, CompletionRecord
from bsk_rl.sats.roles import SpacecraftRole


def channel(**kwargs):
    clock = NS(sim_time=20.0)
    sensors = [
        NS(
            name=f"sensor_{i}",
            role=SpacecraftRole.SENSING_AGENT,
            simulator=clock,
            is_alive=lambda: True,
            dynamics=NS(r_BN_N=np.array([7e6, i * 1e3, 0])),
            data_store=NS(
                catalog=CompletionCatalog(f"sensor_{i}", [3], cooldown_s=100)
            ),
        )
        for i in range(2)
    ]
    comm = CompletionCommunication(link_mode="ideal", **kwargs)
    comm.link_satellites(sensors)
    comm.reset_overwrite_previous()
    comm.reset_post_sim_init()
    record = CompletionRecord("a", "sensor_0", 3, 0, 10, 20, True)
    sensors[0].data_store.catalog.merge_record(record, 20)
    return comm, sensors, clock, record


def test_30_second_broadcast_cannot_arrive_at_5_second_peer_boundary():
    comm, sensors, clock, record = channel(delay_s=7)
    comm.begin_transmission(sensors[0], 20, 30)
    clock.sim_time = 25
    comm.communicate()
    assert not sensors[1].data_store.catalog.records
    clock.sim_time = 50
    comm.communicate()
    assert not sensors[1].data_store.catalog.records
    assert comm.next_event_time(50) == 57
    clock.sim_time = 57
    comm.communicate()
    assert sensors[1].data_store.catalog.records == {"a": record}
    assert comm.backlog("sensor_0") == 0
    assert comm.backlog("sensor_1") == 0


def test_snapshot_excludes_future_captures_and_interrupted_radio_sends_nothing():
    comm, sensors, clock, record = channel()
    comm.begin_transmission(sensors[0], 20, 30)
    sensors[0].data_store.catalog.merge_record(
        replace(record, record_id="b", capture_time=21, completion_time=25), 25
    )
    assert [r.record_id for r in comm.transmissions["sensor_0"].records] == ["a"]
    comm.cancel_transmission("sensor_0", 25)
    clock.sim_time = 60
    comm.communicate()
    assert not comm.pending
    assert not sensors[1].data_store.catalog.records
    assert comm.backlog("sensor_0") == 2


def test_lost_packet_is_retried_until_acknowledged():
    comm, sensors, clock, record = channel(loss_probability=1)
    comm.begin_transmission(sensors[0], 20, 30)
    clock.sim_time = 50
    comm.communicate()
    assert comm.delivery_history[-1]["outcome"] == "lost"
    assert comm.backlog("sensor_0") == 1
    comm.loss_probability = 0
    comm.begin_transmission(sensors[0], 50, 30)
    clock.sim_time = 80
    comm.communicate()
    assert sensors[1].data_store.catalog.records == {"a": record}
    assert comm.backlog("sensor_0") == 0
    assert comm.backlog("sensor_1") == 0


def test_expired_transport_never_clears_a_durable_catalog_entry():
    comm, sensors, clock, record = channel(delay_s=20, ttl_s=10)
    sensors[1].data_store.catalog.merge_record(record, 20)
    comm.begin_transmission(sensors[0], 20, 30)
    clock.sim_time = 50
    comm.communicate()
    clock.sim_time = 70
    comm.communicate()
    assert comm.delivery_history[-1]["outcome"] == "expired"
    assert not sensors[1].data_store.catalog.is_eligible(3, 70)


@pytest.mark.parametrize(
    "case,shared", [("independent", False), ("ideal_completion", True)]
)
def test_automatic_exchange_only_in_ideal_reference(case, shared):
    comm, sensors, clock, record = channel(information_case=case)
    comm.communicate()
    assert bool(sensors[1].data_store.catalog.records) == shared
    assert not sensors[1].data_store.catalog.target(3).pending_record_ids
    assert {f.name for f in fields(record)} == {
        "record_id",
        "source_sensor",
        "target_id",
        "request_epoch",
        "capture_time",
        "completion_time",
        "qualified",
        "delivery_time",
    }


def test_los_loss_during_radio_discards_that_receiver():
    comm, sensors, clock, record = channel()
    comm.link_mode = "los"
    comm.begin_transmission(sensors[0], 20, 30)
    sensors[1].dynamics.r_BN_N = np.array([-7e6, 0, 0])
    clock.sim_time = 25
    comm.communicate()
    sensors[1].dynamics.r_BN_N = np.array([7e6, 1e3, 0])
    clock.sim_time = 50
    comm.communicate()
    assert not sensors[1].data_store.catalog.records
    assert comm.backlog("sensor_0") == 1


def test_failed_exposures_are_local_storage_metadata_not_shared_completions():
    comm, sensors, clock, record = channel(information_case="ideal_completion")
    sensors[0].data_store.catalog.merge_record(
        replace(record, record_id="failed", qualified=False), 20
    )
    comm.communicate()
    assert list(sensors[1].data_store.catalog.records) == ["a"]
    assert comm.backlog("sensor_0") == 0
