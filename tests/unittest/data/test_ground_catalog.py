"""Catalog delivery, identity, validation, and episode lifecycle contracts."""

import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from bsk_rl.data.ground_catalog import CatalogEstimate, CatalogPacket, GroundCatalog

REF = "seconds since 2026-01-01T00:00:00 UTC"


def estimate(target="61", solution=10.0, source="spacecraft-A", **kwargs):
    values = dict(
        target_id=target,
        source_id=source,
        update_id=f"{source}:{target}:{solution}",
        solution_time_s=solution,
        state_epoch_s=solution,
        state=[7e6, 0, 0, 0, 7500, 0],
        covariance_rtn=np.diag([100.0, 400.0, 100.0]),
        time_reference=REF,
    )
    values.update(kwargs)
    return CatalogEstimate(**values)


def packet(*estimates, created=20.0, sender="antenna-1", reference=REF):
    return CatalogPacket(sender, created, estimates, reference)


def test_prior_is_actual_knowledge_and_reset_removes_deliveries():
    prior = estimate(solution=0)
    ground = GroundCatalog(time_reference=REF, initial_estimates=[prior])
    assert ground.snapshot()["61"].estimate == prior
    assert ground.revision == 1
    ground.ingest(packet(estimate()), received_s=21)
    ground.reset()
    assert ground.snapshot()["61"].estimate == prior
    assert ground.revision == 1
    assert GroundCatalog(time_reference=REF).snapshot() == {}


def test_newest_information_wins_per_target_not_state_or_receipt_time():
    ground = GroundCatalog(time_reference=REF)
    ground.ingest(packet(estimate(solution=10, state_epoch_s=20)), received_s=21)
    original = ground.snapshot()["61"]
    assert (
        ground.ingest(
            packet(estimate(solution=10, source="other"), created=30), received_s=31
        )
        == ()
    )
    assert (
        ground.ingest(
            packet(estimate(solution=5, state_epoch_s=40), created=40), received_s=41
        )
        == ()
    )
    assert ground.snapshot()["61"] == original
    assert ground.ingest(
        packet(estimate(solution=15), estimate("62", 12), created=50), received_s=51
    ) == ("61", "62")
    assert ground.revision == 2


def test_frozen_packet_keeps_provenance_during_overlapping_transfers():
    ground, receiver = (
        GroundCatalog(time_reference=REF),
        GroundCatalog(time_reference=REF),
    )
    first = estimate(solution=10)
    ground.ingest(packet(first), received_s=21)
    inflight = ground.make_packet(sender_id="ground", created_s=22)
    ground.ingest(
        packet(estimate(solution=25, source="telescope"), created=26), received_s=27
    )
    receiver.ingest(inflight, received_s=30)
    row = receiver.snapshot()["61"]
    assert row.estimate == first
    assert row.sender_id == "ground" and row.estimate.source_id == "spacecraft-A"
    assert ground.snapshot()["61"].estimate.source_id == "telescope"
    assert row.estimate.solution_time_s == 10 and row.received_s == 30


def test_no_delivery_means_no_remote_change_and_shared_ground_clients_agree():
    ground = GroundCatalog(time_reference=REF)
    operator, scheduler = ground, ground
    onboard = GroundCatalog(time_reference=REF)
    onboard.ingest(packet(estimate()), received_s=21)
    pending = onboard.make_packet(sender_id="spacecraft-A", created_s=22)
    assert not ground.snapshot()  # Link failure: transport does not ingest.
    ground.ingest(pending, received_s=30)
    assert operator.revision == scheduler.revision == 1
    assert operator.snapshot() == scheduler.snapshot()


def test_records_snapshots_and_serialized_copies_do_not_alias():
    state = [7e6, 0, 0, 0, 7500, 0]
    covariance = np.eye(3)
    e = estimate(state=state, covariance_rtn=covariance)
    state[0], covariance[0, 0] = 1, -1
    assert e.state[0] == 7e6 and e.covariance_rtn[0][0] == 1
    ground = GroundCatalog(time_reference=REF)
    ground.ingest(packet(e), received_s=21)
    snapshot = ground.snapshot()
    with pytest.raises(TypeError):
        snapshot["61"] = None
    with pytest.raises(FrozenInstanceError):
        snapshot["61"].estimate.source_id = "wrong"
    serialized = json.loads(json.dumps(ground.state_dict()))
    serialized["entries"][0]["estimate"]["state"][0] = 1
    assert ground.snapshot()["61"].estimate.state[0] == 7e6
    ground.ingest(packet(estimate(solution=25), created=26), received_s=27)
    assert snapshot["61"].estimate.solution_time_s == 10


@pytest.mark.parametrize(
    "change",
    [
        {"target_id": ""},
        {"source_id": ""},
        {"update_id": ""},
        {"time_reference": ""},
        {"solution_time_s": float("nan")},
        {"state_epoch_s": float("inf")},
        {"solution_time_s": "10"},
        {"solution_time_s": True},
        {"state_epoch_s": 9},
        {"state": [0] * 6},
        {"state": [1, 2]},
        {"state": [7e6, 0, 0, float("nan"), 7500, 0]},
        {"covariance_rtn": np.zeros((3, 3))},
        {"covariance_rtn": np.eye(6)},
        {"covariance_rtn": [[1, 1, 0], [0, 1, 0], [0, 0, 1]]},
        {"covariance_rtn": np.diag([-1, 1, 1])},
        {"state_frame": "ITRF"},
        {"covariance_frame": "J2000"},
        {"state_units": ("km", "km/s")},
        {"covariance_units": "km^2"},
    ],
)
def test_estimate_rejects_invalid_contract(change):
    with pytest.raises(ValueError):
        estimate(**change)


@pytest.mark.parametrize(
    "make",
    [
        lambda: packet(estimate(), estimate()),
        lambda: packet(estimate(), created=9),
        lambda: packet(estimate(), reference="different origin"),
        lambda: packet(None),
        lambda: packet(created=float("nan")),
    ],
)
def test_packet_rejects_invalid_contract(make):
    with pytest.raises(ValueError):
        make()


@pytest.mark.parametrize("revision", [-1, True, 1.5])
def test_packet_rejects_invalid_revision(revision):
    with pytest.raises(ValueError):
        CatalogPacket("ground", 20, (estimate(),), REF, catalog_revision=revision)


def test_rejected_delivery_is_atomic_including_mixed_references_and_time_order():
    ground = GroundCatalog(time_reference=REF)
    ground.ingest(packet(estimate()), received_s=21)
    before = ground.state_dict()
    for p, received in [
        (packet(estimate(solution=25), created=26), 25),
        (packet(estimate(), created=15), 20),
        (packet(reference="other"), 30),
        (packet(), float("nan")),
    ]:
        with pytest.raises(ValueError):
            ground.ingest(p, received_s=received)
        assert ground.state_dict() == before


def test_revision_delta_does_not_implicitly_acknowledge_packet():
    ground = GroundCatalog(time_reference=REF)
    ground.ingest(packet(estimate()), received_s=21)
    acknowledged = ground.revision
    ground.ingest(packet(estimate("62", 25), created=26), received_s=27)
    delta = ground.make_packet(
        sender_id="ground", created_s=28, since_revision=acknowledged
    )
    assert [e.target_id for e in delta.estimates] == ["62"]
    assert delta.catalog_revision == ground.revision
    assert delta == ground.make_packet(
        sender_id="ground", created_s=28, since_revision=acknowledged
    )
    for bad in [-1, 999, True, 1.5]:
        with pytest.raises(ValueError):
            ground.make_packet(sender_id="ground", created_s=28, since_revision=bad)
    ground.ingest(packet(estimate(solution=29), created=30), received_s=31)
    assert delta.catalog_revision == acknowledged + 1
    with pytest.raises(ValueError):
        ground.make_packet(sender_id="ground", created_s=20)


def test_equal_time_first_accepted_is_stable_for_explicit_delivery_order():
    def replay():
        ground = GroundCatalog(time_reference=REF)
        for source in ["telescope", "spacecraft-A"]:
            ground.ingest(packet(estimate(source=source)), received_s=21)
        return ground.state_dict()

    assert replay() == replay()
    assert replay()["entries"][0]["estimate"]["source_id"] == "telescope"


def test_prediction_is_explicit_preserves_information_and_does_not_mutate():
    ground = GroundCatalog(time_reference=REF)
    ground.ingest(packet(estimate()), received_s=21)
    before = ground.state_dict()

    def predict(e, time):
        return replace(
            e,
            state_epoch_s=time,
            state=(7e6, (time - e.state_epoch_s) * 7500, 0, 0, 7500, 0),
        )

    predicted = ground.snapshot_at(30, predict=predict)["61"]
    assert predicted.estimate.state_epoch_s == 30
    assert predicted.estimate.solution_time_s == 10
    assert predicted.estimate.state[1] == 150000
    assert ground.state_dict() == before
    with pytest.raises(ValueError):
        ground.snapshot_at(9, predict=predict)
    for change in [
        {"solution_time_s": 11},
        {"source_id": "wrong"},
        {"state_epoch_s": 29},
    ]:
        with pytest.raises(ValueError):
            ground.snapshot_at(
                30,
                predict=lambda e, t, change=change: replace(
                    e, **({"state_epoch_s": t} | change)
                ),
            )
    assert ground.state_dict() == before


def test_json_roundtrip_committed_state_and_reset_prior():
    prior = estimate(solution=0)
    ground = GroundCatalog(time_reference=REF, initial_estimates=[prior])
    ground.ingest(packet(estimate()), received_s=21)
    encoded = json.loads(json.dumps(ground.state_dict()))
    restored = GroundCatalog.from_state_dict(encoded)
    assert restored.state_dict() == ground.state_dict()
    p = ground.make_packet(sender_id="ground", created_s=22)
    assert CatalogPacket.from_dict(json.loads(json.dumps(p.to_dict()))) == p
    restored.reset()
    assert restored.snapshot()["61"].estimate == prior
    assert ground.snapshot()["61"].estimate.solution_time_s == 10
    empty = GroundCatalog(time_reference=REF)
    assert (
        GroundCatalog.from_state_dict(empty.state_dict()).state_dict()
        == empty.state_dict()
    )


def test_restore_cannot_substitute_an_equal_time_prior():
    ground = GroundCatalog(time_reference=REF, initial_estimates=[estimate(solution=0)])
    state = json.loads(json.dumps(ground.state_dict()))
    state["entries"][0]["estimate"]["state"][0] += 1
    with pytest.raises(ValueError, match="equal-time"):
        GroundCatalog.from_state_dict(state)


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.update(schema="unknown"),
        lambda s: s.update(revision=-1),
        lambda s: s.update(revision=True),
        lambda s: s.update(last_received_s=1),
        lambda s: s["entries"].append(s["entries"][0]),
        lambda s: s["entries"][0].update(revision=99),
        lambda s: s["entries"][0].update(received_s=0),
        lambda s: s["entries"][0]["estimate"].update(time_reference="other"),
        lambda s: s.update(entries=[]),
    ],
)
def test_restore_rejects_inconsistent_committed_state(change):
    ground = GroundCatalog(time_reference=REF, initial_estimates=[estimate(solution=0)])
    ground.ingest(packet(estimate()), received_s=21)
    state = json.loads(json.dumps(ground.state_dict()))
    change(state)
    with pytest.raises(ValueError):
        GroundCatalog.from_state_dict(state)
