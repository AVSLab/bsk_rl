"""Synthetic SearchRSO information flow; no sensor or link physics are simulated."""

from bsk_rl.data import CatalogEstimate, CatalogPacket, GroundCatalog

REFERENCE = "seconds since 2026-01-01T00:00:00 UTC"


def solution(source, epoch):
    """Construct an illustrative estimate, not an orbit-determination result."""
    return CatalogEstimate(
        target_id="61",
        source_id=source,
        update_id=f"{source}:61:{epoch}",
        solution_time_s=epoch,
        state_epoch_s=epoch,
        state=(7e6, 0, 0, 0, 7500, 0),
        covariance_rtn=((100.0, 0, 0), (0, 400.0, 0), (0, 0, 100.0)),
        time_reference=REFERENCE,
    )


def run_example():
    """Relay actual received knowledge while keeping spacecraft catalogs separate."""
    prior = solution("initial-catalog", 0)
    ground = GroundCatalog(time_reference=REFERENCE, initial_estimates=[prior])
    spacecraft_a = GroundCatalog(time_reference=REFERENCE, initial_estimates=[prior])
    spacecraft_b = GroundCatalog(time_reference=REFERENCE, initial_estimates=[prior])
    operator, telescope_scheduler = ground, ground

    # A's local estimator reports an observation-derived solution.
    spacecraft_a.ingest(
        CatalogPacket("spacecraft-A", 10, (solution("spacecraft-A", 10),), REFERENCE),
        received_s=10,
    )
    assert ground.snapshot()["61"].estimate.solution_time_s == 0
    spacecraft_packet = spacecraft_a.make_packet(sender_id="spacecraft-A", created_s=11)
    # A pending or failed radio transfer does not call ground.ingest().
    ground.ingest(spacecraft_packet, received_s=20)  # successful radio delivery

    # A telescope processing adapter submits an estimate through the same API.
    telescope_packet = CatalogPacket(
        "telescope-network", 30, (solution("telescope-1", 25),), REFERENCE
    )
    ground.ingest(telescope_packet, received_s=32)
    assert operator.revision == telescope_scheduler.revision
    assert spacecraft_b.snapshot()["61"].estimate.solution_time_s == 0

    uplink = ground.make_packet(sender_id="ground", created_s=35)
    # The transport adapter calls this only after B's successful reception.
    spacecraft_b.ingest(uplink, received_s=40)
    entry = spacecraft_b.snapshot()["61"]
    assert entry.estimate.source_id == "telescope-1"  # original producer retained
    assert entry.sender_id == "ground"  # most recent forwarding hop
    return {
        "ground_revision": ground.revision,
        "B_solution_time_s": entry.estimate.solution_time_s,
        "B_receipt_time_s": entry.received_s,
        "original_source": entry.estimate.source_id,
    }


if __name__ == "__main__":
    print(run_example())
