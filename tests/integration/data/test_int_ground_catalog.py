"""Native scheduling smoke for a catalog adapter, without sensor/link physics."""

from pathlib import Path
from runpy import run_path

from Basilisk.architecture import sysModel
from Basilisk.utilities import SimulationBaseClass, macros

from bsk_rl.data import CatalogEstimate, CatalogPacket, GroundCatalog


def test_catalog_ingestion_runs_on_native_clock_after_delivery_event():
    reference = "native simulation seconds"
    estimate = CatalogEstimate(
        "61",
        "telescope",
        "exposure-1",
        0.0,
        0.0,
        (7e6, 0, 0, 0, 7500, 0),
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        reference,
    )
    packet = CatalogPacket("ground-processing", 0.0, (estimate,), reference)
    ground, onboard = (
        GroundCatalog(time_reference=reference),
        GroundCatalog(time_reference=reference),
    )
    trace = []

    class DeliveryAdapter(sysModel.SysModel):
        def __init__(self):
            super().__init__()
            self.ModelTag = "catalogDeliveryAdapter"

        def UpdateState(self, current_nanos):
            now = current_nanos * macros.NANO2SEC
            if current_nanos == macros.sec2nano(1):
                ground.ingest(packet, received_s=now)
            if current_nanos == macros.sec2nano(3):
                onboard.ingest(
                    ground.make_packet(sender_id="ground", created_s=now),
                    received_s=now,
                )
            trace.append((now, bool(ground.snapshot()), bool(onboard.snapshot())))

    sim = SimulationBaseClass.SimBaseClass()
    process = sim.CreateNewProcess("catalogProcess")
    process.addTask(sim.CreateNewTask("catalogTask", macros.sec2nano(1)))
    adapter = DeliveryAdapter()
    sim.AddModelToTask("catalogTask", adapter)
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(3))
    sim.ExecuteSimulation()
    assert trace == [
        (0.0, False, False),
        (1.0, True, False),
        (2.0, True, False),
        (3.0, True, True),
    ]
    assert onboard.snapshot()["61"].estimate.source_id == "telescope"
    assert onboard.snapshot()["61"].received_s == 3


def test_public_information_flow_example():
    example = Path(__file__).resolve().parents[3] / "examples" / "ground_catalog.py"
    result = run_path(str(example))["run_example"]()
    assert result == {
        "ground_revision": 3,
        "B_solution_time_s": 25.0,
        "B_receipt_time_s": 40.0,
        "original_source": "telescope-1",
    }
