"""Bounded sender-local outgoing inventory, separately encoded for each peer."""

import numpy as np

from bsk_rl.obs.observations import Observation
from bsk_rl.obs.completion_observations import peer_snapshot

OUTGOING_FEATURE_ORDER = (
    "target_identity",
    "public_priority",
    "request_age",
    "capture_age",
    "completion_age",
    "quality",
    "qualified",
    "onboard",
    "delivered",
    "dropped",
    "unacknowledged_revision",
    "record_valid",
)


class OutgoingInventory(Observation):
    """Only own records and ideal ACK history, never recipient private state.

    Match the actual payload ordering/chunk first, then bound the observed prefix.
    Masked/invisible peers have zero rows. Truncation is declared in the schema;
    observing a prefix never changes the physical outgoing packet.
    """

    def __init__(self, slots, norm_time=12000.0):
        """Set a bounded record prefix per visible peer and time normalization."""
        super().__init__(name="outgoing_inventory")
        self.slots, self.norm_time = int(slots), float(norm_time)

    def get_obs(self):
        """Encode own packet records and ACKs without reading peer private state."""
        sensor, now = self.satellite, float(self.simulator.sim_time)
        channel = sensor.completion_communicator
        targets = {int(t.id): t for t in sensor.data_store.data.known}
        result = {}
        for index, peer in enumerate(peer_snapshot(sensor).peers):
            rows = np.zeros((self.slots, len(OUTGOING_FEATURE_ORDER)))
            if peer is not None:
                # Geometry-only control gets identical own information/capacity;
                # only transmit actions and remote receipts are disabled.
                entries = channel._inventory_delta(sensor.name, peer.name)[
                    : channel.inventory_chunk_records
                ]
                ack = channel.acknowledged.get((sensor.name, peer.name), {})
                for slot, entry in enumerate(entries[: self.slots]):
                    fact = entry.fact
                    rows[slot] = [
                        fact.target_id / max(1, len(targets) - 1),
                        targets[fact.target_id].priority / 10.0,
                        (now - fact.request_epoch) / self.norm_time,
                        (now - fact.capture_time) / self.norm_time,
                        (now - fact.completion_time) / self.norm_time,
                        fact.quality,
                        float(fact.qualified),
                        float(entry.status == "onboard"),
                        float(entry.status == "delivered"),
                        float(entry.status == "dropped"),
                        float(ack.get(entry.record_id, 0) < entry.version),
                        1.0,
                    ]
            result[f"peer_{index}"] = rows.reshape(-1)
        return result
