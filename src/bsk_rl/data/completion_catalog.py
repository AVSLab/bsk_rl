"""Record-based local knowledge for persistent completion-only cooperation.

An entry describes a completed exposure, never a reservation or a future action.
Catalogs contain metadata; owning a record does not confer ownership of its image.
"""

from dataclasses import dataclass, replace
from math import isfinite

from bsk_rl.data.multiagent_rso_data import ImageProductRecord, LocalCatalogKnowledge


@dataclass(frozen=True)
class CompletionRecord:
    """Immutable source facts. Delivery is a later version of the same exposure."""

    record_id: str
    source_sensor: str
    target_id: int
    request_epoch: float
    capture_time: float
    completion_time: float
    qualified: bool
    delivery_time: float | None = None

    @property
    def version(self) -> int:
        """Return the monotone acquisition or delivery revision for ACK tracking."""
        return 1 if self.delivery_time is None else 2


class CompletionCatalog(LocalCatalogKnowledge):
    """Sensor-owned facts with separate physical pending state and receive history.

    ``cooldown_s`` is a revisit interval anchored at capture time. Packet expiry
    never changes it. A new request epoch invalidates old service for eligibility
    while retaining those records for history and ground-age calculations.
    """

    def __init__(self, sensor_id, target_ids, *, cooldown_s=0.0, quality_threshold=0.5):
        """Initialize receiver-owned facts for the mission catalog target IDs."""
        super().__init__(sensor_id, target_ids)
        self.cooldown_s = float(cooldown_s)
        self.quality_threshold = float(quality_threshold)
        self.records: dict[str, CompletionRecord] = {}
        self.received_at: dict[str, float] = {}
        # Capture knowledge and its later ground-delivery upgrade can arrive in
        # different packets. Retain the first receipt of each revision separately.
        self.version_received_at: dict[str, dict[int, float]] = {}
        self.request_epochs = {int(t): 0.0 for t in self.targets}
        self.revision = 0

    def set_request_epoch(self, target_id: int, epoch: float) -> None:
        """Open a new service requirement; never erase older exposure history."""
        target_id, epoch = int(target_id), float(epoch)
        if not isfinite(epoch) or epoch < self.request_epochs[target_id]:
            raise ValueError("Request epochs must be finite and nondecreasing.")
        if epoch != self.request_epochs[target_id]:
            self.request_epochs[target_id] = epoch
            self.revision += 1
            self._refresh_target(target_id)

    def merge_record(self, record: CompletionRecord, received_at: float) -> bool:
        """Merge by exposure identity, independent of packet ordering or sender.

        A relayed record retains its original source and times. An old acquisition
        packet cannot undo delivery; an old delivery for another exposure remains
        useful. Conflicting provenance is rejected rather than silently overwritten.
        """
        self.target(record.target_id)  # Reject unknown catalog IDs.
        times = [
            record.capture_time,
            record.completion_time,
            record.request_epoch,
            received_at,
        ]
        if record.delivery_time is not None:
            times.append(record.delivery_time)
        if not all(isfinite(t) for t in times):
            raise ValueError("Completion times must be finite.")
        if (
            not 0
            <= record.request_epoch
            <= record.capture_time
            <= record.completion_time
            <= received_at
        ):
            raise ValueError(
                "A completion cannot precede its request or arrive from the future."
            )
        if (
            record.delivery_time is not None
            and not record.completion_time <= record.delivery_time <= received_at
        ):
            raise ValueError("Invalid ground delivery timestamp.")
        previous = self.records.get(record.record_id)
        if previous is not None:
            if replace(previous, delivery_time=None) != replace(
                record, delivery_time=None
            ):
                raise ValueError(f"Conflicting exposure provenance: {record.record_id}")
            if previous.delivery_time is not None:
                if (
                    record.delivery_time is not None
                    and previous.delivery_time != record.delivery_time
                ):
                    raise ValueError(
                        "Conflicting delivery times for one physical product."
                    )
                self.version_received_at.setdefault(record.record_id, {}).setdefault(
                    record.version, float(received_at)
                )
                return False
            if record.delivery_time is None:
                return False
        self.records[record.record_id] = record
        # The first receipt of the qualifying capture is retained through upgrades.
        self.received_at.setdefault(record.record_id, float(received_at))
        self.version_received_at.setdefault(record.record_id, {}).setdefault(
            record.version, float(received_at)
        )
        self.revision += 1
        self._refresh_target(record.target_id)
        return True

    def service_records(self, target_id, *, local_only=False):
        """Select qualified facts matching the current request generation."""
        epoch = self.request_epochs[int(target_id)]
        return [
            r
            for r in self.records.values()
            if r.target_id == int(target_id)
            and r.request_epoch == epoch
            and r.qualified
            and (not local_only or r.source_sensor == self.sensor_id)
        ]

    def _refresh_target(self, target_id):
        """Materialize compatibility summaries without mixing exposure identities."""
        state = self.target(target_id)
        facts = self.service_records(target_id)
        local = self.service_records(target_id, local_only=True)
        state.latest_acquisition_time = max(
            (r.capture_time for r in facts), default=None
        )
        state.latest_delivery_time = max(
            (r.delivery_time for r in facts if r.delivery_time is not None),
            default=None,
        )
        state.local_cooldown_until = max(
            (r.capture_time + self.cooldown_s for r in local), default=float("-inf")
        )
        state.cooldown_until = max(
            (r.capture_time + self.cooldown_s for r in facts), default=float("-inf")
        )
        # Completed facts are durable; remote pending leases/intents are not used.
        state.remote_pending_sources = ()
        related = [r for r in self.records.values() if r.target_id == target_id]
        state.last_update_time = max(
            (r.delivery_time or r.completion_time for r in related),
            default=float("-inf"),
        )

    def record_capture(self, product: ImageProductRecord, *, completion_time=None):
        """Publish only after exposure/hold completion, with onboard quality known."""
        if product.source_sensor != self.sensor_id:
            raise ValueError("Local captures must belong to this sensor.")
        state = self.target(product.target_id)
        state.pending_record_ids = tuple(
            dict.fromkeys((*state.pending_record_ids, product.record_id))
        )
        completed = (
            (
                product.completion_time
                if product.completion_time is not None
                else product.capture_time
            )
            if completion_time is None
            else float(completion_time)
        )
        return self.merge_record(
            CompletionRecord(
                record_id=product.record_id,
                source_sensor=product.source_sensor,
                target_id=product.target_id,
                request_epoch=product.request_epoch,
                capture_time=product.capture_time,
                completion_time=completed,
                qualified=product.quality >= self.quality_threshold,
            ),
            completed,
        )

    def record_delivery(self, product, cooldown_until=None):
        """Attach ground receipt to the exact locally owned exposure."""
        if product.source_sensor != self.sensor_id or product.delivery_time is None:
            raise ValueError("Delivery requires a locally owned, delivered product.")
        record = self.records[product.record_id]
        state = self.target(product.target_id)
        state.pending_record_ids = tuple(
            r for r in state.pending_record_ids if r != product.record_id
        )
        return self.merge_record(
            replace(record, delivery_time=product.delivery_time), product.delivery_time
        )

    def is_eligible(self, target_id, sim_time):
        # Local storage uses one partition per target: do not overwrite an onboard
        # image. Remote image ownership never creates a local storage reservation.
        """Combine local partition ownership with own and received service freshness."""
        state = self.target(target_id)
        return not state.pending_record_ids and not self.satisfied(target_id, sim_time)

    def is_privately_eligible(self, target_id, sim_time):
        """Evaluate the same eligibility rule with only own completion facts."""
        state = self.target(target_id)
        return not state.pending_record_ids and not self.satisfied(
            target_id, sim_time, local_only=True
        )

    def satisfied(self, target_id, sim_time, *, local_only=False):
        """Test whether a known matching exposure still covers the revisit interval."""
        return any(
            r.capture_time + self.cooldown_s > sim_time
            for r in self.service_records(target_id, local_only=local_only)
        )

    def freshest_delivered_capture(self, target_id):
        """Age of information is based on acquisition, not newest packet/downlink."""
        return max(
            (
                r.capture_time
                for r in self.service_records(target_id)
                if r.delivery_time is not None
            ),
            default=None,
        )


__all__ = ["CompletionCatalog", "CompletionRecord"]
