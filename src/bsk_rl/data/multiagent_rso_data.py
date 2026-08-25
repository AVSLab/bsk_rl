"""Separated physical, local-knowledge, and team-truth data for RSO imaging."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Optional


@dataclass(frozen=True, slots=True)
class ImageProductRecord:
    """One physical image product and its immutable provenance."""

    record_id: str
    source_sensor: str
    target_id: int
    capture_time: float
    delivery_time: Optional[float]
    quality: float
    storage_owner: str

    def delivered(self, delivery_time: float) -> "ImageProductRecord":
        """Return a delivered copy while preserving capture provenance."""
        return replace(self, delivery_time=float(delivery_time))


class SensorProductStore:
    """Physical image products owned by exactly one sensing spacecraft."""

    def __init__(self, sensor_id: str) -> None:
        """Create an empty product store owned by ``sensor_id``."""
        self.sensor_id = str(sensor_id)
        self._products: dict[str, ImageProductRecord] = {}

    @property
    def products(self) -> tuple[ImageProductRecord, ...]:
        """Return onboard products in deterministic record-ID order."""
        return tuple(self._products[key] for key in sorted(self._products))

    def store(self, product: ImageProductRecord) -> None:
        """Put a locally captured, locally owned product onboard."""
        if product.storage_owner != self.sensor_id:
            raise ValueError(
                f"{self.sensor_id} cannot store product owned by {product.storage_owner}."
            )
        if product.source_sensor != self.sensor_id:
            raise ValueError(
                "Image relay is disabled: source and storage owner must match."
            )
        existing = self._products.get(product.record_id)
        if existing is not None and existing != product:
            raise ValueError(f"Conflicting product record_id {product.record_id!r}.")
        self._products[product.record_id] = product

    def downlink(self, record_id: str, delivery_time: float) -> ImageProductRecord:
        """Remove and deliver an onboard product owned by this sensor."""
        try:
            product = self._products.pop(str(record_id))
        except KeyError as exc:
            raise KeyError(
                f"{self.sensor_id} cannot downlink non-onboard product {record_id!r}."
            ) from exc
        return product.delivered(delivery_time)

    def records_for_target(self, target_id: int) -> tuple[ImageProductRecord, ...]:
        """Return all onboard records associated with one target."""
        return tuple(
            product for product in self.products if product.target_id == int(target_id)
        )


@dataclass(slots=True)
class LocalTargetKnowledge:
    """One sensor's latest catalog knowledge for one target."""

    target_id: int
    latest_acquisition_time: Optional[float] = None
    latest_delivery_time: Optional[float] = None
    cooldown_until: float = float("-inf")
    local_cooldown_until: float = float("-inf")
    remote_cooldown_until_by_source: dict[str, float] = field(default_factory=dict)
    pending_record_ids: tuple[str, ...] = ()
    remote_pending_sources: tuple[str, ...] = ()
    remote_pending_expiry_by_source: dict[str, float] = field(default_factory=dict)
    last_update_time: float = float("-inf")
    last_update_source: Optional[str] = None


class LocalCatalogKnowledge:
    """Private, explicitly updated catalog state for one sensing spacecraft."""

    def __init__(self, sensor_id: str, target_ids: Iterable[int]) -> None:
        """Initialize private knowledge for the supplied catalog target IDs."""
        self.sensor_id = str(sensor_id)
        self._targets = {
            int(target_id): LocalTargetKnowledge(target_id=int(target_id))
            for target_id in target_ids
        }

    @property
    def targets(self) -> Mapping[int, LocalTargetKnowledge]:
        """Return the target-indexed local catalog mapping."""
        return self._targets

    def target(self, target_id: int) -> LocalTargetKnowledge:
        """Return local knowledge for ``target_id``."""
        return self._targets[int(target_id)]

    @staticmethod
    def _refresh_effective_state(
        state: LocalTargetKnowledge, sim_time: float
    ) -> None:
        """Expire transient remote status and refresh the effective cooldown."""
        active_remote_pending = {
            source: expiry
            for source, expiry in state.remote_pending_expiry_by_source.items()
            if float(expiry) >= float(sim_time)
        }
        state.remote_pending_expiry_by_source = active_remote_pending
        state.remote_pending_sources = tuple(sorted(active_remote_pending))
        cooldowns = [state.local_cooldown_until]
        cooldowns.extend(state.remote_cooldown_until_by_source.values())
        state.cooldown_until = max(cooldowns, default=float("-inf"))

    def is_privately_eligible(self, target_id: int, sim_time: float) -> bool:
        """Return eligibility from this sensor's own acquisitions and deliveries."""
        state = self.target(target_id)
        return (
            not state.pending_record_ids
            and float(sim_time) >= state.local_cooldown_until
        )

    def is_eligible(self, target_id: int, sim_time: float) -> bool:
        """Return local-policy eligibility without consulting global truth."""
        state = self.target(target_id)
        self._refresh_effective_state(state, sim_time)
        return (
            self.is_privately_eligible(target_id, sim_time)
            and not state.remote_pending_sources
            and float(sim_time) >= state.cooldown_until
        )

    def record_capture(
        self,
        product: ImageProductRecord,
        cooldown_until: Optional[float] = None,
    ) -> None:
        """Record a product captured by this sensor in local knowledge."""
        if product.source_sensor != self.sensor_id:
            raise ValueError("A local capture must originate at the local sensor.")
        state = self.target(product.target_id)
        state.pending_record_ids = tuple(
            dict.fromkeys((*state.pending_record_ids, product.record_id))
        )
        prior = state.latest_acquisition_time
        state.latest_acquisition_time = (
            product.capture_time if prior is None else max(prior, product.capture_time)
        )
        if cooldown_until is not None:
            state.local_cooldown_until = max(
                state.local_cooldown_until, float(cooldown_until)
            )
            self._refresh_effective_state(state, product.capture_time)
        state.last_update_time = float(product.capture_time)
        state.last_update_source = self.sensor_id

    def record_delivery(
        self,
        product: ImageProductRecord,
        cooldown_until: Optional[float] = None,
    ) -> None:
        """Record ground delivery of a locally owned product."""
        if product.delivery_time is None:
            raise ValueError("Delivered local knowledge requires a delivery timestamp.")
        state = self.target(product.target_id)
        state.pending_record_ids = tuple(
            record_id
            for record_id in state.pending_record_ids
            if record_id != product.record_id
        )
        prior = state.latest_delivery_time
        state.latest_delivery_time = (
            product.delivery_time
            if prior is None
            else max(prior, product.delivery_time)
        )
        if cooldown_until is not None:
            state.local_cooldown_until = max(
                state.local_cooldown_until, float(cooldown_until)
            )
            self._refresh_effective_state(state, product.delivery_time)
        state.last_update_time = float(product.delivery_time)
        state.last_update_source = self.sensor_id

    def merge_status(
        self,
        *,
        target_id: int,
        acquisition_time: Optional[float],
        delivery_time: Optional[float],
        cooldown_until: Optional[float],
        lifecycle_status: Optional[str],
        update_time: float,
        expiry_time: Optional[float],
        source_sensor: str,
    ) -> bool:
        """Merge explicit communicated metadata if it is newer."""
        state = self.target(target_id)
        update_key = (float(update_time), str(source_sensor))
        current_key = (state.last_update_time, state.last_update_source or "")
        if update_key <= current_key:
            return False
        if acquisition_time is not None:
            prior = state.latest_acquisition_time
            state.latest_acquisition_time = (
                float(acquisition_time)
                if prior is None
                else max(prior, float(acquisition_time))
            )
        if delivery_time is not None:
            prior = state.latest_delivery_time
            state.latest_delivery_time = (
                float(delivery_time)
                if prior is None
                else max(prior, float(delivery_time))
            )
        remote_cooldowns = dict(state.remote_cooldown_until_by_source)
        if cooldown_until is not None:
            remote_cooldowns[str(source_sensor)] = max(
                remote_cooldowns.get(str(source_sensor), float("-inf")),
                float(cooldown_until),
            )
        elif lifecycle_status in {"delivered", "eligible"}:
            remote_cooldowns.pop(str(source_sensor), None)
        remote_pending_expiry = dict(state.remote_pending_expiry_by_source)
        if lifecycle_status == "pending_verification":
            remote_pending_expiry[str(source_sensor)] = float(
                expiry_time if expiry_time is not None else update_time
            )
        elif lifecycle_status in {"delivered", "eligible", "cooldown"}:
            remote_pending_expiry.pop(str(source_sensor), None)
        state.remote_pending_expiry_by_source = remote_pending_expiry
        state.remote_pending_sources = tuple(sorted(remote_pending_expiry))
        state.remote_cooldown_until_by_source = remote_cooldowns
        self._refresh_effective_state(state, update_time)
        state.last_update_time = float(update_time)
        state.last_update_source = str(source_sensor)
        return True


@dataclass(frozen=True, slots=True)
class ServiceLedgerEntry:
    """Ground-verified team-service result for one image product."""

    product: ImageProductRecord
    unique_service: bool
    successful_duplicate: bool
    credited_value: float


class TeamServiceLedger:
    """Global truth for non-double-counted reward and analysis only."""

    def __init__(
        self,
        *,
        cooldown_s: float,
        quality_threshold: float,
        simultaneous_tolerance_s: float = 1e-6,
    ) -> None:
        """Initialize team truth with service and quality definitions."""
        self.cooldown_s = float(cooldown_s)
        self.quality_threshold = float(quality_threshold)
        self.simultaneous_tolerance_s = float(simultaneous_tolerance_s)
        self.entries: list[ServiceLedgerEntry] = []
        self.capture_attempts: list[ImageProductRecord] = []
        self.duplicate_attempt_count = 0
        self.duplicate_attempt_record_ids: set[str] = set()
        self.successful_duplicate_count = 0
        self._unique_acquisition_count = 0
        self._acquisition_team_value = 0.0
        self._latest_unique_capture_by_target: dict[int, float] = {}
        self._latest_credited_acquisition_by_target: dict[int, float] = {}

    def register_capture_attempt(self, product: ImageProductRecord) -> bool:
        """Log a capture and return whether global truth already had current service."""
        latest = self._latest_credited_acquisition_by_target.get(product.target_id)
        duplicate = (
            latest is not None and product.capture_time < latest + self.cooldown_s
        )
        self.capture_attempts.append(product)
        if duplicate:
            self.duplicate_attempt_count += 1
            self.duplicate_attempt_record_ids.add(product.record_id)
        return duplicate

    def register_acquisitions(
        self,
        products: Iterable[ImageProductRecord],
        target_priorities: Mapping[int, float],
    ) -> dict[str, float]:
        """Allocate one deterministic team acquisition value per service interval."""
        captures = sorted(
            products,
            key=lambda p: (
                p.target_id,
                p.capture_time,
                p.source_sensor,
                p.record_id,
            ),
        )
        credit: dict[str, float] = {}
        i = 0
        while i < len(captures):
            first = captures[i]
            group = [first]
            i += 1
            while i < len(captures):
                candidate = captures[i]
                if (
                    candidate.target_id != first.target_id
                    or abs(candidate.capture_time - first.capture_time)
                    > self.simultaneous_tolerance_s
                ):
                    break
                group.append(candidate)
                i += 1
            qualified = [
                product
                for product in group
                if product.quality >= self.quality_threshold
            ]
            if len(qualified) > 1:
                for duplicate_product in qualified[1:]:
                    if (
                        duplicate_product.record_id
                        not in self.duplicate_attempt_record_ids
                    ):
                        self.duplicate_attempt_record_ids.add(
                            duplicate_product.record_id
                        )
                        self.duplicate_attempt_count += 1
            latest = self._latest_credited_acquisition_by_target.get(first.target_id)
            unique = bool(qualified) and (
                latest is None or first.capture_time >= latest + self.cooldown_s
            )
            if not unique:
                continue
            self._latest_credited_acquisition_by_target[first.target_id] = (
                first.capture_time
            )
            team_value = float(target_priorities[first.target_id])
            self._unique_acquisition_count += 1
            self._acquisition_team_value += team_value
            share = team_value / len(qualified)
            for product in qualified:
                credit[product.source_sensor] = (
                    credit.get(product.source_sensor, 0.0) + share
                )
        return credit

    def register_deliveries(
        self,
        products: Iterable[ImageProductRecord],
        target_priorities: Mapping[int, float],
    ) -> dict[str, float]:
        """Allocate deterministic source credit without changing the team total."""
        delivered = sorted(
            products,
            key=lambda p: (
                p.target_id,
                p.capture_time,
                p.source_sensor,
                p.record_id,
            ),
        )
        credit: dict[str, float] = {}
        i = 0
        while i < len(delivered):
            first = delivered[i]
            group = [first]
            i += 1
            while i < len(delivered):
                candidate = delivered[i]
                if (
                    candidate.target_id != first.target_id
                    or abs(candidate.capture_time - first.capture_time)
                    > self.simultaneous_tolerance_s
                ):
                    break
                group.append(candidate)
                i += 1

            qualified = [
                product
                for product in group
                if product.delivery_time is not None
                and product.quality >= self.quality_threshold
            ]
            latest = self._latest_unique_capture_by_target.get(first.target_id)
            unique = bool(qualified) and (
                latest is None or first.capture_time >= latest + self.cooldown_s
            )
            share = (
                float(target_priorities[first.target_id]) / len(qualified)
                if unique
                else 0.0
            )
            if unique:
                self._latest_unique_capture_by_target[first.target_id] = (
                    first.capture_time
                )

            for product in group:
                is_qualified = product in qualified
                successful_duplicate = bool(is_qualified and not unique)
                if successful_duplicate:
                    self.successful_duplicate_count += 1
                value = share if is_qualified and unique else 0.0
                self.entries.append(
                    ServiceLedgerEntry(
                        product=product,
                        unique_service=bool(
                            is_qualified and unique and product is qualified[0]
                        ),
                        successful_duplicate=successful_duplicate,
                        credited_value=value,
                    )
                )
                credit[product.source_sensor] = (
                    credit.get(product.source_sensor, 0.0) + value
                )
        return credit

    @property
    def team_value(self) -> float:
        """Return non-double-counted delivered team value."""
        return sum(entry.credited_value for entry in self.entries)

    @property
    def unique_service_count(self) -> int:
        """Return the number of unique team-service events."""
        return sum(entry.unique_service for entry in self.entries)

    @property
    def unique_acquisition_count(self) -> int:
        """Return the number of non-duplicated qualifying team acquisitions."""
        return self._unique_acquisition_count

    @property
    def acquisition_team_value(self) -> float:
        """Return non-double-counted priority value acquired by the team."""
        return self._acquisition_team_value


__all__ = [
    "ImageProductRecord",
    "LocalCatalogKnowledge",
    "LocalTargetKnowledge",
    "SensorProductStore",
    "ServiceLedgerEntry",
    "TeamServiceLedger",
]
