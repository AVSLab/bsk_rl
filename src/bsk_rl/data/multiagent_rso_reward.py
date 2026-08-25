"""Multi-sensor RSO reward with local lifecycle state and global team accounting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np
from Basilisk.utilities import orbitalMotion

from bsk_rl.data.base import GlobalReward
from bsk_rl.data.multiagent_rso_data import (
    ImageProductRecord,
    LocalCatalogKnowledge,
)
from bsk_rl.data.rso_targets_data import (
    RSOTargetImageData,
    RSOTargetImageStore,
)
from bsk_rl.sats.roles import SpacecraftRole

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _ServiceEntry:
    """One ground-verified result retained privately by the global rewarder."""

    product: ImageProductRecord
    unique_service: bool
    successful_duplicate: bool
    credited_value: float


class _TeamServiceAccounting:
    """Private, non-observable team accounting used for reward and diagnostics."""

    def __init__(
        self,
        *,
        cooldown_s: float,
        quality_threshold: float,
        simultaneous_tolerance_s: float = 1e-6,
    ) -> None:
        self.cooldown_s = float(cooldown_s)
        self.quality_threshold = float(quality_threshold)
        self.simultaneous_tolerance_s = float(simultaneous_tolerance_s)
        self.entries: list[_ServiceEntry] = []
        self.capture_attempts: list[ImageProductRecord] = []
        self.duplicate_attempt_count = 0
        self.duplicate_attempt_record_ids: set[str] = set()
        self.successful_duplicate_count = 0
        self._unique_acquisition_count = 0
        self._acquisition_team_value = 0.0
        self._latest_unique_capture_by_target: dict[int, float] = {}
        self._latest_credited_acquisition_by_target: dict[int, float] = {}

    def register_capture_attempt(self, product: ImageProductRecord) -> bool:
        latest = self._latest_credited_acquisition_by_target.get(product.target_id)
        duplicate = (
            latest is not None and product.capture_time < latest + self.cooldown_s
        )
        self.capture_attempts.append(product)
        if duplicate:
            self.duplicate_attempt_count += 1
            self.duplicate_attempt_record_ids.add(product.record_id)
        return duplicate

    def _groups(self, products: Iterable[ImageProductRecord]):
        ordered = sorted(
            products,
            key=lambda product: (
                product.target_id,
                product.capture_time,
                product.source_sensor,
                product.record_id,
            ),
        )
        index = 0
        while index < len(ordered):
            first = ordered[index]
            group = [first]
            index += 1
            while index < len(ordered):
                candidate = ordered[index]
                if (
                    candidate.target_id != first.target_id
                    or abs(candidate.capture_time - first.capture_time)
                    > self.simultaneous_tolerance_s
                ):
                    break
                group.append(candidate)
                index += 1
            yield first, group

    def register_acquisitions(
        self,
        products: Iterable[ImageProductRecord],
        target_priorities: Mapping[int, float],
    ) -> dict[str, float]:
        credit: dict[str, float] = {}
        for first, group in self._groups(products):
            qualified = [
                product
                for product in group
                if product.quality >= self.quality_threshold
            ]
            for duplicate in qualified[1:]:
                if duplicate.record_id not in self.duplicate_attempt_record_ids:
                    self.duplicate_attempt_record_ids.add(duplicate.record_id)
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
        credit: dict[str, float] = {}
        for first, group in self._groups(products):
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
            if unique:
                self._latest_unique_capture_by_target[first.target_id] = (
                    first.capture_time
                )
            share = (
                float(target_priorities[first.target_id]) / len(qualified)
                if unique
                else 0.0
            )
            for product in group:
                successful_duplicate = product in qualified and not unique
                if successful_duplicate:
                    self.successful_duplicate_count += 1
                value = share if product in qualified and unique else 0.0
                self.entries.append(
                    _ServiceEntry(
                        product=product,
                        unique_service=bool(
                            product in qualified and unique and product is qualified[0]
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
        return sum(entry.credited_value for entry in self.entries)

    @property
    def unique_service_count(self) -> int:
        return sum(entry.unique_service for entry in self.entries)

    @property
    def unique_acquisition_count(self) -> int:
        return self._unique_acquisition_count

    @property
    def acquisition_team_value(self) -> float:
        return self._acquisition_team_value


class MultiSensorRSOTargetImageStore(RSOTargetImageStore):
    """One sensor's standard datastore, local catalog, and product metadata.

    Basilisk's storage unit remains physical truth. The Python product records below
    describe only products that this same sensor physically owns; they are never copied
    by the communication layer.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize standard RSO data plus local coordination metadata."""
        super().__init__(*args, **kwargs)
        target_ids = [int(target.id) for target in self.data.known]
        self.catalog = LocalCatalogKnowledge(self.satellite.name, target_ids)
        self._products: dict[str, ImageProductRecord] = {}

    @property
    def products(self) -> tuple[ImageProductRecord, ...]:
        """Return the metadata for products physically onboard this sensor."""
        return tuple(self._products[key] for key in sorted(self._products))

    def store_product(self, product: ImageProductRecord) -> None:
        """Register one product after Basilisk storage records its capture."""
        if product.storage_owner != self.satellite.name:
            raise ValueError(
                f"{self.satellite.name} cannot store product owned by "
                f"{product.storage_owner}."
            )
        if product.source_sensor != self.satellite.name:
            raise ValueError("Image-product relay is disabled in this implementation.")
        existing = self._products.get(product.record_id)
        if existing is not None and existing != product:
            raise ValueError(f"Conflicting product record_id {product.record_id!r}.")
        self._products[product.record_id] = product

    def records_for_target(self, target_id: int) -> tuple[ImageProductRecord, ...]:
        """Return onboard products associated with one target."""
        return tuple(
            product for product in self.products if product.target_id == int(target_id)
        )

    def downlink_product(
        self, record_id: str, delivery_time: float
    ) -> ImageProductRecord:
        """Remove and return one locally owned product at ground delivery."""
        try:
            product = self._products.pop(str(record_id))
        except KeyError as exc:
            raise KeyError(
                f"{self.satellite.name} cannot downlink non-onboard product "
                f"{record_id!r}."
            ) from exc
        return product.delivered(delivery_time)

    def get_log_state(self) -> np.ndarray:
        """Read sensor storage or return an inert passive-target state."""
        if self.satellite.role is SpacecraftRole.PASSIVE_TARGET:
            return np.array([], dtype=float)
        return super().get_log_state()

    def compare_log_states(
        self, old_state: np.ndarray, new_state: np.ndarray
    ) -> RSOTargetImageData:
        """Convert sensor storage increases into provenance-preserving captures."""
        if self.satellite.role is SpacecraftRole.PASSIVE_TARGET:
            return RSOTargetImageData()

        pending_records_by_id: dict[int, list[dict[str, Any]]] = {}
        imaged = []
        for idx in np.where(new_state - old_state > 0)[0]:
            target_name = self._target_name_from_storage_index(int(idx))
            target = self._target_from_name(target_name)
            if target is None:
                continue
            record = self._pending_capture_record(
                target, int(idx), old_state, new_state
            )
            pending_records_by_id.setdefault(int(target.id), []).append(record)
            if self._record_quality_value(record) >= self._quality_threshold():
                imaged.append(target)
        return RSOTargetImageData(
            imaged=imaged,
            pending_image_records_by_id=pending_records_by_id,
            hide_pending_targets=self.hide_pending_targets,
        )

    @staticmethod
    def _record_quality_value(record: dict[str, Any]) -> float:
        value = record.get("mean_hold_shadow_factor")
        if value is None:
            value = record.get("capture_shadow_factor", 0.0)
        return float(value)


class MultiSensorRSOTargetImageReward(GlobalReward):
    """Priority-weighted AMOS reward generalized to multiple sensing spacecraft.

    Imaging and delivered-ground-value terms retain the AMOS reward mixture. Team
    truth prevents duplicate service from being counted twice, while credit is assigned
    to source sensors and simultaneous qualifying events split a fixed team total.
    """

    datastore_type = MultiSensorRSOTargetImageStore

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        reward_fn: Callable[[float], float] = lambda value: value,
        reimage_cooldown_orbits: float = 2.0,
        fallback_orbit_period_s: float = 95.0 * 60.0,
        quality_threshold: float = 0.5,
        hide_pending_targets: bool = True,
        duplicate_penalty: float = 0.0,
        communication_penalty: float = 0.0,
    ) -> None:
        """Configure the AMOS reward mixture and separate optional penalties."""
        super().__init__()
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        self.alpha = float(alpha)
        self.reward_fn = reward_fn
        self.reimage_cooldown_orbits = float(reimage_cooldown_orbits)
        self.fallback_orbit_period_s = float(fallback_orbit_period_s)
        self.quality_threshold = float(quality_threshold)
        self.hide_pending_targets = bool(hide_pending_targets)
        self.duplicate_penalty = float(duplicate_penalty)
        self.communication_penalty = float(communication_penalty)

    @property
    def sensing_satellites(self):
        """Return all explicitly marked sensing spacecraft."""
        return [
            satellite
            for satellite in self.scenario.satellites
            if satellite.role is SpacecraftRole.SENSING_AGENT
        ]

    def reset_overwrite_previous(self) -> None:
        """Reset per-episode local and team accounting."""
        super().reset_overwrite_previous()
        self.old_storage_by_sensor: dict[str, np.ndarray] = {}
        self.per_sensor_metrics: dict[str, dict[str, float]] = {}
        self.team_summary: dict[str, float] = {}
        self.reimage_cooldown_s: Optional[float] = None
        self._team_accounting: Optional[_TeamServiceAccounting] = None

    @property
    def service_entries(self) -> tuple[_ServiceEntry, ...]:
        """Read-only team service history for evaluation diagnostics."""
        if self._team_accounting is None:
            return ()
        return tuple(self._team_accounting.entries)

    def initial_data(self, satellite) -> RSOTargetImageData:
        """Give catalog knowledge only to sensing spacecraft."""
        known = (
            self.scenario.target_spacecrafts
            if satellite.role is SpacecraftRole.SENSING_AGENT
            else []
        )
        return RSOTargetImageData(
            known=known,
            hide_pending_targets=self.hide_pending_targets,
        )

    def create_data_store(self, satellite) -> None:
        """Create a role-compatible datastore and sensor-local eligibility filter."""
        super().create_data_store(satellite)
        store = satellite.data_store
        store.verify_image_quality_on_downlink = True
        store.hide_pending_targets = self.hide_pending_targets
        store.image_quality_threshold = self.quality_threshold
        store.data.hide_pending_targets = self.hide_pending_targets

        if satellite.role is not SpacecraftRole.SENSING_AGENT:
            return
        self.per_sensor_metrics[satellite.name] = {
            "captures": 0.0,
            "deliveries": 0.0,
            "duplicate_attempts": 0.0,
            "successful_duplicates": 0.0,
            "communication_actions": 0.0,
            "acquisition_credit": 0.0,
            "delivery_credit": 0.0,
        }

        if not getattr(satellite, "_multi_rso_reimage_filter_added", False):

            def local_reimage_filter(opportunity, sat=satellite):
                if opportunity["type"] != "target":
                    return True
                sim_time = float(getattr(sat.simulator, "sim_time", 0.0))
                target = opportunity["object"]
                local_catalog = getattr(sat.data_store, "catalog", None)
                if local_catalog is not None:
                    return local_catalog.is_eligible(target.id, sim_time)
                return sat.data_store.data.is_target_eligible(target, sim_time)

            satellite.add_access_filter(local_reimage_filter, types="target")
            satellite._multi_rso_reimage_filter_added = True

    @staticmethod
    def _sensor_orbit_period_s(sensor) -> float:
        """Return one sensing spacecraft's osculating two-body period."""
        try:
            r_vec = np.asarray(sensor.dynamics.r_BN_N, dtype=float)
            v_vec = np.asarray(sensor.dynamics.v_BN_N, dtype=float)
            mu = float(orbitalMotion.MU_EARTH * 1e9)
            specific_energy = 0.5 * float(v_vec @ v_vec) - mu / float(
                np.linalg.norm(r_vec)
            )
            semi_major_axis = -mu / (2.0 * specific_energy)
            if specific_energy >= 0.0 or semi_major_axis <= 0.0:
                raise ValueError("non-elliptic sensing orbit")
            return float(2.0 * np.pi * np.sqrt(semi_major_axis**3 / mu))
        except Exception as exc:
            raise ValueError(f"Unable to estimate {sensor.name} orbit period") from exc

    def _estimate_reference_orbit_period_s(self) -> float:
        """Return an order-independent median period for the shared team cooldown."""
        periods = []
        for sensor in self.sensing_satellites:
            try:
                periods.append(self._sensor_orbit_period_s(sensor))
            except ValueError as exc:
                logger.warning("%s", exc)
        if not periods:
            logger.warning("Using fallback orbit period for the team cooldown")
            return self.fallback_orbit_period_s
        return float(np.median(periods))

    def _simulation_time(self) -> float:
        """Return the common simulator time without selecting a sensor by position."""
        sim_times = {
            float(sensor.simulator.sim_time) for sensor in self.sensing_satellites
        }
        if len(sim_times) != 1:
            raise RuntimeError(
                f"Sensing spacecraft have inconsistent times: {sim_times}"
            )
        return sim_times.pop()

    def reset_post_sim_init(self) -> None:
        """Initialize cooldown, team accounting, catalogs, and storage baselines."""
        self.reimage_cooldown_s = max(
            0.0,
            self.reimage_cooldown_orbits * self._estimate_reference_orbit_period_s(),
        )
        self._team_accounting = _TeamServiceAccounting(
            cooldown_s=self.reimage_cooldown_s,
            quality_threshold=self.quality_threshold,
        )
        for sensor in self.sensing_satellites:
            sensor.data_store.cooldown_duration_s = self.reimage_cooldown_s
            storage = np.asarray(
                sensor.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData,
                dtype=float,
            )
            self.old_storage_by_sensor[sensor.name] = np.array(storage)

    def _target_by_id(self, target_id: int):
        return next(
            target
            for target in self.scenario.target_spacecrafts
            if int(target.id) == int(target_id)
        )

    def _target_by_name(self, target_name: str):
        return next(
            (
                target
                for target in self.scenario.target_spacecrafts
                if target.name == target_name
            ),
            None,
        )

    @staticmethod
    def _quality(record: dict[str, Any]) -> float:
        value = record.get("mean_hold_shadow_factor")
        if value is None:
            value = record.get("capture_shadow_factor", 0.0)
        return float(value)

    def _product_from_record(
        self, sensor, record: dict[str, Any]
    ) -> ImageProductRecord:
        capture_time = float(record.get("capture_time", sensor.simulator.sim_time))
        record_id = str(
            record.get(
                "record_id",
                f"{sensor.name}:{record['target_id']}:{capture_time:.9f}",
            )
        )
        return ImageProductRecord(
            record_id=record_id,
            source_sensor=sensor.name,
            target_id=int(record["target_id"]),
            capture_time=capture_time,
            delivery_time=None,
            quality=self._quality(record),
            storage_owner=sensor.name,
        )

    def _capture_products(self, new_data_dict) -> list[ImageProductRecord]:
        products = []
        sensors = {sensor.name: sensor for sensor in self.sensing_satellites}
        assert self._team_accounting is not None
        for sensor_id, new_data in new_data_dict.items():
            sensor = sensors.get(sensor_id)
            if sensor is None:
                continue
            for records in new_data.pending_image_records_by_id.values():
                for record in records:
                    product = self._product_from_record(sensor, record)
                    sensor.data_store.store_product(product)
                    sensor.data_store.catalog.record_capture(product)
                    target = self._target_by_id(product.target_id)
                    sensor.data_store.data.mark_target_pending(target, record)
                    duplicate = self._team_accounting.register_capture_attempt(product)
                    self.per_sensor_metrics[sensor_id]["captures"] += 1.0
                    if duplicate:
                        self.per_sensor_metrics[sensor_id]["duplicate_attempts"] += 1.0
                    products.append(product)
        return products

    def _downlinked_products(self) -> list[ImageProductRecord]:
        products = []
        sim_time = self._simulation_time()
        for sensor in self.sensing_satellites:
            message = sensor.dynamics.storageUnit.storageUnitDataOutMsg.read()
            current = np.asarray(message.storedData, dtype=float)
            previous = self.old_storage_by_sensor[sensor.name]
            for idx in np.where(current - previous < 0)[0]:
                target_name = str(message.storedDataName[int(idx)])
                target = self._target_by_name(target_name)
                if target is None:
                    continue
                onboard = sensor.data_store.records_for_target(target.id)
                if not onboard:
                    continue
                product = min(
                    onboard, key=lambda item: (item.capture_time, item.record_id)
                )
                delivered = sensor.data_store.downlink_product(
                    product.record_id, sim_time
                )
                cooldown_until = (
                    delivered.capture_time + float(self.reimage_cooldown_s)
                    if delivered.quality >= self.quality_threshold
                    else None
                )
                sensor.data_store.catalog.record_delivery(delivered, cooldown_until)
                pending = sensor.data_store.data.pop_pending_record(target)
                sensor.data_store.data.mark_record_verified(
                    pending or {"record_id": delivered.record_id},
                    delivered.quality >= self.quality_threshold,
                )
                if cooldown_until is not None:
                    sensor.data_store.data.mark_target_imaged(target)
                    sensor.data_store.data.mark_target_cooldown(target, cooldown_until)
                else:
                    sensor.data_store.data.clear_target_cooldown(target)
                self.per_sensor_metrics[sensor.name]["deliveries"] += 1.0
                products.append(delivered)
            self.old_storage_by_sensor[sensor.name] = np.array(current)
        return products

    def _operational_adjustments(self, reward: dict[str, float]) -> None:
        for sensor in self.sensing_satellites:
            dynamics = sensor.dynamics
            if getattr(dynamics, "penalties", 0) == 1:
                if dynamics.battery_charge_fraction < 0.2:
                    reward[sensor.name] += float(dynamics.low_battery_penalty)
                if dynamics.storage_level_fraction > 0.991:
                    reward[sensor.name] += float(dynamics.full_storage_penalty)
                if getattr(dynamics, "last_downlink_started_empty", False):
                    reward[sensor.name] += float(dynamics.empty_downlink_penalty)
                    dynamics.last_downlink_started_empty = False
            if (
                str(getattr(sensor, "_current_action_label", ""))
                .lower()
                .startswith("action_broadcast")
            ):
                self.per_sensor_metrics[sensor.name]["communication_actions"] += 1.0
                reward[sensor.name] += self.communication_penalty

    def calculate_reward(self, new_data_dict) -> dict[str, float]:
        """Return source-assigned rewards whose team sum is not duplicated."""
        assert self._team_accounting is not None
        reward = {sensor.name: 0.0 for sensor in self.sensing_satellites}
        priorities = {
            int(target.id): float(target.priority)
            for target in self.scenario.target_spacecrafts
        }
        captures = self._capture_products(new_data_dict)
        acquisition_credit = self._team_accounting.register_acquisitions(
            captures, priorities
        )
        for sensor_id, value in acquisition_credit.items():
            self.per_sensor_metrics[sensor_id]["acquisition_credit"] += float(value)
            reward[sensor_id] += self.reward_fn((1.0 - self.alpha) * value)

        deliveries = self._downlinked_products()
        prior_successful_duplicates = self._team_accounting.successful_duplicate_count
        delivery_credit = self._team_accounting.register_deliveries(
            deliveries, priorities
        )
        for sensor_id, value in delivery_credit.items():
            self.per_sensor_metrics[sensor_id]["delivery_credit"] += float(value)
            reward[sensor_id] += self.reward_fn(self.alpha * value)
        for entry in self._team_accounting.entries:
            if entry.product in deliveries and entry.successful_duplicate:
                self.per_sensor_metrics[entry.product.source_sensor][
                    "successful_duplicates"
                ] += 1.0
                reward[entry.product.source_sensor] += self.duplicate_penalty

        self._operational_adjustments(reward)
        self.team_summary = {
            "team_value": float(self._team_accounting.team_value),
            "team_acquisition_value": float(
                self._team_accounting.acquisition_team_value
            ),
            "unique_acquisition_count": float(
                self._team_accounting.unique_acquisition_count
            ),
            "unique_service_count": float(self._team_accounting.unique_service_count),
            "duplicate_attempt_count": float(
                self._team_accounting.duplicate_attempt_count
            ),
            "successful_duplicate_count": float(
                self._team_accounting.successful_duplicate_count
            ),
            "new_successful_duplicates": float(
                self._team_accounting.successful_duplicate_count
                - prior_successful_duplicates
            ),
        }
        return reward


__all__ = [
    "MultiSensorRSOTargetImageReward",
    "MultiSensorRSOTargetImageStore",
]
