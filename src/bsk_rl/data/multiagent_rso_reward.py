"""Multi-sensor RSO reward with local lifecycle state and global team accounting."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
from Basilisk.utilities import orbitalMotion

from bsk_rl.data.base import GlobalReward
from bsk_rl.data.multiagent_rso_data import (
    ImageProductRecord,
    LocalCatalogKnowledge,
    SensorProductStore,
    TeamServiceLedger,
)
from bsk_rl.data.rso_targets_data import (
    RSOTargetImageData,
    RSOTargetImageStore,
)
from bsk_rl.sats.roles import SpacecraftRole

logger = logging.getLogger(__name__)


class MultiSensorRSOTargetImageStore(RSOTargetImageStore):
    """Per-spacecraft storage logger that is inert for passive propagated targets."""

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
        self.team_ledger: Optional[TeamServiceLedger] = None

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
        }

        if not getattr(satellite, "_multi_rso_reimage_filter_added", False):

            def local_reimage_filter(opportunity, sat=satellite):
                if opportunity["type"] != "target":
                    return True
                sim_time = float(getattr(sat.simulator, "sim_time", 0.0))
                target = opportunity["object"]
                local_catalog = getattr(sat, "local_catalog", None)
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
        """Initialize cooldown, team ledger, local catalogs, and storage baselines."""
        self.reimage_cooldown_s = max(
            0.0,
            self.reimage_cooldown_orbits * self._estimate_reference_orbit_period_s(),
        )
        self.team_ledger = TeamServiceLedger(
            cooldown_s=self.reimage_cooldown_s,
            quality_threshold=self.quality_threshold,
        )
        target_ids = [int(target.id) for target in self.scenario.target_spacecrafts]
        for sensor in self.sensing_satellites:
            sensor.data_store.cooldown_duration_s = self.reimage_cooldown_s
            if not hasattr(sensor, "physical_product_store"):
                sensor.physical_product_store = SensorProductStore(sensor.name)
            if not hasattr(sensor, "local_catalog"):
                sensor.local_catalog = LocalCatalogKnowledge(sensor.name, target_ids)
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
        assert self.team_ledger is not None
        for sensor_id, new_data in new_data_dict.items():
            sensor = sensors.get(sensor_id)
            if sensor is None:
                continue
            for records in new_data.pending_image_records_by_id.values():
                for record in records:
                    product = self._product_from_record(sensor, record)
                    sensor.physical_product_store.store(product)
                    sensor.local_catalog.record_capture(product)
                    target = self._target_by_id(product.target_id)
                    sensor.data_store.data.mark_target_pending(target, record)
                    duplicate = self.team_ledger.register_capture_attempt(product)
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
                onboard = sensor.physical_product_store.records_for_target(target.id)
                if not onboard:
                    continue
                product = min(
                    onboard, key=lambda item: (item.capture_time, item.record_id)
                )
                delivered = sensor.physical_product_store.downlink(
                    product.record_id, sim_time
                )
                cooldown_until = (
                    delivered.capture_time + float(self.reimage_cooldown_s)
                    if delivered.quality >= self.quality_threshold
                    else None
                )
                sensor.local_catalog.record_delivery(delivered, cooldown_until)
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
        assert self.team_ledger is not None
        reward = {sensor.name: 0.0 for sensor in self.sensing_satellites}
        priorities = {
            int(target.id): float(target.priority)
            for target in self.scenario.target_spacecrafts
        }
        captures = self._capture_products(new_data_dict)
        acquisition_credit = self.team_ledger.register_acquisitions(
            captures, priorities
        )
        for sensor_id, value in acquisition_credit.items():
            reward[sensor_id] += self.reward_fn((1.0 - self.alpha) * value)

        deliveries = self._downlinked_products()
        prior_successful_duplicates = self.team_ledger.successful_duplicate_count
        delivery_credit = self.team_ledger.register_deliveries(deliveries, priorities)
        for sensor_id, value in delivery_credit.items():
            reward[sensor_id] += self.reward_fn(self.alpha * value)
        for entry in self.team_ledger.entries:
            if entry.product in deliveries and entry.successful_duplicate:
                self.per_sensor_metrics[entry.product.source_sensor][
                    "successful_duplicates"
                ] += 1.0
                reward[entry.product.source_sensor] += self.duplicate_penalty

        self._operational_adjustments(reward)
        self.team_summary = {
            "team_value": float(self.team_ledger.team_value),
            "unique_service_count": float(self.team_ledger.unique_service_count),
            "duplicate_attempt_count": float(self.team_ledger.duplicate_attempt_count),
            "successful_duplicate_count": float(
                self.team_ledger.successful_duplicate_count
            ),
            "new_successful_duplicates": float(
                self.team_ledger.successful_duplicate_count
                - prior_successful_duplicates
            ),
        }
        return reward


__all__ = [
    "MultiSensorRSOTargetImageReward",
    "MultiSensorRSOTargetImageStore",
]
