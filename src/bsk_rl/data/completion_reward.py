"""Completion catalogs attached to the existing sensor-owned AMOS data pipeline."""

from dataclasses import replace

import numpy as np

from bsk_rl.data.completion_catalog import CompletionCatalog
from bsk_rl.data.multiagent_rso_reward import (
    MultiSensorRSOTargetImageReward,
    MultiSensorRSOTargetImageStore,
)
from bsk_rl.data.rso_targets_data import RSOTargetImageData
from bsk_rl.sats.roles import SpacecraftRole


class CompletionImageStore(MultiSensorRSOTargetImageStore):
    """Delay publishing a capture until its exposure/hold result is available.

    A Basilisk instrument can write bits before its commanded pointing hold ends.
    Teammate boundaries must not turn those bits into an early success. Retain the
    physical increment here until staged hold metadata or interruption resolves it.
    """

    def __init__(self, *args, **kwargs):
        """Attach a completion catalog and per-exposure unresolved volume tracking."""
        super().__init__(*args, **kwargs)
        self.catalog = CompletionCatalog(self.satellite.name, self.catalog.targets)
        self.unresolved_captures = {}
        self.remaining_bits = {}

    def compare_log_states(self, old_state, new_state):
        """Resolve physical image increments only when their hold outcome is known."""
        if self.satellite.role is SpacecraftRole.PASSIVE_TARGET:
            return RSOTargetImageData()
        now = float(self.satellite.simulator.sim_time)
        for idx in np.where(new_state - old_state > 1e-6)[0]:
            target = self._target_from_name(
                self._target_name_from_storage_index(int(idx))
            )
            if target is None:
                continue
            entry = self.unresolved_captures.setdefault(
                int(target.id),
                dict(
                    target=target,
                    storage_index=int(idx),
                    capture_time=now,
                    request_epoch=self.catalog.request_epochs[int(target.id)],
                    bits=0.0,
                ),
            )
            entry["bits"] += float(new_state[idx] - old_state[idx])

        active = getattr(self.satellite, "_active_image_rso_action", None)
        active_target = getattr(getattr(active, "_hold_target", None), "id", None)
        pending, imaged = {}, []
        for target_id, entry in list(self.unresolved_captures.items()):
            target = entry["target"]
            staged = self._pop_staged_capture_metadata(target.name)
            if staged is None and active_target == target_id:
                continue
            # An interrupted exposure still occupies storage, but earns no imaging
            # credit and never tells a peer the request was successfully completed.
            record = staged or dict(
                record_id=f"{self.satellite.name}:{target_id}:{entry['capture_time']:.9f}:incomplete",
                capture_time=entry["capture_time"],
                request_epoch=entry["request_epoch"],
                end_time=now,
                mean_hold_shadow_factor=0.0,
                capture_shadow_factor=0.0,
                success=False,
                reason="incomplete_exposure",
            )
            record.update(
                target_id=target_id,
                target_name=target.name,
                storage_index=entry["storage_index"],
                storage_delta_bits=entry["bits"],
            )
            pending[target_id] = [record]
            if (
                record.get("success", False)
                and MultiSensorRSOTargetImageReward._quality(record)
                >= self._quality_threshold()
            ):
                imaged.append(target)
            del self.unresolved_captures[target_id]
        return RSOTargetImageData(
            imaged=imaged,
            pending_image_records_by_id=pending,
            hide_pending_targets=self.hide_pending_targets,
        )


class CompletionImageReward(MultiSensorRSOTargetImageReward):
    """Preserve AMOS image/ground reward ownership while publishing completion facts."""

    datastore_type = CompletionImageStore

    def reset_post_sim_init(self):
        """Use the same episode cooldown and quality threshold in every local catalog."""
        super().reset_post_sim_init()
        for sensor in self.sensing_satellites:
            sensor.data_store.catalog.cooldown_s = self.reimage_cooldown_s
            sensor.data_store.catalog.quality_threshold = self.quality_threshold

    def _product_from_record(self, sensor, record):
        product = super()._product_from_record(sensor, record)
        sensor.data_store.remaining_bits[product.record_id] = float(
            record["storage_delta_bits"]
        )
        return replace(
            product,
            request_epoch=float(
                record.get(
                    "request_epoch",
                    sensor.data_store.catalog.request_epochs[product.target_id],
                )
            ),
            completion_time=float(record.get("end_time", sensor.simulator.sim_time)),
        )

    def _downlinked_products(self):
        """Credit full products, not the first partial decrease of their partition.

        Multiple products in a target partition drain oldest-first. A receiver's
        catalog never participates in this physical ownership/volume calculation.
        """
        delivered_products = []
        now = self._simulation_time()
        for sensor in self.sensing_satellites:
            store = sensor.data_store
            message = sensor.dynamics.storageUnit.storageUnitDataOutMsg.read()
            current = np.asarray(message.storedData, dtype=float)
            previous = self.old_storage_by_sensor[sensor.name]
            for idx in np.where(previous - current > 1e-6)[0]:
                target = self._target_by_name(str(message.storedDataName[int(idx)]))
                if target is None:
                    continue
                drained = float(previous[idx] - current[idx])
                for product in sorted(
                    store.records_for_target(target.id),
                    key=lambda p: (p.capture_time, p.record_id),
                ):
                    outstanding = store.remaining_bits[product.record_id]
                    consumed = min(outstanding, drained)
                    store.remaining_bits[product.record_id] -= consumed
                    drained -= consumed
                    if store.remaining_bits[product.record_id] > 1e-6:
                        break
                    delivered = store.downlink_product(product.record_id, now)
                    del store.remaining_bits[product.record_id]
                    store.catalog.record_delivery(delivered)
                    pending = store.data.pop_pending_record(target)
                    store.data.mark_record_verified(
                        pending or {"record_id": product.record_id},
                        product.quality >= self.quality_threshold,
                    )
                    self.per_sensor_metrics[sensor.name]["deliveries"] += 1.0
                    delivered_products.append(delivered)
                    if drained <= 1e-6:
                        break
            self.old_storage_by_sensor[sensor.name] = current.copy()
        return delivered_products


__all__ = ["CompletionImageReward", "CompletionImageStore"]
