"""Physical image ownership, complete delivery, and centralized RSO service reward."""

from dataclasses import replace
from itertools import groupby

import numpy as np

from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.utils.rso_imaging import RSOImageRecord


class RSOImageData(Data):
    """Mergeable event records; communicated knowledge does not move image bits."""

    def __init__(self, captures=None, deliveries=None) -> None:
        """Index immutable acquisition and completed-delivery records by record ID."""
        self.captures = {} if captures is None else dict(captures)
        self.deliveries = {} if deliveries is None else dict(deliveries)

    def __add__(self, other):
        """Merge idempotently, rejecting conflicting product provenance."""
        collections = []
        for first, second in (
            (self.captures, other.captures),
            (self.deliveries, other.deliveries),
        ):
            if any(
                key in first and first[key] != value for key, value in second.items()
            ):
                raise ValueError("Conflicting RSO image records.")
            collections.append(first | second)
        return RSOImageData(*collections)


class RSOImageStore(DataStore):
    """Per-imager FIFO ledger backed by actual partitioned storage changes.

    Capture timestamps identify the acquisition command sent to the instrument. Delivery timestamps
    are recognition times at environment step boundaries. Partial transmitted bits
    reduce remaining bits without publishing a completed delivery.
    """

    data_type = RSOImageData

    def __init__(self, satellite, initial_data=None, scenario=None) -> None:
        """Constructor configuration works identically inside composed rewards."""
        super().__init__(satellite, initial_data)
        self.scenario = scenario
        self.products = {}
        self.remaining_bits = {}
        self.is_imager = (
            scenario is not None and satellite.name in scenario.imager_names
        )

    def get_log_state(self):
        """Read physical storage and this observer's confirmed captures."""
        if not self.is_imager:
            return {}, ()
        recorder = self.satellite.fsw.rso_capture
        recorder.check_capture()
        message = self.satellite.dynamics.storageUnit.storageUnitDataOutMsg.read()
        return dict(zip(message.storedDataName, message.storedData)), tuple(
            recorder.records
        )

    def compare_log_states(self, old_state, new_state):
        """Account for transmitted bits, including several products in a step."""
        old_levels, old_records = old_state
        new_levels, records = new_state
        new_captures = records[len(old_records) :]
        captured_bits = {}
        for record in new_captures:
            if (
                record.source_imager != self.satellite.name
                or record.record_id in self.products
            ):
                raise ValueError(
                    "Invalid capture ownership or duplicate physical image."
                )
            partition = self.scenario.partition_name(record.target_id)
            self.products[record.record_id] = record
            self.remaining_bits[record.record_id] = record.size_bits
            captured_bits[partition] = (
                captured_bits.get(partition, 0.0) + record.size_bits
            )
        delivered = {}
        for target_id in self.scenario.targets_by_id if self.is_imager else ():
            partition = self.scenario.partition_name(target_id)
            removed = (
                old_levels.get(partition, 0.0)
                + captured_bits.get(partition, 0.0)
                - new_levels.get(partition, 0.0)
            )
            if removed <= 1e-6:
                continue
            onboard = sorted(
                (
                    product
                    for product in self.products.values()
                    if product.target_id == target_id
                ),
                key=lambda product: (product.capture_time, product.record_id),
            )
            for product in onboard:
                remaining = self.remaining_bits[product.record_id]
                transmitted = min(remaining, removed)
                self.remaining_bits[product.record_id] -= transmitted
                removed -= transmitted
                if self.remaining_bits[product.record_id] <= 1e-6:
                    delivered[product.record_id] = replace(
                        product, delivery_time=float(self.satellite.simulator.sim_time)
                    )
                    del self.products[product.record_id]
                    del self.remaining_bits[product.record_id]
                if removed <= 1e-6:
                    break
            if removed > 1e-6:
                raise ValueError("Storage removed unowned RSO image bits.")
        return RSOImageData(
            {product.record_id: product for product in new_captures}, delivered
        )


class RSOImageReward(GlobalReward):
    """Priority-weighted acquisition and fully delivered, quality-verified service.

    Acquisition and delivery callbacks receive ``(record, target)`` and return
    the scalar value of that stage. By default, acquisition earns zero and a
    complete delivery earns the target's current priority.
    Specify weights, illumination scaling, or time discounts in those callbacks.
    Priorities are evaluated at each reward boundary, before priority events.
    Pending images are centrally visible. A useful delivery establishes a global
    cooldown measured from capture time. In explicit "shared" mode, captures seen
    together at one boundary share
    acquisition credit. Later captures while this service is pending earn no further
    acquisition credit. The first useful delivery group shares delivery credit;
    later duplicate deliveries earn none, including when cooldown is zero.
    "per_imager" gives each owner its full stage value once per pending service.
    Multiple imagers require selecting a mode explicitly.
    """

    data_store_type = RSOImageStore

    def __init__(
        self,
        acquisition_reward_fn=None,
        delivery_reward_fn=None,
        quality_threshold=0.5,
        cooldown_s=0.0,
        multi_imager_credit=None,
    ) -> None:
        """Configure stage callbacks, mean hold illumination, and cooldown.

        Args:
            acquisition_reward_fn: Callable ``(record, target) -> float`` for a
                useful physically acquired image. Defaults to zero.
            delivery_reward_fn: Callable with the same signature for a useful
                complete delivery. Defaults to current priority.
            quality_threshold: Minimum mean illuminated fraction in [0, 1].
            cooldown_s: Global cooldown from capture time, in seconds.
            multi_imager_credit: ``"shared"`` splits each callback's value across
                simultaneous qualified records, while ``"per_imager"`` gives each
                owner its callback's full value. Required for multiple imagers.
        """
        super().__init__()
        if (
            not 0 <= quality_threshold <= 1
            or not np.isfinite(cooldown_s)
            or cooldown_s < 0
            or multi_imager_credit not in (None, "shared", "per_imager")
        ):
            raise ValueError("Invalid RSO image reward settings.")
        for callback in (acquisition_reward_fn, delivery_reward_fn):
            if callback is not None and not callable(callback):
                raise TypeError("RSO stage rewards must be callable.")
        self.acquisition_reward_fn = acquisition_reward_fn or (
            lambda record, target: 0.0
        )
        self.delivery_reward_fn = delivery_reward_fn or (
            lambda record, target: target.priority
        )
        self.quality_threshold = float(quality_threshold)
        self.cooldown_s = float(cooldown_s)
        self.multi_imager_credit = multi_imager_credit

    def link_scenario(self, scenario) -> None:
        """Pass configuration through constructor kwargs, including composition."""
        if len(scenario.imager_names) > 1 and self.multi_imager_credit is None:
            raise ValueError(
                "Multiple imagers require explicit multi_imager_credit: "
                "'shared' or 'per_imager'."
            )
        super().link_scenario(scenario)
        self.data_store_kwargs = dict(scenario=scenario)

    def reset_overwrite_previous(self) -> None:
        """Reset per-episode uniqueness accounting."""
        super().reset_overwrite_previous()
        self.acquisition_credit_times = {}
        self.delivery_credit_times = {}
        self.acquisition_services = set()
        self.delivery_services = set()
        self.record_services = {}
        self.active_services = {}
        self.service_generation = {}
        self.last_reward_components = {"acquisition": {}, "delivery": {}}

    def _credit(
        self, records, credit_times, credited_services, reward_fn, reward
    ) -> None:
        def scope(record):
            return (
                (record.target_id, record.source_imager)
                if self.multi_imager_credit == "per_imager"
                else record.target_id
            )

        qualified = [
            record for record in records if record.quality >= self.quality_threshold
        ]
        qualified.sort(
            key=lambda record: (
                scope(record),
                self.record_services[record.record_id],
                record.capture_time,
                record.source_imager,
                record.record_id,
            )
        )
        for (credit_scope, service), grouped in groupby(
            qualified,
            key=lambda record: (
                scope(record),
                self.record_services[record.record_id],
            ),
        ):
            group = list(grouped)
            service_key = (credit_scope, service)
            if service_key in credited_services:
                continue
            capture_time = min(record.capture_time for record in group)
            last = credit_times.get(credit_scope)
            if last is not None and (
                capture_time <= last + 1e-9 or capture_time < last + self.cooldown_s
            ):
                continue
            target = self.scenario.targets_by_id[group[0].target_id]
            for record in group:
                value = float(reward_fn(record, target))
                if not np.isfinite(value):
                    raise ValueError("RSO stage rewards must return finite scalars.")
                reward[record.source_imager] += value / len(group)
            credit_times[credit_scope] = capture_time
            credited_services.add(service_key)

    def calculate_reward(self, new_data_dict):
        """Update centralized eligibility; credit the actual acquisition/delivery owner."""
        reward = {name: 0.0 for name in new_data_dict}
        captures, deliveries = [], []
        for name, events in new_data_dict.items():
            for record in (*events.captures.values(), *events.deliveries.values()):
                if (
                    name != record.source_imager
                    or name not in self.scenario.imager_names
                ):
                    raise ValueError("RSO reward record attributed to a non-owner.")
            captures.extend(events.captures.values())
            deliveries.extend(events.deliveries.values())
        fresh_captures = [
            record
            for record in captures
            if record.record_id not in self.record_services
        ]
        for target_id in sorted({record.target_id for record in fresh_captures}):
            if not self.scenario.pending.get(target_id):
                generation = self.service_generation.get(target_id, 0) + 1
                self.service_generation[target_id] = generation
                self.active_services[target_id] = (target_id, generation)
            for record in fresh_captures:
                if record.target_id == target_id:
                    self.record_services[record.record_id] = self.active_services[
                        target_id
                    ]
        for record in fresh_captures:
            self.scenario.pending.setdefault(record.target_id, set()).add(
                record.record_id
            )
        for record in deliveries:
            if record.record_id not in self.record_services:
                raise ValueError("RSO delivery has no physically acquired product.")
            self.scenario.pending.setdefault(record.target_id, set()).discard(
                record.record_id
            )
            if record.quality >= self.quality_threshold:
                until = record.capture_time + self.cooldown_s
                self.scenario.cooldown_until[record.target_id] = max(
                    until, self.scenario.cooldown_until.get(record.target_id, -np.inf)
                )
        self._credit(
            captures,
            self.acquisition_credit_times,
            self.acquisition_services,
            self.acquisition_reward_fn,
            reward,
        )
        acquisition_reward = dict(reward)
        self._credit(
            deliveries,
            self.delivery_credit_times,
            self.delivery_services,
            self.delivery_reward_fn,
            reward,
        )
        self.last_reward_components = {
            "acquisition": acquisition_reward,
            "delivery": {
                name: value - acquisition_reward[name] for name, value in reward.items()
            },
        }
        if captures or deliveries:
            self.scenario.revision += 1
        return reward


__doc_title__ = "Space-to-Space RSO Image Data"
__all__ = ["RSOImageData", "RSOImageStore", "RSOImageReward"]
