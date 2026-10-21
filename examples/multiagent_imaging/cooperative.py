"""Opt-in public request tasking; private completion information stays private.

Requests are ground-tasked, synchronized campaigns, not satellite intentions.
Every window has a seeded subset of active requests and a common deadline.
All 200 spacecraft continue propagating. Off-request images remain physically
possible, consume storage/time and earn zero task science credit.
"""

from __future__ import annotations

import math

import numpy as np

from bsk_rl.data.completion_reward import CompletionImageReward


class RequestSchedule:
    def __init__(self, config, seed):
        self.config, self.seed = config, int(seed)

    def epoch(self, time_s):
        return (
            math.floor(max(0.0, time_s) / self.config.request_window_s)
            * self.config.request_window_s
        )

    def active_ids(self, time_s):
        if self.config.request_target_ids:
            return frozenset(self.config.request_target_ids)
        index = int(self.epoch(time_s) / self.config.request_window_s)
        # Separate generator: tasking cannot change physical orbit/priority draws.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index, 731]))
        return frozenset(
            int(i)
            for i in rng.choice(
                self.config.n_targets, self.config.active_request_count, replace=False
            )
        )

    def qualifies(self, product, phase):
        epoch = self.epoch(product.capture_time)
        endpoint = (
            product.completion_time if phase == "acquisition" else product.delivery_time
        )
        return (
            product.target_id in self.active_ids(product.capture_time)
            and product.request_epoch == epoch
            and product.quality >= self.config.request_quality_threshold
            and endpoint is not None
            and endpoint <= epoch + self.config.request_window_s
        )


class CooperativeImageReward(CompletionImageReward):
    """Physical products are retained even when their request credit is zero."""

    def __init__(self, config, **kwargs):
        super().__init__(redundant_capture_cost=config.redundant_capture_cost, **kwargs)
        self.config = config
        self.environment = None

    def reset_post_sim_init(self):
        super().reset_post_sim_init()
        self.request_schedule = RequestSchedule(self.config, self.environment.seed)
        overrides = dict(self.config.request_priority_overrides)
        if overrides:
            targets = self.scenario.target_spacecrafts
            for target in targets:
                target.priority = overrides.get(int(target.id), target.priority)
            scale = 100.0 / sum(t.priority for t in targets)
            for target in targets:
                target.priority *= scale
        self.request_rejected = {"acquisition": 0, "delivery": 0}
        for sensor in self.sensing_satellites:
            sensor.completion_request_schedule = self.request_schedule

    def _science_products(self, products, *, phase):
        qualified = [p for p in products if self.request_schedule.qualifies(p, phase)]
        self.request_rejected[phase] += len(products) - len(qualified)
        return qualified

    def sync_requests(self, now):
        sync_public_requests(self.environment, now)

    def next_request_boundary(self, now):
        return self.request_schedule.epoch(now) + self.config.request_window_s

    def task_summary(self):
        ledger = self._team_accounting
        if len(ledger.entries) != sum(
            m["deliveries"] for m in self.per_sensor_metrics.values()
        ):
            raise AssertionError(
                "Physical delivery history differs from native datastore removals"
            )
        return dict(
            version="public-request-utility-v1",
            acquisition_value=ledger.acquisition_team_value,
            verified_delivery_value=ledger.team_value,
            unique_acquisition_requests=ledger.unique_acquisition_count,
            useful_delivered_requests=ledger.unique_service_count,
            unshaped_weighted_science=(1 - self.alpha) * ledger.acquisition_team_value
            + self.alpha * ledger.team_value,
            redundant_capture_cost=sum(
                m.get("redundant_capture_cost", 0.0)
                for m in self.per_sensor_metrics.values()
            ),
            rejected_acquisitions=self.request_rejected["acquisition"],
            rejected_deliveries=self.request_rejected["delivery"],
            active_request_count=self.config.active_request_count,
            window_s=self.config.request_window_s,
            quality_threshold=self.config.request_quality_threshold,
        )


def sync_public_requests(environment, now):
    """Public ground-task metadata only; never consult the team service ledger."""
    schedule = getattr(environment.rewarder, "request_schedule", None)
    if schedule is None:
        return
    epoch = schedule.epoch(now)
    for target in environment.scenario.target_spacecrafts:
        target.request_epoch_s = epoch
    for sensor in environment.sensing_satellites:
        for target_id in sensor.data_store.catalog.targets:
            sensor.data_store.catalog.set_request_epoch(target_id, epoch)
