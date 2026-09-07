"""Asynchronous completion-aware decisions over real sensing and passive spacecraft."""

from dataclasses import asdict, dataclass
import math

from bsk_rl.gym import NO_ACTION, SensingAgentConstellationTasking
from bsk_rl.obs.completion_observations import (
    CONTINUE_ACTION,
    NON_IMAGING_ACTIONS,
    can_continue,
)
from bsk_rl.sats.roles import SpacecraftRole


@dataclass
class ActiveTask:
    """Own execution state; this object is never transmitted to another agent."""

    mode: str
    action_index: int
    target_id: int | None
    request_epoch: float
    start: float
    deadline: float
    finished: bool = False
    invalidated: bool = False
    receiver: str | None = None


class CompletionConstellationTasking(SensingAgentConstellationTasking):
    """Choose conflict or continuous retasking after receiver-local catalog merges.

    All spacecraft propagate in Basilisk. Only explicitly marked sensors appear
    in the PettingZoo API. ``max_step_duration`` is a declared decision heartbeat;
    physical action endings and queued packet arrivals can stop propagation sooner.
    """

    def __init__(
        self,
        *args,
        retasking_mode="conflict",
        communication_cost_per_s=0.0,
        default_seed=None,
        **kwargs,
    ):
        """Configure the decision rule and optional per-second radio penalty."""
        if retasking_mode not in {"conflict", "continuous"}:
            raise ValueError("retasking_mode must be conflict or continuous.")
        if communication_cost_per_s < 0:
            raise ValueError("Communication cost must be nonnegative.")
        self.default_seed = default_seed
        self._episode_seed_index = 0
        self.retasking_mode = retasking_mode
        self.communication_cost_per_s = float(communication_cost_per_s)
        self.task_history = []
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None):
        """Clear own task progress, decision counts, and candidate caches."""
        # RLlib 2.35's environment runner calls reset without a seed. Use a
        # reproducible episode sequence instead of the base environment's clock.
        if seed is None and self.default_seed is not None:
            seed = (int(self.default_seed) + self._episode_seed_index) % 2**32
            self._episode_seed_index += 1
        self.task_history = []
        self.decision_counts = {s.name: 0 for s in self.sensing_satellites}
        self.communication_time = {s.name: 0.0 for s in self.sensing_satellites}
        for sensor in self.sensing_satellites:
            sensor.completion_task = None
            sensor.completion_candidates = None
            sensor.completion_peers = None
            sensor._active_completion_transmit = None
            sensor.retask_reason = "initial"
        return super().reset(seed=seed, options=options)

    def _close_task(self, sensor, now, reason):
        task = sensor.completion_task
        if task is None or task.finished:
            return
        task.finished = True
        self.task_history.append(
            dict(sensor=sensor.name, **asdict(task), end=float(now), reason=reason)
        )

    def _interrupt(self, sensor, now, reason):
        """Cancel event bookkeeping, never stored image bits or spacecraft dynamics."""
        active_image = getattr(sensor, "_active_image_rso_action", None)
        if active_image is not None:
            active_image._record_imaging_attempt(False, reason)
            active_image._disable_image_success_event()
        active_transmit = getattr(sensor, "_active_completion_transmit", None)
        if active_transmit is not None:
            active_transmit.cancel(reason)
        sensor.disable_timed_terminal_event()
        self.communicator.cancel_transmission(sensor.name, now)
        self._close_task(sensor, now, reason)

    def _step(self, actions):
        now = float(self.simulator.sim_time)
        physical_actions = list(actions)

        # 1. Decode the exact snapshot the actor saw. A policy-selected continue
        # remains in RLlib's trajectory, but does not call the resetting FSW builder.
        for index, (sensor, action) in enumerate(zip(self.satellites, actions)):
            if sensor.role is not SpacecraftRole.SENSING_AGENT:
                continue
            task = sensor.completion_task
            if action is None or action == NO_ACTION:
                if sensor.requires_retasking and sensor.is_alive():
                    raise ValueError(f"{sensor.name} requires a policy action.")
                continue
            if not sensor.is_alive():
                physical_actions[index] = NO_ACTION
                continue
            self.decision_counts[sensor.name] += 1
            target_id = None
            epoch = 0.0
            count = sensor.completion_n_candidates
            peer_start = NON_IMAGING_ACTIONS + count
            receiver = None
            directed = action >= peer_start
            if directed:
                snapshot = sensor.completion_peers
                if snapshot is None or snapshot.time != now:
                    raise ValueError("Missing/stale peer observation snapshot.")
                slot = int(action) - peer_start
                if not 0 <= slot < len(snapshot.peers) or snapshot.peers[slot] is None:
                    raise ValueError("A masked recipient cannot be selected.")
                receiver = snapshot.peers[slot].name
                same_task = (
                    task is not None
                    and task.mode == "broadcast"
                    and task.receiver == receiver
                )
            elif action >= NON_IMAGING_ACTIONS:
                slot = int(action) - NON_IMAGING_ACTIONS
                snapshot = sensor.completion_candidates
                if (
                    snapshot is None
                    or snapshot.time != now
                    or not 0 <= slot < len(snapshot.targets)
                ):
                    raise ValueError("Missing/stale imaging candidate snapshot.")
                target = snapshot.targets[slot]
                if target is None:
                    raise ValueError("A masked imaging action cannot be executed.")
                target_id = int(target.id)
                epoch = sensor.data_store.catalog.request_epochs[target_id]
                same_task = (
                    task is not None
                    and task.mode == "image"
                    and task.target_id == target_id
                    and task.request_epoch == epoch
                )
            else:
                if action == 3 and sensor.completion_communication_mode == "directed":
                    raise ValueError("Broadcast is masked in directed mode.")
                if not 0 <= action <= CONTINUE_ACTION:
                    raise ValueError("Invalid non-imaging action.")
                same_task = (
                    task is not None
                    and task.action_index == action
                    and task.mode != "image"
                )
            if action == CONTINUE_ACTION or (same_task and can_continue(sensor)):
                if not can_continue(sensor):
                    raise ValueError(
                        "Continue is masked after completion/invalidation."
                    )
                sensor.requires_retasking = False
                physical_actions[index] = NO_ACTION
                continue
            if task is not None and not task.finished:
                self._interrupt(sensor, now, "policy_switch")
            modes = ("charge", "downlink", "desat", "broadcast")
            mode = (
                "broadcast"
                if directed
                else modes[int(action)]
                if action < CONTINUE_ACTION
                else "image"
            )
            action_spec = (
                sensor.action_builder.action_spec[6]
                if directed
                else sensor.action_builder.action_spec[5]
                if mode == "image"
                else sensor.action_builder.action_spec[int(action)]
            )
            sensor.completion_task = ActiveTask(
                mode,
                int(action),
                target_id,
                epoch,
                now,
                now + action_spec.duration,
                receiver=receiver,
            )
            if mode == "broadcast":
                # Count a physical start once; peer heartbeats and deliberate
                # continuation decisions must not multiply this operation count.
                self.rewarder.per_sensor_metrics[sensor.name][
                    "communication_actions"
                ] += 1.0

        # 2. Reuse BSK-RL's apply -> simulate -> local logs -> reward -> communicate
        # order. A queued reception is a simulator boundary even if every sensor
        # is still executing its own action. Delays are quantized to simulation ticks.
        configured_max = self.simulator.max_step_duration
        next_event = self.communicator.next_event_time(now)
        if math.isfinite(next_event):
            tick = float(self.simulator.sim_rate)
            wait = max(tick, math.ceil((next_event - now) / tick - 1e-10) * tick)
            self.simulator.max_step_duration = min(configured_max, wait)
        try:
            super()._step(physical_actions)
        finally:
            self.simulator.max_step_duration = configured_max
        end = float(self.simulator.sim_time)

        # 3. Received completions can invalidate an active image. No peer's private
        # catalog or global reward ledger is consulted in this decision rule.
        for sensor in self.sensing_satellites:
            task = sensor.completion_task
            natural_end = bool(sensor.requires_retasking)
            if task is not None:
                deadline = getattr(sensor, "_timed_terminal_time", None)
                if deadline is not None:
                    task.deadline = float(deadline)
                if task.mode == "broadcast" and not task.finished:
                    elapsed = end - now
                    self.communication_time[sensor.name] += elapsed
                    self.reward_dict[sensor.name] -= (
                        self.communication_cost_per_s * elapsed
                    )
                if natural_end or not sensor.is_alive():
                    self._close_task(
                        sensor, end, "action_end" if sensor.is_alive() else "failure"
                    )
            # A request epoch is mission tasking metadata, not a received intention.
            for target in sensor.data_store.data.known:
                sensor.data_store.catalog.set_request_epoch(
                    target.id, float(getattr(target, "request_epoch_s", 0.0))
                )
            conflict = (
                task is not None
                and not task.finished
                and task.mode == "image"
                and (
                    sensor.data_store.catalog.request_epochs[task.target_id]
                    != task.request_epoch
                    or sensor.data_store.catalog.satisfied(task.target_id, end)
                )
            )
            if conflict:
                task.invalidated = True
                self._interrupt(sensor, end, "known_completion_or_new_request")
            sensor.retask_reason = (
                "local_end"
                if natural_end
                else (
                    "known_completion"
                    if conflict
                    else "continuous"
                    if self.retasking_mode == "continuous"
                    else "busy"
                )
            )
            sensor.requires_retasking = bool(
                sensor.is_alive()
                and (natural_end or conflict or self.retasking_mode == "continuous")
            )
            # Invalidate caches after all deliveries, never between observation and
            # action decoding. Every sensor builds its next snapshot from its catalog.
            sensor.completion_candidates = None
            sensor.completion_peers = None
            sensor.observation_builder.obs_dict_cache = None

    def _get_obs(self):
        # RLlib bootstraps time-limit truncations from a real final observation,
        # including agents still executing actions when the horizon is reached.
        original = self.generate_obs_retasking_only
        if self.simulator.sim_time >= self.time_limit:
            self.generate_obs_retasking_only = False
        try:
            return super()._get_obs()
        finally:
            self.generate_obs_retasking_only = original

    def _get_info(self):
        info = super()._get_info()
        for sensor in self.sensing_satellites:
            if sensor.name in info:
                info[sensor.name]["retask_reason"] = getattr(
                    sensor, "retask_reason", "initial"
                )
                # The learner can integrate a per-second cost independently of the
                # arbitrary number of simulator boundaries within an action.
                task = getattr(sensor, "completion_task", None)
                info[sensor.name]["communication_cost_rate"] = (
                    self.communication_cost_per_s
                    if task and task.mode == "broadcast"
                    else 0.0
                )
        return info

    def coordination_metrics(self):
        """Slide metrics from event history, measured in sensor-seconds.

        Global truth is used only here for evaluation. It never updates a catalog,
        candidate mask, or retasking flag. Duplicate and nonduplicate-interruption
        categories are disjoint, unlike concurrent-target-time diagnostics.
        """
        now = float(self.simulator.sim_time)
        attempts = list(self.task_history)
        for sensor in self.sensing_satellites:
            task = sensor.completion_task
            if task is not None and not task.finished:
                attempts.append(
                    dict(sensor=sensor.name, **asdict(task), end=now, reason="horizon")
                )
        truth = self.rewarder._team_accounting.capture_attempts
        wasted_duplicate = wasted_interrupted = 0.0
        for attempt in attempts:
            if attempt["mode"] != "image":
                continue
            competitors = [
                p
                for p in truth
                if p.source_sensor != attempt["sensor"]
                and p.target_id == attempt["target_id"]
                and p.request_epoch == attempt["request_epoch"]
                and p.quality >= self.rewarder.quality_threshold
                and p.capture_time + self.rewarder.reimage_cooldown_s > attempt["start"]
                and (p.completion_time or p.capture_time) < attempt["end"]
            ]
            if competitors:
                first = min(
                    competitors,
                    key=lambda p: (p.completion_time or p.capture_time, p.record_id),
                )
                catalog = self.communicator.catalogs[attempt["sensor"]]
                receipt = catalog.received_at.get(first.record_id, float("inf"))
                wasted_duplicate += max(
                    0.0,
                    min(attempt["end"], receipt)
                    - max(
                        attempt["start"], first.completion_time or first.capture_time
                    ),
                )
            elif attempt["reason"] == "policy_switch":
                wasted_interrupted += attempt["end"] - attempt["start"]
        denominator = len(self.sensing_satellites) * now
        return dict(
            duplicate_sensor_time_s=wasted_duplicate,
            interrupted_nonduplicate_sensor_time_s=wasted_interrupted,
            wasted_time_fraction=(wasted_duplicate + wasted_interrupted) / denominator
            if denominator
            else 0.0,
            communication_time_s=dict(self.communication_time),
            policy_decisions=dict(self.decision_counts),
            task_history=attempts,
        )


__all__ = ["CompletionConstellationTasking"]
