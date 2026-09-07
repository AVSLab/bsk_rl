"""Completion-study actions using existing Basilisk FSW and hold-gated imaging."""

from dataclasses import asdict
import json

import numpy as np
from Basilisk.utilities import macros

from bsk_rl.act.discrete_actions import DiscreteAction, ImageRSO


class BroadcastCompletions(DiscreteAction):
    """Occupy the spacecraft for a finite, interruptible metadata transmission."""

    def __init__(self, duration=30.0):
        """Configure the occupied radio duration in simulation seconds."""
        super().__init__(name="broadcast_completions", n_actions=1)
        self.duration = float(duration)

    def set_action(self, action, prev_action_key=None):
        """Start drifting and freeze completed metadata until the timed radio end."""
        self.satellite.fsw.action_drift()
        self.satellite.update_timed_terminal_event(
            self.simulator.sim_time + self.duration,
            info="completion transmission finished",
        )
        self.satellite.completion_communicator.begin_transmission(
            self.satellite, self.simulator.sim_time, self.duration
        )
        return self.name


class TransmitCompletions(DiscreteAction):
    """Point at an observed peer, then send one frozen completion-only delta.

    A fixed minimum continuous hold defaults to 10 seconds. Optionally extend it
    to fit canonical JSON payload bytes at a declared metadata bit rate. Interrupted
    holds restart the whole packet; partial packets never modify a local catalog.
    """

    def __init__(self, n_peers, duration=300.0, hold_s=10.0, bitrate_bps=None):
        """Configure peer slots, total deadline and continuous packet airtime."""
        super().__init__(name="transmit_completions", n_actions=n_peers)
        self.duration, self.hold_s = float(duration), float(hold_s)
        self.bitrate_bps = bitrate_bps
        self.peer = None

    def set_action(self, action, prev_action_key=None):
        """Decode the observed peer and install pointing, hold and timeout events."""
        snapshot = self.satellite.completion_peers
        if snapshot.time != float(self.simulator.sim_time):
            raise ValueError("Stale peer snapshot.")
        self.peer = snapshot.peers[action]
        if self.peer is None:
            raise ValueError("A masked peer cannot receive a transmission.")
        now = float(self.simulator.sim_time)
        channel = self.satellite.completion_communicator
        records = channel.begin_directed(self.satellite, self.peer, now)
        # The versioned wire-size model includes a 64-byte transport header.
        self.payload_bytes = 64 + len(
            json.dumps(
                [asdict(record) for record in records],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        self.required_hold_s = (
            max(self.hold_s, 8 * self.payload_bytes / self.bitrate_bps)
            if self.bitrate_bps
            else self.hold_s
        )
        self.held_s, self.last_time, self.was_valid = 0.0, now, False
        self.start_time, self.radio_on_s = now, 0.0
        self.satellite._active_completion_transmit = self
        self.satellite.fsw.action_point_peer(self.peer)
        self.event_name = "completion_tx_" + self.satellite.name
        if self.event_name in self.simulator.eventMap:
            self.simulator.delete_event(self.event_name)
        self.simulator.createNewEvent(
            self.event_name,
            macros.sec2nano(self.satellite.fsw.fsw_rate),
            True,
            [
                f"{self.satellite._satellite_command}._active_completion_transmit.check_hold()"
            ],
            [f"{self.satellite._satellite_command}.requires_retasking = True"],
            terminal=True,
        )
        self.satellite.update_timed_terminal_event(
            now + self.duration,
            info="directed transmission deadline",
            extra_actions=[
                f"{self.satellite._satellite_command}._active_completion_transmit.cancel('timeout') if {self.satellite._satellite_command}._active_completion_transmit is not None else None"
            ],
        )
        return self.name + "_" + self.peer.name

    def pointing_valid(self):
        """Use the same MRP/rate requirements as imaging plus discovered LOS."""
        sensor = self.satellite
        if self.peer.name not in sensor.completion_communicator.receivers(sensor):
            return False
        guidance = sensor.fsw.attGuidMsg.read()
        controller = sensor.fsw.insControl
        return bool(
            np.linalg.norm(guidance.sigma_BR) <= controller.attErrTolerance
            and (
                not controller.useRateTolerance
                or np.linalg.norm(guidance.omega_BR_B) <= controller.rateErrTolerance
            )
        )

    def check_hold(self):
        """Basilisk event sampled at every FSW tick, including peer decision gaps."""
        now = float(self.simulator.sim_time)
        dt = max(0.0, now - self.last_time)
        # Action reset writes a zero guidance gateway. Wait for a fresh controller
        # tick so that zero reset message cannot masquerade as pointing lock.
        if now <= self.start_time + 1e-9:
            return False
        valid = self.pointing_valid()
        # Charge radio power for the preceding interval. Counting a hold interval
        # requires valid samples at both ends (conservative one-tick acquisition).
        if self.was_valid:
            self.radio_on_s += dt
        self.held_s = self.held_s + dt if valid and self.was_valid else 0.0
        self.last_time, self.was_valid = now, valid
        self.satellite.dynamics.transmitterPowerSink.powerStatus = int(valid)
        if self.held_s + 1e-9 < self.required_hold_s:
            return False
        channel = self.satellite.completion_communicator
        channel.transmissions[self.satellite.name].end = now
        self._record("hold_complete")
        self.satellite.dynamics.transmitterPowerSink.powerStatus = 0
        return True

    def _record(self, outcome):
        self.satellite.completion_communicator.transmission_history.append(
            dict(
                sender=self.satellite.name,
                receiver=self.peer.name,
                start=self.start_time,
                end=float(self.simulator.sim_time),
                outcome=outcome,
                held_s=self.held_s,
                required_hold_s=self.required_hold_s,
                radio_on_s=self.radio_on_s,
                payload_bytes=self.payload_bytes,
            )
        )

    def cancel(self, reason="policy_switch"):
        """Stop the hold event and radio without modifying any receiver catalog."""
        self.simulator.setEventActivity(self.event_name, False)
        if self.satellite.name in self.satellite.completion_communicator.transmissions:
            self._record(reason)
            self.satellite.completion_communicator.cancel_transmission(
                self.satellite.name, self.simulator.sim_time
            )
        self.satellite.dynamics.transmitterPowerSink.powerStatus = 0
        self.satellite._active_completion_transmit = None


class ContinueTask(DiscreteAction):
    """A deliberate policy decision to retain physical progress and deadline.

    The role-aware environment intercepts this action before the generic action
    builder, which would otherwise delete and recreate timed FSW events.
    """

    def __init__(self):
        """Declare one policy-selectable continuation action."""
        super().__init__(name="continue_task", n_actions=1)

    def set_action(self, action, prev_action_key=None):
        """Reject use without the environment interception that preserves progress."""
        raise RuntimeError("ContinueTask requires CompletionConstellationTasking.")


class ImageCompletion(ImageRSO):
    """Decode the observed candidate snapshot and reuse ImageRSO execution."""

    def set_action(self, action, prev_action_key=None):
        """Task the observed target and install its hold and deadline events."""
        snapshot = self.satellite.completion_candidates
        target = snapshot.targets[action]
        if target is None or snapshot.time != float(self.simulator.sim_time):
            raise ValueError("Imaging action selected an invalid or stale candidate.")
        self.request_epoch = self.satellite.data_store.catalog.request_epochs[
            int(target.id)
        ]
        self.chosen_target_ids.append(int(target.id))
        self.chosen_target_priority.append(float(target.priority))
        self.satellite.dynamics.last_imaging_target_id = int(target.id)
        self._disable_image_success_event()
        key = self.image_rso(target)
        self._enable_image_success_event(target)
        deadline = float(self.simulator.sim_time + self.duration)
        # Keep the inherited opportunity-window termination when an applicable
        # precomputed access window is available; duration remains a hard bound.
        windows = self.satellite.next_opportunities_dict(
            types="target", filter=self.satellite.default_access_filter
        )
        if target in windows:
            deadline = min(deadline, float(windows[target][1]))
        self.satellite.update_timed_terminal_event(
            deadline,
            info="image completion deadline",
            extra_actions=[
                f"{self.satellite._satellite_command}._active_image_rso_action._record_imaging_attempt(False, 'timeout_or_window') if {self.satellite._satellite_command}._active_image_rso_action is not None else None",
                f"{self.satellite._satellite_command}._active_image_rso_action._clear_hold_state() if {self.satellite._satellite_command}._active_image_rso_action is not None else None",
            ],
        )
        return key

    def _stage_capture_metadata(self, *args):
        target = self._hold_target
        super()._stage_capture_metadata(*args)
        records = self.satellite._rso_pending_capture_metadata_by_name.get(
            target.target_spacecraft.name, []
        )
        if records:
            records[-1]["request_epoch"] = self.request_epoch


__all__ = [
    "BroadcastCompletions",
    "TransmitCompletions",
    "ContinueTask",
    "ImageCompletion",
]
