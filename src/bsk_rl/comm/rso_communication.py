"""Information-sharing cases for multi-sensor RSO catalog maintenance."""

from __future__ import annotations

from enum import Enum
from typing import Callable, Iterable, Optional

import numpy as np
from Basilisk.utilities import orbitalMotion

from bsk_rl.comm.communication import CommunicationMethod
from bsk_rl.comm.typed_messages import (
    CentralizedInformationView,
    IntentStatusInbox,
    IntentStatusMessage,
    PerfectMetadataChannel,
)
from bsk_rl.sats.roles import SpacecraftRole
from bsk_rl.utils.coordination import current_target_id, earth_unoccluded


class InformationCase(str, Enum):
    INDEPENDENT = "independent"
    CENTRALIZED_INFORMATION = "centralized_information"
    INTENT_STATUS = "intent_status"


class IntentStatusCommunication(CommunicationMethod):
    """Manage local catalogs and optional directional typed-message exchange.

    ``perfect_metadata_delivery=True`` validates message semantics without link
    constraints. Setting it false requires an explicit directional-pair provider and
    a finite :class:`~bsk_rl.act.BroadcastIntent` action by each sender.
    """

    def __init__(
        self,
        information_case: InformationCase | str,
        *,
        message_ttl_s: float = 600.0,
        perfect_metadata_delivery: bool = True,
        directional_pair_provider: Optional[
            Callable[[float], Iterable[tuple[str, str]]]
        ] = None,
    ) -> None:
        """Configure an information case and its first-stage delivery model."""
        super().__init__(min_period=0.0)
        self.information_case = InformationCase(information_case)
        self.message_ttl_s = float(message_ttl_s)
        self.perfect_metadata_delivery = bool(perfect_metadata_delivery)
        self.directional_pair_provider = directional_pair_provider
        if self.message_ttl_s <= 0.0:
            raise ValueError("message_ttl_s must be positive.")

    def link_satellites(self, satellites) -> None:
        """Link and partition spacecraft by their explicit roles."""
        super().link_satellites(satellites)
        self.sensing_satellites = [
            satellite
            for satellite in satellites
            if satellite.role is SpacecraftRole.SENSING_AGENT
        ]
        self.passive_targets = [
            satellite
            for satellite in satellites
            if satellite.role is SpacecraftRole.PASSIVE_TARGET
        ]
        for sensor in self.sensing_satellites:
            sensor.information_case = self.information_case.value

    def reset_overwrite_previous(self) -> None:
        """Reset episode message ordering and delivery logs."""
        super().reset_overwrite_previous()
        self._sequence_by_sender: dict[str, int] = {}
        self.delivery_history = []

    def reset_post_sim_init(self) -> None:
        """Connect compact communication to each sensor's standard datastore."""
        self.catalogs = {
            sensor.name: sensor.data_store.catalog for sensor in self.sensing_satellites
        }
        self.inboxes = {
            sensor.name: IntentStatusInbox(sensor.name, self.catalogs[sensor.name])
            for sensor in self.sensing_satellites
        }
        self.channel = PerfectMetadataChannel(self.inboxes)
        self.centralized_view = CentralizedInformationView(self.catalogs)
        for sensor in self.sensing_satellites:
            sensor.intent_status_inbox = self.inboxes[sensor.name]
            sensor.centralized_information_view = self.centralized_view

    def communication_pairs(self):
        """Compatibility hook; typed messages use explicit directions instead."""
        return []

    def _next_sequence(self, sender: str) -> int:
        value = self._sequence_by_sender.get(sender, -1) + 1
        self._sequence_by_sender[sender] = value
        return value

    @staticmethod
    def _broadcast_pending(sensor) -> bool:
        return any(
            bool(getattr(action, "broadcast_pending", False))
            for action in sensor.action_builder.action_spec
        )

    @staticmethod
    def _clear_broadcast_pending(sensor) -> None:
        for action in sensor.action_builder.action_spec:
            if hasattr(action, "broadcast_pending"):
                action.broadcast_pending = False

    def _latest_catalog_target_id(self, sensor) -> Optional[int]:
        """Return the most recently updated local target in deterministic order."""
        states = sensor.data_store.catalog.targets.values()
        finite_states = [
            state for state in states if np.isfinite(state.last_update_time)
        ]
        if not finite_states:
            return None
        return int(
            max(
                finite_states,
                key=lambda state: (state.last_update_time, -state.target_id),
            ).target_id
        )

    def _message_for(
        self, sensor, target_id: Optional[int], sim_time: float
    ) -> IntentStatusMessage:
        state = (
            sensor.data_store.catalog.target(target_id)
            if target_id is not None
            else None
        )
        return IntentStatusMessage(
            sender=sensor.name,
            sequence_number=self._next_sequence(sensor.name),
            target_id=int(target_id) if target_id is not None else None,
            action=str(getattr(sensor, "_current_action_label", "unknown")),
            creation_time=float(sim_time),
            expiry_time=float(sim_time) + self.message_ttl_s,
            latest_acquisition_time=(
                state.latest_acquisition_time if state is not None else None
            ),
            latest_delivery_time=(
                state.latest_delivery_time if state is not None else None
            ),
            cooldown_until=(
                state.cooldown_until
                if state is not None and state.cooldown_until != float("-inf")
                else None
            ),
            lifecycle_status=(
                "pending_verification"
                if state is not None
                and (state.pending_record_ids or state.remote_pending_sources)
                else (
                    "cooldown"
                    if state is not None and state.cooldown_until > float(sim_time)
                    else (
                        "delivered"
                        if state is not None and state.latest_delivery_time is not None
                        else ("eligible" if state is not None else None)
                    )
                )
            ),
        )

    def _allowed_receivers(self, sender: str, sim_time: float) -> list[str]:
        if self.perfect_metadata_delivery:
            return [
                sensor.name
                for sensor in self.sensing_satellites
                if sensor.name != sender
            ]
        pair_provider = self.directional_pair_provider
        pairs = (
            pair_provider(sim_time)
            if pair_provider is not None
            else self._geometric_directional_pairs()
        )
        return sorted(
            receiver
            for candidate_sender, receiver in pairs
            if candidate_sender == sender and receiver != sender
        )

    def _geometric_directional_pairs(self) -> list[tuple[str, str]]:
        """Return explicit directions with Earth-unoccluded sensor-to-sensor LOS.

        This is intentionally a geometry-only first model: no RF budget, bandwidth,
        packet loss, or delay is implied.
        """
        pairs = []
        for sender in self.sensing_satellites:
            r_sender = np.asarray(sender.dynamics.r_BN_N, dtype=float)
            for receiver in self.sensing_satellites:
                if sender is receiver:
                    continue
                r_receiver = np.asarray(receiver.dynamics.r_BN_N, dtype=float)
                if earth_unoccluded(
                    r_sender,
                    r_receiver,
                    earth_radius_m=float(orbitalMotion.REQ_EARTH * 1e3),
                ):
                    pairs.append((sender.name, receiver.name))
        return pairs

    def _simulation_time(self) -> float:
        """Return the shared simulator time without selecting a sensor by position."""
        sim_times = {
            float(sensor.simulator.sim_time) for sensor in self.sensing_satellites
        }
        if len(sim_times) != 1:
            raise RuntimeError(
                f"Sensing spacecraft have inconsistent times: {sim_times}"
            )
        return sim_times.pop()

    def communicate(self) -> None:
        """Exchange eligible intent/status metadata at the end of a tasking step."""
        if self.information_case is not InformationCase.INTENT_STATUS:
            return
        sim_time = self._simulation_time()
        for sensor in sorted(self.sensing_satellites, key=lambda sat: sat.name):
            target_id = current_target_id(sensor)
            if not self.perfect_metadata_delivery and not self._broadcast_pending(
                sensor
            ):
                continue
            if target_id is None and not self.perfect_metadata_delivery:
                target_id = self._latest_catalog_target_id(sensor)
            message = self._message_for(sensor, target_id, sim_time)
            self.channel.send(
                message,
                self._allowed_receivers(sensor.name, sim_time),
                available_time=sim_time,
            )
            self._clear_broadcast_pending(sensor)
        self.delivery_history.extend(self.channel.deliver(sim_time))


__all__ = [
    "InformationCase",
    "IntentStatusCommunication",
    "current_target_id",
    "earth_unoccluded",
]
