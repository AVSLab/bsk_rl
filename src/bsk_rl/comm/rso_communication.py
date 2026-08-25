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
from bsk_rl.data.multiagent_rso_data import (
    LocalCatalogKnowledge,
    SensorProductStore,
)
from bsk_rl.sats.roles import SpacecraftRole


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

    def reset_overwrite_previous(self) -> None:
        """Reset episode message ordering and delivery logs."""
        super().reset_overwrite_previous()
        self._sequence_by_sender: dict[str, int] = {}
        self.delivery_history = []

    def reset_pre_sim_init(self) -> None:
        """Allocate per-sensor catalogs, physical stores, and inboxes."""
        target_ids = []
        for target in self.passive_targets:
            rso_target = getattr(target, "rso_target", None)
            if rso_target is None:
                raise RuntimeError(
                    "Passive targets must be registered by the RSO scenario before "
                    "communication initialization."
                )
            target_ids.append(int(rso_target.id))

        self.catalogs = {
            sensor.name: LocalCatalogKnowledge(sensor.name, target_ids)
            for sensor in self.sensing_satellites
        }
        self.product_stores = {
            sensor.name: SensorProductStore(sensor.name)
            for sensor in self.sensing_satellites
        }
        self.inboxes = {
            sensor.name: IntentStatusInbox(sensor.name, self.catalogs[sensor.name])
            for sensor in self.sensing_satellites
        }
        self.channel = PerfectMetadataChannel(self.inboxes)
        self.centralized_view = CentralizedInformationView(self.catalogs)
        for sensor in self.sensing_satellites:
            sensor.local_catalog = self.catalogs[sensor.name]
            sensor.physical_product_store = self.product_stores[sensor.name]
            sensor.intent_status_inbox = self.inboxes[sensor.name]
            sensor.centralized_information_view = self.centralized_view
            sensor.information_case = self.information_case.value

    def communication_pairs(self):
        """Compatibility hook; typed messages use explicit directions instead."""
        return []

    def _next_sequence(self, sender: str) -> int:
        value = self._sequence_by_sender.get(sender, -1) + 1
        self._sequence_by_sender[sender] = value
        return value

    @staticmethod
    def _current_target_id(sensor) -> Optional[int]:
        for action in sensor.action_builder.action_spec:
            chosen = getattr(action, "chosen_target_ids", None)
            if chosen:
                return int(chosen[-1])
        return None

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

    def _message_for(self, sensor, target_id: int, sim_time: float):
        state = self.catalogs[sensor.name].target(target_id)
        return IntentStatusMessage(
            sender=sensor.name,
            sequence_number=self._next_sequence(sensor.name),
            target_id=int(target_id),
            action=str(getattr(sensor, "_current_action_label", "unknown")),
            creation_time=float(sim_time),
            expiry_time=float(sim_time) + self.message_ttl_s,
            latest_acquisition_time=state.latest_acquisition_time,
            latest_delivery_time=state.latest_delivery_time,
            cooldown_until=(
                state.cooldown_until if state.cooldown_until != float("-inf") else None
            ),
            lifecycle_status=(
                "pending_verification"
                if state.pending_record_ids or state.remote_pending_sources
                else (
                    "cooldown"
                    if state.cooldown_until > float(sim_time)
                    else (
                        "delivered"
                        if state.latest_delivery_time is not None
                        else "eligible"
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
        earth_radius_m = float(orbitalMotion.REQ_EARTH * 1e3)
        pairs = []
        for sender in self.sensing_satellites:
            r_sender = np.asarray(sender.dynamics.r_BN_N, dtype=float)
            for receiver in self.sensing_satellites:
                if sender is receiver:
                    continue
                r_receiver = np.asarray(receiver.dynamics.r_BN_N, dtype=float)
                segment = r_receiver - r_sender
                denom = float(segment @ segment)
                if denom == 0.0:
                    visible = True
                else:
                    fraction = float(np.clip(-(r_sender @ segment) / denom, 0.0, 1.0))
                    closest = r_sender + fraction * segment
                    visible = float(np.linalg.norm(closest)) > earth_radius_m
                if visible:
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
            target_id = self._current_target_id(sensor)
            if target_id is None:
                continue
            if not self.perfect_metadata_delivery and not self._broadcast_pending(
                sensor
            ):
                continue
            message = self._message_for(sensor, target_id, sim_time)
            self.channel.send(
                message,
                self._allowed_receivers(sensor.name, sim_time),
                available_time=sim_time,
            )
            self._clear_broadcast_pending(sensor)
        self.delivery_history.extend(self.channel.deliver(sim_time))


__all__ = ["InformationCase", "IntentStatusCommunication"]
