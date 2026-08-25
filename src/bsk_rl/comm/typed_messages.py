"""Directional, typed metadata messages for cooperative catalog maintenance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

from bsk_rl.data.multiagent_rso_data import LocalCatalogKnowledge


class MessageDisposition(str, Enum):
    """Deterministic result of attempting to receive a message."""

    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    STALE = "stale"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class IntentStatusMessage:
    """Compact catalog/intent metadata; never an image product or full datastore."""

    sender: str
    sequence_number: int
    target_id: int
    action: str
    creation_time: float
    expiry_time: float
    latest_acquisition_time: Optional[float] = None
    latest_delivery_time: Optional[float] = None
    cooldown_until: Optional[float] = None
    lifecycle_status: Optional[str] = None

    @property
    def message_id(self) -> tuple[str, int]:
        """Return the sender-scoped message identity."""
        return self.sender, int(self.sequence_number)


class IntentStatusInbox:
    """Receiver-side ordering, expiry, de-duplication, and catalog merge."""

    def __init__(self, receiver: str, catalog: LocalCatalogKnowledge) -> None:
        """Create an inbox attached to one receiver's local catalog."""
        self.receiver = str(receiver)
        self.catalog = catalog
        self._seen: set[tuple[str, int]] = set()
        self._highest_sequence: dict[str, int] = {}
        self.latest_intent_by_sender: dict[str, IntentStatusMessage] = {}
        self.history: list[tuple[IntentStatusMessage, MessageDisposition]] = []

    def receive(
        self, message: IntentStatusMessage, sim_time: float
    ) -> MessageDisposition:
        """Validate and deterministically apply one directional message."""
        if float(sim_time) > float(message.expiry_time):
            disposition = MessageDisposition.EXPIRED
        elif message.message_id in self._seen:
            disposition = MessageDisposition.DUPLICATE
        elif int(message.sequence_number) <= self._highest_sequence.get(
            message.sender, -1
        ):
            disposition = MessageDisposition.STALE
        else:
            self._seen.add(message.message_id)
            self._highest_sequence[message.sender] = int(message.sequence_number)
            self.latest_intent_by_sender[message.sender] = message
            self.catalog.merge_status(
                target_id=message.target_id,
                acquisition_time=message.latest_acquisition_time,
                delivery_time=message.latest_delivery_time,
                cooldown_until=message.cooldown_until,
                lifecycle_status=message.lifecycle_status,
                update_time=message.creation_time,
                expiry_time=message.expiry_time,
                source_sensor=message.sender,
            )
            disposition = MessageDisposition.ACCEPTED
        self.history.append((message, disposition))
        return disposition


@dataclass(frozen=True, slots=True)
class DirectedMessage:
    sender: str
    receiver: str
    message: IntentStatusMessage
    available_time: float


class PerfectMetadataChannel:
    """Perfect directional delivery used to validate message semantics first."""

    def __init__(self, inboxes: dict[str, IntentStatusInbox]) -> None:
        """Create a perfect channel over the provided receiver inboxes."""
        self.inboxes = dict(inboxes)
        self._pending: list[DirectedMessage] = []

    def send(
        self,
        message: IntentStatusMessage,
        receivers: Iterable[str],
        *,
        available_time: Optional[float] = None,
    ) -> None:
        """Queue one message in each requested direction."""
        if available_time is None:
            available_time = message.creation_time
        for receiver in sorted(set(map(str, receivers))):
            if receiver == message.sender:
                continue
            if receiver not in self.inboxes:
                raise KeyError(f"Unknown metadata receiver {receiver!r}.")
            self._pending.append(
                DirectedMessage(
                    sender=message.sender,
                    receiver=receiver,
                    message=message,
                    available_time=float(available_time),
                )
            )

    def deliver(
        self, sim_time: float
    ) -> list[tuple[DirectedMessage, MessageDisposition]]:
        """Deliver all messages available by ``sim_time`` in stable order."""
        ready = [
            item for item in self._pending if item.available_time <= float(sim_time)
        ]
        self._pending = [
            item for item in self._pending if item.available_time > float(sim_time)
        ]
        ready.sort(
            key=lambda item: (
                item.available_time,
                item.receiver,
                item.sender,
                item.message.sequence_number,
            )
        )
        return [
            (item, self.inboxes[item.receiver].receive(item.message, sim_time))
            for item in ready
        ]


class CentralizedInformationView:
    """Ideal metadata snapshot kept separate from each sensor's local catalog."""

    def __init__(self, catalogs: dict[str, LocalCatalogKnowledge]) -> None:
        """Create a read-only ideal view over separate local catalogs."""
        self.catalogs = dict(catalogs)

    def target_snapshot(self, target_id: int) -> dict[str, object]:
        """Return the newest known fields across sensors without mutating any catalog."""
        states = [catalog.target(target_id) for catalog in self.catalogs.values()]
        acquisition_times = [
            state.latest_acquisition_time
            for state in states
            if state.latest_acquisition_time is not None
        ]
        delivery_times = [
            state.latest_delivery_time
            for state in states
            if state.latest_delivery_time is not None
        ]
        return {
            "target_id": int(target_id),
            "latest_acquisition_time": max(acquisition_times, default=None),
            "latest_delivery_time": max(delivery_times, default=None),
            "cooldown_until": max(state.cooldown_until for state in states),
            "pending_anywhere": any(
                state.pending_record_ids or state.remote_pending_sources
                for state in states
            ),
        }


__all__ = [
    "CentralizedInformationView",
    "DirectedMessage",
    "IntentStatusInbox",
    "IntentStatusMessage",
    "MessageDisposition",
    "PerfectMetadataChannel",
]
