"""Finite-duration exchange of completion records, with delayed/lost packets.

Payloads contain completed exposure facts only. The channel has no access to peer
actions, reward truth, or physical image products. Ground downlink is separate.
"""

from dataclasses import dataclass
import math

import numpy as np

from bsk_rl.comm.communication import CommunicationMethod
from bsk_rl.data.completion_catalog import CompletionRecord
from bsk_rl.sats.roles import SpacecraftRole
from bsk_rl.utils.coordination import earth_unoccluded


@dataclass(frozen=True)
class CompletionPacket:
    sender: str
    receiver: str
    sequence: int
    created_at: float
    ready_at: float
    expires_at: float
    records: tuple[CompletionRecord, ...]


@dataclass
class Transmission:
    start: float
    end: float
    records: tuple[CompletionRecord, ...]
    receivers: set[str]
    directed: bool = False


class CompletionCommunication(CommunicationMethod):
    """Independent, ideal-completion, or finite-completion information channels.

    The broadcast baseline samples Earth clearance at environment boundaries.
    Directed actions check sender pointing and LOS every FSW tick and release a
    packet only after a complete continuous hold. Receivers are omnidirectional.
    Packet loss and latency are configurable. Successful delivery acts as an ideal
    small ACK; lost records remain eligible for a later explicit transmission.
    """

    def __init__(
        self,
        information_case="completion",
        *,
        delay_s=0.0,
        loss_probability=0.0,
        ttl_s=600.0,
        link_mode="los",
    ):
        """Configure transport timing, impairment, and geometric link assumptions."""
        super().__init__(min_period=0.0)
        if information_case not in {"independent", "ideal_completion", "completion"}:
            raise ValueError("Unknown completion information case.")
        if link_mode not in {"ideal", "los"}:
            raise ValueError("link_mode must be ideal or los.")
        if (
            not 0 <= loss_probability <= 1
            or not math.isfinite(delay_s)
            or delay_s < 0
            or not math.isfinite(ttl_s)
            or ttl_s <= 0
        ):
            raise ValueError("Invalid packet timing/loss configuration.")
        self.information_case = information_case
        self.delay_s, self.loss_probability, self.ttl_s = (
            float(delay_s),
            float(loss_probability),
            float(ttl_s),
        )
        self.link_mode = link_mode

    def link_satellites(self, satellites):
        """Register sensing spacecraft only as metadata senders and receivers."""
        super().link_satellites(satellites)
        self.sensors = [s for s in satellites if s.role is SpacecraftRole.SENSING_AGENT]
        for sensor in self.sensors:
            sensor.information_case = self.information_case

    def reset_overwrite_previous(self):
        """Clear transport, ACK, and diagnostic state for a new episode."""
        super().reset_overwrite_previous()
        self.transmissions: dict[str, Transmission] = {}
        self.pending: list[CompletionPacket] = []
        self.acknowledged = {}
        self.sequence = 0
        self.delivery_history = []
        self.transmission_history = []
        self.last_exchange = {}
        self.peer_exchange = {}

    def reset_post_sim_init(self):
        """Bind new local catalogs and initialize the episode-local random stream."""
        self.catalogs = {s.name: s.data_store.catalog for s in self.sensors}
        # Reset happens after the environment seeds numpy. Keep channel randomness
        # isolated from subsequent policy/environment random draws.
        self.rng = np.random.default_rng(int(np.random.randint(0, 2**31)))
        for sensor in self.sensors:
            sensor.completion_communicator = self

    def communication_pairs(self):
        # Do not invoke the generic whole-datastore union implementation.
        """Disable generic whole-datastore union in this metadata-only channel."""
        return []

    def receivers(self, sender):
        """Return surviving peers permitted by the configured geometric link."""
        return {
            peer.name
            for peer in self.sensors
            if peer is not sender
            and peer.is_alive()
            and (
                self.link_mode == "ideal"
                or earth_unoccluded(sender.dynamics.r_BN_N, peer.dynamics.r_BN_N)
            )
        }

    def begin_transmission(self, sensor, now, duration):
        """Freeze the payload at action start; no delivery is possible yet."""
        if self.information_case != "completion":
            return
        self.transmissions[sensor.name] = Transmission(
            start=float(now),
            end=float(now + duration),
            records=tuple(
                r for r in sensor.data_store.catalog.records.values() if r.qualified
            ),
            receivers=self.receivers(sensor),
        )

    def delta(self, sender, receiver):
        """Sender-local facts minus its ACK history; never read the peer catalog."""
        ack = self.acknowledged.get((sender, receiver), {})
        return tuple(
            r
            for r in self.catalogs[sender].records.values()
            if r.qualified
            and r.source_sensor != receiver
            and ack.get(r.record_id, 0) < r.version
        )

    def begin_directed(self, sensor, receiver, now):
        """Freeze one recipient's delta. Only the physical hold gate can finish it."""
        if self.information_case != "completion" or receiver.name not in self.receivers(
            sensor
        ):
            raise ValueError("Directed transmission requires a discovered contact.")
        records = self.delta(sensor.name, receiver.name)
        if not records:
            raise ValueError("No unacknowledged completion records for this peer.")
        self.transmissions[sensor.name] = Transmission(
            float(now), float("inf"), records, {receiver.name}, directed=True
        )
        return records

    def cancel_transmission(self, sender, now):
        """Discard an unfinished radio operation without delivering its snapshot."""
        transmission = self.transmissions.pop(sender, None)
        if transmission is not None:
            self.transmission_history.append(
                dict(
                    sender=sender,
                    start=transmission.start,
                    end=float(now),
                    completed=False,
                    records=0,
                )
            )

    def backlog(self, sender):
        """Number of records not yet acknowledged by every configured peer."""
        peers = [p.name for p in self.sensors if p.name != sender]
        return sum(
            any(
                self.acknowledged.get((sender, p), {}).get(r.record_id, 0) < r.version
                for p in peers
                if p != r.source_sensor
            )
            for r in self.catalogs[sender].records.values()
            if r.qualified
        )

    def next_event_time(self, now):
        """Let the environment stop at packet arrival even if all agents are busy."""
        return min(
            [p.ready_at for p in self.pending if p.ready_at > now]
            + [t.end for t in self.transmissions.values() if t.end > now],
            default=float("inf"),
        )

    def _queue(self, sender, receiver, records, now, *, ideal=False):
        ack = self.acknowledged.get((sender, receiver), {})
        # An exposure's original owner already knows the fact it generated.
        delta = tuple(
            r
            for r in records
            if r.source_sensor != receiver and ack.get(r.record_id, 0) < r.version
        )
        if not delta:
            return
        self.sequence += 1
        self.pending.append(
            CompletionPacket(
                sender,
                receiver,
                self.sequence,
                now,
                now + (0.0 if ideal else self.delay_s),
                now + self.ttl_s,
                delta,
            )
        )

    def communicate(self):
        """Finish transmissions first, then deliver only packets ready by now."""
        if self.information_case == "independent":
            return
        now = float(self.sensors[0].simulator.sim_time)
        if self.information_case == "ideal_completion":
            # Snapshot all senders before merging: no accidental within-boundary
            # relay caused by iteration order. This is the ideal metadata reference.
            snapshots = {
                s.name: tuple(
                    r for r in self.catalogs[s.name].records.values() if r.qualified
                )
                for s in self.sensors
            }
            for sender in self.sensors:
                if sender.is_alive():
                    for receiver in self.sensors:
                        if sender is not receiver and receiver.is_alive():
                            self._queue(
                                sender.name,
                                receiver.name,
                                snapshots[sender.name],
                                now,
                                ideal=True,
                            )
        else:
            for sensor in self.sensors:
                tx = self.transmissions.get(sensor.name)
                if tx is None:
                    continue
                if not sensor.is_alive():
                    self.cancel_transmission(sensor.name, now)
                    continue
                # Directed FSW checks LOS and pointing every control tick. A
                # broken contact resets its continuous hold; it may reacquire
                # before the task deadline. Broadcast retains its baseline rule.
                if not tx.directed:
                    tx.receivers.intersection_update(self.receivers(sensor))
                if now + 1e-9 < tx.end:
                    continue  # A teammate's shorter action cannot finish this radio.
                for receiver in sorted(tx.receivers):
                    self._queue(sensor.name, receiver, tx.records, now)
                self.transmission_history.append(
                    dict(
                        sender=sensor.name,
                        start=tx.start,
                        end=now,
                        completed=True,
                        records=len(tx.records),
                        receivers=sorted(tx.receivers),
                        directed=tx.directed,
                    )
                )
                del self.transmissions[sensor.name]

        ready = sorted(
            (p for p in self.pending if p.ready_at <= now + 1e-9),
            key=lambda p: (p.ready_at, p.receiver, p.sender, p.sequence),
        )
        self.pending = [p for p in self.pending if p.ready_at > now + 1e-9]
        alive = {s.name for s in self.sensors if s.is_alive()}
        for packet in ready:
            outcome = "accepted"
            if packet.receiver not in alive:
                outcome = "receiver_unavailable"
            elif now > packet.expires_at:
                outcome = "expired"
            elif (
                self.information_case != "ideal_completion"
                and self.rng.random() < self.loss_probability
            ):
                outcome = "lost"
            changed = 0
            if outcome == "accepted":
                for record in packet.records:
                    changed += self.catalogs[packet.receiver].merge_record(record, now)
                    ack = self.acknowledged.setdefault(
                        (packet.sender, packet.receiver), {}
                    )
                    ack[record.record_id] = max(
                        ack.get(record.record_id, 0), record.version
                    )
                    # Receipt proves that the transmitting peer already knew this
                    # version. Avoid a pointless echo without inspecting its catalog.
                    reverse = self.acknowledged.setdefault(
                        (packet.receiver, packet.sender), {}
                    )
                    reverse[record.record_id] = max(
                        reverse.get(record.record_id, 0), record.version
                    )
                self.last_exchange[packet.receiver] = now
                self.last_exchange[packet.sender] = now
                self.peer_exchange[(packet.sender, packet.receiver)] = now
                self.peer_exchange[(packet.receiver, packet.sender)] = now
            self.delivery_history.append(
                dict(
                    sender=packet.sender,
                    receiver=packet.receiver,
                    sequence=packet.sequence,
                    sent_at=packet.created_at,
                    received_at=now,
                    outcome=outcome,
                    changed=changed,
                    records=len(packet.records),
                )
            )


__all__ = ["CompletionCommunication", "CompletionPacket"]
