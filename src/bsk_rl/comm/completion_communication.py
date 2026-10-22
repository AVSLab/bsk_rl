"""Finite-duration exchange of completion records, with delayed/lost packets.

Payloads contain completed exposure facts only. The channel has no access to peer
actions, reward truth, or physical image products. Ground downlink is separate.
"""

from dataclasses import asdict, dataclass
import json
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
    payload_bytes: int = 0
    inventory_snapshot: object = None


@dataclass
class Transmission:
    start: float
    end: float
    records: tuple[CompletionRecord, ...]
    receivers: set[str]
    directed: bool = False
    payload_bytes: int = 0
    inventory_snapshot: object = None


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
        inventory_chunk_records=256,
    ):
        """Configure transport timing, impairment, and geometric link assumptions."""
        super().__init__(min_period=0.0)
        if information_case not in {"independent", "ideal_completion", "completion", "catalog_tx", "geometry_only"}:
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
        self.inventory_chunk_records = int(inventory_chunk_records)

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
        self.inventory_refresh = {}
        self.inventory_record_refresh = {}

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
        records = tuple(
            r for r in sensor.data_store.catalog.records.values() if r.qualified
        )
        self.transmissions[sensor.name] = Transmission(
            start=float(now),
            end=float(now + duration),
            records=records,
            receivers=self.receivers(sensor),
            payload_bytes=self.wire_payload_bytes(records),
        )

    @staticmethod
    def wire_payload_bytes(records):
        """Canonical completion wire size, including the 64-byte header."""
        return 64 + len(
            json.dumps(
                [asdict(record) for record in records],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )

    def delta(self, sender, receiver):
        """Sender-local facts minus its ACK history; never read the peer catalog."""
        ack = self.acknowledged.get((sender, receiver), {})
        if self.information_case == "geometry_only": return ()
        if self.information_case == "catalog_tx":
            return self._inventory_delta(sender, receiver)
        return tuple(
            r
            for r in self.catalogs[sender].records.values()
            if r.qualified
            and r.source_sensor != receiver
            and ack.get(r.record_id, 0) < r.version
        )

    def _inventory_delta(self, sender, receiver):
        ack = self.acknowledged.get((sender, receiver), {})
        now = float(next(s for s in self.sensors if s.name == sender).simulator.sim_time)
        refresh = now >= self.inventory_refresh.get((sender, receiver), float("-inf")) + self.ttl_s
        return tuple(e for e in sorted(self.catalogs[sender].owned_inventory.values(), key=lambda e: (ack.get(e.record_id, 0) >= e.version, e.revision, e.record_id))
            if now >= self.inventory_record_refresh.get((sender, receiver, e.record_id), float("-inf")) + self.ttl_s or ack.get(e.record_id, 0) < e.version)

    def transmit_eligible(self, sensor, peer):
        eligible = (self.information_case in {"completion", "catalog_tx"} and peer is not None
                and peer.name in self.receivers(sensor) and bool(self.delta(sensor.name, peer.name)))
        if not eligible: return False
        if self.information_case == "catalog_tx":
            snapshot = self.frozen_inventory(sensor.name, peer.name, float(sensor.simulator.sim_time))
            bitrate = getattr(sensor,"completion_metadata_bitrate_bps",None)
            duration = getattr(sensor,"completion_transmit_duration_s",300.)
            hold = getattr(sensor,"completion_transmit_hold_s",10.)
            if bitrate and max(hold,8*snapshot.wire_bytes()/bitrate)>duration: return False
        return True

    def frozen_inventory(self, sender, receiver, now):
        from bsk_rl.data.onboard_inventory import InventorySnapshot
        catalog = self.catalogs[sender]
        all_entries = tuple(sorted(catalog.owned_inventory.values(), key=lambda e: (e.revision, e.record_id)))
        complete = len(all_entries) <= self.inventory_chunk_records
        records = all_entries if complete else self.delta(sender,receiver)[:self.inventory_chunk_records]
        return InventorySnapshot(sender,catalog.inventory_revision,float(now),records,complete)

    def begin_directed(self, sensor, receiver, now):
        """Freeze one recipient's delta. Only the physical hold gate can finish it."""
        if self.information_case not in {"completion", "catalog_tx"} or receiver.name not in self.receivers(
            sensor
        ):
            raise ValueError("Directed transmission requires a discovered contact.")
        if not self.transmit_eligible(sensor,receiver): raise ValueError("No feasible inventory packet")
        records = self.delta(sensor.name, receiver.name)
        if not records:
            raise ValueError("No unacknowledged completion records for this peer.")
        snapshot = None
        if self.information_case == "catalog_tx":
            snapshot = self.frozen_inventory(sensor.name,receiver.name,now)
            records = snapshot.entries
        self.transmissions[sensor.name] = Transmission(
            float(now),
            float("inf"),
            records,
            {receiver.name},
            directed=True,
            payload_bytes=snapshot.wire_bytes() if snapshot else self.wire_payload_bytes(records),
            inventory_snapshot=snapshot,
        )
        return records

    def cancel_transmission(self, sender, now):
        """Discard an unfinished radio operation without delivering its snapshot."""
        transmission = self.transmissions.pop(sender, None)
        if transmission is not None:
            # Failure can cancel in the communication layer before an action
            # callback runs. Preserve powered time in that path as well.
            sensor=next((s for s in self.sensors if s.name==sender),None)
            action=getattr(sensor,"_active_completion_transmit",None)
            radio={}
            if action is not None and not getattr(action,"_audit_recorded",False):
                radio=dict(radio_on_s=action.radio_on_s+(max(0.,float(now)-action.last_time) if action.was_valid else 0.),
                    outcome="cancelled_without_action_audit")
                action._audit_recorded=True
            self.transmission_history.append(
                dict(
                    sender=sender,
                    start=transmission.start,
                    end=float(now),
                    completed=False,
                    records=len(transmission.records),
                    record_ids=[record.record_id for record in transmission.records],
                    payload_bytes=transmission.payload_bytes,
                    receivers=sorted(transmission.receivers),
                    directed=transmission.directed,
                    **radio,
                )
            )

    def backlog(self, sender):
        """Number of records not yet acknowledged by every configured peer."""
        peers = [p.name for p in self.sensors if p.name != sender]
        if self.information_case in {"catalog_tx", "geometry_only"}:
            return len({r.record_id for peer in peers for r in self.delta(sender, peer)})
        return sum(
            any(
                self.acknowledged.get((sender, p), {}).get(r.record_id, 0) < r.version
                for p in peers
                if p != r.source_sensor
            )
            for r in self.catalogs[sender].records.values()
            if r.qualified
        )

    def radio_audit(self, now):
        """Evaluation-only powered durations; include an unfinished final task.

        Action hold rows are authoritative. Generic packet/completion rows do
        not carry radio_on_s and must not be counted a second time.
        """
        rows = [dict(row) for row in self.transmission_history if "radio_on_s" in row]
        for sensor in self.sensors:
            action = getattr(sensor, "_active_completion_transmit", None)
            if action is not None and sensor.name in self.transmissions and not getattr(action,"_audit_recorded",False):
                rows.append(dict(sender=sensor.name, start=action.start_time,
                    end=float(now), outcome="unfinished_at_episode_end",
                    radio_on_s=action.radio_on_s + (max(0., float(now)-action.last_time) if action.was_valid else 0.)))
        return rows

    def next_event_time(self, now):
        """Let the environment stop at packet arrival even if all agents are busy."""
        return min(
            [p.ready_at for p in self.pending if p.ready_at > now]
            + [t.end for t in self.transmissions.values() if t.end > now],
            default=float("inf"),
        )

    def _queue(self, sender, receiver, records, now, *, ideal=False, snapshot=None):
        ack = self.acknowledged.get((sender, receiver), {})
        # An exposure's original owner already knows the fact it generated.
        delta = tuple(
            r
            for r in records
            if r.source_sensor != receiver and ack.get(r.record_id, 0) < r.version
        )
        if snapshot is not None: delta = snapshot.entries
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
                snapshot.epoch + self.ttl_s if snapshot else now + self.ttl_s,
                delta,
                snapshot.wire_bytes() if snapshot else self.wire_payload_bytes(delta),
                snapshot,
            )
        )

    def communicate(self):
        """Finish transmissions first, then deliver only packets ready by now."""
        if self.information_case in {"independent", "geometry_only"}:
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
                    self._queue(sensor.name, receiver, tx.records, now, snapshot=tx.inventory_snapshot)
                self.transmission_history.append(
                    dict(
                        sender=sensor.name,
                        start=tx.start,
                        end=now,
                        completed=True,
                        records=len(tx.records),
                        record_ids=[record.record_id for record in tx.records],
                        payload_bytes=tx.payload_bytes,
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
            elif now >= packet.expires_at:
                outcome = "expired"
            elif (
                self.information_case != "ideal_completion"
                and self.rng.random() < self.loss_probability
            ):
                outcome = "lost"
            changed = 0
            # Frozen-policy evaluation intervention only. Transport, payload,
            # loss draws, timing and ideal ACK bookkeeping remain unchanged.
            # No mission configuration or training default enables this flag.
            suppress_catalog = getattr(self, "_frozen_eval_suppress_catalog_updates", False)
            if outcome == "accepted":
                if packet.inventory_snapshot is not None:
                    if not suppress_catalog:
                        changed += self.catalogs[packet.receiver].receive_inventory(packet.inventory_snapshot, now, self.ttl_s)
                    self.inventory_refresh[(packet.sender, packet.receiver)] = packet.inventory_snapshot.epoch
                    for entry in packet.inventory_snapshot.entries:
                        self.inventory_record_refresh[(packet.sender, packet.receiver,entry.record_id)] = packet.inventory_snapshot.epoch
                for record in packet.records:
                    if packet.inventory_snapshot is None and not suppress_catalog:
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
                    **({"evaluation_intervention": "suppress_receiver_catalog_v1"}
                       if suppress_catalog else {}),
                    sender=packet.sender,
                    receiver=packet.receiver,
                    sequence=packet.sequence,
                    sent_at=packet.created_at,
                    received_at=now,
                    outcome=outcome,
                    changed=changed,
                    records=len(packet.records),
                    record_ids=[record.record_id for record in packet.records],
                    record_versions={
                        record.record_id: record.version for record in packet.records
                    },
                    payload_bytes=packet.payload_bytes,
                    inventory_snapshot=(asdict(packet.inventory_snapshot) if packet.inventory_snapshot else None),
                    acknowledged_versions=(
                        dict(self.acknowledged.get((packet.sender, packet.receiver), {}))
                        if outcome == "accepted"
                        else {}
                    ),
                )
            )


__all__ = ["CompletionCommunication", "CompletionPacket"]
