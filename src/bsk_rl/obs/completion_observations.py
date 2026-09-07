"""The versioned, completion-only observation and candidate/action contract."""

from dataclasses import dataclass

import numpy as np

from bsk_rl.utils.orbital import rv2HN

from bsk_rl.obs.observations import (
    Observation,
    _relative_position_H,
    _relative_velocity_H,
    _angle_to_target,
    _target_distance,
    _target_shadowFactor,
)


OBSERVATION_VERSION = "completion-v2"
PEER_FEATURES = 12
GLOBAL_FEATURES = 26  # 14 inherited own/environment features + 12 below
TARGET_FEATURES = 17
NON_IMAGING_ACTIONS = 5  # charge, downlink, desat, broadcast, deliberate continue
CONTINUE_ACTION = 4
VALID_TARGET_FEATURE = 16
CONTINUE_VALID_FEATURE = 25


@dataclass(frozen=True)
class CandidateSnapshot:
    time: float
    targets: tuple


def candidate_snapshot(sensor, count):
    """Build once per decision: the actor and decoder share these exact IDs.

    Preserve the existing ascending-elevation shortlist, then nearest-distance
    fill. Ineligible targets are never padding; empty slots are explicitly None.
    """
    now = float(sensor.simulator.sim_time)
    cached = getattr(sensor, "completion_candidates", None)
    if cached is not None and cached.time == now:
        return cached
    position = np.asarray(sensor.dynamics.r_BN_N, dtype=float)
    zenith = position / np.linalg.norm(position)
    eligible = []
    for target in sensor.data_store.data.known:
        if (
            target.id in sensor.data_store.unresolved_captures
            or not sensor.data_store.catalog.is_eligible(target.id, now)
        ):
            continue
        relative = np.asarray(target.target_spacecraft.dynamics.r_BN_N) - position
        distance = float(np.linalg.norm(relative))
        if distance <= 0:
            continue
        elevation = float(
            np.degrees(np.arcsin(np.clip(relative @ zenith / distance, -1, 1)))
        )
        eligible.append((target, elevation, distance))
    visible = sorted(
        (row for row in eligible if -21 <= row[1] <= 90),
        key=lambda row: (row[1], int(row[0].id)),
    )
    selected = [row[0] for row in visible[:count]]
    ids = {t.id for t in selected}
    remaining = sorted(
        (row for row in eligible if row[0].id not in ids),
        key=lambda row: (row[2], int(row[0].id)),
    )
    selected.extend(row[0] for row in remaining[: count - len(selected)])
    selected.extend([None] * (count - len(selected)))
    sensor.completion_candidates = CandidateSnapshot(now, tuple(selected))
    return sensor.completion_candidates


def can_continue(sensor):
    task = getattr(sensor, "completion_task", None)
    return bool(task is not None and not task.finished and not task.invalidated)


class CompletionContext(Observation):
    """Own progress and exchange history; no private state of other sensors."""

    def __init__(self, norm=12000.0):
        """Set the normalization time for own execution and exchange history."""
        super().__init__(name="completion_context")
        self.norm = float(norm)

    def get_obs(self):
        """Return the twelve ordered own-state and exchange-history features."""
        sensor, now = self.satellite, float(self.simulator.sim_time)
        task = getattr(sensor, "completion_task", None)
        modes = ("charge", "downlink", "desat", "broadcast", "image")
        mode = task.mode if task is not None else None
        active = getattr(sensor, "_active_image_rso_action", None)
        hold = min(
            1.0,
            getattr(active, "_hold_valid_time_s", 0.0)
            / max(1e-9, getattr(active, "min_pointing_hold_s", 1.0)),
        )
        transmitter = getattr(sensor, "_active_completion_transmit", None)
        if mode == "broadcast" and transmitter is not None:
            hold = min(1.0, transmitter.held_s / transmitter.required_hold_s)
        communicator = sensor.completion_communicator
        exchange = communicator.last_exchange.get(sensor.name)
        return dict(
            mode=[float(mode == m) for m in modes],
            elapsed=(max(0, now - task.start) / self.norm if task else 0),
            remaining=(max(0, task.deadline - now) / self.norm if task else 0),
            hold_fraction=hold,
            unacknowledged_records=communicator.backlog(sensor.name)
            / max(1, len(sensor.data_store.catalog.targets)),
            exchange_age=(0 if exchange is None else (now - exchange) / self.norm),
            exchange_known=float(exchange is not None),
            continue_valid=float(can_continue(sensor)),
        )


class CompletionTargets(Observation):
    """Equal target chunks with service ages, knownness, and explicit padding mask.

    Velocity is inertial velocity difference expressed in own Hill axes, matching
    the inherited imaging model; it is not a rotating-frame position derivative.
    """

    def __init__(self, n_targets, norm_time=12000.0):
        """Configure the fixed candidate count and service-age normalization."""
        super().__init__(name="completion_targets")
        self.n_targets, self.norm_time = int(n_targets), float(norm_time)

    def get_obs(self):
        """Encode the cached candidate tuple with explicit validity and knownness."""
        sensor, now = self.satellite, float(self.simulator.sim_time)
        catalog = sensor.data_store.catalog
        snapshot = candidate_snapshot(sensor, self.n_targets)
        task = getattr(sensor, "completion_task", None)
        result = {}
        for index, target in enumerate(snapshot.targets):
            if target is None:
                result[f"slot_{index}"] = np.zeros(TARGET_FEATURES)
                continue
            opportunity = {"object": target}
            acquired = catalog.target(target.id).latest_acquisition_time
            delivered = catalog.freshest_delivered_capture(target.id)
            values = [
                float(target.priority) / 10.0,
                *(_relative_position_H(sensor, opportunity) / 15960e3),
                *(_relative_velocity_H(sensor, opportunity) / 12000.0),
                _angle_to_target(sensor, opportunity) / 90.0,
                _target_distance(sensor, opportunity) / 15960e3,
                _target_shadowFactor(sensor, opportunity),
                0 if acquired is None else (now - acquired) / self.norm_time,
                float(acquired is not None),
                0 if delivered is None else (now - delivered) / self.norm_time,
                float(delivered is not None),
                float(bool(catalog.target(target.id).pending_record_ids)),
                float(
                    task is not None
                    and task.target_id == target.id
                    and can_continue(sensor)
                ),
                1.0,
            ]
            result[f"slot_{index}"] = np.asarray(values, dtype=float)
        return result


__all__ = ["CompletionContext", "CompletionTargets"]


@dataclass(frozen=True)
class PeerSnapshot:
    time: float
    peers: tuple


def peer_snapshot(sensor):
    """Freeze visible, nonempty-delta recipients under ideal contact discovery.

    The navigation beacon reveals current position/velocity only while LOS is
    available. No resource, intention or private catalog is part of that beacon.
    Stable peer slots use configured sensor names; invisible slots contain None.
    """
    now = float(sensor.simulator.sim_time)
    cached = getattr(sensor, "completion_peers", None)
    if cached is not None and cached.time == now:
        return cached
    channel = sensor.completion_communicator
    contacts = (
        channel.receivers(sensor) if channel.information_case == "completion" else set()
    )
    peers = tuple(
        peer
        if peer.name in contacts and channel.delta(sensor.name, peer.name)
        else None
        for peer in sorted(channel.sensors, key=lambda p: p.name)
        if peer is not sensor
    )
    sensor.completion_peers = PeerSnapshot(now, peers)
    return sensor.completion_peers


class CompletionPeers(Observation):
    """Twelve features per peer: contact geometry and sender-local exchange history."""

    def __init__(self, norm_time=12000.0):
        super().__init__(name="completion_peers")
        self.norm_time = norm_time

    def get_obs(self):
        sensor, now = self.satellite, float(self.simulator.sim_time)
        channel = sensor.completion_communicator
        result = {}
        for index, peer in enumerate(peer_snapshot(sensor).peers):
            if peer is None:
                result[f"peer_{index}"] = np.zeros(PEER_FEATURES)
                continue
            # This read represents an ideal fresh navigation beacon, not arbitrary
            # actor access to uncommunicated peer state. The same beacon drives FSW.
            relative = np.asarray(peer.dynamics.r_BN_N) - sensor.dynamics.r_BN_N
            velocity = np.asarray(peer.dynamics.v_BN_N) - sensor.dynamics.v_BN_N
            hill = rv2HN(sensor.dynamics.r_BN_N, sensor.dynamics.v_BN_N)
            distance = np.linalg.norm(relative)
            boresight = np.asarray(sensor.dynamics.BN).T @ np.asarray(
                sensor.fsw.locPoint.pHat_B
            )
            angle = np.degrees(
                np.arccos(np.clip(boresight @ relative / max(distance, 1e-9), -1, 1))
            )
            exchange = channel.peer_exchange.get((sensor.name, peer.name))
            result[f"peer_{index}"] = np.array(
                [
                    *(hill @ relative / 15960e3),
                    *(hill @ velocity / 12000),
                    angle / 180,
                    distance / 15960e3,
                    len(channel.delta(sensor.name, peer.name))
                    / max(1, len(channel.catalogs[sensor.name].targets)),
                    0 if exchange is None else (now - exchange) / self.norm_time,
                    float(exchange is not None),
                    1.0,
                ]
            )
        return result
