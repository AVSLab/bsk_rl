"""Received orbit estimates shared by ground clients, independently of transport.

The catalog uses strictly newer information epochs, not covariance fusion. States
are J2000 position/velocity in m and m/s; covariance is a 3x3 RTN position
covariance in m^2. All times are seconds in one explicitly named time reference.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np

__all__ = ["CatalogEstimate", "CatalogPacket", "CatalogEntry", "GroundCatalog"]
__doc_title__ = "Ground Catalog"


def _name(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _time(value: float, label: str) -> float:
    if isinstance(value, (str, bool)):
        raise ValueError(f"{label} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


@dataclass(frozen=True)
class CatalogEstimate:
    """Immutable estimate with original producer identity and information time.

    This record carries a position covariance, not a full orbit-estimation
    covariance. Creating or forwarding it never implies a successful delivery.
    """

    target_id: str
    source_id: str
    update_id: str
    solution_time_s: float
    state_epoch_s: float
    state: tuple[float, ...]
    covariance_rtn: tuple[tuple[float, ...], ...]
    time_reference: str
    state_frame: str = "J2000"
    covariance_frame: str = "RTN"
    state_units: tuple[str, str] = ("m", "m/s")
    covariance_units: str = "m^2"

    def __post_init__(self) -> None:
        """Validate and detach all mutable input arrays."""
        for name in ("target_id", "source_id", "update_id", "time_reference"):
            _name(getattr(self, name), name)
        solution = _time(self.solution_time_s, "solution_time_s")
        epoch = _time(self.state_epoch_s, "state_epoch_s")
        if solution > epoch:
            raise ValueError("Solution time must not exceed the state epoch")
        if self.state_frame != "J2000" or self.covariance_frame != "RTN":
            raise ValueError("Supported frames are J2000 state and RTN covariance")
        if tuple(self.state_units) != ("m", "m/s") or self.covariance_units != "m^2":
            raise ValueError("State and covariance must use the documented SI units")
        state = np.asarray(self.state, dtype=float)
        covariance = np.asarray(self.covariance_rtn, dtype=float)
        if state.shape != (6,) or covariance.shape != (3, 3):
            raise ValueError("Expected six state values and a 3x3 position covariance")
        if not np.isfinite(state).all() or not np.isfinite(covariance).all():
            raise ValueError("State and covariance must be finite")
        if (
            np.linalg.norm(state[:3]) == 0
            or np.linalg.norm(np.cross(state[:3], state[3:])) == 0
        ):
            raise ValueError(
                "RTN covariance requires nonzero position and angular momentum"
            )
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError("Covariance must be symmetric")
        if np.linalg.eigvalsh(covariance).min() <= 0:
            raise ValueError("Covariance must be positive definite")
        object.__setattr__(self, "solution_time_s", solution)
        object.__setattr__(self, "state_epoch_s", epoch)
        object.__setattr__(self, "state", tuple(float(x) for x in state))
        object.__setattr__(
            self,
            "covariance_rtn",
            tuple(tuple(float(x) for x in row) for row in covariance),
        )
        object.__setattr__(self, "state_units", ("m", "m/s"))


@dataclass(frozen=True)
class CatalogPacket:
    """Frozen estimates and forwarding identity; contains no live catalog references."""

    sender_id: str
    created_s: float
    estimates: tuple[CatalogEstimate, ...]
    time_reference: str
    catalog_revision: int | None = None

    def __post_init__(self) -> None:
        """Validate the complete packet and freeze its target ordering."""
        _name(self.sender_id, "sender_id")
        _name(self.time_reference, "time_reference")
        if self.catalog_revision is not None and (
            type(self.catalog_revision) is not int or self.catalog_revision < 0
        ):
            raise ValueError("Catalog revision must be a nonnegative integer")
        created = _time(self.created_s, "created_s")
        estimates = tuple(self.estimates)
        if any(not isinstance(e, CatalogEstimate) for e in estimates):
            raise ValueError("Packets contain CatalogEstimate records")
        if len({e.target_id for e in estimates}) != len(estimates):
            raise ValueError("Duplicate target IDs in packet")
        if any(
            e.state_epoch_s > created or e.time_reference != self.time_reference
            for e in estimates
        ):
            raise ValueError(
                "Estimate epoch or time reference differs from packet contract"
            )
        object.__setattr__(self, "created_s", created)
        object.__setattr__(
            self, "estimates", tuple(sorted(estimates, key=lambda e: e.target_id))
        )

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible representation."""
        return {"schema": "bsk-rl-catalog-packet-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> CatalogPacket:
        """Validate and reconstruct a serialized packet."""
        state = dict(state)
        if state.pop("schema") != "bsk-rl-catalog-packet-v1":
            raise ValueError("Unsupported catalog packet schema")
        state["estimates"] = tuple(CatalogEstimate(**e) for e in state["estimates"])
        return cls(**state)


@dataclass(frozen=True)
class CatalogEntry:
    """Accepted estimate with receiver-local delivery and revision metadata."""

    estimate: CatalogEstimate
    received_s: float
    sender_id: str
    packet_created_s: float
    revision: int


class GroundCatalog:
    """Authoritative received knowledge shared by terrestrial clients.

    Create one instance per environment and reset it between episodes. Spacecraft
    use separate instances (or adapters to their own belief stores). A snapshot
    grants access to ground knowledge; it does not simulate an uplink.

    Equal information times retain the first accepted estimate. Callers define
    stable delivery ordering for simultaneous events. No truth or estimator is
    consulted, and no communication timing or ACK state is owned by this class.
    """

    def __init__(
        self,
        *,
        time_reference: str,
        initial_estimates: Iterable[CatalogEstimate] = (),
        initial_received_s: float = 0.0,
    ) -> None:
        """Validate an explicit prior and retain it for episode resets.

        Args:
            time_reference: Common reference for every time tag, such as
                ``seconds since 2026-01-01T00:00:00 UTC``.
            initial_estimates: Actual estimates already available to the ground.
            initial_received_s: Time when the initial catalog is available.
        """
        _name(time_reference, "time_reference")
        self._time_reference = time_reference
        self._prior = CatalogPacket(
            "common-prior", initial_received_s, tuple(initial_estimates), time_reference
        )
        self.reset()

    def reset(self) -> None:
        """Clear deliveries and restore only the explicitly configured prior."""
        self._entries: dict[str, CatalogEntry] = {}
        self._revision = 0
        self._last_received_s = self._prior.created_s
        self.ingest(self._prior, received_s=self._prior.created_s)

    @property
    def time_reference(self) -> str:
        """Return the fixed time origin and scale shared by this catalog's records."""
        return self._time_reference

    @property
    def revision(self) -> int:
        """Return the receiver-local revision, incremented once per useful packet."""
        return self._revision

    def ingest(self, packet: CatalogPacket, *, received_s: float) -> tuple[str, ...]:
        """Atomically accept strictly newer estimates after successful delivery.

        Validate the complete packet even if its estimates would all be stale.
        Failed or pending transfers must not call this method. Information time
        remains unchanged by receipt or forwarding.
        """
        received = _time(received_s, "received_s")
        if packet.time_reference != self.time_reference:
            raise ValueError("Catalog and packet time references differ")
        if received < packet.created_s or received < self._last_received_s:
            raise ValueError("Receipt must follow creation and prior delivery events")
        accepted = tuple(
            e
            for e in packet.estimates
            if e.target_id not in self._entries
            or e.solution_time_s > self._entries[e.target_id].estimate.solution_time_s
        )
        if accepted:
            revision = self.revision + 1
            replacements = {
                e.target_id: CatalogEntry(
                    e, received, packet.sender_id, packet.created_s, revision
                )
                for e in accepted
            }
            self._entries.update(replacements)
            self._revision = revision
        self._last_received_s = received
        return tuple(e.target_id for e in accepted)

    def snapshot(self) -> Mapping[str, CatalogEntry]:
        """Return an immutable, detached snapshot of currently accepted records."""
        return MappingProxyType(dict(sorted(self._entries.items())))

    def snapshot_at(
        self,
        epoch_s: float,
        *,
        predict: Callable[[CatalogEstimate, float], CatalogEstimate],
    ) -> Mapping[str, CatalogEntry]:
        """Predict copies through a supplied model without making information newer.

        The callback must preserve target, source, update ID, information time,
        frame, units, and time reference. Prediction is never written back.
        """
        epoch = _time(epoch_s, "epoch_s")
        result = {}
        for key, entry in self.snapshot().items():
            original = entry.estimate
            if epoch < original.state_epoch_s:
                raise ValueError("Prediction must not run backward")
            predicted = predict(original, epoch)
            preserved = (
                "target_id",
                "source_id",
                "update_id",
                "solution_time_s",
                "time_reference",
                "state_frame",
                "covariance_frame",
                "state_units",
                "covariance_units",
            )
            if (
                not isinstance(predicted, CatalogEstimate)
                or any(
                    getattr(predicted, name) != getattr(original, name)
                    for name in preserved
                )
                or predicted.state_epoch_s != epoch
            ):
                raise ValueError(
                    "Prediction changed identity, information time, or frame contract"
                )
            result[key] = replace(entry, estimate=predicted)
        return MappingProxyType(result)

    def make_packet(
        self, *, sender_id: str, created_s: float, since_revision: int = 0
    ) -> CatalogPacket:
        """Freeze accepted estimates, optionally newer than a receiver-local revision.

        Transport owns the acknowledged revision. Creating this packet does not
        advance that ledger. Original producer provenance travels with estimates.
        """
        if type(since_revision) is not int or not 0 <= since_revision <= self.revision:
            raise ValueError("Invalid catalog revision")
        created = _time(created_s, "created_s")
        if created < self._last_received_s:
            raise ValueError("Packet creation precedes received knowledge")
        return CatalogPacket(
            sender_id,
            created,
            tuple(
                row.estimate
                for row in self.snapshot().values()
                if row.revision > since_revision
            ),
            self.time_reference,
            self.revision,
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize committed knowledge; transport serializes its own pending state."""
        return {
            "schema": "bsk-rl-ground-catalog-v1",
            "prior": self._prior.to_dict(),
            "revision": self.revision,
            "last_received_s": self._last_received_s,
            "entries": [asdict(row) for row in self.snapshot().values()],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> GroundCatalog:
        """Restore validated committed entries and the original reset prior."""
        if state["schema"] != "bsk-rl-ground-catalog-v1":
            raise ValueError("Unsupported ground catalog schema")
        prior = CatalogPacket.from_dict(state["prior"])
        result = cls(
            time_reference=prior.time_reference,
            initial_estimates=prior.estimates,
            initial_received_s=prior.created_s,
        )
        revision = state["revision"]
        last = _time(state["last_received_s"], "last_received_s")
        if (
            type(revision) is not int
            or revision < result.revision
            or last < prior.created_s
        ):
            raise ValueError("Invalid restored revision or receipt time")
        entries = {}
        for row in state["entries"]:
            estimate = CatalogEstimate(**row["estimate"])
            created = _time(row["packet_created_s"], "packet_created_s")
            CatalogPacket(row["sender_id"], created, (estimate,), result.time_reference)
            received = _time(row["received_s"], "received_s")
            if (
                not created <= received <= last
                or type(row["revision"]) is not int
                or not 1 <= row["revision"] <= revision
            ):
                raise ValueError("Invalid restored entry")
            if estimate.target_id in entries:
                raise ValueError("Duplicate restored target")
            entries[estimate.target_id] = CatalogEntry(
                estimate, received, row["sender_id"], created, row["revision"]
            )
        for key, entry in result._entries.items():
            if (
                key not in entries
                or entries[key].estimate.solution_time_s
                < entry.estimate.solution_time_s
            ):
                raise ValueError("Restored catalog lost an initial estimate")
            if (
                entries[key].estimate.solution_time_s == entry.estimate.solution_time_s
                and entries[key].estimate != entry.estimate
            ):
                raise ValueError(
                    "Restored catalog changed an equal-time initial estimate"
                )
        if bool(entries) != bool(revision) or (
            entries and max(e.revision for e in entries.values()) != revision
        ):
            raise ValueError("Restored revision differs from committed entries")
        result._entries, result._revision, result._last_received_s = (
            entries,
            revision,
            last,
        )
        return result
