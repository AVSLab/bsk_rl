"""Independent spacecraft targets for space-to-space imaging.

Unlike :class:`~bsk_rl.scene.RSOPoints`, these targets are whole spacecraft,
not surface points on one nearby RSO. Catalog states are Earth-centered J2000,
in meters and meters per second, at the catalog's optional ``utc_init`` or the
environment's realized epoch when no catalog epoch is specified.
"""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from bsk_rl.scene.scenario import Scenario


@dataclass(frozen=True)
class RSOTarget:
    """Immutable definition of one independently orbiting imaging target.

    IDs are nonempty strings; they have no relationship to simulator indices.
    ``satellite_name`` is :attr:`~bsk_rl.sats.Satellite.name`, the environment's
    agent identifier. It is distinct from the Basilisk spacecraft model's name
    and binds exactly one explicitly supplied satellite, rather than a group.
    Physical parameters are optional ``(sat_args key, scalar value)`` pairs.
    """

    id: str
    satellite_name: str
    rN: tuple[float, float, float]
    vN: tuple[float, float, float]
    priority: float = 1.0
    sigma_init: tuple[float, float, float] = (0.0, 0.0, 0.0)
    omega_init: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wheel_speeds_rpm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    disturbance_torque_Nm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    physical_parameters: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        """Validate and freeze values supplied as lists or numpy vectors."""
        for key in ("id", "satellite_name"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError(f"{key} must be a nonempty string.")
        for key in (
            "rN",
            "vN",
            "sigma_init",
            "omega_init",
            "wheel_speeds_rpm",
            "disturbance_torque_Nm",
        ):
            value = tuple(float(v) for v in getattr(self, key))
            if len(value) != 3 or not np.all(np.isfinite(value)):
                raise ValueError(f"{key} must contain three finite values.")
            object.__setattr__(self, key, value)
        if np.linalg.norm(self.rN) == 0:
            raise ValueError("Target position must be nonzero.")
        if not np.isfinite(self.priority) or self.priority < 0:
            raise ValueError("Priority must be finite and nonnegative.")
        object.__setattr__(self, "priority", float(self.priority))
        from bsk_rl.sim.dyn.base import BasicDynamicsModel
        from bsk_rl.utils.functional import collect_default_args

        allowed = {
            "mass",
            "width",
            "depth",
            "height",
            "mu",
            "min_orbital_radius",
            "dragCoeff",
            "panelArea",
            "u_max",
        }
        defaults = collect_default_args(BasicDynamicsModel)
        parameters = tuple(
            (key, float(value)) for key, value in self.physical_parameters
        )
        if len({key for key, _ in parameters}) != len(parameters):
            raise ValueError("Duplicate physical parameter.")
        if any(
            key not in allowed or not np.isfinite(value) or value <= 0
            for key, value in parameters
        ):
            raise ValueError("Unsupported or invalid target physical parameter.")
        realized = {key: float(defaults[key]) for key in sorted(allowed)}
        realized.update(dict(parameters))
        object.__setattr__(self, "physical_parameters", tuple(realized.items()))

    def sat_args(self) -> dict[str, Any]:
        """Return deterministic target dynamics arguments for this epoch."""
        return dict(
            rN=list(self.rN),
            vN=list(self.vN),
            oe=None,
            sigma_init=list(self.sigma_init),
            omega_init=list(self.omega_init),
            wheelSpeeds=list(self.wheel_speeds_rpm),
            disturbance_vector=list(self.disturbance_torque_Nm),
            basePowerDraw=0.0,
            rwBasePower=0.0,
            batteryStorageCapacity=288000.0,
            storedCharge_Init=288000.0,
            **dict(self.physical_parameters),
        )


@dataclass(frozen=True)
class RSOPriorityEvent:
    """Priority changes applied at the first decision boundary at/after time_s.

    Changes occur after the preceding step's reward and before its observation.
    Events at zero occur before the initial observation. They do not clear pending
    images or cooldowns. Pairs are ``(target ID, new priority)``.
    """

    time_s: float
    priorities: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        """Freeze and validate an event."""
        if not np.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError("Event time must be finite and nonnegative.")
        pairs = tuple((key, float(value)) for key, value in self.priorities)
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("Duplicate priority-event ID.")
        if any(
            not isinstance(key, str) or not key or not np.isfinite(value) or value < 0
            for key, value in pairs
        ):
            raise ValueError("Invalid priority-event value.")
        object.__setattr__(self, "priorities", pairs)


@dataclass(frozen=True)
class RSOTargetCatalog:
    """Replayable target definitions, ordering, optional epoch, and priority events.

    If ``utc_init`` is omitted, target states refer to the environment's realized
    epoch on each reset. After reset,
    :attr:`~bsk_rl.scene.RSOTargets.catalog` contains that explicit epoch and can
    be saved for replay. An explicitly supplied catalog
    epoch must match the environment; states are never silently retimed.
    """

    targets: tuple[RSOTarget, ...]
    utc_init: str | None = None
    priority_events: tuple[RSOPriorityEvent, ...] = ()

    def __post_init__(self) -> None:
        """Validate identities and event references without consuming randomness."""
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "priority_events", tuple(self.priority_events))
        if not self.targets:
            raise ValueError("A catalog requires targets.")
        if self.utc_init is not None and (
            not isinstance(self.utc_init, str) or not self.utc_init
        ):
            raise ValueError("Catalog epoch must be a nonempty string or None.")
        ids = {target.id for target in self.targets}
        names = {target.satellite_name for target in self.targets}
        if len(ids) != len(self.targets) or len(names) != len(self.targets):
            raise ValueError("Target IDs and spacecraft names must be unique.")
        if any(
            key not in ids
            for event in self.priority_events
            for key, _ in event.priorities
        ):
            raise ValueError("Priority event references an unknown target.")
        if list(self.priority_events) != sorted(
            self.priority_events, key=lambda event: event.time_s
        ):
            raise ValueError("Priority events must be ordered by time.")

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned, JSON-compatible mission target definition."""
        return dict(
            schema_version=1,
            frame="Earth-centered J2000",
            position_unit="m",
            velocity_unit="m/s",
            **asdict(self),
        )

    def save(self, path: str | Path) -> None:
        """Save target definitions and their optional epoch without rounding.

        Save ``env.scenario.catalog`` after reset to include the realized epoch
        when the original definition omitted it.
        """
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
        )

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RSOTargetCatalog":
        """Validate and load the supported schema and units."""
        expected = dict(
            schema_version=1,
            frame="Earth-centered J2000",
            position_unit="m",
            velocity_unit="m/s",
        )
        if any(values.get(key) != value for key, value in expected.items()):
            raise ValueError("Unsupported catalog schema, frame, or units.")
        targets = []
        for saved in values["targets"]:
            target = dict(saved)
            # Read catalogs saved before the binding field was clarified.
            if "spacecraft_name" in target:
                legacy_name = target.pop("spacecraft_name")
                if target.get("satellite_name", legacy_name) != legacy_name:
                    raise ValueError(
                        "Conflicting satellite binding names in catalog: "
                        f"satellite_name={target['satellite_name']!r}, "
                        f"spacecraft_name={legacy_name!r}. Supply one binding name."
                    )
                target["satellite_name"] = legacy_name
            targets.append(RSOTarget(**target))
        return cls(
            tuple(targets),
            values.get("utc_init"),
            tuple(RSOPriorityEvent(**event) for event in values["priority_events"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RSOTargetCatalog":
        """Load a saved catalog without resampling any target."""
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class _RuntimeRSOTarget:
    """Episode-local binding; never serialized in the immutable catalog."""

    definition: RSOTarget
    target_spacecraft: Any
    priority: float

    @property
    def id(self) -> str:
        """Stable catalog ID."""
        return self.definition.id


class RSOTargets(Scenario):
    """Bind an immutable catalog to explicitly named imagers and target satellites.

    Other spacecraft are allowed, but do not become targets automatically.
    An omitted catalog epoch follows the environment on each reset; an explicit
    epoch must match it. Targets must provide ``rN``, ``vN``,
    ``oe``, and attitude arguments and an ``EclipseDynModel`` for illumination.
    """

    def __init__(self, catalog: RSOTargetCatalog, imager_names: list[str]) -> None:
        """Select participants by explicit identity, independently of list order."""
        super().__init__()
        self.catalog = catalog
        self.imager_names = tuple(imager_names)
        if not self.imager_names or len(set(self.imager_names)) != len(
            self.imager_names
        ):
            raise ValueError("Imager names must be nonempty and unique.")
        if any(not isinstance(name, str) or not name for name in self.imager_names):
            raise ValueError("Imager names must be nonempty strings.")
        if set(self.imager_names) & {
            target.satellite_name for target in catalog.targets
        }:
            raise ValueError("An imager cannot be its own RSO target.")
        self.reset_overwrite_previous()

    @property
    def catalog(self) -> RSOTargetCatalog:
        """Configured catalog before reset, or the epoch-resolved episode snapshot.

        The input definition remains immutable. Save this property after reset
        to retain the realized epoch without pinning subsequent episode resets.
        """
        if self._episode_catalog is not None:
            return self._episode_catalog
        return self._catalog_definition

    @catalog.setter
    def catalog(self, value: RSOTargetCatalog) -> None:
        """Set the reusable definition and discard the resolved episode snapshot."""
        self._catalog_definition = value
        self._episode_catalog = None

    def validate_satellite_names(self, satellites) -> None:
        """Reject ambiguous/missing original names before the environment renames."""
        names = [sat.name for sat in satellites]
        if len(names) != len(set(names)):
            raise ValueError("Satellite names must be unique.")
        required = set(self.imager_names) | {
            target.satellite_name for target in self.catalog.targets
        }
        if not required <= set(names):
            raise ValueError(
                f"Missing scene participants: {sorted(required - set(names))}"
            )

    def link_satellites(self, satellites) -> None:
        """Bind only the copied participants used by the actual environment."""
        self.validate_satellite_names(satellites)
        super().link_satellites(satellites)

    def reset_overwrite_previous(self) -> None:
        """Drop runtime bindings, centralized eligibility, and event history."""
        self._episode_catalog = None
        self.targets_by_id = {}
        self.imagers = []
        self.pending = {}
        self.cooldown_until = {}
        self.applied_events = []
        self.revision = 0

    def reset_pre_sim_init(self) -> None:
        """Bind this episode's objects and override only catalog target conditions."""
        epoch = getattr(self, "utc_init", None)
        if not isinstance(epoch, str) or not epoch:
            raise ValueError("RSOTargets requires the environment's realized epoch.")
        definition = self._catalog_definition
        if definition.utc_init is not None and epoch != definition.utc_init:
            raise ValueError("World epoch must match RSOTargetCatalog.utc_init.")
        self._episode_catalog = (
            replace(definition, utc_init=epoch)
            if definition.utc_init is None
            else definition
        )
        by_name = {sat.name: sat for sat in self.satellites}
        self.imagers = [by_name[name] for name in self.imager_names]
        for imager in self.imagers:
            if not hasattr(imager.dyn_type, "bind_rso_target") or not hasattr(
                imager.fsw_type, "action_image_rso"
            ):
                raise ValueError(f"Imager {imager.name!r} lacks RSO imaging models.")
        for definition in self.catalog.targets:
            satellite = by_name[definition.satellite_name]
            from bsk_rl.sim.dyn.base import EclipseDynModel

            if not issubclass(satellite.dyn_type, EclipseDynModel):
                raise ValueError(f"Target {satellite.name!r} lacks eclipse dynamics.")
            satellite.sat_args.update(definition.sat_args())
            self.targets_by_id[definition.id] = _RuntimeRSOTarget(
                definition, satellite, definition.priority
            )
        for imager in self.imagers:
            partitions = [
                self.partition_name(target.id) for target in self.catalog.targets
            ]
            if len(set(partitions)) != len(partitions):
                raise ValueError("RSO storage partition names collide.")
            imager.sat_args.update(
                bufferNames=partitions, transmitterNumBuffers=len(partitions)
            )
            imager.rso_scenario = self
            imager._rso_candidate_snapshot = None
        self.after_step(0.0)

    @staticmethod
    def partition_name(target_id: str) -> str:
        """Bounded simulator-safe partition name, independent of array position."""
        import hashlib

        return "rso_" + hashlib.sha256(target_id.encode()).hexdigest()[:32]

    def reset_during_sim_init(self) -> None:
        """Register each target separately in each imager's access model."""
        for imager in self.imagers:
            for target in self.targets_by_id.values():
                imager.dynamics.bind_rso_target(target)

    def is_eligible(self, target_id: str, time_s: float) -> bool:
        """Centralized pending visibility and quality-verified shared cooldown."""
        return not self.pending.get(target_id) and time_s >= self.cooldown_until.get(
            target_id, -np.inf
        )

    def after_step(self, sim_time: float) -> None:
        """Apply due priority events after reward, before the next observation."""
        for index, event in enumerate(self.catalog.priority_events):
            if (
                index in {entry[0] for entry in self.applied_events}
                or event.time_s > sim_time
            ):
                continue
            for target_id, priority in event.priorities:
                self.targets_by_id[target_id].priority = priority
            self.applied_events.append((index, float(sim_time)))
            self.revision += 1


__doc_title__ = "Space-to-Space RSO Targets"
__all__ = ["RSOTarget", "RSOTargetCatalog", "RSOPriorityEvent", "RSOTargets"]
