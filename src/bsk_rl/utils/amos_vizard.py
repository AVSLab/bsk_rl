"""AMOS 2026 Vizard overlays and dynamics-cadence catalog metrics.

The monitor in this module is deliberately attached to the scanner dynamics task.  It
therefore evaluates line of sight and illumination at every simulation cadence instead
of only when the RL agent reaches a decision epoch or requests an imaging action.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import cos, sin, sqrt
from typing import Any

import numpy as np
from Basilisk.architecture import messaging, sysModel
from Basilisk.utilities import macros


# Promotion colors are intentionally outside the ordinary light-to-dark blue
# priority scale.  HIO receives a medium purple and SHIO the deeper purple.
HIO_COLOR = "#8f63a7"
SHIO_COLOR = "#542788"
LIFECYCLE_COLORS = {
    # Cyan is deliberately separated from the light/medium/dark blue priority
    # fills so the lifecycle ring remains legible around every target sprite.
    "eligible": "#00b4d8",
    "cooldown": "#e45756",
    "buffered": "#2a9d5b",
}
PRIORITY_TIER_COLORS = {
    "lower": [158, 202, 225, 255],
    "middle": [66, 146, 198, 255],
    "upper": [8, 81, 156, 255],
}
DESAT_COLOR = "red"
# Match the Trusted Space STTR Phase 2 transceiver treatment: opaque purple,
# deliberately slow rings make an actual data transfer legible at high playback speed.
DOWNLINK_COLOR = "purple"
# At a full-catalog view, a 150-km shell is only a few screen pixels and is
# swallowed by the target sprite.  These visualization-only radii keep the
# lifecycle and promotion shells legible without changing any sensor geometry.
TARGET_STATUS_RING_RADIUS_M = 1_000_000.0
PROMOTION_HALO_RADIUS_M = 1_600_000.0


@dataclass
class AMOSVizardAssets:
    """Objects that must remain alive while Vizard records an episode."""

    ellipsoid_list: list[Any]
    generic_storage_list: list[Any]
    transceiver_list: list[Any]
    rw_effector_list: list[Any]
    thr_effector_list: list[Any]
    sprite_list: list[Any]
    bars: dict[str, Any] = field(default_factory=dict)
    target_outlines: dict[int, Any] = field(default_factory=dict)
    promotion_halos: dict[int, Any] = field(default_factory=dict)
    target_proxy_messages: dict[int, Any] = field(default_factory=dict)
    promotion_marker_messages: dict[int, Any] = field(default_factory=dict)
    promotion_marker_outlines: dict[int, Any] = field(default_factory=dict)
    promotion_marker_ellipsoids: dict[int, list[Any]] = field(default_factory=dict)
    promotion_marker_sprites: dict[int, str] = field(default_factory=dict)
    promotion_marker_names: dict[int, str] = field(default_factory=dict)
    priority_tiers: dict[int, str] = field(default_factory=dict)
    show_target_status_outlines: bool = False
    downlink_transceiver: Any = None
    desat_transceiver: Any = None
    ground_link_line: Any = None
    scanner_display_name: str = "SS1 Space Surveillance Inspector"
    dialog: Any = None


def _disown(obj: Any) -> None:
    """Let the Basilisk container own a SWIG object for the full simulation."""
    try:
        obj.this.disown()
    except Exception:
        pass


def _make_bar(viz_interface, viz_support, label, units, maximum, color="green"):
    bar = viz_interface.GenericStorage()
    bar.label = str(label)
    bar.units = str(units)
    bar.maxValue = max(1.0, float(maximum))
    bar.currentValue = 0.0
    bar.color = viz_interface.IntVector(viz_support.toRGBA255(color))
    _disown(bar)
    return bar


def ground_station_visibility_geometry(
    planet_radius_m: float,
    observer_radius_m: float,
    minimum_elevation_rad: float,
) -> tuple[float, float, float]:
    """Return full FOV, limiting slant range, and surface footprint arc.

    The slant range follows from the triangle formed by the planet center, ground
    station, and observer.  The location cone is centered on the station zenith,
    so its full edge-to-edge angle is twice the zenith angle at minimum elevation.
    """
    planet_radius_m = float(planet_radius_m)
    observer_radius_m = float(observer_radius_m)
    minimum_elevation_rad = float(minimum_elevation_rad)
    if planet_radius_m <= 0.0 or observer_radius_m <= planet_radius_m:
        raise ValueError("observer_radius_m must exceed a positive planet_radius_m")
    if not 0.0 <= minimum_elevation_rad < np.pi / 2.0:
        raise ValueError("minimum_elevation_rad must be in [0, pi/2)")

    slant_range_m = -planet_radius_m * sin(minimum_elevation_rad) + sqrt(
        observer_radius_m**2
        - planet_radius_m**2 * cos(minimum_elevation_rad) ** 2
    )
    cos_central_angle = (
        planet_radius_m**2 + observer_radius_m**2 - slant_range_m**2
    ) / (2.0 * planet_radius_m * observer_radius_m)
    central_angle_rad = float(np.arccos(np.clip(cos_central_angle, -1.0, 1.0)))
    full_fov_rad = np.pi - 2.0 * minimum_elevation_rad
    return full_fov_rad, slant_range_m, planet_radius_m * central_angle_rad


def _priority_terciles(targets: list[Any]) -> dict[int, str]:
    """Assign deterministic lower, middle, and upper initial-priority thirds."""
    ordered = sorted(
        targets,
        key=lambda target: (float(getattr(target, "priority", 0.0)), int(target.id)),
    )
    tiers: dict[int, str] = {}
    for tier, group in zip(("lower", "middle", "upper"), np.array_split(ordered, 3)):
        for target in group.tolist():
            tiers[int(target.id)] = tier
    return tiers


def _promotion_marker_style(target: Any, viz_support: Any) -> tuple[str, list[int]]:
    """Return the immutable Vizard sprite used once a promotion becomes active."""
    priority_kind = str(getattr(target, "priority_event_kind", ""))
    if priority_kind == "HIO":
        return "STAR", list(viz_support.toRGBA255(HIO_COLOR))
    if priority_kind == "SHIO":
        return "TRIANGLE", list(viz_support.toRGBA255(SHIO_COLOR))
    raise ValueError(f"No promotion marker exists for kind {priority_kind!r}")


def _spacecraft_initial_state(spacecraft: Any) -> Any:
    """Construct a state message from a spacecraft's configured initial state."""
    payload = messaging.SCStatesMsgPayload()
    try:
        hub = spacecraft.dynamics.scObject.hub
    except AttributeError:
        return payload
    position = np.asarray(hub.r_CN_NInit, dtype=float).reshape(-1)[:3].tolist()
    velocity = np.asarray(hub.v_CN_NInit, dtype=float).reshape(-1)[:3].tolist()
    payload.r_BN_N = position
    payload.r_CN_N = position
    payload.v_BN_N = velocity
    payload.v_CN_N = velocity
    payload.sigma_BN = (
        np.asarray(hub.sigma_BNInit, dtype=float).reshape(-1)[:3].tolist()
    )
    payload.omega_BN_B = (
        np.asarray(hub.omega_BN_BInit, dtype=float).reshape(-1)[:3].tolist()
    )
    return payload


def _hidden_spacecraft_state() -> Any:
    """Return an Earth-centered state used to hide an initialized Vizard proxy."""
    return messaging.SCStatesMsgPayload()


def prepare_amos_vizard_assets(
    satellites: list[Any],
    viz_interface: Any,
    viz_support: Any,
    *,
    show_text_hud: bool = True,
    show_image_bars: bool = True,
    show_target_status_outlines: bool = False,
    rw_display: str = "all",
) -> AMOSVizardAssets:
    """Create AMOS-specific Vizard objects before visualization is initialized."""
    n_spacecraft = len(satellites)
    scanner = satellites[0]
    n_targets = max(0, n_spacecraft - 1)
    rw_display = str(rw_display).lower()
    valid_rw_displays = {"all", "off"}
    if rw_display not in valid_rw_displays:
        raise ValueError(
            f"rw_display must be one of {sorted(valid_rw_displays)}, not {rw_display!r}"
        )
    ellipsoid_list: list[Any] = [None] * n_spacecraft
    target_outlines: dict[int, Any] = {}
    promotion_halos: dict[int, Any] = {}
    target_proxy_messages: dict[int, Any] = {}
    promotion_marker_messages: dict[int, Any] = {}
    promotion_marker_outlines: dict[int, Any] = {}
    promotion_marker_ellipsoids: dict[int, list[Any]] = {}
    promotion_marker_sprites: dict[int, str] = {}
    promotion_marker_names: dict[int, str] = {}
    target_wrappers = [
        getattr(target_satellite, "rso_target", None)
        for target_satellite in satellites[1:]
    ]
    target_wrappers = [target for target in target_wrappers if target is not None]
    priority_tiers = _priority_terciles(target_wrappers)

    # Preserve a spacecraft silhouette when the observer becomes too small for
    # Vizard to render its full model; the catalog targets remain colored circles.
    sprite_list: list[Any] = [viz_support.setSprite("bskSat")] + [None] * n_targets
    for sc_index, target_satellite in enumerate(satellites[1:], start=1):
        target = getattr(target_satellite, "rso_target", None)
        if target is None:
            continue
        tier = priority_tiers.get(int(target.id), "middle")
        # Event membership is intentionally not revealed at initialization.  Every
        # catalog target begins as an ordinary priority-tier circle; a separate
        # promotion proxy replaces selected targets only after the midpoint event.
        sprite_list[sc_index] = viz_support.setSprite(
            "CIRCLE", color=PRIORITY_TIER_COLORS[tier]
        )
        if show_target_status_outlines:
            outline = viz_interface.Ellipsoid()
            outline.position = [0.0, 0.0, 0.0]
            outline.semiMajorAxes = [TARGET_STATUS_RING_RADIUS_M] * 3
            outline.useBodyFrame = 1
            outline.showGridLines = -1
            outline.isOn = 1
            outline.color = viz_interface.IntVector(
                viz_support.toRGBA255(LIFECYCLE_COLORS["eligible"], alpha=0.90)
            )
            _disown(outline)

            ellipsoid_list[sc_index] = [outline]
            target_outlines[int(target.id)] = outline

        if str(getattr(target, "priority_event_kind", "")) in {"HIO", "SHIO"}:
            shape, promotion_color = _promotion_marker_style(target, viz_support)
            target_id = int(target.id)

            # Vizard reads sprite shape only when a spacecraft first appears.  Both
            # proxies are therefore initialized in frame 1: the blue target proxy at
            # the target, and its eventual purple replacement hidden inside Earth.
            # At promotion the monitor swaps their positions, preventing the blue
            # circle from being drawn over the star or triangle.
            target_proxy = messaging.SCStatesMsg().write(
                _spacecraft_initial_state(target_satellite), 0
            )
            marker_message = messaging.SCStatesMsg().write(
                _hidden_spacecraft_state(), 0
            )
            target_proxy_messages[target_id] = target_proxy
            promotion_marker_messages[int(target.id)] = marker_message
            promotion_marker_sprites[int(target.id)] = viz_support.setSprite(
                shape, color=promotion_color
            )
            promotion_marker_names[int(target.id)] = (
                f"Promoted {getattr(target, 'priority_event_kind', '')} "
                f"target {int(target.id)}"
            )

            marker_outline = None
            if show_target_status_outlines:
                marker_outline = viz_interface.Ellipsoid()
                marker_outline.position = [0.0, 0.0, 0.0]
                marker_outline.semiMajorAxes = [TARGET_STATUS_RING_RADIUS_M] * 3
                marker_outline.useBodyFrame = 1
                marker_outline.showGridLines = -1
                marker_outline.isOn = -1
                marker_outline.color = viz_interface.IntVector(
                    viz_support.toRGBA255(LIFECYCLE_COLORS["eligible"], alpha=0.90)
                )
                _disown(marker_outline)

            promotion_halo = viz_interface.Ellipsoid()
            promotion_halo.position = [0.0, 0.0, 0.0]
            promotion_halo.semiMajorAxes = [PROMOTION_HALO_RADIUS_M] * 3
            promotion_halo.useBodyFrame = 1
            promotion_halo.showGridLines = -1
            promotion_halo.isOn = -1
            promotion_halo.color = viz_interface.IntVector(
                viz_support.toRGBA255(
                    SHIO_COLOR if shape == "TRIANGLE" else HIO_COLOR,
                    alpha=0.70,
                )
            )
            _disown(promotion_halo)
            if marker_outline is not None:
                promotion_marker_outlines[target_id] = marker_outline
            promotion_halos[target_id] = promotion_halo
            promotion_marker_ellipsoids[target_id] = [promotion_halo]
            if marker_outline is not None:
                promotion_marker_ellipsoids[target_id].insert(0, marker_outline)

    downlink_transceiver = viz_interface.Transceiver()
    downlink_transceiver.r_SB_B = [0.0, 0.0, 1.38]
    downlink_transceiver.fieldOfView = float(160.0 * macros.D2R)
    downlink_transceiver.normalVector = [0.0, 0.0, 1.0]
    downlink_transceiver.color = viz_interface.IntVector(
        viz_support.toRGBA255(DOWNLINK_COLOR, alpha=1.0)
    )
    # The state is already conveyed by the animated transceiver and the ground-link
    # line.  A persistent label was visually misleading outside access windows.
    downlink_transceiver.label = ""
    downlink_transceiver.animationSpeed = 2
    downlink_transceiver.transceiverState = 0
    _disown(downlink_transceiver)

    desat_transceiver = viz_interface.Transceiver()
    desat_transceiver.r_SB_B = [0.0, 0.0, -1.38]
    desat_transceiver.fieldOfView = float(160.0 * macros.D2R)
    desat_transceiver.normalVector = [0.0, 0.0, -1.0]
    desat_transceiver.color = viz_interface.IntVector(
        viz_support.toRGBA255(DESAT_COLOR, alpha=1.0)
    )
    desat_transceiver.label = ""
    desat_transceiver.animationSpeed = 2
    desat_transceiver.transceiverState = 0
    _disown(desat_transceiver)
    transceiver_list: list[Any] = [
        [downlink_transceiver, desat_transceiver]
    ] + [None] * n_targets

    bars: dict[str, Any] = {}
    scanner_bars = [
        ("battery", "Battery", "%", 100.0, "green"),
        ("storage", "Onboard storage", "%", 100.0, "steelblue"),
        ("storage_illuminated", "Illuminated", "% capacity", 100.0, "#2a9d8f"),
        (
            "storage_nonilluminated",
            "Non-illuminated",
            "% capacity",
            100.0,
            "#e9a44c",
        ),
        ("eligible", "Catalog eligible", "targets", n_targets, "#4c9f70"),
        ("ever_observable", "Ever observable", "targets", n_targets, "#2a9d8f"),
    ]
    if show_image_bars:
        scanner_bars.extend(
            [
                ("imaged_1", "Imaged 1+", "targets", n_targets, "#2a9d5b"),
                ("imaged_2", "Imaged 2+", "targets", n_targets, "#23864d"),
                ("imaged_3", "Imaged 3+", "targets", n_targets, "#17653a"),
            ]
        )
    scanner_bars.extend(
        [
            ("priority_lower", "Priority: lower third", "", 1.0, PRIORITY_TIER_COLORS["lower"]),
            ("priority_middle", "Priority: middle third", "", 1.0, PRIORITY_TIER_COLORS["middle"]),
            ("priority_upper", "Priority: upper third", "", 1.0, PRIORITY_TIER_COLORS["upper"]),
        ]
    )
    for key, label, units, maximum, color in scanner_bars:
        bars[key] = _make_bar(viz_interface, viz_support, label, units, maximum, color)
    for key in ("priority_lower", "priority_middle", "priority_upper"):
        bars[key].currentValue = 1.0
    generic_storage_list: list[Any] = [list(bars.values())] + [None] * n_targets

    dialog = None
    if show_text_hud:
        dialog = viz_interface.VizEventDialog()
        dialog.eventHandlerID = "SPACE SURVEILLANCE"
        dialog.durationOfDisplay = 0.0
        dialog.displayString = "SPACE SURVEILLANCE\nInitializing..."
        _disown(dialog)

    # Only the agile observer's wheel state belongs in the operations HUD.  Logging
    # every passive target's wheels would needlessly inflate a 45,000-frame playback.
    # Vizard's native RW panel is all-or-nothing for a spacecraft.
    rw_effector = getattr(getattr(scanner, "dynamics", None), "rwStateEffector", None)
    rw_effector_list = [rw_effector if rw_display == "all" else None] + [
        None
    ] * n_targets
    thruster_set = getattr(getattr(scanner, "dynamics", None), "thrusterSet", None)
    thr_effector_list = ([[thruster_set]] if thruster_set is not None else [None]) + [
        None
    ] * n_targets
    return AMOSVizardAssets(
        ellipsoid_list=ellipsoid_list,
        generic_storage_list=generic_storage_list,
        transceiver_list=transceiver_list,
        rw_effector_list=rw_effector_list,
        thr_effector_list=thr_effector_list,
        sprite_list=sprite_list,
        bars=bars,
        target_outlines=target_outlines,
        promotion_halos=promotion_halos,
        target_proxy_messages=target_proxy_messages,
        promotion_marker_messages=promotion_marker_messages,
        promotion_marker_outlines=promotion_marker_outlines,
        promotion_marker_ellipsoids=promotion_marker_ellipsoids,
        promotion_marker_sprites=promotion_marker_sprites,
        promotion_marker_names=promotion_marker_names,
        priority_tiers=priority_tiers,
        show_target_status_outlines=bool(show_target_status_outlines),
        downlink_transceiver=downlink_transceiver,
        desat_transceiver=desat_transceiver,
        dialog=dialog,
    )


class AMOSVizardMonitor(sysModel.SysModel):
    """Update AMOS catalog metrics and Vizard overlays every dynamics step."""

    def __init__(
        self,
        simulator: Any,
        scanner: Any,
        target_satellites: list[Any],
        viz_instance: Any,
        viz_support: Any,
        assets: AMOSVizardAssets,
    ) -> None:
        """Bind the live monitor to one scanner and its Vizard asset objects."""
        super().__init__()
        self.ModelTag = "AMOSVizardMonitor"
        self.simulator = simulator
        self.scanner = scanner
        self.target_satellites = list(target_satellites)
        self.viz_instance = viz_instance
        self.viz_support = viz_support
        self.assets = assets
        self.ever_los_ids: set[int] = set()
        self.ever_observable_ids: set[int] = set()
        self.latest_metrics: dict[str, Any] = {}
        self._last_line_signature = None
        self._last_promotion_halo_kind: dict[int, str] = {}
        self._last_target_outline_state: dict[int, str] = {}
        self._storage_quality_state: dict[str, dict[str, float]] = {}

    @property
    def targets(self) -> list[Any]:
        """Return the catalog wrappers associated with the simulated target spacecraft."""
        known = getattr(getattr(self.scanner, "data_store", None), "data", None)
        known_targets = list(getattr(known, "known", []))
        if known_targets:
            return known_targets
        return [
            target
            for sc in self.target_satellites
            if (target := getattr(sc, "rso_target", None)) is not None
        ]

    def _target_state(self, target: Any, sim_time: float) -> tuple[bool, bool, bool]:
        try:
            access = self.scanner.dynamics.targetLocation.accessOutMsgs[int(target.id)].read()
            los = bool(access.hasAccess)
        except Exception:
            los = False
        try:
            eclipse_index = target.target_spacecraft.dynamics.eclipse_index
            shadow = self.scanner.dynamics.world.eclipseObject.eclipseOutMsgs[
                eclipse_index
            ].read().shadowFactor
            threshold = float(self.scanner.dynamics.eclipse_threshold_for_imaging)
            illuminated = float(shadow) >= threshold
        except Exception:
            illuminated = False
        data_obj = getattr(getattr(self.scanner, "data_store", None), "data", None)
        if data_obj is not None and hasattr(data_obj, "is_target_eligible"):
            eligible = bool(data_obj.is_target_eligible(target, sim_time))
        else:
            eligible = True
        return los, illuminated, eligible

    @staticmethod
    def _record_key(record: dict[str, Any]) -> str:
        return str(
            record.get("record_id")
            or "|".join(
                str(record.get(key))
                for key in ("target_id", "capture_time", "storage_index")
            )
        )

    def _capture_counts(self) -> Counter:
        data_obj = getattr(getattr(self.scanner, "data_store", None), "data", None)
        records: list[dict[str, Any]] = []
        if data_obj is not None:
            records.extend(getattr(data_obj, "verified_useful_records", []))
            for pending in getattr(data_obj, "pending_image_records_by_id", {}).values():
                records.extend(pending)
        for staged in getattr(
            self.scanner, "_rso_pending_capture_metadata_by_name", {}
        ).values():
            records.extend(staged)

        counts: Counter = Counter()
        seen: set[str] = set()
        for record in records:
            key = self._record_key(record)
            if key in seen or not bool(record.get("quality_passed", False)):
                continue
            seen.add(key)
            target_id = record.get("target_id")
            if target_id is not None:
                counts[int(target_id)] += 1

        # Retain compatibility with episodes produced before per-capture records were
        # introduced.  This fallback only supplies the first image of a target.
        if data_obj is not None:
            for target in getattr(data_obj, "imaged", []):
                counts[int(target.id)] = max(1, counts[int(target.id)])
        return counts

    def _storage_split_bits(self) -> tuple[float, float, float]:
        """Split physical onboard storage by illumination at data creation.

        The target-specific Basilisk storage partitions grow during the dynamics
        propagation.  Completed image records are staged only after the hold gate
        succeeds, so using those records here lags the physical increment and used to
        misclassify every not-yet-verified increment as non-illuminated.  Instead, the
        split below samples the corresponding target's shadow factor on the exact
        dynamics tick when its partition grows.
        """
        try:
            msg = self.scanner.dynamics.storageUnit.storageUnitDataOutMsg.read()
        except Exception:
            return 0.0, 0.0, 0.0
        names = [str(name) for name in msg.storedDataName]
        levels = [max(0.0, float(level)) for level in msg.storedData]
        target_by_name: dict[str, Any] = {}
        for target in self.targets:
            target_by_name[str(getattr(target, "name", ""))] = target
            target_spacecraft = getattr(target, "target_spacecraft", None)
            if target_spacecraft is not None:
                target_by_name[str(getattr(target_spacecraft, "name", ""))] = target

        level_by_name = dict(zip(names, levels))
        for name in set(self._storage_quality_state) | set(level_by_name):
            level = float(level_by_name.get(name, 0.0))
            state = self._storage_quality_state.setdefault(
                name, {"level": 0.0, "illuminated": 0.0, "nonilluminated": 0.0}
            )
            previous_level = float(state["level"])
            delta = level - previous_level

            if delta > 1e-6:
                target = target_by_name.get(name)
                illuminated_now = False
                if target is not None:
                    try:
                        eclipse_index = target.target_spacecraft.dynamics.eclipse_index
                        shadow_factor = float(
                            self.scanner.dynamics.world.eclipseObject.eclipseOutMsgs[
                                eclipse_index
                            ]
                            .read()
                            .shadowFactor
                        )
                        threshold = float(
                            self.scanner.dynamics.eclipse_threshold_for_imaging
                        )
                        illuminated_now = shadow_factor >= threshold
                    except Exception:
                        illuminated_now = False
                category = "illuminated" if illuminated_now else "nonilluminated"
                state[category] += delta
            elif delta < -1e-6:
                # A category may only decrease when the physical partition decreases.
                # Proportional removal is stable even when record metadata is moved from
                # the staged queue to the verified archive at a decision boundary.
                fraction_remaining = level / previous_level if previous_level > 0.0 else 0.0
                state["illuminated"] *= fraction_remaining
                state["nonilluminated"] *= fraction_remaining

            state["level"] = level
            classified_level = state["illuminated"] + state["nonilluminated"]
            if classified_level > 0.0 and abs(classified_level - level) > 1e-6:
                scale = level / classified_level
                state["illuminated"] *= scale
                state["nonilluminated"] *= scale

        total = float(sum(levels))
        illuminated_bits = float(
            sum(state["illuminated"] for state in self._storage_quality_state.values())
        )
        nonilluminated_bits = float(
            sum(
                state["nonilluminated"]
                for state in self._storage_quality_state.values()
            )
        )
        return total, illuminated_bits, nonilluminated_bits

    def _target_lifecycle_state(self, target: Any, sim_time: float) -> str:
        """Return eligible, cooldown, or buffered for the target outline."""
        data_obj = getattr(getattr(self.scanner, "data_store", None), "data", None)
        if data_obj is not None:
            try:
                lifecycle = str(data_obj.target_lifecycle_state(target, sim_time))
                if lifecycle == "pending_verification":
                    return "buffered"
                if lifecycle == "cooldown":
                    return "cooldown"
            except Exception:
                pass
        staged_by_name = getattr(
            self.scanner, "_rso_pending_capture_metadata_by_name", {}
        )
        if staged_by_name.get(str(target.name)):
            return "buffered"
        return "eligible"

    def _update_target_visuals(self, sim_time: float) -> None:
        hidden_state = _hidden_spacecraft_state()
        message_time = macros.sec2nano(sim_time)
        for target in self.targets:
            target_id = int(target.id)
            priority_active = bool(getattr(target, "priority_event_active", False))
            priority_kind = str(getattr(target, "priority_event_kind", ""))

            marker_message = self.assets.promotion_marker_messages.get(target_id)
            target_proxy = self.assets.target_proxy_messages.get(target_id)
            if marker_message is not None and target_proxy is not None:
                try:
                    target_state = (
                        target.target_spacecraft.dynamics.scObject.scStateOutMsg.read()
                    )
                    if priority_active:
                        target_proxy.write(hidden_state, message_time)
                        marker_message.write(target_state, message_time)
                    else:
                        target_proxy.write(target_state, message_time)
                        marker_message.write(hidden_state, message_time)
                except Exception:
                    pass

            outline = self.assets.target_outlines.get(target_id)
            marker_outline = self.assets.promotion_marker_outlines.get(target_id)
            if outline is not None or marker_outline is not None:
                lifecycle_state = self._target_lifecycle_state(target, sim_time)
                if (
                    outline is not None
                    and self._last_target_outline_state.get(target_id)
                    != lifecycle_state
                ):
                    outline.isOn = -1 if priority_active else 1
                    outline.color = type(outline.color)(
                        self.viz_support.toRGBA255(
                            LIFECYCLE_COLORS[lifecycle_state], alpha=0.90
                        )
                    )
                    self._last_target_outline_state[target_id] = lifecycle_state
                elif outline is not None:
                    outline.isOn = -1 if priority_active else 1

                if marker_outline is not None:
                    marker_outline.isOn = 1 if priority_active else -1
                    if priority_active:
                        marker_outline.color = type(marker_outline.color)(
                            self.viz_support.toRGBA255(
                                LIFECYCLE_COLORS[lifecycle_state], alpha=0.90
                            )
                        )

            promotion_halo = self.assets.promotion_halos.get(target_id)
            if promotion_halo is not None:
                promotion_halo.isOn = 1 if priority_active else -1
                if (
                    priority_active
                    and self._last_promotion_halo_kind.get(target_id) != priority_kind
                ):
                    is_shio = priority_kind == "SHIO"
                    radius = (
                        1.18 * PROMOTION_HALO_RADIUS_M
                        if is_shio
                        else PROMOTION_HALO_RADIUS_M
                    )
                    promotion_halo.semiMajorAxes = [radius, radius, radius]
                    promotion_halo.color = type(promotion_halo.color)(
                        self.viz_support.toRGBA255(
                            SHIO_COLOR if is_shio else HIO_COLOR, alpha=0.70
                        )
                    )
                    self._last_promotion_halo_kind[target_id] = priority_kind

    def _active_ground_station(self) -> str | None:
        """Return the receiving station only while data is actually leaving storage."""
        commanded = bool(
            getattr(self.scanner.dynamics.transmitterPowerSink, "powerStatus", 0)
        )
        if not commanded:
            return None
        try:
            baud_rate = float(
                self.scanner.dynamics.transmitter.nodeDataOutMsg.read().baudRate
            )
        except Exception:
            return None
        if baud_rate >= -1e-12:
            return None
        for ground_station in getattr(self.scanner.dynamics.world, "groundStations", []):
            try:
                if bool(ground_station.accessOutMsgs[-1].read().hasAccess):
                    model_tag = str(ground_station.ModelTag)
                    return (
                        model_tag[len("GroundStation") :]
                        if model_tag.startswith("GroundStation")
                        else model_tag
                    )
            except Exception:
                continue
        return None

    def _set_line_visible(self, line: Any, visible: bool) -> None:
        lines = self.viz_support.targetLineList
        present = any(item is line for item in lines)
        if visible and not present:
            lines.append(line)
        elif not visible and present:
            for index in range(len(lines) - 1, -1, -1):
                if lines[index] is line:
                    del lines[index]
        else:
            return
        self.viz_support.updateTargetLineList(self.viz_instance)

    def _update_ground_link_line(self, station_name: str | None) -> None:
        line = self.assets.ground_link_line
        if station_name is not None and line is None:
            self.viz_support.createTargetLine(
                self.viz_instance,
                fromBodyName=self.assets.scanner_display_name,
                toBodyName=station_name,
                lineColor=[42, 190, 85, 255],
            )
            line = self.viz_support.targetLineList[-1]
            self.assets.ground_link_line = line
        if line is None:
            return
        target_changed = False
        if station_name is not None:
            target_changed = str(line.toBodyName) != station_name
            line.toBodyName = station_name
            line.lineColor = [42, 190, 85, 255]
        was_present = any(item is line for item in self.viz_support.targetLineList)
        self._set_line_visible(line, station_name is not None)
        if target_changed and was_present:
            self.viz_support.updateTargetLineList(self.viz_instance)

    def _update_pointing_line(self) -> tuple[str, float]:
        fsw = self.scanner.fsw
        line = getattr(fsw, "_rso_line", None)
        active = getattr(self.scanner, "_active_image_rso_action", None)
        state = "inactive"
        hold_fraction = 0.0
        color = None
        if line is not None and active is not None and active._hold_target is not None:
            try:
                valid, _ = active._pointing_constraints_ok(active._hold_target)
            except Exception:
                valid = False
            state = "holding" if valid else "slewing/aligning"
            color = [42, 190, 85, 255] if valid else [245, 190, 45, 255]
            required = max(0.0, float(active.min_pointing_hold_s))
            hold_fraction = (
                min(1.0, float(active._hold_valid_time_s) / required)
                if required > 0.0
                else float(valid)
            )
        signature = (state, tuple(color) if color is not None else None)
        if line is not None:
            line.fromBodyName = self.assets.scanner_display_name
            if color is not None:
                line.lineColor = color
            self._set_line_visible(line, color is not None)
            if signature != self._last_line_signature and color is not None:
                self.viz_support.updateTargetLineList(self.viz_instance)
            self._last_line_signature = signature
        return state, hold_fraction

    def _current_action(self) -> str:
        label = str(getattr(self.scanner, "_current_action_label", "")).strip()
        if label:
            return label
        if bool(getattr(self.scanner.dynamics.thrusterPowerSink, "powerStatus", 0)):
            return "Desat"
        if bool(getattr(self.scanner.dynamics.transmitterPowerSink, "powerStatus", 0)):
            return "Downlink"
        if getattr(self.scanner, "_active_image_rso_action", None) is not None:
            return "Imaging"
        return "Charge"

    def _set_bar(self, key: str, value: float) -> None:
        bar = self.assets.bars.get(key)
        if bar is not None:
            bar.currentValue = float(np.clip(value, 0.0, float(bar.maxValue)))

    def _update_action_rings(
        self, *, downlink_active: bool, current_action: str
    ) -> None:
        """Show purple rings for data transfer and red rings for desaturation."""
        self.assets.downlink_transceiver.transceiverState = (
            1 if downlink_active else 0
        )
        self.assets.desat_transceiver.transceiverState = (
            1 if current_action == "Desat" else 0
        )

    def UpdateState(self, CurrentSimNanos: int) -> None:  # noqa: N802
        """Sample physics state and refresh every AMOS overlay once per dynamics tick."""
        sim_time = float(CurrentSimNanos) * macros.NANO2SEC
        targets = self.targets
        states = {int(target.id): self._target_state(target, sim_time) for target in targets}
        for target_id, (los, illuminated, _) in states.items():
            if los:
                self.ever_los_ids.add(target_id)
            if los and illuminated:
                self.ever_observable_ids.add(target_id)

        counts = self._capture_counts()
        eligible_count = sum(eligible for _, _, eligible in states.values())
        imageable_count = sum(
            los and illuminated and eligible
            for los, illuminated, eligible in states.values()
        )
        imaged_1 = sum(count >= 1 for count in counts.values())
        imaged_2 = sum(count >= 2 for count in counts.values())
        imaged_3 = sum(count >= 3 for count in counts.values())
        total_bits, illuminated_bits, nonilluminated_bits = self._storage_split_bits()
        capacity = max(1.0, float(self.scanner.dynamics.storageUnit.storageCapacity))
        battery_pct = 100.0 * float(self.scanner.dynamics.battery_charge_fraction)
        storage_pct = 100.0 * total_bits / capacity

        self._set_bar("battery", round(battery_pct, 1))
        self._set_bar("storage", storage_pct)
        self._set_bar("storage_illuminated", 100.0 * illuminated_bits / capacity)
        self._set_bar("storage_nonilluminated", 100.0 * nonilluminated_bits / capacity)
        self._set_bar("eligible", eligible_count)
        self._set_bar("ever_observable", len(self.ever_observable_ids))
        self._set_bar("imaged_1", imaged_1)
        self._set_bar("imaged_2", imaged_2)
        self._set_bar("imaged_3", imaged_3)

        active_station = self._active_ground_station()
        downlink_active = active_station is not None
        self._update_ground_link_line(active_station)
        pointing_state, hold_fraction = self._update_pointing_line()
        current_action = self._current_action()
        self._update_action_rings(
            downlink_active=downlink_active,
            current_action=current_action,
        )
        self._update_target_visuals(sim_time)
        wheel_speed_pct = 100.0 * float(
            np.max(np.abs(np.asarray(self.scanner.dynamics.wheel_speeds_fraction)))
        )
        promoted_hio = sum(
            bool(getattr(target, "priority_event_active", False))
            and str(getattr(target, "priority_event_kind", "")) == "HIO"
            for target in targets
        )
        promoted_shio = sum(
            bool(getattr(target, "priority_event_active", False))
            and str(getattr(target, "priority_event_kind", "")) == "SHIO"
            for target in targets
        )

        n_targets = max(1, len(targets))
        self.latest_metrics = {
            "sim_time_s": sim_time,
            "catalog_eligible": eligible_count,
            "imageable_now": imageable_count,
            "ever_los": len(self.ever_los_ids),
            "ever_observable": len(self.ever_observable_ids),
            "imaged_at_least_once": imaged_1,
            "imaged_at_least_twice": imaged_2,
            "imaged_at_least_three_times": imaged_3,
            "battery_percent": battery_pct,
            "storage_percent": storage_pct,
            "downlink_active": downlink_active,
            "active_ground_station": active_station,
            "current_action": current_action,
            "pointing_state": pointing_state,
            "hold_fraction": hold_fraction,
        }

        if self.assets.dialog is not None:
            status_summary = (
                "Status halo: cyan eligible; red cooldown; green onboard"
                if self.assets.show_target_status_outlines
                else "Status halos: omitted for playback performance"
            )
            self.assets.dialog.displayString = (
                "SPACE SURVEILLANCE\n"
                f"Current action: {current_action}; RW max speed: {wheel_speed_pct:.1f}%\n"
                f"Catalog eligible: {eligible_count}/{len(targets)} "
                f"({100.0 * eligible_count / n_targets:.1f}%)\n"
                f"Ever LOS: {len(self.ever_los_ids)}/{len(targets)}; "
                f"ever observable: {len(self.ever_observable_ids)}/{len(targets)}\n"
                f"Imaged 1+ / 2+ / 3+: {imaged_1} / {imaged_2} / {imaged_3}\n"
                f"Battery: {battery_pct:.1f}%; storage: {storage_pct:.1f}%\n"
                f"Stored imagery: illuminated {100.0 * illuminated_bits / capacity:.1f}%; "
                f"non-illuminated {100.0 * nonilluminated_bits / capacity:.1f}%\n"
                "Priority fill: see light / medium / dark blue bars\n"
                f"{status_summary}\n"
                f"Promoted: HIO stars {promoted_hio}; SHIO triangles {promoted_shio}\n"
                f"Imaging: {pointing_state}; hold: {100.0 * hold_fraction:.0f}%\n"
                f"Ground link: {active_station if downlink_active else 'inactive'}"
            )
            self.viz_instance.vizEventDialogs.append(self.assets.dialog)


__all__ = [
    "AMOSVizardAssets",
    "AMOSVizardMonitor",
    "DOWNLINK_COLOR",
    "DESAT_COLOR",
    "HIO_COLOR",
    "PRIORITY_TIER_COLORS",
    "SHIO_COLOR",
    "ground_station_visibility_geometry",
    "prepare_amos_vizard_assets",
]
