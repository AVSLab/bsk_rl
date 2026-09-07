"""Vizard assets and live overlays for multi-sensor RSO imaging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from Basilisk.architecture import sysModel

from bsk_rl.sats.roles import SpacecraftRole
from bsk_rl.utils.amos_vizard import PRIORITY_TIER_COLORS, _priority_terciles


# Match the straightforward pre-Walker Vizard setup: every simulated object is
# passed directly to Vizard and uses its native model/sprite transition.
SENSOR_MODEL_SCALE = 1.5
IMAGING_LINE_SLEW_COLOR = [245, 190, 45, 255]
IMAGING_LINE_HOLD_COLOR = [42, 190, 85, 255]


@dataclass(frozen=True)
class MultiAgentVizardAssets:
    """Static scene data needed to configure the multi-agent playback."""

    sprite_list: list[str | None]
    sensor_names: list[str]
    target_satellites: list[Any]
    target_names_by_tier: dict[str, list[str]]
    priority_tiers: dict[int, str]


def prepare_multiagent_vizard_assets(
    satellites: list[Any], viz_support: Any
) -> MultiAgentVizardAssets:
    """Assign sensor silhouettes and AMOS priority-tercile target colors."""
    sensor_names = [
        satellite.name
        for satellite in satellites
        if satellite.role is SpacecraftRole.SENSING_AGENT
    ]
    target_wrappers = [
        satellite.rso_target
        for satellite in satellites
        if satellite.role is SpacecraftRole.PASSIVE_TARGET
    ]
    priority_tiers = _priority_terciles(target_wrappers)
    target_names_by_tier = {tier: [] for tier in PRIORITY_TIER_COLORS}
    # One sprite entry per physical spacecraft, in scList order. Being an RL
    # agent is independent of how a spacecraft is rendered by Vizard.
    sprite_list: list[str | None] = []
    for satellite in satellites:
        if satellite.role is SpacecraftRole.SENSING_AGENT:
            sprite_list.append(viz_support.setSprite("bskSat"))
            continue
        target = satellite.rso_target
        tier = priority_tiers[int(target.id)]
        target_names_by_tier[tier].append(satellite.name)
        sprite_list.append(
            viz_support.setSprite("CIRCLE", color=list(PRIORITY_TIER_COLORS[tier]))
        )
    return MultiAgentVizardAssets(
        sprite_list=sprite_list,
        sensor_names=sensor_names,
        target_satellites=[
            satellite
            for satellite in satellites
            if satellite.role is SpacecraftRole.PASSIVE_TARGET
        ],
        target_names_by_tier=target_names_by_tier,
        priority_tiers=priority_tiers,
    )


def configure_multiagent_vizard_models(
    viz_instance: Any,
    viz_support: Any,
    assets: MultiAgentVizardAssets,
    *,
    sensor_model_scale: float = SENSOR_MODEL_SCALE,
) -> None:
    """Assign attitude-driven CAD models to the imaging spacecraft only."""
    viz_support.createCustomModel(
        viz_instance,
        simBodiesToModify=assets.sensor_names,
        modelPath="bskSat",
        scale=[float(sensor_model_scale)] * 3,
    )


class MultiAgentVizardMonitor(sysModel.SysModel):
    """Color every sensor-to-RSO line from the live 10-second hold state."""

    def __init__(
        self,
        sensors: list[Any],
        targets: list[Any],
        viz_instance: Any,
        viz_support: Any,
    ) -> None:
        """Bind every sensing agent to the shared Vizard line collection."""
        super().__init__()
        self.ModelTag = "MultiAgentVizardMonitor"
        self.sensors = list(sensors)
        self.targets = list(targets)
        self.viz_instance = viz_instance
        self.viz_support = viz_support
        self._last_signatures: dict[str, tuple[Any, ...]] = {}

    def _set_line_visible(self, line: Any, visible: bool) -> bool:
        lines = self.viz_support.targetLineList
        present = any(item is line for item in lines)
        if visible and not present:
            lines.append(line)
            return True
        if not visible and present:
            for index in range(len(lines) - 1, -1, -1):
                if lines[index] is line:
                    del lines[index]
            return True
        return False

    def update_imaging_lines(self) -> None:
        """Apply AMOS yellow-slew/green-valid-hold semantics to every sensor."""
        dirty = False
        for sensor in self.sensors:
            line = getattr(sensor.fsw, "_rso_line", None)
            if line is None:
                continue
            active = getattr(sensor, "_active_image_rso_action", None)
            target = getattr(active, "_hold_target", None)
            if active is None or target is None:
                dirty |= self._set_line_visible(line, False)
                self._last_signatures[sensor.name] = ("inactive",)
                continue
            try:
                valid, _ = active._pointing_constraints_ok(target)
            except Exception:
                valid = False
            color = IMAGING_LINE_HOLD_COLOR if valid else IMAGING_LINE_SLEW_COLOR
            marker_name = str(target.target_spacecraft.name)
            signature = (marker_name, tuple(color))
            if str(line.fromBodyName) != sensor.name:
                line.fromBodyName = sensor.name
                dirty = True
            if str(line.toBodyName) != marker_name:
                line.toBodyName = marker_name
                dirty = True
            if list(line.lineColor) != color:
                line.lineColor = color
                dirty = True
            dirty |= self._set_line_visible(line, True)
            if self._last_signatures.get(sensor.name) != signature:
                dirty = True
            self._last_signatures[sensor.name] = signature
        if dirty:
            self.viz_support.updateTargetLineList(self.viz_instance)

    def UpdateState(self, CurrentSimNanos: int) -> None:  # noqa: N802, ARG002
        """Refresh line colors immediately before each Vizard frame is serialized."""
        self.update_imaging_lines()


__all__ = [
    "IMAGING_LINE_HOLD_COLOR",
    "IMAGING_LINE_SLEW_COLOR",
    "MultiAgentVizardAssets",
    "MultiAgentVizardMonitor",
    "SENSOR_MODEL_SCALE",
    "configure_multiagent_vizard_models",
    "prepare_multiagent_vizard_assets",
]
