"""Extended Basilisk SimBaseClass for GeneralSatelliteTasking environments."""

import logging
import os
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any
import numpy as np

from Basilisk.utilities import macros as mc
from Basilisk.utilities import SimulationBaseClass
from Basilisk.simulation import simpleInstrument, simpleStorageUnit, partitionedStorageUnit, spaceToGroundTransmitter
from Basilisk.simulation import groundLocation
from Basilisk.utilities import vizSupport
from Basilisk.utilities import unitTestSupport

from Basilisk.simulation import spacecraft
from Basilisk.utilities import macros
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import simIncludeGravBody

from bsk_rl.utils import vizard
from bsk_rl.utils.profiling import profile_section

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite
    from bsk_rl.sim.world import WorldModel

logger = logging.getLogger(__name__)

def strip_prefix(s, prefix="GroundStation"):
    return s[len(prefix):] if s.startswith(prefix) else s

class Simulator(SimulationBaseClass.SimBaseClass):
    """Basilisk simulator for GeneralSatelliteTasking environments."""

    def __init__(
        self,
        satellites: list["Satellite"],
        world_type: type["WorldModel"],
        world_args: dict[str, Any],
        sim_rate: float = 1.0,
        max_step_duration: float = 600.0,
        time_limit: float = float("inf"),
    ) -> None:
        """Basilisk simulator for satellite tasking environments.

        The simulator is reconstructed each time the environment :class:`~bsk_rl.GeneralSatelliteTasking.reset`
        is called, generating a fresh Basilisk simulation.

        Args:
            satellites: Satellites to be simulated
            world_type: Type of world model to be constructed
            world_args: Arguments for world model construction
            sim_rate: [s] Rate for model simulation.
            max_step_duration: [s] Maximum time to propagate sim at a step.
            time_limit: [s] Latest time simulation will propagate to.
        """
        super().__init__()
        self.sim_rate = sim_rate
        self.satellites = satellites
        self.max_step_duration = max_step_duration
        self.time_limit = time_limit
        self.logger = logger
        self.profiler = None

        self.world: WorldModel

        self._set_world(world_type, world_args)

        self.fsw_list = {}
        self.dynamics_list = {}

        for satellite in self.satellites:
            satellite.set_simulator(self)
            self.dynamics_list[satellite.name] = satellite.set_dynamics(self.sim_rate)
            self.fsw_list[satellite.name] = satellite.set_fsw(self.sim_rate)

    def finish_init(self) -> None:
        """Finish simulator initialization."""
        self.set_vizard_epoch()
        self.InitializeSimulation()
        self.ConfigureStopTime(0)
        self.ExecuteSimulation()

    @property
    def sim_time_ns(self) -> int:
        """Simulation time in ns, tied to SimBase integrator."""
        return self.TotalSim.CurrentNanos

    @property
    def sim_time(self) -> float:
        """Simulation time in seconds, tied to SimBase integrator."""
        return self.sim_time_ns * mc.NANO2SEC

    @vizard.visualize
    def setup_vizard(
        self,
        vizard_rate=None,
        amos_hud=False,
        amos_hud_text=True,
        amos_hud_metric_bars=True,
        amos_hud_image_bars=True,
        amos_target_status_outlines=False,
        amos_rw_display="all",
        vizSupport=None,
        **vizard_settings,
    ):
        """Setup Vizard for visualization."""
        save_path = Path(vizard.VIZARD_PATH)
        if not save_path.exists():
            os.makedirs(save_path, exist_ok=True)

        viz_proc_name = "VizProcess"
        viz_proc = self.CreateNewProcess(viz_proc_name, priority=400)

        # Define process name, task name and task time-step
        viz_task_name = "viz_task_name"
        if vizard_rate is None:
            vizard_rate = self.sim_rate
        viz_proc.addTask(self.CreateNewTask(viz_task_name, mc.sec2nano(vizard_rate)))

        customizers = ["spriteList", "genericSensorList"]
        list_data = {}
        for customizer in customizers:
            list_data[customizer] = [
                sat.vizard_data.get(customizer, None) for sat in self.satellites
            ]

        amos_assets = None
        if amos_hud:
            from Basilisk.simulation import vizInterface

            from bsk_rl.utils.amos_vizard import prepare_amos_vizard_assets

            # Basilisk clears static point lines here but not its module-level live
            # target-line cache. Clear it so another episode in the same process
            # cannot inherit a stale imaging or ground-link line.
            del vizSupport.targetLineList[:]

            amos_assets = prepare_amos_vizard_assets(
                self.satellites,
                vizInterface,
                vizSupport,
                show_text_hud=bool(amos_hud_text),
                show_live_metric_bars=bool(amos_hud_metric_bars),
                show_image_bars=bool(amos_hud_image_bars),
                show_target_status_outlines=bool(amos_target_status_outlines),
                rw_display=str(amos_rw_display),
            )
            list_data.update(
                ellipsoidList=amos_assets.ellipsoid_list,
                genericStorageList=amos_assets.generic_storage_list,
                transceiverList=amos_assets.transceiver_list,
                rwEffectorList=amos_assets.rw_effector_list,
                thrEffectorList=amos_assets.thr_effector_list,
                spriteList=amos_assets.sprite_list,
            )
        self.vizInstance = vizSupport.enableUnityVisualization(
            self,
            viz_task_name,
            scList=[sat.dynamics.scObject for sat in self.satellites],
            **list_data,
            saveFile=save_path / f"viz_{time()}",
        )
        viz = self.vizInstance
        if amos_assets is not None:
            # Vizard derives the generic storage-panel title from the visualized
            # spacecraft name.  Change only the visualization label; the simulation,
            # policy, data products, and Basilisk model continue to use SS1.
            self.vizMessenger.scData[0].spacecraftName = (
                amos_assets.scanner_display_name
            )
            # Promotion candidates use a blue proxy that can be moved inside Earth
            # when its immutable purple star/triangle proxy becomes active.
            target_sc_index = {
                int(sat.rso_target.id): index
                for index, sat in enumerate(self.satellites[1:], start=1)
                if getattr(sat, "rso_target", None) is not None
            }
            for target_id, proxy_message in amos_assets.target_proxy_messages.items():
                self.vizMessenger.scData[target_sc_index[target_id]].scStateInMsg.subscribeTo(
                    proxy_message
                )
            # Vizard treats a spacecraft sprite as initialization-only.  Add one
            # immutable purple sprite for each eventual HIO/SHIO.  Its state message
            # starts at Earth's center so Vizard initializes the sprite in frame 1;
            # the monitor moves it to the target only after the midpoint event.
            for target_id, marker_message in sorted(
                amos_assets.promotion_marker_messages.items()
            ):
                marker_data = vizInterface.VizSpacecraftData()
                marker_data.spacecraftName = amos_assets.promotion_marker_names[
                    target_id
                ]
                marker_data.scStateInMsg.subscribeTo(marker_message)
                marker_data.spacecraftSprite = amos_assets.promotion_marker_sprites[
                    target_id
                ]
                # Transparent black is rendered as a dark orbit by Vizard 2.3.1b6.
                # Match the ordinary catalog's opaque-white orbit line instead.
                marker_data.oscOrbitLineColor = vizInterface.IntVector(
                    [255, 255, 255, 255]
                )
                marker_data.trueTrajectoryLineColor = vizInterface.IntVector(
                    [255, 255, 255, 255]
                )
                marker_data.ellipsoidList = vizInterface.EllipsoidVector(
                    amos_assets.promotion_marker_ellipsoids[target_id]
                )
                self.vizMessenger.scData.append(marker_data)

        scanner_radius_m = float(
            np.linalg.norm(self.satellites[0].dynamics.scObject.hub.r_CN_NInit)
        )
        for i in range(len(self.world.groundStations)):
            station = self.world.groundStations[i]
            station_radius_m = float(np.linalg.norm(station.r_LP_P_Init))
            if amos_assets is not None:
                from bsk_rl.utils.amos_vizard import ground_station_visibility_geometry

                station_fov, station_range, _ = ground_station_visibility_geometry(
                    station_radius_m,
                    scanner_radius_m,
                    station.minimumElevation,
                )
            else:
                station_fov = np.radians(160.0)
                station_range = 1000.0 * 1000.0
            vizSupport.addLocation(
                viz,
                stationName=strip_prefix(station.ModelTag),
                parentBodyName=self.world.planet.displayName,
                r_GP_P=unitTestSupport.EigenVector3d2list(station.r_LP_P_Init),
                fieldOfView=station_fov,
                color="green",
                range=station_range,
            )
        if amos_assets is not None:
            # Replace only the observer's default CAD instance so its local-view model
            # is visibly larger while retaining its live body attitude.  Vizard's
            # showSpacecraftAsSprites switch is global: disabling it would replace all
            # 200 target sprites with CAD models, so a per-observer scale is the safe
            # way to delay the observer's distant-view sprite transition.
            vizSupport.createCustomModel(
                viz,
                simBodiesToModify=[amos_assets.scanner_display_name],
                modelPath="bskSat",
                scale=[1.5, 1.5, 1.5],
            )
            viz.settings.spacecraftSizeMultiplier = 2.5
            # A 200-target playback is dominated by Vizard rendering hundreds of
            # orbit histories. Start with these optional histories hidden; they can
            # still be re-enabled from the View menu during playback.
            viz.settings.orbitLinesOn = -1
            viz.settings.trueTrajectoryLinesOn = -1
            viz.settings.showOsculatingGroundTrackLines = -1
            viz.settings.showTruePathGroundTrackLines = -1
            # Keep live imaging and ground-contact lines legible in planet view.
            viz.settings.linesAndFramesLineWidth = 3.0
            viz.settings.useLineRenderersForTargetLinesAndFrames = 1
        else:
            viz.settings.spacecraftSizeMultiplier = 1.5
        # Vizard uses 0 for "use application default," not false.  Explicitly use
        # -1 so locations never draw automatic links to every spacecraft in range;
        # the AMOS monitor owns the single SS1-to-active-station line instead.
        viz.settings.showLocationCommLines = -1 if amos_assets is not None else 1
        viz.settings.showLocationCones = 1
        viz.settings.showLocationLabels = 1
        for key, value in vizard_settings.items():
            setattr(self.vizInstance.settings, key, value)

        if amos_assets is not None:
            from bsk_rl.utils.amos_vizard import AMOSVizardMonitor

            scanner = self.satellites[0]
            self.amos_vizard_monitor = AMOSVizardMonitor(
                simulator=self,
                scanner=scanner,
                target_satellites=self.satellites[1:],
                viz_instance=self.vizInstance,
                viz_support=vizSupport,
                assets=amos_assets,
            )
            self.AddModelToTask(
                viz_task_name,
                self.amos_vizard_monitor,
                # Run immediately before the lower-priority Vizard serializer on the
                # same task, and only at the requested playback sampling cadence.
                ModelPriority=1000,
            )
            vizSupport.setInstrumentGuiSetting(
                self.vizInstance,
                spacecraftName=amos_assets.scanner_display_name,
                showTransceiverLabels=-1,
                showTransceiverFrustum=1,
                # Restore Vizard's native expandable panel.  Vizard hard-codes its
                # title as "<spacecraft> Storage" and exposes no title override.
                showGenericStoragePanel=(
                    1 if amos_assets.show_live_metric_bars else -1
                ),
            )
            show_native_rw = amos_assets.rw_effector_list[0] is not None
            vizSupport.setActuatorGuiSetting(
                self.vizInstance,
                spacecraftName=amos_assets.scanner_display_name,
                viewRWPanel=1 if show_native_rw else -1,
                viewRWHUD=1 if show_native_rw else -1,
                viewThrusterPanel=-1,
                viewThrusterHUD=1,
                showThrusterLabels=-1,
                showRWLabels=1 if show_native_rw else -1,
            )
        vizard.VIZINSTANCE = self.vizInstance


    @vizard.visualize
    def set_vizard_epoch(self, vizInstance=None):
        """Set the Vizard epoch."""
        vizInstance.epochInMsg.subscribeTo(self.world.gravFactory.epochMsg)

    def _set_world(
        self, world_type: type["WorldModel"], world_args: dict[str, Any]
    ) -> None:
        """Construct the simulator world model.

        Args:
            world_type: Type of world model to be constructed.
            world_args: Arguments for world model construction, passed to the world
                from the environment.
        """
        self.world = world_type(self, self.sim_rate, **world_args)

    def run(self) -> None:
        """Propagate the simulator.

        Propagates for a duration up to the ``max_step_duration``, stopping if the
        environment time limit is reached or an event is triggered.
        """
        with profile_section(self, "simulator.run.total"):
            if "max_step_duration" in self.eventMap:
                self.delete_event("max_step_duration")

            with profile_section(self, "simulator.run.create_max_step_event"):
                self.createNewEvent(
                    "max_step_duration",
                    mc.sec2nano(self.sim_rate),
                    True,
                    [
                        f"self.TotalSim.CurrentNanos * {mc.NANO2SEC} >= {self.sim_time + self.max_step_duration}"
                    ],
                    ["self.logger.info('Max step duration reached')"],
                    terminal=True,
                )
            with profile_section(self, "simulator.run.execute"):
                self.ConfigureStopTime(mc.sec2nano(min(self.time_limit, 2**31)))
                self.ExecuteSimulation()

    def delete_event(self, event_name) -> None:
        """Remove an event from the event map.

        Makes event checking faster. Due to a performance issue in Basilisk, it is
        necessary to remove created for tasks that are no longer needed (even if it is
        inactive), or else significant time is spent processing the event at each step.
        """
        # event = self.eventMap[event_name]
        # self.eventList.remove(event)
        del self.eventMap[event_name]

    def __del__(self):
        """Log when simulator is deleted."""
        logger.debug("Basilisk simulator deleted")


__all__ = []
