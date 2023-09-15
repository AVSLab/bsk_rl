import bisect
import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union
from weakref import proxy

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.envs.general_satellite_tasking.types import (
        DynamicsModel,
        FSWModel,
        Simulator,
    )

import chebpy
import numpy as np
from Basilisk.utilities import macros
from gymnasium import spaces

from bsk_rl.envs.general_satellite_tasking.scenario.data import (
    DataStore,
    UniqueImageStore,
)
from bsk_rl.envs.general_satellite_tasking.scenario.environment_features import Target
from bsk_rl.envs.general_satellite_tasking.simulation import dynamics, fsw
from bsk_rl.envs.general_satellite_tasking.utils.functional import (
    collect_default_args,
    safe_dict_merge,
    valid_func_name,
)
from bsk_rl.envs.general_satellite_tasking.utils.orbital import (
    TrajectorySimulator,
    elevation,
)

SatObs = Any
SatAct = Any


class Satellite(ABC):
    dyn_type: type["DynamicsModel"]  # Type of dynamics model used by this satellite
    fsw_type: type["FSWModel"]  # Type of FSW model used by this satellite

    @classmethod
    def default_sat_args(cls, **kwargs) -> dict[str, Any]:
        """Compile default arguments for FSW and dynamics models

        Returns:
            default arguments for satellite models
        """
        defaults = collect_default_args(cls.dyn_type)
        defaults = safe_dict_merge(defaults, collect_default_args(cls.fsw_type))
        for name in dir(cls.fsw_type):
            if inspect.isclass(getattr(cls.fsw_type, name)) and issubclass(
                getattr(cls.fsw_type, name), fsw.Task
            ):
                defaults = safe_dict_merge(
                    defaults, collect_default_args(getattr(cls.fsw_type, name))
                )

        for k, v in kwargs.items():
            assert k in defaults, f"{k} not a parameter"
            defaults[k] = v
        return defaults

    def __init__(
        self,
        name: str,
        sat_args: Optional[dict[str, Any]],
        variable_interval: bool = True,
        **kwargs,
    ) -> None:
        """Base satellite constructor

        Args:
            name: identifier for satellite; does not need to be unique
            sat_args: arguments for FSW and dynamic model construction. {key: value or
                key: function}, where function is called at reset to set the value (used
                for randomization).
            variable_interval: Stop simulation at terminal events
        """
        self.name = name
        if sat_args is None:
            sat_args = self.default_sat_args()
        self.sat_args_generator = self.default_sat_args(**sat_args)
        self.simulator: Simulator
        self.fsw: "FSWModel"
        self.dynamics: "DynamicsModel"
        self.data_store: DataStore
        self.variable_interval = variable_interval
        self._timed_terminal_event_name = None

    @property
    def id(self) -> str:
        """Unique human-readable identifier"""
        return f"{self.name}_{id(self)}"

    def _generate_sat_args(self) -> None:
        """Instantiate sat_args from any randomizers in provided sat_args"""
        self.sat_args = {
            k: v if not callable(v) else v() for k, v in self.sat_args_generator.items()
        }

    def reset_pre_sim(self) -> None:
        """Called in environment reset, before simulator initialization"""
        self.info = []
        self._generate_sat_args()
        assert self.data_store.is_fresh
        self.data_store.is_fresh = False

        self.trajectory = TrajectorySimulator(
            utc_init=self.sat_args["utc_init"],
            rN=self.sat_args["rN"],
            vN=self.sat_args["vN"],
            oe=self.sat_args["oe"],
            mu=self.sat_args["mu"],
        )
        self._timed_terminal_event_name = None

    def set_simulator(self, simulator: "Simulator"):
        """Sets the simulator for models; called during simulator initialization

        Args:
            simulator: Basilisk simulator
        """
        self.simulator = proxy(simulator)

    def set_dynamics(self, dyn_rate: float) -> "DynamicsModel":
        """Create dynamics model; called during simulator initialization

        Args:
            dyn_rate: rate for dynamics simulation [s]

        Returns:
            Satellite's dynamics model
        """
        dynamics = self.dyn_type(self, dyn_rate, **self.sat_args)
        self.dynamics = proxy(dynamics)
        return dynamics

    def set_fsw(self, fsw_rate: float) -> "FSWModel":
        """Create flight software model; called during simulator initialization

        Args:
            fsw_rate: rate for FSW simulation [s]

        Returns:
            Satellite's FSW model
        """
        fsw = self.fsw_type(self, fsw_rate, **self.sat_args)
        self.fsw = proxy(fsw)
        return fsw

    def reset_post_sim(self) -> None:
        """Called in environment reset, after simulator initialization"""
        pass

    @property
    def observation_space(self) -> spaces.Box:
        """Observation space for single satellite, determined from observation

        Returns:
            gymanisium observation space
        """
        return spaces.Box(
            low=-1e16, high=1e16, shape=self.get_obs().shape, dtype=np.float64
        )

    @property
    @abstractmethod  # pragma: no cover
    def action_space(self) -> spaces.Space:
        """Action space for single satellite

        Returns:
            gymanisium action space
        """
        pass

    def is_alive(self) -> bool:
        """Check if the satellite is violating any requirements from dynamics or FSW
        models

        Returns:
            is alive
        """
        return self.dynamics.is_alive() and self.fsw.is_alive()

    @property
    def _satellite_command(self) -> str:
        """Generate string that refers to self in simBase"""
        return (
            "[satellite for satellite in self.satellites "
            + f"if satellite.id=='{self.id}'][0]"
        )

    def _info_command(self, info: str) -> str:
        """Generate command to log to info from an event

        Args:
            info: information to log; cannot include `'` or `"`

        Returns:
            actionList action for simBase.createNewEvent
        """
        return self._satellite_command + f".info.append((self.sim_time, '{info}'))"

    def log_info(self, info: Any) -> None:
        """Record information at the current time

        Args:
            info: Information to log
        """
        self.info.append((self.simulator.sim_time, info))

    def _update_timed_terminal_event(
        self, t_close: float, info: str = "", extra_actions=[]
    ) -> None:
        """Create a simulator event that causes the simulation to stop at a certain time

        Args:
            t_close: Termination time [s]
            info: Additional identifying info to log at terminal time
        """
        self._disable_timed_terminal_event()
        self.log_info(f"setting timed terminal event at {t_close:.1f}")

        # Create new timed terminal event
        self._timed_terminal_event_name = valid_func_name(
            f"timed_terminal_{t_close}_{self.id}"
        )
        self.simulator.createNewEvent(
            self._timed_terminal_event_name,
            macros.sec2nano(self.simulator.sim_rate),
            True,
            [f"self.TotalSim.CurrentNanos * {macros.NANO2SEC} >= {t_close}"],
            [
                self._info_command(f"timed termination at {t_close:.1f} " + info),
            ]
            + extra_actions,
            terminal=self.variable_interval,
        )
        self.simulator.eventMap[self._timed_terminal_event_name].eventActive = True

    def _disable_timed_terminal_event(self) -> None:
        """Turn off simulator termination due to this satellite's window close
        checker"""
        if (
            self._timed_terminal_event_name is not None
            and self._timed_terminal_event_name in self.simulator.eventMap
        ):
            self.simulator.delete_event(self._timed_terminal_event_name)

    @abstractmethod  # pragma: no cover
    def get_obs(self) -> SatObs:
        """Construct the satellite's observation

        Returns:
            satellite observation
        """
        pass

    @abstractmethod  # pragma: no cover
    def set_action(self, action: int) -> None:
        """Enables certain processes in the simulator to command the satellite task.
            Should call an @action from FSW, among other things.

        Args:
            action: action index
        """
        pass


class ImagingSatellite(Satellite):
    dyn_type = dynamics.ImagingDynModel
    fsw_type = fsw.ImagingFSWModel

    def __init__(
        self,
        name: str,
        sat_args: dict[str, Any],
        *args,
        generation_duration: float = 60 * 95 / 10,
        initial_generation_duration: Optional[float] = None,
        target_dist_threshold: float = 1e6,
        **kwargs,
    ) -> None:
        """Satellite with agile imaging capabilities. Ends the simulation when a target
        is imaged or missed

        Args:
            name: Satellite.name
            sat_args: Satellite.sat_args
            n_ahead_observe: Number of upcoming targets to include in observations.
            n_ahead_act: Number of upcoming targets to include in actions.
            generation_duration: Duration to calculate additional imaging windows for
                when windows are exhausted. If `None`, generate for the simulation
                `time_limit` unless the simulation is infinite. [s]
            initial_generation_duration: Duration to initially calculate imaging windows
                [s]
            target_dist_threshold: Distance bound [m] for evaluating imaging windows
                more exactly.
        """
        super().__init__(name, sat_args, *args, **kwargs)
        self.generation_duration = generation_duration
        self.initial_generation_duration = initial_generation_duration
        self.min_elev = sat_args["imageTargetMinimumElevation"]  # Used for window calcs
        self.target_dist_threshold = target_dist_threshold
        self.fsw: ImagingSatellite.fsw_type
        self.dynamics: ImagingSatellite.dyn_type
        self.data_store: UniqueImageStore

    def reset_pre_sim(self) -> None:
        """Set the buffer parameters based on computed windows"""
        super().reset_pre_sim()
        self.sat_args["transmitterNumBuffers"] = len(
            self.data_store.env_knowledge.targets
        )
        self.sat_args["bufferNames"] = [
            target.id for target in self.data_store.env_knowledge.targets
        ]
        self.opportunities: list[dict] = []
        self.window_calculation_time = 0
        self.current_action = None
        self._image_event_name = None
        self.imaged = 0
        self.missed = 0

    def reset_post_sim(self) -> None:
        """Handle initial_generation_duration setting and calculate windows"""
        super().reset_post_sim()
        if self.initial_generation_duration is None:
            if self.simulator.time_limit == float("inf"):
                self.initial_generation_duration = 0
            else:
                self.initial_generation_duration = self.simulator.time_limit
        self.calculate_additional_windows(self.initial_generation_duration)

    def calculate_additional_windows(self, duration: float) -> None:
        """Use a multiroot finding method to evaluate imaging windows for each target;
            data is saved to self.windows.

        Args:
            duration: Time to calculate windows from end of previous window.
        """
        if duration <= 0:
            return
        calculation_start = self.window_calculation_time
        calculation_end = self.window_calculation_time + max(
            duration, self.trajectory.dt * 2, self.generation_duration
        )
        calculation_end = self.generation_duration * np.ceil(
            calculation_end / self.generation_duration
        )

        # Get discrete times and positions for next trajectory segment
        self.trajectory.extend_to(calculation_end)
        r_BP_P_interp = self.trajectory.r_BP_P
        window_calc_span = np.logical_and(
            r_BP_P_interp.x >= calculation_start - 1e-9,
            r_BP_P_interp.x <= calculation_end + 1e-9,
        )  # Account for floating point error in window_calculation_time
        times = r_BP_P_interp.x[window_calc_span]
        positions = r_BP_P_interp.y[window_calc_span]

        for target in self.data_store.env_knowledge.targets:
            candidate_windows = self._find_candidate_windows(
                target.location, times, positions, self.target_dist_threshold
            )

            for candidate_window in candidate_windows:
                roots = self._find_elevation_roots(
                    r_BP_P_interp, target.location, self.min_elev, candidate_window
                )
                new_windows = self._refine_window(
                    roots, candidate_window, (times[0], times[-1])
                )
                for new_window in new_windows:
                    self._add_window(target, new_window, merge_time=times[0])

        self.window_calculation_time = calculation_end

    @staticmethod
    def _find_elevation_roots(
        position_interp,
        location: np.ndarray,
        min_elev: float,
        window: tuple[float, float],
    ):
        def root_fn(t):
            return elevation(position_interp(t), location) - min_elev

        settings = chebpy.UserPreferences()
        with settings:
            settings.eps = 1e-6
            settings.sortroots = True
            roots = chebpy.chebfun(root_fn, window).roots()

        return roots

    @staticmethod
    def _find_candidate_windows(
        location: np.ndarray, times: np.ndarray, positions: np.ndarray, threshold: float
    ) -> list[tuple[float, float]]:
        """Find `times` where a window is plausible; i.e. where a `positions` point is
        within `threshold` of `location`. Too big of a dt in times may miss windows or
        produce bad results."""
        close_times = np.linalg.norm(positions - location, axis=1) < threshold
        close_indices = np.where(close_times)[0]
        groups = np.split(close_indices, np.where(np.diff(close_indices) != 1)[0] + 1)
        groups = [group for group in groups if len(group) > 0]
        candidate_windows = []
        for group in groups:
            t_start = times[max(0, group[0] - 1)]
            t_end = times[min(len(times) - 1, group[-1] + 1)]
            candidate_windows.append((t_start, t_end))
        return candidate_windows

    @staticmethod
    def _refine_window(
        endpoints: Iterable,
        candidate_window: tuple[float, float],
        computation_window: tuple[float, float],
    ) -> list[tuple[float, float]]:
        endpoints = list(endpoints)
        if len(endpoints) % 2 == 1:
            if candidate_window[0] == computation_window[0]:
                endpoints.insert(0, computation_window[0])
            elif candidate_window[-1] == computation_window[-1]:
                endpoints.append(computation_window[-1])
            else:
                raise ValueError()

        new_windows = []
        for t1, t2 in zip(endpoints[0::2], endpoints[1::2]):
            new_windows.append((t1, t2))

        return new_windows

    def _add_window(
        self,
        target: Target,
        new_window: tuple[float, float],
        merge_time: Optional[float] = None,
    ):
        """
        Args:
            target: Target to add window for
            new_window: New window for target
            merge_time: Time at which merges with existing windows will occur. If None,
                check all windows for merges
        """
        if new_window[0] == merge_time or merge_time is None:
            for window in self.opportunities:
                if window["target"] == target and window["window"][1] == new_window[0]:
                    window["window"] = (window["window"][0], new_window[1])
                    return
        bisect.insort(
            self.opportunities,
            {"target": target, "window": new_window},
            key=lambda x: x["window"][1],
        )

    @property
    def windows(self) -> dict[Target, list[tuple[float, float]]]:
        """Access windows via dict of targets -> list of windows"""
        windows = {}
        for opportunity in self.opportunities:
            if opportunity["target"] not in windows:
                windows[opportunity["target"]] = []
            windows[opportunity["target"]].append(opportunity["window"])
        return windows

    @property
    def upcoming_opportunities(self) -> list[dict]:
        """Subset of opportunities that have not yet closed. Attempts to filter out
        known imaged windows if data on imaged windows is accessible."""
        start = bisect.bisect_left(
            self.opportunities, self.simulator.sim_time, key=lambda x: x["window"][1]
        )
        upcoming = self.opportunities[start:]
        try:  # Attempt to filter already known imaged targets
            upcoming = [
                opportunity
                for opportunity in upcoming
                if opportunity["target"] not in self.data_store.data.imaged
            ]
        except AttributeError:
            pass
        return upcoming

    @property
    def upcoming_windows(self) -> dict[Target, list[tuple[float, float]]]:
        """Access upcoming windows in a dict of targets -> list of windows."""
        windows = {}
        for window in self.upcoming_opportunities:
            if window["target"] not in windows:
                windows[window["target"]] = []
            windows[window["target"]].append(window["window"])
        return windows

    @property
    def next_windows(self) -> dict[Target, tuple[float, float]]:
        """Soonest window for each target.

        Returns:
            dict: first non-closed window for each target
        """
        next_windows = {}
        for opportunity in self.upcoming_opportunities:
            if opportunity["target"] not in next_windows:
                next_windows[opportunity["target"]] = opportunity["window"]
        return next_windows

    def upcoming_targets(
        self, n: int, pad: bool = True, max_lookahead: int = 100
    ) -> list[Target]:
        """Find the n nearest targets. Targets are sorted by window close time;
        currently open windows are included.

        Args:
            n: number of windows to look ahead
            pad: if true, duplicates the last target if the number of targets found is
                less than n
            max_lookahead: maximum times to call calculate_additional_windows

        Returns:
            list: n nearest targets, ordered
        """
        if n == 0:
            return []
        for _ in range(max_lookahead):
            soonest = self.upcoming_opportunities
            if len(soonest) < n:
                self.calculate_additional_windows(self.generation_duration)
            else:
                break
        targets = [opportunity["target"] for opportunity in soonest[0:n]]
        if pad:
            targets += [targets[-1]] * (n - len(targets))
        return targets

    def _update_image_event(self, target: Target) -> None:
        """Create a simulator event that causes the simulation to stop when a target is
        imaged

        Args:
            target: Target expected to be imaged
        """
        self._disable_image_event()

        self._image_event_name = valid_func_name(f"image_{self.id}_{target.id}")
        if self._image_event_name not in self.simulator.eventMap.keys():
            data_names = np.array(
                list(
                    self.dynamics.storageUnit.storageUnitDataOutMsg.read().storedDataName
                )
            )
            data_index = int(np.where(data_names == target.id)[0][0])
            current_data_level = (
                self.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData[
                    data_index
                ]
            )
            self.simulator.createNewEvent(
                self._image_event_name,
                macros.sec2nano(self.fsw.fsw_rate),
                True,
                [
                    f"self.dynamics_list['{self.id}'].storageUnit.storageUnitDataOutMsg.read()"
                    + f".storedData[{data_index}] > {current_data_level}"
                ],
                [
                    self._info_command(f"imaged {target}"),
                    self._satellite_command + ".imaged += 1",
                ],
                terminal=self.variable_interval,
            )
        else:
            self.simulator.eventMap[self._image_event_name].eventActive = True

    def _disable_image_event(self) -> None:
        """Turn off simulator termination due to this satellite's imaging checker"""
        if (
            self._image_event_name is not None
            and self._image_event_name in self.simulator.eventMap
        ):
            self.simulator.delete_event(self._image_event_name)
            # self.simulator.eventMap[self._image_event_name].eventActive = False

    def parse_target_selection(self, target_query: Union[int, Target, str]):
        """Identify a target based on upcoming target index, Target object, or target
        id.

        Args:
            target_query: Taret upcoming index, object, or id.
        """
        if np.issubdtype(type(target_query), np.integer):
            target = self.upcoming_targets(target_query + 1)[-1]
        elif isinstance(target_query, Target):
            target = target_query
        elif isinstance(target_query, str):
            target = [
                target
                for target in self.data_store.env_knowledge.targets
                if target.id == target_query
            ][0]
        else:
            raise TypeError(f"Invalid target_query! Cannot be a {type(target_query)}!")

        return target

    def enable_target_window(self, target: Target):
        """Enable the next window close event for target"""
        self._update_image_event(target)
        next_window = self.next_windows[target]
        self.log_info(
            f"{target} window enabled: {next_window[0]:.1f} to {next_window[1]:.1f}"
        )
        self._update_timed_terminal_event(
            next_window[1],
            info=f"for {target} window",
            extra_actions=[self._satellite_command + ".missed += 1"],
        )

    def task_target_for_imaging(self, target: Target):
        """Task the satellite to image a target

        Args:
            target: Selected target
        """
        msg = f"{target} tasked for imaging"
        self.log_info(msg)
        self.fsw.action_image(target.location, target.id)
        self.enable_target_window(target)


#########################
### Convenience Types ###
#########################
class SteeringImagerSatellite(ImagingSatellite):
    dyn_type = dynamics.FullFeaturedDynModel
    fsw_type = fsw.SteeringImagerFSWModel


class FBImagerSatellite(ImagingSatellite):
    dyn_type = dynamics.FullFeaturedDynModel
    fsw_type = fsw.ImagingFSWModel


##########################
### Ready-to-use Types ###
##########################
from Basilisk.utilities import orbitalMotion  # noqa: E402

from bsk_rl.envs.general_satellite_tasking.scenario import (  # noqa: E402
    sat_actions as sa,
)
from bsk_rl.envs.general_satellite_tasking.scenario import (  # noqa: E402
    sat_observations as so,
)


class DoNothingSatellite(sa.DriftAction, so.TimeState):
    dyn_type = dynamics.BasicDynamicsModel
    fsw_type = fsw.BasicFSWModel


class ImageAheadSatellite(
    sa.ImagingActions,
    so.TimeState,
    so.TargetState.configure(n_ahead_observe=3),
    so.NormdPropertyState.configure(
        obs_properties=[
            dict(prop="omega_BP_P", norm=0.03),
            dict(prop="c_hat_P"),
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", norm=7616.5),
            dict(prop="battery_charge_fraction"),
        ]
    ),
    SteeringImagerSatellite,
):
    pass


class FullFeaturedSatellite(
    sa.ImagingActions,
    sa.DesatAction,
    sa.ChargingAction,
    so.TimeState,
    so.TargetState.configure(n_ahead_observe=3),
    so.NormdPropertyState.configure(
        obs_properties=[
            dict(prop="omega_BP_P", norm=0.03),
            dict(prop="c_hat_P"),
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", norm=7616.5),
            dict(prop="battery_charge_fraction"),
        ]
    ),
    SteeringImagerSatellite,
):
    pass
