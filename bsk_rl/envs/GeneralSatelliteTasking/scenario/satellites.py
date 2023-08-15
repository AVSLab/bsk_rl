import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union
from weakref import proxy

if TYPE_CHECKING:
    from bsk_rl.envs.GeneralSatelliteTasking.types import (
        DynamicsModel,
        FSWModel,
        Simulator,
    )

import chebpy
import numpy as np
from Basilisk.utilities import macros
from gymnasium import spaces

from bsk_rl.envs.GeneralSatelliteTasking.scenario.data import (
    DataStore,
    UniqueImageStore,
)
from bsk_rl.envs.GeneralSatelliteTasking.scenario.environment_features import Target
from bsk_rl.envs.GeneralSatelliteTasking.simulation import dynamics, fsw
from bsk_rl.envs.GeneralSatelliteTasking.utils.functional import (
    collect_default_args,
    safe_dict_merge,
    valid_func_name,
)
from bsk_rl.envs.GeneralSatelliteTasking.utils.orbital import (
    TrajectorySimulator,
    elevation,
)


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

    def __init__(self, name: str, sat_args: Optional[dict[str, Any]]) -> None:
        """Base satellite constructor

        Args:
            name: identifier for satellite; does not need to be unique
            sat_args: arguments for FSW and dynamic model construction. {key: value or key: function},
                where function is called at reset to set the value (used for randomization).
        """
        self.name = name
        if sat_args is None:
            sat_args = self.default_sat_args()
        self.sat_args_generator = self.default_sat_args(**sat_args)
        self.simulator: Simulator
        self.fsw: "FSWModel"
        self.dynamics: "DynamicsModel"
        self.data_store: DataStore

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
    def action_space(self) -> spaces.Discrete:
        """Action space for single satellite, computed from n_actions property

        Returns:
            gymanisium action space
        """
        return spaces.Discrete(self.n_actions)

    def is_alive(self) -> bool:
        """Check if the satellite is violating any requirements from dynamics or FSW models

        Returns:
            is alive
        """
        return self.dynamics.is_alive() and self.fsw.is_alive()

    @property
    def _satellite_command(self) -> str:
        """Generate string that refers to self in simBase"""
        return f"[satellite for satellite in self.satellites if satellite.id=='{self.id}'][0]"

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

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Construct the satellite's observation

        Returns:
            satellite observation
        """
        pass

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Number of actions the satellite can take

        Returns:
            int: number of actions
        """
        pass

    @abstractmethod
    def set_action(self, action: int) -> None:
        """Enables certain processes in the simulator to command the satellite task. Should
            call an @action from FSW.

        Args:
            action: action index
        """
        pass


class BasicSatellite(Satellite):
    dyn_type = dynamics.BasicDynamicsModel
    fsw_type = fsw.BasicFSWModel

    def __init__(self, name: str, sat_args: dict[str, Any]) -> None:
        super().__init__(name, sat_args)

    @property
    def n_actions(self) -> int:
        return 2

    def get_obs(self) -> Iterable[float]:
        return np.array([0.0])

    def set_action(self, action: int) -> None:
        if action == 0:
            self.fsw.action_charge()
        elif action == 1:
            self.fsw.action_desat()
        else:
            raise ValueError("Invalid action")


class ImagingSatellite(BasicSatellite):
    dyn_type = dynamics.ImagingDynModel
    fsw_type = fsw.ImagingFSWModel

    def __init__(
        self,
        name: str,
        sat_args: dict[str, Any],
        n_ahead_observe: int = 20,
        n_ahead_act: int = 10,
        generation_duration: float = 60 * 95 / 10,
        initial_generation_duration: Optional[float] = None,
        target_dist_threshold: float = 1e6,
        variable_interval: bool = True,
    ) -> None:
        """Satellite with agile imaging capabilities. Ends the simulation when a target is imaged or missed

        Args:
            name: Satellite.name
            sat_args: Satellite.sat_args
            n_ahead_observe: Number of upcoming targets to include in observations.
            n_ahead_act: Number of upcoming targets to include in actions.
            generation_duration: Duration to calculate additional imaging windows for when windows are exhausted. If
                `None`, generate for the simulation `time_limit` unless the simulation is infinite. [s]
            initial_generation_duration: Duration to initially calculate imaging windows [s]
            target_dist_threshold: Distance bound [m] for evaluating imaging windows more exactly.
            variable_interval: Stop simulation when a target is imaged or a imaging window closes
        """
        super().__init__(name, sat_args)
        self.n_ahead_observe = int(n_ahead_observe)
        self.n_ahead_act = int(n_ahead_act)
        self.generation_duration = generation_duration
        self.initial_generation_duration = initial_generation_duration
        self.min_elev = sat_args["imageTargetMinimumElevation"]  # Used for window calcs
        self.target_dist_threshold = target_dist_threshold
        self.fsw: ImagingSatellite.fsw_type
        self.dynamics: ImagingSatellite.dyn_type
        self.data_store: UniqueImageStore
        self.variable_interval = variable_interval

    def reset_pre_sim(self) -> None:
        """Set the buffer parameters based on computed windows"""
        super().reset_pre_sim()
        self.sat_args["transmitterNumBuffers"] = len(
            self.data_store.env_knowledge.targets
        )
        self.sat_args["bufferNames"] = [
            target.id for target in self.data_store.env_knowledge.targets
        ]
        self.windows = {}
        self.window_calculation_time = 0
        self.current_action = None
        self._image_event_name = None
        self._window_close_event_name = None
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
        self.window_calculation_time += max(
            duration, self.trajectory.dt * 2, self.generation_duration
        )

        # simulate next trajectory segment
        self.trajectory.extend_to(self.window_calculation_time)
        r_BP_P_interp = self.trajectory.r_BP_P
        window_calc_span = np.logical_and(
            r_BP_P_interp.x >= calculation_start,
            r_BP_P_interp.x <= self.window_calculation_time,
        )
        times = r_BP_P_interp.x[window_calc_span]
        positions = r_BP_P_interp.y[window_calc_span]

        for target in self.data_store.env_knowledge.targets:
            # Find times where a window is plausible
            # i.e. where a interpolation point is within target_dist_threshold of the target
            close_times = (
                np.linalg.norm(positions - target.location, axis=1)
                < self.target_dist_threshold
            )
            close_indices = np.where(close_times)[0]
            groups = np.split(
                close_indices, np.where(np.diff(close_indices) != 1)[0] + 1
            )
            groups = [group for group in groups if len(group) > 0]

            for group in groups:
                i_start = max(0, group[0] - 1)
                i_end = min(len(times) - 1, group[-1] + 1)

                root_fn = (
                    lambda t: elevation(r_BP_P_interp(t), target.location)
                    - self.min_elev
                )  # noqa: E731
                settings = chebpy.UserPreferences()
                with settings:
                    settings.eps = 1e-6
                    settings.sortroots = True
                    roots = chebpy.chebfun(
                        root_fn, [times[i_start], times[i_end]]
                    ).roots()

                # Create windows from roots
                if len(roots) == 2:
                    new_window = (roots[0], roots[1])
                elif len(roots) == 1 and times[i_start] == times[0]:
                    new_window = (times[0], roots[0])
                elif len(roots) == 1 and times[i_end] == times[-1]:
                    new_window = (roots[0], times[-1])
                else:
                    break

                # Merge touching windows
                if target in self.windows:
                    if new_window[0] == times[0]:
                        for window in self.windows[target]:
                            if window[1] == new_window[0]:
                                window[1] == new_window[1]
                    else:
                        self.windows[target].append(new_window)
                else:
                    self.windows[target] = [new_window]

    @property
    def upcoming_windows(self) -> dict[Target, list[tuple[float, float]]]:
        """Subset of windows that have not yet closed. Attempts to filter out known imaged windows if data is accessible

        Returns:
            filtered windows
        """
        try:  # Attempt to filter already known imaged targets
            return {
                tgt: [
                    window for window in windows if window[1] > self.simulator.sim_time
                ]
                for tgt, windows in self.windows.items()
                if any(window[1] > self.simulator.sim_time for window in windows)
                and tgt not in self.data_store.data.imaged
            }
        except AttributeError:
            return {
                tgt: [
                    window for window in windows if window[1] > self.simulator.sim_time
                ]
                for tgt, windows in self.windows.items()
                if any(window[1] > self.simulator.sim_time for window in windows)
            }

    @property
    def next_windows(self) -> dict[Target, tuple[float, float]]:
        """Soonest window for each target

        Returns:
            dict: first non-closed window for each target
        """
        return {tgt: windows[0] for tgt, windows in self.upcoming_windows.items()}

    def upcoming_targets(
        self, n: int, pad: bool = True, max_lookahead: int = 100
    ) -> list[Target]:
        """Find the n nearest targets. Targets are sorted by window close time; currently open windows are included.
        Only the first window for a target is accounted for.

        Args:
            n: number of windows to look ahead
            pad: if true, duplicates the last target if the number of targets found is less than n
            max_lookahead: maximum times to call calculate_additional_windows

        Returns:
            list: n nearest targets, ordered
        """
        for _ in range(max_lookahead):
            soonest = sorted(self.next_windows.items(), key=lambda x: x[1][1])
            if len(soonest) < n:
                self.calculate_additional_windows(self.generation_duration)
            else:
                break
        targets = [target for target, _ in soonest[0:n]]
        if pad:
            targets += [targets[-1]] * (n - len(targets))
        return targets

    def get_obs(self) -> Iterable[float]:
        dynamic_state = np.concatenate(
            [
                self.dynamics.omega_BP_P,
                self.fsw.c_hat_P,
                self.dynamics.r_BN_P,
                self.dynamics.v_BN_P,
            ]
        )
        images_state = np.array(
            [
                np.concatenate([[target.priority], target.location])
                for target in self.upcoming_targets(self.n_ahead_observe)
            ]
        )
        images_state = images_state.flatten()

        return np.concatenate((dynamic_state, images_state))

    @property
    def n_actions(self) -> int:
        """Satellite can take n_ahead_act imaging actions"""
        return self.n_ahead_act

    def _disable_window_close_event(self) -> None:
        """Turn off simulator termination due to this satellite's window close checker"""
        if self._window_close_event_name is not None:
            self.simulator.eventMap[self._window_close_event_name].eventActive = False

    def _update_window_close_event(self, t_close: float, info: str = "") -> None:
        """Create a simulator event that causes the simulation to stop at a certain time

        Args:
            t_close: Termination time [s]
            info: Additional identifying info to log at window close
        """
        self._disable_window_close_event()

        # Create new window close event
        self._window_close_event_name = valid_func_name(
            f"window_close_{t_close}_{self.id}"
        )
        self.simulator.createNewEvent(
            self._window_close_event_name,
            macros.sec2nano(self.simulator.sim_rate),
            True,
            [f"self.TotalSim.CurrentNanos * {macros.NANO2SEC} >= {t_close}"],
            [
                self._info_command(f"window closed at {t_close:.1f} " + info),
                self._satellite_command + ".missed += 1",
            ],
            terminal=self.variable_interval,
        )
        self.simulator.eventMap[self._window_close_event_name].eventActive = True

    def _disable_image_event(self) -> None:
        """Turn off simulator termination due to this satellite's imaging checker"""
        if self._image_event_name is not None:
            self.simulator.eventMap[self._image_event_name].eventActive = False

    def _update_image_event(self, target: Target) -> None:
        """Create a simulator event that causes the simulation to stop when a target is imaged

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

    def set_action(self, action: Union[int, Target, str]) -> None:
        """Select the satellite action; does not reassign action if the same target is selected twice

        Args:
            action: image the nth upcoming target (or image by target id or object)
        """
        if action == -1:
            self._disable_window_close_event()
            self._disable_image_event()
            self.fsw.action_drift
            self.current_action = -1
            return

        if np.issubdtype(type(action), np.integer):
            target = self.upcoming_targets(action + 1)[-1]
        elif isinstance(action, Target):
            target = action
        elif isinstance(action, str):
            target = [
                target
                for target in self.data_store.env_knowledge.targets
                if target.id == action
            ][0]
        else:
            raise TypeError(
                f"Invalid action specification! Cannot be a {type(action)}!"
            )

        if self.current_action != target:
            msg = f"{target} tasked for imaging"
            if np.issubdtype(type(action), np.integer):
                msg = msg + f" (index {action})"
            self.log_info(msg)
            self.fsw.action_image(target.location, target.id)
            self._update_image_event(target)
            self._update_window_close_event(
                self.next_windows[target][1], info=f"for {target}"
            )

        self.current_action = target


class SteeringImagerSatellite(ImagingSatellite):
    dyn_type = dynamics.FullFeaturedDynModel
    fsw_type = fsw.SteeringImagerFSWModel


class FBImagerSatellite(ImagingSatellite):
    dyn_type = dynamics.FullFeaturedDynModel
    fsw_type = fsw.ImagingFSWModel


class FullFeaturedSatellite(ImagingSatellite):
    """Imaging satellite that uses MRP steering and can communicate to downlink ground stations"""

    dyn_type = dynamics.FullFeaturedDynModel
    fsw_type = fsw.SteeringImagerFSWModel

    @property
    def n_actions(self) -> int:
        """Satellite can take three non-imaging actions in addition to imaging actions"""
        return super().n_actions + 3

    def set_action(self, action: int) -> None:
        """Select the satellite action; does not reassign action if the same action/target is selected twice

        Args:
            action (int):
                - 0: charge
                - 1: desaturate
                - 2: downlink
                - 3+: image the (n-3)th upcoming target
        """
        if action < 2:
            self._disable_window_close_event()
            self._disable_image_event()

        if action == 0 and self.current_action != 0:
            self.fsw.action_charge()
            self.log_info("charging tasked")
        elif action == 1 and self.current_action != 1:
            self.fsw.action_desat()
            self.log_info("desat tasked")
        elif action == 2 and self.current_action != 2:
            self.fsw.action_downlink()
            self.log_info("downlink tasked")
        else:
            target_action = action
            if isinstance(target_action, int):
                target_action -= 3
            super().set_action(target_action)

        if action < 3:
            self.current_action = action
