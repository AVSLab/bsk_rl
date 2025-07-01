"""Satellites are the agents in the environment."""

import bisect
import logging
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Union

import numpy as np
from Basilisk.utilities import macros
from scipy.optimize import minimize_scalar, root_scalar

from bsk_rl.sats.satellite import Satellite
from bsk_rl.scene.targets import Target
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils import vizard
from bsk_rl.utils.functional import valid_func_name
from bsk_rl.utils.orbital import elevation

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.data.unique_image_data import UniqueImageStore

logger = logging.getLogger(__name__)

SatObs = Any
SatAct = Any


class AccessSatellite(Satellite):
    """Satellite that detects access opportunities for ground locations."""

    def __init__(
        self,
        *args,
        generation_duration: float = 600.0,
        initial_generation_duration: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Satellite that detects access opportunities for ground locations.

        This satellite can be used to computes access opportunities for ground locations
        such as imaging targets or ground stations. The satellite will calculate upcoming
        opportunities for each location and order the opportunities by close time.
        Opportunities are calculated based on a per-location minimum elevation angle.

        Args:
            args: Passed through to :class:`Satellite` constructor.
            generation_duration: [s] Duration to calculate additional opportunities for
                when the simulation time reaches the current calculation time. If
                `None`, generate opportunities for the simulation `time_limit` unless
                the simulation is infinite.
            initial_generation_duration: [s] Period to calculate opportunities for on
                environment reset.
            kwargs: Passed through to :class:`Satellite` constructor.
        """
        super().__init__(*args, **kwargs)
        self.generation_duration = generation_duration
        self.initial_generation_duration = initial_generation_duration
        self.access_filter_functions = []
        self.add_access_filter(lambda opportunity: True)

    def reset_overwrite_previous(self) -> None:
        """Overwrite previous opportunities and locations."""
        super().reset_overwrite_previous()
        self.opportunities: list[dict] = []
        self.window_calculation_time = 0
        self.locations_for_access_checking: list[dict[str, Any]] = []

    def add_location_for_access_checking(
        self,
        object: Any,
        r_LP_P: np.ndarray,
        min_elev: float,
        type: str,
    ) -> None:
        """Add a location to be included in opportunity calculations.

        .. warning::
            The added location will only be considered in future calls to
            :class:`~AccessSatellite.calculate_additional_windows`; opportunities are not
            computed retroactively.

        Args:
            object: Object for with to compute opportunities.
            r_LP_P: [m] Objects planet-fixed location.
            min_elev: [rad] Minimum elevation angle for access.
            type: Category of opportunity target provides.
        """
        location_dict = dict(r_LP_P=r_LP_P, min_elev=min_elev, type=type)
        location_dict[type] = object  # For backwards compatibility, prefer "object" key
        location_dict["object"] = object
        self.locations_for_access_checking.append(location_dict)

    def reset_post_sim_init(self) -> None:
        """Handle initial window calculations for new simulation.

        :meta private:
        """
        super().reset_post_sim_init()
        if self.initial_generation_duration is None:
            if self.simulator.time_limit == float("inf"):
                self.initial_generation_duration = 0
            else:
                self.initial_generation_duration = self.simulator.time_limit
        self.calculate_additional_windows(self.initial_generation_duration)

    def calculate_additional_windows(self, duration: float) -> None:
        """Use a multiroot finding method to evaluate imaging windows for each location.

        Args:
            duration: Time to calculate windows from end of previous window.
        """
        if duration <= 0:
            return

        self.logger.info(
            "Finding opportunity windows from "
            f"{self.window_calculation_time:.2f} to "
            f"{self.window_calculation_time + duration:.2f} seconds"
        )
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

        r_max = np.max(np.linalg.norm(positions, axis=-1))
        access_dist_thresh_multiplier = 1.1
        for location in self.locations_for_access_checking:
            alt_est = r_max - np.linalg.norm(location["r_LP_P"])
            access_dist_threshold = (
                access_dist_thresh_multiplier * alt_est / np.sin(location["min_elev"])
            )
            candidate_windows = self._find_candidate_windows(
                location["r_LP_P"], times, positions, access_dist_threshold
            )

            for candidate_window in candidate_windows:
                roots = self._find_elevation_roots(
                    r_BP_P_interp,
                    location["r_LP_P"],
                    location["min_elev"],
                    candidate_window,
                )
                new_windows = self._refine_window(
                    roots, candidate_window, (times[0], times[-1])
                )
                for new_window in new_windows:
                    self._add_window(
                        location["object"],
                        new_window,
                        type=location["type"],
                        r_LP_P=location["r_LP_P"],
                        merge_time=times[0],
                    )

        self.window_calculation_time = calculation_end

    @staticmethod
    def _find_elevation_roots(
        position_interp,
        location: np.ndarray,
        min_elev: float,
        window: tuple[float, float],
        min_duration: float = 0.1,
    ):
        """Find times where the elevation is equal to the minimum elevation.

        Finds exact times where the satellite's elevation relative to a target is
        equal to the minimum elevation.
        """

        def root_fn(t):
            return -(elevation(position_interp(t), location) - min_elev)

        elev_0, elev_1 = root_fn(window[0]), root_fn(window[1])

        if elev_0 < 0 and elev_1 < 0:
            logger.warning(
                "initial_generation_duration is shorter than the maximum window length; some windows may be neglected."
            )
            return []
        elif elev_0 < 0 or elev_1 < 0:
            return [root_scalar(root_fn, bracket=window).root]
        else:
            res = minimize_scalar(root_fn, bracket=window, tol=1e-4)
            if res.fun < 0:
                window_mid = res.x
                r_open = root_scalar(root_fn, bracket=(window[0], window_mid)).root
                r_close = root_scalar(root_fn, bracket=(window_mid, window[1])).root
                if r_close - r_open > min_duration:
                    return [r_open, r_close]

        return []

    @staticmethod
    def _find_candidate_windows(
        location: np.ndarray, times: np.ndarray, positions: np.ndarray, threshold: float
    ) -> list[tuple[float, float]]:
        """Find `times` where a window is plausible.

        i.e. where a `positions` point is within `threshold` of `location`. Too big of
        a dt in times may miss windows or produce bad results.
        """
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
        """Detect if an exact window has been truncated by a coarse window."""
        endpoints = list(endpoints)

        # Filter endpoints that are too close
        for i, endpoint in enumerate(endpoints[0:-1]):
            if abs(endpoint - endpoints[i + 1]) < 1e-6:
                endpoints[i] = None
        endpoints = [endpoint for endpoint in endpoints if endpoint is not None]

        # Find pairs
        if len(endpoints) % 2 == 1:
            if candidate_window[0] == computation_window[0]:
                endpoints.insert(0, computation_window[0])
            elif candidate_window[-1] == computation_window[-1]:
                endpoints.append(computation_window[-1])
            else:
                return []  # Temporary fix for rare issue.

        new_windows = []
        for t1, t2 in zip(endpoints[0::2], endpoints[1::2]):
            new_windows.append((t1, t2))

        return new_windows

    def _add_window(
        self,
        object: Any,
        new_window: tuple[float, float],
        type: str,
        r_LP_P: np.ndarray,
        merge_time: Optional[float] = None,
    ):
        """Add an opportunity window.

        Args:
            object: Object to add window for
            new_window: New window for target
            type: Type of window being added
            r_LP_P: Planet-fixed location of object
            merge_time: Time at which merges with existing windows will occur. If None,
                check all windows for merges.
        """
        if new_window[0] == merge_time or merge_time is None:
            for opportunity in self.opportunities:
                if (
                    opportunity["type"] == type
                    and opportunity["object"] == object
                    and opportunity["window"][1] == new_window[0]
                ):
                    opportunity["window"] = (opportunity["window"][0], new_window[1])
                    return
        bisect.insort(
            self.opportunities,
            {"object": object, "window": new_window, "type": type, "r_LP_P": r_LP_P},
            key=lambda x: x["window"][1],
        )

    @property
    def upcoming_opportunities(self) -> list[dict]:
        """Ordered list of opportunities that have not yet closed."""
        start = bisect.bisect_left(
            self.opportunities,
            self.simulator.sim_time + 1e-12,
            key=lambda x: x["window"][1],
        )
        upcoming = self.opportunities[start:]
        return upcoming

    def opportunities_dict(
        self,
        types: Optional[Union[str, list[str]]] = None,
        filter: Union[Optional[Callable], list] = None,
    ) -> dict[Any, list[tuple[float, float]]]:
        """Make dictionary of opportunities that maps objects to lists of windows.

        Args:
            types: Types of opportunities to include. If None, include all types.
            filter: Function that takes an opportunity dictionary and returns a boolean
                if the opportunity should be included in the output.
        """
        if isinstance(types, str):
            types = [types]

        if isinstance(filter, list):
            filter_list = filter
            filter = lambda opportunity: opportunity["object"] not in filter_list

        if filter is None:
            filter = self.default_access_filter

        windows = {}
        for opportunity in self.opportunities:
            type = opportunity["type"]
            if (types is None or type in types) and filter(opportunity):
                if opportunity["object"] not in windows:
                    windows[opportunity["object"]] = []
                windows[opportunity["object"]].append(opportunity["window"])
        return windows

    def upcoming_opportunities_dict(
        self,
        types: Optional[Union[str, list[str]]] = None,
        filter: Union[Optional[Callable], list] = None,
    ) -> dict[Any, list[tuple[float, float]]]:
        """Get dictionary of upcoming opportunities.

        Maps objects to lists of windows that have not yet closed.

        Args:
            types: Types of opportunities to include. If None, include all types.
            filter: Function that takes an opportunity dictionary and returns a boolean
                if the opportunity should be included in the output.
        """
        if isinstance(types, str):
            types = [types]

        if isinstance(filter, list):
            filter_list = filter
            filter = lambda opportunity: opportunity["object"] not in filter_list

        if filter is None:
            filter = self.default_access_filter

        windows = {}
        for opportunity in self.upcoming_opportunities:
            type = opportunity["type"]
            if (types is None or type in types) and filter(opportunity):
                if opportunity["object"] not in windows:
                    windows[opportunity["object"]] = []
                windows[opportunity["object"]].append(opportunity["window"])
        return windows

    def next_opportunities_dict(
        self,
        types: Optional[Union[str, list[str]]] = None,
        filter: Union[Optional[Callable], list] = None,
    ) -> dict[Any, tuple[float, float]]:
        """Make dictionary of opportunities that maps objects to the next open windows.

        Args:
            types: Types of opportunities to include. If None, include all types.
            filter: Function that takes an opportunity dictionary and returns a boolean
                if the opportunity should be included in the output.
        """
        if isinstance(types, str):
            types = [types]

        if isinstance(filter, list):
            filter_list = filter
            filter = lambda opportunity: opportunity["object"] not in filter_list

        if filter is None:
            filter = self.default_access_filter

        next_windows = {}
        for opportunity in self.upcoming_opportunities:
            type = opportunity["type"]
            if (types is None or type in types) and filter(opportunity):
                if opportunity["object"] not in next_windows:
                    next_windows[opportunity["object"]] = opportunity["window"]
        return next_windows

    def find_next_opportunities(
        self,
        n: int,
        pad: bool = True,
        max_lookahead: int = 100,
        types: Optional[Union[str, list[str]]] = None,
        filter: Union[Optional[Callable], list] = None,
    ) -> list[dict]:
        """Find the n nearest opportunities, sorted by window close time.

        Args:
            n: Number of opportunities to attempt to include.
            pad: If true, duplicates the last target if the number of opportunities
                found is less than n.
            max_lookahead: Maximum times to call calculate_additional_windows.
            types: Types of opportunities to include. If None, include all types.
            filter: Function that takes an opportunity dictionary and returns a boolean
                if the opportunity should be included in the output.

        Returns:
            ``n`` nearest opportunities, ordered
        """
        if isinstance(types, str):
            types = [types]

        if isinstance(filter, list):
            filter_list = filter
            filter = lambda opportunity: opportunity["object"] not in filter_list

        if filter is None:
            filter = self.default_access_filter

        if n == 0:
            return []

        for _ in range(max_lookahead):
            upcoming_opportunities = self.upcoming_opportunities
            next_opportunities = []
            for opportunity in upcoming_opportunities:
                type = opportunity["type"]
                if (types is None or type in types) and filter(opportunity):
                    next_opportunities.append(opportunity)

                if len(next_opportunities) >= n:
                    return next_opportunities
            self.calculate_additional_windows(self.generation_duration)
        if pad and len(next_opportunities) >= 1:
            next_opportunities += [next_opportunities[-1]] * (
                n - len(next_opportunities)
            )
        else:
            raise RuntimeError(
                "No opportunities found! Use add_location_for_access_checking to add locations."
            )
        return next_opportunities

    def get_access_filter(self):
        """Deprecated function.

        :meta private:
        """
        raise DeprecationWarning(
            "get_access_filter is deprecated. Use add_access_filter and default_access_filter instead."
        )

    def add_access_filter(
        self, access_filter_fn: Callable, types: Optional[Union[str, list[str]]] = None
    ):
        """Add an access filter function to the list of access filters.

        Calls to :class:`~AccessSatellite.opportunities_dict`, :class:`~AccessSatellite.find_next_opportunities`,
        and similar functions will use the boolean AND of all access filter functions,
        unless otherwise specified.

        Access filters are used by various other aspects of the environment to limit
        which opportunities are considered based on the satellite's local knowledge of
        the environment.
        """
        if types is not None:
            if isinstance(types, str):
                types = [types]

            def access_filter_type_restricted(opportunity):
                return opportunity["type"] not in types or access_filter_fn(opportunity)

            self.access_filter_functions.append(access_filter_type_restricted)
        else:
            self.access_filter_functions.append(access_filter_fn)

    @property
    def default_access_filter(self):
        """Generate a default access filter function that combines all access filters.

        :meta private:
        """

        def access_filter(opportunity):
            return all(
                [
                    access_filter_fn(opportunity)
                    for access_filter_fn in self.access_filter_functions
                ]
            )

        return access_filter


class ImagingSatellite(AccessSatellite):
    """Satellite with agile imaging capabilities."""

    dyn_type = dyn.ImagingDynModel
    fsw_type = fsw.ImagingFSWModel

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Satellite with agile imaging capabilities.

        Stop the simulation when a target is imaged or missed so that time is not wasted
        on an inaccessible or already imaged target.
        """
        super().__init__(*args, **kwargs)
        self.fsw: ImagingSatellite.fsw_type
        self.dynamics: ImagingSatellite.dyn_type
        self.data_store: "UniqueImageStore"
        self.target_types = "target"

    @property
    def known_targets(self) -> list["Target"]:
        """List of known targets."""
        try:
            return self.data_store.data.known
        except AttributeError:
            return []

    def reset_overwrite_previous(self) -> None:
        """Overwrite statistics about previous episode."""
        super().reset_overwrite_previous()
        self._image_event_name = None
        self.imaged = 0
        self.missed = 0

    def reset_pre_sim_init(self) -> None:
        """Set the buffer parameters based on computed windows.

        :meta private:
        """
        super().reset_pre_sim_init()
        self.sat_args["bufferNames"] = [
            loc["object"].id
            for loc in self.locations_for_access_checking
            if hasattr(loc["object"], "id")
        ]

        self.sat_args["transmitterNumBuffers"] = len(self.sat_args["bufferNames"])

    def _update_image_event(self, target: "Target") -> None:
        """Create a simulator event that terminates on imaging.

        Causes the simulation to stop when a target is imaged.

        Args:
            target: Target expected to be imaged
        """
        self._disable_image_event()

        self._image_event_name = valid_func_name(f"image_{self.name}_{target.id}")
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

            def side_effect(sim):
                self.logger.info(f"imaged {target}")
                self.imaged += 1
                self.requires_retasking = True
                self.remove_imaging_line()

            self.simulator.createNewEvent(
                self._image_event_name,
                macros.sec2nano(self.fsw.fsw_rate),
                True,
                conditionFunction=lambda sim: self.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData[
                    data_index
                ]
                > current_data_level,
                actionFunction=side_effect,
                terminal=self.variable_interval,
            )
        else:
            self.simulator.eventMap[self._image_event_name].eventActive = True

    def _disable_image_event(self) -> None:
        """Turn off simulator termination due to this satellite's imaging checker."""
        if (
            self._image_event_name is not None
            and self._image_event_name in self.simulator.eventMap
        ):
            self.simulator.delete_event(self._image_event_name)

    def parse_target_selection(self, target_query: Union[int, Target, str]):
        """Identify a target from a query.

        Parses an upcoming target index, Target object, or target id.

        Args:
            target_query: Target upcoming index, object, or id.
        """
        if np.issubdtype(type(target_query), np.integer):
            target = self.find_next_opportunities(
                n=target_query + 1, types=self.target_types
            )[-1]["object"]
        elif isinstance(target_query, Target):
            target = target_query
        elif isinstance(target_query, str):
            try:
                target = [
                    target for target in self.known_targets if target.id == target_query
                ][0]
            except IndexError:
                raise ValueError(f"Target {target_query} not a known target!")
        else:
            raise TypeError(f"Invalid target_query! Cannot be a {type(target_query)}!")

        return target

    def enable_target_window(
        self, target: "Target", max_duration: Optional[float] = None
    ):
        """Enable a timed opportunity close event and a successfully imaged event.

        Args:
            target: Target to terminate the step on imaging or when out of range.
            max_duration: [s] Maximum duration to wait for imaging. If None, use the default
                behavior.
        """
        self._update_image_event(target)
        next_window = self.next_opportunities_dict(
            types=self.target_types,
            filter=self.default_access_filter,
        )[target]
        self.logger.info(
            f"{target} window enabled: {next_window[0]:.1f} to {next_window[1]:.1f}"
        )
        if max_duration is None:
            max_duration = 1e9

        def side_effect(sim):
            if np.isclose(sim.sim_time, next_window[1], atol=1e-9):
                self.missed += 1
            self.remove_imaging_line()

        self.update_timed_terminal_event(
            min(next_window[1], self.simulator.sim_time + max_duration),
            info=f"for {target} window",
            extra_actions=side_effect,
        )

    def task_target_for_imaging(
        self, target: "Target", max_duration: Optional[float] = None
    ):
        """Task the satellite to image a target.

        Args:
            target: Selected target
            max_duration: [s] Maximum duration to wait for imaging. If None, wait until
                the end of the target's access window.
        """
        msg = f"{target} tasked for imaging"
        self.logger.info(msg)
        self.fsw.action_image(target.r_LP_P, target.id)
        self.enable_target_window(target, max_duration=max_duration)
        self.draw_imaging_line(target)

    @vizard.visualize
    def draw_imaging_line(
        self, target: "Target", vizSupport=None, vizInstance=None
    ) -> None:
        """Draw a line from the satellite to the target in vizard."""
        if not hasattr(self, "target_line"):
            vizSupport.createTargetLine(
                vizInstance,
                fromBodyName=self.name,
                toBodyName=target.name,
                lineColor=self.vizard_color,
            )
            self.target_line = vizSupport.targetLineList[-1]
        self.target_line.toBodyName = target.name
        vizSupport.updateTargetLineList(vizInstance)

    @vizard.visualize
    def remove_imaging_line(self, vizSupport=None, vizInstance=None):
        """Remove the imaging line from Vizard."""
        if hasattr(self, "target_line"):
            self.target_line.toBodyName = self.name
            vizSupport.updateTargetLineList(vizInstance)
