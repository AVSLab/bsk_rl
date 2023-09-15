"""Satellite observation types can be used to add information to the observation."""

from copy import deepcopy
from typing import Any, Callable, Optional, Union

import numpy as np
from Basilisk.utilities import orbitalMotion

from bsk_rl.envs.general_satellite_tasking.scenario.satellites import (
    AccessSatellite,
    ImagingSatellite,
    Satellite,
)
from bsk_rl.envs.general_satellite_tasking.utils.functional import (
    bind,
    configurable,
    vectorize_nested_dict,
)


@configurable
class SatObservation(Satellite):
    """Base satellite subclass for composing observations."""

    def __init__(self, *args, obs_type: type = np.ndarray, **kwargs) -> None:
        """Satellite subclass for composing observations.

        Args:
            obs_type: Datatype of satellite's returned observation
            args: Passed through to satellite
            kwargs: Passed through to satellite
        """
        super().__init__(*args, **kwargs)
        self.obs_type = obs_type
        self.obs_fn_list = []
        self.obs_dict_cache = None
        self.obs_cache_time = 0.0

    @property
    def obs_dict(self):
        """Human-readable observation format.

        Cached so only computed once per timestep.
        """
        if (
            self.obs_dict_cache is None
            or self.simulator.sim_time != self.obs_cache_time
        ):
            self.obs_dict_cache = {obs.__name__: obs() for obs in self.obs_fn_list}
            self.obs_cache_time = self.simulator.sim_time
        return deepcopy(self.obs_dict_cache)

    @property
    def obs_ndarray(self):
        """Numpy vector observation format."""
        _, obs = vectorize_nested_dict(self.obs_dict)
        return obs

    @property
    def obs_array_keys(self):
        """Utility to get the keys of the obs_ndarray."""
        keys, _ = vectorize_nested_dict(self.obs_dict)
        return keys

    @property
    def obs_list(self):
        """List observation format."""
        return list(self.obs_ndarray)

    def get_obs(self) -> Union[dict, np.ndarray, list]:
        """Update the observation."""
        if self.obs_type is dict:
            return self.obs_dict
        elif self.obs_type is np.ndarray:
            return self.obs_ndarray
        elif self.obs_type is list:
            return self.obs_list
        else:
            raise ValueError(f"Invalid observation type: {self.obs_type}")

    def add_to_observation(self, obs_element: Callable) -> None:
        """Add a function to be called when constructing observations.

        Args:
            obs_element: Callable to be observed
        """
        self.obs_fn_list.append(obs_element)


@configurable
class NormdPropertyState(SatObservation):
    """Satellite subclass to add satellites properties to the observation."""

    def __init__(
        self, *args, obs_properties: list[dict[str, Any]] = [], **kwargs
    ) -> None:
        """Add a list of properties to the satellite observation.

        Args:
            obs_properties: List of properties that can be found in fsw or dynamics that
                are to be appended to the the observation. Properties are optionally
                normalized by some factor. Specified in the form

                :code-block: python

                    [dict(prop="prop_name", module="fsw"/"dynamics"/None, norm=1.0)]

                If module is not specified or None, the source of the property is
                inferred. If norm is not specified, it is set to 1.0 (no normalization).
            args: Passed through to satellite
            kwargs: Passed through to satellite
        """
        super().__init__(*args, **kwargs)

        for obs_prop in obs_properties:
            self.add_prop_function(**obs_prop)

    def add_prop_function(
        self, prop: str, module: Optional[str] = None, norm: float = 1.0
    ):
        """Add a property to the observation.

        Args:
            prop: Property to query
            module: Module (dynamics or fsw) that holds the property. Can be inferred.
            norm: Value to normalize property by. Defaults to 1.0.
        """
        if module is not None:

            def prop_fn(self):
                return np.array(getattr(getattr(self, module), prop)) / norm

        else:

            def prop_fn(self):
                for module in ["dynamics", "fsw"]:
                    if hasattr(getattr(self, module), prop):
                        return np.array(getattr(getattr(self, module), prop)) / norm
                else:
                    raise AttributeError(f"Property {prop} not found")

        prop_fn.__name__ = prop
        if norm != 1:
            prop_fn.__name__ += "_normd"

        self.add_to_observation(bind(self, prop_fn, prop_fn.__name__ + "_bound"))


@configurable
class TimeState(SatObservation):
    """Satellite subclass to add simulation time to the observation."""

    def __init__(self, *args, normalization_time: Optional[float] = None, **kwargs):
        """Add the sim time to the observation state.

        Automatically normalizes to the sim duration.

        Args:
            normalization_time: Time to normalize by. If None, is set to simulation
                duration
            args: Passed through to satellite
            kwargs: Passed through to satellite
        """
        super().__init__(*args, **kwargs)
        self.normalization_time = normalization_time
        self.add_to_observation(self.normalized_time)

    def reset_post_sim(self):
        """Autodetect normalization time."""
        super().reset_post_sim()
        if self.normalization_time is None:
            self.normalization_time = self.simulator.time_limit

    def normalized_time(self):
        """Return time normalized by normalization_time."""
        assert self.normalization_time is not None
        return self.simulator.sim_time / self.normalization_time


@configurable
class TargetState(SatObservation, ImagingSatellite):
    """Satellite subclass to add upcoming target information to the observation."""

    def __init__(
        self,
        *args,
        n_ahead_observe: int = 1,
        target_properties: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ):
        """Add information about upcoming targets to the observation state.

        Args:
            n_ahead_observe: Number of upcoming targets to consider.
            target_properties: List of properties to include in the observation in the
                format [dict(prop="prop_name", norm=norm)]. If norm is not specified, it
                is set to 1.0 (no normalization). Properties to choose from:
                * priority
                * location
                * window_open
                * window_mid
                * window_close
            args: Passed through to satellite
            kwargs: Passed through to satellite
        """
        super().__init__(*args, n_ahead_observe=n_ahead_observe, **kwargs)
        if target_properties is None:
            target_properties = [
                dict(prop="priority"),
                dict(prop="location", norm=orbitalMotion.REQ_EARTH * 1e3),
            ]
        if "location_norm" in kwargs:
            self.logger.warning(
                "location_norm is ignored and should be specified in target_properties"
            )  # pragma: no cover
        self.n_ahead_observe = int(n_ahead_observe)
        self.target_obs_generator(target_properties)

    def target_obs_generator(self, target_properties):
        """Generate the target_obs function.

        Generates the observation function from the target_properties spec and add it
        to the observation.
        """

        def target_obs(self):
            obs = {}
            for i, opportunity in enumerate(
                self.find_next_opportunities(
                    n=self.n_ahead_observe,
                    filter=self._get_imaged_filter(),
                    types="target",
                )
            ):
                props = {}
                for prop_spec in target_properties:
                    name = prop_spec["prop"]
                    norm = prop_spec.get("norm", 1.0)
                    if name == "priority":
                        value = opportunity["target"].priority
                    elif name == "location":
                        value = opportunity["target"].location
                    elif name == "window_open":
                        value = opportunity["window"][0] - self.simulator.sim_time
                    elif name == "window_mid":
                        value = sum(opportunity["window"]) / 2 - self.simulator.sim_time
                    elif name == "window_close":
                        value = opportunity["window"][1] - self.simulator.sim_time
                    else:
                        raise ValueError(
                            f"Invalid target property: {prop_spec['prop']}"
                        )
                    if norm != 1.0:
                        name += "_normd"
                    props[name] = value / norm
                obs[f"target_{i}"] = props
            return obs

        self.add_to_observation(bind(self, target_obs, "target_obs"))


@configurable
class EclipseState(SatObservation):
    """Satellite subclass to add upcoming eclipse information to the observation."""

    def __init__(self, *args, orbit_period=5700, **kwargs):
        """Add a tuple of the orbit-normalized next orbit start and end.

        Args:
            orbit_period: Normalization factor for eclipse time.
            args: Passed through to satellite
            kwargs: Passed through to satellite
        """
        super().__init__(*args, **kwargs)
        self.orbit_period_eclipse_norm = orbit_period
        self.add_to_observation(self.eclipse_state)

    def eclipse_state(self):
        """Return tuple of normalized next eclipse start and end."""
        eclipse_start, eclipse_end = self.trajectory.next_eclipse(
            self.simulator.sim_time
        )
        return [
            (eclipse_start - self.simulator.sim_time) / self.orbit_period_eclipse_norm,
            (eclipse_end - self.simulator.sim_time) / self.orbit_period_eclipse_norm,
        ]


@configurable
class GroundStationState(SatObservation, AccessSatellite):
    """Satellite subclass to add ground station information to the observation."""

    def __init__(
        self,
        *args,
        n_ahead_observe_downlinks: int = 1,
        downlink_window_properties: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ):
        """Add information about downlink opportunities to the observation state.

        Args:
            n_ahead_observe_downlinks: Number of upcoming downlink opportunities to
                consider.
            downlink_window_properties: List of properties to include in the observation
                in the format [dict(prop="prop_name", norm=norm)]. If norm is not
                specified, it is set to 1.0 (no normalization). Properties to choose
                from:
                * location
                * window_open
                * window_mid
                * window_close
            args: Passed through to satellite
            kwargs: Passed through to satellite
        """
        super().__init__(*args, **kwargs)
        if downlink_window_properties is None:
            downlink_window_properties = [
                dict(prop="window_open", norm=5700),
                dict(prop="window_close", norm=5700),
            ]
        self.ground_station_obs_generator(
            downlink_window_properties, n_ahead_observe_downlinks
        )

    def reset_post_sim(self) -> None:
        """Add downlink ground stations to be considered by the access checker."""
        for ground_station in self.simulator.environment.groundStations:
            self.add_location_for_access_checking(
                object=ground_station.ModelTag,
                location=np.array(ground_station.r_LP_P_Init).flatten(),
                min_elev=ground_station.minimumElevation,
                type="ground_station",
            )
        super().reset_post_sim()

    def ground_station_obs_generator(
        self,
        downlink_window_properties: list[dict[str, Any]],
        n_ahead_observe_downlinks: int,
    ) -> None:
        """Generate the ground_station_obs function.

        Generates an obs function from the downlink_window_properties spec and adds it
        to the observation.
        """

        def ground_station_obs(self):
            obs = {}
            for i, opportunity in enumerate(
                self.find_next_opportunities(
                    n=n_ahead_observe_downlinks, types="ground_station"
                )
            ):
                props = {}
                for prop_spec in downlink_window_properties:
                    name = prop_spec["prop"]
                    norm = prop_spec.get("norm", 1.0)
                    if name == "location":
                        value = opportunity["location"]
                    elif name == "window_open":
                        value = opportunity["window"][0] - self.simulator.sim_time
                    elif name == "window_mid":
                        value = sum(opportunity["window"]) / 2 - self.simulator.sim_time
                    elif name == "window_close":
                        value = opportunity["window"][1] - self.simulator.sim_time
                    else:
                        raise ValueError(
                            f"Invalid ground station property: {prop_spec['prop']}"
                        )
                    if norm != 1.0:
                        name += "_normd"
                    props[name] = value / norm
                obs[f"ground_station_{i}"] = props
            return obs

        self.add_to_observation(bind(self, ground_station_obs, "ground_station_obs"))
