from copy import deepcopy
from typing import Any, Callable, Optional, Union

import numpy as np
from Basilisk.utilities import orbitalMotion

from bsk_rl.envs.GeneralSatelliteTasking.scenario.satellites import (
    ImagingSatellite,
    Satellite,
)
from bsk_rl.envs.GeneralSatelliteTasking.utils.functional import (
    bind,
    configurable,
    vectorize_nested_dict,
)


@configurable
class SatObservation(Satellite):
    def __init__(self, *args, obs_type: type = np.ndarray, **kwargs) -> None:
        """Satellite subclass for composing observations.

        Args:
            obs_type: Datatype of satellite's returned observation
        """
        super().__init__(*args, **kwargs)
        self.obs_type = obs_type
        self.obs_fn_list = []
        self.obs_dict_cache = None
        self.obs_cache_time = 0.0

    @property
    def obs_dict(self):
        """Human-readable observation format. Cached so only computed once per
        timestep."""
        if (
            self.obs_dict_cache is None
            or self.simulator.sim_time != self.obs_cache_time
        ):
            self.obs_dict_cache = {obs.__name__: obs() for obs in self.obs_fn_list}
            self.obs_cache_time = self.simulator.sim_time
        return deepcopy(self.obs_dict_cache)

    @property
    def obs_ndarray(self):
        """Numpy vector observation format"""
        return vectorize_nested_dict(self.obs_dict)

    @property
    def obs_list(self):
        """List observation format"""
        return list(self.obs_ndarray)

    def get_obs(self) -> Union[dict, np.ndarray, list]:
        """Update the observation"""
        if self.obs_type is dict:
            return self.obs_dict
        elif self.obs_type is np.ndarray:
            return self.obs_ndarray
        elif self.obs_list is list:
            return self.obs_list
        else:
            raise ValueError(f"Invalid observation type: {self.obs_type}")

    def add_to_observation(self, obs_element: Callable) -> None:
        """Add a function to be called when constructing observations

        Args:
            obs_element: Callable to be observed
        """
        self.obs_fn_list.append(obs_element)


@configurable
class NormdPropertyState(SatObservation):
    def __init__(
        self, *args, obs_properties: list[dict[str, Any]] = [], **kwargs
    ) -> None:
        """Add a list of properties to the satellite observation.

        Args:
            obs_properties: List of properties that can be found in fsw or dynamics that
                are to be appended to the the observation. Properties are optionally
                normalized by some factor. Specified in
                    [dict(prop="prop_name", module="fsw"/"dynamics"/None, norm=1.0)]
                If module is not specified or None, the source of the property is
                inferred. If norm is not specified, it is set to 1.0 (no normalization).
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

        Returns:
            _type_: _description_
        """
        if module is not None:

            def prop_fn(self):
                return getattr(getattr(self, module), prop) / norm

        else:

            def prop_fn(self):
                for module in ["dynamics", "fsw"]:
                    if hasattr(getattr(self, module), prop):
                        return getattr(getattr(self, module), prop) / norm

        prop_fn.__name__ = prop
        if norm != 1:
            prop_fn.__name__ += "_normd"

        self.add_to_observation(bind(self, prop_fn, prop_fn.__name__ + "_bound"))


@configurable
class TimeState(SatObservation):
    def __init__(self, *args, normalization_time: Optional[float] = None, **kwargs):
        """Adds the sim time to the observation state. Automatically normalizes to the
        sim duration.

        Args:
            normalization_time: Time to normalize by. If None, is set to simulation
                duration
        """

        super().__init__(*args, **kwargs)
        self.normalization_time = normalization_time
        self.add_to_observation(self.normalized_time)

    def reset_post_sim(self):
        """Autodetect normalization time"""
        super().reset_post_sim()
        if self.normalization_time is None:
            self.normalization_time = self.simulator.time_limit

    def normalized_time(self):
        assert self.normalization_time is not None
        return self.simulator.sim_time / self.normalization_time


@configurable
class TargetState(SatObservation, ImagingSatellite):
    def __init__(
        self,
        *args,
        n_ahead_observe: int = 1,
        location_norm: float = orbitalMotion.REQ_EARTH * 1e3,
        **kwargs,
    ):
        """Adds information about upcoming targets to the observation state.

        Args:
            n_ahead_observe: Number of upcoming targets to consider.
        """
        super().__init__(*args, n_ahead_observe=n_ahead_observe, **kwargs)
        self.n_ahead_observe = int(n_ahead_observe)
        self.location_norm = location_norm
        self.add_to_observation(self.target_obs)

    def target_obs(self):
        obs = {}
        for i, target in enumerate(self.upcoming_targets(self.n_ahead_observe)):
            obs[f"tgt_value_{i}"] = target.priority
            loc_name = f"tgt_loc_{i}"
            if loc_name != 1.0:
                loc_name += "_normd"
            obs[loc_name] = target.location / self.location_norm
        return obs


@configurable
class EclipseState(SatObservation):
    def __init__(self, *args, orbit_period=5700, **kwargs):
        """Adds a tuple of the orbit-normalized next orbit start and end.

        Args:
            orbit_period: Normalization factor for eclipse time.
        """

        super().__init__(*args, **kwargs)
        self.orbit_period_eclipse_norm = orbit_period
        self.add_to_observation(self.eclipse_state)

    def eclipse_state(self):
        eclipse_start, eclipse_end = self.trajectory.next_eclipse(
            self.simulator.sim_time
        )
        return [
            (eclipse_start - self.simulator.sim_time) / self.orbit_period_eclipse_norm,
            (eclipse_end - self.simulator.sim_time) / self.orbit_period_eclipse_norm,
        ]
