"""Satellite observation types can be used to add information to the observation.

:class:`Observation` provides an interface for creating new observation types. To
configure the observation, set the ``observation_spec`` attribute of a
:class:`~bsk_rl.env.scenario.satellites.Satellite` subclass. For example:

.. code-block:: python

    class MyObservationSatellite(Satellite):
        observation_spec = [
            SatProperties(
                dict(prop="r_BN_P", module="dynamics", norm=REQ_EARTH * 1e3),
                dict(prop="v_BN_P", module="dynamics", norm=7616.5),
            ),
            obs.TargetProperties(
                dict(prop="priority"),
                dict(prop="location", norm=REQ_EARTH * 1e3),
                n_ahead_observe=16,
            ),
            obs.Time(),
        ]

The format of the observation can setting the ``obs_type`` attribute of the
:class:`~bsk_rl.env.scenario.satellites.Satellite`. The default is ``np.ndarray``, but
it can also be set to a human-readable ``dict`` or a ``list``.

Some commonly used observations are provided:

* :class:`SatProperties` - Add arbitrary ``dynamics`` and ``fsw`` properties.
* :class:`Time` - Add simulation time to the observation.
* :class:`TargetProperties` - Add information about upcoming targets or other ground access points to the observation.
* :class:`Eclipse` - Add a tuple of the next orbit start and end.
"""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from gymnasium import spaces

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.env.types import Satellite, Simulator

import numpy as np
from Basilisk.utilities import orbitalMotion

from bsk_rl.utils.functional import vectorize_nested_dict

logger = logging.getLogger(__name__)


def obs_dict_to_space(obs_dict):
    """Convert an observation dictionary to a gym space.

    Args:
        obs_dict: Observation dictionary

    Returns:
        gym.Space: Observation space

    :meta private:
    """
    if isinstance(obs_dict, dict):
        return spaces.Dict(
            {key: obs_dict_to_space(value) for key, value in obs_dict.items()}
        )
    elif isinstance(obs_dict, list):
        return spaces.Box(
            low=-1e16, high=1e16, shape=(len(obs_dict),), dtype=np.float64
        )
    elif isinstance(obs_dict, (float, int)):
        return spaces.Box(low=-1e16, high=1e16, shape=(1,), dtype=np.float64)
    else:
        return spaces.Box(low=-1e16, high=1e16, shape=obs_dict.shape, dtype=np.float64)


class ObservationBuilder:
    """:meta private:"""

    def __init__(self, satellite: "Satellite", obs_type: type = np.ndarray) -> None:
        """Satellite subclass for composing observations.

        Args:
            satellite: Satellite to observe
            obs_type: Datatype of satellite's returned observation. Can be ``np.ndarray``
                (default), ``dict``, or ``list``.
        """
        self.obs_type = obs_type
        self.obs_dict_cache = None
        self.obs_cache_time = 0.0
        self.satellite = satellite
        self.simulator: "Simulator"
        self.observation_spec = deepcopy(self.satellite.observation_spec)
        name_counts = {}
        for obs in self.observation_spec:
            if obs.name in name_counts:
                name_counts[obs.name] += 1
                obs.name += f"_{name_counts[obs.name]}"
            else:
                name_counts[obs.name] = 1
            obs.link_satellite(self.satellite)

    def reset_post_sim(self) -> None:
        """Perform any once-per-episode setup."""
        self.simulator = self.satellite.simulator  # already a proxy
        self.obs_dict_cache = None
        for obs in self.observation_spec:
            obs.link_simulator(self.simulator)  # already a proxy
            obs.reset_post_sim()

    def obs_dict(self) -> dict[str, Any]:
        """Human-readable observation format.

        Cached so only computed once per timestep.
        """
        if (
            self.obs_dict_cache is None
            or self.simulator.sim_time != self.obs_cache_time
        ):
            self.obs_dict_cache = {
                obs.name: obs.get_obs() for obs in self.observation_spec
            }
            self.obs_cache_time = self.simulator.sim_time
        return deepcopy(self.obs_dict_cache)

    def obs_ndarray(self) -> np.ndarray:
        """Numpy vector observation format."""
        _, obs = vectorize_nested_dict(self.obs_dict())
        return obs

    def obs_array_keys(self) -> list[str]:
        """Get the keys of the obs_ndarray."""
        keys, _ = vectorize_nested_dict(self.obs_dict())
        return keys

    def obs_list(self) -> list:
        """List observation format."""
        return list(self.obs_ndarray())

    def get_obs(self) -> Union[dict, np.ndarray, list]:
        """Update the observation."""
        if self.obs_type is dict:
            return self.obs_dict()
        elif self.obs_type is np.ndarray:
            return self.obs_ndarray()
        elif self.obs_type is list:
            return self.obs_list()
        else:
            raise ValueError(f"Invalid observation type: {self.obs_type}")

    @property
    def observation_space(self) -> spaces.Space:
        """Space of the observation."""
        obs = self.get_obs()
        if isinstance(obs, (list, np.ndarray)):
            return spaces.Box(low=-1e16, high=1e16, shape=obs.shape, dtype=np.float64)
        elif isinstance(obs, dict):
            return obs_dict_to_space(obs)
        else:
            raise ValueError(f"Invalid observation type: {self.obs_type}")

    @property
    def observation_description(self) -> Any:
        """Human-interpretable description of observation space."""
        return self.obs_array_keys()


class Observation(ABC):
    """Base observations class."""

    def __init__(self, name: str = "obs") -> None:
        """Construct an observation.

        Args:
            name: Name of the observation.
        """
        self.name = name
        self.satellite: "Satellite"
        self.simulator: "Simulator"

    def link_satellite(self, satellite: "Satellite") -> None:
        """Link the observation to a satellite.

        Args:
            satellite: Satellite to link to

        :meta private:
        """
        self.satellite = satellite  # already a proxy

    def link_simulator(self, simulator: "Simulator") -> None:
        """Link the observation to a simulator.

        Args:
            simulator: Simulator to link to

        :meta private:
        """
        self.simulator = simulator  # already a proxy

    def reset_post_sim(self) -> None:  # pragma: no cover
        """Perform any once-per-episode setup."""
        pass

    @abstractmethod  # pragma: no cover
    def get_obs(self) -> Any:
        """Return the observation."""
        pass


class SatProperties(Observation):
    """Add arbitrary `dynamics` and `fsw` ."""

    def __init__(
        self, *obs_properties: dict[str, Union[str, float]], name="sat_props"
    ) -> None:
        """Include properties from ``fsw`` and ``dynamics`` in the observation.

        For each desired property, a dictionary specifying the property name and settings
        is passed. For example, to query the position and velocity of the satellite, the
        following would be used:

        .. code-block:: python

            SatProperties(
                dict(prop="r_BN_P", module="dynamics", norm=REQ_EARTH * 1e3),
                dict(prop="v_BN_P", module="dynamics", norm=7616.5, name="velocity"),
            ),


        Args:
            obs_properties: Property that can be found in fsw or dynamics that
                are to be appended to the the observation. Properties are optionally
                normalized by some factor. Each observation is a dictionary with the keys:

                * ``prop``: Name of property in ``fsw`` and ``dynamics`` to query
                * ``module`` `optional`: Module (dynamics or fsw) that holds the property. Can be inferred if ``None``.
                * ``norm`` `optional`: Value to normalize property by. Defaults to 1.0.
                * ``name`` `optional`: Name of the observation element. Defaults to the value of ``prop``.
            name: Name of the observation.

        """
        super().__init__(name=name)
        for obs_property in obs_properties:
            if "norm" not in obs_property:
                obs_property["norm"] = 1.0
            if "name" not in obs_property:
                obs_property["name"] = obs_property["prop"]
                if obs_property["norm"] != 1.0:
                    obs_property["name"] += "_normd"

        self.obs_properties = obs_properties

    def reset_post_sim(self) -> None:
        """If necessary, automatically determine property location.

        :meta private:
        """
        for obs_property in self.obs_properties:
            if "module" not in obs_property:
                for module in ["dynamics", "fsw"]:
                    if hasattr(getattr(self.satellite, module), obs_property["prop"]):
                        obs_property["module"] = module
                        break
                else:
                    raise AttributeError(f"Property {obs_property['prop']} not found")

    def get_obs(self) -> dict[str, Any]:
        """Return the observation.

        :meta private:
        """
        obs = {}
        for obs_property in self.obs_properties:
            prop = obs_property["prop"]
            module = obs_property["module"]
            norm = obs_property["norm"]
            value = getattr(getattr(self.satellite, module), prop)
            if isinstance(value, list):
                value = np.array(value)
            obs[obs_property["name"]] = value / norm
        return obs


class Time(Observation):
    def __init__(self, norm=None, name="time"):
        """Include the simulation time in the observation.

        Args:
            norm: Time to normalize by. If ``None``, the time is normalized by the simulation time limit.
            name: Name of the observation.
        """
        super().__init__(name=name)
        self.norm = norm

    def reset_post_sim(self) -> None:
        """Autodetect normalization time.

        :meta private:
        """
        if self.norm is None:
            self._norm = self.simulator.time_limit
        else:
            self._norm = self.norm

    def get_obs(self) -> float:
        """Return time normalized by normalization_time.

        :meta private:
        """
        return self.simulator.sim_time / self._norm


def _target_angle(sat, opp):
    vector_target_spacecraft_P = opp["location"] - sat.dynamics.r_BN_P
    vector_target_spacecraft_P_hat = vector_target_spacecraft_P / np.linalg.norm(
        vector_target_spacecraft_P
    )
    return np.arccos(np.dot(vector_target_spacecraft_P_hat, sat.fsw.c_hat_P))


class OpportunityProperties(Observation):

    _fn_map = {
        "priority": lambda sat, opp: opp[opp["type"]].priority,
        "location": lambda sat, opp: opp["location"],
        "opportunity_open": lambda sat, opp: opp["window"][0] - sat.simulator.sim_time,
        "opportunity_mid": lambda sat, opp: sum(opp["window"]) / 2
        - sat.simulator.sim_time,
        "opportunity_close": lambda sat, opp: opp["window"][1] - sat.simulator.sim_time,
        "target_angle": _target_angle,
    }

    def __init__(
        self,
        *target_properties: dict[str, Any],
        n_ahead_observe: int,
        type="target",
        name=None,
    ):
        """Include information about upcoming access opportunities in the observation..

        For each desired property, a dictionary specifying the property name and settings
        is passed. These can include preset properties or arbitrary functions of the satellite
        and opportunity.

        .. code-block:: python

            TargetProperties(
                dict(prop="location", norm=REQ_EARTH * 1e3),
                dict(prop="double_priority", fn=lambda sat, opp: opp["target"].priority * 2.0),
                n_ahead_observe=16,
            )

        Args:
            target_properties: Property that is a function of the opportunity to be appended
                to the the observation. Properties are optionally normalized by some factor.
                Each observation is a dictionary with the keys:

                * ``name``: Name of the observation element.
                * ``fn`` `optional`: Function to calculate property, in the form ``fn(satellite, opportunity)``. If not provided, the name will be used to look up a preset function:

                    * ``priority``: Priority of the target.
                    * ``location``: Location of the target in the planet-fixed frame.
                    * ``opportunity_open``: Time until the opportunity opens.
                    * ``opportunity_mid``: Time until the opportunity midpoint.
                    * ``opportunity_close``: Time until the opportunity closes.
                    * ``target_angle``: Angle between the target and the satellite instrument direction.

                * ``norm`` `optional`: Value to normalize property by. Defaults to 1.0.

            n_ahead_observe: Number of upcoming targets to consider.
            type: The type of opportunity to consider. Can be ``target``, ``ground_station``,
                or any other type of opportunity that has been added via
                :obj:`~bsk_rl.env.scenario.satellites.AccessSatellite.add_location_for_access_checking`.
            name: Name of the observation.
        """
        if name is None:
            name = type
        super().__init__(name=name)
        self.type = type
        self.target_properties = target_properties
        for prop_spec in self.target_properties:
            if "norm" not in prop_spec:
                prop_spec["norm"] = 1.0
            if "fn" not in prop_spec:
                try:
                    prop_spec["fn"] = self._fn_map[prop_spec["prop"]]
                except KeyError:
                    raise ValueError(
                        f"Property {prop_spec['prop']} is not predefined and no `fn` was provided."
                    )

            if "name" not in prop_spec:
                prop_spec["name"] = prop_spec["prop"]
                if prop_spec["norm"] != 1.0:
                    prop_spec["name"] += "_normd"

        self.n_ahead_observe = int(n_ahead_observe)

    def reset_post_sim(self) -> None:
        """Add downlink ground stations to be considered by the access checker.

        :meta private:
        """
        if self.type == "ground_station":
            for ground_station in self.simulator.environment.groundStations:
                self.satellite.add_location_for_access_checking(
                    object=ground_station.ModelTag,
                    location=np.array(ground_station.r_LP_P_Init).flatten(),
                    min_elev=ground_station.minimumElevation,
                    type="ground_station",
                )

    def get_obs(self):
        """Iterate over property specs.

        :meta private:
        """
        from bsk_rl.env.scenario.satellites import AccessSatellite

        if not isinstance(self.satellite, AccessSatellite):
            logger.warning(
                "OpportunityProperties observation requires an AccessSatellite"
            )

        obs = {}
        for i, opportunity in enumerate(
            self.satellite.find_next_opportunities(
                n=self.n_ahead_observe,
                filter=self.satellite._get_access_filter(),
                types=self.type,
            )
        ):
            props = {}
            for prop_spec in self.target_properties:
                name = prop_spec["prop"]
                norm = prop_spec["norm"]
                if norm != 1.0:
                    name += "_normd"
                value = prop_spec["fn"](self.satellite, opportunity)
                props[name] = value / norm
            obs[f"{self.name}_{i}"] = props
        return obs


class Eclipse(Observation):
    def __init__(self, norm=5700.0, name="eclipse"):
        """Include a tuple of the next eclipse start and end times in the observation.

        Args:
            norm: Value to normalize by.
            name: Name of the observation.
        """
        super().__init__(name=name)
        self.norm = norm

    def get_obs(self):
        """Return tuple of normalized next eclipse start and end.

        :meta private:
        """
        eclipse_start, eclipse_end = self.satellite.trajectory.next_eclipse(
            self.simulator.sim_time
        )
        return [
            (eclipse_start - self.simulator.sim_time) / self.norm,
            (eclipse_end - self.simulator.sim_time) / self.norm,
        ]
