"""Classes for composing observations for a satellite."""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import numpy as np
from Basilisk.utilities import orbitalMotion
from gymnasium import spaces

from bsk_rl.utils.functional import vectorize_nested_dict
from bsk_rl.utils.orbital import rv2HN
from bsk_rl.utils.profiling import env_flag, profile_section

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite
    from bsk_rl.sim import Simulator


logger = logging.getLogger(__name__)


def nested_obs_to_space(obs_dict):
    """Convert a nested observation dictionary to a gym space.

    Args:
        obs_dict: Observation dictionary

    Returns:
        gym.Space: Observation space

    :meta private:
    """
    if isinstance(obs_dict, dict):
        return spaces.Dict(
            {key: nested_obs_to_space(value) for key, value in obs_dict.items()}
        )
    elif isinstance(obs_dict, list):
        return spaces.Box(
            low=-1e16, high=1e16, shape=(len(obs_dict),), dtype=np.float64
        )
    elif isinstance(obs_dict, (float, int)):
        return spaces.Box(low=-1e16, high=1e16, shape=(1,), dtype=np.float64)
    elif isinstance(obs_dict, np.ndarray):
        return spaces.Box(low=-1e16, high=1e16, shape=obs_dict.shape, dtype=np.float64)
    else:
        raise TypeError(f"Cannot convert {obs_dict} to gym space.")


class ObservationBuilder:
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

    def reset_post_sim_init(self) -> None:
        """Perform any once-per-episode setup."""
        self.simulator = self.satellite.simulator  # already a proxy
        self.obs_dict_cache = None
        for obs in self.observation_spec:
            obs.link_simulator(self.simulator)  # already a proxy
            obs.reset_post_sim_init()

    def obs_dict(self) -> dict[str, Any]:
        """Human-readable observation format.

        Cached so only computed once per timestep.
        """
        with profile_section(self.simulator, "obs.builder.obs_dict"):
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
        with profile_section(self.simulator, "obs.builder.obs_ndarray"):
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
        return nested_obs_to_space(obs)

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

    def reset_post_sim_init(self) -> None:  # pragma: no cover
        """Perform any once-per-episode setup."""
        pass

    @abstractmethod  # pragma: no cover
    def get_obs(self) -> Any:
        """Return the observation."""
        pass


class SatProperties(Observation):
    """Add arbitrary `dynamics` and `fsw` ."""

    def __init__(self, *obs_properties: dict[str, Any], name="sat_props") -> None:
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
                * ``fn`` `optional`: Alternatively, call a function that takes the satellite as an argument.
            name: Name of the observation.

        """
        super().__init__(name=name)
        for obs_property in obs_properties:
            for key in obs_property:
                if key not in ["prop", "module", "norm", "name", "fn"]:
                    raise ValueError(f"Invalid property key: {key}")
            if "norm" not in obs_property:
                obs_property["norm"] = 1.0
            if "name" not in obs_property:
                obs_property["name"] = obs_property["prop"]
                if obs_property["norm"] != 1.0:
                    obs_property["name"] += "_normd"

        self.obs_properties = obs_properties

    def reset_post_sim_init(self) -> None:
        """If necessary, automatically determine property location.

        :meta private:
        """
        for obs_property in self.obs_properties:
            if "module" not in obs_property and "fn" not in obs_property:
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
            if "fn" in obs_property:
                value = obs_property["fn"](self.satellite)
            else:
                module = obs_property["module"]
                value = getattr(getattr(self.satellite, module), prop)
            if isinstance(value, list):
                value = np.array(value)
            norm = obs_property["norm"]
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

    def reset_post_sim_init(self) -> None:
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
    vector_target_spacecraft_P = opp["r_LP_P"] - sat.dynamics.r_BN_P
    vector_target_spacecraft_P_hat = vector_target_spacecraft_P / np.linalg.norm(
        vector_target_spacecraft_P
    )
    return np.arccos(np.dot(vector_target_spacecraft_P_hat, sat.fsw.c_hat_P))


def _target_angle_rate(sat, opp):
    r_BN_P = sat.dynamics.v_BN_P
    v_BN_P = sat.dynamics.v_BN_P
    r_LP_P = opp["object"].r_LP_P
    omega_BP_P = sat.dynamics.omega_BP_P
    omega_CP_ref = (
        omega_BP_P
        - np.cross(v_BN_P, r_LP_P - r_BN_P) / np.linalg.norm(r_LP_P - r_BN_P) ** 2
    )
    return np.linalg.norm(omega_CP_ref)


def _r_LB_H(sat, opp):
    r_LP_P = opp["object"].r_LP_P
    r_BN_N = sat.dynamics.r_BN_N
    r_TB_N = sat.simulator.world.PN.T @ r_LP_P - r_BN_N
    HN = rv2HN(sat.dynamics.r_BN_N, sat.dynamics.v_BN_N)
    return HN @ r_TB_N


def s_hat_H(sat):
    """Dimensionless sun unit vector from the spacecraft in the Hill frame."""
    r_SN_N = (
        sat.simulator.world.gravFactory.spiceObject.planetStateOutMsgs[
            sat.simulator.world.sun_index
        ]
        .read()
        .PositionVector
    )
    r_BN_N = sat.dynamics.r_BN_N
    r_SB_N = np.array(r_SN_N) - np.array(r_BN_N)
    r_SB_H = rv2HN(r_BN_N, sat.dynamics.v_BN_N) @ r_SB_N
    norm = np.linalg.norm(r_SB_H)
    if norm == 0.0:
        return np.zeros(3)
    return r_SB_H / norm


class OpportunityProperties(Observation):
    _fn_map = {
        "priority": lambda sat, opp: opp["object"].priority,
        "r_LP_P": lambda sat, opp: opp["r_LP_P"],
        "r_LB_H": _r_LB_H,
        "opportunity_open": lambda sat, opp: opp["window"][0] - sat.simulator.sim_time,
        "opportunity_mid": lambda sat, opp: sum(opp["window"]) / 2
        - sat.simulator.sim_time,
        "opportunity_close": lambda sat, opp: opp["window"][1] - sat.simulator.sim_time,
        "target_angle": _target_angle,
        "target_angle_rate": _target_angle_rate,
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

            OpportunityProperties(
                dict(prop="r_LP_P", norm=REQ_EARTH * 1e3),
                dict(prop="double_priority", fn=lambda sat, opp: opp["target"].priority * 2.0),
                n_ahead_observe=16,
            )

        Args:
            target_properties: Property that is a function of the opportunity to be appended
                to the the observation. Properties are optionally normalized by some factor.
                Each observation is a dictionary with the keys:

                * ``name`` `optional`: Name of the observation element.
                * ``fn`` `optional`: Function to calculate property, in the form ``fn(satellite, opportunity)``.
                  If not provided, the key ``prop`` will be used to look up a preset function:

                    * ``priority``: Priority of the target.
                    * ``r_LP_P``: Location of the target in the planet-fixed frame.
                    * ``r_LB_H``: Location of the target in the Hill frame.
                    * ``opportunity_open``: Time until the opportunity opens.
                    * ``opportunity_mid``: Time until the opportunity midpoint.
                    * ``opportunity_close``: Time until the opportunity closes.
                    * ``target_angle``: Angle between the target and the satellite instrument direction.
                    * ``target_angle_rate``: Rate difference between the target pointing frame and the body frame.

                * ``norm`` `optional`: Value to normalize property by. Defaults to 1.0.

            n_ahead_observe: Number of upcoming targets to consider.
            type: The type of opportunity to consider. Can be ``target``, ``ground_station``,
                or any other type of opportunity that has been added via
                :obj:`~bsk_rl.sats.AccessSatellite.add_location_for_access_checking`.
            name: Name of the observation.
        """
        if name is None:
            name = type
        super().__init__(name=name)
        self.type = type
        self.target_properties = target_properties
        for i, prop_spec in enumerate(self.target_properties):
            for key in prop_spec:
                if key not in ["fn", "norm", "name", "prop"]:
                    raise ValueError(f"Invalid property key: {key}")

            if "norm" not in prop_spec:
                prop_spec["norm"] = 1.0

            # Determine observation function
            if "fn" not in prop_spec:
                try:
                    prop_spec["fn"] = self._fn_map[prop_spec["prop"]]
                except KeyError:
                    raise ValueError(
                        f"Property prop={prop_spec['prop']} is not predefined and no `fn` was provided."
                    )
            else:
                if "prop" in prop_spec and prop_spec["prop"] in self._fn_map:
                    logger.warning(
                        f"Ignoring default function for `{prop_spec['prop']}` when `fn` is provided."
                    )

            # Determine best name
            if "name" not in prop_spec:
                if "prop" in prop_spec:
                    prop_spec["name"] = prop_spec["prop"]
                else:
                    prop_spec["name"] = f"prop_{i}"

                if prop_spec["norm"] != 1.0:
                    prop_spec["name"] += "_normd"

        self.n_ahead_observe = int(n_ahead_observe)

    def get_obs(self):
        """Iterate over property specs.

        :meta private:
        """
        from bsk_rl.sats import AccessSatellite

        if not isinstance(self.satellite, AccessSatellite):
            logger.warning(
                "OpportunityProperties observation requires an AccessSatellite"
            )

        obs = {}

        for i, opportunity in enumerate(
            self.satellite.find_next_opportunities(
                n=self.n_ahead_observe,
                types=self.type,
                pad=True,
            )
        ):
            props = {}
            for prop_spec in self.target_properties:
                name = prop_spec["name"]
                norm = prop_spec["norm"]
                value = prop_spec["fn"](self.satellite, opportunity)
                props[name] = value / norm
            obs[f"{self.name}_{i}"] = props
        return obs

###ADD a class similar to OpportunityProperties and change that loop of find_next_opportunities
def _relative_position(sat, opp):
    if "_polaris_cache" in opp:
        return opp["_polaris_cache"]["rel_pos_N"]
    sat_pos = np.array(sat.dynamics.r_BN_N)
    target_pos = np.array(opp["object"].target_spacecraft.dynamics.r_BN_N)
    los_vector = target_pos - sat_pos
    return los_vector

def _relative_position_H(sat, opp):
    if "_polaris_cache" in opp:
        return opp["_polaris_cache"]["rel_pos_H"]
    sat_pos = np.array(sat.dynamics.r_BN_N)
    target_pos = np.array(opp["object"].target_spacecraft.dynamics.r_BN_N)
    los_vector = target_pos - sat_pos
    HN = rv2HN(sat.dynamics.r_BN_N, sat.dynamics.v_BN_N)
    return HN @ los_vector

def _r_BN_H(sat, opp):
    if "_polaris_cache" in opp:
        return opp["_polaris_cache"]["target_pos_H"]
    r_BN_N = opp["object"].target_spacecraft.dynamics.r_BN_N
    HN = rv2HN(sat.dynamics.r_BN_N, sat.dynamics.v_BN_N)
    return HN @ r_BN_N


def _target_r_BN_N(sat, opp):
    if "_polaris_cache" in opp:
        return opp["_polaris_cache"]["target_pos_N"]
    return opp["object"].target_spacecraft.dynamics.r_BN_N


def _relative_velocity_H(sat, opp):
    """Relative target velocity in m/s, expressed in the scanner Hill frame."""
    if "_polaris_cache" in opp:
        return opp["_polaris_cache"]["rel_vel_H"]
    sat_vel = np.array(sat.dynamics.v_BN_N, dtype=float)
    target_vel = np.array(opp["object"].target_spacecraft.dynamics.v_BN_N, dtype=float)
    HN = rv2HN(sat.dynamics.r_BN_N, sat.dynamics.v_BN_N)
    return HN @ (target_vel - sat_vel)

def _target_elevation_angle(sat, opp):
    if "_polaris_cache" in opp:
        return opp["_polaris_cache"]["target_elevation_angle"]
    sat_pos = np.array(sat.dynamics.r_BN_N)
    target_pos = np.array(opp["object"].target_spacecraft.dynamics.r_BN_N)
    los_vector = target_pos - sat_pos
    los_unit = los_vector / np.linalg.norm(los_vector)
    zenith = sat_pos / np.linalg.norm(sat_pos)
    elevation_rad = np.arcsin(np.clip(np.dot(los_unit, zenith), -1.0, 1.0))
    elevation_deg = np.degrees(elevation_rad)
    return elevation_deg

def _angle_to_target(sat, opp):
    if "_polaris_cache" in opp:
        return opp["_polaris_cache"]["angle_to_target"]
    vector_target_spacecraft_P = opp["object"].target_spacecraft.dynamics.r_BN_P - sat.dynamics.r_BN_P
    vector_target_spacecraft_P_hat = vector_target_spacecraft_P / np.linalg.norm(
        vector_target_spacecraft_P
    )
    return np.degrees(np.arccos(np.dot(vector_target_spacecraft_P_hat, sat.fsw.c_hat_P)))

def _target_distance(sat, opp):
    if "_polaris_cache" in opp:
        return opp["_polaris_cache"]["target_distance"]
    vector_target_spacecraft_N = (
        np.array(opp["object"].target_spacecraft.dynamics.r_BN_N)
        - np.array(sat.dynamics.r_BN_N)
    )
    return np.linalg.norm(vector_target_spacecraft_N)
def _target_id_extracted(sat, opp):
    id= int((opp["object"].target_spacecraft.id).strip("target_"))
    return id

def _eligible_targets_now(sat, known_targets):
    """Return currently image-eligible targets from the datastore lifecycle state."""
    data_obj = sat.data_store.data
    sim_time = float(sat.simulator.sim_time)
    if hasattr(data_obj, "eligible_targets"):
        return data_obj.eligible_targets(sim_time, known_targets)

    # Backward-compatible fallback for legacy data objects.
    imaged_targets = getattr(data_obj, "imaged", [])
    imaged_ids = {tgt.id for tgt in imaged_targets}
    return [tgt for tgt in known_targets if tgt.id not in imaged_ids]

def _target_imaged(sat, opp):
    target = opp["object"]
    data_obj = sat.data_store.data
    sim_time = float(sat.simulator.sim_time)

    if hasattr(data_obj, "is_target_eligible"):
        # Keep feature name for compatibility; value now means "currently cooling down".
        return int(not data_obj.is_target_eligible(target, sim_time))

    imaged_targets = getattr(data_obj, "imaged", [])
    imaged_ids = {tgt.id for tgt in imaged_targets}
    return int(target.id in imaged_ids)
def _target_shadowFactor(sat, opp):
    if "_polaris_cache" in opp:
        return opp["_polaris_cache"]["target_shadowFactor"]
    return sat.simulator.satellites[0].dynamics.world.eclipseObject.eclipseOutMsgs[opp["object"].target_spacecraft.dynamics.eclipse_index].read().shadowFactor


def _record_dynamic_priority_candidate_access(targets, sim_time: float) -> None:
    """Record tracked targets presented in the policy candidate list."""
    seen_target_ids = set()
    for slot, target in enumerate(targets):
        if not getattr(target, "priority_event_active", False):
            continue
        if getattr(target, "priority_event_kind", "") not in {
            "HIO",
            "SHIO",
            "CONTROL",
        }:
            continue
        target_id = int(target.id)
        if target_id in seen_target_ids:
            continue
        seen_target_ids.add(target_id)
        last_time = getattr(target, "priority_event_last_candidate_log_time", None)
        if last_time == float(sim_time):
            continue
        target.priority_event_candidate_count = (
            int(getattr(target, "priority_event_candidate_count", 0)) + 1
        )
        if getattr(target, "priority_event_first_candidate_time", None) is None:
            target.priority_event_first_candidate_time = float(sim_time)
        candidate_times = getattr(target, "priority_event_candidate_times", None)
        if candidate_times is None:
            candidate_times = []
            target.priority_event_candidate_times = candidate_times
        candidate_slots = getattr(target, "priority_event_candidate_slots", None)
        if candidate_slots is None:
            candidate_slots = []
            target.priority_event_candidate_slots = candidate_slots
        candidate_times.append(float(sim_time))
        candidate_slots.append(int(slot))
        target.priority_event_last_candidate_log_time = float(sim_time)


def _record_dynamic_priority_visible_access(targets, sim_time: float) -> None:
    """Record tracked targets that are eligible and geometrically visible."""
    seen_target_ids = set()
    for target in targets:
        if not getattr(target, "priority_event_active", False):
            continue
        if getattr(target, "priority_event_kind", "") not in {
            "HIO",
            "SHIO",
            "CONTROL",
        }:
            continue
        target_id = int(target.id)
        if target_id in seen_target_ids:
            continue
        seen_target_ids.add(target_id)
        last_time = getattr(target, "priority_event_last_visible_log_time", None)
        if last_time == float(sim_time):
            continue
        target.priority_event_visible_count = (
            int(getattr(target, "priority_event_visible_count", 0)) + 1
        )
        if getattr(target, "priority_event_first_visible_time", None) is None:
            target.priority_event_first_visible_time = float(sim_time)
        visible_times = getattr(target, "priority_event_visible_times", None)
        if visible_times is None:
            visible_times = []
            target.priority_event_visible_times = visible_times
        visible_times.append(float(sim_time))
        target.priority_event_last_visible_log_time = float(sim_time)


class PolarisScTargetProperties(Observation):
    _fn_map = {
        "priority": lambda sat, opp: opp["object"].priority,
        "rel_pos_vector_r_BR_N": _relative_position,
        "rel_pos_vector_r_BR_H": _relative_position_H,
        "rel_vel_vector_v_BR_H": _relative_velocity_H,
        "r_BN_N": _target_r_BN_N,
        "r_BN_H": _r_BN_H,
        # "r_LB_H": _r_LB_H,
        # "opportunity_open": lambda sat, opp: opp["window"][0] - sat.simulator.sim_time,
        # "opportunity_mid": lambda sat, opp: sum(opp["window"]) / 2
        # - sat.simulator.sim_time,
        # "opportunity_close": lambda sat, opp: opp["window"][1] - sat.simulator.sim_time,
        # "target_angle": _target_angle,
        # "target_angle_rate": _target_angle_rate,
        "target_elevation_angle": _target_elevation_angle,
        "angle_to_target": _angle_to_target,
        "target_distance": _target_distance,
        "target_id_info": _target_id_extracted,  # lambda sat, opp: int(opp["object"].target_spacecraft.id).strip("target_"),
        "target_imaged": _target_imaged,
        "target_shadowFactor": _target_shadowFactor,

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

            OpportunityProperties(
                dict(prop="r_LP_P", norm=REQ_EARTH * 1e3),
                dict(prop="double_priority", fn=lambda sat, opp: opp["target"].priority * 2.0),
                n_ahead_observe=16,
            )

        Args:
            target_properties: Property that is a function of the opportunity to be appended
                to the observation. Properties are optionally normalized by some factor.
                Each observation is a dictionary with the keys:

                * ``name`` `optional`: Name of the observation element.
                * ``fn`` `optional`: Function to calculate property, in the form ``fn(satellite, opportunity)``.
                  If not provided, the key ``prop`` will be used to look up a preset function:

                    * ``priority``: Priority of the target.
                    * ``r_LP_P``: Location of the target in the planet-fixed frame.
                    * ``r_LB_H``: Location of the target in the Hill frame.
                    * ``opportunity_open``: Time until the opportunity opens.
                    * ``opportunity_mid``: Time until the opportunity midpoint.
                    * ``opportunity_close``: Time until the opportunity closes.
                    * ``target_angle``: Angle between the target and the satellite instrument direction.
                    * ``target_angle_rate``: Rate difference between the target pointing frame and the body frame.

                * ``norm`` `optional`: Value to normalize property by. Defaults to 1.0.

            n_ahead_observe: Number of upcoming targets to consider.
            type: The type of opportunity to consider. Can be ``target``, ``ground_station``,
                or any other type of opportunity that has been added via
                :obj:`~bsk_rl.sats.AccessSatellite.add_location_for_access_checking`.
            name: Name of the observation.
        """
        if name is None:
            name = type
        super().__init__(name=name)
        self.type = type
        self.target_properties = target_properties
        for i, prop_spec in enumerate(self.target_properties):
            for key in prop_spec:
                if key not in ["fn", "norm", "name", "prop"]:
                    raise ValueError(f"Invalid property key: {key}")

            if "norm" not in prop_spec:
                prop_spec["norm"] = 1.0

            # Determine observation function
            if "fn" not in prop_spec:
                try:
                    prop_spec["fn"] = self._fn_map[prop_spec["prop"]]
                except KeyError:
                    raise ValueError(
                        f"Property prop={prop_spec['prop']} is not predefined and no `fn` was provided."
                    )
            else:
                if "prop" in prop_spec and prop_spec["prop"] in self._fn_map:
                    logger.warning(
                        f"Ignoring default function for `{prop_spec['prop']}` when `fn` is provided."
                    )

            # Determine best name
            if "name" not in prop_spec:
                if "prop" in prop_spec:
                    prop_spec["name"] = prop_spec["prop"]
                else:
                    prop_spec["name"] = f"prop_{i}"

                if prop_spec["norm"] != 1.0:
                    prop_spec["name"] += "_normd"

        self.n_ahead_observe = int(n_ahead_observe)

    def get_obs(self):
        """Iterate over property specs.

        :meta private:
        """
        with profile_section(self.simulator, "obs.polaris_targets.total"):
            if not env_flag("BSK_RL_POLARIS_OBS_CACHE", True):
                return self._get_obs_legacy()
            return self._get_obs_cached()

    def _get_obs_legacy(self):
        """Original target observation path, retained for A/B validation."""
        scanner_pos = np.array(self.satellite.dynamics.r_BN_N)
        known_targets = self.satellite.data_store.data.known
        eligible_targets = _eligible_targets_now(self.satellite, known_targets)
        eligible_ids = {tgt.id for tgt in eligible_targets}

        target_elevations = []
        for target in eligible_targets:
            target_pos = np.array(target.target_spacecraft.dynamics.r_BN_N)
            los_vector = target_pos - scanner_pos
            los_unit = los_vector / np.linalg.norm(los_vector)

            zenith = scanner_pos / np.linalg.norm(scanner_pos)
            cos_angle = np.clip(np.dot(los_unit, zenith), -1.0, 1.0)
            elevation_rad = np.arcsin(cos_angle)
            elev = np.degrees(elevation_rad)
            target_elevations.append((target, elev))

        visible_eligible_targets = [
            (tgt, elev)
            for tgt, elev in target_elevations
            if -21.0 <= elev <= 90.0 and tgt.id in eligible_ids
        ]
        _record_dynamic_priority_visible_access(
            [tgt for tgt, _ in visible_eligible_targets],
            float(self.satellite.simulator.sim_time),
        )

        visible_eligible_targets.sort(key=lambda x: x[1])

        num_actions = self.n_ahead_observe
        final_targets = [tgt for tgt, _ in visible_eligible_targets[:num_actions]]

        if len(final_targets) < num_actions:
            remaining = num_actions - len(final_targets)
            selected_ids = {tgt.id for tgt in final_targets}
            remaining_eligible = [
                tgt for tgt in eligible_targets if tgt.id not in selected_ids
            ]
            remaining_eligible.sort(
                key=lambda tgt: np.linalg.norm(
                    np.array(tgt.target_spacecraft.dynamics.r_BN_N) - scanner_pos
                )
            )
            final_targets += remaining_eligible[:remaining]

        if len(final_targets) < num_actions:
            if len(final_targets) < 1:
                print("no eligible targets available!")
            try:
                final_targets += [final_targets[-1]] * (
                    num_actions - len(final_targets)
                )
            except IndexError:
                print("No eligible targets available; using closest known targets fallback")
                sorted_fallback = sorted(
                    known_targets,
                    key=lambda tgt: np.linalg.norm(
                        np.array(tgt.target_spacecraft.dynamics.r_BN_N) - scanner_pos
                    ),
                )
                final_targets = sorted_fallback[: self.n_ahead_observe]
                if not final_targets:
                    raise RuntimeError("No targets available.")

        obs = {}
        _record_dynamic_priority_candidate_access(
            final_targets,
            float(self.satellite.simulator.sim_time),
        )
        for i, tgt in enumerate(final_targets):
            opportunity = {
                "object": tgt,
                "r_BN_N": np.array(tgt.target_spacecraft.dynamics.r_BN_N),
            }
            props = {}
            for prop_spec in self.target_properties:
                name = prop_spec["name"]
                norm = prop_spec["norm"]
                value = prop_spec["fn"](self.satellite, opportunity)
                props[name] = value / norm
            obs[f"{self.name}_{i}"] = props
        return obs

    def _get_obs_cached(self):
        """Build Polaris target observations with one state read per target."""
        scanner_pos = np.array(self.satellite.dynamics.r_BN_N, dtype=float)
        scanner_vel = np.array(self.satellite.dynamics.v_BN_N, dtype=float)
        scanner_pos_P = np.array(self.satellite.dynamics.r_BN_P, dtype=float)
        c_hat_P = np.array(self.satellite.fsw.c_hat_P, dtype=float)
        HN = rv2HN(scanner_pos, scanner_vel)
        zenith = scanner_pos / np.linalg.norm(scanner_pos)

        with profile_section(self.simulator, "access.polaris_candidate_generation"):
            with profile_section(self.simulator, "obs.polaris_targets.eligible"):
                known_targets = self.satellite.data_store.data.known
                eligible_targets = _eligible_targets_now(self.satellite, known_targets)
                eligible_ids = {tgt.id for tgt in eligible_targets}

            target_cache = {}

            def cache_for_target(target):
                if target in target_cache:
                    return target_cache[target]

                target_pos_N = np.array(
                    target.target_spacecraft.dynamics.r_BN_N, dtype=float
                )
                target_vel_N = np.array(
                    target.target_spacecraft.dynamics.v_BN_N, dtype=float
                )
                target_pos_P = np.array(
                    target.target_spacecraft.dynamics.r_BN_P, dtype=float
                )
                rel_pos_N = target_pos_N - scanner_pos
                rel_vel_N = target_vel_N - scanner_vel
                target_distance = np.linalg.norm(rel_pos_N)
                los_unit = rel_pos_N / target_distance
                elevation_rad = np.arcsin(np.clip(np.dot(los_unit, zenith), -1.0, 1.0))
                vector_target_spacecraft_P = target_pos_P - scanner_pos_P
                vector_target_spacecraft_P_hat = (
                    vector_target_spacecraft_P
                    / np.linalg.norm(vector_target_spacecraft_P)
                )
                angle_to_target = np.degrees(
                    np.arccos(np.dot(vector_target_spacecraft_P_hat, c_hat_P))
                )
                target_shadow_factor = (
                    self.satellite.simulator.satellites[0]
                    .dynamics.world.eclipseObject.eclipseOutMsgs[
                        target.target_spacecraft.dynamics.eclipse_index
                    ]
                    .read()
                    .shadowFactor
                )
                target_cache[target] = {
                    "target_pos_N": target_pos_N,
                    "target_pos_H": HN @ target_pos_N,
                    "rel_pos_N": rel_pos_N,
                    "rel_pos_H": HN @ rel_pos_N,
                    "rel_vel_H": HN @ rel_vel_N,
                    "target_elevation_angle": np.degrees(elevation_rad),
                    "angle_to_target": angle_to_target,
                    "target_distance": target_distance,
                    "target_shadowFactor": target_shadow_factor,
                }
                return target_cache[target]

            with profile_section(self.simulator, "obs.polaris_targets.geometry"):
                target_elevations = [
                    (target, cache_for_target(target)["target_elevation_angle"])
                    for target in eligible_targets
                ]

            with profile_section(self.simulator, "obs.polaris_targets.sort_pad"):
                visible_eligible_targets = [
                    (tgt, elev)
                    for tgt, elev in target_elevations
                    if -21.0 <= elev <= 90.0 and tgt.id in eligible_ids
                ]
                _record_dynamic_priority_visible_access(
                    [tgt for tgt, _ in visible_eligible_targets],
                    float(self.satellite.simulator.sim_time),
                )

                visible_eligible_targets.sort(key=lambda x: x[1])

                num_actions = self.n_ahead_observe
                final_targets = [
                    tgt for tgt, _ in visible_eligible_targets[:num_actions]
                ]

                if len(final_targets) < num_actions:
                    remaining = num_actions - len(final_targets)
                    selected_ids = {tgt.id for tgt in final_targets}
                    remaining_eligible = [
                        tgt
                        for tgt in eligible_targets
                        if tgt.id not in selected_ids
                    ]
                    remaining_eligible.sort(
                        key=lambda tgt: cache_for_target(tgt)["target_distance"]
                    )
                    final_targets += remaining_eligible[:remaining]

                if len(final_targets) < num_actions:
                    if len(final_targets) < 1:
                        print("no eligible targets available!")
                    try:
                        final_targets += [final_targets[-1]] * (
                            num_actions - len(final_targets)
                        )
                    except IndexError:
                        print(
                            "No eligible targets available; using closest known targets fallback"
                        )
                        sorted_fallback = sorted(
                            known_targets,
                            key=lambda tgt: cache_for_target(tgt)["target_distance"],
                        )
                        final_targets = sorted_fallback[: self.n_ahead_observe]
                        if not final_targets:
                            raise RuntimeError("No targets available.")

        obs = {}
        _record_dynamic_priority_candidate_access(
            final_targets,
            float(self.satellite.simulator.sim_time),
        )
        with profile_section(self.simulator, "obs.polaris_targets.properties"):
            for i, tgt in enumerate(final_targets):
                opportunity = {
                    "object": tgt,
                    "r_BN_N": cache_for_target(tgt)["target_pos_N"],
                    "_polaris_cache": cache_for_target(tgt),
                }
                props = {}
                for prop_spec in self.target_properties:
                    name = prop_spec["name"]
                    norm = prop_spec["norm"]
                    value = prop_spec["fn"](self.satellite, opportunity)
                    props[name] = value / norm
                obs[f"{self.name}_{i}"] = props
        return obs

class Eclipse(Observation):
    def __init__(self, norm=1.0, name="eclipse"):
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


__doc_title__ = "Backend"
__all__ = ["ObservationBuilder"]
