from functools import partial
from typing import ClassVar

import numpy as np
from benchmark import BenchmarkEnv
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from bsk_rl import act, data, obs, sats, scene
from bsk_rl.obs.relative_observations import rso_imaged_regions
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import (
    fibonacci_sphere,
    random_orbit,
    random_unit_vector,
    relative_to_chief,
    rv2HN,
)


def sun_hat_chief(self, other):
    r_SN_N = (
        self.simulator.world.gravFactory.spiceObject.planetStateOutMsgs[
            self.simulator.world.sun_index
        ]
        .read()
        .PositionVector
    )
    r_SN_N = np.array(r_SN_N)
    r_SN_N_hat = r_SN_N / np.linalg.norm(r_SN_N)
    HN = other.dynamics.HN
    return HN @ r_SN_N_hat


def SN(sat):
    """Returns the Planet-Sun Hill frame"""
    planet_state = sat.simulator.world.gravFactory.spiceObject.planetStateOutMsgs[
        sat.simulator.world.sun_index
    ].read()
    r_NS_N = -np.array(planet_state.PositionVector)
    v_NS_N = -np.array(planet_state.VelocityVector)

    return rv2HN(r_N=r_NS_N, v_N=v_NS_N)


def eccentricity_vec_N(sat):
    h = h_vec_N(sat)
    r_N = sat.dynamics.r_BN_N
    v_N = sat.dynamics.v_BN_N
    mu = sat.dynamics.mu
    e_N = np.cross(v_N, h) / mu - r_N / np.linalg.norm(r_N)
    return e_N


def eccentricity_vec_S(sat):
    return SN(sat) @ eccentricity_vec_N(sat)


def h_vec_N(sat):
    r_N = sat.dynamics.r_BN_N
    v_N = sat.dynamics.v_BN_N
    h_N = np.cross(r_N, v_N)
    return h_N


def h_vec_S(sat):
    return SN(sat) @ h_vec_N(sat)


class InspectorSat(sats.Satellite):
    observation_spec: ClassVar[list[obs.Observation]] = [
        obs.SatProperties(
            dict(prop="dv_available", norm=10),
            dict(prop="eccentricity_vec_S", fn=eccentricity_vec_S, norm=0.04184),
            dict(prop="h_vec_S", fn=h_vec_S, norm=5e10),
            dict(
                prop="r_BN_S",
                fn=lambda sat: SN(sat) @ sat.dynamics.r_BN_N,
                norm=7e6,
            ),
        ),
        obs.RelativeProperties(
            dict(prop="r_DC_Hc", norm=500),
            dict(prop="v_DC_Hc", norm=5),
            dict(
                prop="imaged_hill",
                fn=partial(
                    rso_imaged_regions,
                    region_centers=fibonacci_sphere(15),
                    frame="chief_hill",
                ),
            ),
            dict(prop="sun_hat_Hc", fn=sun_hat_chief),
            chief_name="RSO",
        ),
        obs.Eclipse(norm=5700),
        obs.Time(),
    ]

    action_spec: ClassVar[list[act.Action]] = [
        act.ImpulsiveThrustHill(
            chief_name="RSO",
            max_dv=1.0,
            max_drift_duration=5700.0 * 2,
            fsw_action="action_inspect_rso",
        )
    ]
    dyn_type = (
        dyn.MaxRangeDynModel,
        dyn.ConjunctionDynModel,
        dyn.RSOInspectorDynModel,
    )
    fsw_type = (
        fsw.MagicOrbitalManeuverFSWModel,
        fsw.RSOInspectorFSWModel,
    )


inspector_sat_args = dict(
    imageAttErrorRequirement=1.0,
    imageRateErrorRequirement=None,
    instrumentBaudRate=1,
    dataStorageCapacity=1e6,
    batteryStorageCapacity=1e9,
    storedCharge_Init=1e9,
    conjunction_radius=2.0,
    dv_available_init=10.0,
    max_range_radius=1000,
    enforce_range_on_chief=True,
    chief_name="RSO",
)


class RSOSat(sats.Satellite):
    observation_spec: ClassVar[list[obs.Observation]] = [
        obs.SatProperties(dict(prop="one", fn=lambda _: 1.0)),
    ]
    action_spec: ClassVar[list[act.Action]] = [act.NadirPoint(duration=1e9)]
    dyn_type = (dyn.ConjunctionDynModel, dyn.RSODynModel)
    fsw_type = fsw.FSWModel


rso_sat_args = dict(conjunction_radius=2.0)


def episode_data_callback(env):
    data = {}

    data["inspected_points"] = sum(env.rewarder.data.point_inspect_status.values())
    if len(env.scenario.rso_points) > 0:
        data["inspected_fraction"] = env.rewarder.data.num_points_inspected / len(
            env.scenario.rso_points
        )
        if env.rewarder.data.num_points_illuminated > 0:
            data["possible_inspected_fraction"] = (
                env.rewarder.data.num_points_inspected
                / env.rewarder.data.num_points_illuminated
            )
        data["all_inspected"] = env.rewarder.bonus_reward_yielded
        if env.rewarder.bonus_reward_yielded:
            data["all_inspected_time"] = env.rewarder.bonus_reward_time

    data["episode_duration"] = env.simulator.sim_time

    return data


def satellite_data_callback(env, satellite):
    data = {}

    if "Inspector" in satellite.name:
        fuel_used = (
            satellite.sat_args_generator["dv_available_init"]
            - satellite.fsw.dv_available
        )
        data["fuel_used"] = fuel_used
        data["mean_thrust_size"] = fuel_used / max(satellite.fsw.thrust_count, 1)
        data["mean_thrust_duration"] = env.simulator.sim_time / max(
            satellite.fsw.thrust_count, 1
        )
        data["thrust_count"] = satellite.fsw.thrust_count
        data["alive"] = satellite.is_alive()
        data["collision"] = not satellite.dynamics.conjunction_valid()
        data["out_of_range"] = not satellite.dynamics.range_valid()
        data["fuel_remaining"] = satellite.fsw.fuel_remaining()
        chief = [sat for sat in env.satellites if sat.name == "RSO"][0]
        data["distance_to_rso"] = np.linalg.norm(
            np.array(satellite.dynamics.r_BN_N) - np.array(chief.dynamics.r_BN_N)
        )

    return data


def sat_arg_randomizer(satellites):
    # Generate the RSO orbit
    R_E = 6371.0  # km
    a = R_E + np.random.uniform(500, 1100)
    e = np.random.uniform(0.0, min(1 - (R_E + 500) / a, (R_E + 1100) / a - 1))
    chief_orbit = random_orbit(a=a, e=e)

    inspectors = [sat for sat in satellites if "Inspector" in sat.name]
    rso = [satellite for satellite in satellites if satellite.name == "RSO"][0]

    # Generate the inspector initial states.
    args = {}
    for inspector in inspectors:
        relative_randomizer = relative_to_chief(
            chief_name="RSO",
            chief_orbit=chief_orbit,
            deputy_relative_state={
                inspector.name: lambda: np.concatenate(
                    (
                        random_unit_vector() * np.random.uniform(250, 750),
                        random_unit_vector() * np.random.uniform(0, 1.0),
                    )
                ),
            },
        )
        args.update(relative_randomizer([rso, inspector]))

    return args


def rewarder_config(
    fuel_penalty_weight=0.1,
    completion_bonus=1.0,
    inspection_reward_scale=1.0,
    completion_threshold=0.90,
):
    return (
        data.RSOInspectionReward(
            inspection_reward_scale=inspection_reward_scale,
            completion_bonus=completion_bonus,
            completion_threshold=completion_threshold,
            min_time_for_completion=5700.0,
        ),
        data.ResourceReward(
            resource_fn=lambda sat: sat.fsw.dv_available
            if isinstance(sat.fsw, fsw.MagicOrbitalManeuverFSWModel)
            else 0.0,
            reward_weight=fuel_penalty_weight,
        ),
    )


env_args = dict(
    satellites=[
        RSOSat("RSO", sat_args=rso_sat_args),
        InspectorSat("Inspector-1", sat_args=inspector_sat_args),
    ],
    sat_arg_randomizer=sat_arg_randomizer,
    scenario=scene.SphericalRSO(
        n_points=100,
        radius=1.0,
        theta_max=np.radians(30),
        range_max=250,
        theta_solar_max=np.radians(60),
    ),
    rewarder=rewarder_config(),
    time_limit=5700.0 * 10,
    sim_rate=5.0,
    episode_data_callback=episode_data_callback,
    satellite_data_callback=satellite_data_callback,
)

policies = {"inspector", "rso"}


def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == "RSO":
        return "rso"
    return "inspector"


module_specs = {
    "inspector": RLModuleSpec(
        model_config_dict={
            "use_lstm": False,
            "fcnet_hiddens": [512, 512],
            "vf_share_layers": False,
        },
    ),
    "rso": RLModuleSpec(
        model_config_dict={
            "use_lstm": False,
            "fcnet_hiddens": [2, 2],
            "vf_share_layers": False,
        },
    ),
}

training_args = dict(
    lr=[[0, 1e-4], [2e6, 1e-4], [3.5e6, 1e-5], [5e6, 1e-6]],
    gamma=0.9999,
    train_batch_size=3600,
    num_sgd_iter=10,
    use_kl_loss=False,
    clip_param=0.1,
    grad_clip=1.0,
    entropy_coeff=0.0,
)

rso_inspection = BenchmarkEnv(
    env_args=env_args,
    policies=policies,
    policy_mapping_fn=policy_mapping_fn,
    module_specs=module_specs,
    training_args=training_args,
)
