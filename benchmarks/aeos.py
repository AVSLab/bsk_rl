from typing import ClassVar

import numpy as np
from Basilisk.utilities import orbitalMotion
from benchmark import BenchmarkEnv
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from bsk_rl import act, comm, data, obs, sats, scene
from bsk_rl.sim import fsw
from bsk_rl.utils.orbital import random_circular_orbit, walker_delta_args


class AEOS(sats.ImagingSatellite):
    action_spec: ClassVar[list[act.Action]] = [act.Image(n_ahead_image=32)]
    observation_spec: ClassVar[list[obs.Observation]] = [
        obs.SatProperties(
            dict(prop="omega_BH_H", norm=0.03),
            dict(prop="c_hat_H"),
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", norm=7616.5),
        ),
        obs.OpportunityProperties(
            dict(prop="priority"),
            dict(prop="r_LB_H", norm=800 * 1e3),
            dict(prop="target_angle", norm=np.pi / 2),
            dict(prop="target_angle_rate", norm=0.03),
            dict(prop="opportunity_open", norm=300.0),
            dict(prop="opportunity_close", norm=300.0),
            type="target",
            n_ahead_observe=32,
        ),
        obs.Time(),
    ]

    fsw_type = (fsw.SteeringFSWModel, fsw.ImagingFSWModel)


SAT_ARGS = dict(
    imageAttErrorRequirement=0.01,
    imageRateErrorRequirement=0.01,
    batteryStorageCapacity=80.0 * 3600 * 100,
    storedCharge_Init=80.0 * 3600 * 100.0,
    dataStorageCapacity=200 * 8e6 * 100,
    u_max=0.4,
    imageTargetMinimumElevation=np.arctan(800 / 500),
    K1=0.25,
    K3=3.0,
    omega_max=np.radians(5),
    servo_Ki=5.0,
    servo_P=150 / 5,
    oe=lambda: random_circular_orbit(alt=800, i=45),
)


def episode_data_callback(env):
    data = {}

    imaged = env.rewarder.data.imaged
    reward = sum(env.rewarder.cum_reward.values())

    data["imaged"] = len(imaged)
    data["reward"] = reward
    data["avg_tgt_val"] = reward / len(imaged) if len(imaged) > 0 else 0.0
    data["duplicates"] = env.rewarder.data.duplicates

    return data


def satellite_data_callback(env, satellite):
    data = {}

    imaged = satellite.imaged
    missed = satellite.missed

    reward = env.rewarder.cum_reward[satellite.name]

    duration = max(env.simulator.sim_time, 0.01)
    orbits = duration / (95 * 60)

    data["imaged"] = imaged
    data["imaged_per_orbit"] = imaged / orbits
    data["missed"] = missed
    data["missed_per_orbit"] = missed / orbits
    data["reward"] = reward
    data["reward_per_orbit"] = reward / orbits
    if imaged == 0:
        data["avg_tgt_val"] = 0
        data["success_rate"] = 0
    else:
        data["avg_tgt_val"] = reward / imaged
        data["success_rate"] = imaged / (imaged + missed)
    data["attempts"] = imaged + missed
    data["attempts_per_orbit"] = (imaged + missed) / orbits

    data["orbits_completed"] = orbits
    data["alive"] = float(satellite.is_alive())

    return data


def gen_env_args(n_satellites=1):
    env_args = dict(
        satellites=[
            AEOS(name=f"EO{i + 1}", sat_args=SAT_ARGS) for i in range(n_satellites)
        ],
        scenario=scene.UniformTargets((100, 10000)),
        rewarder=data.UniqueImageReward(),
        communicator=comm.FreeCommunication(min_period=60),
        sim_rate=0.5,
        max_step_duration=300.0,
        time_limit=5700 * 3,
        failure_penalty=0.0,
        terminate_on_time_limit=True,
        episode_data_callback=episode_data_callback,
        satellite_data_callback=satellite_data_callback,
    )
    if n_satellites > 1:
        env_args["sat_arg_randomizer"] = walker_delta_args(
            n_planes=1, altitude=800, inc=45
        )
    return env_args


policies = {"policy"}


def policy_mapping_fn(agent_id, *args, **kwargs):
    return "policy"


module_specs = {
    "policy": RLModuleSpec(
        model_config_dict={
            "use_lstm": False,
            "fcnet_hiddens": [1024, 1024],
            "vf_share_layers": False,
        },
    ),
}

training_args = dict(
    lr=3e-5,
    gamma=0.997,
    train_batch_size=3000,
    num_sgd_iter=10,
    use_kl_loss=False,
    clip_param=0.2,
    grad_clip=0.5,
)

aeos_single = BenchmarkEnv(
    env_args=gen_env_args(n_satellites=1),
    policies=policies,
    policy_mapping_fn=policy_mapping_fn,
    module_specs=module_specs,
    training_args=training_args,
)

aeos_constellation = BenchmarkEnv(
    env_args=gen_env_args(n_satellites=3),
    policies=policies,
    policy_mapping_fn=policy_mapping_fn,
    module_specs=module_specs,
    training_args=training_args,
    use_async_connectors=True,
)
