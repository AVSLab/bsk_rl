import numpy as np
from benchmark import BenchmarkEnv
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from bsk_rl import act, data, obs, sats, scene
from bsk_rl.sim import dyn, fsw


class ScanningSatellite(sats.AccessSatellite):
    observation_spec = [
        obs.SatProperties(
            dict(prop="storage_level_fraction"),
            dict(prop="battery_charge_fraction"),
            dict(prop="wheel_speeds_fraction"),
        ),
        obs.OpportunityProperties(
            dict(prop="opportunity_open", norm=5700),
            dict(prop="opportunity_close", norm=5700),
            type="ground_station",
            n_ahead_observe=1,
        ),
        obs.Eclipse(norm=5700),
        obs.Time(),
    ]
    action_spec = [
        act.Scan(duration=180.0),
        act.Charge(duration=120.0),
        act.Downlink(duration=60.0),
        act.Desat(duration=60.0),
    ]
    dyn_type = (dyn.ContinuousImagingDynModel, dyn.GroundStationDynModel)
    fsw_type = (fsw.ContinuousImagingFSWModel, fsw.BasicFSWModel)


sat_args = dict(
    # Data
    dataStorageCapacity=5000 * 8e6,  # bits
    storageInit=lambda: np.random.uniform(0.0, 0.8) * 5000 * 8e6,
    instrumentBaudRate=0.5 * 8e6,
    transmitterBaudRate=-50 * 8e6,
    # Power
    batteryStorageCapacity=400 * 3600,  # W*s
    storedCharge_Init=lambda: np.random.uniform(0.3, 1.0) * 400 * 3600,
    basePowerDraw=-10.0,  # W
    instrumentPowerDraw=-30.0,  # W
    transmitterPowerDraw=-25.0,  # W
    thrusterPowerDraw=-80.0,  # W
    panelArea=0.25,
    # Attitude
    imageAttErrorRequirement=0.1,
    imageRateErrorRequirement=0.1,
    disturbance_vector=lambda: np.random.normal(scale=0.002, size=3),  # N*m
    maxWheelSpeed=6000.0,  # RPM
    wheelSpeeds=lambda: np.random.uniform(-3000, 3000, 3),
    desatAttitude="nadir",
)

DURATION = 5 * 5700.0  # About 5 orbits


def episode_data_callback(env):
    reward = env.rewarder.cum_reward
    reward = sum(reward.values()) / len(reward)
    orbits = env.simulator.sim_time / (95 * 60)

    data = dict(
        reward=reward,
        orbits_complete=orbits,
    )
    if orbits > 0:
        data["reward_per_orbit"] = reward / orbits
    if orbits < DURATION / (95 * 60):
        data["orbits_complete_partial_only"] = orbits

    return data


def satellite_data_callback(env, sat):
    orbits = env.simulator.sim_time / (95 * 60)
    data = dict(
        alive=float(sat.is_alive()),
        rw_status_valid=float(sat.dynamics.rw_speeds_valid()),
        battery_status_valid=float(sat.dynamics.battery_valid()),
        orbits_complete=orbits,
        storage_level=sat.dynamics.storage_level_fraction,
        battery_level=sat.dynamics.battery_charge_fraction,
        scanning_time=sat.data_store.data.scanning_time / DURATION,
    )

    return data


env_args = dict(
    satellites=[ScanningSatellite("scanning_sat", sat_args=sat_args)],
    scenario=scene.UniformNadirScanning(value_per_second=1 / DURATION),
    rewarder=data.ScanningTimeReward(),
    time_limit=DURATION,
    failure_penalty=-1.0,
    terminate_on_time_limit=True,
    episode_data_callback=episode_data_callback,
    satellite_data_callback=satellite_data_callback,
)


policies = {"policy"}
def policy_mapping_fn(agent_id, *args, **kwargs):
    return "policy"
module_specs = {
    "policy": RLModuleSpec(
        model_config_dict={
            "use_lstm": False,
            "fcnet_hiddens": [512, 512],
            "vf_share_layers": False,
        },
    ),
}

training_args = dict(
    lr=3e-5,
    gamma=0.99999,
    train_batch_size=3200,
    num_sgd_iter=10,
    use_kl_loss=False,
    clip_param=0.1,
    grad_clip=0.5,
)

nadir_science = BenchmarkEnv(
    env_args=env_args,
    policies=policies,
    policy_mapping_fn=policy_mapping_fn,
    module_specs=module_specs,
    training_args=training_args,
)
