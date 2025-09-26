# put this near your other imports in the same repo
import numpy as np
import gymnasium as gym

from bsk_rl import act, data, obs, scene, sats
from bsk_rl.sim import dyn, fsw, world
from Basilisk.utilities import macros, orbitalMotion

# ---- bring your callbacks in (or pass as args) ----
from updated_train_Polaris import env_metrics_callback, sat_metrics_callback  # adjust import

def _custom_oe_randomizer():
    rLEO = 6871.0 * 1000
    rUpperLEO = 8371.0 * 1000
    oe = orbitalMotion.ClassicElements()
    oe.a = np.random.uniform(1.00 * rLEO, rUpperLEO)
    if oe.a < 2 * rLEO:
        oe.e = np.random.uniform(0.0, 0.02)
        while oe.a * (1 - oe.e) < 6771.0 * 1000:
            oe.e = np.random.uniform(0.0, 0.02)
    else:
        oe.e = np.random.uniform(0.0, 0.2)
    oe.i = np.random.uniform(0, 180) * macros.D2R
    oe.Omega = np.random.uniform(0, 360) * macros.D2R
    oe.omega = np.random.uniform(0, 360) * macros.D2R
    oe.f = np.random.uniform(0, 360) * macros.D2R
    return oe


def make_env_and_args_for_bc(
    *,
    n_targets: int = 100,
    n_targets_ahead: int = 10,
    imaging_duration: float = 300.0,
    extra_time_factor: float = 1.5,
    seed: int = 184,
    failure_penalty: float = -100.0,
    terminate_on_time_limit: bool = False,
    generate_obs_retasking_only: bool = False,
    world_type = world.GroundStationWorldModel,
    episode_cb = env_metrics_callback,
    satellite_cb = sat_metrics_callback,
    vizard_dir: str | None = None,
    vizard_rate: float = 2.0,
    log_level: str = "ERROR",
    use_heuristic: bool = True,
    heuristic_mode: str = "distance",
):
    """
    Build (env, env_args) consistent with your training script.
    - env: Gym env for local BC rollouts ("ConstellationTasking-v1").
    - env_args: dict to pass as RLlib `env_config` for "ConstellationTasking-RLlib".
    """
    total_time = extra_time_factor * n_targets * 300.0

    # ---- Satellite class with obs_v ~ "2/7" from your training script ----
    ScanningSat = type(
        "MyScanningSatellite",
        (sats.AccessSatellite,),
        dict(
            observation_spec=[
                obs.SatProperties(
                    dict(prop="storage_level_fraction"),
                    dict(prop="battery_charge_fraction"),
                    dict(prop="wheel_speeds_fraction"),
                ),
                obs.PolarisScTargetProperties(
                    dict(prop="target_elevation_angle", norm=90.0),
                    dict(prop="rel_pos_vector_r_BR_H", norm=15960 * 1000),
                    dict(prop="angle_to_target", norm=90.0),
                    dict(prop="target_distance", norm=15960 * 1000),
                    dict(prop="target_shadowFactor", norm=1.0),
                    n_ahead_observe=n_targets_ahead,
                ),
                obs.Eclipse(norm=5700),
                obs.OpportunityProperties(
                    dict(prop="opportunity_open", norm=5700.0),
                    dict(prop="opportunity_close", norm=5700.0),
                    type="ground_station",
                    n_ahead_observe=2,
                ),
            ],
            action_spec=[
                act.ImageRSO(n_ahead_image=n_targets_ahead, duration=imaging_duration),
                act.Charge(duration=300.0),
                act.Downlink(duration=300.0),
                act.Desat(duration=150.0),
            ],
            dyn_type=dyn.ImagingSCDynModel,
            fsw_type=fsw.ImagingSCFSWModel,
        ),
    )

    TargetSat = type(
        "MyTargetSatellite",
        (sats.Satellite,),
        dict(
            observation_spec=[obs.Time()],
            action_spec=[act.Drift(duration=total_time)],
            dyn_type=dyn.BasicTargetDynamicsModel,
            fsw_type=fsw.BasicTargetFSWModel,
        ),
    )

    # ---- sat_args (matches training script) ----
    sat_args = {
        "imageAttErrorRequirement": 0.01,              # you set this; left as-is
        "dataStorageCapacity": 50 * 8e6 / 2,
        "storageInit": lambda: np.random.uniform(0.0, 0.0) * 50 * 8e6 / 2,
        "instrumentBaudRate": 0.5 * 8e6,
        "transmitterBaudRate": -0.5 * 8e6,
        "batteryStorageCapacity": 500 * 3600,
        "storedCharge_Init": lambda: np.random.uniform(0.10, 0.4) * 500 * 3600,
        "basePowerDraw": -10.0,
        "instrumentPowerDraw": -30.0,
        "transmitterPowerDraw": -25.0,
        "thrusterPowerDraw": -80.0,
        "panelArea": 1.0,
        "disturbance_vector": lambda: np.random.normal(scale=0.000, size=3),
        "maxWheelSpeed": 6000.0,
        "wheelSpeeds": lambda: np.random.uniform(-500, 500, 3),
        "desatAttitude": "sun",
        "downlink_bonus": 0.0,
        "imaging_bonus": 1.0,  # 1 - downlink_bonus
        "eclipse_threshold_for_imaging": 0.5,
        "eclipse_threshold_for_reward": 0.5,
        "use_heuristic": use_heuristic,
        "heuristic_mode": heuristic_mode,
        # Uncomment if/when you want penalties wired in training:
        # "full_storage_penalty": -1,
        # "low_battery_penalty": -1,
    }

    target_args = dict(
        oe=_custom_oe_randomizer,
        batteryStorageCapacity=1.0,
        storedCharge_Init=0.0,
        basePowerDraw=-10000.0,  # speed trick as in script
    )

    # ---- instantiate satellites ----
    ss1 = ScanningSat(name="SS1", sat_args=sat_args)
    targets = [TargetSat(name=f"target_{i}", sat_args=target_args) for i in range(n_targets)]
    all_sat = [ss1] + targets

    # ---- env_args for RLlib's "ConstellationTasking-RLlib" ----
    env_args = dict(
        satellites=[all_sat],  # RLlib wrapper expects a list of satellite-lists
        scenario=[scene.RandomSatellites("SS1", n_targets=n_targets)],
        rewarder=[data.RSOTargetImageReward()],
        world_type=[world_type],
        time_limit=[total_time],
        failure_penalty=[failure_penalty],
        terminate_on_time_limit=[terminate_on_time_limit],
        generate_obs_retasking_only=[generate_obs_retasking_only],
        episode_data_callback=[episode_cb],
        satellite_data_callback=[satellite_cb],
    )

    # ---- a local Gym env to use for BC rollouts/validation ----
    gym_kwargs = dict(
        satellites=all_sat,
        scenario=scene.RandomSatellites("SS1", n_targets=n_targets),
        rewarder=data.RSOTargetImageReward(),
        world_type=world_type,
        time_limit=total_time,
        log_level=log_level,
        disable_env_checker=True,
    )
    if vizard_dir is not None:
        gym_kwargs["vizard_dir"] = vizard_dir
        gym_kwargs["vizard_settings"] = dict(vizard_rate=vizard_rate)

    env = gym.make("ConstellationTasking-v1", **gym_kwargs)
    env.reset(seed=seed)

    return env, env_args

env, env_args = make_env_and_args_for_bc(seed=184)
print('env args are: ', env_args)