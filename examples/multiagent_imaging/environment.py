"""Construct the role-aware two-sensor AMOS 2026 environment."""

from __future__ import annotations

from functools import partial

import numpy as np
from Basilisk.utilities import macros, orbitalMotion

from bsk_rl import SensingAgentConstellationTasking
from bsk_rl import act, data, obs, sats, scene
from bsk_rl.comm import InformationCase, IntentStatusCommunication
from bsk_rl.sim import dyn, fsw, world
from bsk_rl.sats.roles import SpacecraftRole

from examples.multiagent_imaging.config import MultiAgentImagingConfig


R_EARTH_M = 6371e3


def _sensor_orbit(index: int) -> orbitalMotion.ClassicElements:
    """Distinct initial orbit for a homogeneous sensor spacecraft."""
    orbit = orbitalMotion.ClassicElements()
    orbit.a = R_EARTH_M + (700e3 + 100e3 * (index % 2))
    orbit.e = 0.001
    orbit.i = (97.0 if index % 2 == 0 else 70.0) * macros.D2R
    orbit.Omega = (30.0 * index) * macros.D2R
    orbit.omega = 0.0
    orbit.f = (15.0 * index) * macros.D2R
    return orbit


def _sample_mixed_target_orbit() -> orbitalMotion.ClassicElements:
    regime = np.random.choice(["LEO", "MEO", "GEO"], p=[0.5, 0.3, 0.2])
    orbit = orbitalMotion.ClassicElements()
    if regime == "LEO":
        altitude = np.random.uniform(400e3, 2000e3)
        orbit.e = np.random.uniform(0.0, 0.02)
        inclination = np.random.uniform(0.0, 180.0)
    elif regime == "MEO":
        altitude = np.random.uniform(2000e3, 35000e3)
        orbit.e = np.random.uniform(0.0, 0.1)
        inclination = np.random.uniform(0.0, 120.0)
    else:
        altitude = np.random.uniform(35486e3, 36086e3)
        orbit.e = np.random.uniform(0.0, 0.0015)
        inclination = np.random.uniform(0.0, 15.0)
    orbit.a = R_EARTH_M + altitude
    orbit.i = inclination * macros.D2R
    orbit.Omega = np.random.uniform(0.0, 360.0) * macros.D2R
    orbit.omega = np.random.uniform(0.0, 360.0) * macros.D2R
    orbit.f = np.random.uniform(0.0, 360.0) * macros.D2R
    return orbit


def build_environment(config: MultiAgentImagingConfig):
    """Build a real Basilisk environment with sensing agents and passive RSOs."""

    class SensorSatellite(sats.AccessSatellite):
        observation_spec = [
            obs.SatProperties(
                dict(prop="storage_level_fraction"),
                dict(prop="battery_charge_fraction"),
                dict(prop="wheel_speeds_fraction"),
                dict(prop="s_hat_H", fn=obs.s_hat_H, norm=1.0),
            ),
            obs.Eclipse(norm=5700.0),
            obs.OpportunityProperties(
                dict(prop="opportunity_open", norm=5700.0),
                dict(prop="opportunity_close", norm=5700.0),
                type="ground_station",
                n_ahead_observe=2,
            ),
            obs.PolarisScTargetProperties(
                dict(prop="priority", norm=10.0),
                dict(prop="rel_pos_vector_r_BR_H", norm=15960e3),
                dict(prop="rel_vel_vector_v_BR_H", norm=12000.0),
                dict(prop="angle_to_target", norm=90.0),
                dict(prop="target_distance", norm=15960e3),
                dict(prop="target_shadowFactor", norm=1.0),
                dict(prop="known_cooldown_remaining", norm=12000.0),
                dict(prop="known_pending", norm=1.0),
                dict(prop="known_teammate_intent", norm=1.0),
                n_ahead_observe=config.n_candidates,
            ),
        ]
        action_spec = [
            act.Charge(duration=config.charge_duration_s),
            act.Downlink(
                duration=config.downlink_duration_s,
                variable_duration_downlink=True,
            ),
            act.Desat(duration=config.desat_duration_s),
            act.BroadcastIntent(duration=config.broadcast_duration_s),
            act.ImageRSO(
                n_ahead_image=config.n_candidates,
                duration=config.imaging_duration_s,
                variable_duration_imaging=True,
                min_pointing_hold_s=config.min_pointing_hold_s,
                hold_mode="cumulative",
                require_illumination_during_hold=False,
            ),
        ]
        dyn_type = dyn.ImagingSCDynModel
        fsw_type = fsw.ImagingSCFSWModel

    class PassiveTargetSatellite(sats.Satellite):
        observation_spec = [obs.Time()]
        action_spec = [act.Drift(duration=config.episode_duration_s + 1.0)]
        dyn_type = dyn.BasicTargetDynamicsModel
        fsw_type = fsw.BasicTargetFSWModel

    image_bits = 4e6
    sensor_args = {
        "imageAttErrorRequirement": 0.0025,
        "imageRateErrorRequirement": 0.01,
        "dataStorageCapacity": 50.0 * image_bits,
        "storageInit": 0.0,
        "instrumentBaudRate": image_bits,
        "transmitterBaudRate": -image_bits,
        "batteryStorageCapacity": 500.0 * 3600.0,
        "storedCharge_Init": 0.9 * 500.0 * 3600.0,
        "basePowerDraw": -10.0,
        "instrumentPowerDraw": -30.0,
        "transmitterPowerDraw": -25.0,
        "thrusterPowerDraw": -80.0,
        "panelArea": 1.0,
        "disturbance_vector": np.zeros(3),
        "maxWheelSpeed": 6000.0,
        "wheelSpeeds": np.zeros(3),
        "desatAttitude": "sun",
        "downlink_bonus": config.alpha,
        "imaging_bonus": 1.0 - config.alpha,
        "full_storage_penalty": 0.0,
        "low_battery_penalty": 0.0,
        "eclipse_threshold_for_imaging": 0.5,
        "eclipse_threshold_for_reward": 0.5,
        "empty_downlink_penalty": -1.0,
    }
    sensors = [
        SensorSatellite(
            name=f"sensor_{index}",
            sat_args={**sensor_args, "oe": partial(_sensor_orbit, index)},
            role=SpacecraftRole.SENSING_AGENT,
        )
        for index in range(config.n_sensors)
    ]
    passive_args = {
        "oe": _sample_mixed_target_orbit,
        "batteryStorageCapacity": 1e12,
        "storedCharge_Init": 5e11,
        "basePowerDraw": 0.0,
    }
    targets = [
        PassiveTargetSatellite(
            name=f"target_{index}",
            sat_args=passive_args,
            role=SpacecraftRole.PASSIVE_TARGET,
        )
        for index in range(config.n_targets)
    ]

    scenario = scene.RandomSatellites(
        None,
        n_targets=config.n_targets,
        priority_mode="uniform",
        priority_sum=100.0,
        dynamic_priority_event_enabled=False,
        hio_count=0,
        shio_count=0,
    )
    rewarder = data.MultiSensorRSOTargetImageReward(
        alpha=config.alpha,
        reimage_cooldown_orbits=config.reimage_cooldown_orbits,
        quality_threshold=0.5,
        hide_pending_targets=True,
    )
    communicator = IntentStatusCommunication(
        InformationCase(config.information_case),
        message_ttl_s=config.message_ttl_s,
        perfect_metadata_delivery=config.perfect_metadata_delivery,
    )
    return SensingAgentConstellationTasking(
        satellites=[*sensors, *targets],
        scenario=scenario,
        rewarder=rewarder,
        communicator=communicator,
        world_type=world.GroundStationWorldModel,
        sim_rate=config.sim_rate_s,
        max_step_duration=config.max_step_duration_s,
        time_limit=config.episode_duration_s,
        generate_obs_retasking_only=True,
        log_level="WARNING",
    )


__all__ = ["build_environment"]
