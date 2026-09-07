"""Construct the compact role-aware multi-sensor LEO environment."""

from __future__ import annotations

from functools import partial

import numpy as np
from Basilisk.utilities import macros, orbitalMotion

from bsk_rl.completion_gym import CompletionConstellationTasking
from bsk_rl import act, obs, sats, scene
from bsk_rl.comm.completion_communication import CompletionCommunication
from bsk_rl.data.completion_reward import CompletionImageReward
from bsk_rl.act.completion_actions import (
    BroadcastCompletions,
    TransmitCompletions,
    ContinueTask,
    ImageCompletion,
)
from bsk_rl.obs.completion_observations import (
    CompletionContext,
    CompletionTargets,
    CompletionPeers,
)
from bsk_rl.sim import dyn, fsw, world
from bsk_rl.sats.roles import SpacecraftRole

from examples.multiagent_imaging.config import MultiAgentImagingConfig


R_EARTH_M = 6371e3


def _sensor_orbit(index: int) -> orbitalMotion.ClassicElements:
    """Return the original pre-Walker staggered sensor orbit pattern."""
    orbit = orbitalMotion.ClassicElements()
    orbit.a = R_EARTH_M + (700e3 + 100e3 * (index % 2))
    orbit.e = 0.001
    orbit.i = (97.0 if index % 2 == 0 else 70.0) * macros.D2R
    orbit.Omega = (30.0 * index) * macros.D2R
    orbit.omega = 0.0
    orbit.f = (15.0 * index) * macros.D2R
    return orbit


def _sample_leo_target_orbit() -> orbitalMotion.ClassicElements:
    """Sample the current demonstration catalog entirely in LEO."""
    orbit = orbitalMotion.ClassicElements()
    altitude = np.random.uniform(400e3, 2000e3)
    orbit.e = np.random.uniform(0.0, 0.02)
    inclination = np.random.uniform(0.0, 180.0)
    orbit.a = R_EARTH_M + altitude
    orbit.i = inclination * macros.D2R
    orbit.Omega = np.random.uniform(0.0, 360.0) * macros.D2R
    orbit.omega = np.random.uniform(0.0, 360.0) * macros.D2R
    orbit.f = np.random.uniform(0.0, 360.0) * macros.D2R
    return orbit


def build_environment(
    config: MultiAgentImagingConfig,
    *,
    vizard_dir: str | None = None,
    vizard_settings: dict | None = None,
):
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
            CompletionContext(),
            CompletionTargets(config.n_candidates),
        ] + ([CompletionPeers()] if config.n_peers else [])
        action_spec = [
            act.Charge(duration=config.charge_duration_s),
            act.Downlink(
                duration=config.downlink_duration_s,
                variable_duration_downlink=True,
            ),
            act.Desat(duration=config.desat_duration_s),
            BroadcastCompletions(duration=config.broadcast_duration_s),
            ContinueTask(),
            ImageCompletion(
                n_ahead_image=config.n_candidates,
                duration=config.imaging_duration_s,
                variable_duration_imaging=True,
                min_pointing_hold_s=config.min_pointing_hold_s,
                hold_mode="cumulative",
                require_illumination_during_hold=False,
            ),
        ]
        if config.n_peers:
            action_spec.append(
                TransmitCompletions(
                    config.n_peers,
                    duration=config.transmit_duration_s,
                    hold_s=config.transmit_hold_s,
                    bitrate_bps=config.metadata_bitrate_bps,
                )
            )
        completion_n_candidates = config.n_candidates
        completion_communication_mode = config.communication_mode
        dyn_type = dyn.ImagingSCDynModel
        fsw_type = fsw.ImagingSCFSWModel

    # Targets remain full Basilisk spacecraft. Their role excludes them from the
    # PettingZoo/RLlib agent dictionary; Drift is deterministic simulator plumbing.
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
        "oe": _sample_leo_target_orbit,
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
    rewarder = CompletionImageReward(
        alpha=config.alpha,
        reimage_cooldown_orbits=config.reimage_cooldown_orbits,
        quality_threshold=0.5,
        hide_pending_targets=True,
    )
    communicator = CompletionCommunication(
        config.information_case,
        ttl_s=config.message_ttl_s,
        delay_s=config.message_delay_s,
        loss_probability=config.packet_loss_probability,
        link_mode=config.link_mode,
    )
    return CompletionConstellationTasking(
        retasking_mode=config.retasking_mode,
        default_seed=config.seed,
        communication_cost_per_s=config.communication_cost_per_s,
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
        vizard_dir=vizard_dir,
        vizard_settings=vizard_settings,
    )


__all__ = ["build_environment"]
