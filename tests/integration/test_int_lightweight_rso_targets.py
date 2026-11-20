"""Lightweight passive RSO targets inside a real space-to-space imaging environment."""

from typing import ClassVar

import numpy as np
import pytest

from bsk_rl import ConstellationTasking, act, data, obs, sats, scene
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.passive_orbits import J2Batch, KeplerBatch

EPOCH = "2026 OCT 01 00:00:00.000 (UTC)"
TARGETS = {
    "debris": ((7e6 + 1000, 0, 0), (0, 7500, 0)),
    "uncooperative": ((7e6 + 2000, 0, 0), (0, 7500, 0)),
}


class Camera(sats.Satellite):
    dyn_type = (dyn.SpaceToSpaceImagingDynModel, dyn.GroundStationDynModel)
    fsw_type = fsw.SpaceToSpaceImagingFSWModel
    observation_spec: ClassVar[list[obs.Observation]] = [
        obs.RSOTargetProperties(dict(prop="valid"), n_ahead_observe=2)
    ]
    action_spec: ClassVar[list[act.Action]] = [
        act.ImageRSO(2, min_pointing_hold_s=0, require_illumination_during_hold=False),
        act.Charge(duration=60),
    ]


class FullTarget(sats.Satellite):
    dyn_type = dyn.RSOTargetDynModel
    fsw_type = fsw.FSWModel
    observation_spec: ClassVar[list[obs.Observation]] = [obs.Time()]
    action_spec: ClassVar[list[act.Action]] = [act.Drift(duration=1e9)]


class LightTarget(sats.Satellite):
    dyn_type = dyn.LightweightRSOTargetDynModel
    fsw_type = fsw.PassiveRSOTargetFSWModel
    observation_spec: ClassVar[list[obs.Observation]] = [obs.Time()]
    action_spec: ClassVar[list[act.Action]] = [act.Drift(duration=1e9)]


def make_env(mode=None, time_limit=600):
    definitions = (
        scene.RSOTarget("83", "debris", *TARGETS["debris"], priority=2),
        scene.RSOTarget("rso/209", "uncooperative", *TARGETS["uncooperative"]),
    )
    camera = Camera(
        "camera_a",
        sat_args=dict(
            rN=[7e6, 0, 0],
            vN=[0, 7500, 0],
            oe=None,
            sigma_init=[0, 0, 0],
            omega_init=[0, 0, 0],
            wheelSpeeds=[0.0, 0.0, 0.0],
            batteryStorageCapacity=1e9,
            storedCharge_Init=1e9,
        ),
    )
    if mode is None:
        targets = [FullTarget(name) for name in TARGETS]
    else:
        targets = [
            LightTarget(name, sat_args=dict(target_force_mode=mode)) for name in TARGETS
        ]
    return ConstellationTasking(
        satellites=[targets[0], camera, targets[1]],
        scenario=scene.RSOTargets(
            scene.RSOTargetCatalog(definitions, EPOCH), ["camera_a"]
        ),
        rewarder=data.RSOImageReward(quality_threshold=0),
        world_args=dict(utc_init=EPOCH, gsMinimumElevation=np.radians(-90)),
        time_limit=time_limit,
        max_step_duration=60,
        sim_rate=1,
        log_level="ERROR",
    )


IMAGE, CHARGE = 0, 2


def run(env, image=False, seed=0):
    """Step to the time limit; return target positions and the final pole.

    The camera charges throughout, or with ``image=True`` images the first
    candidate whenever one is available and charges otherwise.
    """
    observation, _ = env.reset(seed=seed)
    step = 0
    while env.agents:
        valid = observation["camera_a"][0] > 0.5
        camera_action = IMAGE if image and valid else CHARGE
        observation, *_ = env.step(
            {
                sat.name: (
                    camera_action
                    if sat.name == "camera_a"
                    else (0 if step == 0 else None)
                )
                for sat in env.satellites
                if sat.name in env.agents
            }
        )
        step += 1
    states = {
        sat.name: np.array(sat.dynamics.scObject.scStateOutMsg.read().r_BN_N)
        for sat in env.satellites
        if sat.name in TARGETS
    }
    return states, env.simulator.sim_time, np.array(env.simulator.world.PN[2])


@pytest.mark.parametrize("mode", ["two_body", "j2"])
def test_published_states_match_direct_propagation(mode):
    states, t, pole = run(make_env(mode))
    r0 = np.array([TARGETS[n][0] for n in TARGETS], dtype=float)
    v0 = np.array([TARGETS[n][1] for n in TARGETS], dtype=float)
    if mode == "two_body":
        expected, _ = KeplerBatch(r0, v0).at(t)
        tolerance = 1e-3  # [m]
    else:
        # The publisher follows the live Earth pole; a fixed final pole differs by
        # well under a microradian over ten minutes.
        expected, _ = J2Batch(r0, v0).at(t, pole=pole)
        tolerance = 1.0  # [m]
    for i, name in enumerate(TARGETS):
        assert np.linalg.norm(states[name] - expected[i]) < tolerance


def test_j2_targets_track_full_basilisk_targets():
    full, t_full, _ = run(make_env(None))
    light, t_light, _ = run(make_env("j2"))
    kepler, _, _ = run(make_env("two_body"))
    assert t_full == t_light
    for name in TARGETS:
        j2_error = np.linalg.norm(light[name] - full[name])
        kepler_error = np.linalg.norm(kepler[name] - full[name])
        assert j2_error < kepler_error
        assert j2_error < J2_TOLERANCE_M


def camera_storage(env):
    return next(sat for sat in env.satellites if sat.name == "camera_a").dynamics.storage_level


def test_imaging_matches_full_targets():
    full, light = make_env(None), make_env("j2")
    run(full, image=True)
    run(light, image=True)
    assert camera_storage(light) > 0
    assert camera_storage(light) == camera_storage(full)


def test_reset_repeats_episode():
    env = make_env("j2", time_limit=180)
    first, _, _ = run(env, image=True)
    # One publish per world tick, including t=0: ticks 0 through 180.
    assert env.simulator.passive_target_publisher.callback_count == 181
    again, _, _ = run(env, image=True)
    for name in TARGETS:
        np.testing.assert_array_equal(first[name], again[name])


def test_lightweight_targets_support_only_drift():
    class ChargingTarget(LightTarget):
        action_spec: ClassVar[list[act.Action]] = [act.Charge(duration=10)]

    env = make_env("j2")
    env.satellites[0].__class__ = ChargingTarget
    env.satellites[0].action_builder.action_spec[0].fsw_action = "action_charge"
    env.reset(seed=0)
    with pytest.raises(ValueError, match="only Drift"):
        env.satellites[0].action_builder.action_spec[0].set_action(0)


def test_backend_selection():
    assert dyn.rso_target_backend("full")[0] is dyn.RSOTargetDynModel
    dyn_type, fsw_type, args = dyn.rso_target_backend("two_body")
    assert dyn_type is dyn.LightweightRSOTargetDynModel
    assert fsw_type is fsw.PassiveRSOTargetFSWModel
    assert args == {"target_force_mode": "two_body"}
    with pytest.raises(ValueError):
        dyn.rso_target_backend("legacy")


# [m] After 600 s from these states, J2 targets differ from full Basilisk targets
# (degree-10 GGM03S gravity plus drag) by 14.0 m; two-body targets by 2.1 km.
J2_TOLERANCE_M = 50.0
