#!/usr/bin/env python3
"""Validate opt-in profiling/speed flags against a small seeded AMOS env.

This script is intentionally small and local-friendly. It compares target recorder
settings while keeping the same seed, actions, rewards, and target observations.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from Basilisk.utilities import macros, orbitalMotion  # noqa: E402
from bsk_rl import ConstellationTasking, act, data, obs, scene, sats  # noqa: E402
from bsk_rl.sim import dyn, fsw, world  # noqa: E402


@contextmanager
def temporary_env(**updates):
    old_values = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _leo_orbit():
    oe = orbitalMotion.ClassicElements()
    oe.a = 6371e3 + np.random.uniform(500e3, 900e3)
    oe.e = np.random.uniform(0.0, 0.01)
    oe.i = np.random.uniform(30.0, 100.0) * macros.D2R
    oe.Omega = np.random.uniform(0.0, 360.0) * macros.D2R
    oe.omega = np.random.uniform(0.0, 360.0) * macros.D2R
    oe.f = np.random.uniform(0.0, 360.0) * macros.D2R
    return oe


def make_env(n_targets: int = 8, n_ahead: int = 4, total_time: float = 900.0):
    class Scanner(sats.AccessSatellite):
        observation_spec = [
            obs.PolarisScTargetProperties(
                dict(prop="priority", norm=2.0),
                dict(prop="target_elevation_angle", norm=90.0),
                dict(prop="rel_pos_vector_r_BR_H", norm=15960 * 1000),
                dict(prop="rel_vel_vector_v_BR_H", norm=16000.0),
                dict(prop="angle_to_target", norm=90.0),
                dict(prop="target_distance", norm=15960 * 1000),
                dict(prop="target_shadowFactor", norm=1.0),
                n_ahead_observe=n_ahead,
            ),
        ]
        action_spec = [act.ImageRSO(n_ahead_image=n_ahead, duration=120.0)]
        dyn_type = dyn.ImagingSCDynModel
        fsw_type = fsw.ImagingSCFSWModel

    class Target(sats.Satellite):
        observation_spec = [obs.Time()]
        action_spec = [act.Drift(duration=total_time)]
        dyn_type = dyn.BasicTargetDynamicsModel
        fsw_type = fsw.BasicTargetFSWModel

    image_bits = 8e6 / 2
    scanner_args = Scanner.default_sat_args(
        oe=_leo_orbit,
        dataStorageCapacity=100 * image_bits,
        storageInit=0.0,
        instrumentBaudRate=0.5 * 8e6,
        transmitterBaudRate=-0.5 * 8e6,
        batteryStorageCapacity=1000 * 500 * 3600,
        storedCharge_Init=1000 * 500 * 3600,
        basePowerDraw=-10.0,
        instrumentPowerDraw=-30.0,
        transmitterPowerDraw=-25.0,
        thrusterPowerDraw=-80.0,
        panelArea=1.0,
        downlink_bonus=0.0,
        imaging_bonus=1.0,
        eclipse_threshold_for_imaging=0.5,
        eclipse_threshold_for_reward=0.5,
        empty_downlink_penalty=-1,
    )
    target_args = Target.default_sat_args(
        oe=_leo_orbit,
        batteryStorageCapacity=1.0,
        storedCharge_Init=1.0,
        basePowerDraw=0.0,
    )
    satellites = [Scanner("SS1", sat_args=scanner_args)] + [
        Target(f"target_{idx}", sat_args=target_args) for idx in range(n_targets)
    ]
    return ConstellationTasking(
        satellites=satellites,
        scenario=scene.RandomSatellites("SS1", n_targets=n_targets),
        rewarder=data.RSOTargetImageReward(
            verify_image_quality_on_downlink=False,
            hide_pending_targets=False,
        ),
        world_type=world.GroundStationWorldModel,
        time_limit=total_time,
        terminate_on_time_limit=False,
    )


def run_case(target_recorders: bool, polaris_obs_cache: bool, seed: int, steps: int):
    with temporary_env(
        BSK_RL_TARGET_RECORDERS=int(target_recorders),
        BSK_RL_POLARIS_OBS_CACHE=int(polaris_obs_cache),
    ):
        env = make_env()
        observations, _ = env.reset(seed=seed)
        ss1_obs = [np.asarray(observations["SS1"], dtype=float)]
        rewards = []
        for _ in range(steps):
            actions = {agent: 0 for agent in env.agents}
            observations, reward, terminated, truncated, _ = env.step(actions)
            ss1_obs.append(np.asarray(observations.get("SS1", []), dtype=float))
            rewards.append(float(reward.get("SS1", 0.0)))
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break
        target_recorders_present = [
            getattr(sat.dynamics, "target_state_recorder", None) is not None
            for sat in env.satellites[1:]
        ]
        result = {
            "ss1_obs": ss1_obs,
            "rewards": np.asarray(rewards, dtype=float),
            "cum_reward": float(env.rewarder.cum_reward.get("SS1", 0.0)),
            "sim_time": float(env.simulator.sim_time),
            "target_recorders_present": target_recorders_present,
            "profile_metrics": env.profiler.metrics(),
        }
        env.close()
        return result


def main():
    seed = int(os.environ.get("BSK_RL_VALIDATION_SEED", "123"))
    steps = int(os.environ.get("BSK_RL_VALIDATION_STEPS", "2"))
    on_result = run_case(
        target_recorders=True, polaris_obs_cache=True, seed=seed, steps=steps
    )
    off_result = run_case(
        target_recorders=False, polaris_obs_cache=True, seed=seed, steps=steps
    )
    legacy_obs_result = run_case(
        target_recorders=True, polaris_obs_cache=False, seed=seed, steps=steps
    )

    assert any(on_result["target_recorders_present"])
    assert not any(off_result["target_recorders_present"])
    assert len(on_result["ss1_obs"]) == len(off_result["ss1_obs"])
    for obs_on, obs_off in zip(on_result["ss1_obs"], off_result["ss1_obs"]):
        np.testing.assert_allclose(obs_on, obs_off, rtol=1e-10, atol=1e-10)
    for obs_cached, obs_legacy in zip(
        on_result["ss1_obs"], legacy_obs_result["ss1_obs"]
    ):
        np.testing.assert_allclose(obs_cached, obs_legacy, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        on_result["rewards"], off_result["rewards"], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        on_result["rewards"],
        legacy_obs_result["rewards"],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        on_result["cum_reward"], off_result["cum_reward"], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        on_result["cum_reward"],
        legacy_obs_result["cum_reward"],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        on_result["sim_time"], off_result["sim_time"], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        on_result["sim_time"],
        legacy_obs_result["sim_time"],
        rtol=1e-10,
        atol=1e-10,
    )

    print("Validation passed.")
    print(f"steps={steps} seed={seed}")
    print(f"sim_time={on_result['sim_time']:.3f}")
    print(f"cum_reward={on_result['cum_reward']:.6f}")
    if on_result["profile_metrics"]:
        top_metrics = sorted(
            (
                (name, value)
                for name, value in on_result["profile_metrics"].items()
                if name.endswith("/total_s")
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:8]
        print("Top profile totals:")
        for name, value in top_metrics:
            print(f"  {name}: {value:.6f}s")


if __name__ == "__main__":
    main()
