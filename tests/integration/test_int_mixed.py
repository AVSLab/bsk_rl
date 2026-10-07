"""Physical imaging across distinct scenes and overlapping reward assignments."""

from typing import ClassVar

import numpy as np
import pytest
from Basilisk.utilities import orbitalMotion

from bsk_rl import ConstellationTasking, act, data, obs, sats, scene
from bsk_rl.comm import FreeCommunication, NoCommunication
from bsk_rl.scene.targets import Target
from bsk_rl.sim import dyn, fsw

EPOCH = "2026 OCT 01 00:00:00.000 (UTC)"


class FixedGroundTargets(scene.UniformTargets):
    """Put a ground target directly below the initial imaging satellite."""

    def regenerate_targets(self):
        self.targets = [
            Target("city", [orbitalMotion.REQ_EARTH * 1e3, 0, 0], priority=3)
        ]

    def reset_post_sim_init(self):
        satellite = self.satellites[0]
        position_P = satellite.dynamics.world.PN @ satellite.dynamics.r_BN_N
        # Access registration retains this array; update it before windows are
        # generated, accounting for the realized Earth orientation at the epoch.
        self.targets[0].r_LP_P[:] = (
            self.radius * position_P / np.linalg.norm(position_P)
        )


class GroundObserver(sats.ImagingSatellite):
    dyn_type = dyn.GroundStationDynModel
    fsw_type = fsw.ImagingFSWModel
    observation_spec: ClassVar[list[obs.Observation]] = [obs.Time()]
    action_spec: ClassVar[list[act.Action]] = [
        act.Image(1, max_duration=30),
        act.Charge(duration=10),
    ]


class RSOObserver(sats.Satellite):
    dyn_type = dyn.SpaceToSpaceImagingDynModel
    fsw_type = fsw.SpaceToSpaceImagingFSWModel
    observation_spec: ClassVar[list[obs.Observation]] = [
        obs.Time(),
        obs.RSOTargetProperties(n_ahead_observe=1),
    ]
    action_spec: ClassVar[list[act.Action]] = [
        act.ImageRSO(
            1,
            max_duration=30,
            min_pointing_hold_s=0,
            require_illumination_during_hold=False,
        ),
        act.Charge(duration=10),
    ]


class PassiveTarget(sats.Satellite):
    dyn_type = dyn.RSOTargetDynModel
    fsw_type = fsw.FSWModel
    observation_spec: ClassVar[list[obs.Observation]] = [obs.Time()]
    action_spec: ClassVar[list[act.Action]] = [act.Drift(duration=1e9)]


def make_env(overlap, communicate):
    satellite_args = dict(
        rN=[7e6, 0, 0],
        vN=[0, 7500, 0],
        oe=None,
        sigma_init=[0, 0, 0],
        omega_init=[0, 0, 0],
        wheelSpeeds=[0.0, 0.0, 0.0],
        imageAttErrorRequirement=2.0,
        instrumentBaudRate=1000,
        batteryStorageCapacity=1e9,
        storedCharge_Init=1e9,
        dataStorageCapacity=1e6,
        storageInit=0,
    )
    ground = GroundObserver("ground", sat_args=satellite_args)
    rso = RSOObserver("observer", sat_args=satellite_args)
    target = PassiveTarget("debris")
    catalog = scene.RSOTargetCatalog(
        (
            scene.RSOTarget(
                "object", "debris", (7e6 + 1000, 0, 0), (0, 7500, 0), priority=2
            ),
        ),
        EPOCH,
    )
    scenarios = {
        "ground_scene": FixedGroundTargets(1),
        "rso_scene": scene.RSOTargets(catalog, ["observer"]),
    }
    participants = {"ground_scene": ["ground"], "rso_scene": ["observer", "debris"]}
    rewarders = {
        "ground_images": data.UniqueImageReward(),
        "rso_images": data.RSOImageReward(
            quality_threshold=0,
            acquisition_reward_fn=lambda record, target: target.priority,
        ),
    }
    assignments = {"ground_images": ["ground"], "rso_images": ["observer"]}
    scenario_mapping = {"ground_images": "ground_scene", "rso_images": "rso_scene"}
    if overlap:
        # One physical agent participates in two scenes. The extra scene measures
        # elapsed time; dual ground/RSO imaging requires additional physical models.
        scenarios["clock_scene"] = scene.Scenario()
        participants["clock_scene"] = ["ground"]
        rewarders["elapsed_time"] = data.ResourceReward(
            reward_weight=-0.25,
            resource_fn=lambda sat: sat.simulator.sim_time,
        )
        assignments["elapsed_time"] = ["ground"]
        scenario_mapping["elapsed_time"] = "clock_scene"
    return ConstellationTasking(
        satellites=[ground, rso, target],
        scenario=scene.MixedScenario(scenarios, participants),
        rewarder=data.MixedReward(rewarders, assignments, scenario_mapping),
        communicator=FreeCommunication() if communicate else NoCommunication(),
        world_args=dict(utc_init=EPOCH),
        time_limit=60,
        max_step_duration=1,
        sim_rate=1,
        log_level="ERROR",
    )


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("communicate", [False, True])
def test_distinct_scenes_and_overlapping_agent(overlap, communicate):
    """Preserve physical attribution, global history, communication, and reset."""
    env = make_env(overlap, communicate)
    try:
        for seed in (0, 1):
            env.reset(seed=seed)
            by_name = {sat.name: sat for sat in env.satellites}
            ground, rso, target = (
                by_name[name] for name in ("ground", "observer", "debris")
            )
            ground_channels = (
                {"ground_images", "elapsed_time"} if overlap else {"ground_images"}
            )
            assert set(ground.data_store.data_stores) == ground_channels
            assert set(rso.data_store.data_stores) == {"rso_images"}
            assert target.data_store.data_stores == {}
            assert (
                env.rewarder.rewarders["ground_images"].scenario
                is env.scenario.scenarios["ground_scene"]
            )
            assert (
                env.rewarder.rewarders["rso_images"].scenario
                is env.scenario.scenarios["rso_scene"]
            )
            if overlap:
                assert env.scenario.scenarios["clock_scene"].satellites == [ground]
                assert (
                    env.rewarder.rewarders["elapsed_time"].scenario
                    is env.scenario.scenarios["clock_scene"]
                )
            totals = {sat.name: 0.0 for sat in env.satellites}
            for step in range(8):
                _, rewards, _, _, _ = env.step(
                    {name: 0 if step == 0 else None for name in by_name}
                )
                for name, value in rewards.items():
                    totals[name] += value
            assert len(ground.data_store.data.data["ground_images"].imaged) == 1
            assert len(rso.data_store.data.data["rso_images"].captures) == 1
            assert env.rewarder.rewarders["ground_images"].cum_reward == {"ground": 3}
            assert env.rewarder.rewarders["rso_images"].cum_reward == {"observer": 2}
            expected_ground = 3 - 0.25 * env.simulator.sim_time if overlap else 3
            assert totals == pytest.approx(
                {"ground": expected_ground, "observer": 2, "debris": 0}
            )
            assert env.rewarder.cum_reward == pytest.approx(totals)
            if overlap:
                assert env.rewarder.rewarders["elapsed_time"].cum_reward[
                    "ground"
                ] == pytest.approx(-0.25 * env.simulator.sim_time)
            for sat in env.satellites:
                assert set(sat.data_store.new_data.data) == set(
                    sat.data_store.data_stores
                )
                for key, store in sat.data_store.data_stores.items():
                    assert store.data is sat.data_store.data.data[key]
                if communicate:
                    assert set(sat.data_store.data.data) == set(env.rewarder.rewarders)
            if communicate:
                assert target.data_store.new_data.data == {}
            # Communicated RSO records never become ground acquisition deltas.
            assert "rso_images" not in ground.data_store.new_data.data
            assert "ground_images" not in rso.data_store.new_data.data
    finally:
        if hasattr(env, "simulator"):
            env.close()
