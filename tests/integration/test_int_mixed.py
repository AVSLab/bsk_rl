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


class DualObserver(sats.ImagingSatellite):
    """Use both pointing tasks and the library's combined buffer configuration."""

    dyn_type = (dyn.SpaceToSpaceImagingDynModel, dyn.GroundStationDynModel)
    fsw_type = fsw.SpaceToSpaceImagingFSWModel
    observation_spec: ClassVar[list[obs.Observation]] = [
        obs.OpportunityProperties(dict(prop="priority"), n_ahead_observe=1),
        obs.RSOTargetProperties(dict(prop="valid"), n_ahead_observe=1),
    ]
    action_spec: ClassVar[list[act.Action]] = [
        act.Image(1, max_duration=30),
        act.ImageRSO(
            1,
            max_duration=30,
            min_pointing_hold_s=0,
            require_illumination_during_hold=False,
        ),
        act.Charge(duration=10),
        act.Downlink(duration=30),
    ]


class DualGroundTargets(FixedGroundTargets):
    """Retain an unobserved candidate after each ground capture."""

    def regenerate_targets(self):
        super().regenerate_targets()
        self.targets.extend(
            Target(f"city_{i}", self.targets[0].r_LP_P.copy(), priority=3)
            for i in range(2)
        )

    def reset_post_sim_init(self):
        super().reset_post_sim_init()
        for target in self.targets[1:]:
            target.r_LP_P[:] = self.targets[0].r_LP_P


def make_dual_env(n_rso=1, **sat_args):
    observer = DualObserver(
        "camera",
        obs_type=dict,
        sat_args=dict(
            rN=[7e6, 0, 0],
            vN=[0, 7500, 0],
            oe=None,
            sigma_init=[0, 0, 0],
            omega_init=[0, 0, 0],
            wheelSpeeds=[0.0, 0.0, 0.0],
            imageAttErrorRequirement=2.0,
            instrumentBaudRate=1000,
            transmitterBaudRate=-1000,
            batteryStorageCapacity=1e9,
            storedCharge_Init=1e9,
            dataStorageCapacity=1e6,
            storageInit=0,
        )
        | sat_args,
    )
    target_names = ["debris", *(f"debris_{i}" for i in range(1, n_rso))]
    catalog = scene.RSOTargetCatalog(
        tuple(
            scene.RSOTarget(
                "object" if i == 0 else f"object_{i}",
                name,
                (7e6 + (i + 1) * 1000, 0, 0),
                (0, 7500, 0),
            )
            for i, name in enumerate(target_names)
        ),
        EPOCH,
    )
    return ConstellationTasking(
        satellites=[observer, *(PassiveTarget(name) for name in target_names)],
        scenario=scene.MixedScenario(
            {
                "ground": DualGroundTargets(3),
                "rso": scene.RSOTargets(catalog, ["camera"]),
            },
            {"ground": ["camera"], "rso": ["camera", *target_names]},
        ),
        rewarder=data.MixedReward(
            {
                "ground": data.UniqueImageReward(),
                "rso": data.RSOImageReward(
                    quality_threshold=0,
                    acquisition_reward_fn=lambda record, target: 0.9 * target.priority,
                    delivery_reward_fn=lambda record, target: 0.1 * target.priority,
                ),
            },
            {"ground": ["camera"], "rso": ["camera"]},
            {"ground": "ground", "rso": "rso"},
        ),
        world_args=dict(utc_init=EPOCH, gsMinimumElevation=np.radians(-90)),
        time_limit=120,
        max_step_duration=1,
        sim_rate=1,
        log_level="ERROR",
    )


@pytest.mark.parametrize("ground_first", [False, True])
def test_same_satellite_switches_ground_and_rso_imaging(ground_first):
    """Separate physical captures, cancel interrupted holds, deliver, and reset."""
    env = make_dual_env()
    try:
        for seed in (0, 1):
            observations, _ = env.reset(seed=seed)
            observer = env.satellites[0]
            flight_software = observer.fsw
            stores = observer.data_store.data_stores
            expected_partitions = [
                observer.buffer_name,
                observer.rso_scenario.partition_name("object"),
            ]
            assert observer.sat_args["bufferNames"] == expected_partitions
            assert observer.sat_args["transmitterNumBuffers"] == len(
                expected_partitions
            )
            assert (
                list(
                    observer.dynamics.storageUnit.storageUnitDataOutMsg.read().storedDataName
                )
                == expected_partitions
            )
            assert set(observations["camera"]) == {"target", "rso_targets"}
            assert observer.action_space.n == 4
            assert flight_software.locPoint is not flight_software.rsoLocPoint
            assert flight_software.insControl is not flight_software.rsoInsControl
            assert len({task.name for task in flight_software.tasks}) == len(
                flight_software.tasks
            )
            assert (
                observer.dynamics.imagingTarget is not observer.dynamics.targetLocation
            )

            def step(action=None):
                return env.step(
                    {
                        "camera": action,
                        "debris": 0 if env.simulator.sim_time == 0 else None,
                    }
                )

            def capture(action, stores=stores):
                def count():
                    if action == 0:
                        return len(stores["ground"].data.imaged)
                    return len(stores["rso"].data.captures)

                previous = count()
                step(action)
                for _ in range(8):
                    if count() > previous:
                        break
                    step()
                assert count() == previous + 1

            expected_ground, expected_rso = 0, 0
            for action in [0, 1] if ground_first else [1, 0]:
                capture(action)
                if action == 0:
                    expected_ground += 1
                    assert flight_software.rso_capture.target is None
                else:
                    expected_rso += 1
                assert len(stores["ground"].data.imaged) == expected_ground
                assert len(stores["rso"].data.captures) == expected_rso

            assert env.rewarder.rewarders["ground"].cum_reward["camera"] == 3
            assert env.rewarder.rewarders["rso"].cum_reward["camera"] == pytest.approx(
                0.9
            )
            step(3)  # Deliver both image types through the shared transmitter.
            for _ in range(8):
                if stores["rso"].data.deliveries:
                    break
                step()
            assert len(stores["rso"].data.deliveries) == 1
            assert observer.dynamics.storage_level == 0
            assert env.rewarder.rewarders["rso"].cum_reward["camera"] == pytest.approx(
                1
            )

            imaging_action = observer.action_builder.action_spec[1]
            imaging_action.min_pointing_hold_s = 100
            step(1)
            step()
            step()
            assert flight_software.rso_capture.held_s > 0
            capture(0)  # A ground command interrupts the unfinished RSO hold.
            assert flight_software.rso_capture.target is None
            assert flight_software.rso_capture.held_s == 0
            assert len(stores["rso"].data.captures) == 1
            assert env.rewarder.rewarders["ground"].cum_reward["camera"] == 6

            level = observer.dynamics.storage_level
            step(2)
            for _ in range(3):
                step()
                assert observer.dynamics.storage_level == level
            imaging_action.min_pointing_hold_s = 0
            capture(1)
            assert len(stores["ground"].data.imaged) == 2
            assert len(stores["rso"].data.captures) == 2
    finally:
        env.close()


@pytest.mark.parametrize("reverse_actions", [False, True])
@pytest.mark.parametrize("ground_first", [False, True])
@pytest.mark.parametrize("target_form", ["id", "definition", "runtime"])
def test_combined_target_overrides(reverse_actions, ground_first, target_form):
    """IDs and objects route to their owning mode independently of action order."""
    env = make_dual_env()
    observer = env.satellites[0]
    if reverse_actions:
        observer.action_builder.action_spec[:2] = observer.action_builder.action_spec[
            :2
        ][::-1]
    try:
        env.reset(seed=0)
        stores = observer.data_store.data_stores
        ground = env.scenario.scenarios["ground"].targets[0]
        definition = observer.rso_scenario.catalog.targets[0]
        runtime = observer.rso_scenario.targets_by_id[definition.id]
        ground_input = ground.id if target_form == "id" else ground
        rso_input = {"id": definition.id, "definition": definition, "runtime": runtime}[
            target_form
        ]
        tasks = [(ground_input, "ground"), (rso_input, "rso")]
        if not ground_first:
            tasks.reverse()
        for target, channel in tasks:
            env.step(
                {"camera": target, "debris": 0 if env.simulator.sim_time == 0 else None}
            )
            for _ in range(8):
                acquired = (
                    stores[channel].data.imaged
                    if channel == "ground"
                    else stores[channel].data.captures
                )
                if acquired:
                    break
                env.step({"camera": None, "debris": None})
            assert acquired
        assert len(stores["ground"].data.imaged) == 1
        assert {
            record.target_id for record in stores["rso"].data.captures.values()
        } == {definition.id}
        assert env.rewarder.rewarders["ground"].cum_reward["camera"] == 3
        assert env.rewarder.rewarders["rso"].cum_reward["camera"] == pytest.approx(0.9)

        # Invalid tasking must retain the active mode, deadline, and previous key.
        previous_key = observer.action_builder.prev_action_key
        event_name = observer._timed_terminal_event_name
        deadline = env.simulator.eventMap[event_name]
        data_name = observer.dynamics.instrument.nodeDataName
        with pytest.raises(ValueError, match="pending or in cooldown"):
            observer.set_action(rso_input)
        with pytest.raises(ValueError, match="No compatible action"):
            observer.set_action("missing_target")
        assert env.simulator.eventMap[event_name] is deadline
        assert observer.action_builder.prev_action_key == previous_key
        assert observer.dynamics.instrument.nodeDataName == data_name
    finally:
        env.close()


@pytest.mark.parametrize("reverse_actions", [False, True])
def test_overlapping_ids_require_objects_and_do_not_skip_mode_switches(reverse_actions):
    """Reject ambiguous strings before tasking, while typed objects remain distinct."""
    env = make_dual_env()
    observer = env.satellites[0]
    if reverse_actions:
        observer.action_builder.action_spec[:2] = observer.action_builder.action_spec[
            :2
        ][::-1]
    try:
        env.reset(seed=0)
        ground = env.scenario.scenarios["ground"].targets[0]
        rso = observer.rso_scenario.catalog.targets[0]
        ground._id = rso.id
        with pytest.raises(ValueError, match="Ambiguous target"):
            observer.set_action(rso.id)
        assert observer.action_builder.prev_action_key is None
        assert observer.fsw.rso_capture.target is None
        assert observer.latest_target is None
        observer.set_action(ground)
        assert observer.dynamics.instrument.nodeDataName == observer.buffer_name
        observer.set_action(rso)
        assert observer.fsw.rso_capture.target.id == rso.id
        observer.set_action(ground)
        assert observer.fsw.rso_capture.target is None
        assert observer.dynamics.instrument.nodeDataName == observer.buffer_name
        # Selecting the same target again retains the existing ground task.
        observer.set_action(ground)
        assert observer.action_builder.prev_action_key == ground.id
    finally:
        env.close()


@pytest.mark.parametrize("extra_partitions", [None, ["aux"]])
def test_ground_and_multiple_rsos_share_capacity_and_partial_downlink(extra_partitions):
    """Keep named per-target accounting within one capacity and transmitter."""
    env = make_dual_env(
        n_rso=2,
        bufferNames=extra_partitions,
        dataStorageCapacity=3000,
        transmitterBaudRate=-100,
    )
    try:
        env.reset(seed=0)
        observer = env.satellites[0]
        stores = observer.data_store.data_stores
        rso_scene = observer.rso_scenario
        expected_names = [
            observer.buffer_name,
            *(extra_partitions or []),
            *(
                rso_scene.partition_name(target_id)
                for target_id in rso_scene.targets_by_id
            ),
        ]
        assert observer.sat_args["bufferNames"] == expected_names
        assert observer.sat_args["transmitterNumBuffers"] == len(expected_names)
        assert observer.dynamics.transmitter.numBuffers == len(expected_names)
        assert observer.dynamics.storageUnit.storageCapacity == 3000

        def step(action=None):
            result = env.step(
                {
                    sat.name: action
                    if sat is observer
                    else (0 if env.simulator.sim_time == 0 else None)
                    for sat in env.satellites
                }
            )
            message = observer.dynamics.storageUnit.storageUnitDataOutMsg.read()
            assert list(message.storedDataName) == expected_names
            assert observer.dynamics.storage_level == pytest.approx(
                sum(message.storedData)
            )
            assert observer.dynamics.storage_level_fraction == pytest.approx(
                observer.dynamics.storage_level / 3000
            )
            return result

        for action, expected_level in ((0, 1000), (1, 2000), (1, 3000)):
            step(action)
            for _ in range(8):
                if observer.dynamics.storage_level == expected_level:
                    break
                step()
            assert observer.dynamics.storage_level == expected_level
        assert len(stores["ground"].data.imaged) == 1
        assert {
            record.target_id for record in stores["rso"].data.captures.values()
        } == {"object", "object_1"}
        assert observer.dynamics.storage_level_fraction == 1

        step(3)
        for _ in range(20):
            if any(0 < bits < 1000 for bits in stores["rso"].remaining_bits.values()):
                break
            step()
        assert any(0 < bits < 1000 for bits in stores["rso"].remaining_bits.values())
        assert len(stores["rso"].data.deliveries) < 2
        level = observer.dynamics.storage_level
        remaining = stores["rso"].remaining_bits.copy()
        step(2)  # Stop downlink while retaining partially transmitted products.
        step()
        assert observer.dynamics.storage_level == level
        assert stores["rso"].remaining_bits == remaining
        step(3)
        for _ in range(40):
            if observer.dynamics.storage_level == 0:
                break
            step()
        assert observer.dynamics.storage_level == 0
        assert len(stores["ground"].data.imaged) == 1
        assert len(stores["rso"].data.deliveries) == 2
        assert stores["rso"].products == {}
        assert stores["rso"].remaining_bits == {}
        assert env.rewarder.rewarders["ground"].cum_reward["camera"] == 3
        assert env.rewarder.rewarders["rso"].cum_reward["camera"] == pytest.approx(2)
    finally:
        env.close()


def test_ground_image_uses_capacity_needed_for_rso_acquisition():
    """RSO readiness uses shared free space, and ground downlink earns no RSO credit."""
    env = make_dual_env(dataStorageCapacity=1500)
    try:
        env.reset(seed=0)
        observer = env.satellites[0]
        stores = observer.data_store.data_stores

        def step(action=None):
            return env.step(
                {"camera": action, "debris": 0 if env.simulator.sim_time == 0 else None}
            )

        step(0)
        for _ in range(8):
            if stores["ground"].data.imaged:
                break
            step()
        assert observer.dynamics.storage_level == 1000
        step(1)
        for _ in range(5):
            step()
        assert observer.dynamics.storage_level == 1000
        assert observer.fsw.rso_capture.records == []
        assert not stores["rso"].data.captures
        step(3)
        for _ in range(8):
            if observer.dynamics.storage_level == 0:
                break
            step()
        assert observer.dynamics.storage_level == 0
        assert not stores["rso"].data.deliveries
        assert env.rewarder.rewarders["rso"].cum_reward["camera"] == 0
        step(1)
        for _ in range(8):
            if stores["rso"].data.captures:
                break
            step()
        assert len(stores["rso"].data.captures) == 1
        assert observer.dynamics.storage_level == 1000
        assert len(stores["ground"].data.imaged) == 1
        assert env.rewarder.rewarders["ground"].cum_reward["camera"] == 3
        assert env.rewarder.rewarders["rso"].cum_reward["camera"] == pytest.approx(0.9)
    finally:
        env.close()


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
@pytest.mark.parametrize("charge", [False, True])
def test_distinct_scenes_and_overlapping_agent(overlap, communicate, charge):
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
                previous_time = env.simulator.sim_time
                action = 1 if charge else 0
                actions = {
                    name: action if step == 0 else None
                    for name in ("ground", "observer")
                }
                actions["debris"] = 0 if step == 0 else None
                _, rewards, _, _, _ = env.step(actions)
                if charge:
                    # Charging produces no imaging reward. The independent clock
                    # channel still penalizes elapsed time in overlapping cases.
                    time_penalty = (
                        -0.25 * (env.simulator.sim_time - previous_time)
                        if overlap
                        else 0
                    )
                    assert rewards == pytest.approx(
                        {"ground": time_penalty, "observer": 0, "debris": 0}
                    )
                    assert not ground.data_store.new_data.data["ground_images"].imaged
                    assert not rso.data_store.new_data.data["rso_images"].captures
                for name, value in rewards.items():
                    totals[name] += value
            expected_images = 0 if charge else 1
            ground_image_reward = 0 if charge else 3
            rso_image_reward = 0 if charge else 2
            assert (
                len(ground.data_store.data.data["ground_images"].imaged)
                == expected_images
            )
            assert (
                len(rso.data_store.data.data["rso_images"].captures) == expected_images
            )
            assert env.rewarder.rewarders["ground_images"].cum_reward == {
                "ground": ground_image_reward
            }
            assert env.rewarder.rewarders["rso_images"].cum_reward == {
                "observer": rso_image_reward
            }
            expected_ground = ground_image_reward - (
                0.25 * env.simulator.sim_time if overlap else 0
            )
            assert totals == pytest.approx(
                {"ground": expected_ground, "observer": rso_image_reward, "debris": 0}
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
