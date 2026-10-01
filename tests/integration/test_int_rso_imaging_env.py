"""Real Basilisk storage/hold, delivery, multi-imager identity, and reset contracts."""

from dataclasses import replace
from typing import ClassVar

import numpy as np
import pytest

from bsk_rl import ConstellationTasking, act, data, obs, sats, scene
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.rso_imaging import illumination_factor

EPOCH = "2026 OCT 01 00:00:00.000 (UTC)"


class Camera(sats.Satellite):
    dyn_type = (dyn.SpaceToSpaceImagingDynModel, dyn.GroundStationDynModel)
    fsw_type = fsw.SpaceToSpaceImagingFSWModel
    observation_spec: ClassVar[list[obs.Observation]] = [
        obs.SatProperties(dict(prop="storage_level")),
        obs.RSOTargetProperties(
            dict(prop="valid"),
            dict(prop="priority"),
            dict(prop="target_distance"),
            dict(prop="target_illumination_factor"),
            n_ahead_observe=2,
        ),
    ]
    action_spec: ClassVar[list[act.Action]] = [
        act.ImageRSO(
            2,
            max_duration=30,
            min_pointing_hold_s=10,
            require_illumination_during_hold=False,
        ),
        act.Downlink(duration=30),
        act.Charge(duration=10),
    ]


class PassiveTarget(sats.Satellite):
    dyn_type = dyn.RSOTargetDynModel
    fsw_type = fsw.FSWModel
    observation_spec: ClassVar[list[obs.Observation]] = [obs.Time()]
    action_spec: ClassVar[list[act.Action]] = [act.Drift(duration=1e9)]


class ReorderedScene(scene.RSOTargets):
    def reset_during_sim_init(self):
        for observer in self.imagers:
            targets = list(self.targets_by_id.values())
            if observer.name == "camera_b":
                targets.reverse()
            for target in targets:
                observer.dynamics.bind_rso_target(target)


def make_env(
    two=False,
    reverse=False,
    variable=True,
    hold_s=10,
    illumination=False,
    events=(),
    composed=False,
    image_size=1000,
    sim_rate=1,
):
    definitions = (
        scene.RSOTarget("83", "debris", (7e6 + 1000, 0, 0), (0, 7500, 0), priority=2),
        scene.RSOTarget(
            "rso/209", "uncooperative", (7e6 + 2000, 0, 0), (0, 7500, 0), priority=1
        ),
    )
    catalog = scene.RSOTargetCatalog(definitions, EPOCH, events)
    names = ["camera_a", "camera_b"] if two else ["camera_a"]
    cameras = [
        Camera(
            name,
            sat_args=dict(
                rN=[7e6, 0, 0],
                vN=[0, 7500, 0],
                oe=None,
                sigma_init=[0, 0, 0],
                omega_init=[0, 0, 0],
                wheelSpeeds=[0.0, 0.0, 0.0],
                imageAttErrorRequirement=2.0,
                instrumentBaudRate=image_size,
                transmitterBaudRate=-100,
                batteryStorageCapacity=1e9,
                storedCharge_Init=1e9,
            ),
        )
        for name in names
    ]
    for camera in cameras:
        imaging = camera.action_builder.action_spec[0]
        imaging.variable_duration_imaging = variable
        imaging.min_pointing_hold_s = hold_s
        imaging.require_illumination_during_hold = illumination
    satellites = [PassiveTarget("debris"), *cameras, PassiveTarget("uncooperative")]
    if reverse:
        satellites.reverse()
    rewarder = data.RSOImageReward(
        multi_imager_credit="shared",
        quality_threshold=0,
        acquisition_reward_fn=lambda record, target: 0.9 * target.priority,
        delivery_reward_fn=lambda record, target: 0.1 * target.priority,
    )
    if composed:
        resource = data.ResourceReward(
            reward_weight=0,
            resource_fn=lambda sat: sat.dynamics.battery_charge_fraction
            if hasattr(sat.dynamics, "battery_charge_fraction")
            else 0,
        )
        rewarder = (resource, rewarder)
    return ConstellationTasking(
        satellites=satellites,
        scenario=ReorderedScene(catalog, names),
        rewarder=rewarder,
        world_args=dict(utc_init=EPOCH, gsMinimumElevation=np.radians(-90)),
        time_limit=120,
        max_step_duration=1,
        sim_rate=sim_rate,
        log_level="ERROR",
    )


def camera(env, name="camera_a"):
    return next(sat for sat in env.satellites if sat.name == name)


def actions(env, imager_action=None, initial=False):
    return {
        sat.name: imager_action
        if sat.name in env.scenario.imager_names
        else (0 if initial else None)
        for sat in env.satellites
    }


def rso_store(satellite):
    store = satellite.data_store
    if hasattr(store, "data_stores"):
        return next(
            component
            for component in store.data_stores
            if isinstance(component, data.RSOImageStore)
        )
    return store


def test_spacecraft_imaging_runs_without_ground_stations():
    class ImagingOnlyCamera(sats.Satellite):
        dyn_type = dyn.SpaceToSpaceImagingDynModel
        fsw_type = fsw.SpaceToSpaceImagingFSWModel
        observation_spec: ClassVar[list[obs.Observation]] = [obs.RSOTargetProperties()]
        action_spec: ClassVar[list[act.Action]] = [
            act.ImageRSO(
                2, min_pointing_hold_s=0, require_illumination_during_hold=False
            )
        ]

    template = make_env()
    scanner = ImagingOnlyCamera(
        "camera_a", sat_args=camera(template).sat_args_generator
    )
    env = ConstellationTasking(
        satellites=[PassiveTarget("debris"), scanner, PassiveTarget("uncooperative")],
        scenario=scene.RSOTargets(template.scenario.catalog, [scanner.name]),
        rewarder=data.RSOImageReward(quality_threshold=0),
        world_args=dict(utc_init=EPOCH),
        time_limit=30,
        max_step_duration=10,
        log_level="ERROR",
    )
    try:
        env.reset(seed=0)
        assert not hasattr(env.simulator.world, "groundStations")
        env.step(actions(env, "83", True))
        observer = camera(env)
        assert observer.dynamics.storage_level == 1000
        assert len(observer.fsw.rso_capture.records) == 1
        assert observer.dynamics.instrument.nodeStatusInMsg.isLinked()
    finally:
        env.close()


@pytest.mark.parametrize("use_default_epoch", [False, True])
def test_unpinned_catalog_follows_realized_environment_epoch_on_each_reset(
    use_default_epoch, tmp_path
):
    env = make_env()
    definition = replace(env.scenario.catalog, utc_init=None)
    env.scenario.catalog = definition
    if use_default_epoch:
        env.world_args_generator = env.world_type.default_world_args()
    else:
        epochs = iter((EPOCH, "2026 OCT 02 00:00:00.000 (UTC)"))
        env.world_args_generator["utc_init"] = lambda: next(epochs)
    try:
        snapshots = []
        for seed in (0, 1):
            env.reset(seed=seed)
            realized = env.scenario.catalog
            assert realized.utc_init == env.world_args["utc_init"]
            assert realized.utc_init == env.scenario.utc_init
            assert realized.targets == definition.targets
            assert definition.utc_init is None
            realized.save(tmp_path / f"catalog_{seed}.json")
            assert (
                scene.RSOTargetCatalog.load(tmp_path / f"catalog_{seed}.json")
                == realized
            )
            snapshots.append(realized)
        assert snapshots[0].utc_init != snapshots[1].utc_init
        assert env.scenario.catalog is snapshots[1]
    finally:
        env.close()


@pytest.mark.parametrize("variable", [False, True])
@pytest.mark.parametrize("sim_rate", [0.5, 1.0])
def test_real_storage_stays_empty_until_hold_and_captures_once(variable, sim_rate):
    env = make_env(variable=variable, sim_rate=sim_rate)
    env.reset(seed=0)
    observer = camera(env)
    levels = []
    env.step(actions(env, 0, True))
    for _ in range(18):
        levels.append((env.simulator.sim_time, observer.dynamics.storage_level))
        env.step(actions(env))
    record = observer.fsw.rso_capture.records[0]
    assert record.hold_valid_time_s >= 10
    assert all(level == 0 for time, level in levels if time < record.capture_time)
    assert record.capture_time >= 10
    assert observer.dynamics.storage_level == 1000
    assert len(observer.fsw.rso_capture.records) == 1
    assert rso_store(observer).remaining_bits[record.record_id] == 1000
    env.close()


@pytest.mark.parametrize("reverse", [False, True])
def test_two_imagers_distinct_access_indices_and_credit(reverse):
    env = make_env(two=True, reverse=reverse)
    env.reset(seed=0)
    env.step(actions(env, 0, True))
    for _ in range(15):
        env.step(actions(env))
    assert sum(env.rewarder.cum_reward.values()) == pytest.approx(1.8)
    for name in env.scenario.imager_names:
        observer = camera(env, name)
        record = observer.fsw.rso_capture.records[0]
        assert record.target_id == "83"
        assert record.source_imager == name
        assert env.rewarder.cum_reward[name] == pytest.approx(0.9)
        assert observer.dynamics.rso_access_messages["83"].read().hasAccess == 1
    assert env.scenario.pending["83"] == {"camera_a:1", "camera_b:1"}
    env.close()


def test_real_partial_downlink_contact_interruption_and_complete_delivery():
    env = make_env(composed=True)
    env.reset(seed=0)
    env.step(actions(env, 0, True))
    for _ in range(12):
        env.step(actions(env))
    observer = camera(env)
    store = rso_store(observer)
    assert len(store.products) == 1
    env.step(actions(env, 2))
    env.step(actions(env))
    env.step(actions(env))
    assert 0 < observer.dynamics.storage_level < 1000
    assert not store.data.deliveries
    env.step(actions(env, 3))
    env.step(actions(env))
    env.step(actions(env, 2))
    for _ in range(20):
        env.step(actions(env))
        if store.data.deliveries:
            break
    assert len(store.data.deliveries) == 1
    assert not store.products
    assert observer.dynamics.storage_level == 0
    assert env.rewarder.cum_reward["camera_a"] == pytest.approx(2.0)
    env.close()


def test_timeout_cancellation_and_illumination_gate():
    for illumination, hold_s in [(False, 100), (True, 10)]:
        env = make_env(illumination=illumination, hold_s=hold_s)
        env.reset(seed=0)
        env.step(actions(env, 0, True))
        for _ in range(6):
            env.step(actions(env))
        env.step(actions(env, 3))
        assert camera(env).dynamics.storage_level == 0
        assert camera(env).fsw.rso_capture.target is None
        env.close()


def test_repeat_reset_and_priority_boundary_ordering():
    events = (
        scene.RSOPriorityEvent(0, (("83", 3),)),
        scene.RSOPriorityEvent(5, (("83", 7),)),
    )
    env = make_env(events=events)
    for _ in range(2):
        initial, _ = env.reset(seed=0)
        assert env.scenario.targets_by_id["83"].priority == 3
        assert env.scenario.pending == {}
        assert camera(env).fsw.rso_capture.records == []
        assert initial["camera_a"][2] == 3
        env.step(actions(env, 0, True))
        for _ in range(15):
            env.step(actions(env))
        assert env.scenario.targets_by_id["83"].priority == 7
        assert env.scenario.applied_events == [(0, 0.0), (1, 5.0)]
        assert env.rewarder.cum_reward["camera_a"] == pytest.approx(6.3)
    env.close()


@pytest.mark.parametrize("mode", ["continuous", "cumulative"])
def test_real_access_interruption_resets_only_continuous_hold(mode):
    env = make_env()
    camera(env).action_builder.action_spec[0].hold_mode = mode
    env.reset(seed=0)
    observer = camera(env)
    env.step(actions(env, 0, True))
    for _ in range(5):
        env.step(actions(env))
    held_before = observer.fsw.rso_capture.held_s
    assert held_before > 0
    observer.dynamics.targetLocation.maximumRange = 1
    for _ in range(3):
        env.step(actions(env))
    assert observer.dynamics.storage_level == 0
    assert observer.fsw.rso_capture.held_s == (
        0 if mode == "continuous" else held_before
    )
    observer.dynamics.targetLocation.maximumRange = -1
    for _ in range(20):
        env.step(actions(env))
        if observer.fsw.rso_capture.records:
            break
    record = observer.fsw.rso_capture.records[0]
    assert record.hold_valid_time_s >= 10
    assert record.capture_time >= (19 if mode == "continuous" else 15)
    assert observer.dynamics.storage_level == record.size_bits
    env.close()


def test_switch_timeout_and_zero_hold_are_physical():
    env = make_env()
    env.reset(seed=0)
    env.step(actions(env, 0, True))
    for _ in range(6):
        env.step(actions(env))
    env.step(actions(env, "rso/209"))
    assert camera(env).fsw.rso_capture.target.id == "rso/209"
    assert camera(env).fsw.rso_capture.held_s == 0
    for _ in range(14):
        env.step(actions(env))
    records = camera(env).fsw.rso_capture.records
    assert [record.target_id for record in records] == ["rso/209"]
    assert records[0].capture_time >= 17
    env.close()

    env = make_env(hold_s=100)
    env.reset(seed=0)
    env.step(actions(env, 0, True))
    for _ in range(35):
        env.step(actions(env))
    assert camera(env).fsw.rso_capture.target is None
    assert camera(env).dynamics.storage_level == 0
    env.close()

    env = make_env(hold_s=0)
    env.reset(seed=0)
    env.step(actions(env, 0, True))
    assert camera(env).dynamics.storage_level == 0  # no preceding valid guidance
    for _ in range(4):
        env.step(actions(env))
    record = camera(env).fsw.rso_capture.records[0]
    assert record.hold_valid_time_s == 0
    assert camera(env).dynamics.storage_level == 1000
    env.close()


def test_real_illumination_and_capacity_block_storage():
    env = make_env(illumination=True)
    env.reset(seed=0)
    observer = camera(env)
    target_dyn = env.scenario.targets_by_id["83"].target_spacecraft.dynamics
    env.step(actions(env, 0, True))
    for _ in range(20):
        env.step(actions(env))
    assert (
        illumination_factor(
            target_dyn.world.eclipseObject.eclipseOutMsgs[
                target_dyn.eclipse_index
            ].read()
        )
        == 0
    )
    assert observer.dynamics.storage_level == 0
    env.close()

    env = make_env()
    env.reset(seed=0)
    observer = camera(env)
    observer.dynamics.storageUnit.storageCapacity = 500
    env.step(actions(env, 0, True))
    for _ in range(15):
        env.step(actions(env))
    assert observer.dynamics.storage_level == 0
    assert observer.fsw.rso_capture.records == []
    env.close()


def test_public_example_and_manifest_replay(tmp_path):
    import runpy
    from pathlib import Path

    example = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/space_to_space_rso_imaging_demo.py")
    )
    result = example["run_demo"](tmp_path)
    assert result["captures"] == result["deliveries"] == 1
    catalog = scene.RSOTargetCatalog.load(tmp_path / "catalog.json")
    env = example["build_environment"](catalog, amos_profile=True)
    initial, _ = env.reset(seed=0)
    assert initial["imager"].shape == (124,)
    assert camera(env, "imager").action_space.n == 13
    manifest = example["mission_manifest"](env)
    env.close()
    replay = example["replay_environment"](manifest)

    # Insert unrelated draws into reset's real path before satellite generation.
    def extra_draws(satellites):
        np.random.random(200)
        return {}

    replay.sat_arg_randomizer = extra_draws
    repeated, _ = replay.reset(seed=999)
    np.testing.assert_array_equal(initial["imager"], repeated["imager"])
    assert replay.scenario.catalog == catalog
    for satellite in replay.satellites:
        assert (
            satellite.sat_args["rN"] == manifest["satellite_args"][satellite.name]["rN"]
        )
    replay.close()


def test_priority_event_follows_reward_at_the_same_boundary():
    event = scene.RSOPriorityEvent(5, (("83", 7),))
    env = make_env(hold_s=3, events=(event,))
    env.reset(seed=0)
    env.step(actions(env, 0, True))
    for _ in range(4):
        env.step(actions(env))
    assert env.simulator.sim_time == 5
    assert camera(env).fsw.rso_capture.records[0].capture_time == 5
    assert env.rewarder.cum_reward["camera_a"] == pytest.approx(1.8)
    assert env.scenario.targets_by_id["83"].priority == 7
    assert env.scenario.applied_events == [(0, 5.0)]
    env.close()


def test_multiple_partition_completions_in_one_physical_step():
    env = make_env(hold_s=0)
    env.reset(seed=0)
    env.step(actions(env, "83", True))
    for _ in range(3):
        env.step(actions(env))
    env.step(actions(env, "rso/209"))
    for _ in range(3):
        env.step(actions(env))
    observer = camera(env)
    assert observer.dynamics.storage_level == 2000
    assert len(observer.data_store.products) == 2
    env.simulator.max_step_duration = 30
    _, reward, _, _, _ = env.step(actions(env, 2))
    assert observer.dynamics.storage_level == 0
    assert len(observer.data_store.data.deliveries) == 2
    assert reward["camera_a"] == pytest.approx(0.3)
    env.close()


def test_multiple_imagers_keep_independent_deadlines():
    env = make_env(two=True, hold_s=100)
    env.max_step_duration = 30
    try:
        env.reset(seed=0)
        env.step(actions(env, 0, True))
        names = [sat._timed_terminal_event_name for sat in env.scenario.imagers]
        assert set(names) == {"timed_terminal_camera_a", "timed_terminal_camera_b"}
        assert all(sat.requires_retasking for sat in env.scenario.imagers)
        assert all(sat.fsw.rso_capture.target is None for sat in env.scenario.imagers)
    finally:
        env.close()


def test_earth_occlusion_prevents_acquisition_of_an_explicitly_tasked_target():
    from dataclasses import replace

    env = make_env(hold_s=0)
    catalog = env.scenario.catalog
    opposite = replace(catalog.targets[0], rN=(-7e6, 0, 0), vN=(0, -7500, 0))
    env.scenario.catalog = scene.RSOTargetCatalog((opposite, catalog.targets[1]), EPOCH)
    env.reset(seed=0)
    env.step(actions(env, "83", True))
    for _ in range(20):
        env.step(actions(env))
    observer = camera(env)
    assert observer.dynamics.rso_access_messages["83"].read().hasAccess == 0
    assert observer.dynamics.storage_level == 0
    assert observer.fsw.rso_capture.records == []
    assert env.scenario.targets_by_id["83"].target_spacecraft.is_alive()
    env.close()


def test_evaluation_physical_products_reward_and_replay(tmp_path, monkeypatch):
    import csv
    import importlib
    import json
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    evaluation = importlib.import_module("space_to_space_rso_imaging_evaluation")
    first = tmp_path / "first"
    result = evaluation.evaluate(first, horizon=240, plots=False)
    assert result["captures"] >= 1 and result["deliveries"] >= 1
    assert result["status"] == "time_limit"
    captures = list(csv.DictReader((first / "captures.csv").open()))
    deliveries = list(csv.DictReader((first / "deliveries.csv").open()))
    assert all(float(row["hold_valid_time_s"]) >= 10 for row in captures)
    acquired_bits = sum(float(row["size_bits"]) for row in captures)
    delivered_bits = sum(float(row["size_bits"]) for row in deliveries)
    assert acquired_bits == delivered_bits + result["storage_bits"]
    assert result["acquisition_reward"] + result["delivery_reward"] == pytest.approx(
        result["total_reward"]
    )
    replay = tmp_path / "replay"
    repeated = evaluation.evaluate(
        replay, manifest_path=first / "mission.json", seed=999, plots=False
    )
    np.testing.assert_array_equal(
        np.load(first / "initial_observation.npy"),
        np.load(replay / "initial_observation.npy"),
    )
    assert repeated["captures"] == result["captures"]
    assert repeated["deliveries"] == result["deliveries"]
    assert (first / "decisions.csv").read_text() == (
        replay / "decisions.csv"
    ).read_text()
    shortened = evaluation.evaluate(
        tmp_path / "shortened",
        manifest_path=first / "mission.json",
        horizon=30,
        plots=False,
    )
    assert shortened["time_s"] == pytest.approx(30, abs=1e-9)
    assert shortened["captures"] == shortened["deliveries"] == 0
    assert (
        json.loads((first / "mission.json").read_text())["catalog"]
        == json.loads((replay / "mission.json").read_text())["catalog"]
    )


def test_evaluation_plots_and_step_budget(tmp_path, monkeypatch):
    import importlib
    from pathlib import Path

    pytest.importorskip("matplotlib")
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    evaluation = importlib.import_module("space_to_space_rso_imaging_evaluation")
    output = tmp_path / "plots"
    result = evaluation.evaluate(output, horizon=240, max_steps=1)
    assert result["status"] == "max_steps"
    assert result["captures"] == 1 and result["deliveries"] == 0
    for name in ("performance.png", "performance.pdf", "actions.png", "actions.pdf"):
        assert (output / name).stat().st_size > 1000
