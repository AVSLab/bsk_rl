"""Completion logic exercised against real Basilisk clocks, events, and spacecraft."""

from dataclasses import replace
import numpy as np
import pytest
from bsk_rl import NO_ACTION
from bsk_rl.data.completion_catalog import CompletionRecord
from examples.multiagent_imaging.config import (
    MultiAgentImagingConfig,
    GLOBAL_FEATURES,
    TARGET_FEATURES,
)
from examples.multiagent_imaging.environment import build_environment


def config(**kwargs):
    return MultiAgentImagingConfig(
        **dict(
            n_targets=4,
            n_candidates=4,
            episode_duration_s=60,
            max_step_duration_s=5,
            charge_duration_s=100,
            information_case="ideal_completion",
            **kwargs,
        )
    )


@pytest.mark.parametrize("mode", ["conflict", "continuous"])
def test_local_completion_invalidates_only_active_matching_task(mode):
    env = build_environment(config(retasking_mode=mode))
    try:
        env.reset(seed=33)
        left, right = env.sensing_satellites
        target_id = left.completion_candidates.targets[0].id
        env.step({left.name: 5, right.name: 0})
        assert env.simulator.sim_time == 5
        right_deadline = right._timed_terminal_time
        right.data_store.catalog.merge_record(
            CompletionRecord("peer_done", right.name, target_id, 0, 0, 5, True), 5
        )
        # Busy agents send the sentinel in conflict mode; continuous agents make
        # a deliberate continue decision at the exact same physical boundary.
        action = NO_ACTION if mode == "conflict" else 4
        observations, _, _, _, infos = env.step({left.name: action, right.name: action})
        assert env.simulator.sim_time == 10
        assert left.requires_retasking
        assert left.completion_task.invalidated
        assert left._active_image_rso_action is None
        assert all(
            t is None or t.id != target_id for t in left.completion_candidates.targets
        )
        assert right._timed_terminal_time == right_deadline
        assert right.requires_retasking == (mode == "continuous")
        assert observations[left.name][25] == 0
        assert infos[left.name]["retask_reason"] == "known_completion"
    finally:
        env.close()


def test_deliberate_continue_and_same_target_preserve_hold_and_deadline():
    env = build_environment(config(retasking_mode="continuous"))
    try:
        env.reset(seed=33)
        sensor = env.sensing_satellites[0]
        env.step({"sensor_0": 5, "sensor_1": 0})
        active = sensor._active_image_rso_action
        target_id = sensor.completion_task.target_id
        start = active._attempt_start_time
        deadline = sensor._timed_terminal_time
        env.step({"sensor_0": 4, "sensor_1": 4})
        slot = next(
            i
            for i, t in enumerate(sensor.completion_candidates.targets)
            if t and t.id == target_id
        )
        env.step({"sensor_0": 5 + slot, "sensor_1": 4})
        assert sensor._active_image_rso_action is active
        assert active._attempt_start_time == start == 0
        assert sensor._timed_terminal_time == deadline
        assert sensor.completion_task.start == 0
        assert env.decision_counts[sensor.name] == 3
    finally:
        env.close()


def test_finite_radio_respects_duration_and_reception_boundary():
    cfg = replace(
        config(), information_case="completion", link_mode="ideal", message_delay_s=7
    )
    env = build_environment(cfg)
    try:
        env.reset(seed=33)
        left, right = env.sensing_satellites
        target_id = next(iter(left.data_store.catalog.targets))
        left.data_store.catalog.merge_record(
            CompletionRecord("done", left.name, target_id, 0, 0, 0, True), 0
        )
        # Heartbeats subdivide transmission without any retasking of either agent.
        env.step({left.name: 3, right.name: 0})
        assert not right.data_store.catalog.records
        while env.simulator.sim_time < 30:
            env.step({left.name: NO_ACTION, right.name: NO_ACTION})
        assert left.requires_retasking
        assert not right.data_store.catalog.records
        assert env.communicator.pending[0].ready_at == 37
        env.step({left.name: 0, right.name: NO_ACTION})
        env.step({left.name: NO_ACTION, right.name: NO_ACTION})
        assert env.simulator.sim_time == 37
        assert right.data_store.catalog.received_at["done"] == 37
        assert env.communication_time[left.name] == pytest.approx(30)
        assert env.rewarder.per_sensor_metrics[left.name]["communication_actions"] == 1
        assert not right.data_store.products
    finally:
        env.close()


def test_completed_targets_are_masked_with_no_ineligible_fallback():
    env = build_environment(config())
    try:
        env.reset(seed=33)
        for sensor in env.sensing_satellites:
            for tid in sensor.data_store.catalog.targets:
                sensor.data_store.catalog.merge_record(
                    CompletionRecord(f"done_{tid}", "sensor_0", tid, 0, 0, 0, True), 0
                )
        observations, *_ = env.step({"sensor_0": 0, "sensor_1": 0})
        # Ask for an actual decision observation after receiver-local refresh.
        env.generate_obs_retasking_only = False
        observations = env._get_obs()
        for sensor in env.sensing_satellites:
            chunks = observations[sensor.name][GLOBAL_FEATURES:].reshape(
                4, TARGET_FEATURES
            )
            assert not chunks.any()
            assert sensor.completion_candidates.targets == (None,) * 4
        with pytest.raises(ValueError, match="masked"):
            env.step({"sensor_0": 5, "sensor_1": NO_ACTION})
    finally:
        env.close()


def test_truncated_busy_agents_have_real_bootstrap_observations():
    env = build_environment(replace(config(), episode_duration_s=10))
    try:
        env.reset(seed=33)
        env.step({"sensor_0": 0, "sensor_1": 0})
        observations, _, _, truncated, _ = env.step(
            {"sensor_0": NO_ACTION, "sensor_1": NO_ACTION}
        )
        assert all(truncated.values())
        assert all(np.any(obs) for obs in observations.values())
        assert all(obs[25] == 1 for obs in observations.values())
    finally:
        env.close()


def test_vizard_contains_passive_spacecraft_with_propagating_state(
    tmp_path, monkeypatch
):
    from bsk_rl.utils import vizard

    # Restore process-global visualization switches before later headless tests.
    monkeypatch.setattr(vizard, "VIZARD_PATH", vizard.VIZARD_PATH)
    monkeypatch.setattr(vizard, "VIZINSTANCE", vizard.VIZINSTANCE)
    env = build_environment(
        config(), vizard_dir=str(tmp_path), vizard_settings={"multiagent_vizard": True}
    )
    try:
        env.reset(seed=33)
        viz = env.simulator.vizInstance
        names = [str(sc.spacecraftName) for sc in viz.scData]
        assert set(names) == {s.name for s in env.satellites}
        assert env.possible_agents == ["sensor_0", "sensor_1"]
        assert not any(
            "RSO marker" in str(location.stationName) for location in viz.locations
        )
        before = {
            s.name: np.array(s.dynamics.scObject.scStateOutMsg.read().r_BN_N)
            for s in env.passive_satellites
        }
        env.step({"sensor_0": 0, "sensor_1": 0})
        for target in env.passive_satellites:
            after = np.array(target.dynamics.scObject.scStateOutMsg.read().r_BN_N)
            assert np.linalg.norm(after - before[target.name]) > 1000
    finally:
        env.close()


def test_global_truth_alone_does_not_retask_and_waste_stops_at_local_receipt():
    from bsk_rl.data.multiagent_rso_data import ImageProductRecord

    env = build_environment(replace(config(), information_case="independent"))
    try:
        env.reset(seed=33)
        left, right = env.sensing_satellites
        tid = left.completion_candidates.targets[0].id
        env.step({left.name: 5, right.name: 0})
        # Seed evaluator-only truth, simulating completion at t=5. It must not
        # become the left actor's knowledge through the global reward machinery.
        product = ImageProductRecord(
            "remote", right.name, tid, 0, None, 1, right.name, completion_time=5
        )
        env.rewarder._team_accounting.capture_attempts.append(product)
        env.step({left.name: NO_ACTION, right.name: NO_ACTION})
        assert not left.requires_retasking
        assert not left.data_store.catalog.records
        assert env.coordination_metrics()["duplicate_sensor_time_s"] == pytest.approx(5)
        left.data_store.catalog.merge_record(
            CompletionRecord("remote", right.name, tid, 0, 0, 5, True), 10
        )
        env.step({left.name: NO_ACTION, right.name: NO_ACTION})
        assert left.requires_retasking
        # Arrival is t=10, so later work before the next boundary cannot extend
        # the prospectus's pre-awareness duplicate interval.
        metrics = env.coordination_metrics()
        assert metrics["duplicate_sensor_time_s"] == pytest.approx(5)
        assert metrics["interrupted_nonduplicate_sensor_time_s"] == 0
        assert metrics["wasted_time_fraction"] == pytest.approx(5 / 30)
    finally:
        env.close()


def test_implicit_training_resets_use_a_reproducible_episode_seed_sequence():
    initial = []
    for _ in range(2):
        env = build_environment(replace(config(), seed=21))
        try:
            env.reset()
            assert env.seed == 21
            first = np.array(env.passive_satellites[0].dynamics.r_BN_N)
            env.reset(
                seed=42
            )  # RLlib's explicit API check must not consume the sequence.
            env.reset()
            assert env.seed == 22
            second = np.array(env.passive_satellites[0].dynamics.r_BN_N)
            initial.append((first, second))
        finally:
            env.close()
    np.testing.assert_array_equal(initial[0], initial[1])
    assert not np.array_equal(*initial[0])
