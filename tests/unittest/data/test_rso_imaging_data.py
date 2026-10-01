"""Ownership, complete delivery, centralized service, and reward composition."""

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from bsk_rl.data import (
    NoReward,
    RSOImageData,
    RSOImageRecord,
    RSOImageReward,
    RSOImageStore,
)
from bsk_rl.data.composition import ComposedReward


def weighted_reward(alpha=0.1, **kwargs):
    """Keep stage weighting in the test scenario, rather than in the rewarder."""
    return RSOImageReward(
        acquisition_reward_fn=lambda record, target: (1 - alpha) * target.priority,
        delivery_reward_fn=lambda record, target: alpha * target.priority,
        **kwargs,
    )


def scene():
    return NS(
        imager_names=("cam-a", "cam-b"),
        targets_by_id={"odd/83": NS(priority=2)},
        partition_name=lambda _: "partition",
        pending={},
        cooldown_until={},
        revision=0,
    )


def image(key="a:1", owner="cam-a", time=10, quality=0.8):
    return RSOImageRecord(key, "odd/83", owner, time, 100, quality, 10)


def test_custom_stage_rewards_receive_the_record_and_current_target():
    mission = scene()
    rewarder = RSOImageReward(
        multi_imager_credit="shared",
        acquisition_reward_fn=lambda record, target: record.quality * target.priority,
        delivery_reward_fn=lambda record, target: target.priority
        / (record.delivery_time - record.capture_time),
    )
    rewarder.link_scenario(mission)
    rewarder.reset_overwrite_previous()
    captured = image()
    assert rewarder.calculate_reward(
        {"cam-a": RSOImageData({captured.record_id: captured})}
    ) == {"cam-a": 1.6}
    mission.targets_by_id[captured.target_id].priority = 5
    delivered = replace(captured, delivery_time=20)
    assert rewarder.calculate_reward(
        {"cam-a": RSOImageData(deliveries={captured.record_id: delivered})}
    ) == {"cam-a": 0.5}


def test_default_reward_is_withheld_until_complete_delivery():
    mission = scene()
    rewarder = RSOImageReward(multi_imager_credit="shared")
    rewarder.link_scenario(mission)
    rewarder.reset_overwrite_previous()
    captured = image()
    assert rewarder.calculate_reward(
        {"cam-a": RSOImageData({captured.record_id: captured})}
    ) == {"cam-a": 0}
    delivered = replace(captured, delivery_time=20)
    assert rewarder.calculate_reward(
        {"cam-a": RSOImageData(deliveries={captured.record_id: delivered})}
    ) == {"cam-a": 2}


def test_partial_and_multiple_delivery_ledger():
    satellite = NS(name="cam-a", simulator=NS(sim_time=40))
    store = RSOImageStore(satellite, scenario=scene())
    first, second = image(), image("a:2", time=20)
    captured = store.compare_log_states(
        ({"partition": 0}, ()), ({"partition": 200}, (first, second))
    )
    assert len(captured.captures) == 2
    partial = store.compare_log_states(
        ({"partition": 200}, (first, second)), ({"partition": 160}, (first, second))
    )
    assert partial.deliveries == {}
    assert store.remaining_bits[first.record_id] == 60
    completed = store.compare_log_states(
        ({"partition": 160}, (first, second)), ({"partition": 0}, (first, second))
    )
    assert len(completed.deliveries) == 2
    assert store.products == {}
    assert store.remaining_bits == {}


def test_owner_reward_priority_and_cooldown():
    mission = scene()
    rewarder = weighted_reward(multi_imager_credit="shared", alpha=0.1, cooldown_s=100)
    rewarder.link_scenario(mission)
    rewarder.reset_overwrite_previous()
    captured = image()
    assert rewarder.calculate_reward(
        {"cam-a": RSOImageData({captured.record_id: captured}), "cam-b": RSOImageData()}
    ) == {"cam-a": 1.8, "cam-b": 0}
    assert mission.pending["odd/83"] == {captured.record_id}
    assert rewarder.last_reward_components == {
        "acquisition": {"cam-a": 1.8, "cam-b": 0},
        "delivery": {"cam-a": 0, "cam-b": 0},
    }
    mission.targets_by_id["odd/83"].priority = 5
    delivered = replace(captured, delivery_time=40)
    result = rewarder.calculate_reward(
        {
            "cam-a": RSOImageData(deliveries={delivered.record_id: delivered}),
            "cam-b": RSOImageData(),
        }
    )
    assert result == {"cam-a": 0.5, "cam-b": 0}
    assert rewarder.last_reward_components == {
        "acquisition": {"cam-a": 0, "cam-b": 0},
        "delivery": {"cam-a": 0.5, "cam-b": 0},
    }
    assert not mission.pending["odd/83"]
    assert mission.cooldown_until["odd/83"] == 110
    assert rewarder.calculate_reward(
        {"cam-a": RSOImageData(deliveries={delivered.record_id: delivered})}
    ) == {"cam-a": 0}
    rewarder.reset_overwrite_previous()
    assert rewarder.last_reward_components == {"acquisition": {}, "delivery": {}}


def test_simultaneous_credit_and_quality():
    rewarder = weighted_reward(multi_imager_credit="shared")
    rewarder.link_scenario(scene())
    rewarder.reset_overwrite_previous()
    first, second = image(), image("b:1", owner="cam-b")
    result = rewarder.calculate_reward(
        {
            "cam-a": RSOImageData({first.record_id: first}),
            "cam-b": RSOImageData({second.record_id: second}),
        }
    )
    assert result == {"cam-a": 0.9, "cam-b": 0.9}
    bad = image("a:bad", time=200, quality=0.1)
    assert rewarder.calculate_reward({"cam-a": RSOImageData({bad.record_id: bad})}) == {
        "cam-a": 0
    }
    with pytest.raises(ValueError, match="owner"):
        rewarder.calculate_reward({"wrong": RSOImageData({first.record_id: first})})


@pytest.mark.parametrize("reverse", [False, True])
def test_composed_reward_retains_scene_and_store_ownership(reverse):
    mission = scene()
    rso_reward = weighted_reward(
        multi_imager_credit="shared", alpha=0.3, cooldown_s=123
    )
    components = (NoReward(), rso_reward) if reverse else (rso_reward, NoReward())
    composed = ComposedReward(*components)
    composed.link_scenario(mission)
    composed.reset_overwrite_previous()
    satellite = NS(name="cam-b")
    composed.create_data_store(satellite)
    store = next(
        store
        for store in satellite.data_store.data_stores
        if isinstance(store, RSOImageStore)
    )
    assert store.scenario is mission
    assert store.is_imager
    assert rso_reward.delivery_reward_fn(
        image(), mission.targets_by_id["odd/83"]
    ) == pytest.approx(0.6)
    assert rso_reward.cooldown_s == 123


def test_cooldown_boundary_and_failed_quality_do_not_double_credit():
    mission = scene()
    rewarder = weighted_reward(multi_imager_credit="shared", alpha=0.1, cooldown_s=100)
    rewarder.link_scenario(mission)
    rewarder.reset_overwrite_previous()
    first = image()
    rewarder.calculate_reward({"cam-a": RSOImageData({first.record_id: first})})
    rewarder.calculate_reward(
        {
            "cam-a": RSOImageData(
                deliveries={first.record_id: replace(first, delivery_time=20)}
            )
        }
    )
    before = image("a:2", time=109)
    assert (
        rewarder.calculate_reward({"cam-a": RSOImageData({before.record_id: before})})[
            "cam-a"
        ]
        == 0
    )
    boundary = image("a:3", time=110)
    assert (
        rewarder.calculate_reward(
            {"cam-a": RSOImageData({boundary.record_id: boundary})}
        )["cam-a"]
        == 1.8
    )
    bad = image("a:bad", time=300, quality=0.1)
    assert (
        rewarder.calculate_reward(
            {
                "cam-a": RSOImageData(
                    {bad.record_id: bad},
                    {bad.record_id: replace(bad, delivery_time=400)},
                )
            }
        )["cam-a"]
        == 0
    )
    assert mission.cooldown_until["odd/83"] == 110
    assert bad.record_id not in mission.pending["odd/83"]


def test_staggered_imagers_do_not_duplicate_credit_while_service_is_pending():
    rewarder = weighted_reward(multi_imager_credit="shared", cooldown_s=0)
    mission = scene()
    rewarder.link_scenario(mission)
    rewarder.reset_overwrite_previous()
    first, second = image(), image("b:1", owner="cam-b", time=15)
    assert rewarder.calculate_reward(
        {"cam-a": RSOImageData({first.record_id: first})}
    ) == {"cam-a": 1.8}
    assert rewarder.calculate_reward(
        {"cam-b": RSOImageData({second.record_id: second})}
    ) == {"cam-b": 0}
    assert rewarder.calculate_reward(
        {
            "cam-a": RSOImageData(
                deliveries={first.record_id: replace(first, delivery_time=30)}
            )
        }
    ) == {"cam-a": 0.2}
    assert rewarder.calculate_reward(
        {
            "cam-b": RSOImageData(
                deliveries={second.record_id: replace(second, delivery_time=35)}
            )
        }
    ) == {"cam-b": 0}
    next_service = image("b:2", owner="cam-b", time=40)
    assert rewarder.calculate_reward(
        {"cam-b": RSOImageData({next_service.record_id: next_service})}
    ) == {"cam-b": 1.8}


def test_multi_imager_scientific_choice_is_required_and_per_owner_mode_works():
    with pytest.raises(ValueError, match="explicit multi_imager_credit"):
        weighted_reward().link_scenario(scene())
    rewarder = weighted_reward(multi_imager_credit="per_imager")
    rewarder.link_scenario(scene())
    rewarder.reset_overwrite_previous()
    first, second = image(), image("b:1", owner="cam-b")
    assert rewarder.calculate_reward(
        {
            "cam-a": RSOImageData({first.record_id: first}),
            "cam-b": RSOImageData({second.record_id: second}),
        }
    ) == {"cam-a": 1.8, "cam-b": 1.8}
    assert rewarder.calculate_reward(
        {
            "cam-a": RSOImageData(
                deliveries={first.record_id: replace(first, delivery_time=30)}
            ),
            "cam-b": RSOImageData(
                deliveries={second.record_id: replace(second, delivery_time=30)}
            ),
        }
    ) == {"cam-a": 0.2, "cam-b": 0.2}
