"""Heterogeneous reward routing, knowledge sharing, and scenario bindings."""

from copy import copy, deepcopy
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from bsk_rl.comm import FreeCommunication
from bsk_rl.data import (
    MixedData,
    MixedReward,
    RSOImageData,
    RSOImageRecord,
    RSOImageReward,
)
from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.data.unique_image_data import UniqueImageData, UniqueImageReward
from bsk_rl.scene import MixedScenario, Scenario
from bsk_rl.scene.targets import Target


class CountData(Data):
    def __init__(self, count=0):
        self.count = count

    def __add__(self, other):
        return type(self)(self.count + other.count)


class CountStore(DataStore):
    data_type = CountData

    def __init__(self, satellite, initial_data=None, increment=1):
        super().__init__(satellite, initial_data)
        self.increment = increment

    def get_log_state(self):
        return self.satellite.value

    def compare_log_states(self, old_state, new_state):
        return CountData((new_state - old_state) * self.increment)


class CountReward(GlobalReward):
    data_store_type = CountStore

    def __init__(self, increment=1):
        super().__init__()
        self.data_store_kwargs = {"increment": increment}
        self.calls = []

    def create_data_store(self, satellite):
        super().create_data_store(satellite)
        satellite.hooks.append(self)

    def calculate_reward(self, new_data_dict):
        self.calls.append(new_data_dict)
        return {name: value.count for name, value in new_data_dict.items()}

    def is_terminated(self, satellite):
        return True


def make_mixed():
    satellites = [NS(name=name, value=0, hooks=[]) for name in ("eo", "sda", "target")]
    scenario = MixedScenario(
        {"ground": Scenario(), "rso": Scenario()},
        {"ground": ["eo"], "rso": ["sda", "target"]},
    )
    scenario.link_satellites(satellites)
    rewarder = MixedReward(
        {"ground": CountReward(2), "rso": CountReward(3)},
        {"ground": ["eo"], "rso": ["sda"]},
        {"ground": "ground", "rso": "rso"},
    )
    rewarder.link_scenario(scenario)
    rewarder.reset_overwrite_previous()
    for sat in satellites:
        rewarder.create_data_store(sat)
        rewarder.data += sat.data_store.data
        sat.data_store.update_from_logs()
    return satellites, scenario, rewarder


def test_mixed_data_union_identity_and_copy():
    first = MixedData({"ground": UniqueImageData(duplicates=2)})
    second = MixedData({"rso": RSOImageData()})
    merged = first + second
    assert set(merged.data) == {"ground", "rso"}
    assert merged.data["ground"].duplicates == 2
    assert merged.data["ground"] is not first.data["ground"]
    assert (merged + MixedData()).data.keys() == merged.data.keys()
    assert (MixedData() + merged).data.keys() == merged.data.keys()
    assert copy(merged).data["ground"] is not merged.data["ground"]
    assert deepcopy(merged).data.keys() == merged.data.keys()
    assert MixedData().__add__(CountData()) is NotImplemented
    with pytest.raises(TypeError, match="ground"):
        first + MixedData({"ground": RSOImageData()})


def test_same_type_channels_stay_separate_and_shared_channels_add():
    first = MixedData({"a": CountData(2), "b": CountData(10)})
    result = first + MixedData({"a": CountData(3)})
    assert result.data["a"].count == 5
    assert result.data["b"].count == 10
    with pytest.raises(AttributeError, match="Ambiguous"):
        _ = result.count


def test_reward_routing_hooks_kwargs_passive_satellites_and_reset():
    satellites, scenario, rewarder = make_mixed()
    for sat in satellites:
        sat.value = 1
    deltas = {sat.name: sat.data_store.update_from_logs() for sat in satellites}
    assert rewarder.reward(deltas) == {"eo": 2, "sda": 3, "target": 0}
    assert len(satellites[0].hooks) == len(satellites[1].hooks) == 1
    assert satellites[2].hooks == []
    assert satellites[2].data_store.data_stores == {}
    assert rewarder.rewarders["ground"].scenario is scenario.scenarios["ground"]
    assert set(rewarder.rewarders["rso"].calls[-1]) == {"sda"}
    assert rewarder.rewarders["ground"].data.count == 2
    assert rewarder.rewarders["rso"].data.count == 3
    assert rewarder.is_terminated(satellites[0])
    assert not rewarder.is_terminated(satellites[2])
    assert rewarder.reward({}) == {}
    rewarder.reset_overwrite_previous()
    assert rewarder.data.data["ground"].count == 0
    assert rewarder.rewarders["rso"].cum_reward == {}


@pytest.mark.parametrize("all_to_all", [False, True])
def test_communicated_knowledge_does_not_become_local_delta(all_to_all):
    satellites, _, rewarder = make_mixed()
    eo, sda, _ = satellites
    eo.value = 1
    sda.value = 1
    for sat in satellites:
        sat.data_store.update_from_logs()
    if all_to_all:
        communicator = FreeCommunication()
        communicator.link_satellites(satellites)
        communicator._communicate_all()
    else:
        eo.data_store.stage_communicated_data(sda.data_store.data)
        eo.data_store.update_with_communicated_data()
    assert set(eo.data_store.data.data) == {"ground", "rso"}
    assert eo.data_store.data_stores["ground"].data is eo.data_store.data.data["ground"]
    eo.value = 2
    delta = eo.data_store.update_from_logs()
    assert set(delta.data) == {"ground"}
    assert rewarder.reward({"eo": delta}) == {"eo": 2}
    assert rewarder.rewarders["rso"].calls == []


def test_overlapping_assignments_sum_rewards():
    sat = NS(name="both", value=0, hooks=[])
    scenario = MixedScenario(
        {"a": Scenario(), "b": Scenario()},
        {"a": ["both"], "b": ["both"]},
    )
    scenario.link_satellites([sat])
    rewarder = MixedReward(
        {"a": CountReward(2), "b": CountReward(3)},
        {"a": ["both"], "b": ["both"]},
        {"a": "a", "b": "b"},
    )
    rewarder.link_scenario(scenario)
    rewarder.reset_overwrite_previous()
    rewarder.create_data_store(sat)
    sat.data_store.update_from_logs()
    sat.value = 1
    assert rewarder.reward({"both": sat.data_store.update_from_logs()}) == {"both": 5}
    assert len(sat.hooks) == 2
    assert rewarder.rewarders["a"].scenario is scenario.scenarios["a"]
    assert rewarder.rewarders["b"].scenario is scenario.scenarios["b"]
    assert rewarder.rewarders["a"].scenario is not rewarder.rewarders["b"].scenario


def test_unique_image_hook_remains_bound_to_its_channel_after_communication():
    target = Target("city", [1, 0, 0], priority=1)
    sat = NS(name="eo", add_access_filter=MagicMock())
    scenario = Scenario()
    scenario.targets = [target]
    scenario.link_satellites([sat])
    rewarder = MixedReward({"images": UniqueImageReward()}, {"images": ["eo"]})
    rewarder.link_scenario(scenario)
    rewarder.reset_overwrite_previous()
    rewarder.create_data_store(sat)
    rewarder.data += sat.data_store.data
    rewarder.reset_post_sim_init()
    assert target in rewarder.rewarders["images"].data.known
    access_filter = sat.add_access_filter.call_args.args[0]
    assert access_filter({"type": "target", "object": target})
    sat.data_store.stage_communicated_data(
        MixedData(
            {
                "images": UniqueImageData(imaged={target}),
                "other_images": UniqueImageData(),
            }
        )
    )
    sat.data_store.update_with_communicated_data()
    assert not access_filter({"type": "target", "object": target})


def test_mixed_scenario_same_class_and_shared_instance_lifecycle():
    first, second = Scenario(), Scenario()
    first.after_step = MagicMock()
    second.after_step = MagicMock()
    mixed = MixedScenario(
        {"a": first, "alias": first, "b": second},
        {"a": ["a"], "alias": ["alias"], "b": ["b"]},
    )
    satellites = [NS(name=name) for name in ("a", "alias", "b")]
    mixed.link_satellites(satellites)
    assert {sat.name for sat in first.satellites} == {"a", "alias"}
    assert second.satellites == [satellites[2]]
    mixed.after_step(10)
    first.after_step.assert_called_once_with(10)
    second.after_step.assert_called_once_with(10)
    with pytest.raises(ValueError, match="unique"):
        mixed.validate_satellite_names([NS(name="a"), NS(name="a")])
    with pytest.raises(ValueError, match="Unknown"):
        mixed.validate_satellite_names(satellites[:1])


def test_reward_mapping_rejects_wrong_scene_membership():
    _, scenario, _ = make_mixed()
    rewarder = MixedReward(
        {"wrong": CountReward()}, {"wrong": ["target"]}, {"wrong": "ground"}
    )
    with pytest.raises(ValueError, match="outside"):
        rewarder.link_scenario(scenario)
    with pytest.raises(ValueError, match="explicit"):
        MixedReward({"a": CountReward()}, {"a": ["eo"]}).link_scenario(scenario)
    repeated = CountReward()
    with pytest.raises(ValueError, match="independent"):
        MixedReward({"a": repeated, "b": repeated}, {"a": [], "b": []})


def test_actual_ground_and_rso_rewards_keep_independent_global_history():
    target = Target("city", [1, 0, 0], priority=3)
    satellites = [
        NS(name=name, add_access_filter=MagicMock()) for name in ("eo", "sda", "target")
    ]
    ground, rso = Scenario(), Scenario()
    ground.targets = [target]
    rso.imager_names = ("sda",)
    rso.targets_by_id = {"object": NS(priority=2)}
    rso.pending = {}
    rso.cooldown_until = {}
    rso.revision = 0
    scenario = MixedScenario(
        {"ground": ground, "rso": rso},
        {"ground": ["eo"], "rso": ["sda", "target"]},
    )
    scenario.link_satellites(satellites)
    rewarder = MixedReward(
        {
            "ground": UniqueImageReward(),
            "rso": RSOImageReward(
                acquisition_reward_fn=lambda record, target: 0.9 * target.priority,
                delivery_reward_fn=lambda record, target: 0.1 * target.priority,
            ),
        },
        {"ground": ["eo"], "rso": ["sda"]},
        {"ground": "ground", "rso": "rso"},
    )
    rewarder.link_scenario(scenario)
    rewarder.reset_overwrite_previous()
    for sat in satellites:
        rewarder.create_data_store(sat)
        rewarder.data += sat.data_store.data
    record = RSOImageRecord("sda:1", "object", "sda", 10, 100, 0.8, 10)
    deltas = {
        "eo": MixedData({"ground": UniqueImageData(imaged={target})}),
        "sda": MixedData({"rso": RSOImageData(captures={record.record_id: record})}),
        "target": MixedData(),
    }
    assert rewarder.reward(deltas) == {"eo": 3, "sda": 1.8, "target": 0}
    assert target in rewarder.rewarders["ground"].data.known
    assert target in rewarder.rewarders["ground"].data.imaged
    assert record.record_id in rewarder.rewarders["rso"].data.captures
    assert rewarder.reward(deltas) == {"eo": 0, "sda": 0, "target": 0}
