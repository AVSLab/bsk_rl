"""Public tasking, private actor, symmetric costs and physical-time credit."""

from dataclasses import replace
from pathlib import Path
import json

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from bsk_rl.data.multiagent_rso_data import ImageProductRecord
from bsk_rl.data.multiagent_rso_reward import _TeamServiceAccounting
from bsk_rl.utils.rllib.target_gnn_module import GNNModule
from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.cooperative import RequestSchedule
from examples.multiagent_imaging.communication_qualification import credit_trace
from examples.multiagent_imaging.readiness import schema, validate_schema
from examples.multiagent_imaging.train import target_attention_config


def config():
    return MultiAgentImagingConfig.from_json(
        Path(__file__).parents[3]
        / "examples/multiagent_imaging/configs/cooperative_v1_shared_catalog_tx.json"
    )


def product(name="p", source="sensor_0", capture=10.0, epoch=0.0, quality=1.0):
    return ImageProductRecord(
        name, source, 3, capture, None, quality, source, epoch, capture + 1
    )


def test_baseline_semantics_remain_opt_in_and_checkpoints_reject_migration():
    c = config()
    baseline = replace(c, experiment_profile="baseline", redundant_capture_cost=0)
    assert "experiment_profile" not in baseline.to_dict()
    assert "redundant_capture_cost" not in baseline.to_dict()
    assert "outgoing_features" not in target_attention_config(baseline)
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_schema(schema(baseline), c)
    validate_schema(schema(c), replace(c, seed=999))
    validate_schema(json.loads(json.dumps(schema(c))), c)
    assert schema(c)["observation"]["size"] == 26 + 24 * 10 + 18 * 3 + 12 * 16 * 3


@pytest.mark.parametrize(
    "changes",
    [
        dict(active_request_count=201),
        dict(redundant_capture_cost=-1),
        dict(request_window_s=float("nan")),
        dict(gae_lambda_per_decision=0.9),
        dict(request_target_ids=(1, 1)),
        dict(request_priority_overrides=((3, -1),)),
    ],
)
def test_invalid_experiments_fail_closed(changes):
    with pytest.raises(ValueError):
        replace(config(), **changes)


def test_public_schedule_is_seeded_independent_and_deadlines_are_separate():
    c = config()
    schedule = RequestSchedule(c, 401)
    assert len(schedule.active_ids(0)) == 16
    assert schedule.active_ids(0) == RequestSchedule(c, 401).active_ids(0)
    assert schedule.active_ids(0) != schedule.active_ids(6000)
    assert schedule.active_ids(0) != RequestSchedule(c, 402).active_ids(0)
    target = next(iter(schedule.active_ids(0)))
    p = replace(product(capture=5998), target_id=target)
    assert schedule.qualifies(p, "acquisition")
    assert not schedule.qualifies(replace(p, completion_time=6001), "acquisition")
    assert not schedule.qualifies(p.delivered(6001), "delivery")
    assert schedule.qualifies(p.delivered(6000), "delivery")
    assert not schedule.qualifies(replace(p, quality=0.1), "acquisition")


def test_duplicate_capture_costs_are_symmetric_once_and_keep_team_credit():
    ledger = _TeamServiceAccounting(cooldown_s=100, quality_threshold=0.5)
    a, b = product(), product("q", "sensor_1")
    assert ledger.redundant_capture_costs([b, a], {3: 8}, 0.05) == {
        "sensor_0": 0.2,
        "sensor_1": 0.2,
    }
    assert ledger.register_acquisitions([a, b], {3: 8}) == {
        "sensor_0": 4,
        "sensor_1": 4,
    }
    assert ledger.redundant_capture_costs([a, b], {3: 8}, 0.05) == {}
    later = product("r", capture=30)
    assert ledger.redundant_capture_costs([later], {3: 8}, 0.05) == {"sensor_0": 0.4}
    assert ledger.register_acquisitions([later], {3: 8}) == {}
    revisit = product("s", capture=110)
    request = product("t", capture=40, epoch=40)
    for p in (revisit, request, product("bad", capture=20, quality=0.1)):
        assert sum(ledger.redundant_capture_costs([p], {3: 8}, 0.05).values()) == 0


def test_uncredited_physical_delivery_is_retained_without_reward_or_duplicate_charge():
    from types import SimpleNamespace as NS
    from bsk_rl.data.multiagent_rso_reward import MultiSensorRSOTargetImageReward
    from bsk_rl.sats.roles import SpacecraftRole

    reward = MultiSensorRSOTargetImageReward()
    sensor = NS(
        name="sensor_0", role=SpacecraftRole.SENSING_AGENT, dynamics=NS(penalties=0)
    )
    reward.scenario = NS(
        satellites=[sensor], target_spacecrafts=[NS(id=3, priority=8.0)]
    )
    reward.per_sensor_metrics = {
        sensor.name: dict(acquisition_credit=0.0, delivery_credit=0.0)
    }
    reward._team_accounting = _TeamServiceAccounting(
        cooldown_s=100, quality_threshold=0.5
    )
    receipt = product().delivered(300.0)
    reward._capture_products = lambda data: []
    reward._downlinked_products = lambda: [receipt]
    reward._science_products = lambda products, phase: []
    assert reward.calculate_reward({}) == {sensor.name: 0.0}
    assert len(reward.service_entries) == 1
    assert reward.service_entries[0].product == receipt
    assert reward.service_entries[0].credited_value == 0
    assert reward.team_summary["unique_service_count"] == 0
    assert reward.calculate_reward({}) == {sensor.name: 0.0}
    assert len(reward.service_entries) == 1


def model_and_obs():
    c = replace(config(), n_candidates=3, outgoing_record_slots=4)
    size = schema(c)["observation"]["size"]
    torch.manual_seed(78)
    model = RLModuleSpec(
        module_class=GNNModule,
        catalog_class=PPOCatalog,
        observation_space=gym.spaces.Box(-np.inf, np.inf, (size,), np.float32),
        action_space=gym.spaces.Discrete(11),
        model_config_dict=target_attention_config(c),
    ).build()
    model.eval()
    x = torch.randn(2, size)
    x[:, 1] = 0.9
    x[:, 25] = 1
    x[:, 26:98].reshape(2, 3, 24)[:, :, 16] = 1
    x[:, 98:152].reshape(2, 3, 18)[:, :, 16:] = 1
    x[:, 152:].reshape(2, 3, 4, 12)[:, :, :, -1] = 1
    return model, x


def test_transmit_scores_have_target_and_message_content_gradients():
    model, x = model_and_obs()
    x.requires_grad_()
    model.pi_head(x)[:, 8:].sum().backward()
    assert x.grad[:, 26:98].abs().max() > 1e-10
    assert x.grad[:, 152:].abs().max() > 1e-10
    assert torch.isfinite(x.grad).all()


def test_peer_permutation_includes_its_outgoing_inventory_and_preserves_value():
    model, x = model_and_obs()
    permutation = [2, 0, 1]
    y = x.clone()
    y[:, 98:152] = x[:, 98:152].reshape(2, 3, 18)[:, permutation].reshape(2, -1)
    y[:, 152:] = x[:, 152:].reshape(2, 3, 4, 12)[:, permutation].reshape(2, -1)
    torch.testing.assert_close(
        model.pi_head(y),
        model.pi_head(x)[:, [0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 9]],
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(model.vf(y), model.vf(x), atol=1e-6, rtol=1e-5)


def test_inventory_padding_and_empty_sets_are_finite_and_masked():
    model, x = model_and_obs()
    records = x[:, 152:].reshape(2, 3, 4, 12)
    records[:, :, 2:, -1] = 0
    y = x.clone()
    y[:, 152:].reshape(2, 3, 4, 12)[:, :, 2:, :-1] = float("nan")
    torch.testing.assert_close(model.pi_head(x), model.pi_head(y))
    torch.testing.assert_close(model.vf(x), model.vf(y))
    x[:] = 0
    x[:, 1] = 0.9
    out = model._forward_train({Columns.OBS: x})
    assert torch.isfinite(out[Columns.ACTION_DIST_INPUTS]).all()
    assert (out[Columns.ACTION_DIST_INPUTS][:, 3:] < -1e8).all()
    out[Columns.VF_PREDS].sum().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_physical_trace_restores_sender_credit_and_is_boundary_invariant():
    c = config()
    raw = dict(
        actions=[16, -1, 0], durations=[120.0, 80.0, 89.0], rewards=[0.0, 0.0, 0.25]
    )
    # NO_ACTION is imported rather than assuming its numeric encoding.
    from bsk_rl import NO_ACTION

    raw["actions"][1] = NO_ACTION
    trace = credit_trace(
        raw, c.discount_per_s, c.effective_gae_lambda, lambda_time="second"
    )
    expected = 0.25 * c.discount_per_s**289 * c.effective_gae_lambda**200
    assert trace["rows"][0]["diagnostic_value_target"] == pytest.approx(
        expected, rel=2e-6
    )
    split = dict(
        actions=[16, NO_ACTION, NO_ACTION, 0],
        durations=[50.0, 70.0, 80.0, 89.0],
        rewards=[0.0, 0.0, 0.0, 0.25],
    )
    other = credit_trace(
        split, c.discount_per_s, c.effective_gae_lambda, lambda_time="second"
    )
    np.testing.assert_allclose(
        [r["diagnostic_value_target"] for r in trace["rows"]],
        [r["diagnostic_value_target"] for r in other["rows"]],
    )
