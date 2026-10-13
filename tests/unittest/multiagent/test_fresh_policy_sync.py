"""Offline pinned-Ray transport, no Ray init, native rollout or optimizer."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import ray
import torch
from ray.rllib.core import COMPONENT_RL_MODULE
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner

from examples.multiagent_imaging import worker_initial_evidence as gate


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pi_head = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout(.1))
        self.vf = torch.nn.Linear(2, 1)
        self.config = NS(inference_only=False)

    def unwrapped(self):
        return self

    def forward(self, *_):
        raise AssertionError("Startup gate must not forward")


class ModuleMap(dict):
    def set_state(self, state):
        for policy, weights in state.items():
            module = self[policy]
            if isinstance(module, Module):
                module.load_state_dict({k: torch.as_tensor(v) for k, v in weights.items()}, strict=False)
            else:
                # Real TorchRLModule.set_state for the actual attention module.
                module.set_state(weights)


class Runner:
    def __init__(self, index, module, fault=None):
        self.worker_index = index
        self.module = ModuleMap(imager=module)
        self._weights_seq_no = 0
        self.fault = fault
        self.state_calls = []
        self.env = NS(unwrapped=NS(par_env=NS(default_seed=100 + index,
            _episode_seed_index=1, communicator=NS(information_case="catalog_tx"))))

    def set_state(self, state):
        self.state_calls.append(list(state))
        assert set(state) == {COMPONENT_RL_MODULE}, "Only weights may be set"
        if self.fault == "reject":
            raise RuntimeError("Fixture set_state rejection")
        if self.fault != "ignore":
            # Execute the actual pinned Ray EnvRunner method, not a duplicate.
            MultiAgentEnvRunner.set_state(self, state)
        if self.fault == "episode":
            self.env.unwrapped.par_env._episode_seed_index += 1
        if self.fault == "rng":
            torch.rand(1)
        if self.fault == "mode":
            self.module["imager"].eval()


class Manager:
    def __init__(self, runners, fault=None):
        self.runners = runners
        self.fault = fault
        self.calls = []

    def actor_ids(self):
        return list(self.runners)

    def is_actor_healthy(self, index):
        return not (self.fault == "unhealthy" and index == 4)

    def total_num_restarts(self):
        return int(self.fault == "restart")

    def foreach_actor(self, reader, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["healthy_only"] is False and kwargs["mark_healthy"] is False
        assert 0 <= kwargs["timeout_seconds"] <= 45
        result = []
        for index, runner in self.runners.items():
            late = len(self.calls) >= 2
            if (self.fault == "omit" or (self.fault == "late_omit" and late)) and index == 4:
                continue
            bad = (self.fault == "rpc" or (self.fault == "late_rpc" and late)) and index == 4
            value = RuntimeError("Fixture failed actor RPC") if bad else reader(runner)
            result.append(NS(actor_id=index, ok=not bad, get=lambda value=value: value))
        return result


def algorithm(n=28, fault=None, modules=None):
    learner = Module() if modules is None else modules[0]
    local = Runner(0, Module() if modules is None else modules[1])
    runners = {i: Runner(i, Module() if modules is None else modules[i+1],
                        fault if i == 4 else None) for i in range(1, n+1)}
    manager = Manager(runners, fault)
    def state(**kwargs):
        assert kwargs == dict(components="learner/rl_module", inference_only=True)
        weights = {k: v.detach().cpu().numpy().copy() for k, v in learner.state_dict().items()
                   if not k.startswith("vf.")}
        return {"learner": {COMPONENT_RL_MODULE: {"imager": weights}}}
    return NS(training_iteration=0, config=NS(num_env_runners=n, num_learners=0),
        learner_group=NS(is_remote=False, _learner=NS(module={"imager": learner}), get_state=state),
        env_runner_group=NS(local_env_runner=local, _worker_manager=manager),
        get_module=lambda _: local.module["imager"])


@pytest.fixture(autouse=True)
def offline_transport(monkeypatch):
    assert not ray.is_initialized()
    monkeypatch.setattr(ray, "put", lambda value: value)
    monkeypatch.setattr(ray, "get", lambda value: value)
    monkeypatch.setattr(ray, "init", lambda *_a, **_k: pytest.fail("Ray init forbidden"))


def test_all_29_fresh_actors_match_learner_without_rng_mode_seed_or_critic_change(tmp_path):
    algo = algorithm()
    old_rng = gate._rng_digest()
    local = algo.get_module("imager")
    old_vf = copy.deepcopy(local.vf.state_dict())
    result = gate.initialize_shared_policy(algo, tmp_path)
    saved = json.loads((tmp_path / "initial-worker-modules.json").read_text())
    assert saved == json.loads(json.dumps(result)) and result["passed"]
    assert not result["before"]["actors_match"]
    assert len(result["before"]["workers"]) == len(result["after"]["workers"]) == 29
    assert result["after"]["actors_match"] and result["learner_unchanged"]
    assert result["driver_rng_unchanged"] and gate._rng_digest() == old_rng
    assert all(c["state_unchanged"] for c in result["worker_state_comparisons"])
    assert all(torch.equal(local.vf.state_dict()[k], v) for k, v in old_vf.items())
    for runner in [algo.env_runner_group.local_env_runner, *algo.env_runner_group._worker_manager.runners.values()]:
        assert runner.state_calls == [[COMPONENT_RL_MODULE]]
        assert runner.module["imager"].pi_head[1].p == .1
    assert '"torch_rng":' not in (tmp_path / "initial-worker-modules.json").read_text()
    with pytest.raises(FileExistsError):
        gate.initialize_shared_policy(algo, tmp_path)
    assert saved == json.loads((tmp_path / "initial-worker-modules.json").read_text())


@pytest.mark.parametrize("fault", ["rpc", "omit", "unhealthy", "restart", "reject", "ignore", "episode", "rng", "mode", "late_rpc", "late_omit"])
def test_failures_persist_full_available_worker_readbacks_before_raising(tmp_path, fault):
    algo = algorithm(fault=fault)
    with pytest.raises(gate.InitialPolicyError):
        gate.initialize_shared_policy(algo, tmp_path)
    saved = json.loads((tmp_path / "initial-worker-modules.json").read_text())
    assert not saved["passed"] and saved["errors"]
    assert saved["before"] and len(saved["before"]["workers"]) >= 28
    if fault in {"rpc", "omit", "unhealthy", "restart"}:
        assert not saved["before"]["topology"]["passed"]
        assert saved["synchronization"] is None
    elif fault in {"late_rpc", "late_omit"}:
        assert saved["before"]["readback_passed"]
        assert len(saved["before"]["workers"]) == 29
        assert len(saved["synchronization"]["workers"]) == 28
        assert saved["synchronization"]["topology"]["missing_indices"] == [4]
    else:
        assert len(saved["synchronization"]["workers"]) == 29
    if fault == "ignore":
        assert saved["after"]["mismatched_worker_indices"] == [4]
        assert len(saved["after"]["workers"]) == 29


def test_malformed_rng_reply_retained_with_remaining_28_workers(tmp_path, monkeypatch):
    original = gate.read_runner_module
    def malformed(runner):
        row = original(runner)
        if runner.worker_index == 4:
            row["seed_state"]["torch_rng"] = [999]
        return row
    monkeypatch.setattr(gate, "read_runner_module", malformed)
    with pytest.raises(gate.InitialPolicyError):
        gate.initialize_shared_policy(algorithm(), tmp_path)
    saved = json.loads((tmp_path / "initial-worker-modules.json").read_text())
    assert len(saved["before"]["workers"]) == 29
    assert saved["before"]["workers"][4]["torch_rng_sha256"] is None
    assert saved["before"]["topology"]["malformed"]


def test_expired_budget_retains_failure_and_does_not_send_weights(tmp_path):
    algo = algorithm(n=0)
    with pytest.raises(gate.InitialPolicyError):
        gate.initialize_shared_policy(algo, tmp_path, timeout_s=0)
    saved = json.loads((tmp_path / "initial-worker-modules.json").read_text())
    assert not saved["passed"] and saved["before"] is None
    assert not algo.env_runner_group.local_env_runner.state_calls


def test_nonfinite_initial_learner_is_retained_and_never_broadcast(tmp_path):
    algo = algorithm(n=0)
    with torch.no_grad():
        algo.learner_group._learner.module["imager"].vf.weight.fill_(float("nan"))
    with pytest.raises(gate.InitialPolicyError):
        gate.initialize_shared_policy(algo, tmp_path)
    saved = json.loads((tmp_path / "initial-worker-modules.json").read_text())
    assert saved["learner_before"]["parameters_finite"] is False
    assert not algo.env_runner_group.local_env_runner.state_calls


def test_completed_algorithm_or_remote_learner_rejected_without_weight_writes(tmp_path):
    for field in ("training_iteration", "remote"):
        algo = algorithm(n=0)
        if field == "remote":
            algo.learner_group.is_remote = True
        else:
            algo.training_iteration = 3
        folder = tmp_path / field
        with pytest.raises(gate.InitialPolicyError):
            gate.initialize_shared_policy(algo, folder)
        assert not algo.env_runner_group.local_env_runner.state_calls
        assert json.loads((folder / "initial-worker-modules.json").read_text())["errors"]


def test_gate_called_only_for_fresh_study_before_sampling():
    from examples.multiagent_imaging import train
    tree = ast.parse(Path(train.__file__).read_text())
    conditions = [n for n in ast.walk(tree) if isinstance(n, ast.If) and
                  ast.unparse(n.test) == "experiment.is_study and resume is None"]
    assert len(conditions) == 1
    calls = [n for n in ast.walk(conditions[0]) if isinstance(n, ast.Call) and
             isinstance(n.func, ast.Name) and n.func.id == "initialize_shared_policy"]
    assert len(calls) == 1
    sample_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                    isinstance(n.func, ast.Attribute) and n.func.attr == "train"]
    assert sample_calls and calls[0].lineno < min(n.lineno for n in sample_calls)


@pytest.mark.parametrize("information_case", ["catalog_tx", "geometry_only"])
def test_actual_attention_module_with_pinned_ray_set_state(tmp_path, information_case):
    from gymnasium.spaces import Box, Discrete
    from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from bsk_rl.utils.rllib.target_gnn_module import GNNModule
    from examples.multiagent_imaging.config import MultiAgentImagingConfig
    from examples.multiagent_imaging.train import target_attention_config

    config = MultiAgentImagingConfig.from_json(Path(__file__).resolve().parents[3] /
        f"examples/multiagent_imaging/configs/walker4_t200_cd1_{information_case}.json")
    modules = [RLModuleSpec(module_class=GNNModule, catalog_class=PPOCatalog,
        observation_space=Box(-np.inf, np.inf, (290,), dtype=np.float32),
        action_space=Discrete(18), model_config_dict=target_attention_config(config),
        inference_only=i > 0).build() for i in range(3)]
    result = gate.initialize_shared_policy(algorithm(n=1, modules=modules), tmp_path)
    assert result["passed"] and result["learner_unchanged"]
    assert result["authoritative_actor_sha256"] == gate.actor_digest(modules[0])
    assert all(v == .1 for worker in result["after"]["workers"] for v in worker["dropout"].values())
    assert not ray.is_initialized()
