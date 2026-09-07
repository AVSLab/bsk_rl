"""Strict completion-policy export, actual weight restore, and PPO resume."""

import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from bsk_rl.utils.rllib.target_gnn_module import GNNModule
from examples.multiagent_imaging.readiness import schema, validate_schema, write_json


def optimizer_digest(algorithm):
    """Digest all Adam moments, step counters and parameter-group settings."""
    optimizer = algorithm.learner_group.get_state(components="learner/optimizer")[
        "learner"
    ]["optimizer"]
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            array = (
                value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor)
                else value
            )
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode())

    visit(optimizer)
    return digest.hexdigest()


def _runner_seed_state(runner):
    env = getattr(runner, "env", None)
    if env is None:
        return None
    task = env.unwrapped.par_env
    return dict(
        worker_index=runner.worker_index,
        next_episode_index=task._episode_seed_index,
        base_seed=task.default_seed,
        torch_rng=torch.get_rng_state().tolist(),
    )


def _restore_runner_seed_state(runner, states):
    state = next(
        (s for s in states if s and s["worker_index"] == runner.worker_index), None
    )
    if state is None or getattr(runner, "env", None) is None:
        return
    task = runner.env.unwrapped.par_env
    if task.default_seed != state["base_seed"]:
        raise ValueError("Resume worker seed stream differs from checkpoint.")
    task._episode_seed_index = state["next_episode_index"]
    runner._needs_initial_reset = True
    torch.set_rng_state(torch.tensor(state["torch_rng"], dtype=torch.uint8))


def save_checkpoint(algorithm, config, output, probes):
    """Save PPO optimizer state plus an independently loadable policy and probes."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    algorithm.save_to_path(str(output / "algorithm"))
    module = algorithm.get_module("imager")
    observations = torch.as_tensor(probes, dtype=torch.float32)
    with torch.no_grad():
        expected = module.forward_inference({Columns.OBS: observations})[
            Columns.ACTION_DIST_INPUTS
        ].cpu()
    state = {
        key: value.detach().cpu().clone() for key, value in module.state_dict().items()
    }
    torch.save(
        dict(weights=state, observations=observations.cpu(), logits=expected),
        output / "policy.pt",
    )
    metadata = dict(
        schema=schema(config),
        config=config.to_dict(),
        model_config=module.config.model_config_dict,
        inference_only=module.config.inference_only,
        policy_sha256=hashlib.sha256((output / "policy.pt").read_bytes()).hexdigest(),
        optimizer_sha256=optimizer_digest(algorithm),
        training_iteration=algorithm.training_iteration,
        worker_seed_states=algorithm.env_runner_group.foreach_worker(
            _runner_seed_state
        ),
    )
    write_json(output / "manifest.json", metadata)
    restored, evidence = load_policy(output, config)
    # Matching stored probes includes logits, not just coincidentally equal actions.
    write_json(output / "restore_validation.json", evidence)
    return evidence


def load_policy(checkpoint, config):
    """Load only after semantic schema and file-integrity validation."""
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "manifest.json").read_text())
    validate_schema(metadata["schema"], config)
    actual_hash = hashlib.sha256((checkpoint / "policy.pt").read_bytes()).hexdigest()
    if actual_hash != metadata["policy_sha256"]:
        raise ValueError("Checkpoint policy digest mismatch.")
    payload = torch.load(
        checkpoint / "policy.pt", map_location="cpu", weights_only=True
    )
    contract = metadata["schema"]
    rng_state = torch.get_rng_state()
    module = RLModuleSpec(
        module_class=GNNModule,
        catalog_class=PPOCatalog,
        observation_space=gym.spaces.Box(
            -np.inf, np.inf, (contract["observation"]["size"],), np.float32
        ),
        action_space=gym.spaces.Discrete(contract["actions"]["size"]),
        model_config_dict=metadata["model_config"],
        inference_only=metadata["inference_only"],
    ).build()
    torch.set_rng_state(rng_state)
    module.load_state_dict(payload["weights"], strict=True)
    module.eval()
    with torch.no_grad():
        restored = module.forward_inference({Columns.OBS: payload["observations"]})[
            Columns.ACTION_DIST_INPUTS
        ]
    torch.testing.assert_close(restored, payload["logits"], rtol=1e-6, atol=1e-6)
    evidence = dict(
        max_logit_error=float((restored - payload["logits"]).abs().max()),
        matched_actions=bool(
            torch.equal(restored.argmax(-1), payload["logits"].argmax(-1))
        ),
        probe_count=int(restored.shape[0]),
        policy_sha256=actual_hash,
    )
    return module, evidence


def policy_callable(module):
    def policy(observation):
        with torch.no_grad():
            batch = {
                Columns.OBS: torch.as_tensor(
                    observation, dtype=torch.float32
                ).unsqueeze(0)
            }
            return int(module.forward_inference(batch)[Columns.ACTIONS][0])

    return policy


def resume_checkpoint(checkpoint, config, *, resources=None):
    """Resume at a complete-episode boundary, restoring optimizer and seed streams."""
    from ray.rllib.algorithms.ppo import PPO
    from ray.tune.registry import register_env
    from examples.multiagent_imaging.train import make_rllib_env

    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "manifest.json").read_text())
    validate_schema(metadata["schema"], config)
    # Validate the exported weights as well before opening the full Ray checkpoint.
    load_policy(checkpoint, config)
    register_env("MultiAgentImaging-RLlib", make_rllib_env)
    algorithm = PPO.from_checkpoint(str(checkpoint.resolve() / "algorithm"))
    # Ray restores the saved topology. Reject a launcher that would otherwise
    # misleadingly report four workers while actually resuming a one-worker run.
    if resources is not None:
        expected = {
            "num_env_runners": resources.num_env_runners,
            "num_cpus_per_env_runner": resources.cpus_per_env_runner,
            "num_learners": resources.num_learners,
            "num_gpus_per_learner": resources.gpus_per_learner,
            "sample_timeout_s": resources.sample_timeout_s,
        }
        different = {
            key: (getattr(algorithm.config, key), value)
            for key, value in expected.items()
            if getattr(algorithm.config, key) != value
        }
        if different:
            algorithm.stop()
            raise ValueError(
                f"Resume launcher differs from saved topology: {different}"
            )
    if (
        "optimizer_sha256" in metadata
        and optimizer_digest(algorithm) != metadata["optimizer_sha256"]
    ):
        algorithm.stop()
        raise AssertionError(
            "PPO optimizer moments or counters did not restore exactly."
        )
    if algorithm.training_iteration != metadata["training_iteration"]:
        algorithm.stop()
        raise AssertionError("PPO training iteration did not restore.")
    payload = torch.load(
        checkpoint / "policy.pt", map_location="cpu", weights_only=True
    )
    with torch.no_grad():
        logits = algorithm.get_module("imager").forward_inference(
            {Columns.OBS: payload["observations"]}
        )[Columns.ACTION_DIST_INPUTS]
    torch.testing.assert_close(logits.cpu(), payload["logits"], atol=1e-6, rtol=1e-6)
    algorithm.env_runner_group.foreach_worker(
        lambda runner: _restore_runner_seed_state(
            runner, metadata["worker_seed_states"]
        )
    )
    return algorithm
