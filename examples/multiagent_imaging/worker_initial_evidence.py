"""Fresh-policy synchronization and read-only evidence before any sampling.

The pinned Ray 2.35 setup does not send the learner's initial weights to remote
EnvRunners. Use its set_state API with bounded acknowledgements, retaining all
readbacks before assertions. No forward, seeding, mode change or environment
step belongs in this gate. Existing full-algorithm resume remains separate.
"""
import hashlib
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from examples.multiagent_imaging.checkpoints import _runner_seed_state, _state_digest
from examples.multiagent_imaging.checkpoint_topology import collect_all_workers, WorkerTopologyError


class InitialPolicyError(AssertionError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__("Initial shared-policy gate failed: " + str(evidence.get("errors", [])))


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _save(path, value):
    # Never replace evidence from an earlier invocation or a completed gate.
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _rng_digest():
    return hashlib.sha256(pickle.dumps(
        (random.getstate(), np.random.get_state(), torch.get_rng_state().numpy()),
        protocol=4)).hexdigest()


def actor_digest(module):
    return actor_state_digest(module.state_dict())


def actor_state_digest(state):
    weights = {k: torch.as_tensor(v) for k, v in state.items() if not k.startswith("vf.")}
    if not weights:
        raise AssertionError("Actor state is empty")
    return _state_digest(weights)


def _module_evidence(module):
    return dict(actor_sha256=actor_digest(module),
                parameters_finite=all(bool(torch.isfinite(v).all()) for v in module.state_dict().values()),
                module_modes={n: bool(m.training) for n, m in module.named_modules()},
                inference_only=module.config.inference_only,
                dropout={n: float(m.p) for n, m in module.named_modules()
                         if isinstance(m, torch.nn.Dropout)})


def read_runner_module(runner):
    before = _runner_seed_state(runner)
    rng_before = _rng_digest()
    module = runner.module["imager"]
    evidence = _module_evidence(module)
    after = _runner_seed_state(runner)
    evidence.update(rng_before_sha256=rng_before, rng_after_sha256=_rng_digest(),
                    episode_state_unchanged=before == after)
    evidence["passed"] = (evidence["episode_state_unchanged"] and
                          evidence["rng_before_sha256"] == evidence["rng_after_sha256"])
    # Return the failure facts too; acceptance happens after their persistence.
    return dict(seed_state=after, module_evidence=evidence)


def _summarize_row(row):
    if not isinstance(row, dict) or not isinstance(row.get("seed_state"), dict):
        return dict(passed=False, error="Malformed worker readback", value_type=type(row).__name__)
    state = row["seed_state"]
    rng = state.get("torch_rng")
    try:
        rng_hash = hashlib.sha256(bytes(rng)).hexdigest() if isinstance(rng, list) else None
    except (TypeError, ValueError):
        rng_hash = None
    try:
        state_hash = _json_digest(state)
    except (TypeError, ValueError):
        state_hash = None
    pending = state.get("pending_completed_episodes")
    return dict(worker_index=state.get("worker_index"), base_seed=state.get("base_seed"),
                next_episode_index=state.get("next_episode_index"),
                torch_rng_sha256=rng_hash, seed_accounting_state_sha256=state_hash,
                study_accounting=state.get("study_audit_state"),
                pending_completed_episodes_sha256=_json_digest(pending),
                pending_completed_episode_count=len(pending) if isinstance(pending, list) else None,
                **row.get("module_evidence", {}))


def _collect_report(algorithm, reader, *, phase, timeout_s):
    try:
        rows, topology = collect_all_workers(algorithm, reader, phase=phase,
                                             timeout_s=timeout_s, require_accounting=True)
    except WorkerTopologyError as error:
        rows, topology = error.readbacks, error.evidence
    return [_summarize_row(row) for row in rows], topology


def initial_worker_evidence(algorithm, *, authoritative_actor=None, require_match=True, timeout_s=45):
    driver = algorithm.get_module("imager")
    rng_before = _rng_digest()
    driver_evidence = _module_evidence(driver)
    reference = authoritative_actor or driver_evidence["actor_sha256"]
    workers, topology = _collect_report(algorithm, read_runner_module,
        phase="initial-module-readback", timeout_s=timeout_s)
    mismatched = [w.get("worker_index") for w in workers if w.get("actor_sha256") != reference]
    rng_after = _rng_digest()
    result = dict(version="initial-worker-module-evidence-v2",
        authoritative_actor_sha256=reference, driver=driver_evidence,
        driver_actor_sha256=driver_evidence["actor_sha256"],
        driver_module_modes=driver_evidence["module_modes"],
        driver_rng_before_sha256=rng_before, driver_rng_after_sha256=rng_after,
        driver_rng_unchanged=rng_before == rng_after, workers=workers, topology=topology,
        mismatched_worker_indices=mismatched, errors=[],
        query="state_dict/named_modules only; no forward, train/eval, seed, reset or sample")
    result["readback_passed"] = (bool(topology.get("passed")) and rng_before == rng_after
                                 and driver_evidence["parameters_finite"]
                                 and all(w.get("passed") and w.get("parameters_finite") for w in workers))
    result["actors_match"] = not mismatched and driver_evidence["actor_sha256"] == reference
    result["passed"] = result["readback_passed"] and result["actors_match"]
    if not result["readback_passed"]:
        result["errors"].append("Incomplete topology or query changed RNG/episode state")
    if not result["actors_match"]:
        result["errors"].append("Initial worker/driver actor weights differ from authoritative learner")
    if not result["readback_passed"] or (require_match and not result["actors_match"]):
        raise InitialPolicyError(result)
    return result


def _apply_initial_state(runner, state_ref):
    import ray
    from ray.rllib.core import COMPONENT_RL_MODULE

    before = read_runner_module(runner)
    errors = []
    try:
        # Only weights. No connector/counter update, reset, sample or mode change.
        runner.set_state({COMPONENT_RL_MODULE: ray.get(state_ref)})
    except Exception as error:
        errors.append(dict(error_type=type(error).__name__, error=str(error)[:1500]))
    after = read_runner_module(runner)
    a, b = before["module_evidence"], after["module_evidence"]
    unchanged = (before["seed_state"] == after["seed_state"] and
                 a["rng_before_sha256"] == b["rng_after_sha256"] and
                 a["module_modes"] == b["module_modes"] and a["dropout"] == b["dropout"])
    b.update(sync_state_unchanged=unchanged, sync_errors=errors,
             passed=bool(a["passed"] and b["passed"] and unchanged and not errors))
    return after


def initialize_shared_policy(algorithm, output, *, timeout_s=45):
    """Fresh-only weight transport with bounded all-actor acknowledgements.

    No Algorithm.train, episode sampling, forward pass or optimizer step.
    Failed reports are saved before raising. This is deliberately separate from
    checkpoint resume, whose saved RNG/episode/optimizer state is authoritative.
    """
    import ray
    from ray.rllib.core import COMPONENT_LEARNER, COMPONENT_RL_MODULE

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "initial-worker-modules.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    started = time.monotonic()
    rng_start = _rng_digest()
    result = dict(version="fresh-policy-sync-v1", passed=False, errors=[],
                  timeout_s=timeout_s, before=None, synchronization=None, after=None)

    def remaining():
        left = timeout_s - (time.monotonic() - started)
        if left <= 0:
            raise TimeoutError("Fresh-policy synchronization exceeded bounded time")
        return left

    try:
        if algorithm.training_iteration != 0:
            raise ValueError("Fresh-policy synchronization is forbidden after update zero")
        if algorithm.config.num_learners != 0 or algorithm.learner_group.is_remote:
            raise ValueError("Fresh-policy gate requires the declared local CPU learner")
        module = algorithm.learner_group._learner.module["imager"].unwrapped()
        learner_before = _module_evidence(module)
        learner_full_before = _state_digest(module.state_dict())
        result["learner_before"] = learner_before
        result["learner_full_state_before_sha256"] = learner_full_before
        if not learner_before["parameters_finite"]:
            raise ValueError("Initial learner actor/critic parameters are nonfinite")
        weights = algorithm.learner_group.get_state(
            components=COMPONENT_LEARNER + "/" + COMPONENT_RL_MODULE,
            inference_only=True)[COMPONENT_LEARNER][COMPONENT_RL_MODULE]
        if set(weights) != {"imager"}:
            raise ValueError("Fresh shared learner must contain only the imager policy")
        reference = actor_state_digest(weights["imager"])
        if reference != learner_before["actor_sha256"]:
            raise ValueError("Learner inference state differs from its actor parameters")
        result["authoritative_actor_sha256"] = reference
        try:
            before = initial_worker_evidence(algorithm, authoritative_actor=reference,
                                             require_match=False, timeout_s=remaining())
        except InitialPolicyError as error:
            before = error.evidence
        result["before"] = before
        _save(output / "initial-worker-before-sync.json", before)
        if not before["readback_passed"]:
            raise InitialPolicyError(before)

        state_ref = ray.put(weights)
        workers, topology = _collect_report(algorithm,
            lambda runner: _apply_initial_state(runner, state_ref),
            phase="fresh-policy-weight-transport", timeout_s=remaining())
        sync = dict(workers=workers, topology=topology,
                    passed=bool(topology.get("passed")) and all(w.get("passed") for w in workers))
        result["synchronization"] = sync
        _save(output / "initial-worker-sync-acknowledgements.json", sync)
        if not sync["passed"]:
            raise InitialPolicyError(dict(errors=["Initial state RPC/readback failed"], **sync))
        try:
            after = initial_worker_evidence(algorithm, authoritative_actor=reference,
                                            timeout_s=remaining())
        except InitialPolicyError as error:
            after = error.evidence
        result["after"] = after
        _save(output / "initial-worker-after-sync.json", after)
        prior = {w["worker_index"]: w for w in before["workers"]}
        checks = []
        for worker in after["workers"]:
            old = prior.get(worker["worker_index"], {})
            same = all(old.get(k) == worker.get(k) for k in (
                "seed_accounting_state_sha256", "rng_before_sha256", "rng_after_sha256",
                "module_modes", "dropout", "inference_only"))
            checks.append(dict(worker_index=worker["worker_index"], state_unchanged=same))
        result["worker_state_comparisons"] = checks
        result["learner_after"] = _module_evidence(module)
        result["learner_full_state_after_sha256"] = _state_digest(module.state_dict())
        result["learner_unchanged"] = (result["learner_after"] == learner_before and
            result["learner_full_state_after_sha256"] == learner_full_before)
        result["driver_modes_unchanged"] = before["driver"] == after["driver"] or (
            before["driver"]["module_modes"] == after["driver"]["module_modes"] and
            before["driver"]["dropout"] == after["driver"]["dropout"] and
            before["driver"]["inference_only"] == after["driver"]["inference_only"])
        result["driver_rng_unchanged"] = rng_start == _rng_digest()
        remaining()
        result["passed"] = (after["passed"] and result["learner_unchanged"] and
            result["driver_modes_unchanged"] and result["driver_rng_unchanged"] and
            all(c["state_unchanged"] for c in checks))
        if not result["passed"]:
            result["errors"].append("Post-sync actor/state/RNG/mode acceptance failed")
    except Exception as error:
        result["errors"].append(dict(error_type=type(error).__name__, error=str(error)[:1500]))
    result["elapsed_s"] = time.monotonic() - started
    # Persist every available success/failure fact, then enforce acceptance.
    _save(report_path, result)
    if not result["passed"]:
        raise InitialPolicyError(result)
    return result
