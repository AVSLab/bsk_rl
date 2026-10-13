"""Fail-closed, bounded state readback for the pinned Ray 2.35 actor manager.

The public foreach_worker API discards error results and defaults to healthy
actors. This adapter deliberately retains every result and never marks actors
healthy, recreates them, samples, or infers state from episode evidence.
"""
import hashlib
import os
import time


class WorkerTopologyError(AssertionError):
    def __init__(self, evidence, *, readbacks=None):
        self.evidence = evidence
        # Preserve successful replies when another actor fails. Checkpoint
        # callers retain their existing fail-closed behavior and evidence shape.
        self.readbacks = [] if readbacks is None else readbacks
        super().__init__("Incomplete or changed worker state: " + str(evidence))


def expected_indices(algorithm):
    return list(range(int(algorithm.config.num_env_runners) + 1))


def validate_seed_states(states, expected, *, require_accounting=False):
    import torch
    rng_size = int(torch.get_rng_state().numel())  # Read only; pinned CPU Torch.
    indices = [s.get("worker_index") for s in states if isinstance(s, dict)]
    duplicates = sorted({i for i in indices if indices.count(i) > 1}, key=str)
    missing = sorted(set(expected) - set(indices))
    unexpected = sorted(set(indices) - set(expected), key=str)
    malformed = []
    for state in states:
        if not isinstance(state, dict):
            malformed.append("null/non-object state")
            continue
        i = state.get("worker_index")
        for field in ("worker_index", "base_seed", "next_episode_index"):
            if not isinstance(state.get(field), int) or state[field] < 0:
                malformed.append(f"worker {i}: invalid {field}")
        rng = state.get("torch_rng")
        if not isinstance(rng, list) or len(rng) != rng_size or any(
            not isinstance(v, int) or not 0 <= v < 256 for v in rng
        ):
            malformed.append(f"worker {i}: absent/invalid full Torch RNG")
        if require_accounting:
            audit = state.get("study_audit_state")
            if not isinstance(audit, dict) or not {"episode_hashes", "true_decisions", "reward"} <= set(audit):
                malformed.append(f"worker {i}: missing lifetime accounting")
            if not isinstance(state.get("pending_completed_episodes"), list):
                malformed.append(f"worker {i}: missing pending episode accounting")
    evidence = dict(expected_indices=expected, actual_indices=indices,
                    missing_indices=missing, unexpected_indices=unexpected,
                    duplicate_indices=duplicates, malformed=malformed, full_torch_rng_bytes=rng_size)
    evidence["passed"] = not (missing or unexpected or duplicates or malformed)
    if not evidence["passed"]:
        raise WorkerTopologyError(evidence)
    return evidence


def _envelope(runner, reader):
    import ray
    identity = dict(pid=os.getpid())
    if ray.is_initialized() and runner.worker_index:
        identity["actor_id"] = str(ray.get_runtime_context().get_actor_id())
    return dict(value=reader(runner), identity=identity)


def collect_all_workers(algorithm, reader, *, phase, timeout_s=45, require_accounting=False):
    """Read all configured actors, including unhealthy actors, retaining errors.

    Local state reads are memory-only. Remote reads share one bounded timeout.
    An unhealthy actor remains a failure even if it replies to this RPC.
    """
    expected = expected_indices(algorithm)
    group = algorithm.env_runner_group
    manager = group._worker_manager  # Explicit pinned-Ray adapter; tested offline.
    ids = list(manager.actor_ids())
    before = {i: bool(manager.is_actor_healthy(i)) for i in ids}
    evidence = dict(phase=phase, expected_indices=expected, configured_actor_ids=ids,
                    health_before=before, timeout_s=timeout_s, errors=[])
    start = time.monotonic()
    values, identities = [], {}
    try:
        local = group.local_env_runner
        if local is None:
            evidence["errors"].append(dict(worker_index=0, error="local worker absent"))
        else:
            reply = _envelope(local, reader)
            values.append(reply["value"])
            identities[0] = reply["identity"]
        if ids:
            results = manager.foreach_actor(
                lambda runner: _envelope(runner, reader), healthy_only=False,
                remote_actor_ids=ids, timeout_seconds=max(0, timeout_s - (time.monotonic() - start)),
                mark_healthy=False,
            )
            for result in results:
                if not result.ok:
                    error = result.get()
                    evidence["errors"].append(dict(worker_index=result.actor_id,
                                                  error_type=type(error).__name__, error=str(error)[:1500]))
                    continue
                reply = result.get()
                value = reply["value"]
                state = value.get("seed_state", value) if isinstance(value, dict) else None
                if not isinstance(state, dict) or state.get("worker_index") != result.actor_id:
                    evidence["errors"].append(dict(worker_index=result.actor_id, error="actor/state identity mismatch"))
                values.append(value)
                identities[result.actor_id] = reply["identity"]
    except Exception as error:
        evidence["errors"].append(dict(error_type=type(error).__name__, error=str(error)[:1500]))
    evidence["elapsed_s"] = time.monotonic() - start
    evidence["health_after"] = {i: bool(manager.is_actor_healthy(i)) for i in manager.actor_ids()}
    evidence["actor_restart_count"] = int(manager.total_num_restarts())
    states = [v.get("seed_state", v) if isinstance(v, dict) else None for v in values]
    try:
        evidence.update(validate_seed_states(states, expected, require_accounting=require_accounting))
    except WorkerTopologyError as error:
        evidence.update(error.evidence)
    evidence["process_identities"] = identities
    evidence["state_sha256"] = {
        s["worker_index"]: hashlib.sha256(__import__("json").dumps(s, sort_keys=True, allow_nan=False).encode()).hexdigest()
        for s in states if isinstance(s, dict) and "worker_index" in s
    }
    baseline = getattr(algorithm, "_checkpoint_worker_identities", identities)
    evidence["changed_process_indices"] = [i for i, identity in identities.items() if baseline.get(i) != identity]
    evidence["passed"] = bool(evidence.get("passed")) and not (
        evidence["errors"] or set(ids) != set(expected[1:])
        or not all(before.values()) or not all(evidence["health_after"].values())
        or evidence["actor_restart_count"] or evidence["changed_process_indices"]
        or evidence["elapsed_s"] > timeout_s
    )
    if not evidence["passed"]:
        raise WorkerTopologyError(evidence, readbacks=values)
    algorithm._checkpoint_worker_identities = identities
    return values, evidence
