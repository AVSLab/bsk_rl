"""Bounded streaming acceptance of a completed cooperative canary/tranche."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.update_evidence import UpdateRecords
from examples.multiagent_imaging.episode_accounting import verify_retained_learner_batch
from examples.multiagent_imaging.cooperative_learning_evidence import (
    sender_credit_evidence,
)
from examples.multiagent_imaging.readiness import write_json
from examples.multiagent_imaging.memory_profile import read_cgroup_memory


def initial_worker_readback(initial, expected_workers=29):
    """Accept both recorded layouts without weakening the full-topology gate."""
    if initial.get("version") == "fresh-policy-sync-v1":
        readback = initial.get("after", {})
        if not initial.get("synchronization", {}).get("passed"):
            raise ValueError("Fresh worker synchronization did not pass")
    elif initial.get("version") == "initial-worker-module-evidence-v2":
        readback = initial
    else:
        raise ValueError("Unknown initial worker evidence schema")
    workers = readback.get("workers", [])
    if (
        not initial.get("passed")
        or not readback.get("passed")
        or not readback.get("topology", {}).get("passed")
        or readback.get("driver_rng_unchanged") is not True
        or len(workers) != expected_workers
        or {w.get("worker_index") for w in workers} != set(range(expected_workers))
        or not all(w.get("passed") for w in workers)
    ):
        raise ValueError("Full worker initial state/readback gate required")
    return readback


def wandb_readback(root, last_update):
    """Bounded live readback, allowing a short summary propagation delay."""
    import wandb

    identity = json.loads((Path(root) / "training/wandb.json").read_text())
    required = [
        "losses/policy_loss",
        "losses/vf_loss",
        "training/gradient_l2",
        "mission/useful_delivered_visits",
        "mission/requests/unshaped_weighted_science",
        "memory/job_cgroup_peak_bytes",
        "status/elapsed_wall_h",
    ]
    required += [f"agent/sensor_{i}/reward" for i in range(4)]
    required += [f"agent/sensor_{i}/storage_fraction" for i in range(4)]
    error = None
    for attempt in range(3):
        try:
            live = wandb.Api(timeout=20).run(
                f"{identity['entity']}/{identity['project']}/{identity['id']}"
            )
            if (
                live.id != identity["id"]
                or live.summary.get("training/iteration") != last_update["iteration"]
                or live.summary.get("lifetime/true_decisions")
                != last_update["policy_decisions_lifetime"]
                or any(live.summary.get(key) is None for key in required)
            ):
                raise ValueError(
                    "Live W&B identity/update/heartbeat/per-agent metrics are incomplete or stale"
                )
            if not all(math.isfinite(float(live.summary[key])) for key in required):
                raise ValueError("Live W&B metrics must be finite")
            result = dict(
                passed=True,
                id=live.id,
                url=live.url,
                state=live.state,
                iteration=live.summary["training/iteration"],
                true_decisions=live.summary["lifetime/true_decisions"],
            )
            write_json(Path(root) / "wandb-readback.json", result)
            return result
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(5)
    raise RuntimeError("Required live W&B readback failed") from error


def accept(
    root,
    config,
    max_updates=None,
    max_episodes=None,
    max_decisions=None,
    post_restore_memory=None,
):
    root = Path(root)
    if (root / "acceptance.json").exists():
        raise FileExistsError("Preserve immutable acceptance; use a fresh job output")
    records = UpdateRecords(root / "training")
    if len(records) < 1 or (max_updates is not None and len(records) > max_updates):
        raise ValueError("Unexpected number of completed updates")
    episodes = decisions = 0
    first = None
    for row in records:
        if first is None:
            first = row
        if (
            not all(math.isfinite(v) for v in row["losses"].values())
            or row["finite_gradients"] != 1
        ):
            raise ValueError("Nonfinite learning")
        if (
            row["parameter_change"]["actor_l2"] <= 0
            or row["parameter_change"]["critic_l2"] <= 0
        ):
            raise ValueError("Actor and critic must both change")
        if row["condensed_learner_rows"] != row["actual_policy_decisions"]:
            raise ValueError("Learner rows differ from true decisions")
        for episode in row["episodes"]:
            if not math.isclose(
                episode["simulated_seconds"], config.episode_duration_s, abs_tol=1e-6
            ):
                raise ValueError("False mission horizon")
        episodes += len(row["episodes"])
        decisions += row["actual_policy_decisions"]
    if (max_episodes is not None and episodes > max_episodes) or (
        max_decisions is not None and decisions > max_decisions
    ):
        raise ValueError("Sampling cap exceeded")
    retained = verify_retained_learner_batch(
        root / "training/learner_batch",
        config,
        first["episodes"],
        first["condensed_learner_rows"],
    )
    credit = sender_credit_evidence(
        root / "training/learner_batch", config, first["episodes"]
    )
    peak = 0
    cgroup_samples = 0
    # The job cgroup peak is cumulative: read it after restore as well, so a
    # restore-only allocation spike cannot disappear from training telemetry.
    memory_source = dict(method="live_current_process_cgroup")
    if post_restore_memory is None:
        final_cgroups = read_cgroup_memory()
    else:
        path = Path(post_restore_memory).resolve()
        if path == (root / "post-restore-memory.json").resolve():
            raise ValueError("Independent audit must preserve its raw memory input")
        if path.stat().st_size > 4 * 1024**2:
            raise ValueError("Post-restore memory metadata exceeds 4 MiB")
        raw = path.read_bytes()
        final_cgroups = json.loads(raw)["cgroups"]
        memory_source = dict(
            method="saved_compute_post_restore_cgroup",
            path=str(path),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
    write_json(
        root / "post-restore-memory.json",
        dict(cgroups=final_cgroups, evidence_source=memory_source),
    )

    def memory_snapshots():
        # Full updates.json is deliberately never read here.
        with (root / "training/memory-profile.jsonl").open() as stream:
            for line in stream:
                yield json.loads(line)
        yield dict(cgroups=final_cgroups)

    final_allocation_observed = any(
        str(
            group["values"].get(
                "memory.max", group["values"].get("memory.limit_in_bytes")
            )
        )
        == "214748364800"
        for group in final_cgroups
    )
    if not final_allocation_observed:
        raise ValueError("Post-restore job cgroup evidence is missing")
    for snapshot in memory_snapshots():
        for group in snapshot.get("cgroups", []):
            values = group["values"]
            limit = values.get("memory.max", values.get("memory.limit_in_bytes", "max"))
            if str(limit) != "214748364800":
                continue
            cgroup_samples += 1
            peak = max(
                peak,
                int(
                    values.get(
                        "memory.peak", values.get("memory.max_usage_in_bytes", 0)
                    )
                ),
            )
            if int(values.get("memory.failcnt", 0)) or any(
                int(v) > 0
                for k, v in (
                    item.split()
                    for item in values.get("memory.events", "").splitlines()
                )
                if k in {"oom", "oom_kill", "max"}
            ):
                raise ValueError("Cgroup memory failure")
    if not cgroup_samples or peak > 160 * 1024**3:
        raise ValueError("Canary requires measured job cgroup peak <=160 GiB")
    initial = json.loads((root / "training/initial-worker-modules.json").read_text())
    initial_readback = initial_worker_readback(initial)
    driver = json.loads((root / "preflight/effective-driver.json").read_text())
    observed_indices = set()
    for path in (root / "training/effective-worker-runtime").glob("*.json"):
        worker = json.loads(path.read_text())
        observed_indices.add(int(path.name.split("-")[1]))
        if (
            not worker["passed"]
            or worker["basilisk"] != driver["basilisk"]
            or worker["bsk_rl"] != driver["bsk_rl"]
        ):
            raise ValueError("Actual worker runtime/native/source differs from driver")
        for name, module in driver["modules"].items():
            for field in ("version", "path", "file_sha256"):
                if worker["modules"][name][field] != module[field]:
                    raise ValueError("Actual worker effective import mismatch")
    if observed_indices != set(range(29)):
        raise ValueError("Missing effective runtime evidence for an actual worker")
    topologies = sorted((root / "training").glob("worker-topology-*.json"))
    if len(topologies) != len(records):
        raise ValueError("Missing update topology evidence")
    for path in topologies:
        topology = json.loads(path.read_text())
        if (
            not topology["passed"]
            or topology["actor_restart_count"]
            or topology["changed_process_indices"]
        ):
            raise ValueError("Worker restart/replacement or failed topology")
    restored = json.loads(
        (root / "zero-update-restore/zero-update-resume.json").read_text()
    )
    if (
        not restored.get("passed", False)
        or len(restored.get("workers", [])) != 29
        or restored.get("new_ppo_updates") != 0
        or restored.get("new_sampled_episodes") != 0
        or not restored.get("exact_logits")
        or restored.get("max_logit_error") != 0
        or any(
            not w.get("matches_checkpoint") or not w.get("no_environment_steps")
            for w in restored["workers"]
        )
    ):
        raise ValueError("Full-topology zero-update restoration failed")
    if not json.loads((root / "checkpoint-after.json").read_text())["passed"]:
        raise ValueError(
            "Checkpoint changed during exact full-topology zero-update restore"
        )
    wandb = wandb_readback(root, row)
    result = dict(
        passed=True,
        updates=len(records),
        episodes=episodes,
        true_decisions=decisions,
        cgroup_peak_bytes=peak,
        cgroup_samples=cgroup_samples,
        post_restore_cgroup_observed=final_allocation_observed,
        post_restore_memory_source=memory_source,
        initial_worker_evidence_version=initial["version"],
        initial_worker_count=len(initial_readback["workers"]),
        retained_batch=retained,
        sender_credit=credit,
        checkpoint_bytes_unchanged=True,
        full_worker_runtime_match=True,
        stable_worker_topology=True,
        wandb=wandb,
        policy_improvement_proven=False,
    )
    write_json(root / "acceptance.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--canary", action="store_true")
    p.add_argument(
        "--post-restore-memory",
        type=Path,
        help="Raw compute-node snapshot for a separate read-only post-run audit",
    )
    a = p.parse_args()
    try:
        r = accept(
            a.root,
            MultiAgentImagingConfig.from_json(a.config),
            post_restore_memory=a.post_restore_memory,
            **(
                dict(max_updates=2, max_episodes=56, max_decisions=120000)
                if a.canary
                else {}
            ),
        )
    except Exception as error:
        failed = a.root / "acceptance-failed.json"
        if not failed.exists():
            write_json(
                failed,
                dict(passed=False, error_type=type(error).__name__, error=str(error)),
            )
        raise
    print(
        json.dumps(
            {k: v for k, v in r.items() if k not in {"retained_batch", "sender_credit"}}
        )
    )


if __name__ == "__main__":
    main()
