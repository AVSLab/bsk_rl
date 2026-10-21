"""Offline fault injection for future compute-job acceptance; no cluster proof."""

import json
from types import SimpleNamespace

import pytest

from examples.multiagent_imaging.cluster import cooperative_accept as gate


def fixture(root, monkeypatch):
    def write(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    rows = [
        dict(
            losses=dict(total_loss=0.1),
            finite_gradients=1,
            parameter_change=dict(actor_l2=0.1, critic_l2=0.2),
            condensed_learner_rows=100,
            actual_policy_decisions=100,
            episodes=[dict(simulated_seconds=45000)],
        )
    ]
    monkeypatch.setattr(gate, "UpdateRecords", lambda p: rows)
    monkeypatch.setattr(
        gate, "verify_retained_learner_batch", lambda *a: dict(passed=True)
    )
    monkeypatch.setattr(gate, "sender_credit_evidence", lambda *a: dict(passed=True))
    monkeypatch.setattr(
        gate,
        "wandb_readback",
        lambda *a: dict(passed=True, id="offline-fault-injection"),
    )
    memory = dict(
        cgroups=[
            dict(
                values={
                    "memory.limit_in_bytes": str(200 * 1024**3),
                    "memory.max_usage_in_bytes": str(120 * 1024**3),
                    "memory.failcnt": "0",
                }
            )
        ]
    )
    write("training/memory-profile.jsonl", memory)
    monkeypatch.setattr(gate, "read_cgroup_memory", lambda: memory["cgroups"])
    write(
        "training/initial-worker-modules.json",
        dict(
            version="initial-worker-module-evidence-v2",
            passed=True,
            topology=dict(passed=True),
            driver_rng_unchanged=True,
            workers=[dict(worker_index=i, passed=True) for i in range(29)],
        ),
    )
    runtime = dict(
        passed=True,
        basilisk="pin",
        bsk_rl="source",
        modules=dict(psutil=dict(version="6.1.0", path="external", file_sha256="hash")),
    )
    write("preflight/effective-driver.json", runtime)
    for i in range(29):
        write(f"training/effective-worker-runtime/worker-{i}-100.json", runtime)
    write(
        "training/worker-topology-0001.json",
        dict(passed=True, actor_restart_count=0, changed_process_indices=[]),
    )
    write(
        "zero-update-restore/zero-update-resume.json",
        dict(
            passed=True,
            new_ppo_updates=0,
            new_sampled_episodes=0,
            exact_logits=True,
            max_logit_error=0,
            workers=[dict(matches_checkpoint=True, no_environment_steps=True)] * 29,
        ),
    )
    write("checkpoint-after.json", dict(passed=True))
    return write, rows, memory


def test_complete_offline_gate_and_immutable_acceptance(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    result = gate.accept(
        tmp_path,
        SimpleNamespace(episode_duration_s=45000),
        max_updates=2,
        max_episodes=56,
        max_decisions=120000,
    )
    assert result["passed"] and not result["policy_improvement_proven"]
    assert result["post_restore_cgroup_observed"]
    with pytest.raises(FileExistsError):
        gate.accept(tmp_path, SimpleNamespace(episode_duration_s=45000))


def test_real_fresh_sync_layout_keeps_all_after_workers(tmp_path, monkeypatch):
    write, _, _ = fixture(tmp_path, monkeypatch)
    path = tmp_path / "training/initial-worker-modules.json"
    after = json.loads(path.read_text())
    initial = dict(
        version="fresh-policy-sync-v1",
        passed=True,
        synchronization=dict(passed=True),
        after=after,
    )
    write("training/initial-worker-modules.json", initial)
    assert "workers" not in initial
    result = gate.accept(tmp_path, SimpleNamespace(episode_duration_s=45000))
    assert result["passed"] and result["initial_worker_count"] == 29
    assert result["initial_worker_evidence_version"] == "fresh-policy-sync-v1"


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "sync", "rng", "failed_worker"]
)
def test_fresh_schema_never_hides_failed_or_missing_worker(
    tmp_path, monkeypatch, fault
):
    fixture(tmp_path, monkeypatch)
    after = json.loads((tmp_path / "training/initial-worker-modules.json").read_text())
    initial = dict(
        version="fresh-policy-sync-v1",
        passed=True,
        synchronization=dict(passed=True),
        after=after,
    )
    if fault == "missing":
        after["workers"].pop()
    elif fault == "duplicate":
        after["workers"][-1]["worker_index"] = 0
    elif fault == "sync":
        initial["synchronization"]["passed"] = False
    elif fault == "rng":
        after["driver_rng_unchanged"] = False
    else:
        after["workers"][-1]["passed"] = False
    with pytest.raises(ValueError):
        gate.initial_worker_readback(initial)


def test_independent_memory_audit_uses_preserved_compute_snapshot(
    tmp_path, monkeypatch
):
    _, _, memory = fixture(tmp_path, monkeypatch)
    original = tmp_path / "original-compute-memory.json"
    original.write_text(json.dumps(memory))
    raw = original.read_bytes()
    monkeypatch.setattr(
        gate, "read_cgroup_memory", lambda: pytest.fail("Login cgroup must not be used")
    )
    result = gate.accept(
        tmp_path,
        SimpleNamespace(episode_duration_s=45000),
        post_restore_memory=original,
    )
    assert result["passed"] and original.read_bytes() == raw
    assert (
        result["post_restore_memory_source"]["method"]
        == "saved_compute_post_restore_cgroup"
    )


@pytest.mark.parametrize("fault", ["peak", "oom", "missing"])
def test_restore_only_memory_fault_cannot_pass_training_peak_gate(
    tmp_path, monkeypatch, fault
):
    fixture(tmp_path, monkeypatch)
    final = dict(
        values={
            "memory.max": str(200 * 1024**3),
            "memory.peak": str((161 if fault == "peak" else 120) * 1024**3),
            "memory.events": "oom 1\noom_kill 0"
            if fault == "oom"
            else "oom 0\noom_kill 0",
        }
    )
    monkeypatch.setattr(
        gate, "read_cgroup_memory", lambda: [] if fault == "missing" else [final]
    )
    with pytest.raises(ValueError, match="memory|cgroup"):
        gate.accept(tmp_path, SimpleNamespace(episode_duration_s=45000))
    assert not (tmp_path / "acceptance.json").exists()


def test_live_wandb_readback_allows_bounded_summary_lag(tmp_path, monkeypatch):
    import sys

    training = tmp_path / "training"
    training.mkdir()
    (training / "wandb.json").write_text(
        json.dumps(dict(entity="entity", project="project", id="run"))
    )
    summary = {
        "training/iteration": 2,
        "lifetime/true_decisions": 100,
        "losses/policy_loss": 0.1,
        "losses/vf_loss": 0.2,
        "training/gradient_l2": 0.3,
        "mission/useful_delivered_visits": 2,
        "mission/requests/unshaped_weighted_science": 5,
        "memory/job_cgroup_peak_bytes": 120 * 1024**3,
        "status/elapsed_wall_h": 0.5,
    }
    for i in range(4):
        summary[f"agent/sensor_{i}/reward"] = 1
        summary[f"agent/sensor_{i}/storage_fraction"] = 0.1
    calls, waits = [], []

    def run(path):
        calls.append(path)
        return SimpleNamespace(
            id="run",
            url="https://wandb.ai/entity/project/runs/run",
            state="finished",
            summary={} if len(calls) == 1 else summary,
        )

    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(Api=lambda **kw: SimpleNamespace(run=run))
    )
    monkeypatch.setattr(gate.time, "sleep", waits.append)
    result = gate.wandb_readback(
        tmp_path, dict(iteration=2, policy_decisions_lifetime=100)
    )
    assert result["passed"] and result["true_decisions"] == 100
    assert calls == ["entity/project/run"] * 2 and waits == [5]


@pytest.mark.parametrize(
    "fault",
    [
        "peak",
        "oom",
        "missing_worker",
        "wrong_import",
        "replacement",
        "restore_steps",
        "changed_checkpoint",
        "false_horizon",
        "sampling_cap",
    ],
)
def test_resource_worker_restore_and_data_faults_never_create_acceptance(
    tmp_path, monkeypatch, fault
):
    write, rows, memory = fixture(tmp_path, monkeypatch)
    if fault == "peak":
        memory["cgroups"][0]["values"]["memory.max_usage_in_bytes"] = str(161 * 1024**3)
        write("training/memory-profile.jsonl", memory)
    elif fault == "oom":
        memory["cgroups"][0]["values"]["memory.failcnt"] = "1"
        write("training/memory-profile.jsonl", memory)
    elif fault == "missing_worker":
        (tmp_path / "training/effective-worker-runtime/worker-28-100.json").unlink()
    elif fault == "wrong_import":
        write(
            "training/effective-worker-runtime/worker-28-100.json",
            dict(passed=True, basilisk="wrong", bsk_rl="source", modules={}),
        )
    elif fault == "replacement":
        write(
            "training/worker-topology-0001.json",
            dict(passed=True, actor_restart_count=1, changed_process_indices=[2]),
        )
    elif fault == "restore_steps":
        write(
            "zero-update-restore/zero-update-resume.json",
            dict(passed=True, workers=[], new_ppo_updates=0, new_sampled_episodes=0),
        )
    elif fault == "changed_checkpoint":
        write("checkpoint-after.json", dict(passed=False))
    elif fault == "false_horizon":
        rows[0]["episodes"][0]["simulated_seconds"] = 44000
    elif fault == "sampling_cap":
        rows[0]["actual_policy_decisions"] = rows[0]["condensed_learner_rows"] = 120001
    with pytest.raises(ValueError):
        gate.accept(
            tmp_path, SimpleNamespace(episode_duration_s=45000), max_decisions=120000
        )
    assert not (tmp_path / "acceptance.json").exists()
