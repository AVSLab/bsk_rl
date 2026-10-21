"""At most two local PPO updates, gated by completed native causal checks.

Full 45000-second missions, current Basilisk, one local runner, no W&B/network.
This establishes PPO wiring and exact local restore, not policy improvement or
Alpine's full 29-worker memory/runtime acceptance.
"""

import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads(args.causal_evidence.read_text())
    if not gate.get("passed") or not gate.get("ppo_gate_passed"):
        raise ValueError("Native causal qualification must pass before PPO")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["BSK_RL_RAY_OBJECT_STORE_BYTES"] = str(256 * 1024**2)
    os.environ["BSK_RL_MEMORY_PROFILE_INTERVAL_S"] = "10"
    os.environ["BSK_RL_TARGET_RECORDERS"] = "0"
    import Basilisk
    import numpy as np
    import ray
    import torch
    from examples.multiagent_imaging.config import MultiAgentImagingConfig
    from examples.multiagent_imaging.train import train_run, LauncherResources
    from examples.multiagent_imaging.optimization import PPOSettings
    from examples.multiagent_imaging.episode_accounting import (
        verify_retained_learner_batch,
    )
    from examples.multiagent_imaging.cooperative_learning_evidence import (
        sender_credit_evidence,
    )
    from examples.multiagent_imaging.checkpoints import (
        resume_checkpoint,
        optimizer_digest,
        _runner_seed_state,
        study_driver_restore_evidence,
    )
    from examples.multiagent_imaging.training_audit import frozen_policy_logits
    from examples.multiagent_imaging.memory_profile import ray_memory_options
    from examples.multiagent_imaging.update_evidence import file_sha256

    config_path = (
        Path(__file__).with_name("configs") / "cooperative_v1_shared_catalog_tx.json"
    )
    config = MultiAgentImagingConfig.from_json(config_path)
    native = Basilisk.getBuildInfo()
    started = time.monotonic()
    cpu_started = time.process_time()
    resources = LauncherResources(num_env_runners=0, ray_cpus=2)
    # A complete local mission exceeds this collection target; one mission per
    # update is expected, and the hard sampler cap forbids a third mission.
    rows = train_run(
        config,
        output=args.output / "training",
        iterations=2,
        train_batch_size=128,
        resources=resources,
        ppo_settings=PPOSettings.from_profile("amos_best"),
        wandb_mode="disabled",
        save_initial=True,
        capture_learner_batch=True,
        max_training_episodes=2,
        max_sampled_true_decisions=50000,
        max_wall_seconds=7200,
    )
    assert len(rows) == 2
    for row in rows:
        assert all(e["simulated_seconds"] == 45000 for e in row["episodes"])
        assert row["actual_policy_decisions"] == row["condensed_learner_rows"]
        assert row["finite_gradients"] == 1 and np.isfinite(row["gradient_l2"])
        assert (
            row["parameter_change"]["actor_l2"] > 0
            and row["parameter_change"]["critic_l2"] > 0
        )
    first = rows[0]
    retained = verify_retained_learner_batch(
        args.output / "training/learner_batch",
        config,
        first["episodes"],
        first["condensed_learner_rows"],
    )
    sender = sender_credit_evidence(
        args.output / "training/learner_batch", config, first["episodes"]
    )
    assert sender["sender_transmit_rows_with_later_shared_science"] > 0
    checkpoint = Path(rows[-1]["checkpoint"])
    before = {
        str(p.relative_to(checkpoint)): file_sha256(p)
        for p in checkpoint.rglob("*")
        if p.is_file()
    }
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    os.environ["BSK_ZERO_UPDATE_RESTORE"] = "1"
    algorithm = None
    ray.init(
        ignore_reinit_error=True,
        num_cpus=2,
        include_dashboard=False,
        **ray_memory_options(),
    )
    try:
        algorithm = resume_checkpoint(checkpoint, config, resources=resources)
        assert algorithm.training_iteration == 2
        assert optimizer_digest(algorithm) == manifest["optimizer_sha256"]
        assert (
            _runner_seed_state(algorithm.env_runner)
            == manifest["worker_seed_states"][0]
        )
        wrapped = algorithm.env_runner.env.unwrapped
        assert (
            wrapped._restore_step_calls == 0 and wrapped.par_env.simulator.sim_time == 0
        )
        driver = study_driver_restore_evidence(algorithm, manifest)
        saved = torch.load(
            checkpoint / "policy.pt", map_location="cpu", weights_only=True
        )
        module = algorithm.get_module("imager")
        assert all(
            torch.equal(value, saved["weights"][key])
            for key, value in module.state_dict().items()
        )
        assert torch.equal(
            frozen_policy_logits(module, saved["observations"]), saved["logits"]
        )
        dropout = {
            name: layer.p
            for name, layer in module.named_modules()
            if isinstance(layer, torch.nn.Dropout)
        }
        assert dropout and all(p == 0.1 for p in dropout.values())
        restore = dict(
            passed=True,
            optimizer_exact=True,
            weights_exact=True,
            logits_exact=True,
            rng_seed_index_accounting_exact=True,
            zero_environment_steps=True,
            dropout=dropout,
            **driver,
        )
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()
        os.environ.pop("BSK_ZERO_UPDATE_RESTORE", None)
    after = {
        str(p.relative_to(checkpoint)): file_sha256(p)
        for p in checkpoint.rglob("*")
        if p.is_file()
    }
    assert before == after and Basilisk.getBuildInfo() == native
    result = dict(
        passed=True,
        new_ppo_updates=2,
        completed_full_missions=sum(len(row["episodes"]) for row in rows),
        true_decisions=sum(row["actual_policy_decisions"] for row in rows),
        updates=[
            dict(
                iteration=row["iteration"],
                losses=row["losses"],
                gradient_l2=row["gradient_l2"],
                parameter_change=row["parameter_change"],
            )
            for row in rows
        ],
        retained_batch=retained,
        sender_credit=sender,
        restore=restore,
        unchanged_checkpoint_files=len(before),
        native_build=native,
        causal_evidence_sha256=file_sha256(args.causal_evidence),
        config_sha256=file_sha256(config_path),
        wall_s=time.monotonic() - started,
        driver_cpu_s=time.process_time() - cpu_started,
        cpu_scope="Driver process including local native simulation and Torch; Ray service CPU is separate in periodic process telemetry",
        full_alpine_topology_tested=False,
        wandb_online_tested=False,
        fixed_batch_fitting_tested=False,
        policy_improvement_proven=False,
        cluster_jobs=0,
    )
    (args.output / "qualification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in {"retained_batch", "sender_credit", "restore", "native_build"}
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
