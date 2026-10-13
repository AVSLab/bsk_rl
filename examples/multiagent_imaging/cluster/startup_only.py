"""Compute-only fresh-policy RPC gate: zero sampling, optimization or checkpoints."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path


def require_allocation():
    if os.environ.get("BSK_STARTUP_ONLY_APPROVED") != "1" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Separate startup-only approval and compute allocation required")
    root = Path(os.environ["BSK_OUTPUT"])
    record = json.loads((root / "allocation-gate.json").read_text())
    if record["allocated_cpus"] != 56 or record["time_limit_s"] != 900 or record["reserved_cpu_h"] != 14:
        raise RuntimeError("Startup-only allocation differs from reviewed 56 CPU / 15 minute / 14 CPU-hour cap")
    memory = json.loads((root / "memory-allocation-profile.json").read_text())
    if not memory["passed"] or memory["effective_limit_bytes"] != 214748364800:
        raise RuntimeError("Effective 200 GiB job cgroup required before Ray initialization")
    if os.environ.get("BSK_REQUIRE_EXTERNAL_PSUTIL") != "1":
        raise RuntimeError("Pinned effective runtime gate must be enabled")


def read_no_steps(runner):
    from examples.multiagent_imaging.worker_initial_evidence import read_runner_module
    row = read_runner_module(runner)
    env = runner.env.unwrapped
    simulator = getattr(env.par_env, "simulator", None)
    row["no_steps"] = dict(guard=bool(getattr(env, "_restore_only_guard", False)),
        attempted_steps=getattr(env, "_restore_step_calls", None),
        simulator_created=simulator is not None,
        simulated_seconds=simulator.sim_time if simulator is not None else None)
    return row


def run(config, output):
    require_allocation()  # Before importing/initializing Ray or building an environment.
    from examples.multiagent_imaging.cluster.effective_imports import require_effective_imports
    from examples.multiagent_imaging.config import MultiAgentImagingConfig
    from examples.multiagent_imaging.memory_profile import MemoryProfile, ray_memory_options
    from examples.multiagent_imaging.optimization import PPOSettings, actual_ppo_settings
    from examples.multiagent_imaging.train import LauncherResources, build_ppo_config
    from examples.multiagent_imaging.worker_initial_evidence import initialize_shared_policy, _rng_digest
    from examples.multiagent_imaging.checkpoint_topology import collect_all_workers, WorkerTopologyError
    import ray
    from ray.rllib.algorithms.ppo import PPO
    import torch

    experiment = MultiAgentImagingConfig.from_json(config)
    if experiment.information_case not in {"catalog_tx", "geometry_only"}:
        raise ValueError("Startup gate is only for the two declared study arms")
    # Existing shell preflight verifies source/runtime/native/support/schema/W&B.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ["BSK_STARTUP_ONLY_GATE"] = "1"
    os.environ["BSK_STUDY_RUNTIME_OUTPUT"] = str(output / "effective-worker-runtime")
    resources = LauncherResources(num_env_runners=28, num_learners=0, ray_cpus=56,
                                  torch_threads=1, sample_timeout_s=3600)
    torch.set_num_threads(1)
    profile = MemoryProfile(output)
    algorithm = None
    result = dict(passed=False, resources=asdict(resources), new_sampled_episodes=0,
                  ppo_updates=0, optimizer_steps=0, checkpoint_writes=0, errors=[])
    def save(name, value):
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    try:
        result["effective_driver"] = require_effective_imports()
        profile.start()
        ray.init(num_cpus=56, include_dashboard=False,
                 _temp_dir=os.environ["BSK_RL_RAY_TMPDIR"], **ray_memory_options())
        algorithm = PPO(build_ppo_config(experiment, train_batch_size=4200,
            resources=resources, ppo_settings=PPOSettings.from_profile("amos_best")))
        profile.mark("after-startup")
        result["initial_shared_policy"] = initialize_shared_policy(algorithm, output)
        result["effective_ppo"] = actual_ppo_settings(algorithm.config)
        before_rng = _rng_digest()
        try:
            rows, topology = collect_all_workers(algorithm, read_no_steps,
                phase="startup-no-steps", require_accounting=True)
        except WorkerTopologyError as error:
            rows, topology = error.readbacks, error.evidence
        no_steps = dict(topology=topology, workers=[dict(worker_index=r["seed_state"]["worker_index"],
                       **r["no_steps"]) for r in rows], driver_rng_unchanged=before_rng == _rng_digest())
        save("no-mission-steps.json", no_steps)
        result["no_mission_steps"] = no_steps
        result["passed"] = (bool(topology["passed"]) and no_steps["driver_rng_unchanged"] and
            all(w["guard"] and w["attempted_steps"] == 0 and
                w["simulated_seconds"] in (None, 0) for w in no_steps["workers"]) and
            algorithm.training_iteration == 0)
        if not result["passed"]:
            raise AssertionError("Startup-only gate attempted stepping or changed state")
    except Exception as error:
        result["errors"].append(dict(error_type=type(error).__name__, error=str(error)[:2000]))
    finally:
        cleanup = [("ray_shutdown", ray.shutdown), ("profile_close", profile.close)]
        if algorithm is not None:
            cleanup.insert(0, ("algorithm_stop", algorithm.stop))
        for phase, close in cleanup:
            try:
                close()
            except Exception as error:
                result["passed"] = False
                result["errors"].append(dict(phase=phase, error_type=type(error).__name__, error=str(error)[:2000]))
    save("startup-only-result.json", result)
    if not result["passed"]:
        raise RuntimeError("Startup-only acceptance failed; retain saved evidence")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
