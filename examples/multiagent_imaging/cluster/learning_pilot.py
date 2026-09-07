"""Two-stage, bounded finite-completion pilot. This module never submits jobs."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from examples.multiagent_imaging.cluster.audit_runtime import audit
from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.readiness import measure, write_json


MODES = ("conflict", "continuous")
HELD_OUT_SEEDS = tuple(range(10000, 10005))


def mission_config(mode):
    return MultiAgentImagingConfig.from_json(
        Path(__file__).resolve().parents[1]
        / "configs"
        / f"mission_preflight_{mode}.json"
    )


def validate_updates(records, config):
    """A returned train() call is insufficient: demand complete physical episodes."""
    if not records:
        raise AssertionError("No PPO update evidence.")
    for record in records:
        if not np.isfinite(list(record["losses"].values())).all():
            raise AssertionError("Nonfinite loss.")
        if record["finite_gradients"] != 1 or not np.isfinite(record["gradient_l2"]):
            raise AssertionError("Nonfinite gradients.")
        if record["parameter_change"]["l2"] <= 0:
            raise AssertionError("No weight update.")
        if not record["restore_validation"]["matched_actions"]:
            raise AssertionError("Restored checkpoint output mismatch.")
        for episode in record["episodes"]:
            if not np.isclose(episode["simulated_seconds"], config.episode_duration_s):
                raise AssertionError("Episode ended before the mission horizon.")
            durations = {}
            for task in episode["coordination"]["task_history"]:
                dt = task["end"] - task["start"]
                if dt < 0:
                    raise AssertionError("Negative physical action duration.")
                durations[task["sensor"]] = durations.get(task["sensor"], 0) + dt
            if len(durations) != config.n_sensors or not all(
                np.isclose(dt, config.episode_duration_s) for dt in durations.values()
            ):
                raise AssertionError(
                    "Physical task durations do not cover each sensor episode."
                )


def evaluate_checkpoint(config, checkpoint, output, seeds):
    """Use restored logits for actions, with matched deterministic heuristic episodes."""
    from examples.multiagent_imaging.checkpoints import load_policy, policy_callable
    from examples.multiagent_imaging.evaluate import run_rollout

    module, evidence = load_policy(checkpoint, config)
    policy = policy_callable(module)
    pairs = []
    for seed in seeds:
        eval_config = replace(config, seed=seed)
        results = {}
        for controller in ("restored", "closest_angle"):
            calls = []

            def counted(observation):
                action = policy(observation)
                calls.append(action)
                return action

            with measure() as measurement:
                result = run_rollout(
                    eval_config,
                    controller="closest_angle",
                    policy=counted if controller == "restored" else None,
                )
            if not np.isclose(result["sim_time_s"], config.episode_duration_s):
                raise AssertionError("Held-out episode ended early.")
            if controller == "restored" and (
                not calls
                or len(calls)
                != sum(result["coordination"]["policy_decisions"].values())
            ):
                raise AssertionError(
                    "Evaluation did not execute the restored policy at every decision."
                )
            result.update(
                controller=controller,
                checkpoint=str(checkpoint) if calls else None,
                restored_policy_calls=len(calls),
                restore_validation=evidence if calls else None,
            )
            measurement["sim_seconds_per_wall_second"] = (
                result["sim_time_s"] / measurement["wall_time_s"]
            )
            result["measurement"] = measurement
            write_json(Path(output) / f"seed_{seed}_{controller}.json", result)
            results[controller] = result
        if (
            results["restored"]["initial_conditions"]
            != results["closest_angle"]["initial_conditions"]
        ):
            raise AssertionError("Held-out policy/heuristic initial conditions differ.")
        pairs.append(
            dict(
                seed=seed,
                restored_reward=sum(results["restored"]["cumulative_reward"].values()),
                heuristic_reward=sum(
                    results["closest_angle"]["cumulative_reward"].values()
                ),
            )
        )
    write_json(Path(output) / "paired_rewards.json", pairs)
    return pairs


def check_gate(gate, runtime):
    if not gate.get("passed") or gate.get("validated_workers") != 1:
        raise ValueError("Four-worker pilot requires a passed one-worker validation.")
    old = gate["runtime"]
    for key in ("source_sha256", "packages"):
        if old[key] != runtime[key]:
            raise ValueError(f"Validation is stale: changed {key}.")
    if old["loaded"].get("basilisk_commit") != runtime["loaded"].get("basilisk_commit"):
        raise ValueError("Validation is stale: changed Basilisk source.")
    if old["loaded"].get("native_sha256") != runtime["loaded"].get("native_sha256"):
        raise ValueError("Validation is stale: changed Basilisk binaries.")


def run(stage, output, *, gate_path=None, updates=8):
    from examples.multiagent_imaging.train import LauncherResources, train_run

    if stage not in {"validate", "learn"} or not 1 <= updates <= 10:
        raise ValueError(
            "Only validation or 1–10 PPO updates per pilot mode are permitted."
        )
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh campaign directory to preserve evidence.")
    output.mkdir(parents=True, exist_ok=True)
    runtime = audit(allocation=True)
    write_json(output / "runtime.json", runtime)
    if not runtime["passed"]:
        raise RuntimeError(f"Runtime gate failed: {runtime['errors']}")
    if stage == "learn":
        if gate_path is None:
            raise ValueError("Supply the one-worker validation gate.")
        gate = json.loads(Path(gate_path).read_text())
        check_gate(gate, runtime)
        # Count the two discarded validation updates too: total authorized work
        # remains at most ten updates per mode across both stages.
        if any(updates + gate["modes"][mode]["updates"] > 10 for mode in MODES):
            raise ValueError(
                "Validation plus learning must stay within ten updates per mode."
            )
    workers = 1 if stage == "validate" else 4
    resources = LauncherResources(
        num_env_runners=workers, ray_cpus=8, torch_threads=1, sample_timeout_s=1800
    )
    record = dict(
        stage=stage,
        runtime=runtime,
        training_seed=0,
        workers=workers,
        max_updates_per_mode=2 if stage == "validate" else updates,
        modes={},
    )
    write_json(output / "plan.json", {**record, "held_out_seeds": HELD_OUT_SEEDS})
    for mode in MODES:
        config = mission_config(mode)
        write_json(output / mode / "config.json", config.to_dict())
        trained = train_run(
            config,
            iterations=1 if stage == "validate" else updates,
            train_batch_size=64,
            resources=resources,
            output=output / mode / "train",
        )
        validate_updates(trained, config)
        checkpoint = trained[-1]["checkpoint"]
        if stage == "validate":
            # A second process lifetime restores Adam, training count, policy logits
            # and worker seed streams, then performs one real resumed update.
            resumed = train_run(
                config,
                iterations=1,
                train_batch_size=64,
                resources=resources,
                resume=checkpoint,
                output=output / mode / "resume",
            )
            validate_updates(resumed, config)
            if resumed[0]["iteration"] != 2:
                raise AssertionError("Resume did not advance iteration 1 to 2.")
            trained += resumed
            checkpoint = resumed[-1]["checkpoint"]
        pairs = evaluate_checkpoint(
            config,
            checkpoint,
            output / mode / "evaluation",
            HELD_OUT_SEEDS[:1] if stage == "validate" else HELD_OUT_SEEDS,
        )
        record["modes"][mode] = dict(
            checkpoint=checkpoint, updates=len(trained), paired_evaluation=pairs
        )
        if stage == "learn":
            plot_learning(trained, output / mode / "learning.png")
    record.update(passed=True, validated_workers=workers)
    write_json(
        output
        / ("validation_gate.json" if stage == "validate" else "pilot_summary.json"),
        record,
    )
    return record


def plot_learning(records, destination):
    """Diagnostic trends from actual updates, not claims of statistical significance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    x = [r["iteration"] for r in records]
    values = [
        (
            "Mean sampled episode reward",
            [
                np.mean([e["constellation_reward"] for e in r["episodes"]])
                for r in records
            ],
        ),
        ("Total PPO loss", [r["losses"]["total_loss"] for r in records]),
        ("Parameter change L2", [r["parameter_change"]["l2"] for r in records]),
        ("Policy decisions sampled", [r["actual_policy_decisions"] for r in records]),
    ]
    for ax, (label, y) in zip(axes.flat, values):
        ax.plot(x, y, marker="o")
        ax.set(xlabel="PPO update", ylabel=label)
        ax.grid(alpha=0.2)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validate", "learn"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-gate", type=Path)
    parser.add_argument("--updates", type=int, choices=range(1, 11), default=8)
    args = parser.parse_args()
    run(args.stage, args.output, gate_path=args.validation_gate, updates=args.updates)
