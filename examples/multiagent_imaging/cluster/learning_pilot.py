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
        / f"walker4_mixed_preflight_{mode}.json"
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


def validate_mission_paths(records, config):
    """Require the random mission gate to exercise the paths it claims to validate."""
    episodes = [episode for record in records for episode in record["episodes"]]
    if not episodes:
        raise AssertionError("No mission episodes were available for path validation.")
    for episode in episodes:
        if set(episode["resources"]) != {
            f"sensor_{index}" for index in range(config.n_sensors)
        }:
            raise AssertionError("Mission episode did not contain all sensing agents.")
        if episode["target_regime_counts"] != {"LEO": 50, "MEO": 30, "GEO": 20}:
            raise AssertionError("Mission episode did not contain the exact mixed population.")
        if not np.isclose(episode["reimage_cooldown_s"], 11834.835756586714):
            raise AssertionError("Walker cooldown differs from the capture-anchored contract.")
    if not any(episode["coverage"]["qualified_exposure_count"] for episode in episodes):
        raise AssertionError("No qualified acquisition occurred in the mission gate.")
    if not any(
        episode["coverage"]["qualified_ground_delivery_count"] for episode in episodes
    ):
        raise AssertionError("No full ground delivery occurred in the mission gate.")
    if not any(
        packet.get("outcome") == "accepted"
        for episode in episodes
        for packet in episode["packets"]
    ):
        raise AssertionError("No directed completion packet was accepted.")
    if not any(
        episode["useful_post_cooldown_revisits"]["count"] for episode in episodes
    ):
        raise AssertionError("No useful post-cooldown revisit occurred.")


def evaluate_checkpoint(config, checkpoint, output, seeds):
    """Pair restored policy, independent, and full-state greedy references by seed."""
    from examples.multiagent_imaging.checkpoints import load_policy, policy_callable
    from examples.multiagent_imaging.evaluate import run_rollout

    module, evidence = load_policy(checkpoint, config)
    policy = policy_callable(module)
    output = Path(output)
    pairs = []
    for seed in seeds:
        policy_config = replace(config, seed=seed)
        independent_config = replace(config, seed=seed, information_case="independent")
        centralized_config = replace(
            config, seed=seed, information_case="ideal_completion"
        )
        results = {}
        runs = (
            ("policy", policy_config, "closest_angle", True),
            (
                "independent",
                independent_config,
                "independent_reference",
                False,
            ),
            (
                "centralized_full_state",
                centralized_config,
                "centralized_full_state_reference",
                False,
            ),
        )
        for label, eval_config, controller, uses_policy in runs:
            calls = []

            def counted(observation):
                action = policy(observation)
                calls.append(action)
                return action

            with measure() as measurement:
                result = run_rollout(
                    eval_config,
                    controller=controller,
                    policy=counted if uses_policy else None,
                )
            if not np.isclose(result["sim_time_s"], config.episode_duration_s):
                raise AssertionError("Held-out episode ended early.")
            if uses_policy and (
                not calls
                or len(calls)
                != sum(result["coordination"]["policy_decisions"].values())
            ):
                raise AssertionError(
                    "Evaluation did not execute the restored policy at every decision."
                )
            if not uses_policy and (
                result["message_diagnostics"]["transmission_history"]
                or any(result["broadcast_time_s"].values())
            ):
                raise AssertionError("Heuristic references must have zero radio activity.")
            result.update(
                controller=(
                    "restored-target-set-attention"
                    if uses_policy
                    else controller
                ),
                checkpoint=str(checkpoint) if calls else None,
                restored_policy_calls=len(calls),
                restore_validation=evidence if calls else None,
            )
            measurement["sim_seconds_per_wall_second"] = (
                result["sim_time_s"] / measurement["wall_time_s"]
            )
            result["measurement"] = measurement
            write_json(output / f"seed_{seed}_{label}.json", result)
            results[label] = result
        hashes = {result["initial_conditions_sha256"] for result in results.values()}
        if len(hashes) != 1:
            raise AssertionError("Held-out policy/reference initial conditions differ.")

        def metrics(result):
            coordination = result["coordination"]
            products = result["product_duplicates"]
            rewards = result["reward_decomposition"]
            return dict(
                acquisition_reward_component=rewards[
                    "acquisition_90_percent_component"
                ],
                ground_delivery_reward_component=rewards[
                    "ground_delivery_10_percent_component"
                ],
                communication_time_adjustment=rewards[
                    "communication_time_adjustment"
                ],
                other_operational_penalties_or_adjustments=rewards[
                    "other_operational_penalties_or_adjustments"
                ],
                total_reward=rewards["total_constellation_reward"],
                first_capture_coverage=result["coverage"][
                    "capture_coverage_fraction"
                ],
                ground_delivery_coverage=result["coverage"][
                    "ground_delivery_coverage_fraction"
                ],
                cooldown_acquisitions=result["team_summary"][
                    "unique_acquisition_count"
                ],
                cooldown_ground_services=result["team_summary"][
                    "unique_service_count"
                ],
                duplicate_attempts=result["team_summary"]["duplicate_attempt_count"],
                successful_duplicates=result["team_summary"][
                    "successful_duplicate_count"
                ],
                duplicate_sensor_seconds=coordination["duplicate_sensor_time_s"],
                interrupted_nonduplicate_sensor_seconds=coordination[
                    "interrupted_nonduplicate_sensor_time_s"
                ],
                wasted_sensor_seconds=coordination["duplicate_sensor_time_s"]
                + coordination["interrupted_nonduplicate_sensor_time_s"],
                wasted_time_fraction=coordination["wasted_time_fraction"],
                stale_cross_sensor_ground_deliveries=products[
                    "stale_cross_sensor_ground_delivery_count"
                ],
                causally_avoidable_stale_ground_deliveries=products[
                    "causally_avoidable_stale_ground_delivery_count"
                ],
                cross_sensor_onboard_overlap_targets=products[
                    "cross_sensor_onboard_overlap_target_count"
                ],
                cross_sensor_onboard_overlap_products=products[
                    "cross_sensor_onboard_overlap_product_count"
                ],
                redundant_acquisitions=products[
                    "cross_sensor_onboard_redundant_acquisition_count"
                ],
                excess_holder_sensor_seconds=products[
                    "cross_sensor_onboard_redundant_sensor_time_s"
                ],
                overlap_sensor_seconds=products[
                    "cross_sensor_onboard_overlap_sensor_time_s"
                ],
                radio_occupancy_s=sum(
                    coordination["communication_time_s"].values()
                ),
                useful_revisits=result["useful_post_cooldown_revisits"]["count"],
            )

        values = {label: metrics(result) for label, result in results.items()}
        differences = {
            f"policy_minus_{reference}": {
                key: values["policy"][key] - values[reference][key]
                for key in values["policy"]
            }
            for reference in ("independent", "centralized_full_state")
        }
        pairs.append(
            dict(
                seed=seed,
                target_population=config.target_population,
                initial_conditions_sha256=hashes.pop(),
                metrics=values,
                differences=differences,
            )
        )
    write_json(output / "paired_effects.json", pairs)
    if len(pairs) > 1:
        plot_paired_effects(pairs, output / "paired_effects.png")
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
        topology_transition=(
            "same one-worker topology; exact optimizer/worker-seed resume"
            if stage == "validate"
            else "fresh policy: one-to-four-worker seed state has no exact continuation mapping"
        ),
        modes={},
    )
    write_json(
        output / "plan.json",
        {
            **record,
            "held_out_seeds": HELD_OUT_SEEDS,
            "target_population": "mixed_50_30_20",
            "leo_only_evaluation": False,
            "constellation": "Walker Delta 4/2/1, 700 km, 97 deg",
        },
    )
    if stage == "validate":
        # Profile a full physical mission outside RLlib before spending the gate
        # allocation on PPO. This also gives a deterministic communication and
        # resource reference for the exact four-sensor configuration.
        from examples.multiagent_imaging.readiness import profile_episode

        profile = profile_episode(mission_config("conflict"), output / "profile")
        if (
            profile["sim_time_s"] != 45000
            or profile["pettingzoo_agents"]
            != [f"sensor_{index}" for index in range(4)]
            or profile["passive_target_count"] != 100
            or profile["target_regime_counts"] != {"LEO": 50, "MEO": 30, "GEO": 20}
        ):
            raise AssertionError("Mission-scale profile did not match the reviewed contract.")
        record["profile"] = dict(
            measurement=profile["measurement"],
            reimage_cooldown_s=profile["reimage_cooldown_s"],
            coverage=profile["coverage"],
            packet_outcomes=profile["message_diagnostics"]["packet_outcome_counts"],
        )
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
            validate_mission_paths(trained, config)
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


def plot_paired_effects(pairs, destination):
    """Plot paired means with 95% t intervals; five seeds do not prove convergence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import t

    labels = {
        "first_capture_coverage": "First-capture coverage (pp)",
        "ground_delivery_coverage": "Ground coverage (pp)",
        "cooldown_acquisitions": "Cooldown acquisitions",
        "cooldown_ground_services": "Cooldown ground services",
        "duplicate_attempts": "Duplicate attempts",
        "wasted_sensor_seconds": "Wasted sensor-seconds",
    }
    references = ("independent", "centralized_full_state")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes.flat, labels.items()):
        means, errors = [], []
        for reference in references:
            values = np.asarray(
                [
                    pair["differences"][f"policy_minus_{reference}"][metric]
                    for pair in pairs
                ],
                dtype=float,
            )
            if "coverage" in metric:
                values *= 100.0
            means.append(float(values.mean()))
            errors.append(
                float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
            )
        ax.errorbar(range(2), means, yerr=errors, fmt="o", capsize=5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(2), ["Independent", "Centralized\nfull-state"])
        ax.set_ylabel(f"Policy minus reference\n{ylabel}")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Held-out mixed-population paired effects (95% t intervals, n=5)")
    fig.savefig(destination, dpi=180)
    fig.savefig(Path(destination).with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validate", "learn"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-gate", type=Path)
    parser.add_argument("--updates", type=int, choices=range(1, 11), default=8)
    args = parser.parse_args()
    run(args.stage, args.output, gate_path=args.validation_gate, updates=args.updates)
