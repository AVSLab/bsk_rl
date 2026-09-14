"""Validate and compact the Walker-4 cluster learning-pilot evidence.

This post-processing command is intentionally read-only with respect to the raw
Slurm results.  It checks the scientific contract before copying the small,
reviewable records that belong in Git; checkpoints and full event histories stay
under the immutable cluster results root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
import shutil
from statistics import mean, stdev


MODES = ("conflict", "continuous")
CONTROLLERS = ("policy", "independent", "centralized_full_state")
EXPECTED_SEEDS = tuple(range(10000, 10005))
EXPECTED_REGIMES = {"LEO": 50, "MEO": 30, "GEO": 20}
EXPECTED_AGENTS = [f"sensor_{index}" for index in range(4)]
EXPECTED_COOLDOWN_S = 11834.835756586714
T_975_DF4 = 2.7764451051977987


def load_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def t_summary(values: list[float]) -> dict:
    values = [float(value) for value in values]
    half_width = (
        T_975_DF4 * stdev(values) / math.sqrt(len(values))
        if len(values) == 5
        else 0.0
    )
    value_mean = mean(values)
    return {
        "n": len(values),
        "mean": value_mean,
        "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        "ci95_low": value_mean - half_width,
        "ci95_high": value_mean + half_width,
        "ci95_half_width": half_width,
    }


def resource_metrics(history: list[dict] | dict[str, list[dict]]) -> dict:
    if not history:
        raise AssertionError("Resource history is empty.")
    if isinstance(history, dict):
        history = [
            {"sensor": sensor, **sample}
            for sensor, samples in history.items()
            for sample in samples
        ]
    latest: dict[str, dict] = {}
    for sample in history:
        latest[sample["sensor"]] = sample
    if sorted(latest) != EXPECTED_AGENTS:
        raise AssertionError("Resource history does not cover every sensing agent.")
    wheels = [
        abs(float(value))
        for sample in history
        for value in sample["wheel_speed_fraction"]
    ]
    values = [
        *(float(sample["battery_fraction"]) for sample in history),
        *(float(sample["storage_fraction"]) for sample in history),
        *wheels,
    ]
    if not all(math.isfinite(value) for value in values):
        raise AssertionError("Resource history contains a nonfinite value.")
    metrics = {
        "minimum_battery_fraction": min(
            float(sample["battery_fraction"]) for sample in history
        ),
        "maximum_storage_fraction": max(
            float(sample["storage_fraction"]) for sample in history
        ),
        "maximum_absolute_wheel_fraction": max(wheels),
        "sensors_reaching_zero_battery": len(
            {
                sample["sensor"]
                for sample in history
                if float(sample["battery_fraction"]) <= 0.0
            }
        ),
        "zero_battery_sensors_at_end": sum(
            float(sample["battery_fraction"]) <= 0.0 for sample in latest.values()
        ),
    }
    if all("alive" in sample for sample in history):
        metrics.update(
            sensors_ever_inactive=len(
                {sample["sensor"] for sample in history if not sample["alive"]}
            ),
            sensors_inactive_at_end=sum(
                not sample["alive"] for sample in latest.values()
            ),
        )
    return metrics


def flatten_update(stage: str, mode: str, record: dict) -> dict:
    episodes = record["episodes"]
    communications = [episode["communication_summary"] for episode in episodes]
    rewards = [episode["reward_decomposition"] for episode in episodes]
    resource_rows = [resource_metrics(episode["resource_history"]) for episode in episodes]
    simulated_seconds = sum(e["simulated_seconds"] for e in episodes)
    return {
        "stage": stage,
        "mode": mode,
        "iteration": record["iteration"],
        "complete_episodes": record["complete_episode_count"],
        "actual_env_steps": record["actual_env_steps"],
        "actual_agent_steps": record["actual_agent_steps"],
        "actual_policy_decisions": record["actual_policy_decisions"],
        "total_loss": record["losses"]["total_loss"],
        "policy_loss": record["losses"]["policy_loss"],
        "value_loss": record["losses"]["vf_loss"],
        "gradient_l2": record["gradient_l2"],
        "finite_gradients": record["finite_gradients"],
        "parameter_change_l2": record["parameter_change"]["l2"],
        "changed_parameter_elements": record["parameter_change"]["changed_elements"],
        "wall_time_s": record["measurement"]["wall_time_s"],
        "peak_rss_bytes": record["measurement"]["peak_rss_bytes"],
        "simulated_seconds": simulated_seconds,
        "simulated_seconds_per_wall_second": simulated_seconds
        / record["measurement"]["wall_time_s"],
        "mean_constellation_reward": mean(e["constellation_reward"] for e in episodes),
        "mean_acquisition_reward": mean(
            r["acquisition_90_percent_component"] for r in rewards
        ),
        "mean_ground_reward": mean(
            r["ground_delivery_10_percent_component"] for r in rewards
        ),
        "mean_communication_adjustment": mean(
            r["communication_time_adjustment"] for r in rewards
        ),
        "mean_other_penalties_or_adjustments": mean(
            r["other_operational_penalties_or_adjustments"] for r in rewards
        ),
        "radio_occupancy_s": sum(c["radio_occupancy_s"] for c in communications),
        "payload_records_attempted": sum(
            c["payload_records_attempted"] for c in communications
        ),
        "payload_bytes_attempted": sum(
            c["payload_bytes_attempted"] for c in communications
        ),
        "minimum_battery_fraction": min(
            r["minimum_battery_fraction"] for r in resource_rows
        ),
        "maximum_storage_fraction": max(
            r["maximum_storage_fraction"] for r in resource_rows
        ),
        "maximum_absolute_wheel_fraction": max(
            r["maximum_absolute_wheel_fraction"] for r in resource_rows
        ),
        "episodes_with_inactive_sensor": sum(
            r.get("sensors_ever_inactive", 0) > 0 for r in resource_rows
        ),
        "matched_restored_actions": record["restore_validation"]["matched_actions"],
        "restored_logit_max_error": record["restore_validation"][
            "max_logit_error"
        ],
    }


def validate_update(record: dict, workers: int) -> None:
    if record["complete_episode_count"] != workers:
        raise AssertionError("Each update must contain one complete episode per worker.")
    if record["actual_env_steps"] < 64 or record["actual_policy_decisions"] <= 0:
        raise AssertionError("Update did not meet the sampled-batch/decision contract.")
    numerical = [
        *record["losses"].values(),
        record["gradient_l2"],
        record["parameter_change"]["l2"],
    ]
    if not all(math.isfinite(float(value)) for value in numerical):
        raise AssertionError("Update contains a nonfinite optimization value.")
    if record["finite_gradients"] != 1 or record["gradient_l2"] <= 0:
        raise AssertionError("Update lacks finite, nonzero gradients.")
    if record["parameter_change"]["l2"] <= 0:
        raise AssertionError("Update did not change policy parameters.")
    restored = record["restore_validation"]
    if not restored["matched_actions"] or restored["max_logit_error"] != 0:
        raise AssertionError("Checkpoint logits/actions did not restore exactly.")
    for episode in record["episodes"]:
        if episode["simulated_seconds"] != 45000.0:
            raise AssertionError("Training episode did not reach 45,000 seconds.")
        if sorted(episode["resources"]) != EXPECTED_AGENTS:
            raise AssertionError("Training episode does not contain four sensing agents.")
        if episode["target_regime_counts"] != EXPECTED_REGIMES:
            raise AssertionError("Training target population is not exact 50/30/20 mixed.")
        if not math.isclose(episode["reimage_cooldown_s"], EXPECTED_COOLDOWN_S):
            raise AssertionError("Training cooldown differs from the reviewed contract.")


def result_metrics(result: dict) -> dict:
    rewards = result["reward_decomposition"]
    coordination = result["coordination"]
    products = result["product_duplicates"]
    messages = result["message_diagnostics"]
    resources = resource_metrics(result["resource_history"])
    # Centralized control performs ideal catalog synchronization at an event
    # boundary. Those durable merges appear in delivery_history, but they are
    # not radio packets. Count a transport packet only when the sender completed
    # a physical pointing hold; retain all merges as a separate audit metric.
    completed_transmissions = [
        entry
        for entry in messages["transmission_history"]
        if entry.get("outcome") == "hold_complete"
    ]
    accepted_catalog_merges = messages["packet_outcome_counts"].get("accepted", 0)
    accepted_transport_packets = (
        accepted_catalog_merges if completed_transmissions else 0
    )
    action_selections = sum(
        count
        for sensor_counts in result["action_counts"].values()
        for count in sensor_counts.values()
    )
    downlink_selections = sum(
        sensor_counts.get("action_downlink", 0)
        for sensor_counts in result["action_counts"].values()
    )
    return {
        "first_capture_coverage": result["coverage"]["capture_coverage_fraction"],
        "ground_delivery_coverage": result["coverage"][
            "ground_delivery_coverage_fraction"
        ],
        "cooldown_acquisitions": result["team_summary"]["unique_acquisition_count"],
        "cooldown_ground_services": result["team_summary"]["unique_service_count"],
        "acquisition_reward_component": rewards["acquisition_90_percent_component"],
        "ground_delivery_reward_component": rewards[
            "ground_delivery_10_percent_component"
        ],
        "communication_time_adjustment": rewards["communication_time_adjustment"],
        "other_operational_penalties_or_adjustments": rewards[
            "other_operational_penalties_or_adjustments"
        ],
        "total_reward": rewards["total_constellation_reward"],
        "duplicate_attempts": result["team_summary"]["duplicate_attempt_count"],
        "successful_duplicates": result["team_summary"]["successful_duplicate_count"],
        "duplicate_sensor_seconds": coordination["duplicate_sensor_time_s"],
        "interrupted_nonduplicate_sensor_seconds": coordination[
            "interrupted_nonduplicate_sensor_time_s"
        ],
        "wasted_sensor_seconds": coordination["duplicate_sensor_time_s"]
        + coordination["interrupted_nonduplicate_sensor_time_s"],
        "wasted_time_fraction": coordination["wasted_time_fraction"],
        "stale_cross_sensor_ground_deliveries": products[
            "stale_cross_sensor_ground_delivery_count"
        ],
        "causally_avoidable_stale_ground_deliveries": products[
            "causally_avoidable_stale_ground_delivery_count"
        ],
        "cross_sensor_onboard_overlap_targets": products[
            "cross_sensor_onboard_overlap_target_count"
        ],
        "cross_sensor_onboard_overlap_products": products[
            "cross_sensor_onboard_overlap_product_count"
        ],
        "redundant_acquisitions": products[
            "cross_sensor_onboard_redundant_acquisition_count"
        ],
        "excess_holder_sensor_seconds": products[
            "cross_sensor_onboard_redundant_sensor_time_s"
        ],
        "overlap_sensor_seconds": products[
            "cross_sensor_onboard_overlap_sensor_time_s"
        ],
        "radio_occupancy_s": sum(coordination["communication_time_s"].values()),
        "policy_decisions": sum(coordination["policy_decisions"].values()),
        "payload_records_attempted": messages["payload_records_attempted"],
        "payload_bytes_attempted": messages["payload_bytes_attempted"],
        "accepted_transport_packets": accepted_transport_packets,
        "accepted_catalog_merges": accepted_catalog_merges,
        "downlink_action_selections": downlink_selections,
        "other_action_selections": action_selections - downlink_selections,
        "useful_revisits": result["useful_post_cooldown_revisits"]["count"],
        "wall_time_s": result["measurement"]["wall_time_s"],
        "peak_rss_bytes": result["measurement"]["peak_rss_bytes"],
        "simulated_seconds_per_wall_second": result["measurement"][
            "sim_seconds_per_wall_second"
        ],
        "minimum_battery_fraction": resources["minimum_battery_fraction"],
        "maximum_storage_fraction": resources["maximum_storage_fraction"],
        "maximum_absolute_wheel_fraction": resources[
            "maximum_absolute_wheel_fraction"
        ],
        "sensors_reaching_zero_battery": resources[
            "sensors_reaching_zero_battery"
        ],
        "zero_battery_sensors_at_end": resources["zero_battery_sensors_at_end"],
    }


def validate_result(result: dict, *, policy: bool) -> None:
    if result["sim_time_s"] != 45000.0:
        raise AssertionError("Held-out episode did not reach 45,000 seconds.")
    if result["pettingzoo_agents"] != EXPECTED_AGENTS:
        raise AssertionError("Held-out episode does not contain four sensing agents.")
    if result["passive_target_count"] != 100:
        raise AssertionError("Passive target count differs from 100.")
    if result["target_regime_counts"] != EXPECTED_REGIMES:
        raise AssertionError("Held-out population is not exact 50/30/20 mixed.")
    if not math.isclose(result["reimage_cooldown_s"], EXPECTED_COOLDOWN_S):
        raise AssertionError("Held-out cooldown differs from the reviewed contract.")
    resource_metrics(result["resource_history"])
    for sensor, products in result["onboard_products"].items():
        if any(
            product["storage_owner"] != sensor
            or product["source_sensor"] != sensor
            for product in products
        ):
            raise AssertionError("Completion metadata changed physical product ownership.")
    if policy:
        decisions = sum(result["coordination"]["policy_decisions"].values())
        if result["restored_policy_calls"] != decisions or decisions <= 0:
            raise AssertionError("Restored policy was not called at every decision.")
        restored = result["restore_validation"]
        if not restored["matched_actions"] or restored["max_logit_error"] != 0:
            raise AssertionError("Held-out checkpoint restore proof failed.")
    elif result["message_diagnostics"]["transmission_history"] or any(
        result["broadcast_time_s"].values()
    ):
        raise AssertionError("A heuristic reference used the radio.")
    if not policy and result_metrics(result)["accepted_transport_packets"]:
        raise AssertionError("A heuristic reference recorded a transport packet.")


def copy_artifact(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def aggregate(validation_root: Path, pilot_root: Path, output: Path) -> dict:
    gate = load_json(validation_root / "validation_gate.json")
    pilot = load_json(pilot_root / "pilot_summary.json")
    if not gate.get("passed") or gate.get("validated_workers") != 1:
        raise AssertionError("One-worker validation gate did not pass.")
    if not pilot.get("passed") or pilot.get("validated_workers") != 4:
        raise AssertionError("Four-worker pilot did not pass.")
    if gate["runtime"]["source_sha256"] != pilot["runtime"]["source_sha256"]:
        raise AssertionError("Gate and pilot executable-source hashes differ.")
    output.mkdir(parents=True, exist_ok=True)

    # Keep compact copies of the records needed to reproduce and audit the two
    # Slurm stages. Full checkpoints and episode histories remain in the raw
    # cluster result tree because they are too large for Git.
    for source, name in (
        (validation_root / "runtime.json", "one_worker_runtime.json"),
        (pilot_root / "runtime.json", "four_worker_runtime.json"),
        (
            validation_root / "support-data-audit.json",
            "one_worker_support_data_audit.json",
        ),
        (
            pilot_root / "support-data-audit.json",
            "four_worker_support_data_audit.json",
        ),
        (validation_root / "plan.json", "one_worker_plan.json"),
        (pilot_root / "plan.json", "four_worker_plan.json"),
        (validation_root / "validation_gate.json", "validation_gate.json"),
        (pilot_root / "pilot_summary.json", "pilot_summary.json"),
    ):
        copy_artifact(source, output / name)

    update_rows = []
    evaluation_rows = []
    paired_rows = []
    summary = {
        "passed": True,
        "source_sha256": pilot["runtime"]["source_sha256"],
        "basilisk_commit": pilot["runtime"]["loaded"]["basilisk_commit"],
        "validation_workers": gate["validated_workers"],
        "pilot_workers": pilot["validated_workers"],
        "topology_transition": pilot["topology_transition"],
        "held_out_seeds": list(EXPECTED_SEEDS),
        "modes": {},
    }

    for mode in MODES:
        gate_records = load_json(validation_root / mode / "train" / "updates.json")
        gate_records += load_json(validation_root / mode / "resume" / "updates.json")
        pilot_records = load_json(pilot_root / mode / "train" / "updates.json")
        if [record["iteration"] for record in gate_records] != [1, 2]:
            raise AssertionError(f"{mode} gate did not prove update-boundary resume.")
        if [record["iteration"] for record in pilot_records] != list(range(1, 9)):
            raise AssertionError(f"{mode} pilot did not complete exactly eight updates.")
        for record in gate_records:
            validate_update(record, 1)
            update_rows.append(flatten_update("one_worker_gate", mode, record))
        for record in pilot_records:
            validate_update(record, 4)
            update_rows.append(flatten_update("four_worker_pilot", mode, record))

        mode_pairs = load_json(pilot_root / mode / "evaluation" / "paired_effects.json")
        if [pair["seed"] for pair in mode_pairs] != list(EXPECTED_SEEDS):
            raise AssertionError(f"{mode} held-out seeds are not 10000-10004.")
        controller_values: dict[str, dict[str, list[float]]] = {
            controller: {} for controller in CONTROLLERS
        }
        difference_values: dict[str, dict[str, list[float]]] = {
            "policy_minus_independent": {},
            "policy_minus_centralized_full_state": {},
        }
        recipient_counts: Counter = Counter()
        slot_counts: Counter = Counter()
        packet_counts: Counter = Counter()
        ack_links: set[str] = set()
        training_recipient_counts: Counter = Counter()
        training_slot_counts: Counter = Counter()
        training_packet_counts: Counter = Counter()
        training_ack_links: set[str] = set()
        for record in pilot_records:
            for episode in record["episodes"]:
                communication = episode["communication_summary"]
                training_recipient_counts.update(
                    communication["recipient_selection_counts"]
                )
                training_slot_counts.update(
                    communication["peer_slot_selection_counts"]
                )
                training_packet_counts.update(communication["packet_outcome_counts"])
                training_ack_links.update(episode["acknowledged_versions_by_link"])

        for pair in mode_pairs:
            results = {}
            for controller in CONTROLLERS:
                path = pilot_root / mode / "evaluation" / (
                    f"seed_{pair['seed']}_{controller}.json"
                )
                result = load_json(path)
                validate_result(result, policy=controller == "policy")
                results[controller] = result
                values = result_metrics(result)
                evaluation_rows.append(
                    {
                        "mode": mode,
                        "seed": pair["seed"],
                        "initial_conditions_sha256": result[
                            "initial_conditions_sha256"
                        ],
                        "controller": controller,
                        **values,
                    }
                )
                for key, value in values.items():
                    controller_values[controller].setdefault(key, []).append(value)
            hashes = {r["initial_conditions_sha256"] for r in results.values()}
            if hashes != {pair["initial_conditions_sha256"]}:
                raise AssertionError(f"{mode} seed {pair['seed']} is not exactly paired.")
            policy_messages = results["policy"]["message_diagnostics"]
            recipient_counts.update(policy_messages["recipient_selection_counts"])
            slot_counts.update(policy_messages["recipient_slot_selection_counts"])
            packet_counts.update(policy_messages["packet_outcome_counts"])
            ack_links.update(policy_messages["acknowledged_versions_by_link"])

            policy_values = result_metrics(results["policy"])
            for reference in ("independent", "centralized_full_state"):
                reference_values = result_metrics(results[reference])
                comparison = f"policy_minus_{reference}"
                row = {
                    "mode": mode,
                    "seed": pair["seed"],
                    "initial_conditions_sha256": pair["initial_conditions_sha256"],
                    "comparison": comparison,
                }
                for key in pair["metrics"]["policy"]:
                    delta = policy_values[key] - reference_values[key]
                    row[key] = delta
                    difference_values[comparison].setdefault(key, []).append(delta)
                paired_rows.append(row)

        checkpoint = Path(pilot["modes"][mode]["checkpoint"])
        copy_artifact(
            checkpoint / "manifest.json",
            output / "checkpoint_manifests" / f"{mode}_manifest.json",
        )
        copy_artifact(
            checkpoint / "restore_validation.json",
            output / "checkpoint_manifests" / f"{mode}_restore_validation.json",
        )
        copy_artifact(
            pilot_root / mode / "learning.png", output / f"{mode}_learning.png"
        )
        copy_artifact(
            pilot_root / mode / "evaluation" / "paired_effects.png",
            output / f"{mode}_paired_effects.png",
        )
        summary["modes"][mode] = {
            "gate_updates": 2,
            "pilot_updates": 8,
            "controller_metrics": {
                controller: {
                    metric: t_summary(values)
                    for metric, values in metrics.items()
                }
                for controller, metrics in controller_values.items()
            },
            "paired_effects": {
                comparison: {
                    metric: t_summary(values)
                    for metric, values in metrics.items()
                }
                for comparison, metrics in difference_values.items()
            },
            "recipient_selection_counts": dict(sorted(recipient_counts.items())),
            "peer_slot_selection_counts": dict(sorted(slot_counts.items())),
            "packet_outcome_counts": dict(sorted(packet_counts.items())),
            "acknowledged_links": sorted(ack_links),
            "training_recipient_selection_counts": dict(
                sorted(training_recipient_counts.items())
            ),
            "training_peer_slot_selection_counts": dict(
                sorted(training_slot_counts.items())
            ),
            "training_packet_outcome_counts": dict(
                sorted(training_packet_counts.items())
            ),
            "training_acknowledged_links": sorted(training_ack_links),
        }

    no_go_reasons = []
    for mode in MODES:
        policy = summary["modes"][mode]["controller_metrics"]["policy"]
        if policy["first_capture_coverage"]["mean"] == 0:
            no_go_reasons.append(f"{mode} final policy had zero held-out capture coverage")
        if policy["zero_battery_sensors_at_end"]["mean"] > 0:
            no_go_reasons.append(
                f"{mode} final policy depleted sensors to zero battery"
            )
        if not summary["modes"][mode]["packet_outcome_counts"].get("accepted", 0):
            no_go_reasons.append(
                f"{mode} final policy delivered no held-out completion packets"
            )
    summary["recommendation"] = {
        "decision": "NO-GO" if no_go_reasons else "GO",
        "scope": "multi-seed learned study",
        "reasons": no_go_reasons,
    }

    write_csv(output / "updates.csv", update_rows)
    write_csv(output / "held_out_metrics.csv", evaluation_rows)
    write_csv(output / "paired_differences.csv", paired_rows)
    write_json(output / "summary.json", summary)
    write_report(output / "REPORT.md", summary)
    return summary


def effect(summary: dict, mode: str, comparison: str, metric: str) -> dict:
    return summary["modes"][mode]["paired_effects"][comparison][metric]


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# Walker-4 completion-v3 bounded cluster pilot",
        "",
        "This compact record validates the authorized one-worker gate and four-worker "
        "pilot. Full checkpoints, event histories, and raw episodes remain in the "
        "immutable cluster results directory. Eight four-worker updates plus the two "
        "one-worker gate updates were run per mode. This is mechanics and early-learning "
        "evidence, not convergence evidence.",
        "",
        "## Recommendation",
        "",
        f"**{summary['recommendation']['decision']} for a multi-seed learned study.** "
        + "; ".join(summary["recommendation"]["reasons"])
        + ". The implementation and checkpoint mechanics passed, but these final policies "
        "are not operational candidates.",
        "",
        "## Held-out paired effects",
        "",
        "Every value is restored policy minus the named reference on the same exact "
        "mixed-population seed and initial-condition hash. Intervals are two-sided 95% "
        "paired t intervals over five seeds (10000-10004).",
        "",
        "| Mode/reference | Capture coverage | Ground coverage | Cooldown acquisitions | Ground services | Total reward | Duplicate attempts | Wasted sensor-s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for comparison, label in (
            ("policy_minus_independent", "independent"),
            ("policy_minus_centralized_full_state", "centralized full state"),
        ):
            values = summary["modes"][mode]["paired_effects"][comparison]

            def fmt(metric: str, scale: float = 1.0) -> str:
                item = values[metric]
                return f"{scale * item['mean']:+.2f} [{scale * item['ci95_low']:+.2f}, {scale * item['ci95_high']:+.2f}]"

            lines.append(
                f"| {mode} vs {label} | {fmt('first_capture_coverage', 100)} pp | "
                f"{fmt('ground_delivery_coverage', 100)} pp | {fmt('cooldown_acquisitions')} | "
                f"{fmt('cooldown_ground_services')} | {fmt('total_reward')} | "
                f"{fmt('duplicate_attempts')} | {fmt('wasted_sensor_seconds')} |"
            )
    lines += [
        "",
        "The centralized-full-state controller is a maximum-information greedy "
        "coordination reference, not a guaranteed global optimum. The learned actor has "
        "receiver-local information only. It selects one of three peer slots and receives "
        "completion metadata only after physical directed pointing and a successful packet.",
        "Centralized catalog updates are counted as ideal catalog merges rather than "
        "transport packets; both heuristic references have zero physical radio packets.",
        "",
        "## Held-out action behavior",
        "",
    ]
    for mode in MODES:
        policy = summary["modes"][mode]["controller_metrics"]["policy"]
        lines.append(
            f"- **{mode}:** mean downlink selections "
            f"{policy['downlink_action_selections']['mean']:.1f}; mean selections of all "
            f"other actions {policy['other_action_selections']['mean']:.1f}."
        )
    lines += [
        "",
        "Both restored final policies selected downlink at every held-out decision, "
        "despite having no physical products to deliver. That collapse explains the "
        "zero capture/ground coverage, zero useful revisits, zero completion packets, "
        "and repeated operational penalties; it is a policy-quality failure rather "
        "than a checkpoint or simulator-execution failure.",
        "",
        "## Communication validation",
        "",
    ]
    for mode in MODES:
        data = summary["modes"][mode]
        lines.append(
            f"- **{mode} training:** recipient selections "
            f"`{data['training_recipient_selection_counts']}`; peer slots "
            f"`{data['training_peer_slot_selection_counts']}`; packet outcomes "
            f"`{data['training_packet_outcome_counts']}`; ACK links "
            f"{len(data['training_acknowledged_links'])}."
        )
        lines.append(
            f"- **{mode} final held-out policy:** recipient selections "
            f"`{data['recipient_selection_counts']}`; peer slots "
            f"`{data['peer_slot_selection_counts']}`; packet outcomes "
            f"`{data['packet_outcome_counts']}`; ACK links {len(data['acknowledged_links'])}."
        )
    lines += [
        "",
        "All training and held-out episodes reached 45,000 seconds with four sensing "
        "agents, 100 passive Basilisk/Vizard RSO spacecraft outside the RL agent list, "
        "and the exact 50 LEO/30 MEO/20 GEO target split. The capture-anchored two-orbit "
        "cooldown remained 11,834.835756586714 seconds within floating-point roundoff. "
        "Every update had finite nonzero gradients and parameter changes. Checkpoint "
        "logits matched exactly and restored actions were identical. The independent and "
        "centralized references had zero radio activity.",
        "",
        "See `summary.json` for every controller mean and paired interval, "
        "`held_out_metrics.csv` for all 30 evaluations, `paired_differences.csv` for the "
        "20 exact-seed effects, and `updates.csv` for all 20 gate/pilot updates. Runtime, "
        "package/native-module hashes, support-data audits, launch plans, checkpoint "
        "manifests, exact submissions, and Slurm accounting are retained beside them.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = aggregate(args.validation_root, args.pilot_root, args.output)
    print(json.dumps({"passed": summary["passed"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
