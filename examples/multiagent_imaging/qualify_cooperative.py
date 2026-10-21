"""Declared native causal fixtures and finite first-action opportunity frontier.

This is a constructed campaign, not representative mission/policy evaluation.
No products, completion facts, spacecraft states or task endings are injected.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.communication_qualification import (
    run_case,
    credit_trace,
)
from examples.multiagent_imaging.readiness import write_json
from examples.multiagent_imaging.update_evidence import file_sha256


def science(result):
    return sum(sum(row["owned_science_reward"].values()) for row in result["steps"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = MultiAgentImagingConfig.from_json(
        Path(__file__).with_name("configs") / "cooperative_v1_shared_catalog_tx.json"
    )
    # Public urgent tasking, chosen to test receiver-exclusive immediate candidate
    # opportunities. All 200 requests remain active; no image action is removed.
    fixture = replace(
        config,
        active_request_count=200,
        request_window_s=900.0,
        request_priority_overrides=((55, 12.0), (80, 10.0)),
    )
    plan = dict(
        kind="constructed native causal fixture",
        seed=401,
        stop_s=2400,
        horizon_s=45000,
        experiment=fixture.to_dict(),
        receiver_controller="highest public priority among locally eligible illuminated candidates",
        alternative_scope="every legal imaging action plus charge, downlink and desat at sender's t=424 decision; fixed declared subsequent controller; not an optimal-policy bound",
        hypotheses=[
            "received completion prevents duplicate target 55",
            "receiver can instead image urgent target 80",
            "benefit exceeds every tested immediate sender alternative",
        ],
        cluster_jobs=0,
        ppo_updates=0,
    )
    write_json(args.output / "plan.json", plan)
    results = {}

    def run(label, case, **kwargs):
        result = run_case(
            case,
            args.output / label,
            experiment=fixture,
            receiver_priority_controller=True,
            **kwargs,
        )
        results[label] = result
        print(
            json.dumps(
                dict(
                    case=label,
                    passed=result["passed"],
                    science=science(result),
                    reward=result["constellation_reward"],
                )
            ),
            flush=True,
        )
        return result

    success = run("delivered", "delivered")
    for case in ("blocked", "expired", "replay"):
        run(case, case)
    snapshot = success["sender_knowledge_before_send"]
    assert 55 not in snapshot["candidates"]
    outgoing = np.asarray(snapshot["observation"])[26 + 24 * 10 + 18 * 3 :].reshape(
        3, 16, 12
    )
    assert np.any((outgoing[:, :, 0] == 55 / 199) & (outgoing[:, :, -1] == 1))
    assert success["receiver_first_image_decision"]["target_id"] == 80
    assert results["blocked"]["receiver_first_image_decision"]["target_id"] == 55
    alternatives = {}
    for slot, target_id in enumerate(snapshot["candidates"]):
        if target_id is not None and snapshot["mask"][5 + slot]:
            label = f"no_comm_image_{target_id}"
            alternatives[label] = run(
                label, "no_comm_image", alternative_target_id=target_id
            )
    for case in ("no_comm_downlink", "no_comm_charge", "no_comm_desat"):
        alternatives[case] = run(case, case)
    # Actual owned-vs-shared traces use the same physical controller and tasking.
    owned = run_case(
        "delivered",
        args.output / "owned_delivered",
        experiment=replace(fixture, reward_credit_mode="owned"),
        receiver_priority_controller=True,
    )
    owned_blocked = run_case(
        "blocked",
        args.output / "owned_blocked",
        experiment=replace(fixture, reward_credit_mode="owned"),
        receiver_priority_controller=True,
    )
    results.update(owned_delivered=owned, owned_blocked=owned_blocked)
    assert success["physical_trace_sha256"] == owned["physical_trace_sha256"]
    assert (
        results["blocked"]["physical_trace_sha256"]
        == owned_blocked["physical_trace_sha256"]
    )
    for label in ("blocked", "expired", "replay", *alternatives):
        assert results[label]["initial_orbit_sha256"] == success["initial_orbit_sha256"]
    assert science(success) == science(results["replay"])
    best = max(science(r) for r in alternatives.values())
    margin = science(success) - best
    assert margin > 0, "No positive unshaped mission value: PPO gate remains closed"
    # Isolate actual primitive science credit from duplicate costs. This avoids
    # claiming a local redundancy fee is additional sender causal science credit.
    credit = {}
    for label, result in (
        ("shared_delivered", success),
        ("shared_blocked", results["blocked"]),
        ("owned_delivered", owned),
        ("owned_blocked", owned_blocked),
    ):
        raw = dict(result["raw_agent_trajectories"]["sensor_0"])
        raw["rewards"] = [
            row["components"]["sensor_0"]["credited_science"] for row in result["steps"]
        ]
        trace = credit_trace(
            raw,
            fixture.discount_per_s,
            fixture.effective_gae_lambda,
            lambda_time="second",
        )
        credit[label] = trace

    def tx_target(trace):
        return next(
            row["diagnostic_value_target"]
            for row in trace["rows"]
            if row["action"] >= 15
        )

    shared = tx_target(credit["shared_delivered"]) - tx_target(credit["shared_blocked"])
    own = tx_target(credit["owned_delivered"]) - tx_target(credit["owned_blocked"])
    assert shared > 0 and abs(own) < 1e-7
    # Rerun the original unfavorable example under its unchanged baseline.
    negative = {}
    for case in ("delivered", "blocked", "no_comm_image"):
        negative[case] = run_case(case, args.output / ("negative_" + case))
    negative_margin = science(negative["delivered"]) - science(
        negative["no_comm_image"]
    )
    assert negative_margin < 0
    summary = dict(
        passed=True,
        kind=plan["kind"],
        representative_evaluation=False,
        unshaped_science=science(success),
        best_alternative_science=best,
        net_unshaped_value=margin,
        information_value=science(success) - science(results["blocked"]),
        alternative_science={k: science(v) for k, v in alternatives.items()},
        negative_baseline_margin=negative_margin,
        sender_shared_diagnostic_target_increment=shared,
        sender_owned_diagnostic_target_increment=own,
        sender_credit=credit,
        native_build=success["native_build"],
        all_initial_orbits_matched=True,
        owned_shared_physics_matched=True,
        outgoing_completed_record_observed=True,
        ppo_gate_passed=True,
        ppo_updates=0,
        cluster_jobs=0,
        fixture_sha256={
            str(p.relative_to(args.output)): file_sha256(p)
            for p in args.output.glob("*/fixture.json")
        },
    )
    write_json(args.output / "qualification.json", summary)
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("sender_credit", "native_build", "fixture_sha256")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
