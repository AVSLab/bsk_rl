"""Independent local mission checks over verified episode files, one at a time."""

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    rows = []
    seen = set()
    for path in sorted((ROOT / "collected/canary/training/episode-spool").glob("*.json")):
        episode = json.loads(path.read_text())
        assert episode["episode_id"] not in seen
        seen.add(episode["episode_id"])
        assert math.isclose(episode["simulated_seconds"], 45000, abs_tol=1e-6)
        captures = episode["capture_products"]
        assert len({p["record_id"] for p in captures}) == len(captures)
        by_id = {p["record_id"]: p for p in captures}
        for item in episode["delivered_products"]:
            product = item["product"]
            original = by_id[product["record_id"]]
            assert product["source_sensor"] == product["storage_owner"] == original["source_sensor"]
            assert product["delivery_time"] >= product["completion_time"] >= product["capture_time"]
        task = episode["cooperative_tasking"]
        active = {}
        qualified = []
        for product in captures:
            index = int(max(0, product["capture_time"]) // task["window_s"])
            if index not in active:
                rng = np.random.default_rng(np.random.SeedSequence([episode["episode_seed"], index, 731]))
                active[index] = set(int(i) for i in rng.choice(200, task["active_request_count"], replace=False))
            if (product["target_id"] in active[index]
                and product["request_epoch"] == index * task["window_s"]
                and product["quality"] >= task["quality_threshold"]
                and product["completion_time"] <= (index + 1) * task["window_s"]):
                qualified.append(product)
        last = {}
        unique = 0
        for product in sorted(qualified, key=lambda p: (p["capture_time"], p["record_id"])):
            key = (product["target_id"], product["request_epoch"])
            if key not in last or product["capture_time"] >= last[key] + episode["reimage_cooldown_s"]:
                unique += 1
                last[key] = product["capture_time"]
        assert unique == task["unique_acquisition_requests"], (episode["episode_id"], unique, task)
        assert len(captures) - len(qualified) == task["rejected_acquisitions"]
        # Historical mission metric counts quality-qualified deliveries. The
        # physical receipt ledger also retains below-threshold products.
        qualified_deliveries = sum(item["product"]["quality"] >= 0.5 for item in episode["delivered_products"])
        assert qualified_deliveries == sum(v["delivered_products"] for v in episode["mission_metrics"]["agents"].values())
        science = 0.9 * task["acquisition_value"] + 0.1 * task["verified_delivery_value"]
        assert math.isclose(science, task["unshaped_weighted_science"], abs_tol=1e-9)
        rows.append(dict(
            episode_id=episode["episode_id"], episode_seed=episode["episode_seed"],
            simulated_seconds=episode["simulated_seconds"], true_decisions=sum(episode["policy_decisions"].values()),
            physical_completed_captures=len(captures), qualified_request_captures=len(qualified),
            unique_request_acquisitions=unique, completed_redundant_request_captures=len(qualified)-unique,
            physical_deliveries=len(episode["delivered_products"]), quality_qualified_deliveries=qualified_deliveries,
            useful_delivered_requests=task["useful_delivered_requests"],
            unshaped_weighted_science=science, redundant_capture_cost=task["redundant_capture_cost"],
            radio_occupancy_s=episode["communication_summary"]["radio_occupancy_s"],
            packet_outcomes=episode["communication_summary"]["packet_outcome_counts"],
        ))
    assert len(rows) == 56
    numeric = [k for k in rows[0] if k not in {"episode_id", "episode_seed", "packet_outcomes", "simulated_seconds"}]
    result = dict(
        passed=True, missions=len(rows), totals={k:sum(r[k] for r in rows) for k in numeric},
        means={k:sum(r[k] for r in rows)/len(rows) for k in numeric}, missions_audit=rows,
        independent_checks=["full horizons", "unique product identities", "physical ownership and delivery ordering", "seeded active request qualification", "completed duplicate versus unique request acquisition", "physical delivery report consistency", "unshaped acquisition/delivery weighting"],
        comparison="One communication-enabled sampling run only. No matched control or held-out evaluation; no reduction or policy improvement claim.",
    )
    destination = ROOT / "mission-audit.json"
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k != "missions_audit"}, indent=2))


if __name__ == "__main__":
    main()
