"""Publish completed frozen evaluations in a separate required-online W&B run.

Argmax and stochastic results retain separate keys. No training experience is
added, no policy is loaded, and an incomplete evaluation cannot be published as
complete. The immutable episode JSON files remain the primary evidence.
"""

import argparse
import json
from pathlib import Path

from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.wandb_logging import TrainingLogger


def publish(root, config):
    root = Path(root)
    summary = json.loads((root / "heldout/summary.json").read_text())
    if (
        not summary["complete"]
        or not summary["initial_conditions_matched"]
        or summary["completed_episodes"] != summary["expected_episodes"]
    ):
        raise ValueError("Only complete matched held-out evaluations can be published")
    output = root / "evaluation-wandb"
    output.mkdir(exist_ok=False)
    source = json.loads((root / "preflight/source-gate.json").read_text())
    logger = TrainingLogger(
        output,
        dict(
            mission=config.to_dict(),
            evidence_kind="frozen heldout; zero PPO updates; no added training experience",
            source=source,
        ),
        mode="online",
        required=True,
    )
    failed = True
    try:
        for row in summary["rows"]:
            prefix = f"heldout/{row['policy']}/{row['mode']}"
            metrics = {
                "heldout/seed": row["seed"],
                "heldout/selection_mode": row["mode"],
            }
            for key in (
                "capture_coverage",
                "ground_coverage",
                "useful_acquisitions",
                "useful_delivered_visits",
                "duplicate_attempts",
                "rollout_wall_s",
            ):
                metrics[f"{prefix}/{key}"] = row[key]
            for key, value in (row.get("cooperative_tasking") or {}).items():
                if isinstance(value, (int, float)):
                    metrics[f"{prefix}/requests/{key}"] = value
            logger._log(metrics)
        failed = False
    finally:
        logger.finish(failed=failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    publish(args.root, MultiAgentImagingConfig.from_json(args.config))


if __name__ == "__main__":
    main()
