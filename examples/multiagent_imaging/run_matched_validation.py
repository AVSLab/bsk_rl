"""Run the four bounded, paired information-case validation rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.evaluate import run_rollout


DEFAULT_CONFIGS = (
    "validation_independent.json",
    "validation_centralized_information.json",
    "validation_intent_perfect.json",
    "validation_intent_los.json",
)
PAIRING_EXCEPTIONS = {"information_case", "perfect_metadata_delivery"}


def _pairing_signature(config: MultiAgentImagingConfig) -> dict:
    return {
        key: value
        for key, value in config.to_dict().items()
        if key not in PAIRING_EXCEPTIONS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/multiagent_imaging/matched_validation"),
    )
    args = parser.parse_args()
    config_dir = Path(__file__).parent / "configs"
    configs = [
        MultiAgentImagingConfig.from_json(config_dir / name) for name in DEFAULT_CONFIGS
    ]
    signatures = [_pairing_signature(config) for config in configs]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError(
            "Matched validation configurations differ beyond information case."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    reference_initial_conditions = None
    for name, config in zip(DEFAULT_CONFIGS, configs):
        result = run_rollout(config)
        if reference_initial_conditions is None:
            reference_initial_conditions = result["initial_conditions"]
        elif result["initial_conditions"] != reference_initial_conditions:
            raise RuntimeError(
                "Paired cases did not reproduce identical initial states."
            )
        case_name = Path(name).stem.removeprefix("validation_")
        output = args.output_dir / f"{case_name}.json"
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        results.append(
            {
                "case": case_name,
                "output": str(output),
                "cumulative_reward": result["cumulative_reward"],
                "team_summary": result["team_summary"],
                "intent_conflicts": result["intent_conflicts"],
                "broadcast_time_s": result["broadcast_time_s"],
                "message_diagnostics": result["message_diagnostics"],
                "target_omission_diagnostics": result["target_omission_diagnostics"],
            }
        )
    summary = {
        "matched_fields": signatures[0],
        "initial_conditions_identical": True,
        "cases": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(summary_path.resolve())


if __name__ == "__main__":
    main()
