"""Run the six paired information/retasking cells on identical initial states."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.evaluate import run_rollout

INFORMATION_CASES = ("independent", "ideal_completion", "completion")
RETASKING_MODES = ("conflict", "continuous")
PAIRING_EXCEPTIONS = {"information_case", "retasking_mode"}


def _pairing_signature(config: MultiAgentImagingConfig) -> dict:
    return {k: v for k, v in config.to_dict().items() if k not in PAIRING_EXCEPTIONS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "completion_conflict.json",
    )
    parser.add_argument("--duration", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/multiagent_imaging/completion_validation"),
    )
    args = parser.parse_args()
    base = MultiAgentImagingConfig.from_json(args.config)
    if args.duration is not None:
        base = replace(base, episode_duration_s=args.duration)
    if args.seed is not None:
        base = replace(base, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, reference = [], None
    for case in INFORMATION_CASES:
        for mode in RETASKING_MODES:
            config = replace(base, information_case=case, retasking_mode=mode)
            result = run_rollout(config)
            if reference is None:
                reference = result["initial_conditions"]
            elif result["initial_conditions"] != reference:
                raise RuntimeError(
                    "Paired cases did not reproduce identical initial states."
                )
            name = f"{case}_{mode}"
            output = args.output_dir / f"{name}.json"
            output.write_text(
                json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            results.append(
                dict(
                    case=name,
                    output=str(output),
                    **{
                        key: result[key]
                        for key in (
                            "cumulative_reward",
                            "team_summary",
                            "coordination",
                            "message_diagnostics",
                            "concurrent_target_conflicts",
                            "target_omission_diagnostics",
                        )
                    },
                )
            )
            print(name, result["sim_time_s"], result["team_summary"], flush=True)
    summary = dict(
        matched_fields=_pairing_signature(base),
        initial_conditions_identical=True,
        cases=results,
    )
    output = args.output_dir / "summary.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
