"""Evaluate frozen policies on matched seeds, separately sampling and using argmax."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import math
import time
import concurrent.futures
import multiprocessing

from examples.multiagent_imaging.checkpoints import load_policy, policy_callable
from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.evaluate import run_rollout
from examples.multiagent_imaging.readiness import write_json


def compare(config, checkpoints, seeds, output, workers=1, modes=(True, False)):
    if workers < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("Positive worker count and unique seeds required")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh validation directory.")
    output.mkdir(parents=True, exist_ok=True)
    if workers > 1:
        return parallel_compare(config, checkpoints, seeds, output, workers, modes)
    references, rows = {}, []
    for label, checkpoint in checkpoints.items():
        module, restore = load_policy(checkpoint, config)
        for seed in seeds:
            for deterministic in modes:
                mode = "deterministic" if deterministic else "stochastic"
                rollout_start=time.monotonic()
                result = run_rollout(
                    replace(config, seed=seed),
                    policy=policy_callable(
                        module, deterministic=deterministic, seed=seed + 1701
                    ),
                )
                result["rollout_wall_s"]=time.monotonic()-rollout_start
                result["selection_mode"]=mode
                initial_hash = result["initial_conditions_sha256"]
                if seed in references and references[seed] != initial_hash:
                    raise AssertionError(
                        "Validation initial conditions are not matched."
                    )
                references[seed] = initial_hash
                result.update(
                    controller=f"{label}_{mode}",
                    checkpoint=str(checkpoint),
                    restore_validation=restore,
                )
                path = output / f"{label}_{mode}_seed{seed}.json"
                write_json(path, result)
                if not math.isclose(
                    result["sim_time_s"],
                    config.episode_duration_s,
                    rel_tol=0,
                    abs_tol=1e-6,
                ):
                    write_json(
                        output / "failed-horizon.json",
                        dict(
                            complete=False,
                            policy=label,
                            mode=mode,
                            seed=seed,
                            expected_horizon_s=config.episode_duration_s,
                            actual_horizon_s=result["sim_time_s"],
                            episode=str(path.resolve()),
                        ),
                    )
                    raise AssertionError(
                        "Held-out episode did not reach the requested horizon; raw failure saved."
                    )
                row = dict(
                    policy=label,
                    mode=mode,
                    seed=seed,
                    episode_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    episode=str(path.resolve()),
                    sim_time_s=result["sim_time_s"],
                    rollout_wall_s=result["rollout_wall_s"],
                    initial_conditions_sha256=initial_hash,
                    reward=sum(result["cumulative_reward"].values()),
                    capture_coverage=result["coverage"]["capture_coverage_fraction"],
                    ground_coverage=result["coverage"][
                        "ground_delivery_coverage_fraction"
                    ],
                    useful_acquisitions=result["team_summary"][
                        "unique_acquisition_count"
                    ],
                    useful_delivered_visits=result["team_summary"][
                        "unique_service_count"
                    ],
                    duplicate_attempts=result["team_summary"][
                        "duplicate_attempt_count"
                    ],
                    useful_revisits=result["useful_post_cooldown_revisits"]["count"],
                    actions=result["action_counts"],
                    cooperative_tasking=result.get("cooperative_tasking"),
                )
                rows.append(row)
                expected = len(checkpoints) * len(seeds) * len(modes)
                write_json(
                    output / "summary.json",
                    dict(
                        initial_conditions_matched=True,
                        expected_episodes=expected,
                        completed_episodes=len(rows),
                        complete=len(rows) == expected,
                        requested_policies=list(checkpoints),
                        requested_seeds=seeds,
                        rows=rows,
                    ),
                )
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in row.items()
                            if k not in {"actions", "episode_sha256"}
                        }
                    ),
                    flush=True,
                )
    return rows


def evaluation_cases(checkpoints, seeds, modes=(True, False)):
    return [(label, checkpoint, seed, deterministic)
            for label, checkpoint in checkpoints.items()
            for seed in seeds for deterministic in modes]


def _evaluation_worker_init():
    import torch
    torch.set_num_threads(1)


def parallel_compare(config, checkpoints, seeds, output, workers, modes):
    """Fresh spawned processes, one frozen mission per case; no Ray or PPO.

    Children retain their raw case evidence. Parent copies verified bytes into
    the existing flat episode schema, checks all matching hashes and writes a
    truthful incremental summary. Independent episode RNGs remain seed keyed.
    """
    cases = evaluation_cases(checkpoints, seeds, modes)
    rows, references, matched = [], {}, None
    def save(complete=False, error=None):
        write_json(output / "summary.json", dict(
            expected_episodes=len(cases), completed_episodes=len(rows),
            complete=complete, initial_conditions_matched=matched,
            requested_policies=list(checkpoints), requested_seeds=seeds,
            evaluation_workers=min(workers, len(cases)), rows=rows, error=error))
    save()
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, len(cases)),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_evaluation_worker_init,
        ) as executor:
            futures = {}
            for index, (label, checkpoint, seed, mode) in enumerate(cases):
                child = output / "cases" / f"case_{index:02d}"
                futures[executor.submit(compare, config, {label: checkpoint}, [seed], child, 1, (mode,))] = child
            for future in concurrent.futures.as_completed(futures):
                case_rows = future.result()
                if len(case_rows) != 1:
                    raise AssertionError("Frozen case returned an unexpected episode count")
                row = dict(case_rows[0])
                source = Path(row['episode'])
                data = source.read_bytes()
                if hashlib.sha256(data).hexdigest() != row['episode_sha256']:
                    raise AssertionError("Frozen case bytes changed before collection")
                if not math.isclose(row['sim_time_s'], config.episode_duration_s, abs_tol=1e-6, rel_tol=0):
                    raise AssertionError("Frozen case did not reach the mission horizon")
                seed = row['seed']
                if seed in references and references[seed] != row['initial_conditions_sha256']:
                    matched = False
                    raise AssertionError("Parallel validation initial conditions are not matched")
                references[seed] = row['initial_conditions_sha256']
                dest = output / source.name
                if dest.exists():
                    raise FileExistsError("Repeated frozen case would overwrite evidence")
                dest.write_bytes(data)
                row['episode'] = str(dest.resolve())
                rows.append(row)
                if len(rows) == len(cases): matched = True
                save(complete=len(rows) == len(cases))
        return rows
    except BaseException as exc:
        save(error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="label=checkpoint-directory",
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent frozen-mission processes, never PPO workers")
    args = parser.parse_args()
    checkpoints = dict(item.split("=", 1) for item in args.checkpoint)
    if len(checkpoints) != len(args.checkpoint) or any(
        not name.replace("_", "").isalnum() for name in checkpoints
    ):
        parser.error(
            "Checkpoint labels must be unique alphanumeric names with optional underscores."
        )
    compare(
        MultiAgentImagingConfig.from_json(args.config),
        checkpoints,
        args.seeds,
        args.output,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
