"""Thin entry point into the unchanged b205fc16 trainer with explicit HPO settings.

No alternate learner, environment or reward implementation. The cooperative
configuration owns gamma/lambda; the HPO transfer supplies optimizer settings.
"""

import argparse
import hashlib
import json
from pathlib import Path


def settings_record(path):
    record = json.loads(Path(path).read_text())
    if record["profile"] != "cooperative-v1-c18-optimizer-transfer-v1":
        raise ValueError("Unexpected HPO transfer profile")
    if record["source_commit"] != "b205fc1603b5ca887aaadcb3c34de2700d563c0c":
        raise ValueError("Wrong pinned trainer commit")
    if record["retained_cooperative_horizons"] is not True:
        raise ValueError("Do not silently replace the cooperative credit horizons")
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ppo-settings", type=Path, required=True)
    p.add_argument("--iterations", type=int, required=True)
    p.add_argument("--train-batch-size", type=int, required=True)
    p.add_argument("--num-env-runners", type=int, required=True)
    p.add_argument("--num-learners", type=int, required=True)
    p.add_argument("--ray-cpus", type=int, required=True)
    p.add_argument("--torch-threads", type=int, required=True)
    p.add_argument("--sample-timeout-s", type=float, required=True)
    p.add_argument("--wandb-mode", choices=["disabled", "online"], required=True)
    p.add_argument("--require-wandb", action="store_true")
    p.add_argument("--save-initial", action="store_true")
    p.add_argument("--capture-learner-batch", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--max-training-episodes", type=int)
    p.add_argument("--max-sampled-true-decisions", type=int)
    p.add_argument("--max-wall-seconds", type=float)
    a = p.parse_args(argv)
    record = settings_record(a.ppo_settings)
    from examples.multiagent_imaging.config import MultiAgentImagingConfig
    from examples.multiagent_imaging.optimization import PPOSettings
    from examples.multiagent_imaging.train import LauncherResources, train_run

    config = MultiAgentImagingConfig.from_json(a.config)
    if not config.is_cooperative or config.gae_lambda_time != "second":
        raise ValueError("HPO transfer is restricted to the cooperative time contract")
    if (a.num_env_runners, a.num_learners, a.ray_cpus, a.torch_threads) != (28, 0, 56, 1):
        raise ValueError("Preserve the approved canary topology")
    if a.iterations < 0 or (a.iterations == 0 and a.resume is None):
        raise ValueError("Zero-update restoration requires an existing checkpoint")
    if a.train_batch_size != 4200:
        raise ValueError("Preserve the reviewed learner batch minimum")
    settings = PPOSettings(**record["ppo_settings"])
    a.output.mkdir(parents=True, exist_ok=True)
    provenance = dict(
        profile=record["profile"],
        specification_sha256=hashlib.sha256(a.ppo_settings.read_bytes()).hexdigest(),
        source_commit=record["source_commit"],
        hpo_source=record["hpo_source"],
        discount_half_life_s=config.discount_half_life_s,
        gae_trace_half_life_s=config.gae_trace_half_life_s,
        hpo_discount_and_lambda_not_transferred=True,
    )
    with (a.output / "hpo-transfer.json").open("x") as stream:
        json.dump(provenance, stream, indent=2)
        stream.write("\n")
    resources = LauncherResources(
        num_env_runners=a.num_env_runners,
        num_learners=a.num_learners,
        ray_cpus=a.ray_cpus,
        torch_threads=a.torch_threads,
        sample_timeout_s=a.sample_timeout_s,
    )
    return train_run(
        config,
        iterations=a.iterations,
        train_batch_size=a.train_batch_size,
        output=a.output,
        resources=resources,
        resume=a.resume,
        ppo_settings=settings,
        wandb_mode=a.wandb_mode,
        require_wandb=a.require_wandb,
        save_initial=a.save_initial,
        capture_learner_batch=a.capture_learner_batch,
        max_training_episodes=a.max_training_episodes,
        max_sampled_true_decisions=a.max_sampled_true_decisions,
        max_wall_seconds=a.max_wall_seconds,
    )


if __name__ == "__main__":
    main()
