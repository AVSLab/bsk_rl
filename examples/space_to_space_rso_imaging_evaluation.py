"""Evaluate an RSO imaging policy and save a replayable mission and plots.

Run ``python examples/space_to_space_rso_imaging_evaluation.py --help``.
The default priority baseline needs no checkpoint or learning framework.
Matplotlib is optional with ``--no-plots``. The
:doc:`space-to-space imaging notebook <space_to_space_rso_imaging>` explains
the environment, reward callbacks, replay, and policy adapters.

The priority, nearest-target, and random baselines first manage battery charge,
wheel speed, and stored images. They then choose an accessible, illuminated
candidate. A saved catalog fixes target states, while a mission manifest also
fixes the imager, world, and action/reward settings. Output CSV files distinguish
attempts, physically stored acquisitions, and fully completed deliveries.

The RLlib adapter loads one PyTorch RLModule for inference. Custom policies use
``module:function`` factories returning ``policy(observation, context)``.
The context includes stable candidate IDs, physical access, resources, and action
indices. Empty slots remain empty. See the notebook for runnable commands.
"""

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from bsk_rl import act, obs, scene

if __package__:
    from .space_to_space_rso_imaging_demo import (
        build_environment,
        make_catalog,
        mission_manifest,
        replay_environment,
    )
else:
    from space_to_space_rso_imaging_demo import (
        build_environment,
        make_catalog,
        mission_manifest,
        replay_environment,
    )


def checkpoint_fingerprint(path):
    """Hash an exact checkpoint file or directory without loading its contents."""
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    files = (
        [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    )
    if not files:
        raise ValueError("Checkpoint contains no files.")
    rows = [
        hashlib.sha256(file.read_bytes()).hexdigest()
        + "  "
        + (file.name if root.is_file() else file.relative_to(root).as_posix())
        + "\n"
        for file in files
    ]
    return dict(
        path=str(root),
        tree_sha256=hashlib.sha256("".join(rows).encode()).hexdigest(),
        hash_algorithm="SHA256 of sorted file-SHA256, two spaces, relative path, newline",
    )


def decision_context(env, imager):
    """Expose IDs and physical diagnostics without changing observation columns."""
    target_observation = next(
        spec
        for spec in imager.observation_builder.observation_spec
        if isinstance(spec, obs.RSOTargetProperties)
    )
    candidates = []
    rows = list(target_observation.get_obs().values())
    for slot, target_id in enumerate(target_observation.candidate_ids()):
        if target_id is None:
            candidates.append(None)
            continue
        target = env.scenario.targets_by_id[target_id]
        dynamics = target.target_spacecraft.dynamics
        candidates.append(
            dict(
                target_id=target_id,
                priority=target.priority,
                distance_m=float(
                    np.linalg.norm(
                        np.asarray(dynamics.r_BN_N) - np.asarray(imager.dynamics.r_BN_N)
                    )
                ),
                illumination=float(rows[slot]["target_illumination_factor"]),
                has_access=bool(
                    imager.dynamics.rso_access_messages[target_id].read().hasAccess
                ),
            )
        )
    actions = []
    for action in imager.action_builder.action_spec:
        for slot in range(action.n_actions):
            actions.append(
                dict(
                    type=type(action).__name__,
                    name=action.name,
                    candidate_slot=slot if isinstance(action, act.ImageRSO) else None,
                )
            )
    return dict(
        time_s=float(env.simulator.sim_time),
        imager_name=imager.name,
        candidates=candidates,
        actions=actions,
        storage_bits=float(imager.dynamics.storage_level),
        battery_fraction=float(imager.dynamics.battery_charge_fraction),
        wheel_fraction=float(np.max(np.abs(imager.dynamics.wheel_speeds_fraction))),
        quality_threshold=env.rewarder.quality_threshold,
    )


def baseline_policy(kind, rng):
    """Use resource guards, downlink stored products, then select an image slot."""

    def policy(observation, context):
        modes = {
            action["type"]: index for index, action in enumerate(context["actions"])
        }
        if context["battery_fraction"] < 0.3:
            return modes["Charge"]
        if context["wheel_fraction"] > 0.8:
            return modes["Desat"]
        if context["storage_bits"] > 0:
            return modes["Downlink"]
        eligible = [
            index
            for index, action in enumerate(context["actions"])
            if action["candidate_slot"] is not None
            and (candidate := context["candidates"][action["candidate_slot"]])
            is not None
            and candidate["has_access"]
            and candidate["illumination"] >= context["quality_threshold"]
        ]
        if not eligible:
            return modes["Charge"]
        if kind == "random":
            return int(rng.choice(eligible))

        def ranking(index):
            candidate = context["candidates"][
                context["actions"][index]["candidate_slot"]
            ]
            return (
                -candidate["priority"]
                if kind == "priority"
                else candidate["distance_m"],
                index,
            )

        return min(eligible, key=ranking)

    return policy


def load_policy(
    kind, env, imager, rng, checkpoint=None, factory=None, stochastic=False
):
    """Load a stateless discrete policy, without creating workers or an algorithm."""
    if kind in ("priority", "nearest", "random"):
        if checkpoint or factory:
            raise ValueError("Checkpoint/factory options require rllib/custom policy.")
        return baseline_policy(kind, rng)
    if kind == "custom":
        if not factory or ":" not in factory:
            raise ValueError(
                "Custom policies require --policy-factory module:function or file.py:function."
            )
        module_name, function_name = factory.rsplit(":", 1)
        if module_name.endswith(".py"):
            spec = importlib.util.spec_from_file_location(
                "rso_user_policy", Path(module_name).resolve()
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)
        policy = getattr(module, function_name)(
            env=env,
            rng=rng,
            checkpoint=checkpoint,
            stochastic=stochastic,
        )
        if not callable(policy):
            raise TypeError(
                "Policy factory must return a callable policy(observation, context)."
            )

        def wrapped(observation, context):
            return policy(observation, context)

        source = Path(module.__file__).resolve()
        wrapped.provenance = dict(
            adapter_source=str(source),
            adapter_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )
        if hasattr(policy, "close"):
            wrapped.close = policy.close
        return wrapped
    if kind != "rllib" or checkpoint is None:
        raise ValueError(
            "RLlib policies require an exact RLModule directory via --checkpoint."
        )
    import torch
    from ray.rllib.core.columns import Columns
    from ray.rllib.core.rl_module.rl_module import RLModule

    module = RLModule.from_checkpoint(Path(checkpoint).expanduser())
    if not isinstance(module, torch.nn.Module):
        raise TypeError(
            "This adapter supports PyTorch RLModules only; use a custom adapter otherwise."
        )
    if module.is_stateful():
        raise ValueError("Recurrent modules require a custom policy adapter.")
    # Ray 2.35 stores spaces on config; newer modules expose them directly.
    spaces = module if hasattr(module, "observation_space") else module.config
    if (
        spaces.observation_space.shape != imager.observation_space.shape
        or spaces.action_space.n != imager.action_space.n
    ):
        raise ValueError(
            "Checkpoint observation/action spaces do not match the environment profile."
        )
    module.eval()  # Disable training-time dropout as well as gradient recording.
    device = next(module.parameters()).device

    def policy(observation, context):
        with torch.inference_mode():
            vector = torch.as_tensor(
                np.asarray(observation, dtype=np.float32), device=device
            ).unsqueeze(0)
            output = module.forward_inference({Columns.OBS: vector})
            logits = output[Columns.ACTION_DIST_INPUTS].detach().cpu().numpy()
        if logits.shape != (1, imager.action_space.n) or not np.isfinite(logits).all():
            raise ValueError(
                "Expected finite discrete logits with shape (1, action_count)."
            )
        logits = logits[0]
        if not stochastic:
            return int(np.argmax(logits))
        probabilities = np.exp(logits - logits.max())
        return int(rng.choice(len(logits), p=probabilities / probabilities.sum()))

    policy.provenance = {
        name: importlib.metadata.version(name) for name in ("ray", "torch")
    }
    return policy


def validate_action(action, context):
    """Reject malformed, out-of-range and padded actions without substitution."""
    if isinstance(action, (bool, np.bool_)) or not isinstance(
        action, (int, np.integer)
    ):
        raise TypeError("Policy must return an integer discrete action.")
    if not 0 <= action < len(context["actions"]):
        raise ValueError("Policy action is outside the environment action space.")
    slot = context["actions"][action]["candidate_slot"]
    if slot is not None and context["candidates"][slot] is None:
        raise ValueError(
            "Policy selected a padded image slot; no real target was substituted."
        )
    return int(action)


def write_csv(path, rows, fields):
    """Keep empty event tables readable with the same explicit schema."""
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(directory, states, decisions):
    """Plot physical products, credited rewards, resources, and action intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.array([row["time_s"] for row in states]) / 60
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8), constrained_layout=True)
    attempt_times = [
        row["start_s"] / 60 for row in decisions if row["target_id"] is not None
    ]
    axes[0].step(
        [times[0], *attempt_times, times[-1]],
        [0, *range(1, len(attempt_times) + 1), len(attempt_times)],
        where="post",
        label="Imaging attempts",
    )
    for key, label in (
        ("captures", "Acquired images"),
        ("deliveries", "Completed deliveries"),
    ):
        axes[0].step(times, [row[key] for row in states], where="post", label=label)
    axes[0].set_ylabel("Cumulative count")
    for key, label in (
        ("acquisition_reward", "Acquisition"),
        ("delivery_reward", "Delivery"),
        ("total_reward", "Total (including penalties)"),
    ):
        axes[1].step(times, [row[key] for row in states], where="post", label=label)
    axes[1].set_ylabel("Cumulative reward")
    axes[2].step(
        times,
        [row["storage_bits"] / 1e6 for row in states],
        where="post",
        label="Onboard images",
    )
    axes[2].set_ylabel("Storage [Mbit]")
    resource_axis = axes[2].twinx()
    resource_axis.plot(
        times,
        [row["battery_fraction"] for row in states],
        color="tab:orange",
        label="Battery",
    )
    resource_axis.plot(
        times,
        [row["wheel_fraction"] for row in states],
        color="tab:green",
        label="Max wheel fraction",
    )
    resource_axis.set_ylabel("Resource fraction")
    resource_axis.legend(loc="upper right")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left")
    axes[-1].set_xlabel("Simulation time [min]")
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"performance.{suffix}", dpi=160)
    plt.close(fig)
    modes = list(dict.fromkeys(row["action_type"] for row in decisions))
    fig, axis = plt.subplots(figsize=(9, 3), constrained_layout=True)
    for row in decisions:
        axis.broken_barh(
            [(row["start_s"] / 60, (row["end_s"] - row["start_s"]) / 60)],
            (modes.index(row["action_type"]) - 0.35, 0.7),
        )
        if row["target_id"] is not None:
            axis.text(
                row["start_s"] / 60,
                modes.index(row["action_type"]),
                row["target_id"],
                fontsize=8,
                va="center",
            )
    axis.set_yticks(range(len(modes)), modes)
    axis.set_xlabel("Simulation time [min]")
    axis.set_title("Commanded actions and selected target IDs")
    axis.grid(axis="x", alpha=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"actions.{suffix}", dpi=160)
    plt.close(fig)


def evaluate(
    output,
    *,
    policy="priority",
    seed=0,
    horizon=None,
    max_steps=200,
    catalog_path=None,
    manifest_path=None,
    profile=None,
    checkpoint=None,
    factory=None,
    stochastic=False,
    plots=True,
):
    """Run one bounded episode; never overwrite an existing output directory.

    Times/resources are sampled at decision boundaries. Capture timestamps come
    from the physical gate; delivery timestamps are boundary recognition times.
    Catalog replay alone fixes targets; mission replay fixes scanner/world too.
    """
    if horizon is not None and (not np.isfinite(horizon) or horizon <= 0):
        raise ValueError("Horizon must be finite and positive.")
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer.")
    if manifest_path and (catalog_path or profile):
        raise ValueError("A mission manifest supplies its own catalog and profile.")
    if profile not in (None, "compact", "amos"):
        raise ValueError("Unknown environment profile.")
    if stochastic and policy not in ("rllib", "custom", "random"):
        raise ValueError("Stochastic mode requires a stochastic policy adapter.")
    if plots:
        import matplotlib  # noqa: F401 -- fail before simulation if extra is absent
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = None
    selected_policy = None
    states, decisions, observations, capture_rows, delivery_rows = [], [], [], [], []
    summary = dict(status="error", policy=policy, seed=seed)
    metadata = dict(
        policy=policy,
        inference_mode="stochastic"
        if stochastic or policy == "random"
        else "deterministic",
        factory=factory,
        evaluator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment_source_sha256=hashlib.sha256(
            Path(sys.modules[build_environment.__module__].__file__).read_bytes()
        ).hexdigest(),
    )
    try:
        if checkpoint:
            metadata["checkpoint"] = checkpoint_fingerprint(checkpoint)
        if manifest_path:
            saved = json.loads(Path(manifest_path).read_text())
            env = replay_environment(saved)
            limit = horizon if horizon is not None else saved["time_limit_s"]
            env.time_limit_generator = lambda: limit
        else:
            catalog = (
                scene.RSOTargetCatalog.load(catalog_path)
                if catalog_path
                else make_catalog(seed)
            )
            env = build_environment(
                catalog,
                seed=seed,
                amos_profile=profile == "amos",
                time_limit=horizon if horizon is not None else 900,
            )
        vectors, _ = env.reset(seed=seed)
        if len(env.scenario.imagers) != 1:
            raise ValueError("This evaluation example supports one imager per episode.")
        imager = env.scenario.imagers[0]
        manifest = mission_manifest(env, checkpoint=metadata)
        manifest["policy_rng"] = dict(
            seed=seed, generator="numpy.PCG64", independent_of_mission_rng=True
        )
        env.scenario.catalog.save(output / "catalog.json")
        # Keep provenance even if adapter loading fails.
        (output / "policy.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (output / "mission.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        np.save(output / "initial_observation.npy", vectors[imager.name])
        rng = np.random.default_rng(seed)
        selected_policy = load_policy(
            policy, env, imager, rng, checkpoint, factory, stochastic
        )
        metadata.update(getattr(selected_policy, "provenance", {}))
        (output / "mission.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        totals = dict(acquisition_reward=0.0, delivery_reward=0.0, total_reward=0.0)
        attempts = 0

        def sample():
            context = decision_context(env, imager)
            states.append(
                dict(
                    time_s=context["time_s"],
                    storage_bits=context["storage_bits"],
                    battery_fraction=context["battery_fraction"],
                    wheel_fraction=context["wheel_fraction"],
                    attempts=attempts,
                    captures=len(imager.data_store.data.captures),
                    deliveries=len(imager.data_store.data.deliveries),
                    **totals,
                )
            )
            return context

        context = sample()
        for _ in range(max_steps):
            vector = np.asarray(vectors[imager.name])
            try:
                action = validate_action(
                    selected_policy(vector.copy(), context), context
                )
            except Exception as error:
                (output / "failed_decision.json").write_text(
                    json.dumps(dict(context=context, error=str(error)), indent=2) + "\n"
                )
                raise
            descriptor = context["actions"][action]
            slot = descriptor["candidate_slot"]
            target_id = (
                None if slot is None else context["candidates"][slot]["target_id"]
            )
            before_captures = set(imager.data_store.data.captures)
            before_deliveries = set(imager.data_store.data.deliveries)
            stage_priorities = {
                key: target.priority
                for key, target in env.scenario.targets_by_id.items()
            }
            commanded = {
                sat.name: 0
                for sat in env.satellites
                if sat.name != imager.name and sat.requires_retasking
            }
            commanded[imager.name] = action
            vectors, rewards, terminated, truncated, _ = env.step(commanded)
            if env.simulator.sim_time <= context["time_s"]:
                raise RuntimeError("Simulation made no time progress.")
            attempts += slot is not None
            components = env.rewarder.last_reward_components
            totals["acquisition_reward"] += components["acquisition"].get(
                imager.name, 0.0
            )
            totals["delivery_reward"] += components["delivery"].get(imager.name, 0.0)
            totals["total_reward"] += rewards.get(imager.name, 0.0)
            row = dict(
                start_s=context["time_s"],
                end_s=float(env.simulator.sim_time),
                action=action,
                action_type=descriptor["type"],
                target_id=target_id,
                reward=rewards.get(imager.name, 0.0),
                acquisition_reward=components["acquisition"].get(imager.name, 0.0),
                delivery_reward=components["delivery"].get(imager.name, 0.0),
                candidate_ids=[
                    None if item is None else item["target_id"]
                    for item in context["candidates"]
                ],
            )
            decisions.append(row)
            observations.append(vector.copy())
            for records, previous, destination in (
                (imager.data_store.data.captures, before_captures, capture_rows),
                (imager.data_store.data.deliveries, before_deliveries, delivery_rows),
            ):
                for key in sorted(set(records) - previous):
                    destination.append(
                        dict(
                            **asdict(records[key]),
                            stage_priority=stage_priorities[records[key].target_id],
                        )
                    )
            context = sample()
            if terminated.get(imager.name) or truncated.get(imager.name):
                summary["status"] = (
                    "terminated" if terminated.get(imager.name) else "time_limit"
                )
                break
        else:
            summary["status"] = "max_steps"
        summary.update(
            states[-1],
            steps=len(decisions),
            pending_products=len(imager.data_store.products),
            remaining_bits=float(sum(imager.data_store.remaining_bits.values())),
        )
        (output / "policy.json").write_text(json.dumps(metadata, indent=2) + "\n")
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_csv(
            output / "states.csv",
            states,
            [
                "time_s",
                "storage_bits",
                "battery_fraction",
                "wheel_fraction",
                "attempts",
                "captures",
                "deliveries",
                "acquisition_reward",
                "delivery_reward",
                "total_reward",
            ],
        )
        csv_decisions = [
            dict(row, candidate_ids=json.dumps(row["candidate_ids"]))
            for row in decisions
        ]
        write_csv(
            output / "decisions.csv",
            csv_decisions,
            [
                "start_s",
                "end_s",
                "action",
                "action_type",
                "target_id",
                "reward",
                "acquisition_reward",
                "delivery_reward",
                "candidate_ids",
            ],
        )
        event_fields = [
            "record_id",
            "target_id",
            "source_imager",
            "capture_time",
            "size_bits",
            "quality",
            "hold_valid_time_s",
            "delivery_time",
            "stage_priority",
        ]
        write_csv(output / "captures.csv", capture_rows, event_fields)
        write_csv(output / "deliveries.csv", delivery_rows, event_fields)
        if observations:
            np.save(output / "decision_observations.npy", np.asarray(observations))
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        try:
            if selected_policy is not None and hasattr(selected_policy, "close"):
                selected_policy.close()
        finally:
            if env is not None:
                env.close()
    if plots:
        try:
            plot_results(output, states, decisions)
        except Exception as error:
            summary.update(
                status="plot_error", error=f"{type(error).__name__}: {error}"
            )
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            raise
    return summary


def main():
    """Command-line entry point for a single reproducible evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; existing directories are rejected.",
    )
    parser.add_argument(
        "--policy",
        choices=["priority", "nearest", "random", "rllib", "custom"],
        default="priority",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--horizon",
        type=float,
        help="Simulation seconds; defaults to 900 or the replayed manifest horizon.",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--catalog",
        dest="catalog_path",
        type=Path,
        help="Fix target conditions; scanner/world still use example settings.",
    )
    inputs.add_argument(
        "--mission",
        dest="manifest_path",
        type=Path,
        help="Replay targets, scanner, world and configuration.",
    )
    parser.add_argument(
        "--profile",
        choices=["compact", "amos"],
        help="New mission observation/action profile; AMOS uses 124 features/13 actions.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Exact RLModule directory or checkpoint passed to a custom adapter.",
    )
    parser.add_argument(
        "--policy-factory",
        dest="factory",
        help="Custom adapter factory as module:function or file.py:function.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample RLlib logits using a separate seeded policy RNG.",
    )
    parser.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        help="Write data only; no Matplotlib dependency.",
    )
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
