"""Independently check one retained real batch, then fit only its private critic.

No environment, sampler, actor update or Ray initialization is performed here.
The uncapped MSE fitting experiment diagnoses representational learnability; it
does not modify PPO's AMOS value-error cap or any campaign checkpoint.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from bsk_rl import NO_ACTION

from examples.multiagent_imaging.checkpoints import load_policy
from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.readiness import write_json


def independently_recompute(directory, config):
    directory = Path(directory)
    raw = json.load(gzip.open(directory / "raw-episodes.json.gz", "rt"))
    boot = np.load(directory / "bootstrap-batch.npz")
    batch = np.load(directory / "learner-batch.npz")
    values = (
        np.load(directory / "bootstrap-values.npz")["vf_preds"]
        .reshape(-1)
        .astype(float)
    )
    gamma = config.discount_per_s
    lengths, condensed, durations, observations, actions, traces = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for ep in raw:
        starts = [i for i, a in enumerate(ep["actions"]) if a != NO_ACTION]
        lengths.append(len(starts) + 1)
        # Save the actual action rewards before the synthetic bootstrap row.
        elapsed_episode = np.cumsum([i["d_ts"] for i in ep["infos"]])
        for j, start in enumerate(starts):
            stop = starts[j + 1] if j + 1 < len(starts) else len(ep["actions"])
            elapsed, reward = 0.0, 0.0
            for i in range(start, stop):
                info = ep["infos"][i + 1]
                dt, rate = (
                    float(info["d_ts"]),
                    float(info.get("communication_cost_rate", 0)),
                )
                impulse = ep["rewards"][i] + rate * dt
                integral = dt if gamma == 1 else (1 - gamma**dt) / -np.log(gamma)
                reward += gamma**elapsed * (gamma**dt * impulse - rate * integral)
                # Sensor metrics expose the physical event that owned the reward.
                before, after = (
                    ep["infos"][i].get("sensor_metrics", {}),
                    info.get("sensor_metrics", {}),
                )
                for kind in ("acquisition_credit", "delivery_credit"):
                    delta = after.get(kind, 0) - before.get(kind, 0)
                    if delta > 0:
                        traces.append(
                            dict(
                                kind=kind,
                                agent=ep["agent_id"],
                                episode_id=ep["id"],
                                multi_agent_episode_id=ep.get("multi_agent_episode_id"),
                                event_index=i,
                                interval_start_s=float(elapsed_episode[i]),
                                event_time_s=float(elapsed_episode[i + 1]),
                                credit_delta=delta,
                                raw_reward=ep["rewards"][i],
                                decision_index=len(actions),
                                seconds_from_action_start=elapsed + dt,
                                endpoint_discount=gamma ** (elapsed + dt),
                            )
                        )
                elapsed += dt
            condensed.append(reward)
            durations.append(elapsed)
            observations.append(ep["observations"][start])
            actions.append(ep["actions"][start])
    assert len(values) == sum(lengths), "Bootstrap episode segmentation mismatch"
    np.testing.assert_array_equal(
        np.asarray(observations, dtype=batch["obs"].dtype), batch["obs"]
    )
    np.testing.assert_array_equal(actions, batch["actions"])
    np.testing.assert_allclose(condensed, batch["rewards"], rtol=2e-5, atol=1e-5)
    # This recurrence is independent of bsk_rl's value-target helper. It retains
    # Ray's terminal-zero and truncation-reset semantics for the bootstrap rows.
    terminal = boot["terminateds"].reshape(-1)
    truncated = boot["truncateds"].reshape(-1)
    rewards = boot["rewards"].reshape(-1)
    dt = np.r_[boot["durations"].reshape(-1)[1:], boot["durations"].reshape(-1)[-1]]
    v = values * (1 - terminal)
    following = np.r_[v[1:], 0.0]
    trace = (
        config.effective_gae_lambda
        if config.gae_lambda_time == "step"
        else config.effective_gae_lambda**dt
    )
    result = np.empty(len(values))
    carry = 0.0
    for i in range(len(values) - 1, -1, -1):
        lam = trace if np.ndim(trace) == 0 else trace[i]
        carry = rewards[i] + gamma ** dt[i] * (
            (1 - lam) * following[i] + (1 - terminal[i]) * lam * carry
        )
        result[i] = carry
        if truncated[i]:
            carry = v[i]
    keep = np.ones(len(values), dtype=bool)
    keep[np.cumsum(lengths) - 1] = False
    np.testing.assert_allclose(durations, dt[keep], rtol=1e-6, atol=1e-5)
    targets = result[keep].astype(np.float32)
    np.testing.assert_allclose(targets, batch["value_targets"], rtol=2e-5, atol=1e-5)
    advantages = targets - values[keep]
    advantages = (advantages - advantages.mean()) / max(1e-4, advantages.std())
    np.testing.assert_allclose(advantages, batch["advantages"], rtol=2e-4, atol=1e-4)
    for record in traces:
        index = record["decision_index"]
        record.update(
            condensed_reward=float(batch["rewards"][index]),
            value_target=float(targets[index]),
            advantage=float(batch["advantages"][index]),
        )
    return batch, dict(
        passed=True,
        raw_agent_episodes=len(raw),
        true_decisions=len(actions),
        bootstrap_rows=len(values) - len(actions),
        target_max_abs_error=float(np.max(np.abs(targets - batch["value_targets"]))),
        forced_no_action_count=int(np.sum(batch["actions"] == NO_ACTION)),
        padded_rows=int(np.sum(~batch["loss_mask"].astype(bool)))
        if "loss_mask" in batch
        else 0,
        physical_reward_traces=traces,
        hashes={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.iterdir()
            if p.is_file()
        },
    )


def fit_critic(batch, checkpoint, config, *, steps=500, max_wall_seconds=600):
    torch.set_num_threads(1)
    torch.manual_seed(3105)
    module, restore = load_policy(checkpoint, config)
    module.train()
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(name.startswith("vf."))
    actor = {
        n: p.detach().clone()
        for n, p in module.named_parameters()
        if not n.startswith("vf.")
    }
    opt = torch.optim.Adam(module.vf.parameters(), lr=1e-3)
    obs = torch.tensor(batch["obs"], dtype=torch.float32)
    target = torch.tensor(batch["value_targets"], dtype=torch.float32)
    history = []
    start = time.monotonic()

    def measure(step):
        module.eval()
        with torch.no_grad():
            prediction = torch.cat(
                [module.vf({Columns.OBS: x}).reshape(-1) for x in obs.split(256)]
            )
            residual = prediction - target
            variance = float(torch.var(target, correction=0))
            history.append(
                dict(
                    step=step,
                    mse=float(residual.square().mean()),
                    explained_variance=1
                    - float(torch.var(residual, correction=0)) / variance
                    if variance > 1e-12
                    else None,
                    amos_squared_error_cap_fraction=float(
                        (residual.square() > 10).float().mean()
                    ),
                )
            )
        module.train()

    measure(0)
    gradient_norms = []
    for step in range(1, steps + 1):
        index = torch.randperm(len(obs))[: min(256, len(obs))]
        prediction = module.vf({Columns.OBS: obs[index]}).reshape(-1)
        loss = (prediction - target[index]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite fixed-batch critic loss")
        opt.zero_grad()
        loss.backward()
        gradients = [p.grad for p in module.vf.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError("Missing/nonfinite fixed-batch critic gradients")
        gradient_norms.append(
            float(torch.sqrt(sum(g.square().sum() for g in gradients)))
        )
        opt.step()
        if any(not torch.isfinite(p).all() for p in module.vf.parameters()):
            raise FloatingPointError("Nonfinite fixed-batch critic parameters")
        if step % 50 == 0 or step == steps:
            measure(step)
        if time.monotonic() - start > max_wall_seconds:
            if history[-1]["step"] != step:
                measure(step)
            break
    assert all(
        torch.equal(actor[n], p.detach())
        for n, p in module.named_parameters()
        if n in actor
    ), "Actor changed during critic-only diagnostic"
    return dict(
        checkpoint=str(checkpoint),
        restore=restore,
        optimizer="Adam lr=0.001, uncapped MSE, critic only",
        steps=step,
        wall_time_s=time.monotonic() - start,
        history=history,
        gradient_l2_min=min(gradient_norms),
        gradient_l2_max=max(gradient_norms),
        actor_unchanged=True,
        can_fit=history[-1]["mse"] < history[0]["mse"] * 0.5
        and history[-1]["explained_variance"] is not None
        and history[-1]["explained_variance"] > 0.5,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--max-wall-seconds", type=float, default=600)
    parser.add_argument("--require-fit", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or not np.isfinite(args.max_wall_seconds) or not 0 < args.max_wall_seconds <= 600:
        parser.error("Require positive steps and a finite wall cap in (0, 600].")
    inputs = args.input_root or args.output
    args.output.mkdir(parents=True, exist_ok=True)
    config = MultiAgentImagingConfig.from_json(args.config)
    batch, audit = independently_recompute(inputs / "learner_batch", config)
    updates = json.loads((inputs / "updates.json").read_text())
    episodes = {e["episode_id"]: e for e in updates[0]["episodes"]}
    for trace in audit["physical_reward_traces"]:
        episode = episodes.get(trace["multi_agent_episode_id"], {})
        acquisition = trace["kind"] == "acquisition_credit"
        products = (
            episode.get("capture_products", [])
            if acquisition
            else [e["product"] for e in episode.get("delivered_products", [])]
        )
        time_key = "completion_time" if acquisition else "delivery_time"
        trace["matching_physical_products"] = [
            p
            for p in products
            if p["source_sensor"] == trace["agent"]
            and p.get(time_key) is not None
            and trace["interval_start_s"] - 1e-6
            <= p[time_key]
            <= trace["event_time_s"] + 1e-6
        ]
        assert trace[
            "matching_physical_products"
        ], "Credited event has no corresponding physical product"
        trace["weighted_reward_component"] = trace["credit_delta"] * (
            1 - config.alpha if acquisition else config.alpha
        )
    audit["acquisition_trace_present"] = any(
        t["kind"] == "acquisition_credit" for t in audit["physical_reward_traces"]
    )
    audit["physical_delivery_trace_present"] = any(
        t["kind"] == "delivery_credit" for t in audit["physical_reward_traces"]
    )
    write_json(args.output / "fixed-batch-target-audit.json", audit)
    ev = updates[-1]["learner_metrics"].get("vf_explained_var")
    if args.require_fit or ev is None or ev < 0.1:
        fit = fit_critic(batch, args.checkpoint, config, steps=args.steps, max_wall_seconds=args.max_wall_seconds)
        write_json(args.output / "fixed-batch-critic-fit.json", fit)
        if args.require_fit and not fit["can_fit"]:
            raise SystemExit("Fixed-batch fit gate failed: require MSE < half and EV > 0.5; evidence preserved")
    else:
        write_json(
            args.output / "fixed-batch-critic-fit.json",
            dict(
                not_tested=True,
                reason="Final sampled critic explained variance >= 0.1",
                sampled_explained_variance=ev,
            ),
        )


if __name__ == "__main__":
    main()
