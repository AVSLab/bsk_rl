"""Episode and update evidence, independent of the reward/policy observations."""

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class CompletionAuditCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, env_runner, **kwargs):
        env_runner.completion_resource_history = []
        self._record_resources(env_runner)

    def on_episode_step(self, *, env_runner, **kwargs):
        self._record_resources(env_runner)

    @staticmethod
    def _record_resources(runner):
        """Event-boundary resource history; do not add peer data to policy inputs."""
        env = runner.env.unwrapped.par_env
        history = getattr(runner, "completion_resource_history", [])
        for sensor in env.sensing_satellites:
            history.append(
                dict(
                    sensor=sensor.name,
                    time_s=float(env.simulator.sim_time),
                    alive=bool(sensor.is_alive()),
                    battery_fraction=float(sensor.dynamics.battery_charge_fraction),
                    storage_fraction=float(sensor.dynamics.storage_level_fraction),
                    wheel_speed_fraction=np.asarray(
                        sensor.dynamics.wheel_speeds_fraction
                    ).tolist(),
                )
            )
        runner.completion_resource_history = history

    def on_episode_end(self, *, env_runner, episode, **kwargs):
        env = env_runner.env.unwrapped.par_env
        resources, products = {}, {}
        for sensor in env.sensing_satellites:
            values = np.asarray(
                [
                    sensor.dynamics.battery_charge_fraction,
                    sensor.dynamics.storage_level_fraction,
                    *sensor.dynamics.wheel_speeds_fraction,
                ]
            )
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite spacecraft resources.")
            resources[sensor.name] = values.tolist()
            products[sensor.name] = len(sensor.data_store.products)
            for product in sensor.data_store.products:
                if (
                    product.storage_owner != sensor.name
                    or product.source_sensor != sensor.name
                ):
                    raise AssertionError(
                        "Completion metadata transferred physical product ownership."
                    )
        probes = [sensor.get_obs().tolist() for sensor in env.sensing_satellites]
        summary = dict(
            simulated_seconds=float(env.simulator.sim_time),
            env_steps=len(episode),
            agent_steps=episode.agent_steps(),
            policy_decisions=dict(env.decision_counts),
            resources=resources,
            resource_history=getattr(env_runner, "completion_resource_history", []),
            products=products,
            constellation_reward=float(episode.get_return()),
            team_summary=dict(env.rewarder.team_summary),
            coordination=env.coordination_metrics(),
            packets=list(env.communicator.delivery_history),
            transmissions=list(env.communicator.transmission_history),
            probe_observations=probes,
            next_episode_index=env._episode_seed_index,
            worker_index=env_runner.worker_index,
        )
        if not hasattr(env_runner, "completion_audits"):
            env_runner.completion_audits = []
        env_runner.completion_audits.append(summary)


def take_audits(runner):
    audits = getattr(runner, "completion_audits", [])
    runner.completion_audits = []
    return audits


def parameter_change(before, after):
    squared, count, largest = 0.0, 0, 0.0
    for key in before:
        delta = np.asarray(after[key]) - np.asarray(before[key])
        if not np.isfinite(delta).all():
            raise FloatingPointError("Nonfinite policy parameter update.")
        squared += float(np.square(delta).sum())
        largest = max(largest, float(np.max(np.abs(delta))))
        count += int(np.count_nonzero(delta))
    if not count:
        raise AssertionError("PPO returned without changing any policy weights.")
    return dict(
        l2=float(np.sqrt(squared)), max_absolute=largest, changed_elements=count
    )
