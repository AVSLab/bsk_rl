"""``bsk_rl.utils.rllib`` is a collection of utilities for working with RLlib."""

from typing import Any

from ray.rllib.algorithms.callbacks import DefaultCallbacks


def unpack_config(env):
    """Create a wrapped version of an env class that unpacks env_config from Ray into kwargs.

    Necessary when setting

    .. code-block:: python

        config.environment(
            env=unpack_config(SatelliteTasking),
            env_config=env_args
        )

    which generates environments that look like

    .. code-block:: python

        SatelliteTasking(**env_args)

    since RLlib expects the environment to take a dictionary called ``kwargs`` instead
    of the actual arguments.

    """

    class UnpackedEnv(env):
        def __init__(self, env_config):
            super().__init__(**env_config)

    UnpackedEnv.__name__ = f"{env.__name__}_Unpacked"

    return UnpackedEnv


class EpisodeDataCallbacks(DefaultCallbacks):
    def __init__(self, *args, **kwargs):
        """Log information at the end of each episode.

        Make a subclass of ``EpisodeDataCallbacks`` and override ``pull_env_metrics`` to
        log environment-specific information at the end of each episode.
        """
        super().__init__(*args, **kwargs)

    def pull_env_metrics(self, env) -> dict[str, float]:
        """Log environment metrics at the end of each episode.

        This function is called whenever ``env`` is terminated or truncated. It should
        return a dictionary of metrics to log.

        Args:
            env: An environment that has completed.
        """
        return {}

    def on_episode_end(
        self,
        env=None,
        metrics_logger=None,
        **kwargs,
    ) -> None:
        """Call pull_env_metrics and log the results.

        :meta private:
        """
        if "base_env" in kwargs:  # Old RLlib stack
            env = kwargs["base_env"].vector_env.envs[0]  # noqa: F841
            env_data = self.pull_env_metrics(env)
            for k, v in env_data.items():
                kwargs["episode"].custom_metrics[k] = v
        else:  # New RLlib stack
            env = env.envs[0]
            env_data = self.pull_env_metrics(env)
            for k, v in env_data.items():
                metrics_logger.log_value(k, v)

    def on_train_result(self, *, algorithm, result, **kwargs):
        """Log frames per second.

        :meta private:
        """
        if "num_env_steps_sampled_this_iter" in result:
            result["fps"] = (
                result["num_env_steps_sampled_this_iter"] / result["time_this_iter_s"]
            )


__doc_title__ = "RLlib Utilities"
__all__ = ["unpack_config", "EpisodeDataCallbacks"]
