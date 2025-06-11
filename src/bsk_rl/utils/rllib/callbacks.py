"""``bsk_rl.utils.rllib.callbacks`` is a collection of utilities for logging additional data in RLlib."""

from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
from deprecated import deprecated
from gymnasium import Wrapper
from pettingzoo.utils import BaseParallelWrapper
from ray.rllib.algorithms.callbacks import DefaultCallbacks

if TYPE_CHECKING:
    from gymnasium import Env
    from pettingzoo import ParallelEnv

    from bsk_rl import GeneralSatelliteTasking
    from bsk_rl.sats import Satellite


@deprecated(
    reason="Only for use with old RLlib stack. Use WrappedEpisodeDataCallbacks instead. This class is not maintained."
)
class EpisodeDataCallbacks(DefaultCallbacks):
    """Deprecated method of logging information at the end of each episode.

    :meta private:
    """

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
            env = env.envs[-1]
            env_data = self.pull_env_metrics(env)
            for k, v in env_data.items():
                metrics_logger.log_value(k, v, clear_on_reduce=True)


@deprecated(
    reason="Use WrappedEpisodeDataCallbacks instead. This class is not maintained."
)
class MultiagentEpisodeDataCallbacks(DefaultCallbacks):
    """Deprecated method of logging information at the end of each episode.

    :meta private:
    """

    def __init__(self, *args, **kwargs):
        """Log information at the end of each episode.

        Make a subclass of :class:`MultiagentEpisodeDataCallbacks` and override
        ``pull_env_metrics`` and ``pull_sat_metrics`` to log environment-specific
        information at the end of each episode. Satellite metrics are logged per-satellite
        and as a mean across all satellites.

        Note that satellites persist in the simulator even after death, so recorded values
        from the end of the episode may not be the same as the values when the agent died.
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

    def pull_sat_metrics(self, env, satellite) -> dict[str, float]:
        """Log per-satellite metrics at the end of each episode.

        This function is called whenever ``env`` is terminated or truncated. It should
        return a dictionary of metrics to log.

        Args:
            env: An environment that has completed.
            satellite: A satellite in the environment.
        """
        return {}

    def on_episode_end(
        self,
        env=None,
        metrics_logger=None,
        **_,
    ) -> None:
        """Call pull_env_metrics and log the results.

        :meta private:
        """
        env = env.par_env
        env_data = self.pull_env_metrics(env)
        for k, v in env_data.items():
            metrics_logger.log_value(k, v, clear_on_reduce=True)

        all_sat_data = []

        for sat in env.satellites:
            sat_data = self.pull_sat_metrics(env, sat)
            all_sat_data.append(sat_data)
            for k, v in sat_data.items():
                metrics_logger.log_value(f"{sat.id}/{k}", v, clear_on_reduce=True)

        for k in all_sat_data[0].keys():
            metrics_logger.log_value(
                f"mean/{k}",
                np.mean([sat_data[k] for sat_data in all_sat_data]),
                clear_on_reduce=True,
            )


class WrappedEpisodeDataCallbacks(DefaultCallbacks):
    def __init__(self, *args, **kwargs):
        """Log information at the end of each episode.

        Logs data from an environment wrapped with :class:`EpisodeDataWrapper` or
        :class:`EpisodeDataParallelWrapper`. See :doc:`/examples/rllib_training` for an
        example of how to use this class.
        """
        super().__init__(*args, **kwargs)

    def on_environment_created(
        self,
        env=None,
        metrics_logger=None,
        **kwargs,
    ):
        """Identify the environment and attach the metrics logger.

        :meta private:
        """
        if "base_env" in kwargs:  # Old RLlib stack
            env = kwargs["base_env"].vector_env.envs[0]  # noqa: F841
        elif hasattr(env, "par_env"):  # New RLlib stack, parallel
            env = env.par_env
        else:  # New RLlib stack, single_agent
            env = env.envs[-1]
        env.set_metrics_logger(metrics_logger)


class EpisodeDataLogger:
    """Base class for logging data at the end of each episode."""

    def __init__(self, episode_data_callback=None, satellite_data_callback=None):
        """Initialize the episode data logger."""
        if episode_data_callback is None:
            episode_data_callback = lambda _: {}
        self.episode_data_callback = episode_data_callback
        if satellite_data_callback is None:
            satellite_data_callback = lambda *_: {}
        self.satellite_data_callback = satellite_data_callback
        self.metrics_logger = None
        self.env: "GeneralSatelliteTasking"
        self.satellites: list["Satellite"]

    def set_metrics_logger(self, metrics_logger):
        """Set the metrics logger for this environment."""
        self.metrics_logger = metrics_logger

    def log_data_on_reset(self):
        """Log data at the end of each episode before resetting the environment."""
        if self.metrics_logger is not None:
            episode_data = self.episode_data_callback(self)
            for k, v in episode_data.items():
                self.metrics_logger.log_value(k, v, clear_on_reduce=True)

            all_sat_data = []

            for sat in self.satellites:
                sat_data = self.satellite_data_callback(self, sat)
                all_sat_data.append(sat_data)
                for k, v in sat_data.items():
                    self.metrics_logger.log_value(
                        f"{sat.id}/{k}", v, clear_on_reduce=True
                    )

            for k in all_sat_data[0].keys():
                self.metrics_logger.log_value(
                    f"mean/{k}",
                    np.mean([sat_data[k] for sat_data in all_sat_data]),
                    clear_on_reduce=True,
                )

    def reset(self, **kwargs):
        """Log data before resetting the environment."""
        self.log_data_on_reset()
        return self.env.reset(**kwargs)


class EpisodeDataWrapper(EpisodeDataLogger, Wrapper):
    def __init__(
        self,
        env: "Env",
        episode_data_callback: Optional[
            Callable[["GeneralSatelliteTasking"], dict[str, float]]
        ] = None,
        satellite_data_callback: Optional[
            Callable[["GeneralSatelliteTasking", "Satellite"], dict[str, float]]
        ] = None,
    ):
        """Wrapper for logging data at the end of each multiagent episode.

        This wrapper should be used with the :class:`WrappedEpisodeDataCallbacks` in
        RLlib. At the end of each episode, the environment will log data using the provided
        callback functions. See :doc:`/examples/rllib_training` for an example of
        how to use this class.

        Args:
            env: The environment to wrap.
            episode_data_callback: A function that takes the environment as an argument
                and returns a dictionary of episode-level metrics.
            satellite_data_callback: A function that takes the environment and a satellite
                as arguments and returns a dictionary of satellite-level metrics.
        """
        EpisodeDataLogger.__init__(self, episode_data_callback, satellite_data_callback)
        Wrapper.__init__(self, env)


class EpisodeDataParallelWrapper(EpisodeDataLogger, BaseParallelWrapper):
    def __init__(
        self,
        env: "ParallelEnv",
        episode_data_callback: Optional[
            Callable[["GeneralSatelliteTasking"], dict[str, float]]
        ] = None,
        satellite_data_callback: Optional[
            Callable[["GeneralSatelliteTasking", "Satellite"], dict[str, float]]
        ] = None,
    ):
        """Wrapper for logging data at the end of each multiagent episode.

        This wrapper should be used with the :class:`WrappedEpisodeDataCallbacks` in
        RLlib. At the end of each episode, the environment will log data using the provided
        callback functions. See :doc:`/examples/async_multiagent_training` for an example of
        how to use this class.

        Args:
            env: The environment to wrap.
            episode_data_callback: A function that takes the environment as an argument
                and returns a dictionary of episode-level metrics.
            satellite_data_callback: A function that takes the environment and a satellite
                as arguments and returns a dictionary of satellite-level metrics.
        """
        EpisodeDataLogger.__init__(self, episode_data_callback, satellite_data_callback)
        BaseParallelWrapper.__init__(self, env)


__doc_title__ = "RLlib Callbacks"
__all__ = [
    "EpisodeDataLogger",
    "EpisodeDataWrapper",
    "EpisodeDataParallelWrapper",
    "WrappedEpisodeDataCallbacks",
]
