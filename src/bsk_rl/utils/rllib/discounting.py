"""``bsk_rl.utils.rllib.discounting`` is a collection of utilities for semi-MDP style discounting.

See the following examples for how to use these utilities:

* :doc:`/examples/time_discounted_gae` - An example of :class:`TimeDiscountedGAEPPOTorchLearner`
  in a single-agent case.
* :doc:`/examples/async_multiagent_training` - An example of the time discounted learner
  and connectors for asynchronous multi-agent training (:class:`ContinuePreviousAction`,
  :class:`MakeAddedStepActionValid`, and :class:`CondenseMultiStepActions`).
"""

from logging import getLogger
from typing import Any, List, Literal, Optional

import numpy as np
from ray.rllib.algorithms.ppo.ppo_learner import PPOLearner
from ray.rllib.algorithms.ppo.tf.ppo_tf_learner import PPOTfLearner
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.postprocessing.episodes import (
    add_one_ts_to_episodes_and_truncate,
    remove_last_ts_from_data,
    remove_last_ts_from_episodes_and_restore_truncateds,
)
from ray.rllib.utils.postprocessing.zero_padding import unpad_data_if_necessary
from ray.rllib.utils.typing import EpisodeType

from bsk_rl import NO_ACTION
from bsk_rl.gym import is_no_action, no_action_like

logger = getLogger(__name__)


class ContinuePreviousAction(ConnectorV2):
    def __init__(self, *args, **kwargs):
        """Override actions with ``NO_ACTION`` on connector pass if the agent does not require retasking."""
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        *,
        data: Optional[Any],
        episodes: List[EpisodeType],
        **_,
    ) -> Any:
        """Override actions with ``NO_ACTION`` on connector pass.

        :meta private:
        """
        for sa_episode in self.single_agent_episode_iterator(
            episodes,
            agents_that_stepped_only=True,
        ):
            if not sa_episode.get_infos(-1)["requires_retasking"]:
                if sa_episode.agent_id is None:
                    assert len(data[Columns.ACTIONS]) == 1
                    id_tuple = list(data[Columns.ACTIONS].keys())[0]
                else:
                    id_tuples = [
                        id_tuple
                        for id_tuple in data[Columns.ACTIONS].keys()
                        if id_tuple[1] == sa_episode.agent_id
                    ]
                    if len(id_tuples) == 0:
                        continue
                    else:
                        id_tuple = id_tuples[0]

                data[Columns.ACTIONS][id_tuple][0] = (
                    data[Columns.ACTIONS][id_tuple][0] * 0 + NO_ACTION
                )

        return data


class ContinuePreviousActionAppended(ConnectorV2):
    def __init__(self, *args, **kwargs):
        """Override actions with ``NO_ACTION`` on connector pass if the agent does not require retasking.

        This additional connector must be appended after NormalizeAndClipActions in continuous
        action spaces. For example:

        .. code-block:: python

            # Append extra connector to the pipeline
            old_connector_builder = config.build_module_to_env_connector

            def new_connector_builder(env):
                pipeline = old_connector_builder(env)
                after = pipeline.insert_after(
                    "NormalizeAndClipActions", ContinuePreviousActionAppended()
                )
                return pipeline

            config.build_module_to_env_connector = new_connector_builder

        """
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        *,
        data: Optional[Any],
        episodes: List[EpisodeType],
        **_,
    ) -> Any:
        """Override actions with ``NO_ACTION`` on connector pass.

        :meta private:
        """
        for id_tuple in data[Columns.ACTIONS_FOR_ENV].keys():
            if is_no_action(data[Columns.ACTIONS][id_tuple][0]):
                data[Columns.ACTIONS_FOR_ENV][id_tuple][0] = no_action_like(
                    data[Columns.ACTIONS_FOR_ENV][id_tuple][0]
                )

        return data


class MakeAddedStepActionValid(ConnectorV2):
    def __init__(self, *args, expected_train_batch_size, **kwargs):
        """Ensure that padded steps are not duplicates of ``NO_ACTION`` steps."""
        super().__init__(*args, **kwargs)
        self.expected_train_batch_size = expected_train_batch_size

    def __call__(
        self,
        *,
        data: Optional[Any],
        episodes: List[EpisodeType],
        **_,
    ) -> Any:
        """Ensure that padded steps are not duplicates of ``NO_ACTION`` steps.

        :meta private:
        """
        total_episodes = 0
        total_steps = 0
        episode_lens = []
        episode_multi_agent_ids = []
        for episode in self.single_agent_episode_iterator(
            episodes, agents_that_stepped_only=False
        ):
            episode_lens.append(len(episode))
            episode_multi_agent_ids.append(episode.multi_agent_episode_id)

        if episode_multi_agent_ids[0] is None:
            total_episodes = len(episode_lens)
            total_steps = sum(episode_lens)
        else:
            total_episodes = len(set(episode_multi_agent_ids))
            max_lens = {}
            for episode_id, length in zip(episode_multi_agent_ids, episode_lens):
                if episode_id not in max_lens or length > max_lens[episode_id]:
                    max_lens[episode_id] = length
            total_steps = sum(max_lens.values())

        one_ts_added = False
        if total_steps == self.expected_train_batch_size + total_episodes:
            one_ts_added = True

        if one_ts_added:
            for episode in self.single_agent_episode_iterator(
                episodes, agents_that_stepped_only=False
            ):
                # Arbitrary filler, will be deleted by remove_last_ts_from_episodes_and_restore_truncateds
                if is_no_action(episode.actions[-1]):
                    episode.actions[-1] = episode._action_space.sample()
                episode.validate()

        return data


class CondenseMultiStepActions(ConnectorV2):
    def __init__(self, *args, **kwargs):
        """Combine steps that used ``NO_ACTION`` on connector pass."""
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        *,
        data: Optional[Any],
        episodes: List[EpisodeType],
        **_,
    ) -> Any:
        """Combine steps that used ``NO_ACTION`` on connector pass.

        :meta private:
        """
        for episode in self.single_agent_episode_iterator(
            episodes, agents_that_stepped_only=False
        ):
            action_idx = list(
                np.argwhere(
                    [not is_no_action(action) for action in episode.actions.data]
                ).flatten()
            )
            obs_idx = action_idx.copy()
            obs_idx.append(len(episode) - 1)

            lookback = episode.actions.data[: episode.actions.lookback]
            new_lookback = episode.actions.lookback
            for action in lookback:
                if is_no_action(action):
                    new_lookback -= 1
                else:
                    break

            # Only keep non-None actions
            episode.actions.data = np.array(episode.actions.data)[action_idx]
            episode.actions.lookback = new_lookback
            for column in episode.extra_model_outputs:
                episode.extra_model_outputs[column].data = episode.extra_model_outputs[
                    column
                ].data[action_idx]
                episode.extra_model_outputs[column].lookback = new_lookback

            # Update episode length
            episode.t = episode.t_started + len(episode.actions)

            # Only keep obs that resulted in those actions
            episode.observations.data = np.array(episode.observations.data)[obs_idx]
            episode.observations.lookback = new_lookback

            # Associate subsequent rewards with prior action
            rewards = []
            requires_retasking = []
            for i, idx_start in enumerate(action_idx):
                if i == len(action_idx) - 1:
                    idx_end = len(episode) - 1
                else:
                    idx_end = action_idx[i + 1] - 1
                rewards.append(
                    sum(episode.rewards[idx_start : idx_end + 1])
                )  # Doesn't discount over course of multistep
                requires_retasking.append(
                    episode.infos.data[idx_start]["requires_retasking"]
                )
            requires_retasking.append(True)
            episode.rewards.data = np.array(rewards)
            episode.rewards.lookback = new_lookback

            # Accumulate d_ts from prior action
            d_ts = []
            for i, idx_end in enumerate(obs_idx):
                if i == 0:
                    idx_start = 0
                else:
                    idx_start = obs_idx[i - 1] + 1
                d_ts.append(
                    sum(
                        info["d_ts"]
                        for info in episode.infos.data[idx_start : idx_end + 1]
                    )
                )
            episode.infos.data = [
                dict(d_ts=d_ts, requires_retasking=requires_retasking)
                for d_ts, requires_retasking in zip(d_ts, requires_retasking)
            ]
            episode.infos.lookback = new_lookback

            episode.validate()

        return data


def compute_value_targets_time_discounted(
    values,
    rewards,
    terminateds,
    truncateds,
    step_durations,
    gamma: float,
    lambda_: float,
    reward_time: Literal["step_start", "step_end"] = "step_end",
):
    """Computes value function (vf) targets given vf predictions and rewards.

    Note that advantages can then easily be computed via the formula:
    advantages = targets - vf_predictions
    """
    assert reward_time in ["step_start", "step_end"]

    # Shift step durations to associate with previous timestep
    # delta_t->t+1 comes with t+1's info, but should be used with t
    step_durations = np.concatenate((step_durations[1:], [step_durations[-1]]))

    # Force-set all values at terminals (not at truncations!) to 0.0.
    orig_values = flat_values = values * (1.0 - terminateds)

    flat_values = np.append(flat_values, 0.0)
    if reward_time == "step_start":
        intermediates = (
            rewards + gamma**step_durations * (1 - lambda_) * flat_values[1:]
        )
    elif reward_time == "step_end":
        intermediates = gamma**step_durations * (
            rewards + (1 - lambda_) * flat_values[1:]
        )
    continues = 1.0 - terminateds

    Rs = []
    last = flat_values[-1]
    for t in reversed(range(intermediates.shape[0])):
        last = (
            intermediates[t]
            + continues[t] * gamma ** step_durations[t] * lambda_ * last
        )
        # last = (
        #     intermediates[t]
        #     + continues[t] * gamma * lambda_ * last
        # )

        Rs.append(last)
        if truncateds[t]:
            last = orig_values[t]

    # Reverse back to correct (time) direction.
    value_targets = np.stack(list(reversed(Rs)), axis=0)

    return value_targets.astype(np.float32)


class TimeDiscountedGAEPPOLearner(PPOLearner):
    def __init__(self, *args, **kwargs):
        """Discount episodes according to the ``d_ts`` value in the info dictionary."""
        super().__init__(*args, **kwargs)

    def _compute_gae_from_episodes(
        self,
        *,
        episodes: Optional[List[EpisodeType]] = None,
    ) -> tuple[Optional[dict[str, Any]], Optional[List[EpisodeType]]]:
        if not episodes:
            raise ValueError(
                "`PPOLearner._compute_gae_from_episodes()` must have the `episodes` "
                "arg provided! Otherwise, GAE/advantage computation can't be performed."
            )

        batch = {}

        sa_episodes_list = list(
            self._learner_connector.single_agent_episode_iterator(
                episodes, agents_that_stepped_only=False
            )
        )
        # Make all episodes one ts longer in order to just have a single batch
        # (and distributed forward pass) for both vf predictions AND the bootstrap
        # vf computations.
        orig_truncateds_of_sa_episodes = add_one_ts_to_episodes_and_truncate(
            sa_episodes_list
        )

        # Call the learner connector (on the artificially elongated episodes)
        # in order to get the batch to pass through the module for vf (and
        # bootstrapped vf) computations.
        batch_for_vf = self._learner_connector(
            rl_module=self.module,
            data={},
            episodes=episodes,
            shared_data={},
        )

        # print(batch_for_vf)
        # Perform the value model's forward pass.
        vf_preds = convert_to_numpy(self._compute_values(batch_for_vf))

        for module_id, module_vf_preds in vf_preds.items():
            # Collect new (single-agent) episode lengths.
            episode_lens_plus_1 = [
                len(e)
                for e in sa_episodes_list
                if e.module_id is None or e.module_id == module_id
            ]

            # Remove all zero-padding again, if applicable, for the upcoming
            # GAE computations.
            module_vf_preds = unpad_data_if_necessary(
                episode_lens_plus_1, module_vf_preds
            )

            # Compute value targets.
            reward_time = self.config.learner_config_dict.get("reward_time", "step_end")
            module_value_targets = compute_value_targets_time_discounted(
                values=module_vf_preds,
                rewards=unpad_data_if_necessary(
                    episode_lens_plus_1,
                    convert_to_numpy(batch_for_vf[module_id][Columns.REWARDS]),
                ),
                terminateds=unpad_data_if_necessary(
                    episode_lens_plus_1,
                    convert_to_numpy(batch_for_vf[module_id][Columns.TERMINATEDS]),
                ),
                truncateds=unpad_data_if_necessary(
                    episode_lens_plus_1,
                    convert_to_numpy(batch_for_vf[module_id][Columns.TRUNCATEDS]),
                ),
                step_durations=unpad_data_if_necessary(
                    episode_lens_plus_1,
                    np.array(
                        [
                            info["d_ts"]
                            for info in batch_for_vf[module_id][Columns.INFOS]
                        ]
                    ),
                ),
                gamma=self.config.gamma,
                lambda_=self.config.lambda_,
                reward_time=reward_time,
            )

            # Remove the extra timesteps again from vf_preds and value targets. Now that
            # the GAE computation is done, we don't need this last timestep anymore in
            # any of our data.
            module_vf_preds, module_value_targets = remove_last_ts_from_data(
                episode_lens_plus_1,
                module_vf_preds,
                module_value_targets,
            )
            module_advantages = module_value_targets - module_vf_preds
            # Drop vf-preds, not needed in loss. Note that in the PPORLModule, vf-preds
            # are recomputed with each `forward_train` call anyway.
            # Standardize advantages (used for more stable and better weighted
            # policy gradient computations).
            module_advantages = (module_advantages - module_advantages.mean()) / max(
                1e-4, module_advantages.std()
            )

            # Restructure ADVANTAGES and VALUE_TARGETS in a way that the Learner
            # connector can properly re-batch these new fields.
            batch_pos = 0
            for eps in sa_episodes_list:
                if eps.module_id is not None and eps.module_id != module_id:
                    continue
                len_ = len(eps) - 1
                self._learner_connector.add_n_batch_items(
                    batch=batch,
                    column=Postprocessing.ADVANTAGES,
                    items_to_add=module_advantages[batch_pos : batch_pos + len_],
                    num_items=len_,
                    single_agent_episode=eps,
                )
                self._learner_connector.add_n_batch_items(
                    batch=batch,
                    column=Postprocessing.VALUE_TARGETS,
                    items_to_add=module_value_targets[batch_pos : batch_pos + len_],
                    num_items=len_,
                    single_agent_episode=eps,
                )
                batch_pos += len_

        # Remove the extra (artificial) timesteps again at the end of all episodes.
        remove_last_ts_from_episodes_and_restore_truncateds(
            sa_episodes_list,
            orig_truncateds_of_sa_episodes,
        )

        return batch, episodes


class TimeDiscountedGAEPPOTorchLearner(PPOTorchLearner, TimeDiscountedGAEPPOLearner):
    pass


class TimeDiscountedGAEPPOTfLearner(PPOTfLearner, TimeDiscountedGAEPPOLearner):
    pass


__doc_title__ = "Semi-MDP Discounting in RLlib"
__all__ = [
    "ContinuePreviousAction",
    "MakeAddedStepActionValid",
    "CondenseMultiStepActions",
    "TimeDiscountedGAEPPOTorchLearner",
    "TimeDiscountedGAEPPOTfLearner",
    "TimeDiscountedGAEPPOLearner",
]
