import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode

from bsk_rl import NO_ACTION
from bsk_rl.utils.rllib.discounting import CondenseMultiStepActions


def test_no_action_steps_condense_and_accumulate_agent_dt():
    episode = SingleAgentEpisode(
        observations=[
            np.array([0]),
            np.array([1]),
            np.array([2]),
            np.array([3]),
            np.array([4]),
        ],
        infos=[
            {"d_ts": 0.0, "requires_retasking": True},
            {"d_ts": 3.0, "requires_retasking": False},
            {"d_ts": 4.0, "requires_retasking": False},
            {"d_ts": 5.0, "requires_retasking": True},
            {"d_ts": 6.0, "requires_retasking": True},
        ],
        actions=[1, NO_ACTION, NO_ACTION, 2],
        rewards=[1.0, 2.0, 3.0, 4.0],
        len_lookback_buffer=0,
    )
    CondenseMultiStepActions()(data=None, episodes=[episode])

    assert list(episode.actions) == [1, 2]
    assert list(episode.rewards) == [6.0, 4.0]
    assert [info["d_ts"] for info in episode.infos] == [0.0, 12.0, 6.0]


def _timed_episode(durations, actions, rewards, rate=0):
    return SingleAgentEpisode(
        observations=[np.array([i]) for i in range(len(actions) + 1)],
        infos=[
            dict(d_ts=dt, requires_retasking=True, communication_cost_rate=rate)
            for dt in [0, *durations]
        ],
        actions=actions,
        rewards=rewards,
        len_lookback_buffer=0,
    )


def test_discounted_return_is_invariant_to_peer_boundaries():
    gamma = 0.99
    unsplit = _timed_episode([10], [3], [8 - 2 * 10], rate=2)
    split = _timed_episode([3, 7], [3, NO_ACTION], [-2 * 3, 8 - 2 * 7], rate=2)
    for ep in (unsplit, split):
        CondenseMultiStepActions(gamma=gamma)(data=None, episodes=[ep])
    expected = gamma**10 * 8 - 2 * (1 - gamma**10) / -np.log(gamma)
    np.testing.assert_allclose(unsplit.rewards.data, [expected])
    np.testing.assert_allclose(split.rewards.data, [expected])
    assert split.observations.data[-1] == 2
    assert split.infos.data[-1]["d_ts"] == 10


def test_start_timed_learner_does_not_discount_condensed_reward_twice():
    from bsk_rl.utils.rllib.discounting import compute_value_targets_time_discounted

    rewards = np.array([0.99**10 * 8, 0])
    targets = compute_value_targets_time_discounted(
        np.zeros(2),
        rewards,
        np.array([0, 1]),
        np.zeros(2),
        np.array([0, 10]),
        0.99,
        1,
        reward_time="step_start",
    )
    np.testing.assert_allclose(targets, [rewards[0], 0])


def test_missing_busy_agent_does_not_skip_other_busy_agents():
    from bsk_rl.utils.rllib.discounting import ContinuePreviousAction
    from ray.rllib.core.columns import Columns
    from unittest.mock import patch

    missing = _timed_episode([1], [0], [0])
    missing.agent_id = "absent"
    busy = _timed_episode([1], [0], [0])
    busy.agent_id = "present"
    for ep in (missing, busy):
        ep.infos.data[-1]["requires_retasking"] = False
    data = {Columns.ACTIONS: {("episode", "present", "imager"): [1]}}
    connector = ContinuePreviousAction()
    with patch.object(
        connector, "single_agent_episode_iterator", return_value=iter([missing, busy])
    ):
        connector(data=data, episodes=[])
    assert data[Columns.ACTIONS][("episode", "present", "imager")][0] == NO_ACTION


def test_two_ppo_connector_passes_do_not_discount_rewards_twice():
    episode = _timed_episode([3, 7], [3, NO_ACTION], [1, 8])
    connector = CondenseMultiStepActions(gamma=0.99)
    connector(data=None, episodes=[episode])
    rewards = episode.rewards.data.copy()
    durations = [i["d_ts"] for i in episode.infos.data]
    connector(data=None, episodes=[episode])
    np.testing.assert_allclose(episode.rewards.data, rewards)
    assert [i["d_ts"] for i in episode.infos.data] == durations


def test_explicit_bootstrap_flag_handles_complete_episode_batch_overshoot():
    from bsk_rl.utils.rllib.discounting import MakeAddedStepActionValid

    episode = _timed_episode(
        [3, 4, 5, 6], [2, NO_ACTION, NO_ACTION, NO_ACTION], [0, 0, 1, 0]
    )
    MakeAddedStepActionValid(expected_train_batch_size=2)(
        data=None, episodes=[episode], shared_data={"bsk_bootstrap_added": True}
    )
    assert episode.actions.data[-1] == 2


def test_trace_half_life_is_measured_in_seconds():
    from bsk_rl.utils.rllib.discounting import compute_value_targets_time_discounted

    gamma = 0.5 ** (1 / 45000)
    trace = 0.5 ** (1 / 6000)
    # A delayed unit advantage retains one half after a 6000-second gap.
    actual = compute_value_targets_time_discounted(
        np.zeros(3),
        np.array([0.0, 1.0, 0.0]),
        np.array([0, 0, 1]),
        np.zeros(3),
        np.array([0.0, 6000.0, 1.0]),
        gamma,
        trace,
        reward_time="step_start",
        lambda_time="second",
    )
    np.testing.assert_allclose(actual[0], gamma**6000 * 0.5, rtol=1e-6)
