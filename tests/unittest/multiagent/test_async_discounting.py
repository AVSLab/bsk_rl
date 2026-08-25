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
