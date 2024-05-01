from unittest.mock import MagicMock, patch

import pytest
from gymnasium import spaces

from bsk_rl.envs.general_satellite_tasking.gym_env import (
    GeneralSatelliteTasking,
    MultiagentSatelliteTasking,
    SingleSatelliteTasking,
)
from bsk_rl.envs.general_satellite_tasking.scenario.satellites import Satellite


class TypeA:
    pass


class TypeB:
    pass


class TypeAprime(TypeA):
    pass


class TestGeneralSatelliteTasking:
    @patch(
        "bsk_rl.envs.general_satellite_tasking.gym_env.GeneralSatelliteTasking.__init__",
        MagicMock(return_value=None),
    )
    def test_generate_env_args(self):
        env = GeneralSatelliteTasking()
        env.env_args_generator = {"a": 1, "b": lambda: 2}
        env._generate_env_args()
        assert env.env_args == {"a": 1, "b": 2}

    @pytest.mark.parametrize(
        "classes,result",
        [
            ([[TypeA]], TypeA),
            ([[TypeA], [TypeA]], TypeA),
            ([[TypeA], [TypeAprime]], TypeAprime),
        ],
    )
    @patch(
        "bsk_rl.envs.general_satellite_tasking.gym_env.GeneralSatelliteTasking.__init__",
        MagicMock(return_value=None),
    )
    def test_minimum_env_model(self, classes, result):
        env = GeneralSatelliteTasking()
        env.satellites = [
            MagicMock(
                dyn_type=MagicMock(_requires_env=MagicMock(return_value=class_list))
            )
            for class_list in classes
        ]
        assert env._minimum_env_model() == result

    @patch(
        "bsk_rl.envs.general_satellite_tasking.gym_env.GeneralSatelliteTasking.__init__",
        MagicMock(return_value=None),
    )
    def test_minimum_env_model_mixed(self):
        env = GeneralSatelliteTasking()
        env.satellites = [
            MagicMock(
                dyn_type=MagicMock(_requires_env=MagicMock(return_value=class_list))
            )
            for class_list in [[TypeA], [TypeB]]
        ]
        model = env._minimum_env_model()
        assert issubclass(model, TypeA)
        assert issubclass(model, TypeB)

    @patch(
        "bsk_rl.envs.general_satellite_tasking.gym_env.Simulator",
    )
    def test_reset(self, mock_sim):
        mock_sat = MagicMock()
        mock_sat.sat_args_generator = {}
        mock_data = MagicMock(env_features=None)
        env = GeneralSatelliteTasking(
            satellites=[mock_sat],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=mock_data,
        )
        env.env_args_generator = {"utc_init": "a long time ago"}
        env.communicator = MagicMock()
        env.reset()
        assert mock_sat.sat_args_generator["utc_init"] == env.env_args["utc_init"]
        mock_sim.assert_called_once()
        mock_sat.reset_pre_sim.assert_called_once()
        mock_data.create_data_store.assert_called_once_with(mock_sat)
        env.communicator.reset.assert_called_once()
        mock_sat.reset_post_sim.assert_called_once()

    def test_get_obs(self):
        env = GeneralSatelliteTasking(
            satellites=[MagicMock(get_obs=MagicMock(return_value=i)) for i in range(3)],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
        )
        assert env._get_obs() == (0, 1, 2)

    def test_get_info(self):
        mock_sats = [MagicMock(info={"sat_index": i}) for i in range(3)]
        env = GeneralSatelliteTasking(
            satellites=mock_sats,
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
        )
        env.latest_step_duration = 10.0
        expected = {sat.id: {"sat_index": i} for i, sat in enumerate(mock_sats)}
        expected["d_ts"] = 10.0
        expected["requires_retasking"] = [sat.id for sat in mock_sats]
        assert env._get_info() == expected

    def test_action_space(self):
        env = GeneralSatelliteTasking(
            satellites=[
                MagicMock(action_space=spaces.Discrete(i + 1)) for i in range(3)
            ],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
        )
        assert env.action_space == spaces.Tuple(
            (spaces.Discrete(1), spaces.Discrete(2), spaces.Discrete(3))
        )

    def test_obs_space_no_sim(self):
        env = GeneralSatelliteTasking(
            satellites=[
                MagicMock(observation_space=spaces.Discrete(i + 1)) for i in range(3)
            ],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
        )
        env.seed = 123
        old_seed = env.seed
        env.reset = MagicMock()
        assert env.observation_space == spaces.Tuple(
            (spaces.Discrete(1), spaces.Discrete(2), spaces.Discrete(3))
        )
        env.reset.assert_called_once_with(seed=old_seed)

    def test_obs_space_existing_sim(self):
        env = GeneralSatelliteTasking(
            satellites=[
                MagicMock(observation_space=spaces.Discrete(i + 1)) for i in range(3)
            ],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
        )
        env.simulator = MagicMock()
        env.reset = MagicMock()
        assert env.observation_space == spaces.Tuple(
            (spaces.Discrete(1), spaces.Discrete(2), spaces.Discrete(3))
        )
        env.reset.assert_not_called()

    def test_step(self):
        mock_sats = [MagicMock() for _ in range(2)]
        env = GeneralSatelliteTasking(
            satellites=mock_sats,
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(
                reward=MagicMock(return_value={sat.id: 12.5 for sat in mock_sats})
            ),
        )
        env.simulator = MagicMock(sim_time=101.0)
        _, reward, _, _, info = env.step((0, 10))
        mock_sats[0].set_action.assert_called_once_with(0)
        mock_sats[1].set_action.assert_called_once_with(10)
        env.simulator.run.assert_called_once()
        assert env.latest_step_duration == 0.0
        for sat in mock_sats:
            sat.data_store.internal_update.assert_called_once()
        assert reward == 25.0

    def test_step_bad_action(self):
        mock_sats = [MagicMock() for _ in range(2)]
        env = GeneralSatelliteTasking(
            satellites=mock_sats,
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(reward=MagicMock(return_value=25.0)),
        )
        env.simulator = MagicMock(sim_time=101.0)
        with pytest.raises(ValueError):
            env.step((0, 10, 20))
        with pytest.raises(ValueError):
            env.step((0,))

    @pytest.mark.parametrize("sat_death", [True, False])
    @pytest.mark.parametrize("timeout", [True, False])
    @pytest.mark.parametrize("terminate_on_time_limit", [True, False])
    def test_step_stopped(self, sat_death, timeout, terminate_on_time_limit):
        mock_sats = [MagicMock() for _ in range(2)]
        env = GeneralSatelliteTasking(
            satellites=mock_sats,
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(
                reward=MagicMock(return_value={sat.id: 12.5 for sat in mock_sats})
            ),
            terminate_on_time_limit=terminate_on_time_limit,
        )
        env.simulator = MagicMock(sim_time=101.0)
        if timeout:
            env.time_limit = 100.0
        else:
            env.time_limit = 1000.0

        if sat_death:
            mock_sats[1].is_alive.return_value = False

        _, _, terminated, truncated, _ = env.step((0, 10))

        assert terminated == (sat_death or (timeout and terminate_on_time_limit))
        assert truncated == timeout

    @patch.multiple(Satellite, __abstractmethods__=set())
    def test_step_retask_needed(self, capfd):
        mock_sat = MagicMock()
        env = SingleSatelliteTasking(
            satellites=[mock_sat],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(reward=MagicMock(return_value={mock_sat.id: 25.0})),
        )
        env.simulator = MagicMock(sim_time=101.0)
        env.step(None)
        assert mock_sat.requires_retasking
        mock_sat.requires_retasking = True
        env.step(None)
        assert mock_sat.requires_retasking

    def test_render(self):
        pass

    def test_close(self):
        env = GeneralSatelliteTasking(
            satellites=[MagicMock()],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
        )
        env.simulator = MagicMock()
        env.close()
        assert not hasattr(env, "simulator")


class TestSingleSatelliteTasking:
    @patch.multiple(
        Satellite,
        __abstractmethods__=set(),
        __init__=MagicMock(return_value=None),
    )
    def test_init(self):
        mock_sat = Satellite("sat", {})
        env = SingleSatelliteTasking(
            satellites=mock_sat,
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
        )
        assert env.satellite == mock_sat

    def test_init_multisat(self):
        with pytest.raises(ValueError):
            SingleSatelliteTasking(
                satellites=[MagicMock(), MagicMock()],
                env_type=MagicMock(),
                env_features=MagicMock(),
                data_manager=MagicMock(),
            )

    @staticmethod
    def make_env():
        mock_sat = MagicMock()
        env = SingleSatelliteTasking(
            satellites=[mock_sat],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
        )
        return env, mock_sat

    def test_action_space(self):
        env, mock_sat = self.make_env()
        assert env.action_space == mock_sat.action_space

    @patch(
        "bsk_rl.envs.general_satellite_tasking.gym_env.GeneralSatelliteTasking.observation_space"
    )
    def test_observation_space(self, obs_patch):
        env, mock_sat = self.make_env()
        env.simulator = MagicMock()
        assert env.observation_space == mock_sat.observation_space

    @patch("bsk_rl.envs.general_satellite_tasking.gym_env.GeneralSatelliteTasking.step")
    def test_step(self, step_patch):
        env, mock_sat = self.make_env()
        env.step("action")
        step_patch.assert_called_once_with(["action"])

    def test_get_obs(self):
        env, mock_sat = self.make_env()
        assert env._get_obs() == mock_sat.get_obs()


class TestMultiagentSatelliteTasking:
    @patch(
        "bsk_rl.envs.general_satellite_tasking.gym_env.Simulator",
    )
    @patch(
        "bsk_rl.envs.general_satellite_tasking.gym_env.MultiagentSatelliteTasking._get_obs",
    )
    @patch(
        "bsk_rl.envs.general_satellite_tasking.gym_env.MultiagentSatelliteTasking._get_info",
    )
    def test_reset(self, mock_sim, obs_fn, info_fn):
        mock_sat_1 = MagicMock()
        mock_sat_2 = MagicMock()
        mock_sat_1.sat_args_generator = {}
        mock_sat_2.sat_args_generator = {}
        mock_data = MagicMock(env_features=None)
        env = MultiagentSatelliteTasking(
            satellites=[mock_sat_1, mock_sat_2],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=mock_data,
        )
        env.env_args_generator = {"utc_init": "a long time ago"}
        env.communicator = MagicMock()
        obs, info = env.reset()
        obs_fn.assert_called_once()
        info_fn.assert_called_once()

    @patch(
        "bsk_rl.envs.general_satellite_tasking.gym_env.GeneralSatelliteTasking._get_truncated",
        MagicMock(return_value=False),
    )
    def test_agents(self):
        env = MultiagentSatelliteTasking(
            satellites=[MagicMock() for i in range(3)],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
        )
        assert env.agents == [sat.id for sat in env.satellites]
        assert env.num_agents == 3
        assert env.possible_agents == [sat.id for sat in env.satellites]
        assert env.max_num_agents == 3

    @patch(
        "bsk_rl.envs.general_satellite_tasking.gym_env.GeneralSatelliteTasking._get_truncated",
        MagicMock(return_value=False),
    )
    def test_get_obs(self):
        env = MultiagentSatelliteTasking(
            satellites=[MagicMock(get_obs=MagicMock(return_value=i)) for i in range(3)],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
        )
        env.newly_dead = []
        assert env._get_obs() == {sat.id: i for i, sat in enumerate(env.satellites)}

    @patch(
        "bsk_rl.envs.general_satellite_tasking.gym_env.GeneralSatelliteTasking._get_truncated",
        MagicMock(return_value=False),
    )
    def test_get_info(self):
        mock_sats = [MagicMock(info={"sat_index": i}) for i in range(3)]
        env = MultiagentSatelliteTasking(
            satellites=mock_sats,
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
        )
        env.newly_dead = []
        env.latest_step_duration = 10.0
        expected = {sat.id: {"sat_index": i} for i, sat in enumerate(mock_sats)}
        expected["d_ts"] = 10.0
        expected["requires_retasking"] = [sat.id for sat in mock_sats]
        assert env._get_info() == expected

    def test_action_spaces(self):
        env = MultiagentSatelliteTasking(
            satellites=[
                MagicMock(action_space=spaces.Discrete(i + 1)) for i in range(3)
            ],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
        )
        assert env.action_spaces == {
            env.satellites[0].id: spaces.Discrete(1),
            env.satellites[1].id: spaces.Discrete(2),
            env.satellites[2].id: spaces.Discrete(3),
        }

    def test_obs_spaces(self):
        env = MultiagentSatelliteTasking(
            satellites=[
                MagicMock(observation_space=spaces.Discrete(i + 1)) for i in range(3)
            ],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
        )
        env.simulator = MagicMock()
        env.reset = MagicMock()
        assert env.observation_spaces == {
            env.satellites[0].id: spaces.Discrete(1),
            env.satellites[1].id: spaces.Discrete(2),
            env.satellites[2].id: spaces.Discrete(3),
        }

    @patch(
        "bsk_rl.envs.general_satellite_tasking.gym_env.GeneralSatelliteTasking._get_truncated",
        MagicMock(return_value=False),
    )
    def test_get_reward(self):
        env = MultiagentSatelliteTasking(
            satellites=[
                MagicMock(is_alive=MagicMock(return_value=False)) for i in range(3)
            ],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
            failure_penalty=-20.0,
        )
        env.newly_dead = [sat.id for sat in env.satellites]
        env.reward_dict = {sat.id: 10.0 for i, sat in enumerate(env.satellites)}
        assert env._get_reward() == {
            sat.id: -10.0 for i, sat in enumerate(env.satellites)
        }

    @pytest.mark.parametrize("timeout", [False, True])
    @pytest.mark.parametrize("terminate_on_time_limit", [False, True])
    def test_get_terminated(self, timeout, terminate_on_time_limit):
        env = MultiagentSatelliteTasking(
            satellites=[
                MagicMock(is_alive=MagicMock(return_value=True if i != 0 else False))
                for i in range(3)
            ],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
            terminate_on_time_limit=terminate_on_time_limit,
            time_limit=100,
        )
        env.simulator = MagicMock(sim_time=101 if timeout else 99)

        if not timeout or not terminate_on_time_limit:
            env.newly_dead = [sat.id for sat in env.satellites]
            assert env._get_terminated() == {
                env.satellites[0].id: True,
                env.satellites[1].id: False,
                env.satellites[2].id: False,
            }
        else:
            env.newly_dead = [sat.id for sat in env.satellites]
            assert env._get_terminated() == {
                env.satellites[0].id: True,
                env.satellites[1].id: True,
                env.satellites[2].id: True,
            }

    @pytest.mark.parametrize("time", [99, 101])
    def test_get_truncated(self, time):
        env = MultiagentSatelliteTasking(
            satellites=[MagicMock() for i in range(3)],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
            time_limit=100,
        )
        env.simulator = MagicMock(sim_time=time)
        env.newly_dead = [sat.id for sat in env.satellites] if time >= 100 else []
        assert env._get_truncated() == {
            env.satellites[0].id: time >= 100,
            env.satellites[1].id: time >= 100,
            env.satellites[2].id: time >= 100,
        }

    def test_close(self):
        env = MultiagentSatelliteTasking(
            satellites=[MagicMock()],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
        )
        env.simulator = MagicMock()
        env.close()
        assert not hasattr(env, "simulator")

    @patch(
        "bsk_rl.envs.general_satellite_tasking.gym_env.GeneralSatelliteTasking._get_truncated",
        MagicMock(return_value=False),
    )
    def test_dead(self):
        env = MultiagentSatelliteTasking(
            satellites=[MagicMock() for _ in range(3)],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
        )
        env.satellites[1].is_alive = MagicMock(return_value=False)
        env.satellites[2].is_alive = MagicMock(return_value=False)
        env.newly_dead = [env.satellites[2].id]
        assert env.previously_dead == [env.satellites[1].id]
        assert env.agents == [env.satellites[0].id]
        assert env.possible_agents == [sat.id for sat in env.satellites]

    mst = "bsk_rl.envs.general_satellite_tasking.gym_env.MultiagentSatelliteTasking."

    @patch(
        "bsk_rl.envs.general_satellite_tasking.gym_env.GeneralSatelliteTasking._get_truncated",
        MagicMock(return_value=False),
    )
    @patch(mst + "_get_obs", MagicMock())
    @patch(mst + "_get_reward", MagicMock())
    @patch(mst + "_get_terminated", MagicMock())
    @patch(mst + "_get_truncated", MagicMock())
    @patch(mst + "_get_info", MagicMock())
    @patch(
        "bsk_rl.envs.general_satellite_tasking.gym_env.GeneralSatelliteTasking._step",
        MagicMock(),
    )
    def test_step(self):
        env = MultiagentSatelliteTasking(
            satellites=[
                MagicMock(is_alive=MagicMock(return_value=True)) for _ in range(3)
            ],
            env_type=MagicMock(),
            env_features=MagicMock(),
            data_manager=MagicMock(),
        )

        def kill_sat_2():
            env.satellites[2].is_alive.return_value = False

        env._step.side_effect = lambda _: kill_sat_2()
        env.satellites[1].is_alive.return_value = False
        env.step(
            {
                env.satellites[0].id: 0,
                env.satellites[2].id: 2,
            }
        )
        env._step.assert_called_with([0, None, 2])
        assert env.newly_dead == [env.satellites[2].id]
