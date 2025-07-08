from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from gymnasium import spaces

from bsk_rl import ConstellationTasking, GeneralSatelliteTasking, SatelliteTasking
from bsk_rl.data.composition import ComposedReward
from bsk_rl.sats import Satellite


class TypeA:
    pass


class TypeB:
    pass


class TypeAprime(TypeA):
    pass


class TestGeneralSatelliteTasking:
    @patch(
        "bsk_rl.GeneralSatelliteTasking.__init__",
        MagicMock(return_value=None),
    )
    def test_generate_world_args(self):
        env = GeneralSatelliteTasking()
        env.unwrapped.world_args_generator = {"a": 1, "b": lambda: 2}
        env._generate_world_args()
        assert env.unwrapped.world_args == {"a": 1, "b": 2}

    @pytest.mark.parametrize(
        "classes,result",
        [
            ([[TypeA]], TypeA),
            ([[TypeA], [TypeA]], TypeA),
            ([[TypeA], [TypeAprime]], TypeAprime),
        ],
    )
    @patch(
        "bsk_rl.GeneralSatelliteTasking.__init__",
        MagicMock(return_value=None),
    )
    def test_minimum_world_model(self, classes, result):
        env = GeneralSatelliteTasking()
        env.unwrapped.satellites = [
            MagicMock(
                dyn_type=MagicMock(_requires_world=MagicMock(return_value=class_list))
            )
            for class_list in classes
        ]
        assert env._minimum_world_model() == result

    @patch(
        "bsk_rl.GeneralSatelliteTasking.__init__",
        MagicMock(return_value=None),
    )
    def test_minimum_world_model_mixed(self):
        env = GeneralSatelliteTasking()
        env.unwrapped.satellites = [
            MagicMock(
                dyn_type=MagicMock(_requires_world=MagicMock(return_value=class_list))
            )
            for class_list in [[TypeA], [TypeB]]
        ]
        model = env._minimum_world_model()
        assert issubclass(model, TypeA)
        assert issubclass(model, TypeB)

    def test_multiple_rewarders(self):
        mock_sat = MagicMock()
        mock_sat.sat_args_generator = {}
        mock_rewarder = [MagicMock(scenario=None), MagicMock(scenario=None)]
        env = GeneralSatelliteTasking(
            satellites=[mock_sat],
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=mock_rewarder,
        )
        assert isinstance(env.rewarder, ComposedReward)

    @patch("bsk_rl.gym.Simulator")
    def test_reset(self, mock_sim):
        mock_sat = MagicMock()
        mock_sat.sat_args_generator = {}
        mock_rewarder = MagicMock(scenario=None)
        env = GeneralSatelliteTasking(
            satellites=[mock_sat],
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=mock_rewarder,
        )
        mock_sat = env.satellites[0]
        mock_rewarder = env.rewarder
        mock_rewarder.data = 0
        mock_sat.data_store.data = 100
        env.unwrapped.world_args_generator = {"utc_init": "a long time ago"}
        env.communicator = MagicMock(last_communication_time=0.0)
        env.reset()
        assert mock_rewarder.data == 100
        mock_sat.generate_sat_args.assert_called_with(utc_init="a long time ago")
        mock_sim.assert_called_once()
        mock_sat.reset_pre_sim_init.assert_called_once()
        mock_rewarder.create_data_store.assert_called_once_with(mock_sat)
        env.communicator.reset_post_sim_init.assert_called_once()
        mock_sat.reset_post_sim_init.assert_called_once()

    @pytest.mark.parametrize(
        "sat_names,expected",
        [
            (["alice"], ["alice"]),
            (["alice", "alice"], ["alice_0", "alice_1"]),
            (["alice", "bob"], ["alice", "bob"]),
            (
                ["alice", "alice", "bob", "charlie", "bob", "alice"],
                ["alice_0", "alice_1", "bob_0", "charlie", "bob_1", "alice_2"],
            ),
            (["alice", "alice_0", "alice"], ["alice_0_0", "alice_0_1", "alice_1"]),
        ],
    )
    def test_name_conflict(self, sat_names, expected):
        mock_sats = [MagicMock() for _ in range(len(sat_names))]
        for sat, name in zip(mock_sats, sat_names):
            sat.name = name
        env = GeneralSatelliteTasking(
            satellites=mock_sats,
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(scenario=None),
        )
        fixed_names = [sat.name for sat in env.satellites]
        assert len(set(fixed_names)) == len(fixed_names)
        assert fixed_names == expected

    def test_get_obs(self):
        env = GeneralSatelliteTasking(
            satellites=[MagicMock(get_obs=MagicMock(return_value=i)) for i in range(3)],
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
        )
        assert env._get_obs() == (0, 1, 2)

    def test_get_obs_retasking_only(self):
        env = GeneralSatelliteTasking(
            satellites=[
                MagicMock(
                    get_obs=MagicMock(return_value=[i + 1]),
                    observation_space=spaces.Box(
                        -1e9, 1e9, shape=(1,), dtype=np.float32
                    ),
                    requires_retasking=(i == 1),
                )
                for i in range(3)
            ],
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
            generate_obs_retasking_only=True,
        )
        assert env._get_obs() == ([0], [2], [0])

    def test_get_info(self):
        mock_sats = [MagicMock(requires_retasking=True) for _ in range(3)]
        for i, sat in enumerate(mock_sats):
            sat.name = f"sat{i}"
        env = GeneralSatelliteTasking(
            satellites=mock_sats,
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
        )
        env.latest_step_duration = 10.0
        expected = {
            sat.name: {"requires_retasking": True} for i, sat in enumerate(mock_sats)
        }
        expected["d_ts"] = 10.0
        assert env._get_info() == expected

    def test_action_space(self):
        env = GeneralSatelliteTasking(
            satellites=[
                MagicMock(action_space=spaces.Discrete(i + 1)) for i in range(3)
            ],
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
        )
        assert env.action_space == spaces.Tuple(
            (spaces.Discrete(1), spaces.Discrete(2), spaces.Discrete(3))
        )

    def test_obs_space_no_sim(self):
        env = GeneralSatelliteTasking(
            satellites=[
                MagicMock(observation_space=spaces.Discrete(i + 1)) for i in range(3)
            ],
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
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
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
        )
        env.unwrapped.simulator = MagicMock()
        env.reset = MagicMock()
        assert env.observation_space == spaces.Tuple(
            (spaces.Discrete(1), spaces.Discrete(2), spaces.Discrete(3))
        )
        env.reset.assert_not_called()

    def test_step(self):
        mock_sats = [MagicMock() for _ in range(2)]
        env = GeneralSatelliteTasking(
            satellites=mock_sats,
            world_type=MagicMock(),
            scenario=MagicMock(),
            communicator=MagicMock(),
            rewarder=MagicMock(
                reward=MagicMock(return_value={sat.name: 12.5 for sat in mock_sats})
            ),
        )
        env._randomize_time_limit()
        mock_sats = env.satellites
        env.unwrapped.simulator = MagicMock(sim_time=101.0)
        _, reward, _, _, info = env.step((0, 10))
        mock_sats[0].set_action.assert_called_once_with(0)
        mock_sats[1].set_action.assert_called_once_with(10)
        env.unwrapped.simulator.run.assert_called_once()
        assert env.latest_step_duration == 0.0
        for sat in mock_sats:
            sat.data_store.update_from_logs.assert_called_once()
        assert reward == 25.0

    def test_step_bad_action(self):
        mock_sats = [MagicMock() for _ in range(2)]
        env = GeneralSatelliteTasking(
            satellites=mock_sats,
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(reward=MagicMock(return_value=25.0)),
        )
        env.unwrapped.simulator = MagicMock(sim_time=101.0)
        with pytest.raises(ValueError):
            env.step((0, 10, 20))
        with pytest.raises(ValueError):
            env.step((0,))

    @pytest.mark.parametrize("sat_death", [True, False])
    @pytest.mark.parametrize("timeout", [True, False])
    @pytest.mark.parametrize("terminate_on_time_limit", [True, False])
    @pytest.mark.parametrize("rewarder_terminated", [True, False])
    @pytest.mark.parametrize("rewarder_truncated", [True, False])
    def test_step_stopped(
        self,
        sat_death,
        timeout,
        terminate_on_time_limit,
        rewarder_terminated,
        rewarder_truncated,
    ):
        mock_sats = [MagicMock() for _ in range(2)]
        env = GeneralSatelliteTasking(
            satellites=mock_sats,
            world_type=MagicMock(),
            scenario=MagicMock(),
            communicator=MagicMock(),
            rewarder=MagicMock(
                reward=MagicMock(return_value={sat.name: 12.5 for sat in mock_sats}),
                is_terminated=MagicMock(return_value=rewarder_terminated),
                is_truncated=MagicMock(return_value=rewarder_truncated),
            ),
            terminate_on_time_limit=terminate_on_time_limit,
        )
        mock_sats = env.satellites
        env.unwrapped.simulator = MagicMock(sim_time=101.0)
        if timeout:
            env.time_limit = 100.0
        else:
            env.time_limit = 1000.0

        if sat_death:
            mock_sats[1].is_alive.return_value = False

        _, _, terminated, truncated, _ = env.step((0, 10))

        assert terminated == (
            sat_death
            or ((timeout or rewarder_truncated) and terminate_on_time_limit)
            or rewarder_terminated
        )
        assert truncated == timeout or rewarder_truncated

    @patch.multiple(Satellite, __abstractmethods__=set())
    def test_step_retask_needed(self, capfd):
        mock_sat = MagicMock()
        env = SatelliteTasking(
            satellite=[mock_sat],
            world_type=MagicMock(),
            scenario=MagicMock(),
            communicator=MagicMock(),
            rewarder=MagicMock(reward=MagicMock(return_value={mock_sat.name: 25.0})),
        )
        env._randomize_time_limit()
        env.unwrapped.simulator = MagicMock(sim_time=101.0)
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
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
        )
        env.unwrapped.simulator = MagicMock()
        env.close()
        assert not hasattr(env, "simulator")


class TestSatelliteTasking:
    @patch.multiple(
        Satellite,
        __abstractmethods__=set(),
        __init__=MagicMock(return_value=None),
    )
    def test_init(self):
        mock_sat = Satellite("sat", {})
        mock_sat.name = "sat"
        env = SatelliteTasking(
            satellite=mock_sat,
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
        )
        assert env.unwrapped.satellite == env.unwrapped.satellites[0]

    def test_init_multisat(self):
        with pytest.raises(ValueError):
            SatelliteTasking(
                satellite=[MagicMock(), MagicMock()],
                world_type=MagicMock(),
                scenario=MagicMock(),
                rewarder=MagicMock(),
            )

    @staticmethod
    def make_env():
        mock_sat = MagicMock()
        env = SatelliteTasking(
            satellite=[mock_sat],
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
        )
        return env, mock_sat

    def test_action_space(self):
        env, mock_sat = self.make_env()
        assert env.action_space == env.satellite.action_space

    @patch("bsk_rl.GeneralSatelliteTasking.observation_space")
    def test_observation_space(self, obs_patch):
        env, mock_sat = self.make_env()
        env.unwrapped.simulator = MagicMock()
        assert env.observation_space == env.satellite.observation_space

    @patch("bsk_rl.GeneralSatelliteTasking.step")
    def test_step(self, step_patch):
        env, mock_sat = self.make_env()
        env.step("action")
        step_patch.assert_called_once_with(["action"])

    def test_get_obs(self):
        env, mock_sat = self.make_env()
        assert env._get_obs() == env.satellite.get_obs()


class TestConstellationTasking:
    @patch("bsk_rl.gym.Simulator")
    @patch("bsk_rl.ConstellationTasking._get_obs")
    @patch("bsk_rl.ConstellationTasking._get_info")
    def test_reset(self, mock_sim, obs_fn, info_fn):
        mock_sat_1 = MagicMock()
        mock_sat_2 = MagicMock()
        mock_sat_1.sat_args_generator = {}
        mock_sat_2.sat_args_generator = {}
        mock_sat_1.name = mock_sat_2.name = "SomeSat"
        mock_data = MagicMock(scenario=None)
        env = ConstellationTasking(
            satellites=[mock_sat_1, mock_sat_2],
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=mock_data,
        )
        env.unwrapped.world_args_generator = {"utc_init": "a long time ago"}
        env.communicator = MagicMock()
        obs, info = env.reset()
        obs_fn.assert_called_once()
        info_fn.assert_called_once()

    @patch(
        "bsk_rl.GeneralSatelliteTasking._get_truncated",
        MagicMock(return_value=False),
    )
    def test_agents(self):
        env = ConstellationTasking(
            satellites=[MagicMock() for i in range(3)],
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
        )
        env._agents_last_compute_time = None
        env.simulator = MagicMock(sim_time=0.0)
        assert env.agents == [sat.name for sat in env.unwrapped.satellites]
        assert env.num_agents == 3
        assert env.possible_agents == [sat.name for sat in env.unwrapped.satellites]
        assert env.max_num_agents == 3

    @patch(
        "bsk_rl.GeneralSatelliteTasking._get_truncated",
        MagicMock(return_value=False),
    )
    def test_get_obs(self):
        env = ConstellationTasking(
            satellites=[MagicMock(get_obs=MagicMock(return_value=i)) for i in range(3)],
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
        )
        env._agents_last_compute_time = None
        env.simulator = MagicMock(sim_time=0.0)
        env.newly_dead = []
        assert env._get_obs() == {
            sat.name: i for i, sat in enumerate(env.unwrapped.satellites)
        }

    @patch(
        "bsk_rl.GeneralSatelliteTasking._get_truncated",
        MagicMock(return_value=False),
    )
    def test_get_info(self):
        mock_sats = [MagicMock(requires_retasking=True) for _ in range(3)]
        for i, sat in enumerate(mock_sats):
            sat.name = f"sat{i}"
        env = ConstellationTasking(
            satellites=mock_sats,
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
        )
        env._agents_last_compute_time = None
        env.simulator = MagicMock(sim_time=0.0)
        env.newly_dead = []
        env.latest_step_duration = 10.0
        expected = {
            f"sat{i}": {"requires_retasking": True, "d_ts": 10.0}
            for i, sat in enumerate(mock_sats)
        }
        expected["__common__"] = {
            "d_ts": 10.0,
        }
        assert env._get_info() == expected

    def test_action_spaces(self):
        env = ConstellationTasking(
            satellites=[
                MagicMock(action_space=spaces.Discrete(i + 1)) for i in range(3)
            ],
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
        )
        assert env.action_spaces == {
            env.unwrapped.satellites[0].name: spaces.Discrete(1),
            env.unwrapped.satellites[1].name: spaces.Discrete(2),
            env.unwrapped.satellites[2].name: spaces.Discrete(3),
        }

    def test_obs_spaces(self):
        env = ConstellationTasking(
            satellites=[
                MagicMock(observation_space=spaces.Box(low=0, high=i + 1, shape=(1,)))
                for i in range(3)
            ],
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
        )
        env.unwrapped.simulator = MagicMock()
        env.reset = MagicMock()
        assert env.observation_spaces == {
            env.unwrapped.satellites[0].name: spaces.Box(low=0, high=1, shape=(1,)),
            env.unwrapped.satellites[1].name: spaces.Box(low=0, high=2, shape=(1,)),
            env.unwrapped.satellites[2].name: spaces.Box(low=0, high=3, shape=(1,)),
        }

    @patch(
        "bsk_rl.GeneralSatelliteTasking._get_truncated",
        MagicMock(return_value=False),
    )
    def test_get_reward(self):
        env = ConstellationTasking(
            satellites=[
                MagicMock(is_alive=MagicMock(return_value=False)) for i in range(3)
            ],
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
            failure_penalty=-20.0,
        )
        env._agents_last_compute_time = None
        env.simulator = MagicMock(sim_time=0.0)
        env.newly_dead = [sat.name for sat in env.unwrapped.satellites]
        env.reward_dict = {
            sat.name: 10.0 for i, sat in enumerate(env.unwrapped.satellites)
        }
        assert env._get_reward() == {
            sat.name: -10.0 for i, sat in enumerate(env.unwrapped.satellites)
        }

    @patch(
        "bsk_rl.GeneralSatelliteTasking._get_truncated",
        MagicMock(return_value=False),
    )
    def test_get_reward_missing_sat(self):
        env = ConstellationTasking(
            satellites=[
                MagicMock(is_alive=MagicMock(return_value=False)) for i in range(3)
            ],
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
            failure_penalty=-20.0,
        )
        env._agents_last_compute_time = None
        env.simulator = MagicMock(sim_time=0.0)
        env.newly_dead = [sat.name for sat in env.unwrapped.satellites]
        env.reward_dict = {env.unwrapped.satellites[0].name: 10.0}
        assert env._get_reward() == {
            env.unwrapped.satellites[0].name: -10.0,
            env.unwrapped.satellites[1].name: -20.0,
            env.unwrapped.satellites[2].name: -20.0,
        }

    @pytest.mark.parametrize("timeout", [False, True])
    @pytest.mark.parametrize("terminate_on_time_limit", [False, True])
    def test_get_terminated(self, timeout, terminate_on_time_limit):
        mock_sats = [
            MagicMock(
                is_alive=MagicMock(return_value=True if i != 0 else False),
            )
            for i in range(3)
        ]
        for i, sat in enumerate(mock_sats):
            sat.name = f"sat{i}"
        env = ConstellationTasking(
            satellites=mock_sats,
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(
                is_terminated=MagicMock(return_value=False),
                is_truncated=MagicMock(return_value=False),
            ),
            terminate_on_time_limit=terminate_on_time_limit,
            time_limit=100,
        )
        env._randomize_time_limit()
        env._agents_last_compute_time = None
        env.simulator = MagicMock(sim_time=0.0)
        env.unwrapped.simulator = MagicMock(sim_time=101 if timeout else 99)

        if not timeout or not terminate_on_time_limit:
            env.newly_dead = [sat.name for sat in env.unwrapped.satellites]
            assert env._get_terminated() == {
                env.unwrapped.satellites[0].name: True,
                env.unwrapped.satellites[1].name: False,
                env.unwrapped.satellites[2].name: False,
            }
        else:
            env.newly_dead = [sat.name for sat in env.unwrapped.satellites]
            assert env._get_terminated() == {
                env.unwrapped.satellites[0].name: True,
                env.unwrapped.satellites[1].name: True,
                env.unwrapped.satellites[2].name: True,
            }

    @pytest.mark.parametrize("time", [99, 101])
    def test_get_truncated(self, time):
        mock_sats = [MagicMock() for _ in range(3)]
        for i, sat in enumerate(mock_sats):
            sat.name = f"sat{i}"
        env = ConstellationTasking(
            satellites=mock_sats,
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(
                is_terminated=MagicMock(return_value=False),
                is_truncated=MagicMock(return_value=False),
            ),
            time_limit=100,
        )
        env._randomize_time_limit()
        env._agents_last_compute_time = None
        env.simulator = MagicMock(sim_time=0.0)
        env.unwrapped.simulator = MagicMock(sim_time=time)
        env.newly_dead = (
            [sat.name for sat in env.unwrapped.satellites] if time >= 100 else []
        )
        assert env._get_truncated() == {
            env.unwrapped.satellites[0].name: time >= 100,
            env.unwrapped.satellites[1].name: time >= 100,
            env.unwrapped.satellites[2].name: time >= 100,
        }

    def test_time_limit_generator(self):
        mock_sats = [MagicMock() for _ in range(3)]
        for i, sat in enumerate(mock_sats):
            sat.name = f"sat{i}"
        env = ConstellationTasking(
            satellites=mock_sats,
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
            time_limit=lambda: np.random.uniform(10, 20),
        )
        env._randomize_time_limit()
        env._agents_last_compute_time = None
        env.simulator = MagicMock(sim_time=0.0)
        env.unwrapped.simulator = MagicMock(sim_time=100.0)
        env.newly_dead = [sat.name for sat in env.unwrapped.satellites]
        assert env._get_truncated() == {
            env.unwrapped.satellites[0].name: True,
            env.unwrapped.satellites[1].name: True,
            env.unwrapped.satellites[2].name: True,
        }

    def test_close(self):
        env = ConstellationTasking(
            satellites=[MagicMock()],
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
        )
        env.unwrapped.simulator = MagicMock()
        env.close()
        assert not hasattr(env, "simulator")

    @patch(
        "bsk_rl.GeneralSatelliteTasking._get_truncated",
        MagicMock(return_value=False),
    )
    def test_dead(self):
        mock_sats = [MagicMock() for _ in range(3)]
        for i, sat in enumerate(mock_sats):
            sat.name = f"sat{i}"
        env = ConstellationTasking(
            satellites=mock_sats,
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
        )
        env._agents_last_compute_time = None
        env.simulator = MagicMock(sim_time=0.0)
        env.unwrapped.satellites[1].is_alive = MagicMock(return_value=False)
        env.unwrapped.satellites[2].is_alive = MagicMock(return_value=False)
        env.newly_dead = [env.unwrapped.satellites[2].name]
        assert env.previously_dead == [env.unwrapped.satellites[1].name]
        assert env.agents == [env.unwrapped.satellites[0].name]
        assert env.possible_agents == [sat.name for sat in env.unwrapped.satellites]

    mst = "bsk_rl.ConstellationTasking."

    @patch(
        "bsk_rl.GeneralSatelliteTasking._get_truncated",
        MagicMock(return_value=False),
    )
    @patch(mst + "_get_obs", MagicMock())
    @patch(mst + "_get_reward", MagicMock())
    @patch(mst + "_get_terminated", MagicMock())
    @patch(mst + "_get_truncated", MagicMock())
    @patch(mst + "_get_info", MagicMock())
    @patch(
        "bsk_rl.GeneralSatelliteTasking._step",
        MagicMock(),
    )
    def test_step(self):
        mock_sats = [MagicMock(is_alive=MagicMock(return_value=True)) for _ in range(3)]
        for i, sat in enumerate(mock_sats):
            sat.name = f"sat{i}"
        env = ConstellationTasking(
            satellites=mock_sats,
            world_type=MagicMock(),
            scenario=MagicMock(),
            rewarder=MagicMock(),
        )
        env._agents_last_compute_time = None
        env.simulator = MagicMock(sim_time=0.0)

        def kill_sat_2():
            env.unwrapped.satellites[2].is_alive.return_value = False

        def side_effect():
            kill_sat_2()
            env.simulator.sim_time += 1.0

        env._step.side_effect = lambda _: side_effect()
        env.unwrapped.satellites[1].is_alive.return_value = False
        env.step(
            {
                env.unwrapped.satellites[0].name: 0,
                env.unwrapped.satellites[2].name: 2,
            }
        )

        env._step.assert_called_with([0, None, 2])
        assert env.newly_dead == [env.unwrapped.satellites[2].name]
