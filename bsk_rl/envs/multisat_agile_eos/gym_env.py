import inspect

import gymnasium as gym
import numpy as np
from Basilisk.utilities import macros as mc
from gymnasium import spaces

from bsk_rl.envs.multisat_agile_eos import env_settings
from bsk_rl.envs.multisat_agile_eos.bsk_sim import MultiSatAgileEOSSimulator

gym.utils.passive_env_checker.logger.setLevel(
    40
)  # Disable annoying gym warnings, we know what's good for us


class MultiSatAgileEOS(gym.Env):
    """
    This Gymnasium environment is designed to simulate the multi-satellite agile EOS
    scheduling problem in which K satellites in a low-Earth orbit Walker-delta formation
    attempt to maximize the number of targets imaged and downlinked while avoiding
    resource constraint violations. Satellites update their local target lists through
    communication with one another. Resource constraint violations include:

    1. Power: The spacecraft must keep its battery charge above zero
    2. Reaction Wheel Saturation: The spacecraft must keep its reaction wheels within
        their speed limits
    3. Data Buffer: The spacecraft must keep its data buffer from overflowing (i.e.
        exceeding or meeting the maximum buffer size)

    Each spacecraft must decide between pointing at any one of J number of ground
    targets for imaging, pointing at the sun to charge, desaturating reaction wheels,
    or downlinking data. This is referred to as the MultiSatAgileEOS environment.

    Action Space (MultiDiscrete):
        For each spacecraft:
            0 - Charging mode
            1 - Downlink mode
            2 - Desat mode
            3:J+3 - Image target j

    Observation Space:
        For each spacecraft:
        ECEF position and velocity - indices 0-5
        Attitude error and attitude rate - indices 6-7
        Reaction wheel speeds - indices 8-11
        Battery charge - indices 12
        Eclipse indicator - indices 13
        Stored data onboard spacecraft - indices 14
        Data transmitted over interval - indices 15
        Amount of time ground stations were accessible (s) - 16-22
        Target Tuples (4 values each) - priority and Hill frame pos

    Reward Function:
        r = +A/priority for for each tgt downlinked for first time
        r = +B/priority for for each tgt imaged for first time
        r = -C if failure

        No reward is given for imaging or downlinking the same target twice
    """

    def __init__(self):
        self.__version__ = "0.0.1"
        print(
            "Basilisk MultiSatAgileEOS Environment - Version {}".format(
                self.__version__
            )
        )

        # General variables defining the environment
        self.max_length = float(270.0)  # Maximum number of minutes
        self.max_steps = 45  # Maximum number of steps
        self.render = False

        #   Tell the environment that it doesn't have a sim attribute...
        self.simulator_init = 0
        self.simulator = None
        self.reward_total = 0

        # Set initial conditions to none (gets assigned in reset)
        self.initial_conditions = None

        # Set the dynRate for the env, which is passed into the simulator
        self.envRate = 1.0
        self.dynRate = 1.0
        self.fswRate = 1.0

        # Set up options, constants for this environment
        self.step_duration = (
            6 * 60.0
        )  # seconds, tune as desired; can be updated mid-simulation
        self.reward_mult = 1.0
        self.failure_penalty = 1
        self.image_reward_fn = lambda priority: 0.1 / priority / 45
        self.downlink_reward_fn = lambda priority: 1.0 / priority / 45
        self.comm_method = "free"
        assert self.comm_method in [
            "free",
            "los",
            "los-multi",
            "none",
        ], f"comm_method {self.comm_method} not valid!"

        # Set the number of spacecraft
        self._n_spacecraft = 2

        # Keep track of rewarded targets
        self.rewarded_targets = []
        self.target_duplication = 0
        self.rewarded_downlinks = []
        self.downlink_duplication = 0

        # Set up targets and action, observation spaces
        self._n_targets = (
            3  # Assumes 3 targets, three other actions (charge, desat, downlink)
        )
        self.target_tuple_size = 4  # Number of elements in target definition
        self.action_space = spaces.MultiDiscrete(
            [3 + self.n_targets] * self.n_spacecraft
        )
        low = -1e16
        high = 1e16
        self.observation_space = spaces.Box(
            low,
            high,
            shape=(self.n_spacecraft, 22 + self.n_targets * self.target_tuple_size),
        )
        self.obs = np.zeros(
            [self.n_spacecraft, 22 + self.n_targets * self.target_tuple_size]
        )
        self.obs_full = np.zeros(
            [self.n_spacecraft, 22 + self.n_targets * self.target_tuple_size]
        )

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []
        self.curr_step = 0
        self.episode_over = False
        self.failure = np.zeros(self.n_spacecraft)

        # Parameters not set yet
        self.params_set = False

    @property
    def n_spacecraft(self):
        return self._n_spacecraft

    @n_spacecraft.setter
    def n_spacecraft(self, n_spacecraft):
        self._n_spacecraft = n_spacecraft
        self.update_spaces()

    @property
    def n_targets(self):
        return self._n_targets

    @n_targets.setter
    def n_targets(self, n_targets):
        self._n_targets = n_targets
        self.update_spaces()

    def sim_attrs(self):
        """
        Creates keyword arguments for instantiating the simulator. If an attribute is
        present that matches a simulatormkeyword, pass it in.
        """
        attrs = inspect.signature(MultiSatAgileEOSSimulator.__init__).parameters.keys()
        return {
            attr: getattr(self, attr)
            for attr in attrs
            if attr != "self" and hasattr(self, attr)
        }

    def set_env_params(self, **kwargs):
        """
        Set arbitrary environment variables. Can be used to override simulator defaults.
        """
        for arg, val in kwargs.items():
            setattr(self, arg, val)

    def set_params(
        self,
        n_spacecraft,
        n_planes,
        rel_phasing,
        inc,
        global_tgts,
        priorities,
        comm_method,
        **kwargs,
    ):
        """
        Updates the constellation and target parameters in the environment. Environment
        must be reset afterwards. Reset is not called within this function so user can
        choose between reset and reset_init.
        """
        # Set these parameters in the environment
        self.set_env_params(
            n_spacecraft=n_spacecraft,
            n_planes=n_planes,
            rel_phasing=rel_phasing,
            inc=inc,
            global_tgts=global_tgts,
            priorities=priorities,
            comm_method=comm_method,
            params_set=True,
            **kwargs,
        )

    def update_spaces(self):
        """
        Updates the size of action and observation spaces and preallocations.
        """
        self.action_space = spaces.MultiDiscrete(
            [3 + self.n_targets] * self.n_spacecraft
        )
        low = -1e16
        high = 1e16
        self.observation_space = spaces.Box(
            low,
            high,
            shape=(self.n_spacecraft, 22 + self.n_targets * self.target_tuple_size),
        )
        self.obs = np.zeros(
            [self.n_spacecraft, 22 + self.n_targets * self.target_tuple_size]
        )
        self.obs_full = np.zeros(
            [self.n_spacecraft, 22 + self.n_targets * self.target_tuple_size]
        )
        self.failure = np.zeros(self.n_spacecraft)

    def _seed(self):
        np.random.seed()
        return

    def step(self, action, return_obs=True):
        """
        :param action: 1 x n_spacecraft list of actions
        :param return_obs: whether or not to return observations
        :return ob: n_spacecraft x dim_obs nparray of observations
        :return reward: 1 x (n_spacecraft + 1) nparray of rewards, last index is global
                reward, other are local rewards
        :return episode over: True/False if episode is over
        :return info: dictionary of info for debugging purposes
        """
        # Check if the simulator has been initialized or not
        if self.simulator_init == 0:
            self.simulator = MultiSatAgileEOSSimulator(**self.sim_attrs())
            self.obs = self.simulator.obs
            self.simulator_init = 1

        # If the simTime in minutes is greater than the planning interval in minutes,
        # end the sim
        if (self.simulator.simTime / 60.0) >= self.max_length:
            print("End of simulation reached", self.simulator.simTime / 60)
            self.episode_over = True

        # Save the previous observations
        prev_ob = self.obs_full

        # Take the action
        downlinked, imaged = self._take_action(action, return_obs)

        if return_obs:
            # Loop through each spacecraft's observations
            for idx in range(0, self.n_spacecraft):
                observation = self.obs_full[idx, :]
                #   If the wheel speeds get too large, end the episode.
                if any(
                    speeds
                    > self.simulator.initial_conditions[str(idx)]["maxSpeed"] * mc.RPM
                    for speeds in observation[8:11]
                ):
                    self.episode_over = True
                    self.failure[0, idx] = True
                    print(
                        "Spacecraft "
                        + str(idx)
                        + " died from wheel explosion. RPMs were norm:"
                        + str(observation[8:11])
                        + ", limit is "
                        + str(6000 * mc.RPM)
                        + ", body rate was "
                        + str(observation[7])
                        + "action taken was "
                        + str(action[idx])
                        + ", env step"
                        + str(self.curr_step)
                    )
                    print(
                        "Prior state was RPM:"
                        + str(prev_ob[idx, 8:11])
                        + " . body rate was: "
                        + str(prev_ob[idx, 7])
                    )
                #   If we run out of power, end the episode.
                elif observation[11] == 0:
                    self.failure[0, idx] = True
                    self.episode_over = True
                    print(
                        "Spacecraft "
                        + str(idx)
                        + " ran out of power. Battery level at: "
                        + str(observation[11])
                        + ", env step "
                        + str(self.curr_step)
                        + ", action taken was "
                        + str(action[idx])
                    )
                #   If we overflow the buffer, end the episode.
                elif (
                    observation[13]
                    >= self.simulator.initial_conditions[str(idx)][
                        "dataStorageCapacity"
                    ]
                    and action[idx] >= 3
                ):
                    self.failure[0, idx] = True
                    self.episode_over = True
                    print(
                        "Spacecraft "
                        + str(idx)
                        + " data buffer overflow. Data storage level at:"
                        + str(observation[13])
                        + ", env step, "
                        + str(self.curr_step)
                        + ", action taken was "
                        + str(action[idx])
                    )
                elif self.sim_over:
                    self.episode_over = True
                    print(
                        "Spacecraft "
                        + str(idx)
                        + " orbit decayed - no penalty, but this one is over."
                    )
                else:
                    self.failure[0, idx] = False

            if self.episode_over:
                info = {
                    "episode": {"r": self.reward_total, "l": self.curr_step},
                    "obs": self._get_state(),
                }
            else:
                info = {"obs": self._get_state()}

            reward = self._get_reward(downlinked, imaged)
            self.reward_total = np.add(self.reward_total, reward)
            self.curr_step += 1

        return (
            self._get_state(),
            reward,
            self.episode_over,
            [False] * self.n_spacecraft,
            info,
        )

    def _take_action(self, action, return_obs=True):
        """
        :param action: 1 x n_spacecraft list of actions
        :param return_obs: whether or not to return observation
        :return downlinked: n_spacecraft x m list of downlinked targets
        :return imaged: n_spacecraft x p list of imaged targets

        Actions:
        0 - Charging Mode
        1 - Desaturation Mode
        2 - Downlinked Mode
        >= 3 - Imaging Mode
        """
        self.action_episode_memory.append(action)
        (
            self.obs,
            self.sim_over,
            self.obs_full,
            downlinked,
            imaged,
        ) = self.simulator.run_sim(action, return_obs)

        return downlinked, imaged

    def _get_reward(self, downlinked, imaged):
        """
        :param downlinked: n_spacecraft x m list of downlinked targets
        :param imaged: n_spacecraft x p list of imaged targets
        :return reward: 1 x (n_spacecraft + 1) nparray of rewards, last index is global
        reward, other are local rewards
        """
        reward = np.zeros((1, self.n_spacecraft + 1))
        for i in range(0, self.n_spacecraft):
            if self.failure[0, i]:
                reward[0, i] = -self.failure_penalty
                reward[
                    0, self.n_spacecraft
                ] -= self.failure_penalty  # Add failure to global reward
            else:
                for data in downlinked[i]:
                    if data not in self.rewarded_downlinks:
                        self.rewarded_downlinks.append(data)
                        r = self.downlink_reward_fn(
                            float(
                                self.simulator.initial_conditions["env_params"][
                                    "globalPriorities"
                                ][data]
                            )
                        )
                        # Add to global reward
                        reward[0, i] += r
                        reward[0, self.n_spacecraft] += r
                    else:
                        self.downlink_duplication += 1
                for image in imaged[i]:
                    if image not in self.rewarded_targets:
                        self.rewarded_targets.append(image)
                        r = self.image_reward_fn(
                            float(
                                self.simulator.initial_conditions["env_params"][
                                    "globalPriorities"
                                ][image]
                            )
                        )
                        # Add to global reward
                        reward[0, i] += r
                        reward[0, self.n_spacecraft] += r
                    else:
                        self.target_duplication += 1

        return reward

    def reset(self, seed=None, options=None):
        """
        Reset the state of the environment and returns an initial observation.
        :return ob: n_spacecraft x dim_obs nparray of observations
        """
        if self.initial_conditions is not None:
            del self.simulator

        if options is not None:
            if "initial_conditions" in options:
                self.initial_conditions = options["initial_conditions"]

        # Check if params were set
        if not self.params_set:
            self.environment_settings = env_settings.env_settings()
            if self.initial_conditions is not None:
                tgt_pos = self.initial_conditions["env_params"]["globalTargets"]
                tgt_priority = self.initial_conditions["env_params"]["globalPriorities"]
            else:
                (
                    tgt_pos,
                    tgt_priority,
                ) = self.environment_settings.generate_global_targets()
            self.set_params(
                n_spacecraft=self.environment_settings.n_spacecraft,
                n_planes=self.environment_settings.n_planes,
                rel_phasing=self.environment_settings.rel_phasing,
                inc=self.environment_settings.inc,
                global_tgts=tgt_pos,
                priorities=tgt_priority,
                comm_method=self.environment_settings.comm_method,
            )
            self.update_spaces()

        self.action_episode_memory = []
        self.episode_over = False
        self.failure = np.zeros((1, self.n_spacecraft))
        self.curr_step = 0
        self.reward_total = 0

        # Create the simulator
        self.simulator = MultiSatAgileEOSSimulator(**self.sim_attrs())

        # Extract initial conditions from instantiation of simulator
        self.initial_conditions = self.simulator.initial_conditions
        self.simulator_init = 1
        self._seed()
        self.obs = self.simulator.obs

        return self.obs

    def _render(self, mode="human", close=False):
        return

    def _get_state(self):
        """
        Return the non-normalized observation to the environment
        :return ob: n_spacecraft x dim_obs np.array of observations
        """
        return self.obs


if __name__ == "__main__":
    env = gym.make("MultiSatAgileEOS-v0")

    env.reset()

    reward_sum = 0
    for idx in range(0, env.max_steps):
        action = env.action_space.sample()
        ob, reward, episode_over, truncated, info = env.step(action)
        reward_sum += reward

        if episode_over:
            print("Episode over at step " + str(idx))
            break

    print("Reward total: " + str(reward_sum))
