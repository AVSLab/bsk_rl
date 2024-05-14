import gymnasium as gym
import numpy as np
from Basilisk.utilities import macros as mc
from gymnasium import spaces

from bsk_rl.scene.agile_eos.bsk_sim import AgileEOSSimulator


class AgileEOS(gym.Env):
    """
    This Gymnasium environment is designed to simulate an agile EOS scheduling problem
    in which a satellite in low-Earth orbit attempts to maximize the number of targets
    imaged and downlinked while avoiding resource constraint violations. Resource
    constraint include:

    1. Power: The spacecraft must keep its battery charge above zero
    2. Reaction Wheel Saturation: The spacecraft must keep its reaction wheels within
        their speed limits
    3. Data Buffer: The spacecraft must keep its data buffer from overflowing (i.e.
        exceeding or meeting the maximum buffer size)

    The spacecraft must decide between pointing at any one of J number of ground targets
    for imaging, pointing at the sun to charge, desaturating reaction wheels, or
    downlinking data. This is referred to as the AgileEOS environment.

    Action Space (Discrete):
    0 - Charging mode
    1 - Downlink mode
    2 - Desat mode
    3:J+3 - Image target j

    Observation Space:
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
    """

    def __init__(self, failure_penalty=1, image_component=0.1, downlink_component=0.9):
        self.__version__ = "0.0.1"
        print("AgileEOS Environment - Version {}".format(self.__version__))

        # General variables defining the environment
        self.max_length = float(270.0)  # Specify the maximum number of minutes
        self.max_steps = 45
        self.render = False

        #   Tell the environment that it doesn't have a sim attribute...
        self.simulator_init = 0
        self.simulator = None
        self.reward_total = 0

        # Set initial conditions to none (gets assigned in reset)
        self.initial_conditions = None

        # Set the dynRate for the env, which is passed into the simulator
        self.dynRate = 1.0
        self.fswRate = 1.0

        # Set up options, constants for this environment
        self.step_duration = 6 * 60.0  # seconds, tune as desired
        self.reward_mult = 1.0

        # Set the reward components
        self.failure_penalty = failure_penalty
        self.image_component = image_component
        self.downlink_component = downlink_component

        # Set up targets and action, observation spaces
        self.n_targets = (
            3  # Assumes 100 targets, three other actions (charge, desat, downlink)
        )
        self.target_tuple_size = 4
        self.action_space = spaces.Discrete(3 + self.n_targets)
        low = -1e16
        high = 1e16
        self.observation_space = spaces.Box(
            low,
            high,
            shape=(22 + self.n_targets * self.target_tuple_size,),
            dtype=np.float64,
        )
        self.obs = np.zeros(22 + self.n_targets * self.target_tuple_size)
        self.obs_full = np.zeros(22 + self.n_targets * self.target_tuple_size)

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []
        self.curr_step = 0
        self.episode_over = False
        self.failure = False

        self.dvc_cmd = 0
        self.act_hold = None

        self.return_obs = True

    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """

        # If the simTime in minutes is greater than the planning interval in minutes,
        # end the sim
        if (self.simulator.simTime / 60.0) >= self.max_length:
            print("End of simulation reached", self.simulator.simTime / 60)
            self.episode_over = True

        downlinked, imaged, info = self._take_action(action)

        # If we want to return observations, do the following
        if self.return_obs:
            reward = 0
            ob = self._get_state()
            self.prev_dvc_cmd = self.dvc_cmd
            self.dvc_cmd = (
                self.simulator.simpleInsControlConfig.deviceCmdOutMsg.read().deviceCmd
            )

            #   If the wheel speeds get too large, end the episode.
            if any(
                speeds > self.simulator.initial_conditions["maxSpeed"] * mc.RPM
                for speeds in self.obs_full[8:11]
            ):
                self.episode_over = True
                self.failure = True
                print(
                    "Died from wheel explosion. RPMs were: "
                    + str(self.obs_full[8:11] / mc.RPM)
                    + ", limit is "
                    + str(self.simulator.initial_conditions["maxSpeed"])
                    + ", body rate was "
                    + str(self.obs_full[7])
                    + ", action taken was "
                    + str(action)
                    + ", env step "
                    + str(self.curr_step)
                )
            #   If we run out of power, end the episode.
            elif self.obs_full[11] == 0:
                self.failure = True
                self.episode_over = True
                print(
                    "Ran out of power. Battery level at: "
                    + str(self.obs_full[11])
                    + ", env step "
                    + str(self.curr_step)
                    + ", action taken was "
                    + str(action)
                )
            #   If we overflow the buffer, end the episode.
            elif self.obs_full[13] >= self.simulator.storageUnit.storageCapacity:
                self.failure = True
                self.episode_over = True
                print(
                    "Data buffer overflow. Data storage level at:"
                    + str(self.obs_full[13])
                    + ", env step, "
                    + str(self.curr_step)
                    + ", action taken was "
                    + str(action)
                )
            elif self.sim_over:
                self.episode_over = True
                print("Orbit decayed - no penalty, but this one is over.")
            else:
                self.failure = False

            reward = self._get_reward(downlinked, imaged)
            self.reward_total += reward

        # Otherwise, return nothing
        else:
            ob = []
            reward = 0

        num_imaged, num_downlinked = self._get_imaged_downlinked()

        # Update the info with the metrics
        info["metrics"] = {
            "downlinked": self.simulator.total_downlinked,
            "sim_length": self.simulator.simTime / 60,
            "total_access": self.simulator.total_access,
            "utilized_access": self.simulator.utilized_access,
            "num_imaged": num_imaged,
            "num_downlinked": num_downlinked,
        }

        self.act_hold = action

        self.curr_step += 1
        return ob.flatten(), reward, self.episode_over, False, info

    def _take_action(self, action):
        """
        Interfaces with the simulator to
        :param action:
        :return:
        """

        self.action_episode_memory[self.curr_episode].append(action)
        (
            self.obs,
            self.sim_over,
            self.obs_full,
            downlinked,
            imaged,
            info,
        ) = self.simulator.run_sim(action, self.return_obs)

        return downlinked, imaged, info

    def _get_reward(self, downlinked, imaged):
        """
        Reward is based on the total number of imaged and downlinked targets, failure i
        f it occurs
        """
        reward = 0
        if self.failure:
            reward = -self.failure_penalty
        else:
            for idx in downlinked:
                reward += (
                    self.downlink_component
                    / float(
                        self.simulator.initial_conditions.get("targetPriorities")[idx]
                    )
                    / self.max_steps
                )
            for idx in imaged:
                reward += (
                    self.image_component
                    / float(
                        self.simulator.initial_conditions.get("targetPriorities")[idx]
                    )
                    / self.max_steps
                )

        return reward

    def reset(self, seed=None, options=None):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        super().reset(seed=seed)
        self.action_episode_memory.append([])
        self.episode_over = False
        self.failure = False
        self.curr_step = 0
        self.reward_total = 0

        if self.simulator is not None:
            del self.simulator

        if options is not None:
            if "initial_conditions" in options:
                self.initial_conditions = options["initial_conditions"]

        # Create the simulator
        self.simulator = AgileEOSSimulator(
            self.dynRate,
            self.fswRate,
            self.step_duration,
            initial_conditions=self.initial_conditions,
            render=self.render,
            n_targets=self.n_targets,
            max_length=self.max_length,
            target_tuple_size=self.target_tuple_size,
        )

        # Extract initial conditions from instantiation of simulator
        self.initial_conditions = self.simulator.initial_conditions
        self.simulator.max_steps = self.max_steps
        self.simulator_init = 1

        return self.simulator.obs.flatten(), self.simulator.info

    def _render(self, mode="human", close=False):
        return

    def _get_state(self):
        """Return the non-normalized observation to the environment"""

        return self.simulator.obs

    def _get_imaged_downlinked(self):
        num_imaged = 0
        num_downlinked = 0
        for idx in range(len(self.simulator.imaged_targets)):
            if self.simulator.imaged_targets[idx] >= 1.0:
                num_imaged += 1

            if self.simulator.downlinked_targets[idx] >= 1.0:
                num_downlinked += 1

        return num_imaged, num_downlinked

    def update_tgt_count(self, n_targets):
        self.n_targets = n_targets
        self.action_space = spaces.Discrete(3 + self.n_targets)
        low = -1e16
        high = 1e16
        self.observation_space = spaces.Box(
            low, high, shape=(22 + self.n_targets * self.target_tuple_size,)
        )
        self.obs = np.zeros(22 + self.n_targets * self.target_tuple_size)
        self.obs_full = np.zeros(22 + self.n_targets * self.target_tuple_size)


if __name__ == "__main__":
    env = gym.make("AgileEOS-v0")

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
