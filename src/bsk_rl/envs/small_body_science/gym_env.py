import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bsk_rl.envs.small_body_science.bsk_sim import SmallBodyScienceSimulator


class SmallBodyScience(gym.Env):
    """
    Small body gym environment where an agent can transition between different
    waypoints defined in the sun anti-momentum frame to image candidate landing sites
    or collect spectroscopy map data while avoiding resource constraint violations.

    Resource constraint violations include:
        - Fuel
        - Power
        - Data storage
        - Collision with the body (not necessarilly a resource, but considered a
            failure condition)

    Action Space (Discrete):
    0 - Charging Mode
    1 - 8 - Transition to waypoint 1-8
    9 - Map
    10 - Downlink
    11 - Image

    Observation Space (Box):
    0-2: Hill-frame position
    3-5: Hill-frame velocity
    6: Eclipse
    7: Data buffer storage
    8: Battery level
    9: dV consumed
    10: Downlink availability
    11-13: Current waypoint
    14-16: Last waypoint
    17: Imaged targets
    18: Downlinked targets
    19-21: Next closest unimaged target position in Hill frame
    22-30: Map regions collected

    Reward Function:
    r = +A each tgt downlinked for first time
    r = +B for each tgt imaged for first time
    r = +C for each map region downlinked for first time
    r = +D for each map region collected for first time
    r = -E for failure
    """

    def __init__(
        self,
        failure_penalty=1,
        target_component=0.25,
        target_downlink_component=0.25,
        map_component=0.25,
        map_downlink_component=0.25,
    ):
        self.__version__ = "0.0.1"
        print("Basilisk Small Body Science Sim - Version {}".format(self.__version__))

        # Set the fidelity of the environment
        self.fidelity = "high"

        # General variables defining the environment
        self.max_length = 10000.0  # Specify the maximum number of minutes
        self.max_steps = 200
        self.render = False

        #   Tell the environment that it doesn't have a sim attribute...
        self.simulator_init = 0
        self.simulator = None
        self.reward_total = 0

        # Set initial conditions to none (gets assigned in reset)
        self.initial_conditions = None

        # Set the dynRate for the env, which is passed into the simulator
        if self.fidelity == "high":
            self.dynRate = 2.0
            self.fswRate = 2.0
            self.mapRate = 180.0
        elif self.fidelity == "low":
            self.dynRate = 1.0
            self.fswRate = 1.0
            self.mapRate = 180.0
        else:
            print(
                "Invalid fidelity, choose either low or high. Selected = ",
                self.fidelity,
            )

        # Set up options, constants for this environment
        self.step_duration = 10000.0  # seconds, tune as desired
        self.reward_mult = 1.0

        self.failure_penalty = failure_penalty
        self.target_component = target_component
        self.target_downlink_component = target_downlink_component
        self.map_component = map_component
        self.map_downlink_component = map_downlink_component

        # Define number of image targets
        self.n_targets = 10

        # Define number of map targets
        self.n_maps = 3
        self.n_map_points = 500  # Number of grid points per map

        self.phi_c = None
        self.lambda_c = None

        # Define observation space
        self.n_states = 31
        self.obs = np.zeros(self.n_states)
        self.obs_full = np.zeros(self.n_states)
        self.observation_space = spaces.Box(-1e16, 1e16, shape=(self.n_states,))

        # Define the action space
        self.action_space = spaces.Discrete(12)

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []
        self.curr_step = 0
        self.episode_over = False
        self.failure = False

        self.return_obs = True

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, truncated, info : tuple
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
        """

        # If the simTime in minutes is greater than the planning interval in minutes,
        # end the sim
        if (self.simulator.simTime / 60.0) >= self.max_length:
            print("End of simulation reached", self.simulator.simTime / 60)
            self.episode_over = True

        downlinked_images, downlinked_maps, imaged, mapped = self._take_action(action)

        # If we want to return observations, do the following
        if self.return_obs:
            ob = self._get_state()

            if any(self.simulator.powerLevel == 0):
                self.failure = True
                self.episode_over = True
                print(
                    "Ran out of power. Battery level at: "
                    + str(self.simulator.powerLevel[-1])
                    + ", env step "
                    + str(self.curr_step)
                    + ", action taken was "
                    + str(action)
                )
            # If we overflow the buffer, end the episode.
            elif (
                self.simulator.storageLevel
                >= self.simulator.dataStorageUnit.storageCapacity
            ):
                self.failure = True
                self.episode_over = True
                print(
                    "Data buffer overflow. Data storage level at:"
                    + str(self.simulator.storageLevel)
                    + ", env step, "
                    + str(self.curr_step)
                    + ", action taken was "
                    + str(action)
                )
            elif self.simulator.dV >= self.simulator.initial_conditions["max_dV"]:
                self.failure = True
                self.episode_over = True
                print(
                    "Ran out of fuel. Total dV consumed: "
                    + str(self.simulator.dV)
                    + ", env step "
                    + str(self.curr_step)
                )
            elif self.simulator.collision:
                self.failure = True
                self.episode_over = True
                print("Collided with the body, env step " + str(self.curr_step))
            else:
                self.failure = False

            reward = self._get_reward(
                downlinked_images, downlinked_maps, imaged, mapped
            )
            self.reward_total += reward

        # Otherwise, return nothing
        else:
            ob = []
            reward = 0

        info = {}
        info["metrics"] = {
            "imaged_targets": self.simulator.imaged_targets,
            "downlinked_targets": self.simulator.downlinked_targets,
            "imaged_maps": self.simulator.imaged_maps,
            "downlinked_maps": self.simulator.downlinked_maps,
            "sim_length": self.simulator.simTime / 60,
        }

        self.curr_step += 1
        return ob, reward, self.episode_over, False, info

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
            downlinked_images,
            downlinked_maps,
            imaged,
            mapped,
        ) = self.simulator.run_sim(action)

        return downlinked_images, downlinked_maps, imaged, mapped

    def _get_reward(self, downlinked_images, downlinked_maps, imaged, mapped):
        """Reward is based on the total amount of imaged data downlinked in MB."""
        reward = 0
        if self.failure:
            reward = -self.failure_penalty
        else:
            # Map image reward
            reward += (
                self.map_component
                / (self.simulator.n_maps * self.simulator.n_map_points)
            ) * len(mapped)
            # Map downlink reward
            reward += (
                self.map_downlink_component
                / (self.simulator.n_maps * self.simulator.n_map_points)
            ) * len(downlinked_maps)
            # Target imaging reward
            reward += (self.target_component / self.simulator.n_targets) * len(imaged)
            # Target downlink reward
            reward += (self.target_downlink_component / self.simulator.n_targets) * len(
                downlinked_images
            )

        return reward

    def reset(self, seed=None, options=None):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.action_episode_memory.append([])
        self.episode_over = False
        self.failure = False
        self.curr_step = 0
        self.reward_total = 0

        # If initial conditions are passed in, use those
        if options is not None:
            if "initial_conditions" in options:
                self.initial_conditions = options["initial_conditions"]

        if self.simulator is not None:
            del self.simulator

        # Create the simulator
        self.simulator = SmallBodyScienceSimulator(
            self.dynRate,
            self.fswRate,
            self.mapRate,
            self.step_duration,
            self.initial_conditions,
            render=self.render,
            n_targets=self.n_targets,
            n_map_points=self.n_map_points,
            max_length=self.max_length,
            n_states=self.n_states,
            n_maps=self.n_maps,
            phi_c=self.phi_c,
            lambda_c=self.lambda_c,
            fidelity=self.fidelity,
        )

        self.simulator.init_tasks_and_processes()

        # Extract initial conditions from instantiation of simulator
        self.initial_conditions = self.simulator.initial_conditions
        self.simulator.max_steps = self.max_steps
        self.simulator_init = 1

        return self.simulator.obs, {}

    def _get_state(self):
        """Get the observation.

        WIP: Work out which error representation to give the algo.
        """

        return self.simulator.obs


if __name__ == "__main__":
    env = gym.make("SmallBodyScience-v0")

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
