# 3rd party modules
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bsk_rl.envs.SmallBodyScience.gym_env import SmallBodyScience
from bsk_rl.envs.SmallBodySciencePOMDP.bsk_sim import SmallBodySciencePOMDPSimulator


class SmallBodySciencePOMDP(SmallBodyScience):
    """
    Small body gym environment where an agent can transition between different
    waypoints defined in the sun anti-momentum frame to image candidate landing sites
    or collect spectroscopy map data while avoiding resource constraint violations. As
    opposed to the SmallBodyScience environment, this environment is utilizes an EKF
    filter for the observation space to simulate a POMDP, which provides a belief state
    for the POMDP.
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
    12 - Navigation Mode

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
    17-20: Location of the next target for imaging
    20-26: Filter covariance diagonals

    Reward Function:
    r = +A each tgt downlinked for first time
    r = +B for each tgt imaged for first time
    r = +C for each map region downlinked for first time
    r = +D for each map region collected for first time
    r = -E for failure
    """

    def __init__(self):
        super().__init__()

        # Set the version
        self.__version__ = "0.0.0"
        print(
            "Small Body Science POMDP Environment - Version {}".format(self.__version__)
        )

        # Modify the observation space
        self.n_states = 26
        self.obs = np.zeros(self.n_states)
        self.obs_full = np.zeros(self.n_states)
        self.observation_space = spaces.Box(-1e16, 1e16, shape=(self.n_states,))

        # Modify the action space to include the navigation mode
        self.action_space = spaces.Discrete(self.action_space.n + 1)

    def step(self, action, return_obs=True):
        ob, reward, self.episode_over, info = super().step(action, return_obs=True)

        return self.modify_ob(ob), reward, self.episode_over, info

    def modify_ob(self, ob):
        """
        Modifies the observation of the MDP such that it conforms to the POMDP
        specification.
        :param ob:
        :return:
        """
        return ob

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
        self.simulator = SmallBodySciencePOMDPSimulator(
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


if __name__ == "__main__":
    env = gym.make("SmallBodySciencePOMDP-v0")

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
