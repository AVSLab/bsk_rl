from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bsk_rl.env.multisensor_eos.bsk_sim import MultiSensorEOSSimulator
from bsk_rl.env.multisensor_eos.env_settings import Settings


class MultiSensorEOS(gym.Env):
    """
    Earth observation environment - simulates a spacecraft with variable imager modes
    attempting to image a ground location.
    Agent must choose between charging, desaturating, and image type(s); also needs to
    choose an appropriate imaging type.
    Taking the image type corresponding to the ground location's required sensor type
    results in full reward, other image types results in no reward.

    Action Space (discrete):

    * 0 - Points solar panels at the sun.
    * 1 - Desaturates the reaction wheels.
    * >1 - Orients the s/c towards the earth; takes image of type _.

    Observation Space:

    * r_sc_I - float[3,] - spacecraft position.
    * v_sc - float[3,] - spacecraft velocity in PCPF.
    * sigma_RB - float [0,1] - norm of the spacecraft error MRP with respect to the
      last reference frame specified.
    * omega_BN - float - norm of the total spacecraft bus rotational velocity with
      respect to the inertial frame.
    * omega_RW - float - norm of the reaction wheel rotational velocities.
    * storedCharge - float [0,batCapacity] - indicates the s/c battery charge level in
      W-s.
    * sun_indicator - float [0, 1] - indicates the flux mitigator due to eclipse.
    * access indicator - access to the next target
    * img_mode norm - float [0,1] - indicates the required imaging mode.

    Reward Function:
    r = 1/(1+ | sigma_RB|) if correct sensor

    Intended to provide a rich reward in action 1 when the spacecraft is pointed
    towards the earth, decaying as sigma^2 as the pointing error increases.
    """

    def __init__(self):
        self.__version__ = "1.0.0"
        print(
            "Earth Observation Sim, State Guarded Actions - Version {}".format(
                self.__version__
            )
        )

        # Set up simulation parameters
        self.settings = Settings()
        self.step_duration = self.settings.STEP_DURATION
        self.max_steps = self.settings.SIM_TIME
        self.dyn_step = self.settings.DYN_STEP
        self.fsw_step = self.settings.FSW_STEP

        # Set up environment parameters
        self.curr_step = None
        self.reward_total = None
        self.initial_conditions = None
        self.simulator = None
        self.episode_over = None
        self.curr_episode = -1
        self.ob = None
        self.state_machine_state = None
        self.prev_ins_spec = None
        self.prev_sim_state = None
        self.sim_state = None
        self.info = {}

        # Set up setting variables locally
        self.wheel_lim = self.settings.WHEEL_LIM
        self.power_max = self.settings.POWER_MAX
        self.obs_defn = self.settings.OBSERVATIONS

        #   Reward function parameters
        self.max_observations = self.settings.N_TGTS
        self.reward_mult = (
            self.settings.REWARD_MULTIPLIER / self.max_observations
        )  # Normalize reward to episode duration; '1' represents 100% target
        # observation with 100% accuracy
        self.failure_penalty = 1  # episode reward of 0 or less is a failed episode
        self.img_modes = self.settings.img_modes

        self.failure = False

        #   Set observation space
        low = -1e16
        high = 1e16
        self.observation_space = spaces.Box(
            low, high, shape=(len(self.obs_defn),), dtype=np.float64
        )

        print("Observation space: ", self.observation_space.shape)

        # Action Space description
        #   0 - sun pointing (power objective)
        #   1 - desaturation (required for long-term pointing)
        #   >1 - imaging types
        self.action_space = spaces.Discrete(2 + self.img_modes)
        self.action_episode_memory = []

        print("Action space: ", self.action_space.n)

        self.render = self.settings.RENDER

        self.return_obs = True

    def step(self, action):
        """
        The agent takes a step in the environment. Note that the simulator must be
        initialized

        Args:
            action: int

        Returns:

        * ob (object): an environment-specific object representing your observation of
          the environment.
        * reward (float): amount of reward achieved by the previous action. The scale
          varies between environments, but the goal is always to increase
          your total reward.
        * episode_over (bool): whether it's time to reset the environment again. Most (but not
          all) tasks are divided up into well-defined episodes, and done
          being True indicates the episode has terminated. (For example,
          perhaps the pole tipped too far, or you lost your last life.)
        * truncated (truncated): set to false. Gymnasium requirement.
        * info (dict): diagnostic information useful for debugging. It can sometimes
          be useful for learning (for example, it might contain the raw
          probabilities behind the environment's last state change).
          However, official evaluations of your agent are not allowed to
          use this for learning.

        """

        self.curr_step += 1
        self.prev_sim_state = deepcopy(self.sim_state)
        self.prev_ins_spec = self.simulator.img_mode
        self._take_action(action)

        #   If the wheel speeds get too large, end the episode.
        if self.sim_state.get("wheel_speed") > self.wheel_lim:
            self.episode_over = True
            self.failure = True
            print("Died from wheel explosion.")
        elif self.sim_state.get("stored_charge") == 0:
            self.episode_over = True
            self.failure = True
            print("Ran out of power.")
        elif self.simulator.sim_over:
            self.episode_over = True

        reward = self._get_reward()
        self.reward_total += reward

        self.info = {
            "episode": {"r": self.reward_total, "l": self.curr_step},
            "obs": self.ob,
            "fsm": self.prev_ins_spec,
        }

        self.info["metrics"] = {
            "sim_length": self.simulator.sim_time / 60,
            "num_imaged": self._get_num_imaged(),
        }

        return self.ob, reward, self.episode_over, False, self.info

    def _take_action(self, action):
        """
        Interfaces with the simulator to
        :param action:
        :return:
        """

        self.action_episode_memory[self.curr_episode].append(action)

        #   Let the simulator handle action management:
        self.sim_state = self.simulator.run_sim(action)
        self.ob = self._get_ob()

    def _get_reward(self):
        """
        Reward is based on time spent with the inertial attitude pointed towards the
        ground within a given tolerance.

        """
        reward = 0
        last_action = self.action_episode_memory[self.curr_episode][-1]
        if self.failure:
            reward = -self.failure_penalty
        elif (last_action > 1) and self.prev_sim_state.get("access_indicator"):
            #   Attitude contribution:
            att_reward = self.reward_mult / (
                1.0 + self.prev_sim_state.get("att_err") ** 2.0
            )
            if last_action == (
                self.prev_ins_spec + 1
            ):  # If the obs taken matches the correction insturment action type
                freq_mult = 1.0
            else:
                freq_mult = 0.0
            reward = att_reward * freq_mult

        return reward

    def _get_ob(self):
        """Get the observation.
        WIP: Work out which error representation to give the algo."""
        ob = np.zeros(len(self.obs_defn))
        for i, ob_key in enumerate(self.obs_defn):
            ob[i] = self.sim_state.get(ob_key)
            if ob_key == "wheel_speed":
                ob[
                    i
                ] /= (
                    self.wheel_lim
                )  # Normalize reaction wheel speed to fraction of limit
            elif ob_key == "stored_charge":
                ob[
                    i
                ] /= (
                    self.power_max
                )  # Normalize current power to fraction of total power

        return ob

    def reset(self, seed=None, options=None):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        super().reset(seed=seed)
        self.curr_episode += 1
        self.action_episode_memory.append([])
        self.curr_step = -1
        self.reward_total = 0
        del (
            self.simulator
        )  # Force delete the sim to make sure nothing funky happens under the hood
        tFinal = self.max_steps * self.step_duration

        if options is None:
            self.settings.generate_new_ic()
            self.initial_conditions = self.settings.INITIAL_CONDITIONS
        else:
            if "initial_conditions" not in options:
                self.settings.generate_new_ic()
                self.initial_conditions = self.settings.INITIAL_CONDITIONS
            else:
                self.initial_conditions = options["initial_conditions"]

        self.simulator = MultiSensorEOSSimulator(
            self.dyn_step,
            self.fsw_step,
            self.step_duration,
            tFinal,
            initial_conditions=self.initial_conditions,
            render=self.render,
            settings=self.settings,
        )
        self.episode_over = self.simulator.sim_over
        self.failure = False
        self.sim_state = self.simulator.get_sim_state(init=True)
        self.ob = self._get_ob()

        return self.ob, self.info

    def _render(self, mode="human", close=False):
        self.render = True
        return

    def _get_num_imaged(self):
        num_imaged = 0
        for idx in range(len(self.simulator.imaged_targets)):
            if self.simulator.imaged_targets[idx] >= 1.0:
                num_imaged += 1


if __name__ == "__main__":
    env = gym.make("MultiSensorEOS-v0")

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
