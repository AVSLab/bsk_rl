import gymnasium as gym
import numpy as np
from Basilisk.architecture import bskLogging
from Basilisk.utilities import orbitalMotion

from bsk_rl.envs.GeneralSatelliteTasking.scenario import data
from bsk_rl.envs.GeneralSatelliteTasking.scenario import satellites as sats
from bsk_rl.envs.GeneralSatelliteTasking.scenario.environment_features import (
    CityTargets,
)
from bsk_rl.envs.GeneralSatelliteTasking.simulation import dynamics, environment, fsw
from bsk_rl.envs.GeneralSatelliteTasking.utils.orbital import random_orbit

bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)
"""
This script demonstrates customization options for a satellite, including:
- Model selection
- State space configuration
- Action space configuration
"""


# Define a new satellite class, selecting a similar class as a starting point
class CustomSat(sats.ImagingSatellite):
    # Change the attitude controller by redefining fsw_type. In this case, we are using a MRP Feedback based controller
    # instead of a the default PD feedback-based controller.
    fsw_type = fsw.SteeringImagerFSWModel

    # In some cases, the specific model you want may not exactly exists. Models are designed to be easily composed, so
    # a new model based on existing models can be quickly defined.

    class CustomDynModel(dynamics.ImagingDynModel, dynamics.LOSCommDynModel):
        pass

    dyn_type = CustomDynModel

    # Model compatibility between FSW, dynamics, and the environment should be automatically checked in most cases.

    # A more common customization requirement is designing the observation and action spaces. Three functions are most
    # commonly overridden to achieve this: get_obs, set_action, and n_actions

    # Define a custom observation. Various properties from the Basilisk simulation are exposed through the satellite
    # class to make this process easier, including r_BN_B, omega_BN_B, and many more. Typically, this function should
    # return a 1-dimensional numpy array. In this example, the satellite's dynamic state and information about upcoming
    # targets are normalized.
    def get_obs(self):
        dynamic_state = np.concatenate(
            [
                self.dynamics.omega_BP_P / 0.03,
                self.fsw.c_hat_P,
                self.dynamics.r_BN_P / (orbitalMotion.REQ_EARTH * 1e3),
                self.dynamics.v_BN_P / 7616.5,
            ]
        )
        images_state = np.array(
            [
                np.concatenate(
                    [
                        [target.priority],
                        target.location / (orbitalMotion.REQ_EARTH * 1e3),
                    ]
                )
                for target in self.upcoming_targets(self.n_ahead_observe)
            ]
        )
        images_state = images_state.flatten()

        return np.concatenate((dynamic_state, images_state))

    # Define a custom action function. In most discrete RL contexts, this function should accept a single integer;
    # however, any parameterization is possible with this package. An important note: it is generally undesireable
    # to retask the same action twice in a row as controller states will get reset. Good set_action defintions should
    # include protections against this. In this example:
    #   - 0: charge
    #   - 1: desaturate
    #   - 2+: image the (n-3)th upcoming target
    def set_action(self, action):
        if action == 0 and self.current_action != 0:
            # Use functions defined in FSW with the @action decorator to interact with the Basilisk sim.
            self.fsw.action_charge()
            # Save data to the info dictonary for debugging help
            self.log_info("charging tasked")
        elif action == 1 and self.current_action != 1:
            self.fsw.action_desat()
            self.log_info("desat tasked")
        else:
            target_action = action
            if isinstance(target_action, int):
                target_action -= 2
            # Use the standard ImagingSatellite tasking function
            super().set_action(target_action)

        if action < 2:
            self.current_action = action

    # The action space cannot be inferred; explicitly tell gymnasium how many actions the satellite can take
    @property
    def n_actions(self) -> int:
        return super().n_actions + 2


# Configure the environent
env_features = CityTargets(n_targets=5000)
data_manager = data.UniqueImagingManager(env_features)
# Use the CustomSat type
sat_type = CustomSat
sat_args = sat_type.default_sat_args(
    imageAttErrorRequirement=0.01,
    imageRateErrorRequirement=0.01,
    oe=random_orbit,
)
satellite = sat_type(
    "EO1",
    sat_args,
    n_ahead_observe=3,
    n_ahead_act=3,
    variable_interval=False,
)

# Make the environment with Gymnasium
env = gym.make(
    "SingleSatelliteTasking-v1",
    satellites=satellite,
    # Select an EnvironmentModel compatible with the models in the satellite
    env_type=environment.BasicEnvironmentModel,
    env_args=environment.BasicEnvironmentModel.default_env_args(),
    env_features=env_features,
    data_manager=data_manager,
    sim_rate=0.5,
    max_step_duration=600.0,
    time_limit=95 * 60 * 3,
)

# Run the simulation until timeout or agent failure
total_reward = 0.0
observation, info = env.reset()

while True:
    print(f"<time: {env.simulator.sim_time:.1f}s>")

    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample()  # Task random actions
    )

    # Show the custom normalized observation vector
    print("\tObservation:", observation)

    # Print info dict messages from each sat
    msg_list = []
    for sat, msgs in info.items():
        if isinstance(msgs, list):
            for time, message in msgs:
                msg_list.append(
                    (
                        time,
                        f"\t<{'_'.join(sat.split('_')[0:-1])} at {time:.1f}>\t{message}",
                    )
                )
    for time, message in sorted(msg_list):
        print(message)

    total_reward += reward
    print(f"\tReward: {reward:.3f} ({total_reward:.3f} cumulative)")
    if terminated or truncated:
        print("Episode complete.")
        break
