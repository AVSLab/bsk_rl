import gymnasium as gym
from Basilisk.architecture import bskLogging
from Basilisk.utilities import orbitalMotion

from bsk_rl.envs.general_satellite_tasking.scenario import data
from bsk_rl.envs.general_satellite_tasking.scenario import sat_actions as sa
from bsk_rl.envs.general_satellite_tasking.scenario import sat_observations as so
from bsk_rl.envs.general_satellite_tasking.scenario import satellites as sats
from bsk_rl.envs.general_satellite_tasking.scenario.environment_features import (
    CityTargets,
)
from bsk_rl.envs.general_satellite_tasking.simulation import dynamics, environment, fsw
from bsk_rl.envs.general_satellite_tasking.utils.orbital import random_orbit

bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

# This script demonstrates customization options for a satellite, including:
# - Model selection
# - Observation space configuration
# - Action space configuration

# There are two primary methods of customization. The preferred option (1) is to use the
# built-in observation space and actions space satellite subclasses. Alternatively,
# option (2) is to manually override methods for observations and actions in a satellite
# subclass.


# OPTION 1: Define a new satellite class by composing existing types.
class CustomSatComposed(
    # Action classes. Discrete actions are added in reverse order
    # Thus produces an action space of the form:
    #   {'0': 'action_charge', '1': 'action_desat', '2-4': 'image'}
    sa.ImagingActions.configure(n_ahead_act=3),
    sa.DesatAction.configure(action_duration=120.0),
    sa.ChargingAction.configure(action_duration=60.0),
    # Observation classes. In the vectorized observation, these will be composed in
    # reverse order. Default arguments for __init__ can be overriden with configure() to
    # bake them into the class definition prior to instantiation.
    # This produces an observaiton in the form:
    #    omega_BP_P_normd:  [ 0.01489828  0.0004725  -0.08323254]
    #    c_hat_P:  [ 0.66675533 -0.69281445  0.27467338]
    #    r_BN_P_normd:  [ 0.09177786 -0.80203809 -0.7120501 ]
    #    v_BN_P_normd:  [ 0.91321553 -0.11810811  0.25020653]
    #    battery_charge_fraction:  0.740410440543005
    #    target_obs:  {'tgt_value_0': 0.1878322060213219,
    #                  'tgt_loc_0_normd': array([ 0.21883092, -0.72328348, -0.6549610]),
    #                  'tgt_value_1': 0.8484751150377395,
    #                  'tgt_loc_1_normd': array([ 0.23371944, -0.73369242, -0.6380208]),
    #                  'tgt_value_2': 0.14482123441765427,
    #                  'tgt_loc_2_normd': array([ 0.23645694, -0.73721533, -0.63293101])
    #                 }
    #    normalized_time:  0.22505847953216376
    so.TimeState,
    so.TargetState.configure(n_ahead_observe=3),
    so.NormdPropertyState.configure(
        obs_properties=[
            dict(prop="omega_BP_P", norm=0.03),
            dict(prop="c_hat_P"),
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", norm=7616.5),
            dict(prop="battery_charge_fraction"),
        ]
    ),
    # Base class for this satellite
    sats.ImagingSatellite,
):
    # Change the attitude controller by redefining fsw_type. In this case, we are using
    # a MRP Feedback based controller instead of a the default PD feedback-based
    # controller.
    fsw_type = fsw.SteeringImagerFSWModel

    # In some cases, the specific model you want may not exactly exists. Models are
    # designed to be easily composed, so a new model based on existing models can be
    # quickly defined.

    class CustomDynModel(dynamics.ImagingDynModel, dynamics.LOSCommDynModel):
        pass

    dyn_type = CustomDynModel
    # Model compatibility between FSW, dynamics, and the environment should be
    # automatically checked in most cases.


# OPTION 2: Define a new satellite class manually, selecting a similar class as a
# starting point
# class CustomSatManual(sats.ImagingSatellite):
#     # Select FSW and dynamics as in option 1
#     fsw_type = fsw.SteeringImagerFSWModel

#     class CustomDynModel(dynamics.ImagingDynModel, dynamics.LOSCommDynModel):
#         pass

#     dyn_type = CustomDynModel

#     # A more common customization requirement is designing the observation and action
#     # spaces. Three functions are most commonly overridden to achieve this: get_obs,
#     # set_action, and n_actions

#     # Define a custom observation. Various properties from the Basilisk simulation are
#     # exposed through the satellite class to make this process easier, including
#     # r_BN_B, omega_BN_B, and many more. Typically, this function should return a
#     # 1-dimensional numpy array. In this example, the satellite's dynamic state and
#     # information about upcoming targets are normalized.
#     def get_obs(self):
#         dynamic_state = np.concatenate(
#             [
#                 self.dynamics.omega_BP_P / 0.03,
#                 self.fsw.c_hat_P,
#                 self.dynamics.r_BN_P / (orbitalMotion.REQ_EARTH * 1e3),
#                 self.dynamics.v_BN_P / 7616.5,
#             ]
#         )
#         images_state = np.array(
#             [
#                 np.concatenate(
#                     [
#                         [target.priority],
#                         target.location / (orbitalMotion.REQ_EARTH * 1e3),
#                     ]
#                 )
#                 for target in self.upcoming_targets(self.n_ahead_observe)
#             ]
#         )
#         images_state = images_state.flatten()

#         return np.concatenate((dynamic_state, images_state))

#     # Define a custom action function. In most discrete RL contexts, this function
#     # should accept a single integer; however, any parameterization is possible with
#     # this package. An important note: it is generally undesirable to retask the same
#     # action twice in a row as controller states will get reset. Good set_action
#     # defintions should include protections against this. In this example:
#     #   - 0: charge
#     #   - 1: desaturate
#     #   - 2+: image the (n-3)th upcoming target
#     def set_action(self, action):
#         if action == 0 and self.current_action != 0:
#             # Use functions defined in FSW with the @action decorator to interact with
#             # the Basilisk sim.
#             self.fsw.action_charge()
#             # Save data to the info dictonary for debugging help
#             self.log_info("charging tasked")
#         if action == 1 and self.current_action != 1:
#             self.fsw.action_desat()
#             self.log_info("desat tasked")
#         else:
#             target_action = action
#             if isinstance(target_action, int):
#                 target_action -= 2
#             # Use the standard ImagingSatellite tasking function
#             super().set_action(target_action)

#         if action < 2:
#             self.current_action = action

#     # The action space cannot be inferred; explicitly tell gymnasium how many actions
#     # the satellite can take
#     @property
#     def action_space(self):
#         return gym.spaces.Discrete(self.n_ahead_act + 2)


# Configure the environent
env_features = CityTargets(n_targets=5000)
data_manager = data.UniqueImagingManager(env_features)
# Use the CustomSat type
sat_type = CustomSatComposed
sat_args = sat_type.default_sat_args(
    imageAttErrorRequirement=0.01,
    imageRateErrorRequirement=0.01,
    oe=random_orbit,
)
satellite = sat_type(
    "EO1",
    sat_args,
    variable_interval=True,
)
# The composed satellite action space returns a human-readable action map
print("Actions:", satellite.action_map)

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
    log_level="INFO",
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

    # Using the composed satellite features also provides a human-readable state:
    for k, v in env.satellite.obs_dict.items():
        print(f"\t\t{k}:  {v}")

    total_reward += reward
    print(f"\tReward: {reward:.3f} ({total_reward:.3f} cumulative)")
    if terminated or truncated:
        print("Episode complete.")
        break
