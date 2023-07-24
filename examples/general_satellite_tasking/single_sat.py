import gymnasium as gym

from bsk_rl.envs.GeneralSatelliteTasking.scenario import data
from bsk_rl.envs.GeneralSatelliteTasking.scenario import satellites as sats
from bsk_rl.envs.GeneralSatelliteTasking.scenario.environment_features import (
    StaticTargets,
)
from bsk_rl.envs.GeneralSatelliteTasking.simulation import environment
from bsk_rl.envs.GeneralSatelliteTasking.utils.orbital import random_orbit

"""
This script demonstrates the configuration of an environment with a single imaging satellite.
"""

# Data environment contains 5000 targets randomly distributed
env_features = StaticTargets(n_targets=5000)
# Data manager records and rewards uniquely imaged targets
data_manager = data.UniqueImagingManager(env_features)

# Construct satellites of the FullFeaturedSatellite type
sat_type = sats.FullFeaturedSatellite
"""Satellite configuration arguments are inferred from the satellite type. The function default_sat_args
collects all of the parameters that must be set for FSW and dynamics in the Basilisk simulation. Any
parameters that are to be overridden can be set as arguments to default_sat_args, and an error will be raised if the
parameter is not valid for the satellite type.
"""
sat_args = sat_type.default_sat_args(
    imageAttErrorRequirement=0.01,  # Change a default parameter
    imageRateErrorRequirement=0.01,
    # Parameters can also be set as a function that is called each time the environment is reset
    oe=random_orbit,
)
print(sat_args)

# Instantiate the satellite object. Arguments to the satellite class are set here.
satellite = sat_type("EO1", sat_args, n_ahead_observe=30, n_ahead_act=15)

# Make the environment with Gymnasium
env = gym.make(
    # The SingleSatelliteTasking environment takes actions and observations directly from the satellite, instead of
    # wrapping them in a tuple
    "SingleSatelliteTasking-v1",
    satellites=satellite,
    # Pick the type for the Basilisk environment model. Note that it is not instantiated here.
    env_type=environment.GroundStationEnvModel,
    # Like default_sat_args, default_env_args infers model parameters from the type and specific parameters can be
    # overridden or randomized.
    env_args=environment.GroundStationEnvModel.default_env_args(),
    # Pass configuration objects
    env_features=env_features,
    data_manager=data_manager,
    # Integration frequency in seconds
    sim_rate=0.5,
    # Environment will be propagated by at most max_step_duration before needing new actions selected; however, some
    # satellites will instead end the step when the current task is finished
    max_step_duration=600.0,
    # Set 3-orbit long episodes
    time_limit=95 * 60 * 3,
    # Send the terminated signal in addition to the truncated signal at the end of the episode. Needed for some RL
    # algorithms to work correctly.
    terminate_on_time_limit=True,
)

# Run the simulation until timeout or agent failure
total_reward = 0.0
observation, info = env.reset()

while True:
    print(f"<time: {env.simulator.sim_time:.1f}s>")

    """
    Task random actions. Look at the set_action function for the chosen satellite type to see what actions do. In this
    case, the action mapping is as follows:
            - 0: charge
            - 1: desaturate
            - 2: downlink
            - 3+: image the (n-3)th upcoming target

    """
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )

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
