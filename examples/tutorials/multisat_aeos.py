"""
Multisat AEOS
=============

some text here
"""

import gymnasium as gym
import numpy as np

from bsk_rl.env.scenario import communication, data
from bsk_rl.env.scenario import satellites as sats
from bsk_rl.env.scenario.environment_features import CityTargets
from bsk_rl.utils.orbital import walker_delta


def run():
    """Demonstrate the configuration of an environment with multiple imaging satellites."""
    # Data environment contains 5000 targets located near random cities, which are
    # randomized on reset()
    env_features = CityTargets(n_targets=500, location_offset=10e3)
    # Data manager records and rewards uniquely imaged targets
    data_manager = data.UniqueImagingManager(env_features)

    # Generate orbital parameters for each satellite in the constellation
    oes = walker_delta(
        n_spacecraft=3,  # Number of satellites
        n_planes=1,
        rel_phasing=0,
        altitude=500 * 1e3,
        inc=45,
        clustersize=3,  # Cluster all 3 satellites together
        clusterspacing=30,  # Space satellites by a true anomaly of 30 degrees
    )

    # Construct satellites of the FullFeaturedSatellite type
    satellites = []
    sat_type = sats.FullFeaturedSatellite
    for i, oe in enumerate(oes):
        # Satellite configuration arguments are inferred from the satellite type. The
        # function default_sat_args collects all of the parameters that must be set for FSW
        # and dynamics in the Basilisk simulation. Any parameters that are to be overridden
        # can be set as arguments to default_sat_args, and an error will be raised if the
        # parameter is not valid for the satellite type.

        sat_args = sat_type.default_sat_args(
            oe=oe,
            imageAttErrorRequirement=0.01,  # Change a default parameter
            imageRateErrorRequirement=0.01,
            # Parameters can also be set as a function that is called each time the
            # environment is reset
            panelEfficiency=lambda: 0.2 + np.random.uniform(-0.01, 0.01),
        )

        # As an example, look at the arguments for one of the satellites
        if i == 0:
            print(sat_args)

        # Instantiate the satellite object. Arguments to the satellite class are set here.
        satellite = sat_type(
            "EO" + str(i + 1), sat_args, n_ahead_observe=15, n_ahead_act=15
        )
        satellites.append(satellite)

    # Instantiate the communication method
    communicator = communication.LOSMultiCommunication(satellites)

    # Make the environment with Gymnasium
    env = gym.make(
        "GeneralSatelliteTasking-v1",
        satellites=satellites,
        # Pass configuration objects
        env_features=env_features,
        data_manager=data_manager,
        communicator=communicator,
        # Integration frequency in seconds
        sim_rate=0.5,
        # Environment will be propagated by at most max_step_duration before needing new
        # actions selected; however, some satellites will instead end the step when the
        # current task is finished
        max_step_duration=600.0,
        # Set 3-orbit long episodes
        time_limit=95 * 60,
        log_level="INFO",
    )

    # Run the simulation until timeout or agent failure
    total_reward = 0.0
    observation, info = env.reset()

    while True:
        """
        Task random actions. Look at the set_action function for the chosen satellite type
        to see what actions do. In this case, the action mapping is as follows:
                - 0: charge
                - 1: desaturate
                - 2: downlink
                - 3+: image the (n-3)th upcoming target

        """
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )

        total_reward += reward
        print(f"\tReward: {reward:.3f} ({total_reward:.3f} cumulative)")

        if terminated or truncated:
            print("Episode complete.")
            break


if __name__ == "__main__":
    run()
