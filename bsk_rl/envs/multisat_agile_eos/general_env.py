import gymnasium as gym
import numpy as np

from bsk_rl.envs.general_satellite_tasking.scenario import communication, data
from bsk_rl.envs.general_satellite_tasking.scenario import sat_actions as sa
from bsk_rl.envs.general_satellite_tasking.scenario import sat_observations as so
from bsk_rl.envs.general_satellite_tasking.scenario import satellites as sats
from bsk_rl.envs.general_satellite_tasking.scenario.environment_features import (
    StaticTargets,
)
from bsk_rl.envs.general_satellite_tasking.simulation import dynamics, environment, fsw
from bsk_rl.utilities.initial_conditions import leo_orbit

sim_rate = 1.0
step_duration = 6 * 60.0
n_targets_state = 3
intervals = 45
target_tuple_size = 4
n_targets = 1040
failure_penalty = -1
oes = leo_orbit.walker_delta(
    n_spacecraft=2,  # Number of satellites
    n_planes=2,
    rel_phasing=0,
    altitude=500 * 1e3,
    inc=45,
)


class CustomSatComposed(
    sa.ImagingActions.configure(n_ahead_act=n_targets_state),
    sa.DesatAction,
    sa.DownlinkAction,
    sa.ChargingAction,
    so.TargetState.configure(n_ahead_observe=n_targets_state),  # TODO put in hill frame
    so.NormdPropertyState.configure(
        obs_properties=[
            dict(prop="r_BN_P", norm=1),
            dict(prop="v_BN_P", norm=1),
            # att error TODO
            # att rate error TODO
            dict(prop="wheel_speeds_fraction"),
            dict(prop="battery_charge_fraction"),
            # in_eclipse TODO
            dict(prop="storage_level_fraction")
            # np.sum(transmitterBaud)* self.dynRate/ (transmitterBaudRate * self.step_duration),
            # np.sum(accessIndicator1) * self.dynRate / self.step_duration,
            # np.sum(accessIndicator2) * self.dynRate / self.step_duration,
            # np.sum(accessIndicator3) * self.dynRate / self.step_duration,
            # np.sum(accessIndicator4) * self.dynRate / self.step_duration,
            # np.sum(accessIndicator5) * self.dynRate / self.step_duration,
            # np.sum(accessIndicator6) * self.dynRate / self.step_duration,
            # np.sum(accessIndicator7) * self.dynRate / self.step_duration,
        ]
    ),
    sats.ImagingSatellite,
):
    class CustomDynModel(dynamics.ImagingDynModel, dynamics.LOSCommDynModel):
        pass

    dyn_type = CustomDynModel
    fsw_type = fsw.ImagingFSWModel


# Construct satellites of the FullFeaturedSatellite type
satellites = []
sat_type = sats.FullFeaturedSatellite
for i, oe in enumerate(oes):
    sat_args = sat_type.default_sat_args(
        oe=oe,
        imageAttErrorRequirement=0.01,
        imageRateErrorRequirement=None,
    )

    # Instantiate the satellite object. Arguments to the satellite class are set here.
    satellite = sat_type(
        "EO" + str(i + 1),
        sat_args,
        variable_interval=False,
    )
    satellites.append(satellite)

# print states and actions

# Make the environment with Gymnasium
env = gym.make(
    "GeneralSatelliteTasking-v1",
    satellites=satellites,
    env_type=environment.GroundStationEnvModel,
    env_args=environment.GroundStationEnvModel.default_env_args(
        utc_init="2021 MAY 04 07:47:48.965 (UTC)"
    ),
    env_features=StaticTargets(
        n_targets=n_targets, priority_distribution=lambda: np.random.randint(1, 4)
    ),
    data_manager=data.UniqueImagingManager(reward_fn=lambda x: 1 / x / intervals),
    communicator=communication.LOSMultiCommunication(satellites),
    sim_rate=sim_rate,
    max_step_duration=step_duration,
    time_limit=step_duration * intervals,
)
