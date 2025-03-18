import gymnasium as gym
from requests.packages import target

import os
import numpy as np

# Ensure the data directory exists
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

from bsk_rl import act, data, obs, scene, sats
from bsk_rl.sim import dyn, fsw

from Basilisk.architecture import bskLogging
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

class MyScanningSatellite(sats.Satellite):
    observation_spec = [
        obs.SatProperties(
            dict(prop="storage_level_fraction"),
            dict(prop="battery_charge_fraction")
        ),
        obs.Eclipse(),
    ]
    action_spec = [
        act.ImageRSO(n_ahead_image=n_targets,duration=2000),  # Scan for 1 minute
        act.Charge(duration=600.0),  # Charge for 10 minutes
    ]
    dyn_type = dyn.ImagingSCDynModel
    fsw_type = fsw.ImagingSCFSWModel

MyScanningSatellite.default_sat_args() # why is this needed?

sat_args = {}

# Set some parameters as constants
sat_args["imageAttErrorRequirement"] = 0.05
sat_args["dataStorageCapacity"] = 1e10
sat_args["instrumentBaudRate"] = 1e7
sat_args["storedCharge_Init"] = 50000.0

# Randomize the initial storage level on every reset
sat_args["storageInit"] = lambda: np.random.uniform(0.25, 0.75) * 1e10


class MyTargetSatellite(sats.Satellite):
    observation_spec = [
        obs.Time(),
    ]
    action_spec = [
        act.Drift(duration=total_time),  # Scan for 1 minute
        # act.Charge(duration=600.0),  # Charge for 10 minutes
    ]
    dyn_type = dyn.BasicTargetDynamicsModel  # Passed as a type
    fsw_type = fsw.BasicTargetFSWModel
# Make the satellite
sat = MyScanningSatellite(name="SO1", sat_args=sat_args) # SO1 for satellite observer 1

env = gym.make(
    "SatelliteTasking-v1",
    satellite=sat,
    scenario=scene.RandomSatellites(50),
    rewarder=data.RSOTargetImageReward(),
    time_limit=5700.0,  # approximately 1 orbit
    log_level="INFO",
    disable_env_checker=True,
)

observation, info = env.reset(seed=1)
inspector_sigmaBN = []
inspector_omegaBN = []
inspector_r_BN_N = []
target_r_BN_N = []

print("Initial data level:", observation[0], "(randomized by sat_args)")
for _ in range(3):
    observation, reward, terminated, truncated, info = env.step(action=0)
    inspector_sigmaBN.append(env.satellites[0].dynamics.sigma_BN)
    inspector_omegaBN.append(env.satellites[0].dynamics.omega_BN_B)
    inspector_r_BN_N.append(env.satellites[0].dynamics.r_BN_N)
    target_r_BN_N.append(env.satellites[0].dynamics.simpleTargetNav.transOutMsg.read().r_BN_N)
print("  Final data level:", observation[0])



while not truncated:
    observation, reward, terminated, truncated, info = env.step(action=0)
    inspector_sigmaBN.append(env.satellites[0].dynamics.sigma_BN)
    inspector_omegaBN.append(env.satellites[0].dynamics.omega_BN_B)
    inspector_r_BN_N.append(env.satellites[0].dynamics.r_BN_N)
    target_r_BN_N.append(env.satellites[0].dynamics.simpleTargetNav.transOutMsg.read().r_BN_N)

    print(f"Charge level: {observation[1]:.3f} ({env.unwrapped.simulator.sim_time:.1f} seconds)\n\tEclipse: start: {observation[2]:.1f} end: {observation[3]:.1f}")

print('Inspector sigma BN', inspector_sigmaBN[0], inspector_sigmaBN[-1])
print('Inspector r_BN_N', inspector_r_BN_N[0], inspector_r_BN_N[-1])
print('Target r_BN_N', target_r_BN_N[0:10])

# Convert to numpy arrays and save
np.save(os.path.join(data_dir, "inspector_sigmaBN.npy"), np.array(inspector_sigmaBN))
np.save(os.path.join(data_dir, "inspector_omegaBN.npy"), np.array(inspector_omegaBN))
np.save(os.path.join(data_dir, "inspector_r_BN_N.npy"), np.array(inspector_r_BN_N))
np.save(os.path.join(data_dir, "target_r_BN_N.npy"), np.array(target_r_BN_N))

print("Data saved successfully in 'data/' folder.")