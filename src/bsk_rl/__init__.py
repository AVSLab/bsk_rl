from gymnasium.envs.registration import register

register(id="SimpleEOS-v0", entry_point="bsk_rl.envs.simple_eos.gym_env:SimpleEOS")

register(
    id="MultiSensorEOS-v0",
    entry_point="bsk_rl.envs.multisensor_eos.gym_env:MultiSensorEOS",
)

register(id="AgileEOS-v0", entry_point="bsk_rl.envs.agile_eos.gym_env:AgileEOS")

register(
    id="MultiSatAgileEOS-v0",
    entry_point="bsk_rl.envs.multisat_agile_eos.gym_env:MultiSatAgileEOS",
)

register(
    id="SmallBodyScience-v0",
    entry_point="bsk_rl.envs.small_body_science.gym_env:SmallBodyScience",
)

register(
    id="SmallBodySciencePOMDP-v0",
    entry_point="bsk_rl.envs.small_body_science_pomdp.gym_env:SmallBodySciencePOMDP",
)

register(
    id="GeneralSatelliteTasking-v1",
    entry_point="bsk_rl.envs.general_satellite_tasking.gym_env:GeneralSatelliteTasking",
)

register(
    id="SingleSatelliteTasking-v1",
    entry_point="bsk_rl.envs.general_satellite_tasking.gym_env:SingleSatelliteTasking",
)
