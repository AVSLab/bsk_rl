from gymnasium.envs.registration import register

register(id="SimpleEOS-v0", entry_point="bsk_rl.envs.SimpleEOS.gym_env:SimpleEOS")

register(
    id="MultiSensorEOS-v0",
    entry_point="bsk_rl.envs.MultiSensorEOS.gym_env:MultiSensorEOS",
)

register(id="AgileEOS-v0", entry_point="bsk_rl.envs.AgileEOS.gym_env:AgileEOS")

register(
    id="MultiSatAgileEOS-v0",
    entry_point="bsk_rl.envs.MultiSatAgileEOS.gym_env:MultiSatAgileEOS",
)

register(
    id="SmallBodyScience-v0",
    entry_point="bsk_rl.envs.SmallBodyScience.gym_env:SmallBodyScience",
)

register(
    id="SmallBodySciencePOMDP-v0",
    entry_point="bsk_rl.envs.SmallBodySciencePOMDP.gym_env:SmallBodySciencePOMDP",
)
