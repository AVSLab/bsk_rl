from gymnasium.envs.registration import register

from bsk_rl._check_bsk_version import _check_bsk_version

register(
    id="GeneralSatelliteTasking-v1",
    entry_point="bsk_rl.env.gym_env:GeneralSatelliteTasking",
)

register(
    id="SingleSatelliteTasking-v1",
    entry_point="bsk_rl.env.gym_env:SingleSatelliteTasking",
)


_check_bsk_version()
