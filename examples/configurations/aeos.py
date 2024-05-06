class TargetInfoSat(
    sa.ImagingActions,
    so.TimeState,
    DensityState.configure(
        density_interval=60 * 5, density_windows=20, density_normalization=5
    ),
    so.TargetState.configure(
        target_properties=[
            dict(prop="priority"),
            dict(prop="r_TB_H", norm=800 * 1e3),
            dict(prop="theta_error", norm=np.pi / 2),
            dict(prop="omega_error", norm=0.03),
        ]
    ),
    so.NormdPropertyState.configure(
        obs_properties=[
            dict(prop="omega_BH_H", module="dynamics", norm=0.03),
            dict(prop="c_hat_H", module="fsw"),
            dict(prop="r_BN_P", module="dynamics", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", module="dynamics", norm=7616.5),
        ]
    ),
    sats.SteeringImagerSatellite,
):
    pass


class TargetInfoSat(sats.SteeringImagerSatellite, sa.ImagingActions):
    observation_spec = [
        obs.SatProperty(
            dict(prop="priority"),
            dict(prop="r_TB_H", norm=800 * 1e3),
            dict(prop="theta_error", norm=np.pi / 2),
            dict(prop="omega_error", norm=0.03),
        ),
        obs.TargetProperties(
            dict(prop="omega_BH_H", module="dynamics", norm=0.03),
            dict(prop="c_hat_H", module="fsw"),
            dict(prop="r_BN_P", module="dynamics", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", module="dynamics", norm=7616.5),
            n_ahead_observe=32,
        ),
        obs.Time(),
    ]
