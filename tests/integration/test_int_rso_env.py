import types
from functools import partial

import numpy as np
import pytest

from bsk_rl import ConstellationTasking, act, data, obs, sats, scene
from bsk_rl.obs.relative_observations import rso_imaged_regions
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import fibonacci_sphere, random_circular_orbit


class RSOSat(sats.Satellite):
    observation_spec = [
        obs.SatProperties(dict(prop="one", fn=lambda _: 1.0)),
    ]
    action_spec = [act.Drift(duration=1e9)]
    dyn_type = dyn.RSODynModel
    fsw_type = fsw.BasicFSWModel


class InspectorSat(sats.Satellite):
    observation_spec = [
        obs.RelativeProperties(
            dict(
                prop="rso_imaged_regions",
                fn=partial(
                    rso_imaged_regions,
                    region_centers=fibonacci_sphere(15),
                    frame="chief_hill",
                ),
            ),
            chief_name="RSO",
        ),
        obs.Eclipse(),
    ]
    action_spec = [
        act.ImpulsiveThrustHill(
            chief_name="RSO",
            max_dv=1.0,
            max_drift_duration=5700.0 * 2,
            fsw_action="action_inspect_rso",
        )
    ]
    dyn_type = types.new_class("Dyn", (dyn.MaxRangeDynModel, dyn.RSOInspectorDynModel))
    fsw_type = types.new_class(
        "FSW",
        (
            fsw.SteeringFSWModel,
            fsw.MagicOrbitalManeuverFSWModel,
            fsw.RSOInspectorFSWModel,
        ),
    )


def inspector_sat_args(**kwargs):
    return dict(
        imageAttErrorRequirement=1.0,
        imageRateErrorRequirement=None,
        instrumentBaudRate=1,
        dataStorageCapacity=1e6,
        batteryStorageCapacity=1e9,
        storedCharge_Init=1e9,
        dv_available_init=10.0,
        chief_name="RSO",
        u_max=1.0,
        **kwargs,
    )


@pytest.mark.repeat(10)
def test_inspection():
    env = ConstellationTasking(
        satellites=[
            RSOSat(
                "RSO",
                sat_args=dict(oe=random_circular_orbit(i=0, alt=500, Omega=0, f=0)),
            ),
            InspectorSat(
                "Inspector",
                sat_args=inspector_sat_args(
                    oe=random_circular_orbit(i=0, alt=500.01, Omega=0, f=0)
                ),
                obs_type=dict,
            ),
        ],
        scenario=scene.SphericalRSO(
            n_points=100,
            radius=1.0,
            theta_max=np.radians(90),  # Generous observation angle
            range_max=250,
            theta_solar_max=np.radians(90),  # Generous illumination angle
        ),
        rewarder=(data.RSOInspectionReward()),
        time_limit=60000,
        sim_rate=1.0,
        # log_level="INFO",
    )

    observation, _ = env.reset()
    in_eclipse_start = (
        observation["Inspector"]["eclipse"][0] > observation["Inspector"]["eclipse"][1]
    )

    duration = 100
    observation, reward, _, _, _ = env.step(
        dict(RSO=0, Inspector=[0.0, 0.0, 0.0, duration])
    )
    in_eclipse_end = (
        observation["Inspector"]["eclipse"][0] > observation["Inspector"]["eclipse"][1]
    )

    # Check that nothing is imaged in eclipse
    if in_eclipse_start and in_eclipse_end:
        assert sum(observation["Inspector"]["rel_props"]["rso_imaged_regions"]) == 0
    else:
        # Orbital setup means that if not in eclipse, inspector is on the illuminated side
        assert sum(observation["Inspector"]["rel_props"]["rso_imaged_regions"]) > 0

    # Check that reward was yielded correctly
    if sum(observation["Inspector"]["rel_props"]["rso_imaged_regions"] > 0):
        assert reward["Inspector"] > 0
    else:
        assert "Inspector" not in reward or reward["Inspector"] == 0
