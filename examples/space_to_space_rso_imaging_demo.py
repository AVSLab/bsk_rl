"""A small public space-to-space RSO imaging example.

Run ``python examples/space_to_space_rso_imaging_demo.py`` after installing BSK-RL.
The demo uses standard ground stations and saves a replayable catalog/manifest.
It requires neither a trained policy nor Vizard. The
:doc:`accompanying notebook <space_to_space_rso_imaging>` walks through each
configuration step and explains acquisition, delivery, and replay.

Ground-station support is explicitly composed with the spacecraft imaging
dynamics. The native Basilisk instrument controller checks pointing, access,
illumination, and hold timing before the instrument generates data. The Python
image recorder confirms that a complete product reached storage. Downlink reward
is assigned only when all its bits have left the imager's storage.

``make_rewarder`` defines the example's 90% acquisition and 10% delivery weights
as callbacks. The reusable rewarder has no mission-specific alpha parameter.
The optional AMOS profile lives in this example and preserves the 124-input,
13-action schema used by its research policies. It does not reconstruct paper
conditions without a saved mission manifest.
"""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import ClassVar

import numpy as np
from Basilisk import __file__ as basilisk_path
from Basilisk.utilities import orbitalMotion

from bsk_rl import ConstellationTasking, act, data, obs, sats, scene
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import random_circular_orbit, rv2HN

EPOCH = "2026 JUN 21 00:00:00.000 (UTC)"


def sun_in_hill(satellite):
    """Dimensionless Sun direction in the imager Hill frame, as in AMOS v9."""
    world = satellite.simulator.world
    sun = np.asarray(
        world.gravFactory.spiceObject.planetStateOutMsgs[world.sun_index]
        .read()
        .PositionVector
    )
    relative = sun - satellite.dynamics.r_BN_N
    hill = rv2HN(satellite.dynamics.r_BN_N, satellite.dynamics.v_BN_N) @ relative
    return hill / np.linalg.norm(hill)


def make_catalog(seed=0, n_targets=3):
    """Realize target conditions with an isolated RNG, then save/replay the result."""
    rng = np.random.default_rng(seed)
    targets = []
    for index in range(n_targets):
        elements = orbitalMotion.ClassicElements()
        elements.a, elements.e = 7.2e6 + index * 1e5, 0.0
        elements.i = np.radians(rng.uniform(0, 90))
        elements.Omega, elements.omega = 0.0, 0.0
        elements.f = rng.uniform(0, 0.1)
        rN, vN = orbitalMotion.elem2rv(orbitalMotion.MU_EARTH * 1e9, elements)
        targets.append(
            scene.RSOTarget(
                f"rso-{83 + 17 * index}",
                f"object_{index}",
                rN,
                vN,
                priority=float(index + 1),
            )
        )
    return scene.RSOTargetCatalog(tuple(targets))


def amos_observation_spec(n_candidates=10):
    """AMOS v9 full-action layout: 14 global + 11*K target features."""
    return [
        obs.SatProperties(
            dict(prop="storage_level_fraction"),
            dict(prop="battery_charge_fraction"),
            dict(prop="wheel_speeds_fraction"),
            dict(prop="s_hat_H", fn=sun_in_hill),
        ),
        obs.Eclipse(norm=5700),
        obs.OpportunityProperties(
            dict(prop="opportunity_open", norm=5700),
            dict(prop="opportunity_close", norm=5700),
            type="ground_station",
            n_ahead_observe=2,
        ),
        obs.RSOTargetProperties(n_ahead_observe=n_candidates),
    ]


def make_rewarder(acquisition_weight=0.0, delivery_weight=1.0, **kwargs):
    """Define this example's reward recipe outside the generic image rewarder.

    Callbacks receive an image record and the current runtime target. The default
    recipe assigns all priority credit to useful complete delivery. To provide
    an earlier training signal, pass acquisition_weight=0.9, delivery_weight=0.1.
    The explicit weights can be saved and reconstructed by the replay helper.
    """
    rewarder = data.RSOImageReward(
        acquisition_reward_fn=lambda record, target: acquisition_weight
        * target.priority,
        delivery_reward_fn=lambda record, target: delivery_weight * target.priority,
        **kwargs,
    )
    rewarder.example_weights = dict(
        acquisition_weight=acquisition_weight, delivery_weight=delivery_weight
    )
    return rewarder


def build_environment(catalog=None, *, seed=0, amos_profile=False, time_limit=1200):
    """Construct one imager and explicitly bound passive spacecraft targets.

    The AMOS profile preserves observation dimensions, action indices, hold mode,
    and alpha mapping. Correct acquisition/delivery semantics intentionally differ
    from the historical bugs; this is not a bit-for-bit research reproduction.

    Catalogs can omit their epoch. This example fixes the world epoch once, and
    the scene inherits its realized value. An explicit catalog epoch is also
    used for the world. Save ``env.scenario.catalog`` after reset for replay.
    """
    if catalog is None:
        catalog = make_catalog(seed)
    count = 10 if amos_profile else 3

    class Imager(sats.AccessSatellite):
        dyn_type = (dyn.SpaceToSpaceImagingDynModel, dyn.GroundStationDynModel)
        fsw_type = fsw.SpaceToSpaceImagingFSWModel
        observation_spec: ClassVar[list[obs.Observation]] = (
            amos_observation_spec(count)
            if amos_profile
            else [
                obs.SatProperties(dict(prop="storage_level_fraction")),
                obs.RSOTargetProperties(
                    dict(prop="valid"),
                    dict(prop="priority"),
                    dict(prop="target_distance"),
                    dict(prop="target_illumination_factor"),
                    n_ahead_observe=count,
                ),
            ]
        )
        action_spec: ClassVar[list[act.Action]] = [
            act.Charge(duration=300 if amos_profile else 60),
            act.Downlink(duration=300 if amos_profile else 90),
            act.Desat(duration=150 if amos_profile else 60),
            act.ImageRSO(
                count,
                max_duration=300,
                min_pointing_hold_s=10,
                hold_mode="cumulative",
                require_illumination_during_hold=False,
            ),
        ]

    class PassiveRSO(sats.Satellite):
        dyn_type = dyn.RSOTargetDynModel
        fsw_type = fsw.FSWModel
        observation_spec: ClassVar[list[obs.Observation]] = [obs.Time()]
        action_spec: ClassVar[list[act.Action]] = [act.Drift(duration=1e9)]

    scanner_args = dict(
        oe=random_circular_orbit(i=45.0, alt=600.0, Omega=0.0, f=0.0),
        sigma_init=[0, 0, 0],
        omega_init=[0, 0, 0],
        wheelSpeeds=[0.0, 0.0, 0.0],
        maxWheelSpeed=6000.0,
        batteryStorageCapacity=1e9,
        storedCharge_Init=1e9,
    )
    if amos_profile:
        scanner_args.update(
            imageAttErrorRequirement=0.0025,
            imageRateErrorRequirement=0.01,
            dataStorageCapacity=50 * 4e6,
            instrumentBaudRate=4e6,
            transmitterBaudRate=-4e6,
            batteryStorageCapacity=500 * 3600,
            storedCharge_Init=500 * 3600,
            basePowerDraw=-10.0,
            instrumentPowerDraw=-30.0,
            transmitterPowerDraw=-25.0,
            thrusterPowerDraw=-80.0,
            panelArea=1.0,
            maxWheelSpeed=6000.0,
            desatAttitude="sun",
        )
    scanner = Imager(
        "imager",
        sat_args=scanner_args,
    )
    targets = [PassiveRSO(target.satellite_name) for target in catalog.targets]
    return ConstellationTasking(
        satellites=[*targets, scanner],
        scenario=scene.RSOTargets(catalog, [scanner.name]),
        rewarder=make_rewarder(
            acquisition_weight=0.9 if amos_profile else 0.0,
            delivery_weight=0.1 if amos_profile else 1.0,
            cooldown_s=0,
        ),
        world_args=dict(
            utc_init=catalog.utc_init if catalog.utc_init is not None else EPOCH
        ),
        time_limit=time_limit,
        max_step_duration=300,
        log_level="ERROR",
    )


def mission_manifest(env, *, checkpoint=None):
    """Record realized mission settings; keep checkpoint details explicit if supplied."""

    def serializable(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return [serializable(item) for item in value]
        if isinstance(value, dict):
            return {key: serializable(item) for key, item in value.items()}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, orbitalMotion.ClassicElements):
            return {
                "orbital_elements": {
                    key: float(getattr(value, key))
                    for key in ("a", "e", "i", "Omega", "omega", "f")
                }
            }
        if isinstance(value, float) and not np.isfinite(value):
            return {"special_float": str(value)}
        if value is None or isinstance(value, (str, float, int, bool)):
            return value
        raise TypeError(f"Cannot serialize mission value {type(value)}")

    imager = next(
        sat for sat in env.satellites if sat.name in env.scenario.imager_names
    )
    action_settings = []
    for action in imager.action_builder.action_spec:
        fields = {
            key: value
            for key, value in vars(action).items()
            if key not in ("satellite", "simulator", "event_name", "chosen_target_ids")
        }
        action_settings.append(
            dict(type=type(action).__name__, settings=serializable(fields))
        )
    catalog = env.scenario.catalog.to_dict()
    import bsk_rl

    source = Path(bsk_rl.__file__).resolve()
    try:
        # An installed package inside another repository's .venv must not inherit
        # that repository's HEAD as its own source revision.
        subprocess.check_output(
            [
                "git",
                "-C",
                str(source.parent),
                "ls-files",
                "--error-unmatch",
                str(source),
            ],
            stderr=subprocess.DEVNULL,
        )
        commit = subprocess.check_output(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return dict(
        schema_version=1,
        catalog=catalog,
        catalog_sha256=hashlib.sha256(
            json.dumps(catalog, sort_keys=True).encode()
        ).hexdigest(),
        seed=env.seed,
        amos_profile=imager.observation_space.shape == (124,),
        numpy_random_state=serializable(np.random.get_state()),
        bsk_rl_commit=commit,
        bsk_rl_import_path=str(source),
        bsk_rl_source_tree_sha256=hashlib.sha256(
            "".join(
                hashlib.sha256(path.read_bytes()).hexdigest()
                + "  "
                + path.relative_to(source.parent).as_posix()
                + "\n"
                for path in sorted(source.parent.rglob("*.py"))
            ).encode()
        ).hexdigest(),
        python=platform.python_version(),
        versions={
            name: importlib.metadata.version(name)
            for name in ("bsk_rl", "bsk", "numpy", "gymnasium")
        },
        basilisk_import_path=basilisk_path,
        basilisk_binary_hashes={
            str(path.relative_to(Path(basilisk_path).parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(Path(basilisk_path).parent.rglob("*.so"))
        },
        world_args=serializable(env.world_args),
        satellite_args={sat.name: serializable(sat.sat_args) for sat in env.satellites},
        rates=dict(
            sim_s=env.sim_rate,
            dynamics_s=imager.dynamics.dyn_rate,
            fsw_s=imager.fsw.fsw_rate,
        ),
        time_limit_s=env.time_limit,
        max_step_duration_s=env.max_step_duration,
        observation_keys=imager.observation_builder.obs_array_keys(),
        observation_dtype=str(imager.observation_builder.dtype),
        rso_properties=[
            spec.properties
            for spec in imager.observation_builder.observation_spec
            if isinstance(spec, obs.RSOTargetProperties)
        ],
        actions=action_settings,
        reward=dict(
            **env.rewarder.example_weights,
            quality_threshold=env.rewarder.quality_threshold,
            cooldown_s=env.rewarder.cooldown_s,
            multi_imager_credit=env.rewarder.multi_imager_credit,
        ),
        checkpoint=checkpoint,
    )


def replay_environment(manifest):
    """Freeze scanner and world separately from the replayed target catalog.

    This helper supports this example's two profiles. For a different environment,
    replay the catalog and apply the manifest to that environment's own constructor.
    """

    def restore(value):
        if isinstance(value, list):
            return [restore(item) for item in value]
        if isinstance(value, dict):
            if "orbital_elements" in value:
                elements = orbitalMotion.ClassicElements()
                for key, item in value["orbital_elements"].items():
                    setattr(elements, key, item)
                return elements
            if "special_float" in value:
                return float(value["special_float"])
            return {key: restore(item) for key, item in value.items()}
        return value

    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported mission manifest version.")
    digest = hashlib.sha256(
        json.dumps(manifest["catalog"], sort_keys=True).encode()
    ).hexdigest()
    if digest != manifest["catalog_sha256"]:
        raise ValueError("Mission catalog hash does not match manifest.")
    env = build_environment(
        scene.RSOTargetCatalog.from_dict(manifest["catalog"]),
        amos_profile=manifest["amos_profile"],
        time_limit=manifest["time_limit_s"],
    )
    env.world_args_generator = restore(manifest["world_args"])
    for satellite in env.satellites:
        satellite.sat_args_generator = restore(
            manifest["satellite_args"][satellite.name]
        )
    env.sim_rate = manifest["rates"]["sim_s"]
    env.max_step_duration = manifest["max_step_duration_s"]
    for satellite in env.satellites:
        if satellite.name in env.scenario.imager_names:
            for action, saved in zip(
                satellite.action_builder.action_spec, manifest["actions"]
            ):
                if type(action).__name__ != saved["type"]:
                    raise ValueError("Manifest action layout does not match example.")
                for key, value in saved["settings"].items():
                    setattr(action, key, restore(value))
    reward_settings = dict(manifest["reward"])
    # Replay manifests written before stage weights moved into example callbacks.
    if "alpha" in reward_settings:
        alpha = reward_settings.pop("alpha")
        reward_settings.update(acquisition_weight=1 - alpha, delivery_weight=alpha)
    env.rewarder = make_rewarder(**reward_settings)
    env.rewarder.link_scenario(env.scenario)
    return env


def run_demo(output_directory=None):
    """Run a bounded acquisition/downlink demo using standard ground access."""
    catalog = scene.RSOTargetCatalog.from_dict(make_catalog().to_dict())
    env = build_environment(catalog)
    env.reset(seed=0)
    manifest = mission_manifest(env)
    if output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        env.scenario.catalog.save(output_directory / "catalog.json")
        (output_directory / "mission.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    reward_total = 0.0
    for step in range(30):
        imager = next(sat for sat in env.satellites if sat.name == "imager")
        store = imager.data_store
        # Downlink stored images; otherwise image a valid observation slot.
        if store.products:
            choice = 1
        else:
            row = imager.observation_builder.obs_dict()["rso_targets"]["rso_targets_0"]
            choice = 3 if row["valid"] else 0
        _, reward, terminated, truncated, _ = env.step(
            {sat.name: choice if sat.name == "imager" else 0 for sat in env.satellites}
        )
        reward_total += reward.get("imager", 0.0)
        if store.data.deliveries or all(terminated.values()) or all(truncated.values()):
            break
    result = dict(
        sim_time_s=env.simulator.sim_time,
        captures=len(env.rewarder.data.captures),
        deliveries=len(env.rewarder.data.deliveries),
        reward=reward_total,
    )
    env.close()
    return result


if __name__ == "__main__":
    print(json.dumps(run_demo("rso_imaging_demo"), indent=2))
