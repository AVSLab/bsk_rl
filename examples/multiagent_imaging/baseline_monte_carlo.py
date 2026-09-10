"""Matched deterministic three-sensor baselines, isolated from learned-policy studies.

The centralized controller is a full-state *greedy controller*, not a learned
policy or a proof of optimal coverage. It may inspect every sensor's current
execution/resource state and completion facts. The independent controller sees
only its own spacecraft, own catalog, and the declared target ephemerides.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
from Basilisk.utilities import macros, orbitalMotion

from bsk_rl import NO_ACTION
from bsk_rl.obs.completion_observations import NON_IMAGING_ACTIONS, candidate_snapshot
from bsk_rl.obs.observations import _angle_to_target, _target_shadowFactor
from bsk_rl.utils.coordination import earth_unoccluded
from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.environment import R_EARTH_M, build_environment

CAMPAIGN_VERSION = "three-sensor-full-state-baselines-v2"
CASES = ("independent", "centralized_full_state")
ENVIRONMENTS = ("leo", "mixed")
CELLS = tuple((case, regime) for case in CASES for regime in ENVIRONMENTS)
ALTITUDE_BANDS_M = {
    "LEO": (400e3, 2000e3),
    "MEO": (2000e3, 35000e3),
    "GEO": (35786e3 - 300e3, 35786e3 + 300e3),
}


@dataclass(frozen=True)
class BaselineConfig:
    """A separate experiment contract; saved completion-v2 checkpoints do not change."""

    n_sensors: int = 3
    n_targets: int = 100
    n_candidates: int = 10
    episode_duration_s: float = 45000.0
    max_step_duration_s: float = 300.0
    imaging_duration_s: float = 300.0
    downlink_duration_s: float = 300.0
    charge_duration_s: float = 300.0
    desat_duration_s: float = 150.0
    min_pointing_hold_s: float = 10.0
    reimage_cooldown_orbits: float = 2.0
    alpha: float = 0.1
    battery_charge_threshold: float = 0.3
    wheel_desaturation_threshold: float = 0.7
    storage_downlink_threshold: float = 0.8
    retasking_mode: str = "conflict"

    def __post_init__(self):
        if self.retasking_mode != "conflict":
            raise ValueError("This four-cell campaign fixes conflict retasking.")
        for name in (
            "battery_charge_threshold",
            "wheel_desaturation_threshold",
            "storage_downlink_threshold",
        ):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be between zero and one.")
        # Reuse the physics configuration validation without extending its schema.
        self.environment_config("independent", 0)

    def environment_config(self, case, seed):
        if case not in CASES:
            raise ValueError(f"Unknown information case: {case}")
        physics = asdict(self)
        for key in (
            "battery_charge_threshold",
            "wheel_desaturation_threshold",
            "storage_downlink_threshold",
        ):
            physics.pop(key)
        return MultiAgentImagingConfig(
            **physics,
            seed=int(seed),
            information_case="independent"
            if case == "independent"
            else "ideal_completion",
            communication_mode="broadcast",
            communication_cost_per_s=0.0,
        )


def task_spec(task_id):
    """Map one Slurm array index to a case and its matched seed, exactly once."""
    if isinstance(task_id, bool) or int(task_id) != task_id or not 0 <= task_id < 200:
        raise ValueError("task_id must be an integer in 0..199.")
    case, regime = CELLS[int(task_id) // 50]
    return {
        "task_id": int(task_id),
        "case": case,
        "target_environment": regime,
        "seed": int(task_id) % 50,
    }


def exact_regimes(count, environment, seed):
    """AMOS evaluation's 50/30/20 mix, largest remainders, then seeded ID shuffle."""
    if environment == "leo":
        return ["LEO"] * count
    if environment != "mixed":
        raise ValueError("target_environment must be leo or mixed.")
    raw = count * np.asarray([0.5, 0.3, 0.2])
    counts = np.floor(raw).astype(int)
    order = sorted(range(3), key=lambda i: (-(raw[i] - counts[i]), i))
    for index in order[: count - int(counts.sum())]:
        counts[index] += 1
    names = [r for r, n in zip(ALTITUDE_BANDS_M, counts) for _ in range(n)]
    return list(np.random.default_rng(seed).permutation(names))


def sample_orbit(regime):
    """Reuse the documented AMOS altitude, eccentricity, inclination distributions.

    Draw from the environment's reset-seeded NumPy stream, not wall-clock entropy.
    Unlike the old AMOS speed shortcut, these targets remain live spacecraft.
    """
    orbit = orbitalMotion.ClassicElements()
    orbit.a = R_EARTH_M + np.random.uniform(*ALTITUDE_BANDS_M[regime])
    e_max = {"LEO": 0.02, "MEO": 0.10, "GEO": 0.0015}[regime]
    orbit.e = np.random.uniform(0.0, e_max)
    while orbit.a * (1 - orbit.e) < R_EARTH_M + 400e3:
        orbit.e = np.random.uniform(0.0, e_max)
    orbit.i = (
        np.random.uniform(0.0, {"LEO": 180, "MEO": 120, "GEO": 15}[regime]) * macros.D2R
    )
    orbit.Omega, orbit.omega, orbit.f = np.random.uniform(0.0, 360.0, 3) * macros.D2R
    return orbit


def build_baseline(config, case, target_environment, seed):
    """Configure orbital randomizers before reset, leaving the training builder intact."""
    env = build_environment(config.environment_config(case, seed))
    regimes = exact_regimes(config.n_targets, target_environment, seed)
    for target, regime in zip(env.passive_satellites, regimes):
        target.sat_args_generator["oe"] = partial(sample_orbit, regime)
        target.baseline_regime = regime
    return env


def _operational_action(sensor, config):
    """The same own-resource rules apply in both information cases."""
    if sensor.dynamics.battery_charge_fraction < config.battery_charge_threshold:
        return 0
    if (
        np.max(np.abs(sensor.dynamics.wheel_speeds_fraction))
        > config.wheel_desaturation_threshold
    ):
        return 2
    storage = sensor.dynamics.storage_level_fraction
    contact = any(
        bool(message.read().hasAccess)
        for message in sensor.dynamics.ground_station_access_messages.values()
    )
    if storage > 0 and (contact or storage >= config.storage_downlink_threshold):
        return 1
    return None


def local_choices(sensor, config):
    """Build legal choices using only this sensor's state, catalog and target geometry.

    The candidate ID tuple is precisely the tuple used by ImageCompletion's
    decoder. Avoid imaging through Earth or selecting currently shadowed targets.
    Coverage-first means never-observed targets rank before revisits *within the
    ten-slot shortlist*; it is not an omniscient all-catalog optimizer.
    """
    if not sensor.requires_retasking:
        return [
            {
                "action": NO_ACTION,
                "target_id": None,
                "new": 0,
                "angle": 0.0,
                "priority": 0.0,
            }
        ]
    operational = _operational_action(sensor, config)
    if operational is not None:
        return [
            {
                "action": operational,
                "target_id": None,
                "new": 0,
                "angle": 0.0,
                "priority": 0.0,
            }
        ]
    choices = []
    observed = {
        r.target_id for r in sensor.data_store.catalog.records.values() if r.qualified
    }
    for slot, target in enumerate(
        candidate_snapshot(sensor, config.n_candidates).targets
    ):
        if target is None:
            continue
        opportunity = {"object": target}
        if not earth_unoccluded(
            sensor.dynamics.r_BN_N, target.target_spacecraft.dynamics.r_BN_N
        ):
            continue
        if _target_shadowFactor(sensor, opportunity) < 0.5:
            continue
        angle = float(_angle_to_target(sensor, opportunity))
        if not math.isfinite(angle):
            continue
        choices.append(
            {
                "action": NON_IMAGING_ACTIONS + slot,
                "target_id": int(target.id),
                "new": int(target.id not in observed),
                "angle": angle,
                "priority": float(target.priority),
            }
        )
    choices.sort(
        key=lambda row: (-row["new"], row["angle"], -row["priority"], row["target_id"])
    )
    # Charge is a legal fallback if geometry leaves no useful candidate, or when
    # the central assignment gives the only remaining candidate to the teammate.
    choices.append(
        {"action": 0, "target_id": None, "new": 0, "angle": 0.0, "priority": 0.0}
    )
    return choices


def choose_joint(options, reserved=()):
    """Exactly enumerate the current multi-sensor greedy assignment.

    This joint controller knows ongoing assignments. Maximize new-target jobs,
    then total image jobs; minimize aggregate current pointing angle. It has no
    future-trajectory optimizer and cannot guarantee globally optimal coverage.
    """
    names = sorted(options)
    feasible = []
    for rows in product(*(options[name] for name in names)):
        ids = [row["target_id"] for row in rows if row["target_id"] is not None]
        if len(set(ids)) != len(ids) or set(ids).intersection(reserved):
            continue
        key = (
            -sum(row["new"] for row in rows),
            -len(ids),
            sum(row["angle"] for row in rows),
            -sum(row["priority"] for row in rows),
            tuple(row["action"] for row in rows),
        )
        feasible.append((key, rows))
    if not feasible:
        raise RuntimeError("Central assignment must retain an operational fallback.")
    rows = min(feasible, key=lambda item: item[0])[1]
    return {name: row["action"] for name, row in zip(names, rows)}


def centralized_state_snapshot(env):
    """Expose every mission-relevant sensor state to the centralized controller.

    This snapshot is an explicit audit boundary.  It contains live navigation,
    attitude/resources, physical product ownership, durable completion records,
    and the active task for every sensing spacecraft.  It is used only by the
    no-radio centralized baseline and never enters a learned policy observation.
    """
    now = float(env.simulator.sim_time)
    sensors = {}
    for sensor in env.sensing_satellites:
        if sensor.name not in env.agents:
            continue
        task = sensor.completion_task
        sensors[sensor.name] = {
            "position_N_m": np.asarray(sensor.dynamics.r_BN_N).tolist(),
            "velocity_N_m_s": np.asarray(sensor.dynamics.v_BN_N).tolist(),
            "attitude_MRP": np.asarray(sensor.dynamics.sigma_BN).tolist(),
            "body_rate_rad_s": np.asarray(sensor.dynamics.omega_BN_B).tolist(),
            "battery_fraction": float(sensor.dynamics.battery_charge_fraction),
            "storage_fraction": float(sensor.dynamics.storage_level_fraction),
            "wheel_fraction": np.asarray(
                sensor.dynamics.wheel_speeds_fraction
            ).tolist(),
            "requires_retasking": bool(sensor.requires_retasking),
            "active_task": (
                None
                if task is None
                else {
                    "mode": task.mode,
                    "target_id": task.target_id,
                    "start": task.start,
                    "deadline": task.deadline,
                    "finished": bool(task.finished),
                }
            ),
            "onboard_products": [
                asdict(product) for product in sensor.data_store.products
            ],
            "completion_records": [
                asdict(record)
                for record in sensor.data_store.catalog.records.values()
            ],
            "request_epochs": dict(sensor.data_store.catalog.request_epochs),
        }
    return {"time_s": now, "sensors": sensors}


def _globally_reserved_targets(snapshot, cooldown_s):
    """Find targets a full-information coordinator knows are already serviced.

    Active imaging jobs reserve their targets immediately.  Qualified completed
    exposures reserve the matching request generation until the existing
    capture-anchored cooldown expires, whether or not the image has reached ground.
    """
    now = snapshot["time_s"]
    reserved = set()
    for state in snapshot["sensors"].values():
        task = state["active_task"]
        if (
            task is not None
            and task["mode"] == "image"
            and not task["finished"]
            and task["target_id"] is not None
        ):
            reserved.add(int(task["target_id"]))
        for record in state["completion_records"]:
            target_id = int(record["target_id"])
            if (
                record["qualified"]
                and float(record["request_epoch"])
                == float(state["request_epochs"][target_id])
                and float(record["capture_time"]) + float(cooldown_s) > now
            ):
                reserved.add(target_id)
    return reserved


def _update_central_audit(audit, snapshot):
    """Record compact proof that every live sensor was read at a decision boundary."""
    if audit is None:
        return
    states = snapshot["sensors"].values()
    audit["decision_boundaries"] += 1
    audit["sensor_state_reads"] += len(snapshot["sensors"])
    audit["minimum_sensors_visible"] = min(
        audit["minimum_sensors_visible"], len(snapshot["sensors"])
    )
    audit["maximum_catalog_records_visible"] = max(
        audit["maximum_catalog_records_visible"],
        sum(len(state["completion_records"]) for state in states),
    )
    audit["maximum_onboard_products_visible"] = max(
        audit["maximum_onboard_products_visible"],
        sum(len(state["onboard_products"]) for state in states),
    )
    audit["last_snapshot_sha256"] = digest(snapshot)


def choose_actions(env, config, case, *, central_audit=None):
    """Choose one event-boundary action per live sensor under the selected contract."""
    sensors = [s for s in env.sensing_satellites if s.name in env.agents]
    options = {s.name: local_choices(s, config) for s in sensors}
    if case == "independent":
        # Do not inspect another sensor's choice/task/catalog in this branch.
        return {name: rows[0]["action"] for name, rows in options.items()}

    # The centralized reference reads a coherent full-team snapshot on every
    # asynchronous event boundary.  Use its global completion history both for
    # the coverage-first score and for current target reservations.
    snapshot = centralized_state_snapshot(env)
    _update_central_audit(central_audit, snapshot)
    globally_observed = {
        int(record["target_id"])
        for state in snapshot["sensors"].values()
        for record in state["completion_records"]
        if record["qualified"]
    }
    for rows in options.values():
        for row in rows:
            if row["target_id"] is not None:
                row["new"] = int(int(row["target_id"]) not in globally_observed)
    reserved = _globally_reserved_targets(snapshot, env.rewarder.reimage_cooldown_s)
    return choose_joint(options, reserved)


def initial_conditions(env):
    """Persist the physical state and catalog priorities that pairing must reproduce."""
    result = {}
    for satellite in env.satellites:
        row = {
            "position_N_m": np.asarray(satellite.dynamics.r_BN_N).tolist(),
            "velocity_N_m_s": np.asarray(satellite.dynamics.v_BN_N).tolist(),
            "role": satellite.role.value,
        }
        if satellite in env.passive_satellites:
            row.update(
                target_id=int(satellite.rso_target.id),
                priority=float(satellite.rso_target.priority),
                regime=satellite.baseline_regime,
            )
        else:
            row.update(
                battery_fraction=float(satellite.dynamics.battery_charge_fraction),
                storage_fraction=float(satellite.dynamics.storage_level_fraction),
                wheel_fraction=np.asarray(
                    satellite.dynamics.wheel_speeds_fraction
                ).tolist(),
                attitude_MRP=np.asarray(satellite.dynamics.sigma_BN).tolist(),
                body_rate_rad_s=np.asarray(satellite.dynamics.omega_BN_B).tolist(),
            )
        result[satellite.name] = row
    return result


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def source_record():
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    # Include tracked and untracked executable/configuration inputs. Documentation
    # updates (including this campaign's execution report) must not invalidate the
    # remaining paired tasks. HEAD is recorded separately for the full Git history.
    paths = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "src",
                "examples/multiagent_imaging",
                "pyproject.toml",
            ],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    hashes = {
        p: hashlib.sha256((root / p).read_bytes()).hexdigest()
        for p in sorted(set(paths))
        if p
        and (root / p).is_file()
        and Path(p).suffix
        in {".py", ".toml", ".json", ".slurm", ".sh", ".txt", ".yaml", ".yml"}
    }
    import Basilisk
    import importlib.metadata

    versions = {}
    for name in ("numpy", "scipy", "gymnasium", "pettingzoo", "torch", "ray"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "source_hashes": hashes,
        "source_fingerprint": digest(hashes),
        "python": sys.version,
        "python_executable": sys.executable,
        "basilisk_path": Basilisk.__file__,
        "basilisk_version": getattr(Basilisk, "__version__", None),
        "versions": versions,
        "platform": platform.platform(),
    }


def coverage_metrics(
    target_ids, captures, services, quality_threshold, sensor_names=None
):
    """Distinct target coverage and exposure/service counts are different quantities."""
    all_ids = set(target_ids)
    qualified = [p for p in captures if p.quality >= quality_threshold]
    deliveries = [
        s.product
        for s in services
        if s.product.quality >= quality_threshold
        and s.product.delivery_time is not None
    ]
    captured_ids = {p.target_id for p in qualified}
    delivered_ids = {p.target_id for p in deliveries}
    per_sensor = {}
    sensor_names = sorted(
        sensor_names
        if sensor_names is not None
        else {p.source_sensor for p in [*qualified, *deliveries]}
    )
    sources_by_target = {
        target_id: {p.source_sensor for p in qualified if p.target_id == target_id}
        for target_id in captured_ids
    }
    for sensor in sensor_names:
        captured = {p.target_id for p in qualified if p.source_sensor == sensor}
        delivered = {p.target_id for p in deliveries if p.source_sensor == sensor}
        per_sensor[sensor] = {
            "capture_target_ids": sorted(captured),
            "ground_delivery_target_ids": sorted(delivered),
            "capture_coverage_fraction": len(captured) / len(all_ids),
            "ground_delivery_coverage_fraction": len(delivered) / len(all_ids),
        }
    return {
        "catalog_target_count": len(all_ids),
        "capture_target_count": len(captured_ids),
        "ground_delivery_target_count": len(delivered_ids),
        "capture_coverage_fraction": len(captured_ids) / len(all_ids),
        "ground_delivery_coverage_fraction": len(delivered_ids) / len(all_ids),
        "capture_target_ids": sorted(captured_ids),
        "ground_delivery_target_ids": sorted(delivered_ids),
        "never_captured_target_ids": sorted(all_ids - captured_ids),
        "never_delivered_target_ids": sorted(all_ids - delivered_ids),
        "qualified_exposure_count": len(qualified),
        "unqualified_exposure_count": len(captures) - len(qualified),
        "qualified_ground_delivery_count": len(deliveries),
        # A target counts once when two or more distinct sensors acquired it.
        # This generalizes the former two-set intersection to any team size.
        "cross_sensor_capture_overlap_count": sum(
            len(sources) >= 2 for sources in sources_by_target.values()
        ),
        "per_sensor": per_sensor,
    }


def duplicate_product_metrics(captures, services, quality_threshold, episode_end_s):
    """Measure stale ground products and simultaneous cross-sensor ownership.

    The stale-delivery metric implements the user's first definition: a qualified
    ground-delivered product is stale if another sensor also delivered a newer
    capture of the same target.  The onboard metric implements the stricter
    coverage definition by integrating intervals in which different sensors
    physically held qualified products for the same target.
    """
    delivered = {
        entry.product.record_id: entry.product
        for entry in services
        if entry.product.quality >= quality_threshold
        and entry.product.delivery_time is not None
    }
    qualified = [p for p in captures if p.quality >= quality_threshold]

    stale_ids = set()
    causally_avoidable_ids = set()
    for image in delivered.values():
        newer = [
            other
            for other in delivered.values()
            if other.target_id == image.target_id
            and other.source_sensor != image.source_sensor
            and other.capture_time > image.capture_time
        ]
        if newer:
            stale_ids.add(image.record_id)
        if any(other.delivery_time <= image.delivery_time for other in newer):
            # This subset was already known to be stale at the time it arrived.
            causally_avoidable_ids.add(image.record_id)

    intervals_by_target = {}
    for image in qualified:
        delivered_version = delivered.get(image.record_id)
        end = (
            float(delivered_version.delivery_time)
            if delivered_version is not None
            else float(episode_end_s)
        )
        start = float(image.capture_time)
        if end + 1e-9 < start:
            raise ValueError("A product cannot leave storage before capture.")
        if end > start + 1e-9:
            intervals_by_target.setdefault(int(image.target_id), []).append(
                (start, end, image.source_sensor, image.record_id)
            )

    overlap_targets = set()
    overlap_products = set()
    redundant_acquisitions = 0
    redundant_sensor_time_s = 0.0
    overlap_sensor_time_s = 0.0
    for target_id, intervals in intervals_by_target.items():
        starts, ends = {}, {}
        for start, end, sensor, record_id in intervals:
            starts.setdefault(start, []).append((record_id, sensor))
            ends.setdefault(end, []).append((record_id, sensor))
        active = {}
        previous = min((*starts, *ends))
        for boundary in sorted(set(starts) | set(ends)):
            active_sensors = set(active.values())
            elapsed = boundary - previous
            if len(active_sensors) >= 2 and elapsed > 0:
                overlap_targets.add(target_id)
                overlap_products.update(active)
                redundant_sensor_time_s += (len(active_sensors) - 1) * elapsed
                overlap_sensor_time_s += len(active_sensors) * elapsed

            # Treat ownership as [capture, delivery): a delivery and a new capture
            # at exactly the same instant do not create an artificial overlap.
            for record_id, _ in ends.get(boundary, ()):
                active.pop(record_id, None)
            existing_sensors = set(active.values())
            incoming = starts.get(boundary, ())
            incoming_sensors = {sensor for _, sensor in incoming}
            redundant_acquisitions += max(
                0,
                len(existing_sensors | incoming_sensors)
                - max(1, len(existing_sensors)),
            )
            active.update(incoming)
            previous = boundary

    return {
        "stale_cross_sensor_ground_delivery_count": len(stale_ids),
        "stale_cross_sensor_ground_delivery_target_count": len(
            {delivered[record_id].target_id for record_id in stale_ids}
        ),
        "causally_avoidable_stale_ground_delivery_count": len(
            causally_avoidable_ids
        ),
        "stale_cross_sensor_ground_delivery_record_ids": sorted(stale_ids),
        "cross_sensor_onboard_overlap_target_count": len(overlap_targets),
        "cross_sensor_onboard_overlap_product_count": len(overlap_products),
        "cross_sensor_onboard_redundant_acquisition_count": redundant_acquisitions,
        "cross_sensor_onboard_redundant_sensor_time_s": redundant_sensor_time_s,
        "cross_sensor_onboard_overlap_sensor_time_s": overlap_sensor_time_s,
    }


def run_episode(config, specification):
    """Run one complete deterministic episode; no radio action or policy training."""
    started = time.perf_counter()
    env = build_baseline(
        config,
        specification["case"],
        specification["target_environment"],
        specification["seed"],
    )
    try:
        env.reset(seed=specification["seed"])
        initial = initial_conditions(env)
        resources = {s.name: [] for s in env.sensing_satellites}
        rewards = {s.name: 0.0 for s in env.sensing_satellites}
        action_counts = {s.name: Counter() for s in env.sensing_satellites}
        sampled = {
            int(t.rso_target.id): {"candidate_samples": 0, "illuminated_los_samples": 0}
            for t in env.passive_satellites
        }
        central_audit = {
            "enabled": specification["case"] == "centralized_full_state",
            "decision_boundaries": 0,
            "sensor_state_reads": 0,
            "minimum_sensors_visible": (
                config.n_sensors
                if specification["case"] == "centralized_full_state"
                else None
            ),
            "maximum_catalog_records_visible": 0,
            "maximum_onboard_products_visible": 0,
            "last_snapshot_sha256": None,
            "fields_visible": [
                "navigation position and velocity",
                "attitude and body rate",
                "battery, storage, and wheel state",
                "active task and target reservation",
                "physical onboard product metadata and owner",
                "durable time-tagged completion and ground-delivery records",
                "current request epochs",
            ],
        }
        steps = 0

        # Capture each event boundary, including reset; these sampled diagnostics
        # are not a proof that a never-visible target lacked an intervening window.
        def sample_state():
            for sensor in env.sensing_satellites:
                resources[sensor.name].append(
                    {
                        "time_s": float(env.simulator.sim_time),
                        "alive": bool(sensor.is_alive()),
                        "battery_fraction": float(
                            sensor.dynamics.battery_charge_fraction
                        ),
                        "storage_fraction": float(
                            sensor.dynamics.storage_level_fraction
                        ),
                        "max_wheel_fraction": float(
                            np.max(np.abs(sensor.dynamics.wheel_speeds_fraction))
                        ),
                    }
                )
                if sensor.name not in env.agents:
                    continue
                for target in env.passive_satellites:
                    if (
                        earth_unoccluded(sensor.dynamics.r_BN_N, target.dynamics.r_BN_N)
                        and _target_shadowFactor(sensor, {"object": target.rso_target})
                        >= 0.5
                    ):
                        sampled[int(target.rso_target.id)][
                            "illuminated_los_samples"
                        ] += 1
                if sensor.requires_retasking:
                    for target in candidate_snapshot(
                        sensor, config.n_candidates
                    ).targets:
                        if target is not None:
                            sampled[int(target.id)]["candidate_samples"] += 1

        sample_state()
        while env.agents:
            actions = choose_actions(
                env,
                config,
                specification["case"],
                central_audit=central_audit,
            )
            if 3 in actions.values():
                raise AssertionError(
                    "Baseline campaigns never execute a transmit/broadcast action."
                )
            for name, action in actions.items():
                action_counts[name][str(action)] += 1
            _, reward, terminated, truncated, _ = env.step(actions)
            for name, value in reward.items():
                rewards[name] += float(value)
            sample_state()
            steps += 1
            if all(terminated.values()) or all(truncated.values()):
                break
        captures = env.rewarder._team_accounting.capture_attempts
        services = env.rewarder.service_entries
        metrics = coverage_metrics(
            sampled,
            captures,
            services,
            env.rewarder.quality_threshold,
            sensor_names=[sensor.name for sensor in env.sensing_satellites],
        )
        coverage_by_regime = {}
        for regime in ALTITUDE_BANDS_M:
            ids = {
                int(t.rso_target.id)
                for t in env.passive_satellites
                if t.baseline_regime == regime
            }
            if ids:
                coverage_by_regime[regime] = coverage_metrics(
                    ids,
                    [p for p in captures if p.target_id in ids],
                    [s for s in services if s.product.target_id in ids],
                    env.rewarder.quality_threshold,
                    sensor_names=[sensor.name for sensor in env.sensing_satellites],
                )
        now = float(env.simulator.sim_time)
        product_duplicates = duplicate_product_metrics(
            captures, services, env.rewarder.quality_threshold, now
        )
        wall = time.perf_counter() - started
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
        return {
            "campaign_version": CAMPAIGN_VERSION,
            **specification,
            "controller": f"{specification['case']}_coverage_first_greedy",
            "baseline_config": asdict(config),
            "physics_config": config.environment_config(
                specification["case"], specification["seed"]
            ).to_dict(),
            "initial_conditions": initial,
            "initial_conditions_sha256": digest(initial),
            "pettingzoo_agents": list(env.possible_agents),
            "passive_target_count": len(env.passive_satellites),
            "regime_counts": dict(
                Counter(t.baseline_regime for t in env.passive_satellites)
            ),
            "reimage_cooldown_s": float(env.rewarder.reimage_cooldown_s),
            "sim_time_s": now,
            "horizon_reached": now >= config.episode_duration_s,
            "wall_time_s": wall,
            "simulated_seconds_per_wall_second": now / wall,
            "peak_process_rss_bytes": peak_rss,
            "event_steps": steps,
            "total_constellation_reward": sum(rewards.values()),
            "cumulative_reward": rewards,
            "coverage": metrics,
            "coverage_by_regime": coverage_by_regime,
            "product_duplicates": product_duplicates,
            "team_summary": env.rewarder.team_summary,
            "coordination": env.coordination_metrics(),
            "centralized_information_audit": central_audit,
            "action_counts": {k: dict(v) for k, v in action_counts.items()},
            "resource_history": resources,
            "sampled_target_geometry": sampled,
            "capture_records": [asdict(p) for p in captures],
            "ground_delivery_records": [asdict(entry.product) for entry in services],
            "completion_records": {
                s.name: [asdict(r) for r in s.data_store.catalog.records.values()]
                for s in env.sensing_satellites
            },
            "catalog_receipts": {
                s.name: s.data_store.catalog.version_received_at
                for s in env.sensing_satellites
            },
            "onboard_products": {
                s.name: [asdict(p) for p in s.data_store.products]
                for s in env.sensing_satellites
            },
            "communication": {
                "radio_action_count": 0,
                "ideal_information_updates": len(env.communicator.delivery_history),
            },
        }
    finally:
        env.close()


def write_json(path, value):
    """Write complete JSON atomically so an interrupted array task is never counted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def make_manifest(config):
    return {
        "campaign_version": CAMPAIGN_VERSION,
        "baseline_config": asdict(config),
        "n_episodes": 200,
        "tasks": [task_spec(i) for i in range(200)],
        "information_assumptions": {
            "independent": "Own state/catalog and declared target ephemerides only; no peer catalog, assignment or resource access.",
            "centralized_full_state": "One joint greedy controller reads every live sensor's navigation, attitude, resources, active task, physical products, request epochs, and durable time-tagged catalog at every event decision boundary; no communication action or information latency.",
        },
        "centralized_bound_interpretation": "Maximum instantaneous mission information and joint current assignment, but not a mathematical performance upper bound because the controller has no future-trajectory optimizer.",
        "cooldown_contract": "The existing capture-anchored revisit cooldown is preserved. Ground-confirmed coverage and stale ground delivery are separate endpoint metrics.",
        "pairing": "Identical initial spacecraft states and target priorities between information cases for each environment and seed; LEO and mixed target orbits intentionally differ.",
        "source": source_record(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest_parser = sub.add_parser(
        "manifest", help="Prepare exactly 200 configurations; do not run simulations."
    )
    manifest_parser.add_argument("--config", type=Path)
    manifest_parser.add_argument("--output", type=Path, required=True)
    run_parser = sub.add_parser("run", help="Execute exactly one manifest task.")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--task-id", type=int, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        config = (
            BaselineConfig(**json.loads(args.config.read_text()))
            if args.config
            else BaselineConfig()
        )
        write_json(args.output, make_manifest(config))
        print(args.output.resolve())
    else:
        manifest = json.loads(args.manifest.read_text())
        if manifest["campaign_version"] != CAMPAIGN_VERSION or manifest["tasks"] != [
            task_spec(i) for i in range(200)
        ]:
            raise ValueError("Unexpected campaign version/task mapping.")
        current_source = source_record()
        if (
            current_source["source_fingerprint"]
            != manifest["source"]["source_fingerprint"]
        ):
            raise ValueError(
                "Source differs from manifest; regenerate and review the manifest before running."
            )
        specification = task_spec(args.task_id)
        output = args.output_dir / f"episode_{args.task_id:03d}.json"
        if output.exists():
            raise FileExistsError(f"Refusing to replace completed episode {output}.")
        result = run_episode(
            BaselineConfig(**manifest["baseline_config"]), specification
        )
        result.update(source=current_source, manifest_sha256=digest(manifest))
        write_json(output, result)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "task_id",
                        "case",
                        "target_environment",
                        "seed",
                        "sim_time_s",
                        "wall_time_s",
                        "coverage",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
