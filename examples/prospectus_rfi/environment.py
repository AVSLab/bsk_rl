"""Matched AMOS-2025 environment for the Research Focus I study.

The only intended physical changes from the archived AMOS setup are the fixed
100 s imaging action, a 45,000 s episode for every catalog size, a uniformly
sampled configured target-catalog range, and a 20--60 percent initial scanner
charge. Both learned policies consume the exact same padded observation and
validity mask. The physical scenario contains ``config.catalog_max`` target
spacecraft; only the first ``scenario.n_targets`` are exposed in an episode.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from Basilisk.utilities import macros, orbitalMotion

from bsk_rl import act, data, obs, scene, sats
from bsk_rl.act.discrete_actions import DiscreteActionBuilder
from bsk_rl.obs.observations import _eligible_targets_now
from bsk_rl.sim import dyn, fsw, world

from .config import EnvironmentConfig

GLOBAL_FEATURE_COUNT = 11
TARGET_PHYSICAL_FEATURE_COUNT = 7
TARGET_FEATURE_COUNT = TARGET_PHYSICAL_FEATURE_COUNT + 1
TARGET_MASK_INDEX = TARGET_FEATURE_COUNT - 1

# Exact policy I/O recorded in the August 13, 2025 checkpoint artifact.  This
# deliberately differs from the masked layout used to train the new policies.
LEGACY_AMOS2025_CANDIDATE_COUNT = 10
LEGACY_AMOS2025_TARGET_FEATURE_COUNT = 7
LEGACY_AMOS2025_GROUND_STATION_COUNT = 5
LEGACY_AMOS2025_OBSERVATION_SIZE = (
    5
    + LEGACY_AMOS2025_CANDIDATE_COUNT * LEGACY_AMOS2025_TARGET_FEATURE_COUNT
    + 2
    + 2 * LEGACY_AMOS2025_GROUND_STATION_COUNT
)
LEGACY_AMOS2025_ACTION_COUNT = LEGACY_AMOS2025_CANDIDATE_COUNT + 3
LEGACY_AMOS2025_OBSERVATION_CONTRACT = "amos2025_obs_v2_checkpoint_exact"
STUDY_MASKED_OBSERVATION_CONTRACT = "amos2025_obs_v2_masked"
AMOS2025_ATTENTION_CONTROL_OBSERVATION_CONTRACT = (
    "amos2025_obs_v2_checkpoint_attention_masked"
)
AMOS2025_ATTENTION_CONTROL_GLOBAL_FEATURE_COUNT = 17
AMOS2025_ATTENTION_CONTROL_TARGET_START = 5
BRECKENRIDGE2026_OBSERVATION_CONTRACT = "breckenridge2026_obs_v7"
BRECKENRIDGE2026_OBSERVATION_SIZE = 81


class StudyDiscreteActionBuilder(DiscreteActionBuilder):
    """Record action allocation without changing action execution semantics."""

    def reset_post_sim_init(self) -> None:
        super().reset_post_sim_init()
        # The archived ImageRSO timeout hook checks this satellite attribute even
        # in fixed-duration mode, where no success event initializes it.  Keep the
        # compatibility state local to this study instead of modifying historical
        # AMOS source files.
        self.satellite._active_image_rso_action = None
        self.satellite.study_action_counts = {
            "image": 0,
            "charge": 0,
            "downlink": 0,
            "desaturate": 0,
        }

    def set_action(self, action: int) -> None:
        if np.issubdtype(type(action), np.integer):
            candidate_count = self.action_spec[0].n_actions
            if int(action) < candidate_count:
                label = "image"
            else:
                label = ("charge", "downlink", "desaturate")[
                    int(action) - candidate_count
                ]
            self.satellite.study_action_counts[label] += 1
        else:
            # Object/string overrides are only supported by ImageRSO here.
            self.satellite.study_action_counts["image"] += 1
        super().set_action(action)


def zero_padded_target_rows(
    rows: np.ndarray, valid_count: int, mask_index: int = TARGET_MASK_INDEX
) -> np.ndarray:
    """Zero invalid rows and write a binary mask without mutating the input."""

    result = np.asarray(rows, dtype=np.float32).copy()
    if result.ndim != 2:
        raise ValueError("target rows must have shape [slots, features]")
    if not 0 <= valid_count <= result.shape[0]:
        raise ValueError("valid_count must lie within the target slot count")
    if not 0 <= mask_index < result.shape[1]:
        raise ValueError("mask_index is outside the feature dimension")
    result[valid_count:] = 0.0
    result[:valid_count, mask_index] = 1.0
    return result


def _zero_like(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.zeros_like(value, dtype=float)
    if isinstance(value, list):
        return [0.0 for _ in value]
    return 0.0


class MaskedPolarisTargetProperties(obs.PolarisScTargetProperties):
    """AMOS per-target features with true zero padding and a validity mask.

    ``PolarisScTargetProperties`` historically repeats the last eligible target
    when fewer candidates than action slots remain.  Repetition leaks padded
    values into a policy.  This subclass retains the historical candidate
    ordering for valid rows, replaces repetitions with zeros, and appends
    ``valid_target`` to every target row.
    """

    def get_obs(self) -> dict[str, dict[str, Any]]:
        result = super().get_obs()
        known = self.satellite.data_store.data.known
        valid_count = min(
            len(_eligible_targets_now(self.satellite, known)), self.n_ahead_observe
        )
        for slot in range(self.n_ahead_observe):
            row = result[f"{self.name}_{slot}"]
            if slot >= valid_count:
                for key, value in row.items():
                    row[key] = _zero_like(value)
            row["valid_target"] = float(slot < valid_count)
        return result


class RandomCatalogSatellites(scene.RandomSatellites):
    """Sample catalog size at every reset, or hold it fixed for evaluation."""

    def __init__(
        self,
        *args: Any,
        catalog_min: int,
        catalog_max: int,
        fixed_catalog_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.catalog_min = int(catalog_min)
        self.catalog_max = int(catalog_max)
        self.fixed_catalog_size = (
            None if fixed_catalog_size is None else int(fixed_catalog_size)
        )
        self.sampled_catalog_size: int | None = None
        initial_size = self.fixed_catalog_size or self.catalog_max
        super().__init__(*args, n_targets=initial_size, **kwargs)
        self._validate_catalog_settings()

    def _validate_catalog_settings(self) -> None:
        if self.catalog_min < 1 or self.catalog_min > self.catalog_max:
            raise ValueError("invalid catalog sampling range")
        if self.fixed_catalog_size is not None and not (
            self.catalog_min <= self.fixed_catalog_size <= self.catalog_max
        ):
            raise ValueError("fixed catalog size lies outside the configured range")

    def reset_overwrite_previous(self) -> None:
        if self.fixed_catalog_size is None:
            # np.random is seeded by the environment before reset hooks execute.
            self.n_targets = int(
                np.random.randint(self.catalog_min, self.catalog_max + 1)
            )
        else:
            self.n_targets = self.fixed_catalog_size
        self.sampled_catalog_size = self.n_targets
        super().reset_overwrite_previous()


def amos2025_target_orbit() -> orbitalMotion.ClassicElements:
    """Exact LEO target distribution from the late-summer AMOS 2025 trainer."""

    lower_a_m = 6871.0e3
    upper_a_m = 8371.0e3
    elements = orbitalMotion.ClassicElements()
    elements.a = np.random.uniform(lower_a_m, upper_a_m)
    elements.e = np.random.uniform(0.0, 0.02)
    while elements.a * (1.0 - elements.e) < 6771.0e3:
        elements.e = np.random.uniform(0.0, 0.02)
    elements.i = np.random.uniform(0.0, 180.0) * macros.D2R
    elements.Omega = np.random.uniform(0.0, 360.0) * macros.D2R
    elements.omega = np.random.uniform(0.0, 360.0) * macros.D2R
    elements.f = np.random.uniform(0.0, 360.0) * macros.D2R
    return elements


def make_satellite_types(
    config: EnvironmentConfig,
    *,
    observation_contract: str | None = None,
):
    """Construct satellite classes with an explicitly selected I/O contract.

    The legacy contract exists only to evaluate the frozen AMOS 2025 policy.  It
    reproduces the checkpoint's 87-value input exactly: five spacecraft values,
    ten seven-value target rows, two eclipse values, and five pairs of ground-
    station opportunity values.  It intentionally has no validity mask because
    the historical policy was not trained with one.
    """

    config.validate()
    observation_contract = observation_contract or config.observation_layout
    if observation_contract not in {
        STUDY_MASKED_OBSERVATION_CONTRACT,
        LEGACY_AMOS2025_OBSERVATION_CONTRACT,
        AMOS2025_ATTENTION_CONTROL_OBSERVATION_CONTRACT,
        BRECKENRIDGE2026_OBSERVATION_CONTRACT,
    }:
        raise ValueError(f"unsupported observation contract: {observation_contract}")
    if (
        observation_contract
        in {
            LEGACY_AMOS2025_OBSERVATION_CONTRACT,
            BRECKENRIDGE2026_OBSERVATION_CONTRACT,
        }
        and config.candidate_count != LEGACY_AMOS2025_CANDIDATE_COUNT
    ):
        raise ValueError("the frozen monolithic policies require exactly 10 candidates")

    class StudyImageRSO(act.ImageRSO):
        builder_type = StudyDiscreteActionBuilder

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # ImageRSO's historical diagnostic list is populated before its
            # full-catalog heuristic override.  Keep a study-local record of the
            # target that was actually tasked so policy/heuristic comparisons use
            # the same command-time definition.
            self.actual_target_ids: list[int] = []
            self.actual_target_illumination_status: list[float] = []

        def set_action(self, action: int, prev_action_key: str | None = None) -> str:
            action_key = super().set_action(action, prev_action_key)
            actual_id = int(self.satellite.dynamics.last_imaging_target_id)
            actual_target = next(
                target
                for target in self.satellite.data_store.data.known
                if int(target.id) == actual_id
            )
            self.actual_target_ids.append(actual_id)
            self.actual_target_illumination_status.append(
                self._target_shadow_factor(actual_target)
            )
            return action_key

    class StudyCharge(act.Charge):
        builder_type = StudyDiscreteActionBuilder

    class StudyDownlink(act.Downlink):
        builder_type = StudyDiscreteActionBuilder

    class StudyDesat(act.Desat):
        builder_type = StudyDiscreteActionBuilder

    class StudyScanningSatellite(sats.AccessSatellite):
        if observation_contract == LEGACY_AMOS2025_OBSERVATION_CONTRACT:
            # Exact order and normalization recovered from the saved run params.
            observation_spec = [
                obs.SatProperties(
                    dict(prop="storage_level_fraction"),
                    dict(prop="battery_charge_fraction"),
                    dict(prop="wheel_speeds_fraction"),
                ),
                obs.PolarisScTargetProperties(
                    dict(prop="target_elevation_angle", norm=1.0),
                    dict(prop="rel_pos_vector_r_BR_H", norm=1596.0e3),
                    dict(prop="angle_to_target", norm=1.0),
                    dict(prop="target_distance", norm=1596.0e3),
                    dict(prop="target_shadowFactor", norm=1.0),
                    n_ahead_observe=LEGACY_AMOS2025_CANDIDATE_COUNT,
                ),
                obs.Eclipse(norm=1.0),
                obs.OpportunityProperties(
                    dict(prop="opportunity_open", norm=5700.0),
                    dict(prop="opportunity_close", norm=5700.0),
                    type="ground_station",
                    n_ahead_observe=LEGACY_AMOS2025_GROUND_STATION_COUNT,
                ),
            ]
        elif observation_contract == BRECKENRIDGE2026_OBSERVATION_CONTRACT:
            # Exact observation-version-7 contract used by the October 14,
            # 2025 Breckenridge imaging-versus-downlink alpha sweep.
            observation_spec = [
                obs.SatProperties(
                    dict(prop="storage_level_fraction"),
                    dict(prop="battery_charge_fraction"),
                    dict(prop="wheel_speeds_fraction"),
                ),
                obs.PolarisScTargetProperties(
                    dict(prop="target_elevation_angle", norm=90.0),
                    dict(prop="rel_pos_vector_r_BR_H", norm=15960.0e3),
                    dict(prop="angle_to_target", norm=90.0),
                    dict(prop="target_distance", norm=15960.0e3),
                    dict(prop="target_shadowFactor", norm=1.0),
                    n_ahead_observe=LEGACY_AMOS2025_CANDIDATE_COUNT,
                ),
                obs.Eclipse(norm=5700.0),
                obs.OpportunityProperties(
                    dict(prop="opportunity_open", norm=5700.0),
                    dict(prop="opportunity_close", norm=5700.0),
                    type="ground_station",
                    n_ahead_observe=2,
                ),
            ]
        elif observation_contract == STUDY_MASKED_OBSERVATION_CONTRACT:
            observation_spec = [
                # Five global values: storage, battery, and three wheel speeds.
                obs.SatProperties(
                    dict(prop="storage_level_fraction"),
                    dict(prop="battery_charge_fraction"),
                    dict(prop="wheel_speeds_fraction"),
                ),
                # Two eclipse values.
                obs.Eclipse(norm=5700.0),
                # Four ground-station access values (open/close for two passes).
                obs.OpportunityProperties(
                    dict(prop="opportunity_open", norm=5700.0),
                    dict(prop="opportunity_close", norm=5700.0),
                    type="ground_station",
                    n_ahead_observe=2,
                ),
                # Seven physical values plus one binary validity mask per slot.
                MaskedPolarisTargetProperties(
                    dict(prop="target_elevation_angle", norm=90.0),
                    dict(prop="rel_pos_vector_r_BR_H", norm=15960.0e3),
                    dict(prop="angle_to_target", norm=90.0),
                    dict(prop="target_distance", norm=15960.0e3),
                    dict(prop="target_shadowFactor", norm=1.0),
                    n_ahead_observe=config.candidate_count,
                ),
            ]
        else:
            # Same physical fields, order, and normalizations as the archived
            # August 13 AMOS 2025 checkpoint, with one validity bit appended to
            # each target row so the target-set attention policy cannot learn
            # from repeated/padded candidate values.
            observation_spec = [
                obs.SatProperties(
                    dict(prop="storage_level_fraction"),
                    dict(prop="battery_charge_fraction"),
                    dict(prop="wheel_speeds_fraction"),
                ),
                MaskedPolarisTargetProperties(
                    dict(prop="target_elevation_angle", norm=1.0),
                    dict(prop="rel_pos_vector_r_BR_H", norm=1596.0e3),
                    dict(prop="angle_to_target", norm=1.0),
                    dict(prop="target_distance", norm=1596.0e3),
                    dict(prop="target_shadowFactor", norm=1.0),
                    n_ahead_observe=LEGACY_AMOS2025_CANDIDATE_COUNT,
                ),
                obs.Eclipse(norm=1.0),
                obs.OpportunityProperties(
                    dict(prop="opportunity_open", norm=5700.0),
                    dict(prop="opportunity_close", norm=5700.0),
                    type="ground_station",
                    n_ahead_observe=LEGACY_AMOS2025_GROUND_STATION_COUNT,
                ),
            ]
        action_spec = [
            StudyImageRSO(
                n_ahead_image=config.candidate_count,
                duration=config.imaging_duration_s,
                variable_duration_imaging=False,
                # These fields only gate the early-success event.  Fixed-duration
                # mode deliberately retains the complete configured tasking interval.
                min_pointing_hold_s=0.0,
                require_illumination_during_hold=False,
            ),
            StudyCharge(duration=config.charge_duration_s),
            StudyDownlink(
                duration=config.downlink_duration_s,
                variable_duration_downlink=False,
            ),
            StudyDesat(duration=config.desaturation_duration_s),
        ]
        dyn_type = dyn.ImagingSCDynModel
        fsw_type = fsw.ImagingSCFSWModel

    class StudyTargetSatellite(sats.Satellite):
        observation_spec = [obs.Time()]
        action_spec = [act.Drift(duration=config.episode_duration_s)]
        dyn_type = dyn.BasicTargetDynamicsModel
        fsw_type = fsw.BasicTargetFSWModel

    return StudyScanningSatellite, StudyTargetSatellite


def make_environment_args(
    config: EnvironmentConfig,
    *,
    fixed_catalog_size: int | None = None,
    episode_data_callback: Any | None = None,
    satellite_data_callback: Any | None = None,
    historical_heuristic: bool = False,
    historical_heuristic_mode: str | None = None,
    observation_contract: str | None = None,
) -> dict[str, Any]:
    """Create one directly usable BSK-RL environment configuration."""

    config.validate()
    if fixed_catalog_size is not None and not (
        config.catalog_min <= fixed_catalog_size <= config.catalog_max
    ):
        raise ValueError("fixed_catalog_size lies outside the study range")
    if historical_heuristic_mode not in {None, "angle", "distance"}:
        raise ValueError(
            "historical_heuristic_mode must be None, 'angle', or 'distance'"
        )

    scanner_type, target_type = make_satellite_types(
        config, observation_contract=observation_contract
    )
    scanner_args = {
        "imageAttErrorRequirement": config.image_attitude_error_requirement,
        "dataStorageCapacity": config.storage_capacity_bits,
        "storageInit": 0.0,
        "instrumentBaudRate": config.instrument_baud_rate,
        "transmitterBaudRate": config.transmitter_baud_rate,
        "batteryStorageCapacity": config.battery_capacity_ws,
        "storedCharge_Init": lambda: np.random.uniform(
            config.initial_battery_fraction_min,
            config.initial_battery_fraction_max,
        )
        * config.battery_capacity_ws,
        "basePowerDraw": config.base_power_draw_w,
        "instrumentPowerDraw": config.instrument_power_draw_w,
        "transmitterPowerDraw": config.transmitter_power_draw_w,
        "thrusterPowerDraw": config.thruster_power_draw_w,
        "panelArea": config.panel_area_m2,
        "disturbance_vector": lambda: np.zeros(3),
        "maxWheelSpeed": config.max_wheel_speed_rpm,
        "wheelSpeeds": lambda: np.random.uniform(-500.0, 500.0, 3),
        "desatAttitude": "sun",
        "downlink_bonus": config.downlink_bonus,
        "imaging_bonus": config.imaging_bonus,
        "eclipse_threshold_for_imaging": config.eclipse_threshold,
        "eclipse_threshold_for_reward": config.eclipse_threshold,
        "use_heuristic": bool(
            historical_heuristic or historical_heuristic_mode is not None
        ),
        "heuristic_mode": historical_heuristic_mode or "angle",
    }
    target_args = {
        "oe": amos2025_target_orbit,
        "batteryStorageCapacity": 1.0,
        "storedCharge_Init": 0.0,
        "basePowerDraw": -10_000.0,
    }

    scanner = scanner_type(name="SS1", sat_args=scanner_args)
    physical_target_count = fixed_catalog_size or config.catalog_max
    targets = [
        target_type(name=f"target_{index}", sat_args=target_args)
        for index in range(physical_target_count)
    ]
    scenario = RandomCatalogSatellites(
        "SS1",
        catalog_min=config.catalog_min,
        catalog_max=config.catalog_max,
        fixed_catalog_size=fixed_catalog_size,
        priority_mode="constant",
        priority_sum=None,
        rescale_priorities_to_sum=False,
        priority_constant=config.target_priority,
    )
    rewarder = data.RSOTargetImageReward(
        # An infinite cooldown reproduces the AMOS 2025 one-image-per-target rule.
        reimage_cooldown_orbits=np.inf,
        verify_image_quality_on_downlink=False,
        hide_pending_targets=True,
    )

    env_args: dict[str, Any] = {
        "satellites": [scanner, *targets],
        "scenario": scenario,
        "rewarder": rewarder,
        "world_type": world.GroundStationWorldModel,
        "time_limit": config.episode_duration_s,
        "failure_penalty": config.failure_penalty,
        "terminate_on_time_limit": False,
        "generate_obs_retasking_only": False,
        "log_level": "ERROR",
    }
    if episode_data_callback is not None:
        env_args["episode_data_callback"] = episode_data_callback
    if satellite_data_callback is not None:
        env_args["satellite_data_callback"] = satellite_data_callback
    return env_args


def environment_contract(config: EnvironmentConfig) -> dict[str, Any]:
    """Machine-readable observation/action/reward contract for run metadata."""

    if config.observation_layout == AMOS2025_ATTENTION_CONTROL_OBSERVATION_CONTRACT:
        global_features = AMOS2025_ATTENTION_CONTROL_GLOBAL_FEATURE_COUNT
        target_start = AMOS2025_ATTENTION_CONTROL_TARGET_START
        field_order = [
            "spacecraft[5]",
            "masked_target_rows[10,8]",
            "eclipse[2]",
            "ground_station_windows[5,2]",
        ]
    else:
        global_features = GLOBAL_FEATURE_COUNT
        target_start = GLOBAL_FEATURE_COUNT
        field_order = ["global[11]", "masked_target_rows[K,8]"]
    return {
        "environment": asdict(config),
        "observation": {
            "global_features": global_features,
            "target_start": target_start,
            "target_slots": config.candidate_count,
            "target_physical_features_per_slot": TARGET_PHYSICAL_FEATURE_COUNT,
            "mask_features_per_slot": 1,
            "flattened_size": global_features
            + config.candidate_count * TARGET_FEATURE_COUNT,
            "field_order": field_order,
            "padding": "zero",
            "mask_name": "valid_target",
            "mask_index_within_target_row": TARGET_MASK_INDEX,
        },
        "action": {
            "target_actions": config.candidate_count,
            "non_target_actions": ["charge", "downlink", "desaturate"],
            "total_actions": config.action_count,
            "target_actions_first": True,
        },
        "reward_equation": (
            "R = (1-alpha)*illuminated_unique_observation_value "
            "+ alpha*useful_downlink_value + operational_penalties"
        ),
        "alpha_interpretation": "alpha=0 is observation-only; it is not AlphaZero",
    }


def legacy_amos2025_policy_contract() -> dict[str, Any]:
    """Return the exact archived module signature and observation semantics."""

    return {
        "name": LEGACY_AMOS2025_OBSERVATION_CONTRACT,
        "flattened_observation_size": LEGACY_AMOS2025_OBSERVATION_SIZE,
        "action_count": LEGACY_AMOS2025_ACTION_COUNT,
        "candidate_count": LEGACY_AMOS2025_CANDIDATE_COUNT,
        "field_order": [
            "spacecraft[5]",
            "target_rows[10,7]",
            "eclipse[2]",
            "ground_station_windows[5,2]",
        ],
        "target_field_order": [
            "target_elevation_angle/1",
            "rel_pos_vector_r_BR_H[3]/1596000",
            "angle_to_target/1",
            "target_distance/1596000",
            "target_shadowFactor/1",
        ],
        "padding": "historical repeated-target behavior; no validity mask",
    }


__all__ = [
    "AMOS2025_ATTENTION_CONTROL_GLOBAL_FEATURE_COUNT",
    "AMOS2025_ATTENTION_CONTROL_OBSERVATION_CONTRACT",
    "AMOS2025_ATTENTION_CONTROL_TARGET_START",
    "BRECKENRIDGE2026_OBSERVATION_CONTRACT",
    "BRECKENRIDGE2026_OBSERVATION_SIZE",
    "GLOBAL_FEATURE_COUNT",
    "LEGACY_AMOS2025_ACTION_COUNT",
    "LEGACY_AMOS2025_CANDIDATE_COUNT",
    "LEGACY_AMOS2025_OBSERVATION_CONTRACT",
    "LEGACY_AMOS2025_OBSERVATION_SIZE",
    "TARGET_FEATURE_COUNT",
    "TARGET_MASK_INDEX",
    "MaskedPolarisTargetProperties",
    "RandomCatalogSatellites",
    "STUDY_MASKED_OBSERVATION_CONTRACT",
    "amos2025_target_orbit",
    "environment_contract",
    "legacy_amos2025_policy_contract",
    "make_environment_args",
    "make_satellite_types",
    "zero_padded_target_rows",
]
