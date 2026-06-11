"""Image-count helpers shared by policy evaluation scripts.

The older evaluation scripts counted only ``rewarder.imaged_illuminated``.
That list is still correct for older runs and for images that have already been
verified on downlink, but newer downlink-verification runs can also have useful
illuminated images still sitting onboard as pending records. These helpers add
those pending records without double-counting records mirrored across data stores.
"""

from __future__ import annotations

from collections import OrderedDict
import math
from typing import Any


def _safe_len(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def _iter_wrapped_objects(obj: Any):
    """Yield likely env/rewarder wrappers without depending on one wrapper stack."""
    seen = set()
    stack = [obj]
    while stack:
        current = stack.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr in ("unwrapped", "env", "par_env"):
            try:
                child = getattr(current, attr)
            except Exception:
                continue
            if child is not current:
                stack.append(child)


def _find_rewarder(obj: Any):
    for candidate in _iter_wrapped_objects(obj):
        rewarder = getattr(candidate, "rewarder", None)
        if rewarder is not None:
            return rewarder
    if hasattr(obj, "imaged_illuminated"):
        return obj
    return None


def _iter_satellites(obj: Any, rewarder: Any):
    """Yield satellites from either an env-like object or the rewarder scenario."""
    seen = set()
    containers = []
    if rewarder is not None:
        containers.append(getattr(rewarder, "scenario", None))
    containers.extend(_iter_wrapped_objects(obj))

    for container in containers:
        satellites = getattr(container, "satellites", None)
        if satellites is None:
            continue
        for satellite in satellites:
            if id(satellite) not in seen:
                seen.add(id(satellite))
                yield satellite


def _record_key(target_id: Any, record: dict[str, Any]) -> str:
    """Stable key so the same pending record mirrored in stores counts once."""
    if record.get("record_id") is not None:
        return str(record["record_id"])
    return "|".join(
        str(record.get(key))
        for key in (
            "source_satellite",
            "target_id",
            "target_name",
            "capture_time",
            "storage_index",
        )
    ) or str(target_id)


def _pending_records(obj: Any, rewarder: Any) -> list[dict[str, Any]]:
    """Collect unique pending records from rewarder/global and satellite stores."""
    data_sources = []
    if rewarder is not None:
        data_sources.append(getattr(rewarder, "data", None))
    for satellite in _iter_satellites(obj, rewarder):
        data_store = getattr(satellite, "data_store", None)
        data_sources.append(getattr(data_store, "data", None))

    records_by_key: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for data_source in data_sources:
        pending_by_id = getattr(data_source, "pending_image_records_by_id", None)
        if not pending_by_id:
            continue
        for target_id, records in pending_by_id.items():
            for record in records:
                record = dict(record)
                record.setdefault("target_id", target_id)
                records_by_key[_record_key(target_id, record)] = record
    return list(records_by_key.values())


def _bool_like(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
        return None
    return bool(value)


def _quality_threshold(record: dict[str, Any], obj: Any, rewarder: Any) -> float:
    for value in (
        record.get("quality_threshold"),
        getattr(rewarder, "image_quality_threshold", None),
    ):
        if value is not None:
            return float(value)

    for satellite in _iter_satellites(obj, rewarder):
        dynamics = getattr(satellite, "dynamics", None)
        for attr in ("eclipse_threshold_for_reward", "eclipse_threshold_for_imaging"):
            value = getattr(dynamics, attr, None)
            if value is not None:
                return float(value)
    return 0.5


def _record_quality_value(record: dict[str, Any]) -> float | None:
    for key in ("mean_hold_shadow_factor", "capture_shadow_factor", "quality_value"):
        value = record.get(key)
        if value is not None:
            return float(value)
    return None


def _is_pending_record_illuminated(
    record: dict[str, Any], obj: Any, rewarder: Any
) -> bool:
    quality_passed = _bool_like(record.get("quality_passed"))
    if quality_passed is not None:
        return quality_passed

    quality_value = _record_quality_value(record)
    if quality_value is None:
        # If a very old pending record does not contain illumination metadata, do
        # not guess. Older versions without pending records still fall back to the
        # confirmed imaged_illuminated count below.
        return False
    return quality_value >= _quality_threshold(record, obj, rewarder)


def illuminated_image_metrics(obj: Any) -> dict[str, int]:
    """Return confirmed + onboard-pending illuminated image counts.

    ``total_illuminated_images`` is the metric the plots/summaries should use.
    It equals the old ``len(rewarder.imaged_illuminated)`` on older code paths
    and adds uniquely pending onboard records only when the new pending image
    lifecycle exists.
    """
    rewarder = _find_rewarder(obj)
    confirmed = _safe_len(getattr(rewarder, "imaged_illuminated", []))
    pending_records = _pending_records(obj, rewarder)
    pending_illuminated = sum(
        1
        for record in pending_records
        if _is_pending_record_illuminated(record, obj, rewarder)
    )
    return {
        "confirmed_illuminated_images": int(confirmed),
        "pending_illuminated_images_onboard": int(pending_illuminated),
        "pending_images_onboard": int(len(pending_records)),
        "total_illuminated_images": int(confirmed + pending_illuminated),
    }


def illuminated_image_count(obj: Any) -> int:
    """Backwards-compatible scalar count for existing plots."""
    return illuminated_image_metrics(obj)["total_illuminated_images"]


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return float((ordered[midpoint - 1] + ordered[midpoint]) / 2.0)


def _duration_from_record(record: dict[str, Any]) -> float | None:
    explicit_duration = _finite_float(record.get("duration_s"))
    if explicit_duration is not None:
        return explicit_duration

    start = _finite_float(record.get("start_time"))
    end = _finite_float(record.get("end_time"))
    if start is None or end is None:
        return None
    duration = end - start
    return duration if math.isfinite(duration) and duration >= 0.0 else None


def imaging_attempt_series(
    action_spec: Any, command_times: Any | None = None
) -> dict[str, Any]:
    """Return per-command timing arrays from ImageRSO attempt records.

    The accepted AMOS-2026 acquisition event is the hold-gated image-success
    action termination. A lower-level instrument/storage trigger can occur
    earlier, but that raw first-capture time should not be used as the headline
    acquisition duration because the policy cannot retask until the action ends.
    """

    records = list(getattr(action_spec, "imaging_attempt_records", []))
    command_times_list = None
    if command_times is not None:
        try:
            command_times_list = [float(value) for value in command_times]
        except TypeError:
            command_times_list = None

    n_rows = len(command_times_list) if command_times_list is not None else len(records)

    def nan_list() -> list[float]:
        return [float("nan")] * n_rows

    accepted_acq_times = nan_list()
    accepted_acq_dt = nan_list()
    first_capture_times = nan_list()
    first_capture_dt = nan_list()
    action_end_times = nan_list()
    action_durations = nan_list()
    slew_times = nan_list()
    hold_valid_times = nan_list()
    target_ids = [None] * n_rows
    reasons = [None] * n_rows
    success_flags = [False] * n_rows

    used_indices: set[int] = set()

    def row_index(record_index: int, record: dict[str, Any]) -> int | None:
        if command_times_list is None:
            return record_index if record_index < n_rows else None
        if len(records) == n_rows:
            return record_index

        start_time = _finite_float(record.get("start_time"))
        if start_time is None:
            return None
        best_index = None
        best_abs_error = float("inf")
        for index, command_time in enumerate(command_times_list):
            if index in used_indices:
                continue
            abs_error = abs(command_time - start_time)
            if abs_error < best_abs_error:
                best_abs_error = abs_error
                best_index = index
        # Times are normally exact. Keep a small tolerance for recorder/event
        # scheduling differences without accidentally matching the next command.
        if best_index is not None and best_abs_error <= 1.0:
            return best_index
        return None

    for record_index, record in enumerate(records):
        index = row_index(record_index, record)
        if index is None or not 0 <= index < n_rows:
            continue
        used_indices.add(index)

        start_time = _finite_float(record.get("start_time"))
        if start_time is None and command_times_list is not None:
            start_time = command_times_list[index]
        end_time = _finite_float(record.get("end_time"))
        first_capture_time = _finite_float(record.get("first_capture_time"))
        duration = _duration_from_record(record)
        success = bool(record.get("success"))

        target_ids[index] = record.get("target_id")
        reasons[index] = record.get("reason")
        success_flags[index] = success

        if end_time is not None:
            action_end_times[index] = end_time
            if start_time is not None:
                action_durations[index] = end_time - start_time
        elif duration is not None and start_time is not None:
            action_end_times[index] = start_time + duration
            action_durations[index] = duration

        if first_capture_time is not None:
            first_capture_times[index] = first_capture_time
            if start_time is not None:
                first_capture_dt[index] = first_capture_time - start_time

        slew_time = _finite_float(record.get("slew_time_s"))
        if slew_time is not None:
            slew_times[index] = slew_time

        hold_valid_time = _finite_float(record.get("hold_valid_time_s"))
        if hold_valid_time is not None:
            hold_valid_times[index] = hold_valid_time

        if success and action_end_times[index] == action_end_times[index]:
            accepted_acq_times[index] = action_end_times[index]
            if start_time is not None:
                accepted_acq_dt[index] = action_end_times[index] - start_time

    return {
        "accepted_acq_times": accepted_acq_times,
        "accepted_acq_dt": accepted_acq_dt,
        "accepted_acq_success": [int(flag) for flag in success_flags],
        "first_capture_times": first_capture_times,
        "first_capture_dt": first_capture_dt,
        "action_end_times": action_end_times,
        "action_durations_sec": action_durations,
        "slew_times_sec": slew_times,
        "hold_valid_times_sec": hold_valid_times,
        "target_ids": target_ids,
        "reasons": reasons,
    }


def imaging_attempt_metrics(action_spec: Any) -> dict[str, Any]:
    """Summarize ImageRSO's per-attempt timing records.

    This uses the action-side records because they know the actual variable
    imaging action time and the slew time, rather than the old nominal 300 s
    imaging window.
    """
    records = list(getattr(action_spec, "imaging_attempt_records", []))

    durations: list[float] = []
    successful_durations: list[float] = []
    unsuccessful_durations: list[float] = []
    slew_times: list[float] = []
    successful_slew_times: list[float] = []
    unsuccessful_slew_times: list[float] = []
    success_flags: list[bool] = []

    for record in records:
        success = bool(record.get("success"))
        success_flags.append(success)

        duration = _duration_from_record(record)
        if duration is not None:
            durations.append(duration)
            if success:
                successful_durations.append(duration)
            else:
                unsuccessful_durations.append(duration)

        slew_time = _finite_float(record.get("slew_time_s"))
        if slew_time is not None:
            slew_times.append(slew_time)
            if success:
                successful_slew_times.append(slew_time)
            else:
                unsuccessful_slew_times.append(slew_time)

    accepted_series = imaging_attempt_series(action_spec)
    accepted_acq_dt = [
        value
        for value in accepted_series["accepted_acq_dt"]
        if _finite_float(value) is not None
    ]

    return {
        "num_imaging_attempts": int(len(records)),
        "imaging_attempt_success_rate": (
            float(sum(success_flags) / len(success_flags))
            if success_flags
            else float("nan")
        ),
        "total_imaging_action_time_sec": float(sum(durations)),
        "mean_imaging_action_duration_sec": _mean(durations),
        "median_imaging_action_duration_sec": _median(durations),
        "mean_accepted_acquisition_time_sec": _mean(accepted_acq_dt),
        "median_accepted_acquisition_time_sec": _median(accepted_acq_dt),
        "mean_successful_imaging_action_duration_sec": _mean(successful_durations),
        "median_successful_imaging_action_duration_sec": _median(successful_durations),
        "mean_unsuccessful_imaging_action_duration_sec": _mean(unsuccessful_durations),
        "median_unsuccessful_imaging_action_duration_sec": _median(
            unsuccessful_durations
        ),
        "mean_imaging_slew_time_sec": _mean(slew_times),
        "median_imaging_slew_time_sec": _median(slew_times),
        "mean_successful_imaging_slew_time_sec": _mean(successful_slew_times),
        "median_successful_imaging_slew_time_sec": _median(successful_slew_times),
        "mean_unsuccessful_imaging_slew_time_sec": _mean(unsuccessful_slew_times),
        "median_unsuccessful_imaging_slew_time_sec": _median(
            unsuccessful_slew_times
        ),
        "imaging_action_durations_sec": durations,
        "successful_imaging_action_durations_sec": successful_durations,
        "unsuccessful_imaging_action_durations_sec": unsuccessful_durations,
        "imaging_slew_times_sec": slew_times,
        "successful_imaging_slew_times_sec": successful_slew_times,
        "unsuccessful_imaging_slew_times_sec": unsuccessful_slew_times,
    }
