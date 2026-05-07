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
