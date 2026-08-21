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

import numpy as np


def _merge_time_spans(
    spans: Any,
    *,
    gap_tolerance_sec: float = 1e-6,
) -> list[tuple[float, float]]:
    """Return sorted, merged finite time spans."""
    cleaned = []
    for span in spans:
        try:
            start, stop = float(span[0]), float(span[1])
        except (IndexError, TypeError, ValueError):
            continue
        if not (np.isfinite(start) and np.isfinite(stop)) or stop < start:
            continue
        cleaned.append((start, stop))
    if not cleaned:
        return []

    cleaned.sort()
    merged = [cleaned[0]]
    for start, stop in cleaned[1:]:
        previous_start, previous_stop = merged[-1]
        if start <= previous_stop + gap_tolerance_sec:
            merged[-1] = (previous_start, max(previous_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def ground_station_window_dict(
    satellite: Any,
    *,
    time_limit_sec: float | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """Extract authoritative ground-station access windows from the satellite.

    ``AccessSatellite`` calculates these windows from continuous elevation roots
    during environment reset. Reading the stored opportunities avoids assigning a
    station identity to the rotating ``ground_station_0``/``ground_station_1``
    observation slots and avoids any decision-epoch sampling offset.
    """
    windows: OrderedDict[str, list[tuple[float, float]]] = OrderedDict()
    station_number_by_object: dict[int, int] = {}
    for opportunity in getattr(satellite, "opportunities", []):
        if opportunity.get("type") != "ground_station":
            continue
        station = opportunity.get("object")
        object_id = id(station)
        if object_id not in station_number_by_object:
            station_number_by_object[object_id] = len(station_number_by_object)
        fallback = f"ground_station_{station_number_by_object[object_id]}"
        station_name = str(
            getattr(station, "ModelTag", None)
            or getattr(station, "name", None)
            or fallback
        )
        try:
            start, stop = map(float, opportunity["window"])
        except (KeyError, TypeError, ValueError):
            continue
        if time_limit_sec is not None:
            start = max(0.0, start)
            stop = min(float(time_limit_sec), stop)
        if stop >= start:
            windows.setdefault(station_name, []).append((start, stop))

    return {
        station_name: _merge_time_spans(spans)
        for station_name, spans in windows.items()
    }


def ground_station_window_rows(
    windows: dict[str, list[tuple[float, float]]],
) -> list[dict[str, float | str]]:
    """Flatten a station-to-window mapping for CSV or JSON serialization."""
    return [
        {
            "ground_station": station_name,
            "window_open_sec": float(start),
            "window_close_sec": float(stop),
            "window_duration_sec": float(stop - start),
        }
        for station_name, spans in windows.items()
        for start, stop in spans
    ]


def interval_overlap_seconds(
    start: float,
    stop: float,
    spans: Any,
) -> float:
    """Return the union overlap between one interval and a set of spans."""
    overlaps = []
    for span_start, span_stop in _merge_time_spans(spans):
        overlap_start = max(float(start), span_start)
        overlap_stop = min(float(stop), span_stop)
        if overlap_stop > overlap_start:
            overlaps.append((overlap_start, overlap_stop))
    return float(sum(stop_i - start_i for start_i, stop_i in _merge_time_spans(overlaps)))


def annotate_downlink_window_alignment(
    step_log: Any,
    windows: dict[str, list[tuple[float, float]]],
) -> list[dict[str, Any]]:
    """Annotate each Downlink decision with exact access-window overlap."""
    all_spans = [span for station_spans in windows.values() for span in station_spans]
    rows = []
    for row in step_log:
        if str(row.get("action_category")) != "Downlink":
            continue
        annotated = dict(row)
        start = float(row.get("t_cmd", np.nan))
        stop = float(row.get("t_after", start))
        annotated["ground_station_overlap_sec"] = interval_overlap_seconds(
            start, stop, all_spans
        )
        annotated["starts_in_ground_station_window"] = any(
            span_start <= start <= span_stop for span_start, span_stop in all_spans
        )
        annotated["ends_in_ground_station_window"] = any(
            span_start <= stop <= span_stop for span_start, span_stop in all_spans
        )
        annotated["storage_reduction_fraction"] = max(
            0.0,
            float(row.get("storage_frac_cmd", np.nan))
            - float(row.get("storage_frac_after", np.nan)),
        )
        annotated["useful_deliveries_during_action"] = max(
            0,
            int(row.get("useful_downlinks_after", 0))
            - int(row.get("useful_downlinks_cmd", 0)),
        )
        rows.append(annotated)
    return rows


def cumulative_count_axis_limit(
    *series: Any,
    minimum: float = 300.0,
    increment: float = 100.0,
) -> float:
    """Return a readable upper limit for cumulative-count plot axes.

    The axis retains the paper's 300-count minimum.  Once any plotted cumulative
    series exceeds that value, the limit advances to the next ``increment`` boundary.
    Non-finite and missing values are ignored.
    """
    if minimum <= 0.0:
        raise ValueError("minimum must be positive")
    if increment <= 0.0:
        raise ValueError("increment must be positive")

    finite_values: list[float] = []
    for values in series:
        try:
            array = np.asarray(values, dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        finite_values.extend(array[np.isfinite(array)].tolist())

    observed_max = max(finite_values, default=0.0)
    if observed_max <= minimum:
        return float(minimum)
    return float(increment * math.ceil(observed_max / increment))


def decision_target_state_metrics(satellite: Any) -> dict[str, float | int]:
    """Measure target availability at the current policy decision epoch.

    ``geometric_visible_eligible`` reproduces the elevation test used to construct
    the Polaris candidate list. ``imageable_eligible`` is the stricter operational
    intersection of datastore eligibility, Basilisk line of sight, and target
    illumination.  The latter is the useful diagnostic for deciding whether an
    apparent idle action occurred while a valid image opportunity existed.
    """
    data_obj = satellite.data_store.data
    known_targets = list(getattr(data_obj, "known", []))
    sim_time = float(satellite.simulator.sim_time)
    scanner_position = np.asarray(satellite.dynamics.r_BN_N, dtype=float)
    scanner_radius = float(np.linalg.norm(scanner_position))
    zenith = (
        scanner_position / scanner_radius
        if scanner_radius > 0.0
        else np.zeros(3, dtype=float)
    )
    illumination_threshold = float(
        getattr(satellite.dynamics, "eclipse_threshold_for_imaging", 0.5)
    )
    imaged_ids = {
        int(target.id) for target in getattr(data_obj, "imaged", [])
    }

    counts = {
        "catalog_target_count": len(known_targets),
        "eligible_target_count": 0,
        "pending_verification_target_count": 0,
        "cooldown_target_count": 0,
        "never_imaged_eligible_count": 0,
        "previously_imaged_eligible_count": 0,
        "geometric_visible_eligible_count": 0,
        "los_eligible_count": 0,
        "illuminated_eligible_count": 0,
        "imageable_eligible_count": 0,
    }

    for target in known_targets:
        lifecycle_state = (
            data_obj.target_lifecycle_state(target, sim_time)
            if hasattr(data_obj, "target_lifecycle_state")
            else None
        )
        if lifecycle_state == "pending_verification":
            counts["pending_verification_target_count"] += 1
        elif lifecycle_state == "cooldown":
            counts["cooldown_target_count"] += 1
        eligible = bool(
            data_obj.is_target_eligible(target, sim_time)
            if hasattr(data_obj, "is_target_eligible")
            else int(target.id) not in imaged_ids
        )
        if not eligible:
            continue

        target_id = int(target.id)
        counts["eligible_target_count"] += 1
        if target_id in imaged_ids:
            counts["previously_imaged_eligible_count"] += 1
        else:
            counts["never_imaged_eligible_count"] += 1

        target_position = np.asarray(
            target.target_spacecraft.dynamics.r_BN_N, dtype=float
        )
        relative_position = target_position - scanner_position
        relative_norm = float(np.linalg.norm(relative_position))
        if scanner_radius > 0.0 and relative_norm > 0.0:
            elevation_deg = float(
                np.degrees(
                    np.arcsin(
                        np.clip(
                            np.dot(relative_position / relative_norm, zenith),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            if -21.0 <= elevation_deg <= 90.0:
                counts["geometric_visible_eligible_count"] += 1

        try:
            los = bool(
                satellite.dynamics.targetLocation.accessOutMsgs[target_id]
                .read()
                .hasAccess
            )
        except Exception:
            los = False
        try:
            eclipse_index = target.target_spacecraft.dynamics.eclipse_index
            shadow_factor = float(
                satellite.dynamics.world.eclipseObject.eclipseOutMsgs[eclipse_index]
                .read()
                .shadowFactor
            )
            illuminated = shadow_factor >= illumination_threshold
        except Exception:
            illuminated = False

        counts["los_eligible_count"] += int(los)
        counts["illuminated_eligible_count"] += int(illuminated)
        counts["imageable_eligible_count"] += int(los and illuminated)

    return counts


def decision_state_summary(step_log: Any) -> dict[str, dict[str, Any]]:
    """Aggregate target/resource conditions by executed action category."""
    metric_names = (
        "eligible_target_count",
        "pending_verification_target_count",
        "cooldown_target_count",
        "never_imaged_eligible_count",
        "previously_imaged_eligible_count",
        "geometric_visible_eligible_count",
        "los_eligible_count",
        "illuminated_eligible_count",
        "imageable_eligible_count",
        "wheel_speed_max_fraction_cmd",
        "battery_frac_cmd",
        "storage_frac_cmd",
    )
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in step_log:
        category = str(row.get("action_category", "Unknown"))
        grouped.setdefault(category, []).append(row)

    summary: dict[str, dict[str, Any]] = {}
    for category, rows in grouped.items():
        category_summary: dict[str, Any] = {"decision_count": len(rows)}
        for metric_name in metric_names:
            values = [
                number
                for row in rows
                if (number := _finite_float(row.get(metric_name))) is not None
            ]
            if not values:
                continue
            category_summary[metric_name] = {
                "min": float(min(values)),
                "median": _median(values),
                "mean": _mean(values),
                "max": float(max(values)),
                "zero_fraction": float(sum(value == 0.0 for value in values) / len(values)),
            }
        summary[category] = category_summary
    return summary


def desat_availability_summary(step_log: Any) -> dict[str, Any]:
    """Summarize target availability and wheel state at Desat decisions."""
    desat_rows = [
        row for row in step_log if str(row.get("action_category")) == "Desat"
    ]
    if not desat_rows:
        return {"desat_decision_count": 0}

    imageable = [int(row.get("imageable_eligible_count", 0)) for row in desat_rows]
    wheel = [float(row.get("wheel_speed_max_fraction_cmd", np.nan)) for row in desat_rows]
    shadows = [float(row.get("sat_shadow_cmd", np.nan)) for row in desat_rows]
    return {
        "desat_decision_count": len(desat_rows),
        "desat_with_zero_imageable": int(sum(value == 0 for value in imageable)),
        "desat_with_at_most_one_imageable": int(sum(value <= 1 for value in imageable)),
        "desat_with_at_most_three_imageable": int(sum(value <= 3 for value in imageable)),
        "desat_in_observer_umbra": int(sum(value < 0.5 for value in shadows)),
        "imageable_at_desat_min": int(min(imageable)),
        "imageable_at_desat_median": float(np.median(imageable)),
        "imageable_at_desat_max": int(max(imageable)),
        "wheel_speed_fraction_at_desat_median": float(np.nanmedian(wheel)),
        "wheel_speed_fraction_at_desat_max": float(np.nanmax(wheel)),
    }


def plot_target_availability_desat_diagnostic(
    step_log: Any,
    *,
    cooldown_orbits: float,
    seed: int,
    target_count: int,
    special_title: str | None = None,
):
    """Build a decision-level target-availability and Desat diagnostic figure."""
    import matplotlib.pyplot as plt

    rows = list(step_log)
    if not rows:
        raise ValueError("step_log is empty")
    t_cmd = np.asarray([float(row["t_cmd"]) for row in rows], dtype=float)
    t_after = np.asarray([float(row.get("t_after", row["t_cmd"])) for row in rows])
    eligible = np.asarray([float(row.get("eligible_target_count", np.nan)) for row in rows])
    imageable = np.asarray([float(row.get("imageable_eligible_count", np.nan)) for row in rows])
    never_imaged = np.asarray([float(row.get("never_imaged_eligible_count", np.nan)) for row in rows])
    pending_or_cooldown = float(target_count) - eligible
    wheel_pct = 100.0 * np.asarray(
        [float(row.get("wheel_speed_max_fraction_cmd", np.nan)) for row in rows]
    )
    in_umbra = np.asarray(
        [float(row.get("sat_shadow_cmd", 1.0)) < 0.5 for row in rows], dtype=bool
    )
    desat_mask = np.asarray(
        [str(row.get("action_category")) == "Desat" for row in rows], dtype=bool
    )

    fig, (ax_availability, ax_wheel) = plt.subplots(
        2,
        1,
        figsize=(12.0, 7.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08},
    )
    y_top = max(float(target_count), float(np.nanmax(eligible)))
    ax_availability.fill_between(
        t_cmd,
        0.0,
        y_top,
        where=in_umbra,
        step="post",
        color="0.82",
        alpha=0.55,
        label="Observer umbra",
        zorder=0,
    )
    ax_availability.step(
        t_cmd, eligible, where="post", color="#3569A8", linewidth=1.6,
        label="Eligible targets",
    )
    ax_availability.step(
        t_cmd, imageable, where="post", color="#168A65", linewidth=1.8,
        label="Imageable now (LOS + illuminated + eligible)",
    )
    ax_availability.step(
        t_cmd, never_imaged, where="post", color="#D18B16", linewidth=1.3,
        linestyle="--", label="Never-imaged and eligible",
    )
    ax_availability.step(
        t_cmd, pending_or_cooldown, where="post", color="0.38", linewidth=1.2,
        linestyle=":", label="Pending verification or cooldown",
    )
    for index in np.flatnonzero(desat_mask):
        ax_availability.axvspan(
            t_cmd[index], t_after[index], color="#C83349", alpha=0.22, zorder=1
        )
    if np.any(desat_mask):
        ax_availability.scatter(
            t_cmd[desat_mask], imageable[desat_mask], color="#C83349", marker="D",
            s=34, edgecolor="white", linewidth=0.5, zorder=4, label="Desat decision",
        )
    ax_availability.set_ylabel("Target count")
    ax_availability.set_ylim(0.0, y_top)
    ax_availability.grid(True, alpha=0.22)
    ax_availability.legend(loc="upper right", ncol=2, fontsize=9)

    ax_wheel.step(
        t_cmd, wheel_pct, where="post", color="#6F4E7C", linewidth=1.4,
        label="Maximum wheel-speed fraction",
    )
    if np.any(desat_mask):
        ax_wheel.scatter(
            t_cmd[desat_mask], wheel_pct[desat_mask], color="#C83349", marker="D",
            s=34, edgecolor="white", linewidth=0.5, zorder=4,
            label="Wheel state when Desat was selected",
        )
    ax_wheel.set_xlabel("Simulation time [s]")
    ax_wheel.set_ylabel("Max. wheel speed [%]")
    ax_wheel.set_xlim(0.0, float(np.nanmax(t_after)))
    ax_wheel.set_ylim(bottom=0.0)
    ax_wheel.grid(True, alpha=0.22)
    ax_wheel.legend(loc="upper right", fontsize=9)

    title = special_title or (
        f"Target availability and Desat decisions — seed {seed}, "
        f"{target_count} targets, {cooldown_orbits:g}-orbit cooldown"
    )
    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


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
