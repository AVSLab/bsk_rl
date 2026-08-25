"""Plot per-sensor and team diagnostics from a multi-agent rollout JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ACTION_ORDER = ("image", "downlink", "charge", "desat", "broadcast", "other")
ACTION_COLORS = {
    "image": "#2ca25f",
    "downlink": "#3182bd",
    "charge": "#f2c14e",
    "desat": "#9467bd",
    "broadcast": "#f28e2b",
    "other": "#9e9e9e",
}


def _action_category(label: str) -> str:
    lowered = str(label).lower()
    for category in ACTION_ORDER[:-1]:
        if category in lowered:
            return category
    return "other"


def _action_totals(action_times: dict[str, float]) -> dict[str, float]:
    totals = {category: 0.0 for category in ACTION_ORDER}
    for label, duration in action_times.items():
        totals[_action_category(label)] += float(duration)
    return totals


def _all_products(result: dict) -> list[dict]:
    """Return one capture record per product, whether onboard or delivered."""
    products: dict[str, dict] = {}
    for records in result.get("onboard_products", {}).values():
        for record in records:
            products[str(record["record_id"])] = dict(record)
    for record in result.get("team_service_history", []):
        product = {
            key: record.get(key)
            for key in (
                "record_id",
                "source_sensor",
                "target_id",
                "capture_time",
                "delivery_time",
                "quality",
                "storage_owner",
            )
        }
        products[str(product["record_id"])] = product
    return sorted(
        products.values(),
        key=lambda product: (
            float(product["capture_time"]),
            str(product["source_sensor"]),
            int(product["target_id"]),
        ),
    )


def _save_vector_and_preview(fig, output_base: Path) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(".pdf"), output_base.with_suffix(".png")]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def _plot_sensor(result: dict, sensor: str, output_dir: Path) -> list[Path]:
    color = "#20639b"
    reward_history = result.get("reward_history", {}).get(sensor, [])
    resource_history = result.get("resource_history", {}).get(sensor, [])
    action_totals = _action_totals(result.get("action_time_s", {}).get(sensor, {}))

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.8), constrained_layout=True)
    fig.suptitle(f"{sensor}: deterministic shared-controller diagnostics", fontsize=13)

    if reward_history:
        time_min = [float(row["time_s"]) / 60.0 for row in reward_history]
        reward = [float(row["cumulative_reward"]) for row in reward_history]
        axes[0].step(time_min, reward, where="post", color=color, linewidth=2.0)
    axes[0].set_ylabel("Cumulative reward")
    axes[0].set_xlabel("Simulation time [min]")
    axes[0].grid(alpha=0.25)

    if resource_history:
        time_min = np.asarray([float(row["time_s"]) for row in resource_history]) / 60.0
        battery = [float(row["battery_fraction"]) for row in resource_history]
        storage = [float(row["storage_fraction"]) for row in resource_history]
        wheel = [
            max(map(abs, row["wheel_speed_fraction"]), default=0.0)
            for row in resource_history
        ]
        axes[1].plot(time_min, battery, label="Battery", linewidth=1.8)
        axes[1].plot(time_min, storage, label="Storage", linewidth=1.8)
        axes[1].plot(time_min, wheel, label="Max wheel speed", linewidth=1.8)
    axes[1].set_ylim(-0.03, 1.05)
    axes[1].set_ylabel("Normalized resource")
    axes[1].set_xlabel("Simulation time [min]")
    axes[1].legend(loc="best", ncols=3, fontsize=9)
    axes[1].grid(alpha=0.25)

    categories = [key for key in ACTION_ORDER if action_totals[key] > 0.0]
    durations = [action_totals[key] / 60.0 for key in categories]
    axes[2].bar(
        categories,
        durations,
        color=[ACTION_COLORS[key] for key in categories],
        edgecolor="0.25",
        linewidth=0.6,
    )
    axes[2].set_ylabel("Accumulated time [min]")
    axes[2].set_xlabel("Action category")
    axes[2].grid(axis="y", alpha=0.25)

    return _save_vector_and_preview(fig, output_dir / f"{sensor}_diagnostics")


def _step_counts(times: Iterable[float]) -> tuple[list[float], list[int]]:
    ordered = sorted(map(float, times))
    return [0.0, *[value / 60.0 for value in ordered]], list(range(len(ordered) + 1))


def _plot_team(result: dict, sensors: list[str], output_dir: Path) -> list[Path]:
    products = _all_products(result)
    team = result.get("team_summary", {})
    conflict = result.get("intent_conflicts", {})
    colors = {
        sensor: plt.get_cmap("tab10")(index % 10)
        for index, sensor in enumerate(sensors)
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True)
    fig.suptitle(
        f"Multi-agent catalog overview ({len(sensors)} sensing spacecraft)\n"
        f"{int(team.get('unique_acquisition_count', 0))} unique acquisitions; "
        f"{int(team.get('unique_service_count', 0))} unique deliveries; "
        f"ground value={float(team.get('team_value', 0.0)):.2f}; "
        f"same-target conflict={float(conflict.get('time_s', 0.0)) / 60.0:.1f} min",
        fontsize=12.5,
    )

    for sensor in sensors:
        captures = [
            row["capture_time"] for row in products if row["source_sensor"] == sensor
        ]
        x, y = _step_counts(captures)
        axes[0, 0].step(x, y, where="post", label=sensor, color=colors[sensor])
    axes[0, 0].set_xlabel("Simulation time [min]")
    axes[0, 0].set_ylabel("Cumulative captures")
    axes[0, 0].legend(loc="upper left", fontsize=9)
    axes[0, 0].grid(alpha=0.25)

    offsets = np.linspace(-0.18, 0.18, len(sensors))
    for sensor, offset in zip(sensors, offsets):
        sensor_products = [
            product for product in products if product["source_sensor"] == sensor
        ]
        axes[0, 1].scatter(
            [float(row["capture_time"]) / 60.0 for row in sensor_products],
            [int(row["target_id"]) + offset for row in sensor_products],
            s=34,
            color=colors[sensor],
            edgecolor="white",
            linewidth=0.5,
            label=sensor,
            zorder=3,
        )
        delivered = [
            row for row in sensor_products if row.get("delivery_time") is not None
        ]
        axes[0, 1].scatter(
            [float(row["capture_time"]) / 60.0 for row in delivered],
            [int(row["target_id"]) + offset for row in delivered],
            marker="x",
            s=44,
            color="black",
            linewidth=1.0,
            zorder=4,
        )
    axes[0, 1].set_xlabel("Capture time [min]")
    axes[0, 1].set_ylabel("RSO target ID")
    axes[0, 1].set_title(
        "Catalog capture raster; × indicates later delivery", fontsize=10
    )
    axes[0, 1].grid(alpha=0.25)

    bottoms = np.zeros(len(sensors))
    for category in ACTION_ORDER:
        values = np.asarray(
            [
                _action_totals(result.get("action_time_s", {}).get(sensor, {}))[
                    category
                ]
                / 60.0
                for sensor in sensors
            ]
        )
        if not np.any(values > 0.0):
            continue
        axes[1, 0].bar(
            sensors,
            values,
            bottom=bottoms,
            label=category.capitalize(),
            color=ACTION_COLORS[category],
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms += values
    axes[1, 0].set_ylabel("Accumulated action time [min]")
    axes[1, 0].set_xlabel("Sensing spacecraft")
    axes[1, 0].legend(loc="upper center", ncols=3, fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.25)

    metric_keys = ("captures", "deliveries", "duplicate_attempts")
    metric_labels = ("Captures", "Deliveries", "Duplicate attempts")
    x = np.arange(len(sensors), dtype=float)
    width = 0.23
    per_sensor = result.get("per_sensor_metrics", {})
    for index, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        axes[1, 1].bar(
            x + (index - 1) * width,
            [float(per_sensor.get(sensor, {}).get(key, 0.0)) for sensor in sensors],
            width=width,
            label=label,
        )
    axes[1, 1].set_xticks(x, sensors)
    axes[1, 1].set_ylabel("Event count")
    axes[1, 1].set_xlabel("Sensing spacecraft")
    axes[1, 1].legend(loc="upper center", ncols=3, fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.25)

    return _save_vector_and_preview(fig, output_dir / "multiagent_overview")


def plot_evaluation(result: dict, output_dir: str | Path) -> list[Path]:
    """Write per-sensor plots and, for multi-sensor runs, one team overview."""
    output_dir = Path(output_dir)
    sensors = list(result.get("pettingzoo_agents", []))
    paths = []
    for sensor in sensors:
        paths.extend(_plot_sensor(result, sensor, output_dir))
    if len(sensors) > 1:
        paths.extend(_plot_team(result, sensors, output_dir))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.input.read_text())
    for path in plot_evaluation(result, args.output_dir):
        print(path.resolve())


if __name__ == "__main__":
    main()
