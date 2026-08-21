#!/usr/bin/env python3
"""Rebuild AMOS evaluation timing plots from a saved run directory.

This script is intentionally independent of the live simulator so older runs can
be replotted with authoritative ground-station windows and decision-level Desat
diagnostics without rerunning a 45,000-second episode.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from evaluation_image_metrics import (  # noqa: E402
    annotate_downlink_window_alignment,
    cumulative_count_axis_limit,
    desat_availability_summary,
    plot_target_availability_desat_diagnostic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--ground-station-windows", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--cooldown-orbits", type=float, default=2.0)
    parser.add_argument("--special-title", default=None)
    return parser.parse_args()


def load_windows(path: Path) -> dict[str, list[tuple[float, float]]]:
    frame = pd.read_csv(path)
    windows: dict[str, list[tuple[float, float]]] = {}
    for row in frame.itertuples(index=False):
        windows.setdefault(str(row.ground_station), []).append(
            (float(row.window_open_sec), float(row.window_close_sec))
        )
    return windows


def cumulative_at(sample_times: np.ndarray, event_times: np.ndarray) -> np.ndarray:
    event_times = np.sort(event_times[np.isfinite(event_times)])
    return np.searchsorted(event_times, sample_times, side="right")


def action_spans(frame: pd.DataFrame, category: str):
    selected = frame[frame["action_category"] == category]
    return zip(selected["t_cmd"].to_numpy(), selected["t_after"].to_numpy())


def main() -> None:
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    steps = pd.read_csv(args.run_dir / "steps.csv")
    images = pd.read_csv(args.run_dir / "images.csv")
    deliveries = pd.read_csv(args.run_dir / "verified_deliveries.csv")
    windows = load_windows(args.ground_station_windows)

    sim_times = np.load(args.run_dir / "sim_times.npy")
    battery = np.load(args.run_dir / "battery_levels.npy")
    storage = np.load(args.run_dir / "storage_levels.npy")
    useful_deliveries = deliveries[deliveries["useful_delivery"].astype(bool)]
    # Verified useful records carry the authoritative target-illumination result.
    # Successful acquisitions absent from the verification table are still onboard;
    # their command-time target illumination is the best saved older-run proxy.
    successful_images = images[images["acq_success"].astype(bool)].copy()
    verified_capture_times = pd.to_numeric(
        deliveries["capture_time"], errors="coerce"
    ).to_numpy()
    successful_capture_times = pd.to_numeric(
        successful_images["first_capture_time"], errors="coerce"
    ).to_numpy()
    verified_match = np.isin(
        np.round(successful_capture_times, 6),
        np.round(verified_capture_times, 6),
    )
    pending_useful = successful_images[
        (~verified_match)
        & (pd.to_numeric(successful_images["target_shadow_cmd"], errors="coerce") >= 0.5)
    ]
    useful_acquisition_times = np.concatenate(
        [
            pd.to_numeric(useful_deliveries["capture_time"], errors="coerce").to_numpy(),
            pd.to_numeric(pending_useful["first_capture_time"], errors="coerce").to_numpy(),
        ]
    )
    cumulative_images = cumulative_at(
        sim_times,
        useful_acquisition_times,
    )
    cumulative_deliveries = cumulative_at(
        sim_times,
        pd.to_numeric(useful_deliveries["downlink_time"], errors="coerce").to_numpy(),
    )

    fig, resource_axis = plt.subplots(figsize=(12.0, 6.0))
    first_window = True
    for station_spans in windows.values():
        for start, stop in station_spans:
            resource_axis.axvspan(
                start,
                stop,
                color="#39A845",
                alpha=0.14,
                label="Ground-station window" if first_window else "",
                zorder=0,
            )
            first_window = False
    # Decision-to-decision shadow state is sufficient for contextual shading here;
    # the access-window audit uses the independent continuous root solutions above.
    umbra_mask = pd.to_numeric(steps["sat_shadow_cmd"], errors="coerce") < 0.5
    for row in steps[umbra_mask].itertuples():
        resource_axis.axvspan(
            float(row.t_cmd),
            float(row.t_after),
            color="0.55",
            alpha=0.18,
            zorder=0,
        )

    resource_axis.plot(sim_times, battery, color="#3569A8", linewidth=1.3, label="Battery fraction")
    resource_axis.plot(sim_times, storage, color="#E07A28", linewidth=1.3, label="Storage fraction")
    for number, (start, stop) in enumerate(action_spans(steps, "Downlink")):
        resource_axis.axvline(
            start,
            color="#C227B9",
            linestyle="--",
            linewidth=0.65,
            alpha=0.48,
            label="Downlink command start" if number == 0 else "",
        )
        resource_axis.axvspan(
            start,
            stop,
            ymin=0.965,
            ymax=0.985,
            color="#C227B9",
            alpha=0.8,
            label="Downlink action duration" if number == 0 else "",
        )
    for number, (start, _stop) in enumerate(action_spans(steps, "Desat")):
        resource_axis.axvline(
            start,
            color="#C83349",
            linestyle="--",
            linewidth=0.7,
            alpha=0.55,
            label="Desat command start" if number == 0 else "",
        )

    count_axis = resource_axis.twinx()
    count_axis.step(
        sim_times,
        cumulative_images,
        where="post",
        color="#168A65",
        linewidth=1.5,
        label="Useful images acquired (cumulative)",
    )
    count_axis.step(
        sim_times,
        cumulative_deliveries,
        where="post",
        color="#B43A3A",
        linewidth=1.5,
        label="Useful images delivered (cumulative)",
    )
    resource_axis.set(xlabel="Simulation time [s]", ylabel="Battery and storage fraction", ylim=(0, 1))
    resource_axis.set_xlim(0, float(np.nanmax(sim_times)))
    count_axis.set_ylabel("Cumulative count")
    count_axis.set_ylim(0, cumulative_count_axis_limit(cumulative_images, cumulative_deliveries))
    resource_axis.grid(True, linestyle="-.", alpha=0.28)
    handles_a, labels_a = resource_axis.get_legend_handles_labels()
    handles_b, labels_b = count_axis.get_legend_handles_labels()
    resource_axis.legend(handles_a + handles_b, labels_a + labels_b, loc="upper left", fontsize=8.7, ncol=2)
    fig.tight_layout()
    resource_path = args.plot_dir / f"{args.name}_resource_delivery_groundstation_timing.pdf"
    fig.savefig(resource_path, bbox_inches="tight")
    plt.close(fig)

    target_count = int(pd.to_numeric(steps["catalog_target_count"], errors="coerce").max())
    fig = plot_target_availability_desat_diagnostic(
        steps.to_dict("records"),
        cooldown_orbits=args.cooldown_orbits,
        seed=int(pd.to_numeric(steps.get("seed", pd.Series([0])), errors="coerce").fillna(0).iloc[0]),
        target_count=target_count,
        special_title=args.special_title,
    )
    diagnostic_path = args.plot_dir / f"{args.name}_target_availability_desat_{args.cooldown_orbits:g}orbit_cooldown.pdf"
    fig.savefig(diagnostic_path, bbox_inches="tight")
    plt.close(fig)

    alignment = pd.DataFrame(
        annotate_downlink_window_alignment(steps.to_dict("records"), windows)
    )
    alignment["useful_delivery_count_at_action_end"] = [
        int(np.count_nonzero(np.isclose(useful_deliveries["downlink_time"], row.t_after)))
        for row in alignment.itertuples()
    ]
    alignment_path = args.run_dir / "downlink_ground_station_window_alignment.csv"
    alignment.to_csv(alignment_path, index=False)

    summary = desat_availability_summary(steps.to_dict("records"))
    useful_action_mask = alignment["useful_delivery_count_at_action_end"] > 0
    print(f"Saved: {resource_path}")
    print(f"Saved: {diagnostic_path}")
    print(f"Saved: {alignment_path}")
    print(
        "Downlink audit:",
        {
            "downlink_actions": int(len(alignment)),
            "actions_overlapping_access": int((alignment["ground_station_overlap_sec"] > 0).sum()),
            "useful_delivery_actions": int(useful_action_mask.sum()),
            "useful_delivery_actions_without_access_overlap": int(
                (useful_action_mask & (alignment["ground_station_overlap_sec"] <= 0)).sum()
            ),
        },
    )
    print("Desat audit:", summary)


if __name__ == "__main__":
    main()
