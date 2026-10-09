"""Matched baseline Vizard recordings and evidence-backed comparison plots.

Recording uses an explicitly selected historical source snapshot. Color changes
are applied to the saved Vizard stream from physical product records afterward;
they cannot change the controller's observations, reservations, or decisions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import sys

AGENT_COLORS = {
    "sensor_0": [245, 166, 35, 255],
    "sensor_1": [49, 187, 112, 255],
    "sensor_2": [185, 117, 242, 255],
}
DUPLICATE_COLOR = [242, 61, 70, 255]
BASE_COLORS = {
    "lower": [158, 202, 225, 255],
    "middle": [66, 146, 198, 255],
    "upper": [8, 81, 156, 255],
}
CASE_COLORS = {"independent": "#3573B9", "centralized_full_state": "#B77E18"}
CASE_LABELS = {
    "independent": "Independent local greedy",
    "centralized_full_state": "Centralized full-state joint greedy",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def pending_holders(episode, time_s):
    """Half-open physical ownership intervals; metadata never implies ownership.

    Qualified capture is displayed at exposure completion, not at the initial
    shutter timestamp. A product is removed only on its complete ground delivery.
    """
    deliveries = {
        p["record_id"]: p["delivery_time"] for p in episode["ground_delivery_records"]
    }
    held = defaultdict(set)
    for p in episode["capture_records"]:
        start = p.get("completion_time") or p["capture_time"]
        end = deliveries.get(p["record_id"], float("inf"))
        if p["quality"] >= 0.5 and start <= time_s < end:
            held[p["target_id"]].add(p["source_sensor"])
    return dict(held)


def target_state(holders):
    if len(holders) >= 2:
        return "duplicate"
    return next(iter(holders)) if holders else "base"


@lru_cache(maxsize=1)
def proto_types():
    """Wire-compatible projection of Basilisk vizMessage.proto.

    Unknown protobuf fields are retained on parsing/serialization. This avoids
    needing protoc and preserves all native Vizard settings, planets, HUDs, etc.
    Field numbers/types are from Basilisk/src/utilities/vizProtobuffer/vizMessage.proto.
    """
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    f = descriptor_pb2.FileDescriptorProto(
        name="baseline_viz_projection.proto", package="baseline_viz", syntax="proto3"
    )

    def msg(name):
        m = f.message_type.add()
        m.name = name
        return m

    def field(m, name, num, typ, repeated=False, type_name=None):
        x = m.field.add(name=name, number=num, type=typ, label=3 if repeated else 1)
        if type_name:
            x.type_name = ".baseline_viz." + type_name

    t = msg("TimeStamp")
    field(t, "frameNumber", 1, 3)
    field(t, "simTimeElapsed", 2, 1)
    s = msg("Spacecraft")
    field(s, "spacecraftName", 1, 9)
    for name, n in [("position", 2), ("velocity", 3), ("rotation", 4)]:
        field(s, name, n, 1, True)
    field(s, "spacecraftSprite", 9, 9)
    line = msg("PointLine")
    field(line, "fromBodyName", 1, 9)
    field(line, "toBodyName", 2, 9)
    field(line, "lineColor", 3, 5, True)
    live = msg("LiveSettings")
    field(live, "targetLines", 1, 11, True, "PointLine")
    v = msg("VizMessage")
    field(v, "currentTime", 1, 11, type_name="TimeStamp")
    field(v, "spacecraft", 3, 11, True, "Spacecraft")
    field(v, "liveSettings", 10, 11, type_name="LiveSettings")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(f)
    descriptor = pool.FindMessageTypeByName("baseline_viz.VizMessage")
    return (
        message_factory.GetMessageClass(descriptor)
        if hasattr(message_factory, "GetMessageClass")
        else message_factory.MessageFactory(pool).GetPrototype(descriptor)
    )


def read_varint(stream):
    value = shift = 0
    while True:
        b = stream.read(1)
        if not b:
            if shift:
                raise ValueError("Truncated Vizard length prefix")
            return None
        value |= (b[0] & 127) << shift
        if not b[0] & 128:
            return value
        shift += 7
        if shift > 35:
            raise ValueError("Invalid Vizard length prefix")


def encode_varint(value):
    out = bytearray()
    while value > 127:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def frames(path):
    Viz = proto_types()
    with path.open("rb") as f:
        while (size := read_varint(f)) is not None:
            data = f.read(size)
            if len(data) != size:
                raise ValueError("Truncated Vizard frame")
            m = Viz()
            m.ParseFromString(data)
            yield m


def color_recording(raw, episode, output):
    """Create immutable sprite proxies and switch their visible positions per frame.

    Vizard initializes sprites once, so changing spacecraftSprite every frame is
    insufficient. Each target gets three owner proxies and one red duplicate
    proxy, all initialized in frame one. Inactive proxies are hidden inside Earth.
    They are visualization objects only, never simulated spacecraft or agents.
    """
    if output.exists():
        raise FileExistsError(output)
    stats = Counter()
    timeline = []
    with output.open("wb") as f:
        for index, m in enumerate(frames(raw)):
            t = m.currentTime.simTimeElapsed / 1e9
            held = pending_holders(episode, t)
            active_names = {}
            frame_counts = Counter()
            for sc in list(m.spacecraft):
                name = sc.spacecraftName
                if name in AGENT_COLORS:
                    sc.spacecraftSprite = "TRIANGLE " + " ".join(
                        map(str, AGENT_COLORS[name])
                    )
                    continue
                if not name.startswith("target_"):
                    continue
                target_id = int(name.split("_")[-1])
                state = target_state(held.get(target_id, set()))
                frame_counts[state] += 1
                original_position = list(sc.position)
                velocity = list(sc.velocity)
                rotation = list(sc.rotation)
                active_names[name] = name if state == "base" else f"{name}__{state}"
                for key, color in [
                    *AGENT_COLORS.items(),
                    ("duplicate", DUPLICATE_COLOR),
                ]:
                    proxy = m.spacecraft.add()
                    proxy.spacecraftName = f"{name}__{key}"
                    proxy.position.extend(
                        original_position if state == key else [0.0, 0.0, 0.0]
                    )
                    proxy.velocity.extend(velocity if state == key else [0.0, 0.0, 0.0])
                    proxy.rotation.extend(rotation)
                    if index == 0:
                        proxy.spacecraftSprite = (
                            "STAR " if key == "duplicate" else "CIRCLE "
                        ) + " ".join(map(str, color))
                if state != "base":
                    sc.position[:] = [0.0, 0.0, 0.0]
                    sc.velocity[:] = [0.0, 0.0, 0.0]
            for line in m.liveSettings.targetLines:
                line.toBodyName = active_names.get(line.toBodyName, line.toBodyName)
                if line.fromBodyName in AGENT_COLORS:
                    line.lineColor[:] = AGENT_COLORS[line.fromBodyName]
            data = m.SerializeToString()
            f.write(encode_varint(len(data)))
            f.write(data)
            stats["frames"] += 1
            stats["duplicate_target_frames"] += frame_counts["duplicate"]
            stats["max_red_targets"] = max(
                stats["max_red_targets"], frame_counts["duplicate"]
            )
            timeline.append(
                {
                    "time_s": t,
                    **{
                        key: frame_counts[key]
                        for key in ["base", *AGENT_COLORS, "duplicate"]
                    },
                }
            )
    stats["first_time_s"] = timeline[0]["time_s"]
    stats["last_time_s"] = timeline[-1]["time_s"]
    write_json(
        output.with_suffix(".colors.json"),
        {
            "palette": AGENT_COLORS,
            "red": DUPLICATE_COLOR,
            "stats": dict(stats),
            "frames": timeline,
        },
    )
    return dict(stats)


def record(args):
    source = args.source_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(source / "src"), str(source)]
    from examples.multiagent_imaging import baseline_monte_carlo as b
    import Basilisk
    from Basilisk.utilities import vizSupport

    assert vizSupport.vizFound, "Basilisk Vizard recorder unavailable"
    manifest = json.loads(args.manifest.read_text())
    mismatches = [
        p
        for p, h in manifest["source"]["source_hashes"].items()
        if not (source / p).is_file()
        or hashlib.sha256((source / p).read_bytes()).hexdigest() != h
    ]
    if mismatches:
        raise ValueError(f"Snapshot differs from historical manifest: {mismatches}")
    original_builder = b.build_environment

    def with_recording(config):
        return original_builder(
            config,
            vizard_dir=str(output / "raw"),
            vizard_settings={
                "vizard_rate": args.rate_s,
                "multiagent_vizard": True,
                "showSpacecraftAsSprites": 1,
                "showSpacecraftLabels": -1,
                "orbitLinesOn": -1,
                "trueTrajectoryLinesOn": -1,
                "showOsculatingGroundTrackLines": -1,
                "showTruePathGroundTrackLines": -1,
                "showLocationCones": -1,
                "showLocationCommLines": -1,
                "showLocationLabels": -1,
                "spacecraftSizeMultiplier": 2.5,
                "linesAndFramesLineWidth": 2.0,
                "useLineRenderersForTargetLinesAndFrames": 1,
            },
        )

    b.build_environment = with_recording
    task = (
        args.seed
        + (100 if args.case == "centralized_full_state" else 0)
        + (50 if args.environment == "mixed" else 0)
    )
    config = b.BaselineConfig(**manifest["baseline_config"])
    print(
        f'Recording {args.case}, {args.environment}, seed {args.seed}, {config.episode_duration_s:g} s, historical source {manifest["source"]["git_head"]}',
        flush=True,
    )
    result = b.run_episode(config, b.task_spec(task))
    result["visualization_provenance"] = {
        "historical_manifest_sha256": digest(manifest),
        "historical_source_fingerprint": manifest["source"]["source_fingerprint"],
        "historical_git_head": manifest["source"]["git_head"],
        "source_snapshot": str(source),
        "python": sys.version,
        "basilisk_path": Basilisk.__file__,
        "basilisk_version": getattr(Basilisk, "__version__", None),
        "platform": platform.platform(),
        "sample_rate_s": args.rate_s,
        "rendering": "Postprocessed qualified physical capture-completion/full-delivery records; ownership colors do not affect control",
        "visualizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "heuristic": CASE_LABELS[args.case],
    }
    write_json(output / "episode.json", result)
    import gc

    gc.collect()
    bins = list((output / "raw").rglob("*.bin"))
    if len(bins) != 1:
        raise ValueError(f"Expected one native recording, found {bins}")
    name = f"{args.case}_{args.environment}_seed{args.seed}_45000s_UnityViz.bin"
    stats = color_recording(bins[0], result, output / name)
    print(
        json.dumps(
            {
                "case": args.case,
                "initial_conditions_sha256": result["initial_conditions_sha256"],
                "time_s": result["sim_time_s"],
                "capture_coverage": result["coverage"]["capture_coverage_fraction"],
                "ground_coverage": result["coverage"][
                    "ground_delivery_coverage_fraction"
                ],
                "duplicates": result["team_summary"]["duplicate_attempt_count"],
                "vizard": str(output / name),
                "color_stats": stats,
            }
        ),
        flush=True,
    )


def plots(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    episodes = [
        json.loads(p.read_text())
        for p in sorted(args.episodes_dir.glob("episode_*.json"))
    ]
    if len(episodes) != 200:
        raise ValueError("Plot campaign must contain all 200 episodes")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#242424",
            "text.color": "#242424",
            "axes.titleweight": "bold",
        }
    )

    def save(fig, name):
        for ext in ["png", "pdf"]:
            fig.savefig(root / f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    def heading(fig, title, subtitle):
        fig.suptitle(title, x=0.07, y=0.99, ha="left", fontsize=19)
        fig.text(
            0.07, 1 - 0.68 / fig.get_figheight(), subtitle, fontsize=11, color="#505050"
        )

    groups = {
        (c, e): [d for d in episodes if d["case"] == c and d["target_environment"] == e]
        for c in CASE_LABELS
        for e in ["leo", "mixed"]
    }
    metrics = [
        (
            "Capture coverage (%)",
            lambda d: 100 * d["coverage"]["capture_coverage_fraction"],
        ),
        (
            "Ground-delivered coverage (%)",
            lambda d: 100 * d["coverage"]["ground_delivery_coverage_fraction"],
        ),
        (
            "Cooldown-qualified acquisitions",
            lambda d: d["team_summary"]["unique_acquisition_count"],
        ),
        (
            "Ground-confirmed unique services",
            lambda d: d["team_summary"]["unique_service_count"],
        ),
        ("Constellation reward", lambda d: d["total_constellation_reward"]),
        (
            "Duplicate image attempts",
            lambda d: d["team_summary"]["duplicate_attempt_count"],
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    heading(
        fig,
        "Independent versus centralized information",
        "Deterministic coverage-first greedy baselines · 3 sensors · 100 targets · 10 candidates · 45,000 s · 50 matched seeds/environment",
    )
    for ax, (title, fn) in zip(axes.flat, metrics):
        for j, (case, label) in enumerate(CASE_LABELS.items()):
            x = np.array([0, 1]) + (j - 0.5) * 0.32
            vals = [
                np.array([fn(d) for d in groups[case, e]]) for e in ["leo", "mixed"]
            ]
            ax.bar(
                x,
                [a.mean() for a in vals],
                width=0.30,
                color=CASE_COLORS[case],
                edgecolor="#333333",
                linewidth=0.6,
                label=label,
                hatch="//" if j == 0 else None,
            )
            for pos, a in zip(x, vals):
                ax.scatter(
                    pos + np.linspace(-0.09, 0.09, len(a)),
                    a,
                    s=7,
                    color="#242424",
                    alpha=0.20,
                    zorder=3,
                )
                ax.text(
                    pos,
                    a.mean() + max(fn(d) for d in episodes) * 0.025,
                    f"{a.mean():.2f}",
                    ha="center",
                    fontsize=9,
                )
        ax.set(title=title, xticks=[0, 1], xticklabels=["LEO", "Mixed LEO/MEO/GEO"])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
        if "coverage" in title.lower():
            ax.set_ylim(0, 108)
    fig.legend(
        *axes.flat[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
    )
    fig.text(
        0.07,
        0.005,
        "Black points are individual seeds. Coverage counts each target once; qualified services include revisits after the capture-anchored cooldown.",
        fontsize=10,
        color="#505050",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.91))
    save(fig, "campaign_comparison")
    # Paired differences show the seed-level contrast, with seed-blocked pooled intervals.
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    heading(
        fig,
        "Matched information effects",
        "Centralized minus independent · effects computed within the same environment/seed · pooled CI resamples 50 seed blocks",
    )
    effect_specs = [
        ("Qualified acquisitions", metrics[2][1]),
        ("Unique ground services", metrics[3][1]),
        ("Duplicate attempts", metrics[5][1]),
    ]
    rng = np.random.default_rng(20260906)
    for ax, (title, fn) in zip(axes, effect_specs):
        arrays = []
        for e in ["leo", "mixed"]:
            indexed = {
                (d["case"], d["seed"]): d
                for d in episodes
                if d["target_environment"] == e
            }
            arrays.append(
                np.array(
                    [
                        fn(indexed["centralized_full_state", s])
                        - fn(indexed["independent", s])
                        for s in range(50)
                    ]
                )
            )
        for x, (label, a) in enumerate(
            zip(["LEO", "Mixed", "Pooled"], [*arrays, np.concatenate(arrays)])
        ):
            block = (arrays[0] + arrays[1]) / 2 if x == 2 else a
            ci = np.quantile(
                rng.choice(block, (10000, len(block)), replace=True).mean(axis=1),
                [0.025, 0.975],
            )
            ax.scatter(
                x + np.linspace(-0.12, 0.12, len(a)),
                a,
                s=17,
                facecolors="none",
                edgecolors="#B77E18",
                alpha=0.50,
            )
            ax.errorbar(
                x,
                a.mean(),
                yerr=[[a.mean() - ci[0]], [ci[1] - a.mean()]],
                fmt="D",
                color="#242424",
                capsize=5,
                zorder=5,
            )
            ax.annotate(
                f"{a.mean():+.2f}",
                (x, a.mean()),
                xytext=(12, 6),
                textcoords="offset points",
                fontsize=10,
            )
        ax.axhline(0, color="#777777", lw=0.8, ls=":")
        ax.set(title=title, xticks=[0, 1, 2], xticklabels=["LEO", "Mixed", "Pooled"])
        ax.grid(axis="y", alpha=0.15)
    fig.tight_layout(rect=(0, 0.02, 1, 0.84))
    save(fig, "paired_information_effects")
    # Campaign seed-zero timeline corresponds to the selected recordings' initial scenario.
    pair = [groups[c, "leo"][0] for c in CASE_LABELS]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    heading(
        fig,
        "Matched LEO seed 0: physical products over time",
        "Same initial scenario · independent local greedy versus centralized full-state joint greedy · original campaign event records",
    )
    for d in pair:
        c = d["case"]
        linestyle = "--" if c == "independent" else "-"
        for ax, key, when in [
            (axes[0], "capture_records", "completion_time"),
            (axes[1], "ground_delivery_records", "delivery_time"),
        ]:
            events = sorted(
                (p[when] or p["capture_time"], p["target_id"])
                for p in d[key]
                if p["quality"] >= 0.5
            )
            seen = set()
            x = [0.0]
            y = [0]
            for t, tid in events:
                if tid not in seen:
                    seen.add(tid)
                    x.append(t / 3600)
                    y.append(len(seen))
            x.append(12.5)
            y.append(y[-1])
            ax.step(
                x,
                y,
                where="post",
                lw=2,
                color=CASE_COLORS[c],
                ls=linestyle,
                label=CASE_LABELS[c],
            )
        times = np.arange(0, 45001, 60)
        reds = [
            sum(len(h) >= 2 for h in pending_holders(d, float(t)).values())
            for t in times
        ]
        axes[2].step(
            times / 3600,
            reds,
            where="post",
            color=CASE_COLORS[c],
            ls=linestyle,
            lw=2,
            label=CASE_LABELS[c],
        )
    for ax, title in zip(
        axes,
        [
            "Distinct captured targets / 100",
            "Distinct ground-delivered targets / 100",
            "Targets with multiple onboard holders",
        ],
    ):
        ax.set(
            title=title, xlabel="Simulated time (hours)", xlim=(0, 12.5), ylim=(0, None)
        )
        ax.grid(alpha=0.15)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.84))
    save(fig, "matched_seed0_timeline")
    # Portable color key, shared by both Vizard streams.
    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    ax.axis("off")
    ax.set_title(
        "Vizard target colors · same key in both recordings",
        loc="left",
        fontsize=18,
        pad=20,
    )
    labels = [
        ("Sensor 0 owns image", AGENT_COLORS["sensor_0"], "o"),
        ("Sensor 1 owns image", AGENT_COLORS["sensor_1"], "o"),
        ("Sensor 2 owns image", AGENT_COLORS["sensor_2"], "o"),
        ("Multiple sensors hold images", DUPLICATE_COLOR, "*"),
    ]
    for i, (label, color, marker) in enumerate(labels):
        x = 0.035 + i * 0.25
        ax.scatter(
            [x],
            [0.70],
            s=230,
            marker=marker,
            color=np.array(color[:3]) / 255,
            transform=ax.transAxes,
        )
        ax.text(
            x + 0.025, 0.70, label, transform=ax.transAxes, va="center", fontsize=11
        )
    for i, (tier, color) in enumerate(BASE_COLORS.items()):
        x = 0.10 + i * 0.30
        ax.scatter(
            [x], [0.40], s=120, color=np.array(color[:3]) / 255, transform=ax.transAxes
        )
        ax.text(
            x + 0.025,
            0.40,
            f"{tier.title()} priority\nNo onboard image",
            transform=ax.transAxes,
            va="center",
            fontsize=10,
        )
    ax.text(
        0.02,
        0.12,
        "Red marks physical duplicate holdings, including valid revisits after cooldown.\nAfter full downlink: remaining holder color, or priority blue if all products have left storage.",
        transform=ax.transAxes,
        fontsize=11,
        color="#505050",
    )
    save(fig, "vizard_color_key")
    write_json(
        root / "plot_provenance.json",
        {
            "source": str(args.episodes_dir.resolve()),
            "episodes": len(episodes),
            "seed": 0,
            "environment": "leo",
            "heuristics": CASE_LABELS,
            "colors": AGENT_COLORS,
            "red_semantics": "two or more distinct sensors holding qualified products at that time",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    print(root, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("record")
    r.add_argument("--source-root", type=Path, required=True)
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--case", choices=CASE_LABELS, required=True)
    r.add_argument("--environment", choices=["leo", "mixed"], default="leo")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--rate-s", type=float, default=10)
    r.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("plots")
    p.add_argument("--episodes-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "record":
        record(args)
    else:
        plots(args)


if __name__ == "__main__":
    main()
