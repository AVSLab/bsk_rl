#!/usr/bin/env python3
"""Run the 200-RSO AMOS 2026 priority-event scenario in Vizard."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = REPO_ROOT / "examples" / "updated_policy_evaluation.py"
DEFAULT_POLICY = (
    REPO_ROOT
    / "artifacts"
    / "amos_2026"
    / "policies"
    / "mixed_a0p1"
    / "checkpoint_000119"
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "amos_2026" / "vizard"
DEFAULT_PLOTS = Path(__file__).resolve().parent / "plots"
DEFAULT_N_TARGETS = 200
DEFAULT_INTEREST_FRACTION = 0.10
GROUND_CONFIRMATION_COOLDOWN_ORBITS = 0.0


def interest_object_count(n_targets: int, fraction: float) -> int:
    """Convert a catalog fraction to an exact, non-overlapping tier count."""
    n_targets = int(n_targets)
    fraction = float(fraction)
    if n_targets <= 0:
        raise ValueError("n_targets must be positive")
    if not 0.0 <= fraction <= 0.5:
        raise ValueError("interest fraction must be in [0, 0.5]")
    raw_count = n_targets * fraction
    count = round(raw_count)
    if not math.isclose(raw_count, count, abs_tol=1e-9):
        raise ValueError(
            f"{fraction:g} of {n_targets} targets is not an integer target count"
        )
    return int(count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save a smooth Vizard playback for the selected mixed-trained alpha=0.1 "
            "AMOS 2026 policy. The default scenario has 200 targets; at mid-episode "
            "10% become HIOs and a disjoint 10% become SHIOs. Re-imaging is enabled "
            "immediately after ground verification."
        )
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=Path(os.environ.get("AMOS2026_POLICY_PATH", DEFAULT_POLICY)),
        help=(
            "RLlib checkpoint directory. Defaults to AMOS2026_POLICY_PATH or the "
            "canonical local checkpoint location under artifacts/amos_2026."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vizard-rate-hz",
        type=float,
        default=1.0,
        help=(
            "Recorded Vizard samples per simulation second (default: 1 Hz)."
        ),
    )
    parser.add_argument(
        "--n-targets",
        type=int,
        choices=[100, 200],
        default=DEFAULT_N_TARGETS,
        help="Evaluation catalog size. Both configurations use a 45,000 s episode.",
    )
    parser.add_argument(
        "--interest-fraction",
        type=float,
        default=DEFAULT_INTEREST_FRACTION,
        help=(
            "Independent catalog fraction promoted to HIO and SHIO at mid-episode "
            "(default: 0.10 for each tier)."
        ),
    )
    parser.add_argument(
        "--reimage-cooldown-orbits",
        type=float,
        default=GROUND_CONFIRMATION_COOLDOWN_ORBITS,
        help=(
            "Additional cooldown after useful ground verification (default: 0; the "
            "image remains ineligible while it is onboard awaiting verification)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS,
        help="Directory for the three episode-summary PDF plots.",
    )
    parser.add_argument(
        "--rw-display",
        choices=["all", "off"],
        default="off",
        help="Show the full native reaction-wheel panel or omit it (default).",
    )
    parser.add_argument(
        "--no-text-hud", action="store_true", help="Hide the live text summary."
    )
    parser.add_argument(
        "--no-metric-bars",
        action="store_true",
        help=(
            "Hide every live metric bar and skip their 200-target calculations while "
            "retaining action rings, promotion markers, and pointing overlays."
        ),
    )
    parser.add_argument(
        "--no-image-bars",
        action="store_true",
        help="Hide the targets-imaged >=1, >=2, and >=3 bars.",
    )
    parser.add_argument(
        "--target-status-outlines",
        action="store_true",
        help=(
            "Enable the optional cyan/red/green lifecycle outlines. They are omitted "
            "by default to reduce Vizard scene and playback load."
        ),
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Disable all AMOS-specific Vizard overlays and live metrics.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact evaluator command without executing it.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the evaluator's per-action simulation-time progress messages.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace the canonical Vizard file for this seed, catalog size, and "
            "sampling rate after the new run completes successfully."
        ),
    )
    return parser.parse_args()


def vizard_tag(args: argparse.Namespace) -> str:
    hz_label = f"{args.vizard_rate_hz:g}Hz".replace(".", "p")
    interest_pct = f"{100.0 * args.interest_fraction:g}pct".replace(".", "p")
    cooldown_label = (
        "groundConfirm"
        if math.isclose(args.reimage_cooldown_orbits, 0.0, abs_tol=1e-12)
        else f"{args.reimage_cooldown_orbits:g}orbitCooldown".replace(".", "p")
    )
    return (
        f"AMOS2026_mixed_a0p1_{args.n_targets}targets_"
        f"{interest_pct}HIO_{interest_pct}SHIO_{cooldown_label}_"
        f"seed{args.seed}_{hz_label}"
    )


def build_command(args: argparse.Namespace) -> list[str]:
    if args.vizard_rate_hz <= 0.0:
        raise ValueError("--vizard-rate-hz must be positive")
    if args.reimage_cooldown_orbits < 0.0:
        raise ValueError("--reimage-cooldown-orbits must be non-negative")
    interest_count = interest_object_count(args.n_targets, args.interest_fraction)
    python = REPO_ROOT / ".venv" / "bin" / "python"
    command = [
        str(python if python.exists() else Path(sys.executable)),
        "-u",
        str(EVALUATOR),
        "--policy_path",
        str(args.policy_path.expanduser().resolve()),
        "--policy_name",
        (
            "amos2026_MIXED_GAT_fullActions_10d90i_"
            f"evaluation{args.n_targets}targets"
        ),
        "--policy_layout",
        "gat_full",
        "--obs_v",
        "9",
        "--policy_mode",
        "latest",
        "--reward_alpha",
        "0.1",
        "--target_env",
        "mixed",
        "--mix_weights",
        '{"LEO":0.5,"MEO":0.3,"GEO":0.2}',
        "--exact_mix_counts",
        "--n_targets",
        str(args.n_targets),
        "--n_targets_ahead",
        "10",
        "--priority_sum",
        str(args.n_targets),
        "--total_time_sec",
        "45000",
        "--reimage_cooldown_orbits",
        str(args.reimage_cooldown_orbits),
        "--dynamic_priority_event",
        "on",
        "--dynamic_priority_event_fraction",
        "0.5",
        "--hio_count",
        str(interest_count),
        "--hio_priority_max_multiplier",
        "5",
        "--shio_count",
        str(interest_count),
        "--shio_priority_max_multiplier",
        "10",
        "--dynamic_priority_event_seed",
        str(args.seed),
        "--seed",
        str(args.seed),
        "--save_vizard",
        "--vizard_rate",
        str(1.0 / args.vizard_rate_hz),
        "--vizard_dir",
        str(args.output_dir.expanduser().resolve()),
        "--vizard_tag",
        vizard_tag(args),
        "--plot_dir",
        str(args.plots_dir.expanduser().resolve()),
        "--no_show_plots",
        "--no_save_data",
    ]
    if not args.no_hud:
        command.append("--amos_vizard_hud")
        command.extend(["--amos_vizard_rw_display", args.rw_display])
        if args.no_text_hud:
            command.append("--no_amos_vizard_text")
        if args.no_metric_bars:
            command.append("--no_amos_vizard_metric_bars")
        if args.no_image_bars:
            command.append("--no_amos_vizard_image_bars")
        if args.target_status_outlines:
            command.append("--amos_vizard_target_status_outlines")
    if args.quiet:
        command.append("--quiet")
    return command


def main() -> int:
    args = parse_args()
    command = build_command(args)
    interest_count = interest_object_count(args.n_targets, args.interest_fraction)
    print("Vizard episode runner:", Path(__file__).resolve())
    print("Evaluator:", EVALUATOR)
    print("Policy:", args.policy_path.expanduser().resolve())
    print("Catalog:", f"{args.n_targets} targets over 45,000 s")
    print(
        "Midpoint promotion:",
        f"{interest_count} HIO + {interest_count} SHIO "
        f"({100.0 * args.interest_fraction:g}% each, disjoint)",
    )
    print(
        "Re-imaging gate:",
        "ground confirmation only"
        if math.isclose(args.reimage_cooldown_orbits, 0.0, abs_tol=1e-12)
        else f"ground confirmation + {args.reimage_cooldown_orbits:g} orbit(s)",
    )
    print("Vizard sampling:", f"{args.vizard_rate_hz:g} Hz")
    print(
        "Progress logging:",
        "suppressed" if args.quiet else "enabled once per policy action",
    )
    print("Output:", args.output_dir.expanduser().resolve())
    print("Plots:", args.plots_dir.expanduser().resolve())
    print("Command:", " ".join(command))
    if args.dry_run:
        return 0
    if not EVALUATOR.is_file():
        raise SystemExit(f"Evaluator not found: {EVALUATOR}")
    if not args.policy_path.expanduser().is_dir():
        raise SystemExit(
            "The paper's checkpoint is not available locally. Copy checkpoint_000119 "
            f"to {DEFAULT_POLICY} or set AMOS2026_POLICY_PATH."
        )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.expanduser().mkdir(parents=True, exist_ok=True)
    existing = {path.resolve() for path in output_dir.rglob("*") if path.is_file()}
    return_code = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
    if return_code != 0 or not args.overwrite:
        return return_code

    created = sorted(
        (
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path.resolve() not in existing
        ),
        key=lambda path: path.stat().st_mtime,
    )
    created_bins = [path for path in created if path.suffix == ".bin"]
    if not created_bins:
        print("WARNING: no new Vizard .bin file was found to canonicalize")
        return return_code

    newest = created_bins[-1]
    canonical = newest.with_name(f"{vizard_tag(args)}_UnityViz.bin")
    if canonical.exists() and canonical.resolve() != newest.resolve():
        canonical.unlink()
    if newest.resolve() != canonical.resolve():
        newest.replace(canonical)
    for stale in canonical.parent.glob(f"{vizard_tag(args)}_*_UnityViz.bin"):
        if stale.resolve() != canonical.resolve():
            stale.unlink()
    print("Canonical Vizard file:", canonical)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
