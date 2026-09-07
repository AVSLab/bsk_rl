"""Reproducible measurement and schema helpers for local and Slurm preflight."""

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import threading
import time

import psutil

from bsk_rl.obs.completion_observations import OBSERVATION_VERSION


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def schema(config):
    """Semantic contract, not just matching flat dimensions.

    Scenario seed may differ for evaluation. Changing targets, timing, masking,
    communication assumptions or physical discounting requires a new checkpoint.
    """
    settings = config.to_dict()
    settings.pop("seed")
    return dict(
        version=OBSERVATION_VERSION,
        policy="shared-target-peer-set-attention-v2",
        observation=dict(
            own=26,
            target_features=17,
            candidates=config.n_candidates,
            peer_features=12,
            peers=config.n_peers,
            size=26 + 17 * config.n_candidates + 12 * config.n_peers,
        ),
        actions=dict(
            operational=["charge", "downlink", "desat", "broadcast", "continue"],
            image_start=5,
            transmit_start=5 + config.n_candidates,
            size=5 + config.n_candidates + config.n_peers,
        ),
        settings=settings,
    )


def validate_schema(saved, config):
    expected = schema(config)
    if saved != expected:
        raise ValueError(
            "Checkpoint observation/action/formulation schema mismatch; no implicit migration is allowed."
        )


def basilisk_source_commit(module_path):
    """Identify a source build without mistaking an enclosing bsk_rl repo for it."""
    import os

    directory = os.environ.get("BASILISK_SOURCE_ROOT", module_path)
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=directory,
        text=True,
        capture_output=True,
    )
    if root.returncode or not (Path(root.stdout.strip()) / "src/architecture").is_dir():
        return None
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=directory, text=True
    ).strip()


def provenance(config):
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    versions = {}
    for name in (
        "ray",
        "torch",
        "numpy",
        "gymnasium",
        "pettingzoo",
        "pyarrow",
        "scipy",
        "psutil",
    ):
        versions[name] = importlib.metadata.version(name)
    import Basilisk

    sources = {}
    for directory in (root / "src/bsk_rl", root / "examples/multiagent_imaging"):
        for path in sorted(directory.rglob("*.py")):
            sources[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return dict(
        config=config.to_dict(),
        schema=schema(config),
        git_commit=git("rev-parse", "HEAD"),
        git_branch=git("branch", "--show-current"),
        git_status=git("status", "--short"),
        source_sha256=sources,
        python=platform.python_version(),
        platform=platform.platform(),
        packages=versions,
        basilisk_version=getattr(Basilisk, "__version__", "unknown"),
        basilisk_path=str(Basilisk.__path__[0]),
        basilisk_git_commit=basilisk_source_commit(Basilisk.__path__[0]),
        discount_half_life_s=config.discount_half_life_s,
        gae_trace_half_life_s=config.gae_trace_half_life_s,
    )


@contextmanager
def measure():
    """Sample resident memory of this process and its children (including Ray)."""
    record = {"peak_rss_bytes": 0}
    stop = threading.Event()
    process = psutil.Process()

    def sample():
        while not stop.is_set():
            total = 0
            for child in [process, *process.children(recursive=True)]:
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            record["peak_rss_bytes"] = max(record["peak_rss_bytes"], total)
            stop.wait(0.2)

    thread = threading.Thread(target=sample, daemon=True)
    start = time.perf_counter()
    thread.start()
    try:
        yield record
    finally:
        stop.set()
        thread.join()
        record["wall_time_s"] = time.perf_counter() - start


def profile_episode(config, output):
    """Measure a complete mission episode before starting the PPO preflight."""
    from examples.multiagent_imaging.evaluate import run_rollout

    output = Path(output)
    with measure() as measurement:
        result = run_rollout(config, controller="closest_angle")
    measurement["simulated_seconds"] = result["sim_time_s"]
    measurement["sim_seconds_per_wall_second"] = (
        result["sim_time_s"] / measurement["wall_time_s"]
    )
    result["measurement"] = measurement
    write_json(output / "profile_episode.json", result)
    write_json(output / "profile_provenance.json", provenance(config))
    print(json.dumps(measurement), flush=True)
    return result


def save_reproduction_bundle(output):
    """Preserve edited/untracked source as well as installed dependency versions."""
    import sys
    import tarfile

    output = Path(output)
    root = Path(__file__).resolve().parents[2]
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (output / "pip-freeze.txt").write_text(freeze)
    with tarfile.open(output / "source-snapshot.tar.gz", "w:gz") as archive:
        for directory in (root / "src/bsk_rl", root / "examples/multiagent_imaging"):
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix in {
                    ".py",
                    ".json",
                    ".md",
                    ".slurm",
                    ".txt",
                    ".sh",
                }:
                    archive.add(path, arcname=path.relative_to(root))
        for directory in (
            root / "tests/unittest/multiagent",
            root / "tests/integration/multiagent",
        ):
            for path in sorted(directory.glob("*.py")):
                archive.add(path, arcname=path.relative_to(root))
        archive.add(root / "pyproject.toml", arcname="pyproject.toml")


if __name__ == "__main__":
    import argparse
    from examples.multiagent_imaging.config import MultiAgentImagingConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile_episode(MultiAgentImagingConfig.from_json(args.config), args.output)
