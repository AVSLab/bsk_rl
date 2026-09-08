"""Read-only runtime gate; safe to run on a login node (no simulation or Ray)."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


BASILISK_COMMIT = "8fcb54b2fb28388efb711786630501944fddec28"
ROOT = Path(__file__).resolve().parents[3]


def _normalized_distribution_name(name):
    """Use the same separator-insensitive identity as Python package metadata."""
    return name.lower().replace("-", "_").replace(".", "_")


def distribution_record(name):
    """Resolve an installed distribution without accepting Ray's private vendors.

    Importing RLlib prepends ``ray/thirdparty_files`` to ``sys.path``.  That
    directory includes compatibility metadata for packages such as psutil, but
    it is not the environment installation imported by this pilot.  Calling
    ``importlib.metadata.version`` after the RLlib import therefore reports the
    vendor's version.  Record every match and select the first non-vendored
    distribution so pre-import and post-import audits have the same meaning.
    """
    expected_name = _normalized_distribution_name(name)
    matches = []
    for distribution in importlib.metadata.distributions():
        found_name = distribution.metadata.get("Name", "")
        if _normalized_distribution_name(found_name) != expected_name:
            continue
        metadata_path = str(getattr(distribution, "_path", ""))
        normalized_path = metadata_path.replace("\\", "/")
        matches.append(
            {
                "version": distribution.version,
                "metadata_path": metadata_path,
                "ray_vendored": "/ray/thirdparty_files/" in normalized_path,
            }
        )
    selected = next((match for match in matches if not match["ray_vendored"]), None)
    return {"selected": selected, "matches": matches}


def command(*args, cwd=None):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def audit(*, allocation=False):
    """Require the tested ABI/package pins and record the actual loaded sources."""
    errors = []
    packages = {}
    for line in (Path(__file__).with_name("requirements.txt")).read_text().splitlines():
        if "==" not in line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        name = name.split("[")[0]
        resolution = distribution_record(name)
        selected = resolution["selected"]
        actual = selected["version"] if selected else None
        packages[name] = {
            "expected": expected,
            "actual": actual,
            "metadata_path": selected["metadata_path"] if selected else None,
            "shadowed": [
                match for match in resolution["matches"] if match is not selected
            ],
        }
        # A CPU wheel may carry a local +cpu suffix with the same pinned release.
        if actual is None or actual.split("+")[0] != expected:
            errors.append(f"{name}: expected {expected}, found {actual}")
    if sys.version_info[:2] != (3, 11):
        errors.append(
            "This preflight requires Python 3.11; the old 3.10 environment is not validated."
        )
    loaded = {}
    try:
        import Basilisk
        from Basilisk.simulation import spacecraft, simpleNav
        from Basilisk.fswAlgorithms import locationPointing
        import bsk_rl

        loaded = {
            "basilisk": str(Basilisk.__path__[0]),
            "bsk_rl": str(Path(bsk_rl.__file__).resolve()),
            "native_modules": [
                m.__name__ for m in (spacecraft, simpleNav, locationPointing)
            ],
        }
        if not Path(bsk_rl.__file__).resolve().is_relative_to(ROOT / "src"):
            errors.append("Loaded bsk_rl is not the reviewed project source.")
        source = Path(os.environ.get("BASILISK_SOURCE_ROOT", Basilisk.__path__[0]))
        top = command("git", "rev-parse", "--show-toplevel", cwd=source)
        if top["returncode"] or not (Path(top["stdout"]) / "src/architecture").is_dir():
            errors.append(
                "Basilisk source checkout is unknown; set BASILISK_SOURCE_ROOT."
            )
        else:
            if (
                not Path(Basilisk.__path__[0])
                .resolve()
                .is_relative_to(Path(top["stdout"]).resolve())
            ):
                errors.append(
                    "Loaded Basilisk is not inside the recorded source build; do not pair an old wheel with a new source checkout."
                )
            loaded["basilisk_commit"] = command("git", "rev-parse", "HEAD", cwd=source)[
                "stdout"
            ]
            loaded["basilisk_dirty"] = command(
                "git", "status", "--porcelain", "--untracked-files=no", cwd=source
            )["stdout"]
            if loaded["basilisk_commit"] != BASILISK_COMMIT or loaded["basilisk_dirty"]:
                errors.append("Basilisk source differs from the clean tested commit.")
        # Record the binaries actually imported, since a source HEAD alone does
        # not prove when or from which sources an existing binary was built.
        binaries = {}
        for module in (spacecraft, simpleNav, locationPointing):
            for path in Path(module.__file__).parent.glob(
                "_" + module.__name__.split(".")[-1] + ".*"
            ):
                if path.is_file() and path.suffix in {".so", ".pyd", ".dylib"}:
                    binaries[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        loaded["native_sha256"] = binaries
    except Exception as exc:
        errors.append(f"Native import/source check failed: {exc}")

    slurm = {
        key: value for key, value in os.environ.items() if key.startswith("SLURM_")
    }
    if allocation:
        if not slurm.get("SLURM_JOB_ID"):
            errors.append(
                "Run simulations in an authorized Slurm allocation, not on the login node."
            )
        if int(slurm.get("SLURM_CPUS_PER_TASK", "0")) < 8:
            errors.append("Learning pilot requires eight allocated CPUs.")
        if int(slurm.get("SLURM_JOB_NUM_NODES", "0")) != 1:
            errors.append("Learning pilot requires exactly one node.")
        if int(slurm.get("SLURM_MEM_PER_NODE", "0")) < 32768:
            errors.append("Learning pilot requires at least 32 GiB per node.")
    paths = list((ROOT / "src/bsk_rl").rglob("*.py"))
    paths += [
        p
        for p in (ROOT / "examples/multiagent_imaging").rglob("*")
        if p.is_file() and p.suffix in {".py", ".json", ".slurm", ".sh", ".txt"}
    ]
    return dict(
        passed=not errors,
        errors=errors,
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        hostname=platform.node(),
        root=str(ROOT),
        packages=packages,
        loaded=loaded,
        slurm=slurm,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)
        },
        git=command("git", "rev-parse", "HEAD", cwd=ROOT),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-allocation", action="store_true")
    args = parser.parse_args()
    result = audit(allocation=args.require_allocation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("passed", "errors", "executable", "root")},
            indent=2,
        )
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
