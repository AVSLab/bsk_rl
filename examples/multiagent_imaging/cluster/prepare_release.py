"""Make an immutable source overlay and base Git bundle; never upload or submit."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def prepare(output, *, root=None):
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[3]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use an empty release directory.")
    names = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "src/bsk_rl",
                "examples/multiagent_imaging",
                "tests/unittest/multiagent",
                "tests/integration/multiagent",
                "pyproject.toml",
                "README.md",
                "LICENSE",
                ".gitignore",
            ],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    # Inherited global ignore rules can hide import-critical __init__.py and
    # utils.py. An immutable release must include the actual package tree,
    # rather than assuming Git visibility establishes executable completeness.
    required_python = [str(path.relative_to(root)) for folder in
        (root / "src/bsk_rl", root / "examples/multiagent_imaging")
        for path in folder.rglob("*.py") if "__pycache__" not in path.parts
        and "evidence" not in path.parts]
    # Evidence and bytecode are historical/runtime artifacts, not release source.
    names = sorted(name for name in set(names) | set(required_python)
                   if "evidence" not in Path(name).parts
                   and "__pycache__" not in Path(name).parts
                   and not name.endswith((".pyc", ".pyo")))
    hashes = {}
    with tarfile.open(output / "source-overlay.tar.gz", "w:gz") as archive:
        for name in sorted(set(names)):
            path = root / name
            if not name or not path.is_file():
                continue
            archive.add(path, arcname=name)
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    missing = sorted(set(required_python) - set(hashes))
    if missing:
        raise RuntimeError("Import-critical Python files missing from release: " + repr(missing))
    # Base history is small enough to bundle; this avoids relying on an unpushed
    # branch name. Dirty/new readiness files are preserved in the overlay above.
    subprocess.run(
        ["git", "bundle", "create", str(output / "base.bundle"), "HEAD"],
        cwd=root,
        check=True,
    )
    (output / "deploy_release.sh").write_bytes(
        Path(__file__).with_name("deploy_release.sh").read_bytes()
    )
    record = dict(
        base_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"],cwd=root,text=True).strip(),
        source_sha256=hashes,
        archives={
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("base.bundle", "source-overlay.tar.gz", "deploy_release.sh")
        },
    )
    from examples.multiagent_imaging.cluster.release_manifest import (
        STUDY_CONFIGS, validate_study_manifest,
    )
    record["release_id"] = (
        "multiagent-space-imaging-" + record["base_commit"][:12]
        + "-" + record["archives"]["source-overlay.tar.gz"][:12]
    )
    record["config_sha256"] = {
        arm: hashes[name] for arm, name in STUDY_CONFIGS.items()
    }
    validate_study_manifest(root, record)
    (output / "release.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha}  {name}\n" for name, sha in record["archives"].items())
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
