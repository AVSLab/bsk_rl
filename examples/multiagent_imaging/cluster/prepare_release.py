"""Make an immutable source overlay and base Git bundle; never upload or submit."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def prepare(output):
    root = Path(__file__).resolve().parents[3]
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
    hashes = {}
    with tarfile.open(output / "source-overlay.tar.gz", "w:gz") as archive:
        for name in sorted(set(names)):
            path = root / name
            if not name or not path.is_file():
                continue
            archive.add(path, arcname=name)
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
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
        branch="multi-agent-space-imaging-2026",
        source_sha256=hashes,
        archives={
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("base.bundle", "source-overlay.tar.gz", "deploy_release.sh")
        },
    )
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
