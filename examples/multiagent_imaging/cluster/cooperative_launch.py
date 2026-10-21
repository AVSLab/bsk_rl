"""Prepare immutable inputs and print reviewed sbatch commands; never submits.

No Ray/Basilisk imports, dependency installation, credentials or remote mutation.
Use the printed command yourself after reviewing its budget and release manifest.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
ACCOUNT = "ucb550_asc3"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def prepare(output):
    if git("branch", "--show-current") != "multi-agent-space-imaging-2026" or git(
        "status", "--porcelain"
    ):
        raise ValueError(
            "Prepare from the clean canonical branch after committing passing changes"
        )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = []
    for name in git(
        "ls-files", "src/bsk_rl", "examples/multiagent_imaging"
    ).splitlines():
        path = Path(name)
        if "evidence" in path.parts:
            continue
        if (
            path.suffix in {".py", ".sh", ".slurm", ".manifest"}
            or path.parent == Path("examples/multiagent_imaging/configs")
            or name
            in {
                "examples/multiagent_imaging/cluster/requirements.txt",
                "src/bsk_rl/bsk_version_req.txt",
            }
        ):
            paths.append(name)
    manifest = dict(
        version="cooperative-release-v1",
        commit=git("rev-parse", "HEAD"),
        branch=git("branch", "--show-current"),
        prepared_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={name: sha256(ROOT / name) for name in paths},
        runtime_policy="unchanged gate-parent-runtime.manifest and gate-parent-support.manifest; compute preflight required",
        account=ACCOUNT,
        partition="acpu",
        qos="cpu-normal",
        cpus=56,
        memory_gib=200,
        no_jobs_submitted=True,
    )
    (output / "release.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output / "release.json"


def validate_release(path, root=ROOT):
    record = json.loads(Path(path).read_text())
    if (
        record.get("version") != "cooperative-release-v1"
        or record.get("account") != ACCOUNT
    ):
        raise ValueError("Unexpected cooperative release/account")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        != record["commit"]
    ):
        raise ValueError("Deployed Git commit differs from reviewed release")
    for name, digest in record["source_sha256"].items():
        if (
            not (root / name).is_relative_to(root)
            or ".." in Path(name).parts
            or sha256(root / name) != digest
        ):
            raise ValueError("Source mismatch: " + name)
    return dict(
        passed=True,
        commit=record["commit"],
        source_files=len(record["source_sha256"]),
        release_sha256=sha256(path),
    )


def command(
    release,
    phase,
    output,
    *,
    arm="catalog_tx",
    variant="shared",
    wall_minutes=None,
    cpu_hour_budget=None,
    iterations=None,
    episodes=None,
    decisions=None,
    checkpoints=(),
    seeds=(),
):
    release = Path(release).resolve()
    if not release.is_file():
        raise ValueError("Reviewed release manifest required")
    if phase == "canary":
        wall_minutes, iterations, episodes, decisions = 90, 2, 56, 120000
        if cpu_hour_budget is not None and cpu_hour_budget < 84:
            raise ValueError("90-minute canary reserves 84 allocated CPU-hours")
    else:
        if not isinstance(wall_minutes, int) or not 15 <= wall_minutes <= 1440:
            raise ValueError(
                "Choose an explicit wall limit between 15 and 1440 minutes"
            )
        cost = 56 * wall_minutes / 60
        required_cost = 2 * cost if phase == "train" else cost
        if cpu_hour_budget is None or cpu_hour_budget < required_cost:
            raise ValueError(
                "Explicit allocated CPU-hour budget must cover the matched pair (training) or single evaluation job"
            )
        if phase == "train" and any(
            not isinstance(v, int) or v <= 0 for v in (iterations, episodes, decisions)
        ):
            raise ValueError(
                "Explicit positive update, mission and true-decision caps required"
            )
    if arm not in {"catalog_tx", "geometry_only"} or variant not in {
        "shared",
        "owned",
        "delivery",
    }:
        raise ValueError("Unknown treatment/variant")
    if phase == "evaluate" and (
        not checkpoints or not seeds or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            "Evaluation requires labelled checkpoints and explicit unique held-out seeds"
        )
    record = json.loads(release.read_text())
    if record["account"] != ACCOUNT:
        raise ValueError("Allocation mismatch")
    config = f"examples/multiagent_imaging/configs/cooperative_v1_{variant}_{arm}.json"
    script = ROOT / "examples/multiagent_imaging/cluster/cooperative.slurm"
    argv = [
        "sbatch",
        "--account=" + ACCOUNT,
        "--partition=acpu",
        "--qos=cpu-normal",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=56",
        "--mem=200G",
        "--no-requeue",
        "--export=ALL",
        "--time=" + str(wall_minutes),
        "--signal=B:USR1@600",
        "--job-name=cooperative-v1-" + phase + "-" + arm,
        "--output=" + str(Path(output).resolve()) + ".slurm-%j.out",
        str(script),
        phase,
        str(release),
        config,
        str(Path(output).resolve()),
        str(wall_minutes * 60),
        str(iterations or 0),
        str(episodes or 0),
        str(decisions or 0),
        *["checkpoint=" + str(p) for p in checkpoints],
        *["seed=" + str(s) for s in seeds],
    ]
    return dict(
        command=shlex.join(argv),
        argv=argv,
        allocated_cpu_hours=56 * wall_minutes / 60,
        pair_allocated_cpu_hours=2 * 56 * wall_minutes / 60
        if phase == "train"
        else None,
        account=ACCOUNT,
        phase=phase,
        config=config,
        release_commit=record["commit"],
        submitted=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="operation", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--release", type=Path, required=True)
    plan.add_argument("--phase", choices=["canary", "train", "evaluate"], required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument(
        "--arm", choices=["catalog_tx", "geometry_only"], default="catalog_tx"
    )
    plan.add_argument(
        "--variant", choices=["shared", "owned", "delivery"], default="shared"
    )
    plan.add_argument("--wall-minutes", type=int)
    plan.add_argument("--cpu-hour-budget", type=float)
    plan.add_argument("--iterations", type=int)
    plan.add_argument("--episodes", type=int)
    plan.add_argument("--decisions", type=int)
    plan.add_argument("--checkpoint", action="append", default=[])
    plan.add_argument("--seeds", type=int, nargs="+", default=[])
    a = p.parse_args()
    if a.operation == "prepare":
        print(prepare(a.output))
    else:
        print(
            json.dumps(
                command(
                    a.release,
                    a.phase,
                    a.output,
                    arm=a.arm,
                    variant=a.variant,
                    wall_minutes=a.wall_minutes,
                    cpu_hour_budget=a.cpu_hour_budget,
                    iterations=a.iterations,
                    episodes=a.episodes,
                    decisions=a.decisions,
                    checkpoints=a.checkpoint,
                    seeds=a.seeds,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
