"""Read-only post-terminal audit for the confirmed initial-evidence reader defect.

Use allowlisted Termius downloads, then run locally against the collected files
and a hash-matched b205fc16 source snapshot. No submission, sampling, policy
inference or dependency change. Preserve the original Slurm failure and output.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

JOB = "34752183"
COMMIT = "b205fc1603b5ca887aaadcb3c34de2700d563c0c"
SOURCE = Path("/projects/dahu1128/bsk_rl-cooperative-v1-b205fc16-20261021")
RAW = Path(
    "/scratch/alpine/dahu1128/cooperative-v1-c18-canary-b205fc16-20261021/canary"
)
EVIDENCE = Path("/projects/dahu1128/cooperative-v1-canary-b205fc16-evidence-20261021")


def small_json(path):
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("Metadata exceeds 4 MiB: " + str(path))
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--verifier-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--raw-output", type=Path, default=RAW)
    parser.add_argument(
        "--release", type=Path, default=EVIDENCE / "release/release.json"
    )
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--terminal-sacct", type=Path)
    args = parser.parse_args()
    if socket.gethostname().startswith("login"):
        raise ValueError(
            "Numerical batch audit must run locally after verified downloads"
        )
    raw_root, source_root = args.raw_output.resolve(), args.source_root.resolve()
    if args.terminal_sacct:
        if args.terminal_sacct.stat().st_size > 4 * 1024**2:
            raise ValueError("Scheduler receipt exceeds 4 MiB")
        scheduler = args.terminal_sacct.read_text()
    elif os.environ.get("SLURM_JOB_ID"):
        scheduler = subprocess.check_output(
            [
                "sacct",
                "-nP",
                "-X",
                "-j",
                JOB,
                "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS",
            ],
            text=True,
            timeout=15,
        )
    else:
        raise ValueError("Provide the independently transferred final sacct receipt")
    rows = [line.split("|") for line in scheduler.splitlines() if line.strip()]
    owned = [row for row in rows if row[0] == JOB]
    if len(owned) != 1 or owned[0][1] not in {"FAILED", "COMPLETED"}:
        raise ValueError(
            "Require terminal canary; never audit a live/failed-training run"
        )
    failure = small_json(raw_root / "acceptance-failed.json")
    if failure.get("error_type") != "KeyError" or failure.get("error") != "'workers'":
        raise ValueError("Only the demonstrated reader-schema defect is covered")
    if (raw_root / "training/training-failed.json").exists():
        raise ValueError("Training failure needs its own investigation")
    if hashlib.sha256(args.release.read_bytes()).hexdigest() != args.release_sha256:
        raise ValueError("Release receipt differs from its verified transfer digest")
    release = small_json(args.release)
    if release["commit"] != COMMIT or len(release["source_sha256"]) != 186:
        raise ValueError("Original trainer source identity differs")
    for name, digest in release["source_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source manifest entry")
        with (source_root / relative).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("Archived trainer source changed: " + name)
    validated = dict(
        commit=COMMIT,
        source_files=186,
        source_root=str(source_root),
        release_sha256=args.release_sha256,
        source_file_hashes_passed=True,
    )
    verifier_hash = hashlib.sha256(args.verifier.read_bytes()).hexdigest()
    if verifier_hash != args.verifier_sha256:
        raise ValueError("Independent verifier differs from its reviewed digest")
    memory = raw_root / "post-restore-memory.json"
    small_json(memory)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in (
        "training",
        "preflight",
        "zero-update-restore",
        "checkpoint-after.json",
    ):
        (args.output / name).symlink_to(
            raw_root / name, target_is_directory=(raw_root / name).is_dir()
        )
    sys.path[:0] = [str(source_root), str(source_root / "src")]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    spec = importlib.util.spec_from_file_location(
        "independent_cooperative_accept", args.verifier
    )
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    import torch

    torch.set_num_threads(1)
    from examples.multiagent_imaging.config import MultiAgentImagingConfig

    record = dict(
        observed_utc=datetime.now(timezone.utc).isoformat(),
        job_id=JOB,
        original_scheduler_record=scheduler,
        original_acceptance_failure=failure,
        original_source=validated,
        independent_verifier_sha256=verifier_hash,
        new_PPO_updates=0,
        new_native_missions=0,
        scope="Corrected evidence verification; original Slurm exit status remains unchanged",
        passed=False,
    )
    try:
        record["acceptance"] = gate.accept(
            args.output,
            MultiAgentImagingConfig.from_json(
                source_root
                / "examples/multiagent_imaging/configs/cooperative_v1_shared_catalog_tx.json"
            ),
            max_updates=2,
            max_episodes=56,
            max_decisions=120000,
            post_restore_memory=memory,
        )
        record["passed"] = True
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        with (args.output / "independent-audit-outcome.json").open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
    print(
        json.dumps(
            {k: record[k] for k in ("passed", "job_id", "scope", "new_PPO_updates")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
