"""Read-only post-terminal audit for the confirmed initial-evidence reader defect.

Run through Termius. No submission, sampling, policy inference or dependency
change. Preserve the original Slurm failure and raw output without rewriting it.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

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
    args = parser.parse_args()
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
    rows = [line.split("|") for line in scheduler.splitlines() if line.strip()]
    owned = [row for row in rows if row[0] == JOB]
    if len(owned) != 1 or owned[0][1] not in {"FAILED", "COMPLETED"}:
        raise ValueError(
            "Require terminal canary; never audit a live/failed-training run"
        )
    failure = small_json(RAW / "acceptance-failed.json")
    if failure.get("error_type") != "KeyError" or failure.get("error") != "'workers'":
        raise ValueError("Only the demonstrated reader-schema defect is covered")
    if (RAW / "training/training-failed.json").exists():
        raise ValueError("Training failure needs its own investigation")
    from examples.multiagent_imaging.cluster.cooperative_launch import validate_release

    validated = validate_release(EVIDENCE / "release/release.json", root=SOURCE)
    if validated["commit"] != COMMIT:
        raise ValueError("Original trainer source changed")
    verifier_hash = hashlib.sha256(args.verifier.read_bytes()).hexdigest()
    if verifier_hash != args.verifier_sha256:
        raise ValueError("Independent verifier differs from its reviewed digest")
    memory = RAW / "post-restore-memory.json"
    small_json(memory)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in (
        "training",
        "preflight",
        "zero-update-restore",
        "checkpoint-after.json",
    ):
        (args.output / name).symlink_to(
            RAW / name, target_is_directory=(RAW / name).is_dir()
        )
    spec = importlib.util.spec_from_file_location(
        "independent_cooperative_accept", args.verifier
    )
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
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
                SOURCE
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
