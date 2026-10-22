"""Allowlisted read-only collection for terminal canary 34752183, via Termius.

No training/runtime imports, submissions, cancellations, or source/output edits.
Only the fresh collection directory and archive are written.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

JOB = "34752183"
ROOT = Path("/scratch/alpine/dahu1128/cooperative-v1-c18-canary-b205fc16-20261021/canary")
EVIDENCE = Path("/projects/dahu1128/cooperative-v1-canary-b205fc16-evidence-20261021")
FINAL_MONITOR = EVIDENCE / "monitor-20261021T223520451731Z.json"
DIRECTORIES = {"preflight", "native-transport", "training", "zero-update-restore"}
ROOT_FILES = {
    "acceptance-failed.json", "allocation-gate.json", "allocation.txt",
    "checkpoint-after.json", "checkpoint-before.json", "job-exit.json",
    "post-restore-memory.json", "sacct-at-exit.txt",
}
TRAINING_ENTRIES = {
    "checkpoint_0000", "checkpoint_0001", "checkpoint_0002",
    "effective-worker-runtime", "episode-spool", "hpo-transfer.json",
    "initial-worker-after-sync.json", "initial-worker-before-sync.json",
    "initial-worker-modules.json", "initial-worker-sync-acknowledgements.json",
    "initial_policy_gate.json", "launcher.json", "learner_batch",
    "lifetime-state.json", "memory-configuration.json", "memory-profile.jsonl",
    "pip-freeze.txt", "provenance.json", "sampling-budget.json",
    "source-snapshot.tar.gz", "startup.json", "training_hyperparameters.json",
    "tranche-stop.json", "update-journal", "updates.json", "wandb.json",
    "worker-topology-0001.json", "worker-topology-0002.json",
}
RESTORE_ENTRIES = {
    "effective-worker-runtime", "hpo-transfer.json", "launcher.json",
    "memory-configuration.json", "memory-profile.jsonl", "pip-freeze.txt",
    "provenance.json", "sampling-budget.json", "source-snapshot.tar.gz",
    "startup.json", "training_hyperparameters.json", "update-journal",
    "updates.json", "wandb.json", "zero-update-resume.json",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    receipt = subprocess.check_output([
        "sacct", "-nP", "-X", "-j", JOB,
        "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS",
    ], text=True, timeout=20)
    rows = [line.split("|") for line in receipt.splitlines() if line.strip()]
    owned = [row for row in rows if row[0] == JOB]
    if len(owned) != 1 or owned[0][1] not in {"FAILED", "COMPLETED"}:
        raise ValueError("Require the exact terminal canary before collecting")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = EVIDENCE / ("terminal-collection-" + stamp)
    destination.mkdir(exist_ok=False)
    scheduler = destination / "final-sacct.txt"
    scheduler.write_text(receipt)
    selected = {}

    def add(path, relative):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Only regular allowlisted files: " + str(path))
        if any(word in part.lower() for part in path.parts
               for word in ("secret", "credentials", "wandb_key", ".netrc")):
            raise ValueError("Credential-like path refused")
        selected[relative] = path

    for entry in sorted(ROOT.iterdir()):
        if entry.name in ROOT_FILES:
            add(entry, "canary/" + entry.name)
        elif entry.name in DIRECTORIES:
            if entry.name in {"training", "zero-update-restore"}:
                allowed = TRAINING_ENTRIES if entry.name == "training" else RESTORE_ENTRIES
                unknown = {p.name for p in entry.iterdir()} - allowed - {"wandb"}
                if unknown:
                    raise ValueError("Review new output entries: " + repr(unknown))
            for path in sorted(entry.rglob("*")):
                relative = path.relative_to(ROOT)
                if "wandb" in relative.parts or "__pycache__" in relative.parts:
                    continue
                if path.is_file() or path.is_symlink():
                    add(path, "canary/" + relative.as_posix())
        else:
            raise ValueError("Review new root entry: " + entry.name)
    for name in ROOT_FILES:
        if "canary/" + name not in selected:
            raise ValueError("Missing expected terminal receipt: " + name)
    add(scheduler, "receipts/final-sacct.txt")
    add(FINAL_MONITOR, "receipts/" + FINAL_MONITOR.name)
    add(EVIDENCE / "release/release.json", "receipts/release.json")
    add(EVIDENCE / "monitor_canary.py", "receipts/monitor_canary.py")
    add(Path(__file__), "receipts/collect_terminal.py")
    add(ROOT.parent / ("canary.slurm-" + JOB + ".out"), "receipts/canary.slurm-" + JOB + ".out")
    for path in sorted((EVIDENCE / "launch-envelope-v2").rglob("*")):
        if path.is_file() or path.is_symlink():
            add(path, "receipts/launch-envelope-v2/" + path.relative_to(EVIDENCE / "launch-envelope-v2").as_posix())
    manifest = dict(job_id=JOB, collected_utc=datetime.now(timezone.utc).isoformat(),
                    original_scheduler=receipt, excluded=["wandb cache/log directories"],
                    files=[dict(path=name, size=path.stat().st_size, sha256=digest(path))
                           for name, path in sorted(selected.items())])
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    archive = destination / "canary-evidence.tar.gz"
    with tarfile.open(archive, "x:gz") as bundle:
        bundle.add(manifest_path, arcname="manifest.json", recursive=False)
        for name, path in sorted(selected.items()):
            bundle.add(path, arcname=name, recursive=False)
    result = dict(job_id=JOB, archive=str(archive), size=archive.stat().st_size,
                  sha256=digest(archive), files=len(selected),
                  manifest_sha256=digest(manifest_path), scheduler=receipt)
    (destination / "transfer-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
