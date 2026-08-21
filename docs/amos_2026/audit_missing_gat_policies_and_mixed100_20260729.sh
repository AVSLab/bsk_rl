#!/usr/bin/env bash
set -euo pipefail

AUDIT_DATE=${1:-20260729}
HOST=${AMOS2026_CLUSTER_HOST:-dahu1128@login-ci5.rc.colorado.edu}
DEST="/Users/dahu1128/Downloads/AMOS2026_targeted_audit_${AUDIT_DATE}"
ARCHIVE="$DEST/amos2026_targeted_audit_${AUDIT_DATE}.tgz"

mkdir -p "$DEST"

echo "Connecting to $HOST."
echo "This audit is read-only. Enter your Alpine password when prompted."

ssh "$HOST" 'bash -s' >"$ARCHIVE" <<'REMOTE'
set -euo pipefail

MC_BASE="/scratch/alpine/$USER/amos2026_mc"
POLICY_ROOT="/scratch/alpine/$USER/rllib_results"
TMP_DIR=$(mktemp -d)
OUT="$TMP_DIR/amos2026_targeted_audit"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$OUT/manifests"

{
    printf 'created_at_utc='
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    printf 'host=%s\n' "$(hostname -f)"
    printf 'mc_base=%s\n' "$MC_BASE"
    printf 'policy_root=%s\n' "$POLICY_ROOT"
} >"$OUT/context.txt"

find "$MC_BASE" -mindepth 1 -maxdepth 1 -type d \
    -printf '%TY-%Tm-%TdT%TH:%TM:%TSZ\t%p\n' \
    | sort >"$OUT/campaign_roots.tsv"

printf 'policy_tag,run_dir,checkpoint_count,latest_checkpoint\n' \
    >"$OUT/checkpoint_inventory.csv"

for tag in 60d40i 70d30i 80d20i 90d10i; do
    while IFS= read -r run_dir; do
        checkpoint_count=$(
            find "$run_dir" -mindepth 1 -maxdepth 4 -type d \
                -name 'checkpoint_*' -print | wc -l | tr -d ' '
        )
        latest_checkpoint=$(
            find "$run_dir" -mindepth 1 -maxdepth 4 -type d \
                -name 'checkpoint_*' -print | sort -V | tail -n 1
        )
        printf '"%s","%s",%s,"%s"\n' \
            "$tag" "$run_dir" "$checkpoint_count" "$latest_checkpoint" \
            >>"$OUT/checkpoint_inventory.csv"
    done < <(
        find "$POLICY_ROOT" -mindepth 1 -maxdepth 1 \
            \( -type d -o -type l \) \
            -iname "*gat*${tag}*" -print | sort
    )
done

python3 - "$MC_BASE" "$OUT" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

mc_base = Path(sys.argv[1])
out = Path(sys.argv[2])
focus_policies = {"60d40i", "70d30i", "80d20i", "90d10i"}

records = []
for status_path in sorted(mc_base.rglob("mc_status.json")):
    try:
        status = json.loads(status_path.read_text())
    except Exception as exc:
        records.append(
            {
                "campaign": status_path.relative_to(mc_base).parts[0],
                "policy_tag": "",
                "seed": "",
                "state": "bad_json",
                "target_env": "",
                "n_targets": "",
                "total_time_sec": "",
                "policy_name": "",
                "obs_version": "",
                "status_path": str(status_path),
                "error": str(exc),
            }
        )
        continue

    relative = status_path.relative_to(mc_base)
    policy_name = str(status.get("policy_name", ""))
    records.append(
        {
            "campaign": relative.parts[0],
            "policy_tag": str(status.get("policy_tag", "")),
            "seed": status.get("seed", ""),
            "state": str(status.get("state", "")),
            "target_env": str(status.get("target_env", "")),
            "n_targets": status.get("n_targets", ""),
            "total_time_sec": status.get("total_time_sec", ""),
            "policy_name": policy_name,
            "obs_version": "9" if "obs_v9" in policy_name.lower() else "",
            "status_path": str(status_path),
            "error": "",
        }
    )

fields = [
    "campaign",
    "policy_tag",
    "seed",
    "state",
    "target_env",
    "n_targets",
    "total_time_sec",
    "policy_name",
    "obs_version",
    "status_path",
    "error",
]
with (out / "all_mc_runs.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)

counts = Counter(
    (
        row["campaign"],
        row["target_env"],
        str(row["n_targets"]),
        row["policy_tag"],
        row["state"],
        row["obs_version"],
    )
    for row in records
)
with (out / "mc_policy_counts.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "campaign",
            "target_env",
            "n_targets",
            "policy_tag",
            "state",
            "obs_version",
            "run_count",
        ]
    )
    for key, count in sorted(counts.items()):
        writer.writerow([*key, count])

relevant = [
    row
    for row in records
    if row["policy_tag"] in focus_policies
    or (
        row["target_env"].lower() == "mixed"
        and str(row["n_targets"]) == "100"
    )
]
with (out / "focus_runs.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(relevant)

summary = {
    "status_file_count": len(records),
    "focus_run_count": len(relevant),
    "missing_alpha_policy_counts": {
        policy: sum(
            row["policy_tag"] == policy and row["state"] == "completed"
            for row in records
        )
        for policy in sorted(focus_policies)
    },
    "mixed_100_completed_count": sum(
        row["target_env"].lower() == "mixed"
        and str(row["n_targets"]) == "100"
        and row["state"] == "completed"
        for row in records
    ),
    "mixed_100_campaigns": sorted(
        {
            row["campaign"]
            for row in records
            if row["target_env"].lower() == "mixed"
            and str(row["n_targets"]) == "100"
        }
    ),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY

while IFS= read -r manifest; do
    campaign=$(basename "$(dirname "$(dirname "$manifest")")")
    cp "$manifest" "$OUT/manifests/${campaign}__$(basename "$manifest")"
done < <(find "$MC_BASE" -type f -path '*/manifests/*.json' -print | sort)

tar -C "$TMP_DIR" -czf - "$(basename "$OUT")"
REMOTE

tar -xzf "$ARCHIVE" -C "$DEST"

echo
echo "Targeted audit copied to:"
echo "  $DEST"
echo
echo "Key files:"
echo "  $DEST/amos2026_targeted_audit/summary.json"
echo "  $DEST/amos2026_targeted_audit/checkpoint_inventory.csv"
echo "  $DEST/amos2026_targeted_audit/mc_policy_counts.csv"
echo "  $DEST/amos2026_targeted_audit/focus_runs.csv"
