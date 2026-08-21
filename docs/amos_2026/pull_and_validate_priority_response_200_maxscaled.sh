#!/usr/bin/env bash

# Run on the Mac. Validate and analyze the completed max-scaled priority-response
# campaign on Alpine, package the exact response outputs, and copy them locally.

set -euo pipefail

remote=${AMOS2026_CLUSTER_HOST:-dahu1128@login-ci5.rc.colorado.edu}
remote_repo=/projects/dahu1128/bsk_rl
campaign_name=gat_mixedtrained_a0p1_priority_response_mixed_exact100LEO60MEO40GEO_200targets_prioritySum200_HIO5xMax_SHIO10xMax_controls8_45000s_20260804T010953Z
remote_root=/scratch/alpine/dahu1128/amos2026_mc/$campaign_name
local_root=/Users/dahu1128/Downloads/AMOS2026_priority_response_maxscaled_20260804
archive_name=amos2026_priority_response_maxscaled_exact_outputs.tgz
socket="${TMPDIR:-/tmp}/amos2026_pull_maxscaled_$$"

cleanup() {
    ssh -S "$socket" -O exit "$remote" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "$local_root"

echo "Connecting to Alpine. Enter your CURC password when prompted."
ssh -M -S "$socket" -o ControlPersist=600 -fN "$remote"

ssh -S "$socket" "$remote" bash -s -- "$remote_root" "$remote_repo" "$archive_name" \
    <<'REMOTE' | tee "$local_root/pull_and_validation.log"
set -euo pipefail

root=$1
repo=$2
archive_name=$3

if [[ ! -d "$root" ]]; then
    echo "Campaign root not found: $root" >&2
    exit 2
fi

cd "$repo"
source "/projects/$USER/.venv/bin/activate"
export MPLBACKEND=Agg

echo "===== Slurm accounting ====="
sacct -j 30844211,30844212,30844213,30844214,30844215,30844216,30844217,30844218,30844219,30844220 \
    --format=JobID,JobName%24,State,Elapsed,ExitCode -X || true

echo "===== Campaign validation ====="
python3 - "$root" <<'PY'
import json
import math
import pathlib
import sys

import pandas as pd

root = pathlib.Path(sys.argv[1])
statuses = sorted(root.rglob("mc_status.json"))
responses = sorted(root.rglob("priority_response_targets.csv"))
catalogs = sorted(root.rglob("target_catalog.csv"))
deliveries = sorted(root.rglob("verified_deliveries.csv"))
metrics = sorted(root.rglob("metrics_*.json"))

expected = 100
counts = {
    "mc_status.json": len(statuses),
    "priority_response_targets.csv": len(responses),
    "target_catalog.csv": len(catalogs),
    "verified_deliveries.csv": len(deliveries),
    "metrics_*.json": len(metrics),
}
print(json.dumps(counts, indent=2))
for name, count in counts.items():
    if count != expected:
        raise SystemExit(f"Expected {expected} {name} files, found {count}")

status_payloads = [json.loads(path.read_text()) for path in statuses]
bad_states = [
    (payload.get("policy_tag"), payload.get("seed"), payload.get("state"), payload.get("returncode"))
    for payload in status_payloads
    if payload.get("state") != "completed" or payload.get("returncode") != 0
]
if bad_states:
    raise SystemExit(f"Incomplete or failed runs: {bad_states[:10]}")

seeds = sorted(int(payload["seed"]) for payload in status_payloads)
if seeds != list(range(expected)):
    raise SystemExit(f"Seed coverage is not 0:100: {seeds}")

for payload in status_payloads:
    required = {
        "policy_tag": "mixed_a0p1",
        "n_targets": 200,
        "priority_sum": 200.0,
        "priority_uniform_low": 0.0,
        "priority_uniform_high": 2.0,
        "target_env": "mixed",
        "exact_mix_counts": True,
        "hio_priority_max_multiplier": 5.0,
        "shio_priority_max_multiplier": 10.0,
        "priority_control_count": 8,
    }
    for key, expected_value in required.items():
        if payload.get(key) != expected_value:
            raise SystemExit(
                f"Seed {payload.get('seed')} has {key}={payload.get(key)!r}, "
                f"expected {expected_value!r}"
            )

response = pd.concat((pd.read_csv(path) for path in responses), ignore_index=True)
if len(response) != 1600:
    raise SystemExit(f"Expected 1,600 response rows, found {len(response)}")
class_counts = response.groupby("response_class").size().to_dict()
expected_class_counts = {"CONTROL": 800, "HIO": 500, "SHIO": 300}
if class_counts != expected_class_counts:
    raise SystemExit(f"Unexpected response-class counts: {class_counts}")
if response[["seed", "target_id"]].duplicated().any():
    raise SystemExit("Duplicate seed/target rows found in priority response data")

ratios = response["event_priority"] / response["realized_initial_priority_max"]
for label, expected_ratio in (("HIO", 5.0), ("SHIO", 10.0)):
    values = ratios[response["response_class"].eq(label)]
    if not values.map(lambda value: math.isclose(value, expected_ratio, rel_tol=1e-10, abs_tol=1e-10)).all():
        raise SystemExit(f"{label} event-priority ratio failed validation")

catalog = pd.concat((pd.read_csv(path) for path in catalogs), ignore_index=True)
per_seed = catalog.groupby("seed")["initial_priority"].agg(["size", "sum", "min", "max"])
if not per_seed["size"].eq(200).all():
    raise SystemExit("At least one seed does not contain 200 catalog targets")
if not per_seed["sum"].map(lambda value: math.isclose(value, 200.0, rel_tol=1e-10, abs_tol=1e-8)).all():
    raise SystemExit("At least one seed does not have baseline priority sum 200")
if (per_seed["min"] < 0.0).any():
    raise SystemExit("Negative baseline priorities found")

print(f"seed_coverage={seeds[0]}:{seeds[-1] + 1}")
print(f"response_rows={len(response)}")
print(f"response_class_counts={class_counts}")
print(
    "realized_initial_priority_max_range="
    f"{per_seed['max'].min():.6f}:{per_seed['max'].max():.6f}"
)
print("HIO_ratio_to_realized_max=5.0")
print("SHIO_ratio_to_realized_max=10.0")
print("VALIDATION_PASSED")
PY

echo "===== Focused analysis ====="
python3 examples/amos_2026/analyze_gat_priority_response_mc.py \
    --input-root "$root" \
    --expected-seeds 0:100

echo "===== Packaging exact outputs ====="
cd "$root"
find . -type f \
    \( -name 'mc_status.json' \
       -o -name 'priority_response_targets.csv' \
       -o -name 'verified_deliveries.csv' \
       -o -name 'target_catalog.csv' \
       -o -name 'metrics_*.json' \
       -o -path './manifests/*' \
       -o -path './priority_response_analysis/*' \) \
    -print0 | tar --null -T - -czf "$archive_name"

ls -lh "$archive_name"
echo "REMOTE_ROOT=$root"
echo "ARCHIVE=$root/$archive_name"
REMOTE

rsync -ah --progress -e "ssh -S $socket" \
    "$remote:$remote_root/$archive_name" "$local_root/"

tar -xzf "$local_root/$archive_name" -C "$local_root"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$local_root/PULL_COMPLETE_UTC.txt"

echo
echo "Copied and unpacked to: $local_root"
