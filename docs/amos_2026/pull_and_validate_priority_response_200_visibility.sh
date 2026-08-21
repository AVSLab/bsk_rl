#!/usr/bin/env bash

# Run on the Mac after jobs 30846501--30846510 finish. This validates the
# visibility-enabled max-scaled campaign, packages the exact target response
# outputs, copies them locally, and regenerates the paper's Figure 5 data.

set -euo pipefail

remote=${AMOS2026_CLUSTER_HOST:-dahu1128@login-ci5.rc.colorado.edu}
remote_repo=/projects/dahu1128/bsk_rl
campaign_name=gat_mixedtrained_a0p1_priority_response_mixed_exact100LEO60MEO40GEO_200targets_prioritySum200_HIO5xMax_SHIO10xMax_controls8_45000s_20260804T052230Z
remote_root=/scratch/alpine/dahu1128/amos2026_mc/$campaign_name
local_root=/Users/dahu1128/Downloads/AMOS2026_priority_response_visibility_20260804
paper_root=/Users/dahu1128/Documents/PhD/Conferences/AMOS2026/AMOS_conference_paper_2026
archive_name=amos2026_priority_response_visibility_exact_outputs.tgz
socket="${TMPDIR:-/tmp}/amos2026_pull_visibility_$$"

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
sacct -j 30846501,30846502,30846503,30846504,30846505,30846506,30846507,30846508,30846509,30846510 \
    --format=JobID,JobName%24,State,Elapsed,ExitCode -X

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

counts = {
    "mc_status.json": len(statuses),
    "priority_response_targets.csv": len(responses),
    "target_catalog.csv": len(catalogs),
    "verified_deliveries.csv": len(deliveries),
    "metrics_*.json": len(metrics),
}
print(json.dumps(counts, indent=2))
for name, count in counts.items():
    if count != 100:
        raise SystemExit(f"Expected 100 {name} files, found {count}")

payloads = [json.loads(path.read_text()) for path in statuses]
bad = [
    (item.get("seed"), item.get("state"), item.get("returncode"))
    for item in payloads
    if item.get("state") != "completed" or item.get("returncode") != 0
]
if bad:
    raise SystemExit(f"Incomplete or failed runs: {bad[:10]}")
if sorted(int(item["seed"]) for item in payloads) != list(range(100)):
    raise SystemExit("Seed coverage is not exactly 0:100")

response = pd.concat((pd.read_csv(path) for path in responses), ignore_index=True)
if len(response) != 1600:
    raise SystemExit(f"Expected 1,600 response rows, found {len(response)}")
expected_classes = {"CONTROL": 800, "HIO": 500, "SHIO": 300}
if response.groupby("response_class").size().to_dict() != expected_classes:
    raise SystemExit("Unexpected HIO/SHIO/control row counts")

required_telemetry = {
    "first_candidate_delay_sec",
    "first_eligible_visible_delay_sec",
    "first_successful_image_delay_sec",
    "first_useful_downlink_delay_sec",
}
missing = required_telemetry - set(response.columns)
if missing:
    raise SystemExit(f"Missing response telemetry columns: {sorted(missing)}")

response["class"] = response["response_class"].replace({"CONTROL": "Control"})
candidate_coverage = (
    response.assign(has_candidate=response["first_candidate_delay_sec"].notna())
    .groupby(["seed", "class"])["has_candidate"]
    .any()
)
if not candidate_coverage.all():
    raise SystemExit(
        "At least one seed/class has no candidate-list event: "
        f"{list(candidate_coverage[~candidate_coverage].index)[:10]}"
    )

ratios = response["event_priority"] / response["realized_initial_priority_max"]
for label, expected in (("HIO", 5.0), ("SHIO", 10.0)):
    values = ratios[response["response_class"].eq(label)]
    if not values.map(
        lambda value: math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-10)
    ).all():
        raise SystemExit(f"{label} event-priority ratio failed validation")

catalog = pd.concat((pd.read_csv(path) for path in catalogs), ignore_index=True)
per_seed = catalog.groupby("seed")["initial_priority"].agg(["size", "sum"])
if not per_seed["size"].eq(200).all():
    raise SystemExit("At least one seed does not contain 200 targets")
if not per_seed["sum"].map(
    lambda value: math.isclose(value, 200.0, rel_tol=1e-10, abs_tol=1e-8)
).all():
    raise SystemExit("At least one seed does not have priority sum 200")

print(f"candidate_target_coverage_pct={100.0 * response['first_candidate_delay_sec'].notna().mean():.2f}")
print(f"visible_target_coverage_pct={100.0 * response['first_eligible_visible_delay_sec'].notna().mean():.2f}")
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
REMOTE

rsync -ah --progress -e "ssh -S $socket" \
    "$remote:$remote_root/$archive_name" "$local_root/"

tar -xzf "$local_root/$archive_name" -C "$local_root"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$local_root/PULL_COMPLETE_UTC.txt"

source_csv=$local_root/priority_response_analysis/priority_response_targets_combined.csv
if [[ ! -f "$source_csv" ]]; then
    echo "Combined response output is missing after extraction: $source_csv" >&2
    exit 3
fi

echo "===== Regenerating paper Figure 5 ====="
AMOS2026_PRIORITY_RESPONSE_SOURCE="$source_csv" \
    /Users/dahu1128/Repositories/bsk_rl/.venv/bin/python \
    "$paper_root/analyze_maxscaled_priority_response.py"

echo
echo "Copied, validated, and regenerated Figure 5 from: $local_root"
