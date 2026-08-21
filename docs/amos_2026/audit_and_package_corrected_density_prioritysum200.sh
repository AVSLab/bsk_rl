#!/usr/bin/env bash
set -euo pipefail

# Run on Alpine after, or while, the corrected density campaigns execute.
# It validates every completed status against the intended configuration. Once
# both campaigns are complete, it runs both aggregate analyses and packages the
# compact paper inputs for transfer to the Mac.

requested_root=${1:-}
if [[ -n "$requested_root" ]]; then
    BASE_ROOT=$requested_root
else
    BASE_ROOT=$(find "/scratch/alpine/$USER/amos2026_mc" -maxdepth 1 -type d \
        -name 'corrected_density_prioritySum200_*' -print | sort | tail -1)
fi

if [[ -z "${BASE_ROOT:-}" || ! -f "$BASE_ROOT/campaign_paths.env" ]]; then
    echo "Could not locate a corrected priority-sum-200 campaign." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$BASE_ROOT/campaign_paths.env"
cd "/projects/$USER/bsk_rl"
source "/projects/$USER/.venv/bin/activate"
export MPLBACKEND=Agg
export MPLCONFIGDIR="/scratch/alpine/$USER/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR" "$BASE_ROOT/audit"

python3 - "$LEO_OUTPUT_ROOT" "$MIXED_OUTPUT_ROOT" \
    "$BASE_ROOT/audit/status_audit.json" "$POLICY_TAGS" <<'PY'
import json
import sys
import csv
from collections import Counter, defaultdict
from pathlib import Path

roots = {"leo": Path(sys.argv[1]), "mixed": Path(sys.argv[2])}
output = Path(sys.argv[3])
policy_tags = tuple(tag for tag in sys.argv[4].split(",") if tag)
expected_seeds = set(range(100))
report = {"expected_priority_sum": 200.0, "expected_n_targets": 200, "environments": {}}

for expected_env, root in roots.items():
    states = Counter()
    completed = defaultdict(set)
    invalid = []
    statuses = sorted(root.glob("seeds_*/*/seed_*/mc_status.json"))
    for path in statuses:
        try:
            status = json.loads(path.read_text())
        except Exception as exc:
            invalid.append({"path": str(path), "reason": f"unreadable: {exc}"})
            continue
        state = str(status.get("state", "unknown"))
        states[state] += 1
        reasons = []
        if status.get("target_env") != expected_env:
            reasons.append(f"target_env={status.get('target_env')!r}")
        if int(status.get("n_targets", -1)) != 200:
            reasons.append(f"n_targets={status.get('n_targets')!r}")
        if abs(float(status.get("priority_sum", -1.0)) - 200.0) > 1e-9:
            reasons.append(f"priority_sum={status.get('priority_sum')!r}")
        if status.get("evaluation_reward_mix") != "100d00i":
            reasons.append(f"evaluation_reward_mix={status.get('evaluation_reward_mix')!r}")
        tag = str(status.get("policy_tag", ""))
        seed = int(status.get("seed", -1))
        if tag not in policy_tags:
            reasons.append(f"policy_tag={tag!r}")
        if seed not in expected_seeds:
            reasons.append(f"seed={seed!r}")
        if reasons:
            invalid.append({"path": str(path), "reason": ", ".join(reasons)})
        elif state == "completed" and status.get("returncode", 0) in (0, None):
            catalog_paths = sorted(path.parent.glob("target_catalog.csv"))
            catalog_paths.extend(sorted(path.parent.glob("*/target_catalog.csv")))
            if not catalog_paths:
                invalid.append({"path": str(path), "reason": "missing target_catalog.csv"})
                continue
            catalog_path = max(catalog_paths, key=lambda item: item.stat().st_mtime)
            try:
                with catalog_path.open(newline="") as handle:
                    catalog = list(csv.DictReader(handle))
                initial_sum = sum(float(row["initial_priority"]) for row in catalog)
            except Exception as exc:
                invalid.append(
                    {"path": str(catalog_path), "reason": f"invalid target catalog: {exc}"}
                )
                continue
            if len(catalog) != 200 or abs(initial_sum - 200.0) > 1e-8:
                invalid.append(
                    {
                        "path": str(catalog_path),
                        "reason": (
                            f"catalog rows={len(catalog)}, "
                            f"initial_priority_sum={initial_sum:.12g}"
                        ),
                    }
                )
                continue
            completed[tag].add(seed)

    missing = {
        tag: sorted(expected_seeds - completed[tag])
        for tag in policy_tags
        if completed[tag] != expected_seeds
    }
    report["environments"][expected_env] = {
        "root": str(root),
        "status_file_count": len(statuses),
        "states": dict(sorted(states.items())),
        "valid_completed_count": sum(len(seeds) for seeds in completed.values()),
        "expected_count": len(policy_tags) * len(expected_seeds),
        "missing_by_policy": missing,
        "invalid_statuses": invalid,
        "complete": not missing and not invalid,
    }

report["all_complete"] = all(
    item["complete"] for item in report["environments"].values()
)
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY

all_complete=$(python3 - "$BASE_ROOT/audit/status_audit.json" <<'PY'
import json
import sys
print("1" if json.load(open(sys.argv[1]))["all_complete"] else "0")
PY
)

if [[ "$all_complete" != "1" ]]; then
    echo
    echo "Campaign is not complete yet. Audit saved at:"
    echo "  $BASE_ROOT/audit/status_audit.json"
    exit 0
fi

for root in "$LEO_OUTPUT_ROOT" "$MIXED_OUTPUT_ROOT"; do
    python3 examples/amos_2026/analyze_gat_reward_sweep_mc.py \
        --input-root "$root" --expected-seeds 0:100 --policy-tags "$POLICY_TAGS"
    python3 examples/amos_2026/analyze_gat_reward_sweep_mc_detailed.py \
        --input-root "$root" --expected-seeds 0:100 --policy-tags "$POLICY_TAGS"
done

archive="$BASE_ROOT/amos2026_corrected_density_prioritySum200_results.tgz"
tar -czf "$archive" -C "$BASE_ROOT" \
    campaign_paths.env submitted_jobs.tsv SUBMISSION_COMPLETE_UTC.txt audit manifests \
    "$(basename "$LEO_OUTPUT_ROOT")/analysis" \
    "$(basename "$LEO_OUTPUT_ROOT")/analysis_detailed" \
    "$(basename "$MIXED_OUTPUT_ROOT")/analysis" \
    "$(basename "$MIXED_OUTPUT_ROOT")/analysis_detailed"

echo
echo "Validated and packaged corrected results:"
echo "  $archive"
