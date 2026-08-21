# AMOS 2026 Targeted Alpine Audit

Audit date: 2026-07-29

Source package:

`/Users/dahu1128/Downloads/AMOS2026_targeted_audit_20260729`

## Findings

- The Alpine policy root contains compatible observation-version-9 GAT
  training runs for `60d40i`, `70d30i`, `80d20i`, and `90d10i`.
- Each of those four training runs contains four checkpoint directories.
- The audit parsed 3,300 `mc_status.json` files under the AMOS 2026 Monte
  Carlo root.
- None of those status files belongs to `60d40i`, `70d30i`, `80d20i`, or
  `90d10i`; those policies therefore have no completed Monte Carlo campaign.
- No status file has both `target_env=mixed` and `n_targets=100`. A compatible
  mixed-100 Monte Carlo campaign is not currently available.
- Local metadata records observation-version-9 mixed-random GAT training
  configurations at alpha 0.1 for both a fixed 100-target catalog and a
  variable 100--300-target catalog. The local copies contain configuration
  YAML files but no checkpoints or training histories, so completion of those
  training runs is not established by the copied data.
- A separate, completed 100-seed mixed-100 campaign exists under
  `rllib_results/breckenridge2026_mc`, but it is not compatible with the AMOS
  sweep. It uses an observation-version-7 alpha-0.1 policy, fixed-duration
  actions, action shielding, uniform target priorities, and no HIO/SHIO event.
  It must not be added to the AMOS Figure 4 comparison.

## Consequences for the paper

- The current eight-policy plots correctly show only policies with matched
  100-seed results.
- Alpha 0.6, 0.7, 0.8, and 0.9 should be added after a common-seed evaluation
  is run with the same 200-target mixed scenario and evaluation reward.
- Figure 4 cannot yet include a mixed-100 curve without running a new
  compatible campaign. The figure generator will load
  `data/comparisons/mixed_100_summary_by_policy.csv` automatically when that
  result is available.
- The missing Figure 4 curve is an evaluation of the same frozen,
  observation-version-9 LEO-trained reward-sweep policies in a 100-target
  mixed catalog. It is distinct from evaluating a mixed-trained alpha-0.1
  checkpoint, which would be a separate training-domain ablation and only one
  policy point unless additional mixed-trained reward weights are produced.

## Preserved audit files

- `summary.json`: headline counts.
- `checkpoint_inventory.csv`: matching GAT run and checkpoint inventory.
- `mc_policy_counts.csv`: campaign, policy, configuration, and state counts.
- `focus_runs.csv`: runs matching the missing policy tags or mixed-100 setup.
- `manifests/`: copies of all AMOS 2026 frozen checkpoint manifests found on
  Alpine.
