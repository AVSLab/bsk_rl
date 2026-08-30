# Research Focus I campaign status — 2026-08-30

This record separates last-confirmed Alpine state from work that is merely
implemented or submitted. Live Alpine state could not be queried from the local
Codex session because CURC requires interactive authentication.

## Trained policies

- The six memory-safe 100-second exploratory policies completed: monolithic MLP
  and target-set attention at K=5, 10, and 20, seed 10001. Each ran for about 47
  hours and produced final/retained checkpoints. They do not need retraining for
  the exploratory comparison.
- The isolated N=100, K=10, 300-second target-set attention control completed its
  gate and three training allocations (last recorded jobs 31561686--31561689).
  It needs held-out checkpoint selection, not another training run.
- The requested Breckenridge imaging-versus-downlink alpha=0 monolithic policy
  is the October 14, 2025 observation-v7 `0d100i/checkpoint_000145` artifact.
  Its 81-input, 13-action, 8,757,262-parameter module has SHA-256
  `0d8033272f14cdd408192d7ab6ee819b18691c9385fca87be24044fc950464d2`.

## Training failures and interpretation

- The original N=100--400, 28-runner design suffered Ray worker deaths/OOM and
  must not be resumed. The memory-safe replacement uses N=100--200, 12 runners,
  16 requested CPUs, and 230 GiB.
- Early stress jobs ended at the advance SIGTERM boundary before a PPO iteration
  completed. That was a gate timing problem, not evidence that the simulator or
  policy was broken.
- Comparing PPO iteration counts was misleading: the exploratory study uses
  4,200 transitions per iteration versus 180 in the historical trainer. The
  six final runs reached roughly 290k--319k environment transitions each.
- Rising training return is only a diagnostic. Architecture claims must use the
  held-out checkpoint and paired Monte Carlo results.

## Evaluation state

- The first six-cell validation job 31614765 timed out after 24 hours because
  each array task serially ran about 90 full episodes and wrote only at the end.
  This was an evaluation-layout failure; individual simulations were advancing.
- Atomic one-episode validation was submitted as jobs 31722716 (400 tasks),
  31722718 (140 tasks), and collector 31722719. The last confirmed state on
  August 27 was one task running and the others pending for priority. Completion
  is not confirmed locally. The gate is exactly 540 atomic validation rows and
  six `best_validation` links.

## Monte Carlo inventory

- Complete historical closest-angle heuristic: N=100, 200, 400; seeds 0--99;
  fixed 100-second imaging; 300 episodes at
  `/scratch/alpine/$USER/prospectus_rfi/heuristic_mc/amos2025_closest_angle_100s_20260815T183838Z`.
- Complete August AMOS frozen-policy transfer: N=100, 200, 400; seeds 0--99;
  the 300-second-trained policy evaluated at 100 seconds; 300 episodes at
  `/scratch/alpine/$USER/prospectus_rfi/legacy_policy_mc/amos2025_alpha0_300s_to_100s_20260817T004436Z`.
  This is an out-of-distribution transfer baseline and is not the Breckenridge
  alpha=0 policy requested for the final 300-second comparison.
- The 600-task acquisition-timeline replay was launched as job 31360337, but a
  final completion audit was not recorded. Do not use its analysis until all
  expected replay outputs validate.
- The 100-second architecture Monte Carlo has not been confirmed as launched.
  After validation it now evaluates MLP, attention, full-catalog smallest-angle,
  full-catalog closest-distance, and K-candidate matched-angle methods over
  K=5/10/20, N=100/200/300/400, and seeds 700000--700099: 6,000 episode rows.
- The matched 300-second comparison is implemented but not submitted: exact
  Breckenridge alpha=0 MLP, 300-second target-set attention, full-catalog
  smallest-angle, and full-catalog closest-distance over seeds 0--99: 400 rows.

## Immediate cluster order

1. Audit jobs 31722716, 31722718, and 31722719 and count validation rows. Resume
   only missing atomic tasks until the collector creates all six links.
2. Launch the 6,000-row 100-second paired campaign with
   `submit_memorysafe_paired_mc.sh`; do not retrain the six exploratory policies.
3. Locate the completed 300-second attention run directory and launch
   `submit_amos2025_matched_300s.sh`. This selects its checkpoint on held-out
   seeds before the 400-row evaluation.
4. Run the strict collectors/analysis. Only then replace the preliminary values
   in `sec_focus1.tex`.
5. If the prospectus will claim architecture superiority rather than present an
   exploratory result, train confirmatory seeds 20001 and 30001 for the selected
   MLP and attention configurations. One training seed per cell is insufficient
   for a final architecture-level claim.
