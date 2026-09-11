# Three-sensor full-state baseline campaign

## Result

The `three-sensor-full-state-baselines-v2` campaign completed all **200**
45,000-second episodes: 50 matched seeds for each combination of independent or
centralized information and LEO-only or mixed LEO/MEO/GEO targets. Strict
aggregation found no missing or duplicate task IDs and validated all **100**
information-case pairs.

Centralized information did what this reference is intended to test. It turned
large amounts of cross-sensor duplicate work into useful cooldown-qualified
services. Averaged across the two target environments, centralized minus
independent produced **27.85 more qualified acquisition services**, **29.88 more
ground-confirmed unique services**, and **28.65 more reward**, while removing
**429.50 duplicate attempts** and **406.93 successful duplicate deliveries** per
episode. First catalog coverage rose only **0.25 percentage points** because the
finite horizon and observation geometry, rather than duplicate work alone, limited
most of the remaining targets.

## Experiment contract

- Three sensing spacecraft at 700/800/700 km are the only PettingZoo agents.
- The 100 RSO targets are propagated Basilisk/Vizard spacecraft and never agents.
- Each controller receives ten target candidates and uses conflict retasking.
- Every complete episode lasts 45,000 simulated seconds.
- The existing `reimage_cooldown_orbits=2.0` derives a sensing-team cooldown of
  11,834.835756586714 seconds, subject only to floating-point variation of
  2e-12 seconds. The cooldown starts at qualified capture time.
- Ground-confirmed coverage changes only after full physical product delivery.
- Neither controller has a broadcast or directed-transmission action. Recorded
  radio occupancy is zero in all episodes.

The independent controller reads only the deciding sensor's resources, products,
catalog, and declared target ephemerides. The centralized controller reads every
live sensor's navigation, attitude and rate, resources, current task and target
reservation, physical onboard products and owners, request epochs, and durable
time-tagged capture/completion/delivery catalog at every asynchronous decision
boundary. It then jointly assigns current actions while excluding same-target,
fresh, or in-progress duplicates. This is a maximum-information coordination
reference. Its joint assignment remains greedy and therefore is not a proof of a
globally optimal trajectory.

## Matched-seed analysis

For each target environment and seed, the independent and centralized runs have
the same initial-condition hash. The primary comparison is therefore

`effect(environment, seed) = centralized(environment, seed) - independent(environment, seed)`.

LEO and mixed each provide 50 paired effects. The two environments reuse seed IDs,
so the pooled result contains 100 environment-level effects but only 50 independent
seed blocks. Environment-specific 95% intervals bootstrap the 50 paired seeds. The
pooled interval resamples 50 blocks, keeping each seed's LEO and mixed effects
together. This preserves the noise reduction of pairing without claiming 100
independent random initializations.

Earlier project validation also used matched initial conditions. The six-cell
deterministic rollouts and the phase-two information-case validation in
`TEST_RESULTS.md` compared controllers on shared initial states. Those were smoke
and small deterministic checks. This campaign is the first 50-seed-per-environment
baseline analysis in this branch with paired inferential intervals.

## Coverage, useful services, and reward

Cell values are means over 50 seeds. The intervals are 95% bootstrap intervals.
Coverage counts each of the 100 catalog targets once. A unique service can count a
valid revisit after the shared cooldown, so service counts can exceed 100.

| Cell | Capture coverage | Ground coverage | Unique acquisitions | Unique ground services | Reward |
|---|---:|---:|---:|---:|---:|
| Independent LEO | 93.92% [93.34, 94.48] | 93.68% [93.08, 94.26] | 281.60 [279.00, 284.12] | 263.94 [261.16, 266.76] | 280.77 [277.35, 284.05] |
| Centralized LEO | 94.04% [93.46, 94.62] | 93.68% [93.08, 94.28] | 308.96 [305.78, 312.00] | 290.08 [286.26, 293.88] | 308.40 [304.55, 312.22] |
| Independent mixed | 96.78% [96.30, 97.28] | 96.60% [96.10, 97.12] | 322.60 [320.46, 324.78] | 302.20 [299.34, 305.08] | 320.01 [317.34, 322.72] |
| Centralized mixed | 97.16% [96.74, 97.60] | 96.96% [96.52, 97.42] | 350.94 [348.58, 353.32] | 335.82 [331.94, 339.62] | 349.67 [346.86, 352.46] |

The paired effects confirm that centralized coordination produces more productive
work even though first coverage is already near saturation.

| Centralized minus independent | LEO, 50 pairs | Mixed, 50 pairs | Pooled, 50 seed blocks |
|---|---:|---:|---:|
| Capture coverage | +0.12 pp [-0.10, +0.34] | +0.38 pp [+0.18, +0.64] | +0.25 pp [+0.09, +0.45] |
| Ground coverage | +0.00 pp [-0.24, +0.26] | +0.36 pp [+0.14, +0.64] | +0.18 pp [0.00, +0.39] |
| Unique acquisitions | +27.36 [+25.76, +28.96] | +28.34 [+26.68, +29.98] | +27.85 [+26.51, +29.21] |
| Unique ground services | +26.14 [+23.40, +28.86] | +33.62 [+30.36, +36.82] | +29.88 [+27.15, +32.50] |
| Reward | +27.63 [+25.76, +29.54] | +29.67 [+27.80, +31.56] | +28.65 [+27.17, +30.09] |

No LEO episode reached 100% capture coverage. Three independent and three
centralized mixed episodes reached 100% capture coverage; two episodes in each
mixed cell reached 100% ground coverage.

## Duplicate work

The independent sensors intentionally cannot see peer work. The centralized
controller removes nearly every causal duplicate but may retain tiny physical
overlaps when actions already in progress cross asynchronous boundaries.

| Cell | Duplicate attempts | Duplicate task time | Wasted fraction | Stale ground deliveries | Avoidable stale deliveries | Redundant onboard acquisitions | Excess-holder time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Independent LEO | 370.56 | 14.04 h | 37.43% | 506.54 | 113.52 | 229.40 | 87.95 h |
| Centralized LEO | 0.00 | 0.00 h | 0.00% | 145.42 | 0.50 | 2.80 | 0.43 h |
| Independent mixed | 488.44 | 16.98 h | 45.29% | 665.62 | 147.88 | 301.90 | 110.79 h |
| Centralized mixed | 0.00 | 0.00 h | 0.00% | 187.42 | 0.78 | 3.52 | 0.68 h |

“Stale” is intentionally broader than “causally avoidable.” A product is stale if
another sensor eventually delivers a newer capture of the same target; it is
causally avoidable only if the newer product had already reached the ground before
the stale product was delivered. The onboard-overlap metric separately detects
two or more sensors physically holding qualified products for the same target.

## Reward audit

The baseline reward is

`0.9 × qualified acquisition value + 0.1 × ground-delivered value + adjustments`.

Both value components are nonnegative. Across all 200 episodes, the maximum
absolute adjustment was **0**, the number of sensors with negative cumulative
reward was **0**, and the minimum per-sensor cumulative reward was positive. The
positive-components plot therefore does not discard a hidden loss term. The
configured empty-downlink penalty never contributed, and duplicate and
communication penalties were zero for this baseline. PPO training will still use
the physical-time discount in its optimization objective; that is separate from
the undiscounted episode performance total reported here.

## Why coverage was below 100%

For missed targets, the event-boundary geometry audit found:

| Cell | Mean targets missed | Missed with zero sampled illuminated LOS | Missed with a positive sampled opportunity |
|---|---:|---:|---:|
| Independent LEO | 6.08 | 5.52 | 0.56 |
| Centralized LEO | 5.96 | 5.52 | 0.44 |
| Independent mixed | 3.22 | 2.68 | 0.54 |
| Centralized mixed | 2.84 | 2.70 | 0.14 |

Thus most of the residual gap was associated with no Earth-clear illuminated LOS
sample during a returned decision boundary. The counters are diagnostic samples,
not a continuous-time reachability proof. The remaining fraction can also reflect
the ten-candidate shortlist and the greedy policy. Centralized information can
redirect available work but cannot create a viewing opportunity.

## Runtime and provenance

All 200 Slurm episode allocations completed. Median elapsed time was 262 seconds
(range 183–389), and median batch MaxRSS was 1.781 GiB (range 1.710–2.091 GiB).
Slurm allocated two CPUs per task despite the one-CPU request, for 29.237 allocated
CPU-hours. Two strict report jobs completed in 39 seconds each; the second was an
accidental duplicate aggregation and reproduced the same complete result.

The runtime audit used Python 3.11, Basilisk 2.12.0b0 from commit
`8fcb54b2fb28388efb711786630501944fddec28`, and the separate completion-v2
checkout/environment. The first ten episodes record source Git commit `01d766d`;
the remainder record cluster commit `0453fc6`, which added evidence files only.
All 200 episodes carry the identical executable source fingerprint
`ec552c0b71e5f4469cda03762a311d3eab4a27faa2df35aeca70977e6899f569`.
The AMOS checkout and environment were not modified.

## Recommendation

The baseline campaign is complete and internally consistent. It supports a bounded
three-sensor learned-policy pilot with directed finite completion sharing. Three
sensors are needed because receiver selection is trivial with two. Train fresh
weights because the extra peer slot changes the observation/action contract. Keep
completion-only, time-tagged catalog deltas; retain the capture-anchored cooldown so
shared knowledge prevents immediate duplication but permits useful revisits after
11,834.835756586714 seconds. Add the exact LEO and mixed target samplers to the
learned environment before comparing trained policies with these baselines.

Do one complete-episode, one-worker checkpoint restore/resume gate first. Only if
that passes should a bounded four-worker pilot run conflict and continuous
retasking for at most ten PPO updates each. The broad six-cell or multi-seed
learned-policy study remains a later decision.

## Artifacts

- `summary.json`: cell and paired bootstrap statistics plus metric audit.
- `episodes.csv`: one row per episode.
- `paired_differences.csv`: centralized-minus-independent effects by environment
  and seed.
- `campaign_validation.json`: strict structural and physics checks.
- `slurm_summary.json` and `slurm_accounting.psv`: cluster resource evidence.
- `manifest.json`, `runtime-login-audit.json`, and `support-data-audit.json`:
  reproducibility records.
- PNG/PDF plots: coverage, productive services/reward, action waste, catalog and
  product duplication, overlap time, and paired information effects.
