# 3-sensor deterministic Monte Carlo baselines

Completed 200/200 episodes; verified 100 matched information pairs.

Coverage counts the union of distinct qualified targets over the 100-target mission catalog. Ground coverage additionally requires full physical downlink. Repeat services do not increase first-coverage percentages. Stale ground deliveries and simultaneous cross-sensor onboard ownership are reported separately.

| Cell | N | Mean capture coverage | Mean ground coverage | Unique acquisition services | Unique ground services | Successful duplicates |
|---|---:|---:|---:|---:|---:|---:|
| independent_leo | 50 | 93.92% | 93.68% | 281.60 | 263.94 | 348.56 |
| independent_mixed | 50 | 96.78% | 96.60% | 322.60 | 302.20 | 466.58 |
| centralized_full_state_leo | 50 | 94.04% | 93.68% | 308.96 | 290.08 | 0.50 |
| centralized_full_state_mixed | 50 | 97.16% | 96.96% | 350.94 | 335.82 | 0.78 |

## Matched information effects

Each effect is centralized minus independent for an identical environment/seed initial condition. The pooled row contains 100 paired observations in 50 seed blocks; its bootstrap keeps each seed's LEO and mixed effects together.

| Environment | Pairs | Capture coverage | Ground coverage | Unique acquisitions | Unique ground services | Reward | Duplicate attempts |
|---|---:|---:|---:|---:|---:|---:|---:|
| LEO | 50 | +0.12 pp | +0.00 pp | +27.36 | +26.14 | +27.63 | -370.56 |
| MIXED | 50 | +0.38 pp | +0.36 pp | +28.34 | +33.62 | +29.67 | -488.44 |
| Pooled, seed-blocked | 100 | +0.25 pp | +0.18 pp | +27.85 | +29.88 | +28.65 | -429.50 |

See summary.json for all intervals and paired metrics. Early terminations remain in these statistics. Geometric visibility is sampled at event boundaries and does not prove feasibility or infeasibility.

These are deterministic heuristic information/control baselines. Centralized full-state access is an information advantage; this greedy controller is not a global optimality bound, and no 100% coverage claim is assumed.
