# Completion observation/action contract: completion-v2

The example uses ordinary composable BSK-RL observations. In broadcast mode the shape
is `26 + 17*K` with `K+5` actions. Directed mode appends `P = N_sensors-1` peer rows:
`26 + 17*K + 12*P` observations and `K+5+P` actions. The full two-sensor/ten-candidate
configuration therefore has **208 observations and 16 actions**. Passive RSO count does
not change these dimensions. `completion-v1` and legacy intent checkpoints require
retraining: attention, pooling and the operational head have changed even for broadcast.

Checkpoint validation compares the complete semantic contract, not just dimensions.
Only scenario seed may change during held-out evaluation. The manifest includes model
layout, communication mode, retasking, timings, discounting and packet assumptions.

## Global features in flattening order

| Index | Meaning | Normalization |
|---|---|---|
| 0 | Own storage fraction | Capacity |
| 1 | Own battery fraction | Capacity |
| 2–4 | Own wheel-speed fractions | Configured max speeds |
| 5–7 | Own Sun direction in Hill axes | Unit vector |
| 8–9 | Own eclipse timing (existing `obs.Eclipse`) | 5700 s |
| 10–13 | Next two ground-window open/close pairs | 5700 s |
| 14–18 | Own task mode one-hot: charge, downlink, desat, broadcast, image | 0/1 |
| 19 | Elapsed time since own task start | 12000 s |
| 20 | Remaining time to own task deadline | 12000 s |
| 21 | Own imaging hold or continuous transmit hold / required hold | Capped at 1 |
| 22 | Records not acknowledged by every configured peer | Catalog target count |
| 23 | Time since last successful exchange | 12000 s |
| 24 | A successful exchange has occurred | 0/1 |
| 25 | Current own task can continue | 0/1 |

At reset, mode/progress are zero. At a task ending, mode and elapsed time describe the
last own task while continue-valid is zero. Unknown exchange age is zero with knownness
zero. Backlog is zero for a single sensor and counts qualified record versions; it may
exceed one when multiple exposures per target accumulate. It is an own sender history
signal, not a remote catalog inspection.

## Per-candidate features

| Chunk offset | Meaning | Normalization |
|---|---|---|
| 0 | Target priority | 10 |
| 1–3 | Target minus sensor position, expressed in own Hill axes | 15960 km |
| 4–6 | Inertial target minus sensor velocity, expressed in own Hill axes | 12 km/s |
| 7 | Own pointing angle to target | 90 degrees |
| 8 | Own range to target | 15960 km |
| 9 | Target illumination/shadow factor | 0–1 |
| 10 | Age of newest known qualified acquisition for current request | 12000 s |
| 11 | Acquisition time is known | 0/1 |
| 12 | Age of freshest known *delivered capture* for current request | 12000 s |
| 13 | Delivered-capture time is known | 0/1 |
| 14 | Own pending product indicator | 0/1 |
| 15 | This is the own active imaging target | 0/1 |
| 16 | Valid candidate/action mask | 0/1 |

Time ages are `now - capture_time`, not time since packet receipt or last downlink.
Unknown ages are zero with separate knownness flags. Age values are normalized, not
clipped, to preserve information about long revisits. The velocity convention reuses
`_relative_velocity_H`; it is **not** the derivative of rotating-Hill-frame position.
Target geometry/illumination retains the original environment's exact simulation-state
assumption. Raw target IDs, private peer catalogs/resources and intentions are never actor inputs.
Directed mode additionally exposes the declared contact beacon features below. A navigation-estimation study would need a
separate target ephemeris/error model.

## Candidate identity and masking

`candidate_snapshot(sensor, K)` builds one timestamped tuple from sensor-local eligibility.
It retains the prior ascending-elevation shortlist within −21 to 90 degrees, filling
remaining slots by nearest distance and using target ID only as a deterministic tie-break.
This shortlist is an ordering heuristic, not a physical visibility guarantee. The existing
FSW access/pointing checks still govern capture success.

The observation and `ImageCompletion` decode that same tuple. The action builder does not
recompute a different ranking. Completed, cooldown-blocked, locally pending, or unresolved
image partitions are excluded. Missing slots are `None` with all-zero chunks; unavailable
targets never reappear as padding. Every environment boundary invalidates the next snapshot
after local updates and communication. Cached IDs remain unchanged between observation
and action execution. Invalid or stale imaging actions raise a clear error.

The pending field is normally zero because pending targets are excluded; it reserves an
explicit contract field for future candidate-list variants. Active-target relation can
also be absent when a target leaves the shortlist or its raw exposure is unresolved;
own mode, hold, and continue-valid remain observable. Candidate count is fixed for a run.

## Actions and learner masking

| Index | Action |
|---|---|
| 0 | Charge |
| 1 | Downlink owned physical image data to ground |
| 2 | Desaturate |
| 3 | Broadcast completed metadata |
| 4 | Deliberately continue the own unfinished task |
| 5…K+4 | Image the corresponding candidate slot |

Directed mode adds actions `K+5 ... K+4+P` for corresponding peer slots. Index 3 is
reserved and masked in directed mode. Independent and ideal-completion policies mask
all finite communication actions. Charge, downlink and desaturation remain available;
continue is valid only for an unfinished own task. These are eligibility masks, not a
complete safety shield.

Target and peer validity masks apply in **training, exploration and inference**. Padding
is zeroed before encoding and excluded from attention keys and mean/max pooling. An
all-empty set has a zero pooled representation; own resource context feeds operational
logits directly. The attention kernel gets a zero dummy key for empty rows, preventing
undefined all-masked softmax. Final invalid logits use the finite sentinel `-1e9`.
Metamorphic tests vary padding contents and permute both sets: target/peer scores follow
the corresponding permutation; operational logits and the critic remain invariant.

## Directed peer rows

Slots use the same timestamped peer snapshot for observation and action decoding, ordered
by configured peer name. A slot is eligible only in finite completion mode, while that
peer is alive and geometrically reachable, and when the sender has an unacknowledged
qualified completion delta for it. Otherwise the entire row is zero and its action masked.
No peer identity number is a neural feature.

| Offset | Meaning | Normalization |
|---|---|---|
| 0–2 | Beacon peer-minus-sender position in sender Hill axes | 15960 km |
| 3–5 | Beacon inertial velocity difference in sender Hill axes | 12 km/s |
| 6 | Sender boresight angle to peer | 180 degrees |
| 7 | Range to peer | 15960 km |
| 8 | Sender's unacknowledged record versions for this receiver | Catalog target count |
| 9 | Age of sender's last successful exchange with this peer | 12000 s |
| 10 | Such an exchange is known | 0/1 |
| 11 | Eligible peer/action | 0/1 |

**Declared discovery model:** a visible peer supplies an ideal current navigation beacon;
its SimpleNav message drives sender FSW while LOS is checked every control tick. Beacon
age and estimation error are zero in this model. This is a public contact-navigation
assumption, not permission to inspect arbitrary uncommunicated peer state. Discovery and
small ACKs have ideal negligible airtime/power; receivers use an omnidirectional antenna
and need not interrupt their own task. There is no receiver pointing or contention model.
The completion payload itself contains no navigation state.

The peer encoder/scorer shares weights over peers. Its masked pooled context conditions
the target-set network; the critic uses the same local information. Target scores use
shared target encoding and self-attention. This is parameter-sharing independent PPO,
not a centralized critic or a monolithic flattened-catalog MLP.

Selecting the same unfinished target or operational mode also preserves progress. Under
conflict retasking, a busy sensor's connector inserts `NO_ACTION` without policy choice;
that forced continuation is distinct from action 4 and is condensed before learning.
Under continuous retasking every boundary permits a policy decision, including action 4.

## Verification and interpretation

Regression tests cover schema shape across one/two/three sensors, actor masks in all
forward modes, candidate/action identity, all-targets-blocked padding, hold/deadline
preservation, no metadata transfer without the selected channel, conflict interruption,
finite radio timing, and real truncation observations. See [TEST_RESULTS.md](TEST_RESULTS.md)
for executed checks. Three information cases crossed with two retasking modes form the
matched study. Keep initial states, action bounds, heartbeat, quality threshold, cooldown,
reward weights, and per-second discount identical across cells. A deterministic heuristic
rollout validates plumbing; it does not establish learned collaboration performance.
