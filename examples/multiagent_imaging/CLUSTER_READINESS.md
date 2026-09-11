# Completion-v2 training and cluster-readiness evidence

**September 11 status:** the separate Alpine runtime, one-worker two-sensor PPO
restore/resume gate, and the complete 200-episode three-sensor independent versus
centralized baseline campaign have run successfully. See
[`cluster/evidence/three_sensor_v2_full_campaign`](cluster/evidence/three_sensor_v2_full_campaign/REPORT.md).
The four-worker learned pilot and broad six-cell learned-policy study have not run.
The next policy gate should use three sensors after versioning the extra peer/action
slot and adding the baseline's exact LEO/mixed target-environment selection.

**September 6 historical preparation:** the live Alpine audit found a different AMOS checkout
and an incompatible Python/Basilisk environment. Prepared deployment, runtime build,
one-worker validation, bounded learning, and the 200-episode baseline array are in
[cluster/README.md](cluster/README.md) and [BASELINE_MONTE_CARLO.md](BASELINE_MONTE_CARLO.md).
They had not yet been submitted at that point. The local evidence below remains the
pre-deployment record.

The local preflight passed on `multi-agent-space-imaging-2026`. Directed transmission,
target-set attention, complete-episode PPO, checkpoint evaluation and training resume
execute against real Basilisk spacecraft. At the time of this local record, no cluster
job or six-cell statistical study had launched. The later cluster work completed the
runtime, one-worker gate, and deterministic baseline campaign; it still does not show
learned-policy convergence or superiority of either retasking rule.

## Formulation and implementation

The production preflight configurations use **two sensing agents, 100 passive RSO
spacecraft, ten candidates and 45,000-second episodes**. Two is the sensor count in the
existing full configuration. All 102 spacecraft propagate in Basilisk; only two enter
PettingZoo/RLlib. Passive targets remain spacecraft in Vizard, not markers or policy agents.

The shared policy encodes candidate targets with shared small networks and self-attention.
Padding is excluded inside attention and pooling, not merely at the action logits. Own
resource context remains available when every target is blocked. A shared peer encoder
adds receiver scores and pooled contact context. The critic uses the same local information.
For this configuration, the contract is **208 observation values and 16 discrete actions**.
Candidate and peer permutations move the corresponding logits while preserving operational
logits and value predictions. The contract is versioned `completion-v2`.

`TransmitCompletions` selects an observed peer slot. Existing Basilisk `locationPointing`
tracks that peer's SimpleNav message through `ImagingSCFSWModel.action_point_peer`. The
instrument and ground data transmitter remain disabled. The sender must satisfy Earth
LOS, MRP attitude error ≤0.0025 and angular rate error ≤0.01 rad/s at each FSW tick. After
slew, transmission requires a continuous **10-second** valid hold, extended when needed
to fit the frozen payload at **64 kbit/s**. A 64-byte transport header is included. The
whole attempt has a **300-second deadline**. Loss of lock resets packet progress. Valid
transmission draws **25 W** from the existing power sink without draining image data.

Only the selected receiver receives the completion packet after hold, delay/loss/TTL
processing. The FSW controls pointing; the communication layer merges catalog metadata
at the next simulation boundary. The original 30-second drift/broadcast remains selectable
with `communication_mode="broadcast"`. The mission configurations select `"directed"`.

Peer knowledge is explicit: ideal current navigation beacons from discovered LOS contacts,
plus sender-local exchange/ACK history. No private peer catalog, intent or resource vector
is exposed. ACKs and discovery are assumed to have negligible airtime/power. Receivers have
omnidirectional antennas and can receive while executing their own task.

Catalog facts preserve source, exposure ID, target and request generation. Capture,
completion, full ground delivery and receiver receipt are separate timestamps. Each
receiver keeps the first receipt of each record revision as well as its first knowledge
of that exposure. Merging is by exposure identity with monotone delivery upgrades; an
old packet cannot overwrite newer acquisition knowledge or erase a ground delivery.
Actor summaries use the newest relevant fact while durable exposure history is retained.
Receiving completion metadata never transfers or deletes physical image products.

The reward discount half-life is **45,000 seconds**:
`gamma_s = 0.9999845968479506`. A reward one nominal 6000-second orbit away retains about
0.912 weight, and one at the horizon retains 0.5. The former `gamma_s=0.999` retained only
about 0.00247 after one orbit. GAE trace half-life is **6000 seconds**, also applied using
actual elapsed seconds. Complete-episode sampling and existing action condensation keep
forced continuation separate from policy decisions and avoid double-discounting rewards.

See [ARCHITECTURE.md](ARCHITECTURE.md) and
[OBSERVATION_VALIDATION.md](OBSERVATION_VALIDATION.md) for code responsibilities and every
observation field/action index. The code includes explanatory comments at the navigation,
hold, catalog merge, padding, timing and checkpoint handoffs.

## Measured local evidence

Runtime: macOS ARM, Python 3.11.9, Ray 2.35.0, Torch 2.4.1, NumPy 1.26.4. Basilisk is a
clean source build at `8fcb54b2fb28388efb711786630501944fddec28` (its package reports
`0.0.0`, so that package string alone is insufficient for reproduction).

| Run | Complete simulated seconds | Wall seconds | Peak RSS, MiB | Simulated seconds / wall second |
|---|---:|---:|---:|---:|
| Full deterministic heuristic profile | 45,000 | 238.53 | 507.64 | 188.66 |
| Conflict, one PPO update | 45,000 | 248.69 | 1148.11 | 180.95 |
| Continuous, one PPO update | 45,000 | 221.79 | 1349.63 | 202.89 |

PPO wall time above measures sampling and optimization, excluding roughly three seconds
of algorithm setup and checkpoint writing. RSS is sampled every 0.2 seconds for the
process and its children; shared pages may be counted more than once, and short peaks can
be missed. It is not an exact Slurm reservation requirement.

| PPO validation | Conflict | Continuous |
|---|---:|---:|
| Requested minimum environment batch | 64 | 64 |
| Actual environment steps | 480 | 342 |
| Actual agent transition rows before condensation | 960 | 684 |
| Actual policy decisions | 485 | 684 |
| Complete episodes | 1 | 1 |
| Total loss | 8.12622 | 9.38409 |
| Policy loss | −0.39068 | 0.22864 |
| Value loss | 8.51689 | 9.15545 |
| Last measured gradient L2 norm | 0.41186 | 0.87174 |
| Parameter-change L2 norm | 0.03749 | 0.03710 |
| Changed parameter elements | 70,642 | 69,687 |
| Restored-logit maximum error | 0 | 0 |

Every gradient computation checks finiteness; the norm reported is the final measured
minibatch norm. Each sensor's physical task durations sum to exactly 45,000 seconds in
both runs. The different actual batch sizes are expected with complete episodes and
asynchronous decision sets. They must be recorded when comparing training budgets.

Coverage from the full heuristic profile: 137 unique acquisition services, 130 unique
ground services, 14 successful duplicate ground services, and 67 accepted directed
packets. The counters include revisit services, not just distinct target identities.
The two PPO sampling episodes also exercised capture, ground delivery and communication:
132/150 acquisition services, 80/115 ground services, and 31/24 completed directed holds
for conflict/continuous respectively. End-of-episode storage reached 82%/82% in conflict
and 94%/10% in continuous. Batteries and wheels remained finite and operational.
These are coverage observations from the initial policies, **not matched learned-policy
performance comparisons**. The heuristic and PPO also use different seed streams.

Checkpoint validation loads actual exported weights and compares logits and actions on
saved observations. A separate integration test evaluates a restored policy through a
real held-out simulator episode and counts its calls; no heuristic fallback is possible.
It restores Adam moments/counters, training iteration and episode seed position, then runs
another update and verifies further finite parameter changes. The full mission checkpoint
also passes optimizer/logit restoration. Strict schema checks reject changed candidate
count, communication mode, retasking, timing, discounting or old observation versions.

The local two-rollout-worker preflight collected four complete 360-second episodes with
three sensors. Worker indices 1 and 2 used seeds 3262728894 and 1402144644 respectively;
both advanced their own episode sequence. It produced finite losses/gradients, changed
weights and exact restored logits. Peak sampled RSS was about 1820 MiB. This validates
local Ray process separation; it does not validate Linux/Slurm or GPU execution.

Final regressions: **190 tests passed**: 85 multi-agent unit/integration checks, plus 105
core action, observation, environment and simulator checks. The core simulator tests run
in a separate process because of pre-existing monkeypatch isolation behavior. Directed
recipient selection is tested with three real sensors, both fixed hold and payload-driven
airtime. Other checks cover interrupted lock, out-of-order receipts, partial/full downlink,
revisit eligibility, actual parameter updates, seed streams and strict checkpoint resume.
Ruff passes for the changed completion/readiness modules; the Slurm script passes `bash -n`.

## Artifacts and reproduction

Artifacts are under `results/multiagent_imaging/cluster_readiness/`:

- `profile/profile_episode.json`: full physical profile and event/resource histories.
- `conflict/updates.json`, `continuous/updates.json`: batch, loss, gradient, parameter,
  resource, timing, communication and complete-episode evidence.
- Each `checkpoint_0001/`: full Ray algorithm checkpoint, `policy.pt`, semantic manifest
  and restored-output evidence. Continuous additionally has `full_resume_validation.json`.
- `worker_preflight/`: two-worker launch configuration, seed states and update evidence.
- `summary.json`: compact full-scale results.
- `diagnostics/`: resource and physical-discount figures in PNG/PDF. These are validation
  figures, not six-cell reward rankings.
- `final_provenance.json`, `pip-freeze.txt`, `source-snapshot.tar.gz`: final source/dependency
  records. Individual run manifests retain the source hashes present for each run.

Some metadata audit/export refinements were added after the initial profile/conflict run;
their per-run source hashes are intentionally retained. They do not alter the two-sensor
policy contract or learned outputs. Final tests validate the finished source. Result files
are git-ignored; implementation/configuration/docs/tests remain reviewable in the worktree.

## Recommended production configuration and limits

Keep the mission configurations above, one Torch/BLAS thread per rollout process, complete
episodes, shared target-set attention, receiver-local eligibility and completion-only
payloads. Begin on **one node, eight CPUs, 32 GiB RAM, four rollout workers and a CPU
learner**. The small network is unlikely to benefit from a GPU before simulator throughput
is measured on the cluster. This is an initial resource reservation, not a benchmarked
Linux requirement. The prepared script is `cluster/train.slurm` with pinned dependencies
and build notes in `cluster/README.md`.

The measured local cost is about four minutes per full episode/update on one runner.
Four workers could sample four episodes in roughly that wall time if per-core performance
and scaling were similar, but contention and Linux hardware can change this substantially.
Use 1800-second rollout timeouts rather than RLlib's 60-second default. A one-hour initial
cluster validation is a reasonable bounded budget; the supplied script's four-hour limit
is configurable. Measure actual batch sizes again when increasing workers.

Remaining limits:

- One PPO update proves functioning optimization, not convergence, policy quality or a
  preferred retasking mode. Production learning-rate, batch and value-loss settings need
  a bounded learning-curve pilot before the six-cell study.
- Sender pointing/power is physical, but receiver antenna, beacon/ACK cost, RF link budget,
  contention, propagation light time and noisy/stale navigation are idealized explicitly.
  There is no packet fragmentation for deltas larger than an attempt can transmit.
- Broadcast and directed baselines have different physical radio assumptions; report
  that distinction when comparing them.
- Resource dynamics are active, but eligibility masks are not a full safety shield.
  High storage occupancy in the first PPO episodes warrants monitoring.
- Full episode histories grow with mission activity. Memory extrapolation to many
  workers, longer missions or larger constellations needs measurement.
- Resume is at complete-episode boundaries with the same worker layout/runtime. Changing
  worker counts or devices is not promised to reproduce the same stochastic trajectory.
- The Slurm script has been prepared and syntax checked. Cluster software installation,
  scheduler behavior, Linux execution and GPU/distributed learners remain untested.

The next task is a bounded cluster-environment validation and learning pilot after the
cluster project path, environment and allocation are known. Use [NEXT_PROMPT.md](NEXT_PROMPT.md).
