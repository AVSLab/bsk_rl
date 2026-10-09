# Prepared four-sensor learned-policy study

**Status: local implementation/offline validation only.** No new mission, PPO
optimizer update, cluster submission or new online W&B run has occurred. The
passed resume/critic gate 33505950 is historical and will never be resubmitted by
these scripts. Fixed-batch critic fitting is not policy convergence.

## Declared experiment

Both arms use a fresh parameter-sharing decentralized PPO policy, trained from
all four sensing agents' real decisions. This is not reuse of the trained AMOS
single-sensor checkpoint. Common architecture/seed initialization is identical;
action availability and received information differ explicitly.

| Setting | Both arms |
|---|---|
| Sensors | Walker-Delta T/P/F=4/2/1, 700 km, 97 degrees |
| Targets | 200 passive mixed targets: 100 LEO / 60 MEO / 40 GEO |
| Mission | 45,000 seconds; ten target candidate slots |
| Cooldown | One sensor-reference orbital period, anchored at physical capture |
| Actions/physics | Conflict retasking; physical storage and ground downlink; 300 s image/downlink/charge/transmit; 150 s desat; explicit continue |
| Reward | Owned acquisition/delivery reward; alpha=0.1; no team credit |
| PPO | `amos_best`, minimum batch 4200, 30 epochs, minibatch 128, cap 10, local CPU learner, one Torch thread; one runner in serial readiness, 28 in the separately gated parallel launch |
| Discount | gamma per second 0.9915045428565076; lambda per condensed decision 0.8713548569911232, explicitly approved AMOS objective |
| Architecture | Private actor/critic inputs, target-set attention, matching capacity; no memory or full-storage imaging shield |

`amos_best` preserves the declared AMOS learning-rate schedule, entropy, gradient
clipping and PPO clipping parameters in `optimization.py`. Its native Ray schedule
counter is preserved; it is not silently converted to the new true-decision axis.
The old `gae_trace_half_life_s` field is inactive when lambda-per-decision is set.
The reward discount half-life is approximately 81 seconds, not 45,000 seconds.

### Information arms

- `geometry_only`: ideal public position/velocity navigation beacons when direct
  Earth-clear LOS exists. No peer catalog, resources, action or current task can
  enter the actor/critic/masks. All transmit and broadcast logits are masked.
- `catalog_tx`: identical public navigation, plus only physically received
  `completed-onboard-v1` inventory. Owned completed low-quality and qualified
  products contain capture/request/quality; owner revisions and delivery/drop
  tombstones carry snapshot time. No image bytes, intent or resources are shared.

Navigation and contact discovery are explicit ideal LOS-only assumptions, with
zero beacon latency/cost. Peer receivers are omnidirectional and do not occupy a
receiver task. Small successful-packet ACKs retain the declared negligible-cost
assumption. The image inventory packet itself has a 64-byte header plus canonical
JSON UTF-8 bytes, 64,000 bit/s rate, continuous LOS/pointing hold >=10 s, 300 s
maximum action, delay=0 s, loss probability=0 and snapshot TTL=600 s. Payloads
freeze at action start. These defaults are declared physical-channel settings,
not a loss/latency study. Up to 256 entries fit one chunk; partial snapshots never
infer removal. Explicit tombstones are retained; authoritative complete absence
means unknown/absent, never an invented ground delivery. Dropping metadata does
not itself remove physical storage: the physical caller owns that operation.
Completion/cooldown history survives status expiry/removal. Receivers never own
another sensor's physical products or reward.

The versioned schema is `walker-inventory-navigation-v1`: 290 observation
features (26 own/context +10x21 target +3x18 peer), 18 logits. Peer observation
validity index 16 is separate from transmit eligibility index 17. Three stable
peer slots are sorted by configured sensor name. The old 100-target checkpoint
schema is incompatible; do not reshape it into this observation.

Only ready agents rebuild expensive observations. True final bootstrap observes
all live agents. Forced NO_ACTION, artificial bootstrap and padding are excluded
by the existing condensation/terminal learner connectors. Four real trajectories
map to one `imager` module; this does not convert event steps or learner
presentations into real decisions.

## Metrics and state

`study_metrics.py` is evaluation-only. Raw captures, full physical deliveries,
request epochs, tasks, receipt times, reward increments and sparse event-boundary
resources remain in saved episode records. Global truth is absent from policy
infos and from observation/mask implementations.

- Useful delivered visits are credited physical ground-delivery requests.
  Positive-credit product counts are separate (simultaneous products can share
  one visit). Metadata receipts/ACKs never count as ground downlinks.
- Coverage uses distinct target union /200. Per-sensor coverage is labeled
  separately. Legal post-cooldown revisits are not fresh duplicates.
- Fresh duplicate acquisitions count excess distinct sensor holdings of a
  qualified same-target/request product before delivery or capture+cooldown.
  Simultaneous completions count k-1; delivery ends before a same-time new start.
- Redundant execution unions imaging-task time covered by qualified peer
  completion for the same request. Clip at cooldown, task end and actual horizon.
  Unaware time additionally clips at first receipt. This is an observed
  opportunity-cost proxy, not proven counterfactual recoverable time.
- Fresh overlap uses target-seconds /target-sensor-seconds. Physical onboard
  overlap includes valid old revisits and is explicitly descriptive.
- Radio task occupancy is assigned transmit-task time. `radio_on_s` integrates
  actual powered intervals sampled by FSW, including interrupted/unfinished
  tasks; generic packet audit rows are not counted twice. Neither is packet count.
- Primary latency is capture-to-full-ground-delivery for positive-credit
  products; secondary latency is all qualified physically delivered products.
  Report actual delays, valid n and medians, plus right-censored count/ages and
  undelivered fraction. An empty latency population is null, not zero.
- Owned reward includes actual API penalties. Sum agent increments/returns to
  constellation return; agent actions are actual selections, not physical images.
- Failure live-time denominators use the first observed failure boundary.
  This is boundary-resolved death timing, not sub-tick death reconstruction.

Episode products, inventory, packet/ACK state, rewards, actions and trajectories
reset. Base seed/next episode index/Torch RNG, model/optimizer, lifetime dedup
IDs, completed-episode metric aggregation and pending callback evidence persist
in checkpoints. Replay conflicts fail. Lifetime pooled medians use actual product
delays; rates aggregate numerators/denominators. Driver wall/CPU cost survives
resume as a separate driver-only account. Full allocation cost (including tests,
preflight, restore and frozen evaluation) comes from job-exit and final sacct;
do not call driver cost full-job billing.

## W&B and plots

Only the driver initializes/publishes W&B. Separate fresh IDs per arm; all
continuations and zero-update restores use saved identity with `resume=must`.
Primary x-axis is cumulative **true decisions**. Iteration, episode and lifetime
driver wall-time axes are declared separately. Thirty-second heartbeats publish
status and current-mission worker-spool rewards/actions/storage; these reset with
the episode and never increment completed lifetime experience. Raw episode
telemetry retains simulation time and identity. Failed horizon evidence fails
readiness; no silent partial-success summary.

`wandb-dashboard-study.json` defines panels/units/valid counts, not an already
created live dashboard. `study_plotting.py` generates three independent PNG/PDF
scatter figures from compatible saved 200-target/one-orbit records, retaining
seed/sensor rows and input hashes. It rejects mixed argmax/stochastic modes or
historical configuration. **Plasma = catalog transmit; viridis = geometry only**,
four shades/markers; solid/dashed lines for arm identity. Templates have no fake
points and explicitly say no missions have been sampled. Historical
100-target/two-orbit figures remain in their original directories.

## Prepared compute work (approval required)

`study_readiness.slurm`: one 8-CPU/32-GiB/4-hour allocation, account ucb550_asc2,
acpu/cpu-normal. Effective <=9 CPUs, hard reservation cap36 allocated CPU-hours.
At most one PPO update per arm, <=6 sampled training missions and <=30,000 true
decisions per arm; 16 total frozen held-out missions: initial/final x401/402 x
argmax/stochastic x2 arms. No seed sweep or second automatic job. It gates source,
installed/loaded runtime, native/support hashes, online W&B, actual worker pins,
finite learning, physical product metrics, all horizons, same initial weights,
matched seed hashes and full-topology zero-update restore/byte preservation.
Record all training seeds: the common initial seed intersection must match;
training episode counts may differ because trajectories/episode lengths differ.
All held-out initial states must match exactly within/across arms and modes.

The local runtime is Apple/Python3.11.9/Basilisk2.13; intended Alpine is
Python3.11.13/Basilisk2.12.0b0 at8fcb54b2fb28388efb711786630501944fddec28,
Ray2.35, Torch2.4.1+cpu, NumPy1.26.4, W&B0.26.1, effective externalpsutil6.1.
Retain shared packages/extras/GCC14.2 support paths. New-source compute behavior
has not been tested. Initialization checks spaces without a hidden mission step;
a real initial observation reset remains explicitly part of the later compute
initial-policy gate, separate from sampled training/held-out mission counts.

## Parallel resource revision (6 October 2026)

The current long-launch preparation requests **32 CPUs / 100 GiB RAM / 24 hours
per arm**, with **28 remote simulation runners**, a local CPU learner and one
Torch thread per process. Each environment still contains four spacecraft;
28 runners mean 28 independent constellation missions sampled in parallel.
All real decisions from all spacecraft train the same shared `imager` policy.

See `PARALLEL_RESOURCE_PLAN.md` for the source-backed AMOS comparison, new-source
readiness requirement, total cost, experience-budget derivation and approval
boundary. The immutable v4 readiness currently submitted as job33510037 remains
8 requested CPUs /32GiB /4hours; its source was not modified.

Both parallel long scripts reject timing from a one-runner or different-source
readiness result. Common exposure targets are rounded down to50k true-decision
milestones from65% of the slower measured arm's throughput and a common safe
collection cutoff. Hard decision caps reserve two measured batches of overrun;
mission/update caps are recorded in budget.json. This is an estimate, not a
promise of future throughput, exactly matched exposure or convergence.

The parallel caps are invocation-wide and partitioned deterministically across
workers. A reset without a policy action consumes no sampled mission. Exhaustion
fails before another physical action; partial failure remains evidence.
Lifetime metrics and worker seed/RNG streams retain their checkpoint semantics.
A fresh continuation invocation receives a new explicit sampling budget while
restoring the same optimizer, policy and W&B identity. Continuation requires
review; no successor is submitted automatically.

The two24-hour reservations total1536 allocatedCPU-hours. Another24-hour pair
would reserve another1536 and requires separate approval. Both arms retain the
approved mission, private observation/model/action schema, owned reward,
AMOS gamma/lambda and PPO optimizer; resources and evidence collection are the
changes declared here. A full-topology restore cannot silently change28 workers
to one or vice versa.

Human labels are **With communication** (plasma) and **Without communication**
(viridis). Plot spacecraft1..4 map to internal sensor_0..3. No figure containing
new200-target data exists until actual episode records are loaded.
