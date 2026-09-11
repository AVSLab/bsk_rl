# Three-sensor independent and centralized baselines

This campaign completed **200 deterministic heuristic episodes**, not policy training:
three sensors, 100 passive RSO spacecraft, ten candidates, 45,000-second horizons,
and conflict retasking. Seeds 0–49 repeat in each information/environment cell.
The full paired results and plots are in
[`cluster/evidence/three_sensor_v2_full_campaign`](cluster/evidence/three_sensor_v2_full_campaign/REPORT.md).

| Array IDs | Information/controller | Target catalog | Seeds |
|---|---|---|---|
| 0–49 | Independent local greedy | LEO | 0–49 |
| 50–99 | Independent local greedy | Mixed | 0–49 |
| 100–149 | Centralized full-state joint greedy | LEO | 0–49 |
| 150–199 | Centralized full-state joint greedy | Mixed | 0–49 |

The baseline configuration is `configs/baseline_mc.json`. It is a separate schema
from completion-v2 policy observations and checkpoints. The learned-policy task,
its directed transmission, and its existing configurations are unchanged.

## Information and control assumptions

**Independent:** each sensor applies the same deterministic rule using its own
resource state, own physical image products, own completion history, and declared
target ephemerides/illumination. It cannot inspect any peer's catalog, action,
resource state, or choices. A common target catalog and ephemerides are mission
inputs, not peer communication. Independent sensors can duplicate each other's
work. Their *union* coverage must be measured; three independent sensors do not
mathematically guarantee 100% coverage in a finite horizon.

**Centralized full state:** one joint controller is invoked at every asynchronous
environment decision boundary. At that exact simulated time it reads every live
sensor's position, velocity, attitude, body rate, battery, storage, wheel state,
active task, target reservation, physical onboard products and owners, current
request epochs, and durable capture/completion/delivery catalog. It applies the
same local resource constraints, then jointly assigns currently available target
slots, avoiding another assigned, onboard-fresh, or in-progress target.
The existing ideal-completion mechanism supplies immediate completion facts to
local eligibility and conflict-retasking logic; the **additional full-state joint
controller** uses current assignments/resources directly. Therefore this case is
stronger than an `ideal_completion` shared decentralized policy. It has no radio
action, network latency, bandwidth cost, or peer-state observation restrictions.
Intent knowledge is isolated to this explicitly omniscient baseline and is not
added to completion-v2 policy observations or transmitted payloads. Every episode
saves a compact centralized-information audit with the number of boundaries and
sensor-state reads, peak visible catalog/product counts, and a final snapshot hash.

All controllers prioritize a target they have never qualifiedly captured, then
minimize current pointing angle, then prefer higher priority. The centralized
controller enumerates current three-sensor combinations, maximizing new-target jobs,
then total image jobs, then minimizing total angle. It does not solve a future
trajectory optimization. Centralized knowledge is an information advantage; this
particular greedy policy is **not a guaranteed optimum or upper performance bound**.

All use the same ten-slot existing candidate shortlist, actual Earth-clear LOS,
and current target illumination above the existing reward threshold. Priority is
normalized to a catalog total of 100 using the existing scenario. All charge below
30% battery, desaturate above 70% wheel limit, and downlink stored products during
current ground contact or above 80% storage. They charge if no usable candidate
remains. Physical image holding, ground delivery, charging, and wheel dynamics
remain Basilisk/BSK-RL operations. No broadcast or directed-transmit action is ever
selected. One orbit can contain multiple event decisions; busy tasks continue.

## Existing cooldown is preserved

Per the user's final instruction to use the implementation already present, this
campaign **does not introduce a new one-orbit, fixed-5700-second, or
delivery-anchored cooldown**. The existing full mission configuration has
`reimage_cooldown_orbits=2.0`. `MultiSensorRSOTargetImageReward.reset_post_sim_init`
derives `cooldown_s = 2 × median(initial sensing-spacecraft orbital periods)`.
For the three-sensor 700/800/700-km orbit pattern this reset derives
**11,834.835756586714 seconds**; every episode records the value derived from its
initialized sensing orbits.
It is the same in LEO-only and mixed catalogs, independent of each target's orbit.
Every episode records the actual derived `reimage_cooldown_s`.

A qualified acquisition suppresses revisits until capture time plus that cooldown.
An own pending physical product additionally blocks that sensor's same-target
storage partition until its full ground delivery. Receiving metadata does not move
image bits or confer ownership. In the independent case a peer acquisition is
unknown; in the centralized case immediate shared facts suppress all sensors for
the same cooldown. First-coverage metrics remain separate from revisit service
counts, which can exceed 100 over a long episode.

“Ground confirmation” is kept as an evaluation boundary: ground coverage changes
only after a complete physical downlink. It is not the start of the existing
revisit cooldown. Moving the cooldown anchor from capture to delivery would change
the task-generation semantics and must be a separately versioned sensitivity case.

## LEO and mixed orbital populations

The target distributions reproduce `examples/updated_policy_evaluation.py` in the
AMOS worktree (sampler at lines 1744–1829):

| Regime | Altitude above 6371-km Earth radius | Eccentricity | Inclination |
|---|---|---|---|
| LEO | Uniform 400–2000 km | Uniform 0–0.02 | Uniform 0–180° |
| MEO | Uniform 2000–35000 km | Uniform 0–0.10 | Uniform 0–120° |
| GEO | Uniform 35,786 ±300 km | Uniform 0–0.0015 | Uniform 0–15° |

RAAN, argument of periapsis, and true anomaly are uniform 0–360°. Eccentricity
is resampled if needed to keep perigee at least 400 km. The mixed evaluation uses
the AMOS exact-count option: **50 LEO, 30 MEO, 20 GEO**, with regime-to-target-ID
assignment shuffled deterministically by seed. It does not randomize mixture
weights, unlike a separate AMOS training variant. The three sensors retain the
branch's existing staggered orbit generator: sensor 0 is 700 km/97°, sensor 1 is
800 km/70°, and sensor 2 is 700 km/97°, with distinct RAAN and anomaly offsets.
All 100 RSOs are live propagated Basilisk/Vizard spacecraft; only the three sensors
are PettingZoo agents. There is no old target-killing speed shortcut.

For each environment and seed, independent and centralized cases must reproduce
identical initial positions, velocities, sensor attitude/resources, regime labels,
and target priorities. The aggregator rejects mismatches. LEO and mixed targets
intentionally have different orbital states even when seed numbers match.

## Metrics and statistical interpretation

Each episode saves distinct qualified **capture coverage** and distinct fully
**ground-delivered coverage**, each divided by the entire 100-target catalog.
It also saves per-sensor and per-regime coverage, overlapping capture target IDs,
qualified/unqualified exposures, repeated services, reward, duplicate attempts,
duplicate and nonduplicate-interrupted sensor-seconds, wasted-time fraction,
actual task durations/decision counts, zero radio occupancy, time-tagged exposure
and delivery records, catalog receipt versions, and physical onboard ownership.

Two additional duplicate families implement the requested definitions:

1. **Stale cross-sensor ground delivery:** a qualified delivered product counts
   when another sensor also delivered a newer capture of the same target. The
   report also gives the causal subset for which the newer product had already
   reached ground before the stale one arrived.
2. **Cross-sensor onboard overlap:** qualified physical products are represented
   by half-open storage intervals `[capture_time, delivery_time)`, or through the
   episode end when still onboard. The report counts affected targets and products,
   redundant acquisitions, excess sensor-seconds `integral max(0, holders-1) dt`,
   and all sensor-seconds during multi-sensor overlap.

These do not replace the existing cooldown-relative duplicate-attempt and wasted
task-time metrics. They answer different questions: data freshness at ground and
catalog-coverage opportunity cost while products remain in storage.
Event-boundary battery/storage/wheel histories, sensor survival, wall time,
peak process RSS, and simulated-seconds/wall-second are recorded. Geometric and
candidate occurrence counts help explain omitted targets, but are sampled at event
boundaries and are not a continuous visibility feasibility certificate.

Aggregation validates all 200 expected IDs and source/config manifests, rejects
duplicate/mismatched runs, keeps early-death episodes in statistics, writes episode
and paired-difference CSVs, and computes 95% bootstrap intervals over initial-state
seeds. Central-minus-independent differences are paired by environment/seed. The
environment-specific intervals resample 50 paired seeds. The pooled analysis has 100
LEO/mixed effects but resamples 50 seed blocks, keeping both environments together
when a reused seed is drawn. This avoids treating the same seed ID as two independent
initializations.
`--allow-partial` exists only for diagnostic partial reports and labels incompleteness.
Plots show all seed outcomes, means, paired effects, productive cooldown-qualified
services, positive reward components, and each duplicate-work family. The baseline
reward decomposes exactly into 90% acquisition value and 10% ground-delivered value;
all saved penalty/adjustment terms were zero. No 95%→100% improvement is assumed.
These results concern deterministic heuristics, not learned-policy convergence or a
direct controlled comparison to a previous single-agent 95% result.

## Cluster preparation and commands

Live cluster work verified account `ucb550_asc2`, partition `acpu`, and QOS
`cpu-normal` (24-hour maximum); the old `amilan` partition is obsolete. The separate
completion-v2 checkout, Python 3.11 environment, and Basilisk source/runtime now
exist at the paths below and passed the recorded runtime build/audit. The existing
`/projects/dahu1128/bsk_rl` AMOS checkout and its old environment remain separate.
Pull the reviewed three-sensor commit and regenerate a new v2 manifest before any
new baseline run. The already completed two-sensor tasks 0 and 100 belong to the v1
manifest and cannot be mixed into this campaign. The array requires no Ray or GPU:
each task is one CPU and one Basilisk simulation. The template requests
**4 GiB and one hour per episode**, at most eight concurrent episodes. These are
conservative given the measured two-sensor runs (about five minutes and 1.97 GB
per episode); validate the first three-sensor pair before expanding. The existing
learned-policy preflight uses a different eight-CPU/32-GiB allocation.

Generate the manifest **after the final source is deployed**, from its Git checkout.
It hashes tracked/untracked executable/configuration inputs, exact configuration,
versions and all 200 task mappings; a task rejects subsequent input changes.
Documentation-only edits do not invalidate the campaign; each result separately
records its full Git HEAD. Put outputs in
`results/`, not inside source folders. The project and Python paths below are deployment destinations, not an assertion that they already exist:

```bash
export BSK_PROJECT_ROOT=/projects/dahu1128/bsk_rl-multi-agent-space-imaging-2026
export BSK_RL_PYTHON=/projects/dahu1128/.venv-completion-v2/bin/python
export BASILISK_SOURCE_ROOT=/projects/dahu1128/basilisk-completion-v2
cd "$BSK_PROJECT_ROOT"
export PYTHONPATH="$BSK_PROJECT_ROOT/src:$BSK_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BSK_MC_ROOT="$BSK_PROJECT_ROOT/results/multiagent_imaging/baseline-mc-3sensor-v2"
export BSK_MC_MANIFEST="$BSK_MC_ROOT/manifest.json"
export BSK_MC_OUTPUT="$BSK_MC_ROOT/episodes"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.baseline_monte_carlo manifest \
  --config examples/multiagent_imaging/configs/baseline_mc.json \
  --output "$BSK_MC_MANIFEST"
```

After explicit authorization, submit a first LEO information pair before the full
campaign, with the account/partition from the actual cluster allocation:

```bash
sbatch --export=ALL --account=ucb550_asc2 --partition=acpu --qos=cpu-normal \
  --array=0,100%2 examples/multiagent_imaging/cluster/baseline_mc.slurm
```

Once these have completed and been checked, submit the **remaining 198 IDs** so
the campaign still has exactly 200 episodes:

```bash
sbatch --export=ALL --account=ucb550_asc2 --partition=acpu --qos=cpu-normal \
  --array=1-99,101-199%8 examples/multiagent_imaging/cluster/baseline_mc.slurm
```

Alternatively, after equivalent preflight and authorization, submit the template's
complete `0-199%8` array once. Existing episode files are never overwritten. A failed
episode with no complete atomic JSON may be retried by its exact array ID.

Aggregate after the full campaign:

```bash
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.aggregate_baseline_monte_carlo \
  --manifest "$BSK_MC_MANIFEST" --episodes-dir "$BSK_MC_OUTPUT" \
  --output-dir "$BSK_MC_ROOT/report" --plots
```

Every array task checks for a Slurm job and executes `cluster/audit_runtime.py`
before simulation, saving its own runtime report. The baseline audit does not use
`--require-allocation`, which specifically checks the separate eight-CPU/32-GiB
learning allocation. Baseline tasks request one CPU and 4 GiB each.

Source files: `baseline_monte_carlo.py` contains the scenario, controller and runner;
`aggregate_baseline_monte_carlo.py` validates pairing and creates tables/reports;
`cluster/baseline_mc.slurm` maps array tasks to episodes. Code comments explain the
information boundary and physical versus catalog ownership.

## Validation evidence (2026-09-11)

The complete cluster campaign has **200/200 episodes**, four cells of 50 seeds, and
100 validated matched information pairs. Every run reached 45,000 seconds with all
three sensors alive, exactly 100 passive spacecraft, and zero radio activity. The
centralized episodes recorded one full-team snapshot per decision boundary and three
sensor-state reads per snapshot; independent episodes recorded no centralized reads.
All paired initial-condition hashes match. The derived cooldown differs from
11,834.835756586714 seconds only by floating-point roundoff.

The strict local rerun and **18 focused tests** pass. The tests cover campaign mapping,
matched initialization, joint assignment, coverage and duplicate definitions,
reward-component separation, paired/seed-blocked statistics, centralized access, and
four short no-radio episodes. Ruff passes; test output contains only existing Basilisk
deprecation warnings.

Centralized coordination averaged 27.85 more cooldown-qualified acquisitions and
29.88 more ground-confirmed unique services than independent control in the pooled
paired comparison. It removed 429.50 duplicate attempts and 406.93 successful
duplicate deliveries per episode. Coverage rose by 0.25 capture percentage points;
most residual missed targets had no event-boundary illuminated LOS sample. See the
full evidence report for intervals, metric definitions, resource use, and limitations.
