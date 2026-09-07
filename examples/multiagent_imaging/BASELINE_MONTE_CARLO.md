# Two-sensor independent and centralized baselines

This campaign prepares **200 deterministic heuristic episodes**, not policy training:
two sensors, 100 passive RSO spacecraft, ten candidates, 45,000-second horizons,
and conflict retasking. Seeds 0–49 repeat in each information/environment cell.
No job is submitted by these preparation commands.

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
target ephemerides/illumination. It cannot inspect its teammate's catalog, action,
resource state, or choices. A common target catalog and ephemerides are mission
inputs, not peer communication. Independent sensors can duplicate each other's
work. Their *union* coverage must be measured; two independent sensors do not
mathematically guarantee 100% coverage in a finite horizon.

**Centralized full state:** one joint controller has instantaneous access to both
sensors' current physical state, resources, physical ownership, catalogs, and
ongoing tasks. It applies the same local resource constraints, then jointly assigns
currently available target slots, avoiding another assigned or in-progress target.
The existing ideal-completion mechanism supplies immediate completion facts to
local eligibility and conflict-retasking logic; the **additional full-state joint
controller** uses current assignments/resources directly. Therefore this case is
stronger than an `ideal_completion` shared decentralized policy. It has no radio
action, network latency, bandwidth cost, or peer-state observation restrictions.
Intent knowledge is isolated to this explicitly omniscient baseline and is not
added to completion-v2 policy observations or transmitted payloads.

Both controllers prioritize a target they have never qualifiedly captured, then
minimize current pointing angle, then prefer higher priority. The centralized
controller enumerates current two-sensor combinations, maximizing new-target jobs,
then total image jobs, then minimizing total angle. It does not solve a future
trajectory optimization. Centralized knowledge is an information advantage; this
particular greedy policy is **not a guaranteed optimum or upper performance bound**.

Both use the same ten-slot existing candidate shortlist, actual Earth-clear LOS,
and current target illumination above the existing reward threshold. Priority is
normalized to a catalog total of 100 using the existing scenario. Both charge below
30% battery, desaturate above 70% wheel limit, and downlink stored products during
current ground contact or above 80% storage. They charge if no usable candidate
remains. Physical image holding, ground delivery, charging, and wheel dynamics
remain Basilisk/BSK-RL operations. No broadcast or directed-transmit action is ever
selected. One orbit can contain multiple event decisions; busy tasks continue.

## Existing cooldown is preserved

Per the user's final instruction to use the implementation already present, this
campaign **does not introduce a new one-orbit, fixed-5700-second, or ground-only
cooldown**. The existing full mission configuration has
`reimage_cooldown_orbits=2.0`. `MultiSensorRSOTargetImageReward.reset_post_sim_init`
derives `cooldown_s = 2 × median(initial sensing-spacecraft orbital periods)`.
For the existing 700/800-km two-sensor orbits this is **11,960.807123947807 seconds**.
It is the same in LEO-only and mixed catalogs, independent of each target's orbit.
Every episode records the actual derived `reimage_cooldown_s`.

A qualified acquisition suppresses revisits until capture time plus that cooldown.
An own pending physical product additionally blocks that sensor's same-target
storage partition until its full ground delivery. Receiving metadata does not move
image bits or confer ownership. In the independent case a peer acquisition is
unknown; in the centralized case immediate shared facts suppress both sensors for
the same cooldown. First-coverage metrics remain separate from revisit service
counts, which can exceed 100 over a long episode.

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
weights, unlike a separate AMOS training variant. Both sensors retain the full
multi-agent branch's existing staggered LEO orbits (700/800 km, 97°/70° inclination).
All 100 RSOs are live propagated Basilisk/Vizard spacecraft; only the two sensors
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
Event-boundary battery/storage/wheel histories, sensor survival, wall time,
peak process RSS, and simulated-seconds/wall-second are recorded. Geometric and
candidate occurrence counts help explain omitted targets, but are sampled at event
boundaries and are not a continuous visibility feasibility certificate.

Aggregation validates all 200 expected IDs and source/config manifests, rejects
duplicate/mismatched runs, keeps early-death episodes in statistics, writes episode
and paired-difference CSVs, and computes 95% bootstrap intervals over initial-state
seeds. Central-minus-independent differences are paired by environment/seed.
`--allow-partial` exists only for diagnostic partial reports and labels incompleteness.
Optional coverage plots show all seed outcomes and means. No 95%→100% improvement
is assumed. These results concern two deterministic heuristics, not learned-policy
convergence or a direct controlled comparison to a previous single-agent 95% result.

## Cluster preparation and commands

Live cluster inspection verified account `ucb550_asc2`, partition `acpu`, and QOS
`cpu-normal` (24-hour maximum); the old `amilan` partition is obsolete. The requested
`/projects/dahu1128/bsk_rl-multi-agent-space-imaging-2026` directory was absent at
inspection. The existing `/projects/dahu1128/bsk_rl` is the AMOS checkout and its
`/projects/dahu1128/.venv/bin/python` environment does not meet the pinned readiness
dependencies. Deploy the supplied snapshot and create/validate the separate Python
3.11 environment using the parent cluster preparation instructions first. Do not
run this campaign in the old checkout/environment. The array requires no Ray or
GPU: each task is one CPU and one Basilisk simulation. The template requests
**4 GiB and one hour per episode**, at most eight concurrent episodes. These are
conservative starting allocations, to revise from cluster preflight measurements;
they do not claim measured cluster performance. The existing learned-policy
preflight uses a different eight-CPU/32-GiB allocation.

Generate the manifest **after the final source is deployed**, from its Git checkout.
It hashes tracked/untracked relevant sources, exact configuration, versions and
all 200 task mappings; a task rejects subsequent source changes. Put outputs in
`results/`, not inside source folders. The project and Python paths below are deployment destinations, not an assertion that they already exist:

```bash
export BSK_PROJECT_ROOT=/projects/dahu1128/bsk_rl-multi-agent-space-imaging-2026
export BSK_RL_PYTHON=/projects/dahu1128/.venv-completion-v2/bin/python
export BASILISK_SOURCE_ROOT=/projects/dahu1128/basilisk-completion-v2
cd "$BSK_PROJECT_ROOT"
export PYTHONPATH="$BSK_PROJECT_ROOT/src:$BSK_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BSK_MC_MANIFEST="$BSK_PROJECT_ROOT/results/multiagent_imaging/baseline-mc/manifest.json"
export BSK_MC_OUTPUT="$BSK_PROJECT_ROOT/results/multiagent_imaging/baseline-mc/episodes"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.baseline_monte_carlo manifest \
  --config examples/multiagent_imaging/configs/baseline_mc.json \
  --output "$BSK_MC_MANIFEST"
```

After explicit authorization, submit a first LEO information pair before the full
campaign, with the account/partition from the actual cluster allocation:

```bash
sbatch --account=ucb550_asc2 --partition=acpu --qos=cpu-normal \
  --array=0,100%2 examples/multiagent_imaging/cluster/baseline_mc.slurm
```

Once these have completed and been checked, submit the **remaining 198 IDs** so
the campaign still has exactly 200 episodes:

```bash
sbatch --account=ucb550_asc2 --partition=acpu --qos=cpu-normal \
  --array=1-99,101-199%8 examples/multiagent_imaging/cluster/baseline_mc.slurm
```

Alternatively, after equivalent preflight and authorization, submit the template's
complete `0-199%8` array once. Existing episode files are never overwritten. A failed
episode with no complete atomic JSON may be retried by its exact array ID.

Aggregate after the full campaign:

```bash
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.aggregate_baseline_monte_carlo \
  --manifest "$BSK_MC_MANIFEST" --episodes-dir "$BSK_MC_OUTPUT" \
  --output-dir "$BSK_PROJECT_ROOT/results/multiagent_imaging/baseline-mc/report" --plots
```

Every array task checks for a Slurm job and executes `cluster/audit_runtime.py`
before simulation, saving its own runtime report. The baseline audit does not use
`--require-allocation`, which specifically checks the separate eight-CPU/32-GiB
learning allocation. Baseline tasks request one CPU and 4 GiB each.

Source files: `baseline_monte_carlo.py` contains the scenario, controller and runner;
`aggregate_baseline_monte_carlo.py` validates pairing and creates tables/reports;
`cluster/baseline_mc.slurm` maps array tasks to episodes. Code comments explain the
information boundary and physical versus catalog ownership.

## Local validation evidence (2026-09-06)

The new campaign tests passed: **12 tests** covering the 200-ID/four-cell mapping,
exact mixed population, reproducible orbital sampling, duplicate/reserved-target
joint assignment, coverage union denominators, source/pairing validation, real
Basilisk initial-state matching, and four short no-radio episodes. Ruff and shell
syntax checks passed. Existing Basilisk deprecation warnings remain.

Four additional saved diagnostic episodes used six targets, three candidates,
1800-second horizons and seed 0, one per cell. All completed; qualified union
capture counts were 4/6 for each LEO information case and 5/6 for each mixed case.
These short episodes contained no ground deliveries and do not validate long-run
coverage or superiority. The partial aggregator verified both initial-state pairs
and labeled the result 4/200 incomplete. Artifacts are under
`results/multiagent_imaging/baseline_mc_validation/` (manifest, raw episodes,
CSV tables and partial report). The full mission baseline campaign has not run.
