# Research Focus I campaign status and reusable prompts

This runbook turns the live AMOS 2025 architecture-comparison campaign into
explicit, gated tasks. Each prompt is intended to be pasted into a fresh Codex task
when its entry condition is satisfied. Do not execute later prompts merely because a
calendar date has arrived; verify the gate first.

## Status snapshot: 2026-08-16 23:47 MDT

- Study branch: `amos2025-architecture-comparison`.
- Expected cluster checkout: `/projects/dahu1128/bsk_rl-rfi`.
- Expected branch commit at this snapshot: `9185bb9`.
- Completed closest-angle heuristic campaign:
  `/scratch/alpine/dahu1128/prospectus_rfi/heuristic_mc/amos2025_closest_angle_100s_20260815T183838Z`.
- Completed frozen AMOS 2025 policy transfer campaign:
  `/scratch/alpine/dahu1128/prospectus_rfi/legacy_policy_mc/amos2025_alpha0_300s_to_100s_20260817T004436Z`.
- Both completed campaigns contain 300 validated episodes: catalog sizes 100, 200,
  and 400, with matched seeds 0 through 99.
- Frozen policy checkpoint:
  `/projects/dahu1128/policy_artifacts/amos2025_alpha0_best_iter427/inspector`.
- Frozen policy `module_state.pt` SHA-256:
  `6db5bcd4fda20205977dfab377441f625051ef9e9dfaebde5e8db5ec1ab0e2c4`.
- Acquisition-timeline replay array: job `31360337`, 600 independent tasks.
  Tasks 0 through 89 were complete with exit code `0:0`; tasks 90 onward were
  advancing. The remaining-array throttle was effectively increased to 100.
- Timeline task map:

  | Array IDs | Method | Catalog size | Seeds |
  | --- | --- | ---: | ---: |
  | 0--99 | closest-angle heuristic | 100 | 0--99 |
  | 100--199 | closest-angle heuristic | 200 | 0--99 |
  | 200--299 | closest-angle heuristic | 400 | 0--99 |
  | 300--399 | frozen AMOS 2025 policy | 100 | 0--99 |
  | 400--499 | frozen AMOS 2025 policy | 200 | 0--99 |
  | 500--599 | frozen AMOS 2025 policy | 400 | 0--99 |

- Architecture training consists of six exploratory, single-seed runs:

  | Architecture | K | Active first-segment job |
  | --- | ---: | --- |
  | fixed-input monolithic MLP | 5 | `31323362_0` (original 48-hour allocation) |
  | fixed-input monolithic MLP | 10 | `31344908_1` |
  | fixed-input monolithic MLP | 20 | `31344911_2` |
  | target-set attention policy | 5 | `31344914_3` |
  | target-set attention policy | 10 | `31344917_4` |
  | target-set attention policy | 20 | `31344920_5` |

- The K=10/K=20 MLP and K=5/K=10/K=20 attention runs have independent continuation
  chains waiting on dependencies. Their pending dependency state is intentional and
  is not evidence that a job is stuck.
- W&B project: `amos2025-architecture-comparison`.
- W&B group: `rfi-alpha0-100s-candidate-sweep`.

The message `31360337_30-89: Job has already finished` from `scontrol update` was
harmless: Slurm could not alter already-completed elements, while the remaining tasks
subsequently expanded to approximately 100 concurrent tasks. Do not resubmit those
completed elements.

## Rules that apply to every prompt

1. Start with read-only inspection: branch, commit, `git status`, `squeue`, `sacct`,
   manifests, and expected output counts.
2. Do not cancel, hold, requeue, or duplicate a live job unless an actual failed or
   missing unit has been identified and documented.
3. Do not recursively scan or analyze the GPFS scratch tree from a login-node Python
   process. Submit such work as a small `acpu` batch job and write its stdout to
   `/projects/dahu1128` so it is easy to inspect.
4. Preserve every accepted raw result. Recovery jobs must target only missing or invalid
   task IDs and must write atomically into the existing campaign root.
5. Call the old checkpoint a **frozen AMOS 2025 policy transfer baseline**. It was
   trained with 300-second imaging actions and is being evaluated without retraining in
   the 100-second environment; it is not a matched newly trained policy.
6. Call the newer learned architecture a **target-set attention policy**, not a GAT or
   GNN, unless explicit graph nodes, edges, or adjacency are demonstrated in the code.
7. The current six architecture/K runs use one training seed (`10001`) and are
   exploratory. They cannot establish training-seed variance or architecture
   superiority.
8. Do not claim that a policy is faster or better until the paired results support it.
   Target saturation is a hypothesis to test, not an assumed conclusion.
9. Report exact commands, job IDs, roots, counts, failures, and the next decision gate.

## To-do 1: monitor the live jobs now

**Entry condition:** Run this while timeline job `31360337` or any of the six first
architecture-training allocations remains active.

**Prompt to paste:**

> Work in the local BSK-RL repository on branch
> `amos2025-architecture-comparison` and help me audit the currently running Alpine
> Research Focus I campaign. This is a read-only monitoring task. Do not submit,
> cancel, hold, requeue, or modify any Slurm job.
>
> The acquisition-timeline array is job `31360337`, with 600 tasks. Map IDs 0--99 to
> heuristic/N=100, 100--199 to heuristic/N=200, 200--299 to heuristic/N=400,
> 300--399 to frozen-policy/N=100, 400--499 to frozen-policy/N=200, and 500--599 to
> frozen-policy/N=400. Use `squeue` and `sacct -X` to count tasks in COMPLETED,
> RUNNING, PENDING, FAILED, CANCELLED, TIMEOUT, and OUT_OF_MEMORY states. Confirm that
> all completed tasks have exit code `0:0`. Summarize completion separately for each
> method/catalog-size block. Do not use `tail` as a substitute for counting all task
> states. Estimate remaining wall time from the median duration of completed tasks,
> clearly labeling that estimate as approximate.
>
> Also audit the six architecture-training runs without changing them. The expected
> first allocations are `31323362_0`, `31344908_1`, `31344911_2`, `31344914_3`,
> `31344917_4`, and `31344920_5`. Inspect their continuation dependencies and explain
> whether each pending job is correctly waiting for its predecessor. Confirm the
> architecture/K mapping, current elapsed time, state, latest checkpoint or training
> metric timestamp, W&B project/group, and whether the logs contain `Traceback`,
> `Error`, NaN losses, or repeated restarts. Do not recursively scan scratch from the
> login node; if checking multiple files is needed, give me a small `sbatch --wrap`
> audit command.
>
> End with three sections: `Healthy`, `Needs attention`, and `Next check`. If there is
> no actual failure, explicitly tell me to leave the jobs alone and when to check again.

**Expected decision:** If there are no failures, wait. For the timeline array, a
30--60-minute check is reasonable given roughly three- to four-minute task runtimes and
100-way concurrency. Long training only needs a few checks per day.

## To-do 2: validate the completed timeline grid and recover only gaps

**Entry condition:** `squeue -j 31360337` is empty. A job leaving the queue is not by
itself proof that all 600 tasks succeeded.

**Prompt to paste:**

> The AMOS 2025 acquisition-timeline array `31360337` has left the Alpine queue. Audit
> it before any analysis. Work on branch `amos2025-architecture-comparison`, pull with
> `--ff-only`, and preserve all existing results. The heuristic campaign root is
> `/scratch/alpine/dahu1128/prospectus_rfi/heuristic_mc/amos2025_closest_angle_100s_20260815T183838Z`.
> The frozen-policy campaign root is
> `/scratch/alpine/dahu1128/prospectus_rfi/legacy_policy_mc/amos2025_alpha0_300s_to_100s_20260817T004436Z`.
>
> First inspect `sacct -X -j 31360337` and report the exact count of every terminal
> state and nonzero exit code. Then use the repository's timeline scan/recovery logic
> to verify the full Cartesian grid: two methods, N in {100, 200, 400}, and seeds
> 0--99, for exactly 600 timeline CSV/metadata pairs. Run any multi-file validation as
> a small `acpu`, `cpu-normal` compute job, with stdout under `/projects/dahu1128`.
> Do not run a recursive GPFS scan in a login-node foreground process.
>
> Validate more than file existence. For every timeline, require parseable CSV and
> JSON, correct method/N/seed identity, a time range beginning at 0 and reaching the
> 45,000-second episode endpoint, nondecreasing cumulative illuminated observations,
> no impossible negative counts, the expected scenario fingerprint, and final totals
> matching the previously accepted one-row episode result. Confirm that the policy
> replay used checkpoint SHA-256
> `6db5bcd4fda20205977dfab377441f625051ef9e9dfaebde5e8db5ec1ab0e2c4`.
>
> If and only if files are missing or invalid, produce the exact missing array IDs and
> use `submit_acquisition_timeline_mc.sh` against the same two roots so it schedules
> only those units. Do not create a new campaign root or rerun successful units. After
> recovery, rerun the audit and provide a machine-readable completion manifest. End by
> saying either `timeline analysis gate passed` or listing the exact unresolved IDs.

**Expected output:** A 600/600 validated grid, no unexplained terminal failures, and a
saved audit manifest. Analysis remains blocked until this passes.

## To-do 3: run the saturation-aware acquisition-speed analysis tomorrow

**Entry condition:** To-do 2 reports `timeline analysis gate passed`.

**Prompt to paste:**

> Analyze the completed paired acquisition timelines for the closest-angle heuristic
> and the frozen AMOS 2025 policy transfer baseline. Work in the BSK-RL study branch
> and use the existing script
> `examples/prospectus_rfi/analyze_acquisition_timelines.py`. The heuristic root is
> `/scratch/alpine/dahu1128/prospectus_rfi/heuristic_mc/amos2025_closest_angle_100s_20260815T183838Z`;
> the policy root is
> `/scratch/alpine/dahu1128/prospectus_rfi/legacy_policy_mc/amos2025_alpha0_300s_to_100s_20260817T004436Z`;
> and the analysis output root should be
> `/scratch/alpine/dahu1128/prospectus_rfi/acquisition_timeline_analysis/amos2025_frozen_policy_vs_heuristic`.
>
> Submit the analysis as a small Alpine `acpu` compute job rather than running it on a
> login node. Preserve raw timeline files and save both raw and plotted data. Resample
> each irregular decision history by forward-filling onto the common 100-second grid
> from 0 through 45,000 seconds; do not interpolate counts or apply smoothing. Plot
> cumulative unique illuminated observations throughout the full episode, separately
> for N=100, 200, and 400. Show the mean or median central curve as explicitly labeled,
> with a transparent 95% paired-bootstrap uncertainty band based on the 100 matched
> scenario seeds. Also plot the paired policy-minus-heuristic curve and its confidence
> band at every grid time.
>
> Produce tables at 15,000, 30,000, and 45,000 seconds, but do not reduce the figures
> to those three checkpoints. Calculate normalized acquisition AUC and time to reach
> 50%, 80%, 90%, and 95% of each episode's own final illuminated count. Report paired
> differences, paired 95% bootstrap confidence intervals, paired tests, multiplicity
> correction, and the predeclared practical-equivalence classification. Save PDF and
> SVG figures plus CSV/JSON data and record the commit/configuration in metadata.
>
> Test the saturation hypothesis honestly: determine whether endpoints converge while
> one method reaches the plateau sooner. Do not assume that unobserved targets are
> geometrically unreachable without evidence. Discuss the transfer mismatch: the
> frozen policy was trained with 300-second imaging actions, then evaluated here with
> 100-second actions. Clearly separate final acquisition count from acquisition speed
> and do not claim that this transfer baseline represents the newly trained policies.
>
> Verify every expected output after the job finishes and summarize numerical findings,
> figure paths, limitations, and wording safe to use in a prospectus. If the evidence
> does not show earlier acquisition by the policy, say so directly.

**Expected output:** Continuous 100-second acquisition curves and paired statistics,
not just final episode totals.

## To-do 4: package and pull the completed Monte Carlo and timeline results

**Entry condition:** The timeline audit and acquisition analysis are complete.

**Prompt to paste:**

> Package the completed AMOS 2025 heuristic, frozen-policy transfer, timeline, and
> acquisition-analysis artifacts on Alpine and give me exact commands to pull them to
> my local Mac. Do not modify or delete cluster results. First inventory file counts,
> sizes, campaign manifests, completion JSON, raw episode CSVs, timeline CSV/metadata,
> analysis tables, and PDF/SVG figures. Exclude transient W&B caches and unrelated
> AMOS 2026 data.
>
> Build a checksum-verified archive from a compute job if reading the scratch tree is
> substantial. Place the archive, inventory, and SHA-256 file under
> `/projects/dahu1128/prospectus_rfi_exports/`, where login-node transfer is stable.
> Preserve the campaign directory hierarchy in the archive. Then give me an `rsync`
> command to download into
> `/Users/dahu1128/Repositories/bsk_rl/results/prospectus_rfi/cluster_downloads/`.
> Verify the local checksum and extract into a timestamped directory without
> overwriting an earlier download.
>
> After extraction, run the repository collectors/analysis in local read-only mode as
> a reproducibility check where practical. Tell me which outputs are suitable for Git
> (small metadata, scripts, and final figures) and which raw or checkpoint artifacts
> should remain ignored. Report exact archive paths, sizes, checksums, and validation
> counts.

**Expected output:** One immutable cluster export, one verified local copy, and no raw
Monte Carlo bulk accidentally committed to Git.

## To-do 5: audit architecture training each day

**Entry condition:** Any of the six architecture/K campaigns is incomplete.

**Prompt to paste:**

> Perform a read-only daily health audit of the AMOS 2025 Research Focus I training
> campaign on Alpine. Do not cancel, resubmit, resume, or alter dependencies unless you
> first prove a real failure and ask me before making a state-changing scheduler action.
> The architecture/K cells are fixed-input monolithic MLP K={5,10,20} and target-set
> attention policy K={5,10,20}, all with exploratory seed 10001.
>
> Inspect the original job `31323362_0`, first segmented jobs `31344908_1`,
> `31344911_2`, `31344914_3`, `31344917_4`, and `31344920_5`, plus their continuation
> and cleanup dependencies recorded in the campaign manifest. The known dependency
> job IDs are `31344909_1`, `31344910_1`, `31344912_2`, `31344913_2`,
> `31344915_3`, `31344916_3`, `31344918_4`, `31344919_4`, `31344921_5`, and
> `31344922_5`. Reconstruct each chain rather than assuming numeric order implies its
> purpose.
>
> For each cell, report scheduler state, segment number, elapsed allocation time,
> cumulative training wall time, environment steps, samples per second, last metric
> timestamp, newest valid checkpoint, W&B run ID/URL, and whether the deterministic W&B
> run resumed rather than forked. Check logs for tracebacks, NaNs, OOM, timeouts,
> repeated startup loops, checkpoint corruption, or failure to advance environment
> steps. Confirm that the current configuration still uses a 45,000-second episode,
> 100-second imaging, alpha=0 observation-only reward, catalog N sampled uniformly from
> 100--400 at reset, randomized 20--60% initial battery, no re-imaging, and the intended
> K.
>
> Compare progress for health monitoring but do not rank architectures by raw PPO
> training return. Explain whether each continuation is correctly waiting, running, or
> complete and whether every cell is still on track for the same cumulative 48-hour
> training allowance. End with an action table containing `leave alone`, `inspect`, or
> `recovery needed`; include a specific reason for any state other than `leave alone`.

**Expected decision:** Most daily checks should end with `leave alone`. A Slurm
dependency wait is normal.

## To-do 6: validate and select checkpoints after all six runs finish

**Entry condition:** Every architecture/K chain has reached its intended cumulative
48-hour training allowance and has a final checkpoint. No upstream training job is
still running or eligible to resume.

**Prompt to paste:**

> All six exploratory architecture/K training campaigns appear complete. Before
> evaluation, audit cumulative wall time, environment steps, checkpoint integrity, and
> W&B continuity for fixed-input monolithic MLP and target-set attention at K=5, 10,
> and 20, seed 10001. Do not use a partially trained or failed cell. Save a completion
> table and identify any run that did not receive the common cumulative 48-hour budget.
>
> If the completion gate passes, submit
> `examples/prospectus_rfi/slurm/validate_candidate_sweep.sbatch` with no stale
> dependency on the old array ID. Validate periodic and final checkpoints on the same
> held-out physical seed set, using the common operational score and matched physical
> metrics rather than raw training return. Record successful observations, illuminated
> observations, useful deliveries, backlog, action allocation, resource-constraint
> interventions, and inference cost. Keep K explicit in all rows.
>
> Select and materialize one `checkpoints/best_validation` checkpoint for each of the
> six cells according to the predeclared selection rule. Verify the symlink or copied
> checkpoint can load independently. Save all validation rows, rejected checkpoint
> candidates, the selection rule, and selected checkpoint hashes. Report whether
> apparent differences are robust across held-out scenarios, but label them exploratory
> because training seed variance has not yet been measured. Provide the exact validation
> job ID, output paths, and any cell that blocks paired evaluation.

**Expected output:** Six loadable `best_validation` checkpoints with a transparent
selection audit.

## To-do 7: launch the paired Monte Carlo for the newly trained policies

**Entry condition:** To-do 6 has produced six valid `best_validation` checkpoints.

**Prompt to paste:**

> Launch and supervise the paired Monte Carlo evaluation for the six newly trained
> exploratory policies using
> `examples/prospectus_rfi/slurm/evaluate_paired_mc.sbatch`. Begin with `--test-only`,
> inspect the task map, and confirm the checkpoint paths before submission. Evaluate
> fixed-input monolithic MLP and target-set attention policies for K={5,10,20}, plus
> the historical and information-matched heuristic comparisons encoded by the study.
>
> Use fixed catalog sizes N={100,200,300,400}, 100 scenario seeds per method/cell, and
> the identical scenario seed for every paired method comparison. Do not retrain at the
> evaluation catalog sizes. Confirm the environment remains the 45,000-second AMOS
> 2025 physical scenario with 100-second imaging, observation-only alpha=0, no
> re-imaging, matched action feasibility/resource shield, and the configured battery
> distribution. Distinguish catalog N from presented-candidate K everywhere. Document
> any heuristic information advantage.
>
> Submit all independent blocks with the repository's intended array concurrency and
> positive nice value so this campaign does not disrupt unrelated training. Record the
> job ID, commit, root, task map, checkpoint hashes, and expected episode count. Monitor
> by state counts, not scrolling logs. After the array leaves the queue, validate every
> expected method/K/N/seed row and recover only missing or invalid blocks. Do not draw
> conclusions from a partial grid.

**Expected output:** A complete, paired evaluation grid with machine-readable
provenance and no unpaired seed substitutions.

## To-do 8: produce the exploratory architecture comparison

**Entry condition:** The full paired Monte Carlo grid from To-do 7 is validated.

**Prompt to paste:**

> Run the complete Research Focus I exploratory analysis with
> `examples/prospectus_rfi/analyze.py --input-root /scratch/alpine/dahu1128/prospectus_rfi`.
> Run substantial scratch reads on a compute node. Save raw plotted data plus PDF and
> SVG figures. Verify every figure and table against the input row counts before
> interpreting results.
>
> Compare training performance against environment steps and wall-clock hours; use
> held-out validation performance rather than raw training return as the operational
> curve. Compare catalog-size generalization at N=100, 200, 300, and 400. Report
> successful/illuminated observations, deliveries, backlog, action allocation,
> resource-constraint interventions, parameter count, inference time, sample
> throughput, and time to predeclared thresholds. Keep K=5,10,20 visible and separate
> total catalog size N from presented-candidate count K.
>
> For primary paired metrics report mean, standard deviation, median, IQR, paired
> method differences, paired 95% bootstrap confidence intervals, an appropriate paired
> test, Holm multiplicity correction, and the predeclared practical-equivalence margin.
> Do not smooth away instability. If two methods are statistically or practically
> equivalent, say so. If one K seems best from this single seed, describe it as a
> candidate for confirmatory replication, not a proven optimum.
>
> Update `prospectus_results.md` with exact numerical values, uncertainty intervals,
> figure paths, limitations, and language suitable for the prospectus. Explicitly state
> that each architecture/K cell currently has one training seed and therefore does not
> support claims about training stochasticity or definitive architecture superiority.
> Cross-reference the separate frozen-policy transfer/saturation analysis without
> conflating it with newly trained 100-second policies.

**Expected output:** Publication-quality exploratory figures and honest, qualified
prospectus wording.

## To-do 9: design the confirmatory multi-seed campaign next week

**Entry condition:** The exploratory validation and paired evaluation are complete.
This is a design and compute-budget gate; do not launch long jobs without explicit
approval.

**Prompt to paste:**

> Design the confirmatory Research Focus I campaign using the completed exploratory
> evidence. Do not submit jobs yet. Preserve equal scientific treatment of the
> fixed-input monolithic MLP and target-set attention policy. First audit whether the
> originally required equal-compute, architecture-specific hyperparameter tuning was
> completed; if not, plan separate bounded tuning studies with the same total compute
> allowance for each architecture and held-out-seed selection using a common physical
> performance score.
>
> Use the exploratory K={5,10,20} results to predeclare which K values will be carried
> into confirmation. Do not choose K solely from noisy raw training return. At minimum,
> plan three independent training seeds for every retained architecture/K cell, each
> with the same 48-hour wall-clock allowance on equivalent hardware. Include learning
> rate, batch sizes, minibatch size, PPO epochs, clip parameter, entropy/value
> coefficients, gradient clipping, discount/continuous-time discount treatment, GAE
> lambda, width/depth, and the attention embedding/heads/blocks/feed-forward width in
> the tuning record.
>
> Prepare a compute table containing number of tuning trials, final runs, nodes/CPUs,
> node-hours, expected storage, checkpoint cadence, validation episodes, paired Monte
> Carlo episodes, and expected W&B runs. Build restartable Slurm arrays and continuation
> chains that do not duplicate completed work. Predeclare the validation seeds,
> performance thresholds, primary metrics, bootstrap method, statistical tests,
> multiplicity correction, and practical-equivalence margins before the final results
> are examined.
>
> Return a staged proposal with `minimum defensible`, `preferred`, and `expanded`
> compute options, exact config/script changes, launch commands, expected cost, and
> decision points. Ask for my approval before submitting any long-running confirmatory
> job.

**Expected output:** An auditable multi-seed protocol and compute request, not a launch.

## To-do 10: audit redundant, abandoned, or invalid research folders

**Entry condition:** Primary campaign data is safely packaged locally. This is secondary
work and must not delay or modify active runs.

**Prompt to paste:**

> Perform a read-only provenance and storage audit of BSK-RL research folders on my
> local machine and, where necessary, Alpine. Do not delete, move, rename, or overwrite
> anything. Focus on identifying redundant copies, abandoned planning folders, failed
> campaigns, partial data, known-wrong results, superseded analyses, and authoritative
> artifacts for AMOS 2025 Research Focus I versus unrelated AMOS 2026 work.
>
> Start from Git history, manifests, commit hashes, configuration fingerprints,
> checkpoint hashes, completion JSON, expected row counts, job IDs, timestamps, and
> README/prospectus references. Never classify a folder as redundant based only on its
> name or age. Hash large files only when needed, and avoid recursive GPFS scans on a
> login node. Build a CSV/Markdown inventory with path, owner campaign, size, status,
> evidence, authoritative replacement, reproducibility value, and recommended action.
>
> Use conservative categories: `authoritative`, `retain for provenance`, `active`,
> `duplicate verified by hash`, `incomplete but recoverable`, `invalid with documented
> reason`, `unreferenced/uncertain`, and `candidate for archival`. For anything believed
> wrong, cite the exact defect and the corrected replacement. Produce a proposed archive
> and deletion plan with estimated space savings, but stop for my approval before any
> destructive action. Keep the AMOS 2025 and AMOS 2026 research records clearly
> separated.

**Expected output:** A defensible inventory and proposed cleanup plan with zero files
changed.

## Calendar and decision gates

### Now: late August 16 / early August 17

1. Use To-do 1 to monitor timeline job `31360337` and the six training runs.
2. Leave healthy jobs alone. Do not submit another timeline array.
3. Once `31360337` leaves the queue, move to To-do 2.

### Tomorrow: August 17

1. Validate all 600 acquisition timelines and recover only genuine gaps (To-do 2).
2. Run the saturation-aware continuous acquisition analysis (To-do 3).
3. Package and pull the completed heuristic/policy/timeline analysis (To-do 4).
4. Run one daily architecture-training health audit (To-do 5).

### The following several days

1. Continue To-do 5 while the 24-hour training segments and their dependency chains
   advance.
2. Do not start checkpoint validation until all six cells have reached the common
   cumulative training budget.
3. Preserve W&B run continuity and verify new checkpoints after every segment boundary.

### Next week

1. Validate and select the six exploratory checkpoints (To-do 6).
2. Run the paired Monte Carlo for the newly trained policies (To-do 7).
3. Produce the exploratory architecture/K analysis and prospectus text (To-do 8).
4. Design the confirmatory equal-tuning/multi-seed campaign and request approval before
   launch (To-do 9).
5. Only after the primary results are safely packaged, perform the read-only folder
   hygiene audit (To-do 10).

Dates are planning targets, not permission to skip a gate. Scheduler delays or failed
units may shift later work without changing the scientific order.
