# Multi-agent space-imaging policy learning: systematic recovery prompt

Prepared 2026-10-05 from the retained failure audit, current executable source, earlier Polaris trainers, and the supplied GNC abstract. This is a work instruction, not evidence of a new training result.

**October 8 continuation:** start with [CANONICAL_WORKSPACE.md](CANONICAL_WORKSPACE.md)
and [STUDY_IMPLEMENTATION.md](STUDY_IMPLEMENTATION.md). The retained recovery
instructions below describe the earlier 100-target, 232-feature completion case.
The later opt-in 200-target inventory study has 290 features and separately
declared optimization settings. Verify the selected config and schema before
continuing; historical recovery settings are not the current study defaults.

## Copy everything below into the next task

Work in `/Users/dahu1128/Repositories/bsk_rl-multi-agent-space-imaging-2026`.

### Objective and authorized scope

Diagnose and fix the smallest demonstrated obstacles to learning useful decentralized multi-spacecraft imaging, physical downlink, and completion-message coordination for the multi-agent space-imaging framework. Establish measurable learning progress rather than treating finite losses, changing weights, or a few successful images as sufficient evidence.

Complete the evidence audit, necessary implementation, focused local tests, bounded local diagnostics, and a reviewable cluster launch package. Do not submit Slurm jobs, launch distributed cluster training, push changes, or start a broad seed study without my explicit approval. Preserve all prior evidence and use new output directories. Prepare the exact next cluster command and stop for approval at that concrete launch gate.

Use the existing implementation and training infrastructure on branch `multi-agent-space-imaging-2026` in the canonical checkout above. Read `CANONICAL_WORKSPACE.md` and preserve dirty patches and source fingerprints there before changes. All multi-agent source, tests, configuration, and new local evidence stay on this branch. Do not create a separate GNC-named implementation branch or checkout based on the paper venue. Historical source snapshots and sealed releases retain their original identities.

### Read first and reconcile what has already been done

Paths below are relative to the working repository:

- `examples/multiagent_imaging/cluster/evidence/walker4_failure_audit_20260929/REPORT.md`, `CONTRACT.md`, `REPRODUCE.md`, `analysis_final/learner_summary.json`, and `analysis_final/learner_minibatches.csv`.
- `examples/multiagent_imaging/cluster/evidence/walker4_completion_v3_cluster_pilot/REPORT.md`.
- `examples/multiagent_imaging/cluster/evidence/walker4_policy_recovery_v4_local/REPORT.md` and its corresponding raw result directories.
- `examples/multiagent_imaging/NEXT_PROMPT_POLICY_RECOVERY.md`, `WALKER4_LEARNED_PILOT.md`, `CLUSTER_READINESS.md`, `cluster/README.md`, and `cluster/EXECUTION.md`.
- `examples/multiagent_imaging/{config,environment,train,training_audit,checkpoints,evaluate}.py`, relevant configs, `cluster/learning_pilot.py`, and the prepared audit `one_worker_gate.py`.
- `src/bsk_rl/completion_gym.py`, `src/bsk_rl/{obs/completion_observations,act/completion_actions,data/completion_reward,data/multiagent_rso_reward,comm/completion_communication}.py`, and `src/bsk_rl/utils/rllib/{target_gnn_module,discounting}.py`.

Compare prior training practice using `/Users/dahu1128/Repositories/bsk_rl/examples/train_Polaris_gat_full_actions_wandb.py`, its curriculum/mixed variants, `examples/amos_2026/sbatch_train_polaris_gat_full_actions_mixed_fixed_100targets_reward_sweep_48h.sh`, and retained run/checkpoint manifests. Read those reference files; implement changes only in the multiagent checkout. Distinguish configured training allowances from actual sampled/trained decisions and completed jobs.

Search for newer recovery results before assuming the September gate remains unrun. Reconcile stale documentation against dated raw artifacts. In particular, the September 29 audit supersedes earlier statements that final v3 probabilities were unavailable. Preserve historical jobs 32528119 and 32530057 and all their evidence.

Historical starting reference (superseded by the October 8 consolidation; verify current HEAD): `e7b0d40988770ac8618bdd9392b087ea6dec2c19` plus a dirty v4 patch. The reviewed trainer SHA-256 was `1d4b3dd72c82d7680e159233a9f60b6d421cf9dd7974cec7fbded3727a33bc5d`; re-establish current provenance rather than assuming it is unchanged.

### Scientific contract

The multi-agent imaging problem is a decentralized partially observable semi-Markov decision process: local decisions occur asynchronously and physical actions have different durations. Shared policy parameters and centrally collected training data are compatible with decentralized execution; they do not grant an actor private peer state.

The earlier recovery configuration is Walker Delta 4/2/1: four sensing agents at 700 km and 97 degrees, 100 passive Basilisk targets with exactly 50 LEO/30 MEO/20 GEO, ten candidate slots and three peer slots. Passive targets must never become RL agents. Its contract is 232 float32 observation values and 18 actions; inspect the selected case's actual schema before relying on these sizes. Preserve imaging, charge, desaturation, downlink, explicit continue, and directed transmit action identities.

An actor may use its own resources, physical products, active task, observations, durable local completion history, declared target ephemerides, and permitted contact/navigation beacons. Another sensor's catalog, action, resources, active task, products or choice may affect its local knowledge only through a successfully received authorized message. Test that changing private peer state while holding local observations, received records, and allowed beacons fixed cannot change actor inputs, masks, or logits. Diagnostic logging may inspect team truth but must not feed it back to the actor or local teacher.

Keep completion-only, durable, time-tagged metadata; sender-local ACK/delta state; receiver-local eligibility; physical SimpleNav/locationPointing; Earth LOS; imaging-equivalent attitude/rate requirements; a 10-second minimum hold; 64-kbit/s metadata rate; a 64-byte header; a 300-second deadline; and 25-W draw. Sending metadata does not move image bits, physical ownership, or ground-delivery credit. Receipt must change only the receiving catalog at the correct boundary. Never replace these constraints with instantaneous information exchange to make training pass.

Reconcile the abstract's phrase “fulfilled but not yet downlinked” with the actual capture and full-delivery record revisions. Explicitly state what is sent and when; do not silently introduce intent sharing or remove documented delivery revisions. Preserve the two-orbit, capture-anchored cooldown, presently 11834.835756586714 seconds for this configuration. Preserve the 45,000-second reward half-life and 6,000-second GAE trace half-life initially.

Distinguish the three information cases: independent without intersatellite messages; centralized full-state high-information greedy reference; decentralized learned policy with physical completion messages. `ideal_completion` is an idealized completion-sharing control, not automatically a centralized full-state joint controller. The three-sensor completed baseline campaign is historical context; comparisons with a Walker-4 learned policy require fresh matched Walker-4 controls. Do not call centralized greedy a mathematical upper bound or global optimum. Keep the existing local critic for the initial recovery; adding a centralized critic would be a separate formulation change.

### Evidence to verify, not rediscover through an uncontrolled rerun

1. v3 deterministic deployment repeatedly chose empty downlink. Original final conflict probes put downlink at only 8.41–11.59% probability, with normalized entropy 0.9887–0.9932. These terminal probes reject categorical collapse in those states; they do not establish stochastic full-mission performance.
2. v4 already implements empty-storage and low-battery start masks, calibrated output heads, entropy coefficient 0.02, gradient clipping, probability/sample audits, and checkpoint compatibility checks. Do not repeat this work or load incompatible v3 checkpoints as corrected v4 policies.
3. v4's retained 1,200-second evaluations capture images but contain no full ground delivery. This is insufficient evidence for sustained catalog maintenance.
4. The fresh local audit update has 86 distinct decisions and 96 minibatch presentations. Value explained variance is approximately 0.001; 39.58% of presentations have squared value errors beyond the Ray 2.35 default clamp of 10. This is a measured current-runtime concern, not proof of the historical cluster optimization cause.
5. The current trainer uses one SGD epoch per PPO update; the earlier Polaris full-action trainer configures 30, larger batches, a much larger step allowance, and different KL/clipping settings. Quantify effective learner samples, optimizer steps, reuse and wall time before choosing changes. A requested batch minimum of 64 is not necessarily 64 actual decisions under complete-episode collection. Do not blindly copy single-agent discount values or optimizer settings into asynchronous training.
6. Acquisition and delivery rewards are assigned to the source sensor. Transmit has no direct positive reward. Trace whether the sender's advantage contains the benefit its message gives another sensor. Parameter sharing alone does not supply that missing cross-agent credit path.

Classify each claim as Confirmed, Failed, Uncertain, or Not tested, with an artifact, code line, or exact test output. Keep transport, learning mechanics, learned stochastic behavior, deterministic deployment, restore/resume, and mission performance as distinct claims.

### Stage 1: prove the experience and credit pipeline

Record Git/dirty/source fingerprints, config/schema hashes, interpreter and package versions, imported Python paths, native Basilisk binary hashes/build identity, and support-data hashes. Use the verified local interpreter `/Users/dahu1128/Repositories/bsk_rl/.venv/bin/python` only after checking what it imports. Local macOS regressions cannot certify the historical cluster Basilisk runtime.

Audit one complete experience path: timestamped observation and candidate/peer snapshot → selected action/log probability → physical start and duration → capture completion/full delivery/packet receipt → reward component → condensed decision → return/advantage → minibatch → actor and critic gradient.

Reconcile forced NO_ACTION with explicit continue; ensure only true policy choices reach PPO. Verify that condensing rewards occurs once, action/log-probability identities survive asynchronous peer boundaries, and gamma**dt/lambda**dt use physical seconds. Check true termination versus horizon truncation, final bootstrap behavior, dead-agent masks and complete episode accounting. Reconcile per-agent rewards with the unique team ledger without double credit.

Log the catalog-eligible count, returned candidate count, LOS-and-illumination imageable count for that sensor, and action-valid count separately. A valid candidate task need not be imageable now. Check that action slot i still refers to the same target in the observation, actor mask, controller and learner batch. Diagnose shortlist bottlenecks before changing selection rules. Trace current-task and onboard-product reservations, request epochs, cooldown expiry, full-product drainage and physical ownership.

Retain decision-level logits/probabilities/masks, chosen and argmax actions, entropy normalized by the available action set, top-two logit margin, local resources, action durations, packet delta size, and each reward component. Retain raw and standardized advantages, returns, predictions, value residuals/clamp fraction, KL, ratio clipping, and gradient norms for image/operational/peer heads and critic before and after clipping. Count opportunities and decisions separately: no transmit choice is interpretable only when useful peer opportunities existed.

Use focused existing tests; add tests only for a demonstrated defect or new learning contract. The initial relevant commands from the repo root are:

```sh
PYTHONPATH=src:. /Users/dahu1128/Repositories/bsk_rl/.venv/bin/python -m pytest -q \
  tests/unittest/multiagent/test_completion_policy_mask.py \
  tests/unittest/multiagent/test_completion_set_attention.py \
  tests/unittest/multiagent/test_async_discounting.py \
  tests/unittest/multiagent/test_directed_hold.py \
  tests/unittest/multiagent/test_shared_policy.py \
  tests/unittest/multiagent/test_data_boundaries.py \
  tests/unittest/multiagent/test_completion_catalog.py \
  tests/unittest/multiagent/test_completion_products.py

PYTHONPATH=src:. /Users/dahu1128/Repositories/bsk_rl/.venv/bin/python -m pytest -q \
  tests/integration/multiagent/test_completion_execution.py \
  tests/integration/multiagent/test_directed_transmission.py \
  tests/integration/multiagent/test_checkpoint_resume.py \
  tests/integration/multiagent/test_rllib_smoke.py
```

Inspect their markers/dependencies first and report skips separately from passes. Report exact commands/output; do not imply these commands have already run in this task.

### Stage 2: discriminate optimization hypotheses cheaply

First reproduce the value-clamp measurement from retained tensors using the installed learner. Ray 2.35 clamps the squared value residual itself, so residuals beyond the cap have zero gradient for that value-loss term. Verify this with autograd. Compare capped and appropriately scaled uncapped/high-cap critic fits on a fixed audited batch with identical initialization, data order and optimizer budget. Choose any trial cap from measured residual/return ranges and record the exact value. This is an offline fitting diagnostic; it is not an on-policy learning result or permission to train PPO indefinitely on old data.

Expose optimization settings in a versioned configuration instead of hiding them in hardcoded defaults. Prepare one-variable comparisons in this order: value-loss scale/cap, then update reuse (one versus a modest three or ten epochs), then learning rate/KL/entropy only if diagnostics justify it. Keep physical reward, gamma/lambda and mission semantics fixed in the first comparison. Increased SGD reuse must improve validation, not merely reduce training loss. Keep the calibrated initialization check at initialization; a useful trained policy need not remain nearly uniform.

Use a small local-only heuristic demonstration set to test whether the current actor can fit meaningful state-dependent legal decisions. Cover imaging, resource management and downlink conditions, including valid alternatives and candidate/peer permutations. Labels must use only actor-available information. Never behavior-clone the centralized joint controller into a claimed decentralized teacher. Downlink demonstrations must contain owned physical products and real ground opportunity.

Treat inability to fit these simple labels as evidence for an input, head, masking or representation problem. Successful imitation proves representation/optimization capacity, not autonomous learning or generalization. A POMDP does not by itself establish that recurrence is necessary. Audit which relevant history the local catalog/task summaries preserve; prepare a recurrent-policy comparison only after a specific history ambiguity survives the simpler checks.

### Stage 3: separate mission skills from communication credit

Start with shared-policy independent PPO for imaging, charging/desaturation and physical downlink, with transmit disabled. Keep four sensors, the mixed catalog, tensor/action identities, and task physics; version the information-case contract and any intentional compatible weight transfer. This tests whether the foundational mission tradeoff learns before adding the cost and credit difficulty of messages.

Then compare decentralized completion PPO with the same initialization and evaluation states. Add an explicit communication-credit diagnostic: a message can save the receiver from a duplicate while the sender earns no corresponding own reward. Compare sender advantage and receiver outcome under a controlled send/no-send intervention with identical initial state and reproducible action draws. Report opportunity cost, physical feasibility, changed receiver eligibility and downstream useful deliveries. Do not require equal action counts across sensors or award a bonus merely for transmitting.

If the own-reward credit path cannot represent a cooperative benefit, propose the smallest mathematically explicit cooperative-return or credit-assignment change. A training-only team reward does not automatically add a centralized actor/critic, but it is an objective/training-information change that must be declared and evaluated separately. Preserve the physical event ledger and reported mission metrics. Prepare that change for review rather than silently rewriting the task or adding arbitrary communication rewards.

Prepare a local-information heuristic warm start as a separate experiment if random-start PPO remains ineffective. Compare random-start PPO, behavior cloning alone, and cloning followed by PPO on matched validation seeds and measured budgets. Improvement over random initialization is not sufficient to credit PPO after cloning; compare against the cloned policy itself. Retain demonstrations, teacher provenance, label balance, loss curves and held-out imitation performance.

### Bounded local run and curriculum

Authorize focused local tests, offline fixed-batch diagnostics, and at most three fresh 1,200-second diagnostic PPO updates total in this task. Use conflict retasking first. Do not attempt every proposed ablation; select the smallest test justified by the preceding evidence and keep other variants as prepared configs.

If a full mission is needed to resolve the remaining credit/delivery question, run at most one instrumented fresh conflict PPO update with existing v4 settings and at most six complete 45,000-second episodes total across collection and before/after deterministic/stochastic evaluation. Make the collector respect the declared episode cap and report actual decisions. If the existing launcher cannot guarantee this budget, prepare a bounded launcher before running. Do not spend this budget on a broad sweep or both retasking modes. A single update is a diagnostic, not a convergence study.

Prepare 3,000 → 12,000 → 45,000-second curriculum stages. Initially change only the horizon; retain mission task deadlines, four sensors, exact target mix, resources, masks and cooldown. Existing 1,200-second evidence also shortened task deadlines, so it is not a clean horizon-only comparison. Verify by physical control rollouts that the proposed middle stage actually includes a ground pass and a feasible post-cooldown revisit; lengthen it if necessary. Never shorten cooldown to manufacture revisit success. Short-stage tests do not need ground delivery if no ground opportunity exists, but the full stage must include complete delivery and revisit evidence.

Inventory seed use before selecting new splits. Seeds 7000/7001 and 10000–10004 have already been inspected; treat them as historical diagnostics rather than untouched final tests. Declare training, validation, action-RNG and new reserved test seeds. Match initial-condition hashes across every control and information comparison. Evaluate initialized and subsequent checkpoints separately with deterministic argmax and seeded categorical sampling; select checkpoints using validation only.

### Prepare the exact cluster progression

Reuse and reconcile `cluster/pilot.slurm` with `cluster/evidence/walker4_failure_audit_20260929/{one_worker_gate.py,gate.slurm}`. Their run budgets differ; choose one exact next gate and document its semantics. Prepare an initial unchanged-v4, full-horizon one-worker diagnostic if the full-horizon evidence gap remains. Do not jump straight to the prior four-worker stage because historical simulator/restore checks passed.

Historical preparation requests one node, eight CPUs, 32 GiB and four hours in account `ucb550_asc2`, partition `acpu`, QoS `cpu-normal`. Verify current available resources through read-only cluster access when available; otherwise mark them unverified. The four-hour request is at most 32 requested CPU-hours, not a measured runtime or billing estimate. Profile the actual mixed mission and learner, estimate episode/decision throughput, peak RSS, checkpoint size and diagnostic-log volume, and reserve time for restores/evaluation. Do not borrow the old AMOS `amilan`/48-hour launcher without checking current availability.

Specify reviewed cluster checkout/commit plus patch fingerprint, Python executable, Basilisk build and native libraries, isolated support-data cache, thread caps, Ray temp directory, seed streams and a fresh output path. Keep credentials out of logs. Use `sbatch --export=ALL` in the prepared command because the historical shell alias suppressed exports; confirm current behavior. Validate shell syntax, config/schema, output nonexistence, runtime imports, and dry-run plan locally. A Slurm script passing syntax checks is not a completed job.

After approval, the first cluster diagnostic must preserve full policy and optimizer checkpoints, restore in a fresh process without an extra train call, prove frozen logit/action agreement and optimizer/seed/counter continuity, and save failed runs for diagnosis. A later topology change from one to four workers is a separate experiment with explicitly documented continuation semantics.

Prepare a subsequent learning run only after the diagnostic identifies a viable configuration. Set its budget by actual condensed agent decisions, episodes and optimizer steps as well as wall time; do not use “ten updates” as a convergence target. Plan periodic full-mission validation, best-checkpoint retention, exact resume, and learning curves against the initialized/BC policy. First resolve conflict mode; continuous mode and multiple training seeds follow a demonstrated useful policy. Prepare the later study, with independent learned operation and matched four-sensor greedy controls, as a separate owner-approved expansion.

### Acceptance and deliverables

Separate three gates:

1. Mechanics: valid observations/actions, finite losses and gradients, real parameter changes, correct physical-time credit and unique ledger, no peer-state leakage, and exact schema-compatible restore/resume. This gate does not prove learning.
2. Operational mission: every expected live sensor survives the declared horizon; zero empty-storage downlinks; no repeated invalid-action loop; physically qualified capture and complete ground delivery; useful post-cooldown revisit on a scenario with verified opportunity; more than one purposeful action class; and no persistent storage saturation/resource exhaustion. Retain the existing minimum 5% battery, 95% maximum storage, at least one qualifying image and full delivery as explicit diagnostic thresholds where applicable, not as claims of competitive mission performance. Report per-sensor outcomes, storage dwell time and service opportunities. Test physical transport separately so an intentionally independent actor is not required to transmit.
3. Learning: improvement over the same configuration's initialized policy on fresh fixed validation seeds, under a predeclared budget and checkpoint rule. Prefer credited delivered priority value as the primary catalog-maintenance metric; report credited delivered visits/useful downlinks, distinct coverage, valid revisits, duplicate images, acquisition-to-full-ground latency, resources and message cost. For a warm start, require a comparison to BC alone. Declare practical improvement and safety thresholds before reading new outcomes, using serviceable matched controls and pilot variability. Report uncertainty; do not demand an arbitrary significance cutoff from a tiny canary or call mechanics a performance win. Learned communication additionally needs useful information delivery and an outcome comparison with messages disabled, not packet count alone.

For latency include median/p90 among delivered products plus the undelivered fraction/age at horizon; reporting only successful deliveries can hide bad scheduling. Count coverage as a union across targets and repeat visits separately. Plot validation performance against actual sampled decisions, optimizer steps and wall time; also show state-conditional action distributions, value-fit/clamp trends and per-sensor workload. Use seed/mission as the sampling unit for paired uncertainty; sensors sharing a mission and policy are not independent replications.

Deliver a concise plain-language diagnosis; updated claim ledger; ranked failure tree with falsifying tests; code/config patch and source manifest; exact test commands/output; bounded local before/after results; comparison to earlier Polaris training settings and realized budgets; checkpoint selection/resume rule; and one complete dry-run-validated cluster launch package with resource estimate and explicit approval gate. Include a smallest-next-experiment recommendation for every unresolved branch.

Stop expensive escalation when a demonstrated correctness defect invalidates experience, return/advantage attribution fails, resources collapse, outputs are nonfinite, or the runtime cannot be verified. Repair and revalidate within the bounded local scope. Missing raw historical learner artifacts block an exact historical root-cause claim; they do not block fresh instrumented local tests. If cluster authentication is unavailable, state the missing access/artifact and finish the local package without implying a cluster result.

Finish only when the next action is concrete and reviewable. Do not end with generic “try more training” advice. Wait for my explicit approval before submitting the prepared cluster gate.

## Primary implementation references

The retained repository audit supports the numerical observations above. The installed learner and the [pinned Ray 2.35 PPO learner source](https://github.com/ray-project/ray/blob/ray-2.35.0/rllib/algorithms/ppo/torch/ppo_torch_learner.py) support the squared-error clamp interpretation. The [original PPO paper](https://arxiv.org/abs/1707.06347) describes minibatch update reuse; it does not establish the appropriate epoch count or learning budget for this spacecraft experiment.
