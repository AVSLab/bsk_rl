> Historical pre-implementation review. Option 2 is now implemented; see [ARCHITECTURE.md](ARCHITECTURE.md) for the current design and [TEST_RESULTS.md](TEST_RESULTS.md) for verification.

Review dated September 5, 2026. Recommendation: retain the current AMOS-derived branch, reuse BSK-RL's event/continuation interfaces, and implement completion-sharing semantics plus an explicit retasking selector. Use Option 2 below as the research implementation, with Option 1 as its validation stage.

The supplied slides and the user's completion-only direction take precedence over older prospectus and branch text that proposes intent sharing. This is a review and implementation proposal; it does not change simulation, training, or communication behavior.

**Evidence and scope.** Reviewed `multi-agent-space-imaging-2026` at `d6a386e1e291d721ffdbea4574e2e4997f50de44`, including the existing uncommitted worktree edits; refreshed upstream `develop` to `3f4c4f16d1778f6dd1a69468ea154a1f3a5f98d9`; inspected `feature/144-relative-motion`, `feature/fires`, and the observation-branch history. Also read the local prospectus at `/Users/dahu1128/Documents/PhD/Comprehensive Exam/prospectus_20page/sec_focus3.tex`. Existing user edits were preserved.

The current branch descends from AMOS commit `0a05f2bd72872dc8272da673b550b3f1c9daafab`. Its common ancestor with current upstream develop is `0d60cb7efa0d3d08cc20430c6a98cb9738aef906`. The two branches have 280 and 219 commits, respectively, after their merge base. Moving onto develop is therefore a substantial migration, especially because upstream replaced monolithic `sim/dyn.py` and `sim/fsw.py` with modular packages.

**What the related work actually observes.** These are distinct experiments, rather than one universal BSK-RL observation contract.

| Reference | Per-agent policy information | Communication and peer information |
|---|---|---|
| Stephenson, Mantovani, Schaub, AAS 24-192 (2024), Table 1 | Own angular rate, pointing, Earth-fixed position/velocity, episode time, charge/eclipse; upcoming target rewards/positions and reward density | Completion or intent changes eligible requests. The table has no explicit teammate position/velocity block. It compares retasking all agents with retasking agents whose tasks are finished or invalidated. |
| Stephenson and Schaub, AMOS 2025 inspection, Table 1 | Own RSO-relative position/velocity, Sun direction, orbital eccentricity/angular-momentum/position vectors, fuel, time/eclipse, and 15 regional inspection fractions | The multi-agent-trained variant additionally receives each other inspector's position and velocity relative to the RSO: `6(N-1)` features. The single-agent-trained variant deployed with multiple inspectors omits these. Free exchange supports joint inspected-region knowledge. Fixed peer slots prevent transfer to arbitrary agent counts. |
| Stephenson and Schaub, IEEE Aerospace 2026 tip-and-cue, Table 1 | Own rate, boresight, Earth-fixed position/velocity; priorities, Hill-frame target positions, pointing errors, and opportunity windows for 32 known, unimaged targets | Free constellation-wide updates of newly scanned and imaged targets every 50 seconds. No peer position, velocity, resources, or intended-action block appears in the policy table. |

Sources: [2024 constellation paper](https://hanspeterschaub.info/Papers/Stephenson2024b.pdf); [2025 inspection paper](https://amostech.com/TechnicalPapers/2025/Machine-Learning-for-SDA-Applications/Stephenson.pdf), especially pp. 3, 5–6; [2026 tip-and-cue paper](https://hanspeterschaub.info/Papers/Stephenson2026.pdf), especially pp. 2–3. The inspection table was also rendered and visually checked.

The public upstream `benchmarks/rso_inspection.py` and `examples/rso_inspection.ipynb` configure the own-state/chief-relative observation variant. `obs.RelativeProperties` can evaluate functions against a named chief; `rso_imaged_regions` reads the inspector's datastore. These examples are not evidence that the multi-agent paper omitted peers: the paper explicitly includes them for multi-agent training. Collision-checking and shielding can also consume peer dynamics independently of the policy input. Availability of simulator state does not automatically authorize exposing it to a decentralized actor.

The old `feature/144-relative-motion` tip (`26dc2da`, June 2025) is an outline with unfinished observation functions. [PR 214](https://github.com/AVSLab/bsk_rl/pull/214) explicitly says its contents were being split into other PRs and that the outline would not be merged. The useful inspection implementation is now in develop. `feature/fires` (`99d8618`, September 2024) contains a preliminary fire example, not a verified reproduction of the 2026 tip-and-cue experiment. `feature/more-observations` is still older, using the pre-refactor package layout. None is an appropriate new base for this study. The exact paper experiment training repositories were not located among the available public branches; paper claims above are distinguished from inspected library examples.

**What already exists and what needs work.**

| Component | Existing support | Required change for the slides |
|---|---|---|
| Event propagation | Basilisk terminal events; BSK-RL timed action termination and earliest-event propagation | Reuse. Add explicit message arrival/completion events when communication has independent timing. |
| Busy-agent continuation | `requires_retasking`, `NO_ACTION`, `ContinuePreviousAction`, `CondenseMultiStepActions` | Reuse. Their presence alone does not implement conflict retasking on received completion. |
| Continuous retasking | Environment accepts new actions for all agents at each boundary | Add a named mode, produce real observations for every retasked agent, and keep the continuation connector consistent with the selected decision set. |
| Conflict retasking | Local completion/timeout sets `requires_retasking` | Add receiver-local active-request invalidation after catalog merge, before observation generation. |
| Local knowledge | `LocalCatalogKnowledge` and typed directional inbox/channel | Retain ownership boundaries; replace target-wide last-writer ordering with fact/service-record merge semantics. |
| Completion exchange | Capture/delivery/cooldown fields exist inside intent/status messages | Remove future-task fields; send completion deltas for all relevant updated records, not just the active or latest target. |
| Ground data and reward | Sensor-owned products and private team accounting | Preserve. Sharing a completion never transfers an image or its delivery credit. |
| Observation representation | Shared target attention, `14 + 13K` inputs | Remove teammate intent; add useful time-tagged service features and explicit validity. Unify candidate/action selection. |

There is no ready-made, named continuous/conflict retasking implementation in the inspected modern library branches that can simply be imported. The reusable machinery is present; the domain-specific completion trigger is the missing layer. Basilisk owns physics and event execution; BSK-RL should own the decision rule and information boundaries.

**Specific review findings.**

1. Current configuration is local-event continuation, not the slide's full conflict rule. `environment.py` sets `generate_obs_retasking_only=True`; training installs `ContinuePreviousAction`; `gym.py::_step` runs communication after reward updates, but no subsequent hook marks a busy imager for retasking when its request becomes locally known completed. A message may update knowledge while the current action continues until its own event.

2. Observation and action candidates can disagree. `obs/observations.py::_eligible_targets_now` intersects datastore eligibility with the local catalog. `act/discrete_actions.py::ImageRSO._eligible_targets_now` reads only datastore eligibility. A direct executable check with target 1 blocked by received knowledge produced observation candidates `[2]` and action candidates `[1, 2]`. Thus an index can select a different RSO from the one represented by the observation. Both builders also have closest-known-target fallbacks when eligibility is empty. Use one candidate snapshot and a mask; never repopulate a completed-only catalog with blocked targets just to fill slots.

3. The finite broadcast duration is not enforced as a delivery gate. `BroadcastIntent.set_action` sets `broadcast_pending` immediately. `communicate()` consumes that flag on the next global boundary. A focused executable check started a 30-second broadcast at time 0 and delivered its message at time 5. Upstream `Broadcast`/`BroadcastCommunication` uses the same pending-flag pattern, so importing it does not by itself fix this asynchronous timing case. Require an explicit ready time or transmission-completed event; cancel or account for a preempted transmission.

4. One timestamp for the whole target loses independent facts. `LocalCatalogKnowledge.merge_status` rejects an entire incoming update when `(update_time, source)` is older than the target's last update. A reproduced case accepted C's acquisition at time 100, then rejected A's previously unknown delivery at time 90, leaving delivery knowledge absent. Those are different facts and can both be useful. Sender-wide sequence rejection also loses an older packet for target A after a newer packet for target B arrives unless each newer packet is a cumulative replacement snapshot. That is not true of these single-target packets.

5. Message selection is not catalog synchronization. Perfect exchange emits the active imaging target, or a message without a target. Finite exchange falls back to the single most recently updated target. Multiple completions and delivery updates during other modes can remain unshared. Repeatedly transmitting old knowledge with a new `creation_time` also obscures its true age and provenance. Keep original event timestamps, and maintain a queue of unsent/changed service records; periodically reconcile summaries to recover lost deltas.

6. The stated reward-time setting is currently ineffective. `train.py` requests `reward_time="step_start"`, but this branch's `compute_value_targets_time_discounted` has no such argument and discounts the reward at step end. Its condensing logic also sums intermediate rewards without within-action time discounting. These details can change an apparent continuous-versus-conflict advantage because the modes create different action boundaries. Choose and verify a common physical-time reward convention before scientific comparison.

7. The current `intent_conflict_time_s` metric measures whether any sensors are concurrently targeting the same RSO. It does not implement the slide's wasted sensor-time expression: it lacks the distinction between interruption loss, post-completion duplication before receipt, and sensor multiplicity. Keep it as a separate diagnostic and add event-based accounting.

8. Current centralized information is a metadata reference, not the prospectus's full-state reference. Its pending/cooldown features use an aggregated view, but candidate filtering does not consistently consume that same view. Even after fixing this, more information is only an information upper bound: a separately trained policy can perform worse in finite training. Do not label measured reward as a guaranteed optimization upper bound.

**Encode retasking as a decision-set selector.** Keep information mode, retasking mode, and physical communication model as separate configuration axes.

Let `D(t)` be living sensors whose own action has finished, timed out, failed its opportunity, or requires a safety response. Let `C_i(t)` mean that sensor i's current imaging request has become satisfied according to its own catalog after local events and messages available by t. Then:

- Continuous: every living sensing agent belongs to the decision set at each declared decision epoch.
- Conflict: an agent belongs to the decision set if it is in `D(t)` or `C_i(t)` is true.
- Optional diagnostic baseline: only `D(t)`, which corresponds most closely to current behavior.

An unrelated target completion does not interrupt charge, desaturation, downlink, or transmission in conflict mode. Those operations still terminate on their own resource/window/deadline events. A target completion must be for the active request/service requirement, not merely some historical observation of the same RSO.

The boundary order should be: propagate → collect local physical events → account for reward → deliver all messages available now → merge local knowledge → calculate decision sets → invalidate affected candidate caches → produce observations and masks → select and apply actions. Resolve simultaneous physical completions as a batch, so Python iteration order does not award one satellite knowledge before another satellite's simultaneous capture is accounted for.

Retask because a peer completed a target at the receiver's receipt time, not retroactively at the peer's capture time. With delayed messages, receipt can be the next event. A maximum propagation chunk is not automatically a policy decision: either declare it a common decision cadence or continue through it without adding mode-dependent decisions. The slide's globally triggered continuous mode assumes a common notification/decision clock; fully decentralized hardware cannot know that an out-of-contact peer finished without an explicit signal. Label this idealized scheduling assumption and keep it identical across information comparisons.

Reselecting the same task must preserve mid-action physical progress when intended as continuation. Presently `ImageRSO.set_action` disables/recreates the success event and resets hold tracking. Add task identity and an explicit continuation path that retains accumulated hold, controller state, and original deadline. A policy-selected continuation remains a learned decision; `NO_ACTION` means the policy did not decide. Do not conflate the two when condensing transitions. Changed tasks need an interruption reason and cleanup of old events, without deleting real onboard data.

**Time-tagged local catalogs: keep the newest facts, not the newest packet.**

The useful persistent entry is a materialized summary derived from service records. A compact completion record should include:

| Field | Purpose |
|---|---|
| `target_id`, `request_id` or service-generation identifier | Distinguish repeated requests, promotions, and revisits of the same RSO |
| `record_id`, `source_sensor` | Preserve origin and deduplicate the same physical acquisition across retransmission |
| `capture_time` | Epoch of the observation itself |
| Capture/quality/verification status | Distinguish collected imagery from a qualifying fulfilled request |
| `delivery_time` associated with this record, when known | Ground receipt event for this exact acquisition |
| Record version and original event timestamp | Handle revisions and delayed/out-of-order updates |
| Locally recorded `received_at` | Measure information delay without modifying source event time |

Use immutable record identities and monotone facts where applicable: a known qualifying completion remains known; a delivery attaches to its acquisition; repeated receipt is idempotent. Rejections or quality revisions require explicit versions for that record. Preserve local physical product ownership separately. A packet lifetime or lost contact must not erase a confirmed completion. Request expiry/revisit eligibility is a mission rule, separate from transport expiry and peer-contact age.

Keep the latest qualifying acquisition summary and latest delivered-observation summary separately. For information age at the ground, the important quantity is `now - capture_time_of_freshest_qualifying_delivered_product`, not `now - most_recent_downlink_time`. A delayed old image arriving later must not make the catalog look fresher than a newer image already delivered. Do not combine maxima of independent fields into a fictitious product.

Example: A captures at 100, B captures at 120, B's image reaches ground at 130, and A's reaches ground at 150. The freshest delivered observation remains B's capture at 120. The latest delivery event is 150, but the service age at 160 is 40 seconds, not 10 or 60.

For a one-shot request, remove a locally known fulfilled request from eligibility permanently. For persistent maintenance, remove it only for the appropriate revisit interval or service generation; retain its catalog history. Current code starts a qualifying delivery's cooldown at `capture_time + cooldown`, not at delivery time. Preserve that distinction if it is the intended RF II service definition.

The meaning of completed must be explicit. Recommended default for acquisition deconfliction: a qualifying completed exposure satisfies the imaging request, and ground delivery is a separate service milestone. If image quality is genuinely only available after ground verification, share `captured_unverified` as evidence, not as confirmed success. Decide whether temporary suppression while verification is pending is part of the baseline or a separate ablation. A latest-capture field alone cannot distinguish these cases. The current branch verifies on downlink and tracks pending images, so this is a scientific contract change, not just renaming `IntentStatusMessage`.

**Observation proposal.** Retain the established BSK-RL pattern of composable observation objects with explicit property functions, units, and normalization, then use the shared target encoder. Start from the branch's 14 own/environment features and ten geometric/value features per target. Remove `known_teammate_intent`; retain useful own pending/service state, but note that pending/cooldown features are usually zero if such targets have already been filtered out.

For persistent service, prefer normalized age since latest known qualifying acquisition, normalized age of freshest known delivered observation, validity/knownness indicators, and relevant local product state. An unknown age must not be encoded as a fresh observation. Information provenance remains in the catalog, not as arbitrary numeric IDs fed into the network. Policy vector size must be derived from the chosen schema and recorded with checkpoints; existing `14 + 13K` checkpoints do not automatically remain compatible.

At continuous retasking epochs, add own current mode, elapsed/residual action duration, and imaging progress or remaining hold as needed. The policy otherwise cannot distinguish a fresh task from an almost-finished one. Include an active-target relation/continuation option without sharing it to other agents. Own angular rate/pointing context deserves a small ablation because decisions can now happen mid-slew.

Produce exactly one candidate ID snapshot per agent and epoch, with a padding/action mask. Both the observation and the action decoder must use it. Reserve a valid non-imaging action when no imaging candidate is eligible. Preserve an active eligible target or an explicit continue option when it falls outside a freshly sorted top-K list.

Target ephemeris is different from teammate ephemeris. Retaining target relative position/velocity from the RF II model does not imply sharing peer trajectories. Document whether target ephemerides are assumed known exactly. Also keep frame conventions explicit: the branch's `_relative_velocity_H` rotates inertial velocity difference into Hill axes; upstream `v_DC_Hc` additionally subtracts the rotating-frame `omega × r` term. They represent different quantities and should not be substituted solely because both are called Hill-frame relative velocity.

**Three implementation options.** Both retasking modes are available in all three; the options vary research scope and observation/communication fidelity.

| Option | Implementation and observations | What it enables | Prospectus fit and limitation |
|---|---|---|---|
| 1. Minimal completion-sharing experiment | Existing branch; completion records; ideal instantaneous exchange; shared candidate/mask code; conflict and continuous selectors; own/target observations with completion filtering and own action progress | Cleanly validates retasking semantics and quantifies duplicate/interruption tradeoffs before link physics | Fastest credible RF III proof of concept. It isolates information value but does not establish when costly communication is worthwhile. A one-shot subcase should be labeled a diagnostic, not persistent service. |
| 2. Persistent completion catalog with finite communication — recommended | Option 1 plus record-aware acquisition/delivery merge, revisit semantics, reliable delta exchange, finite transmission completion and receive events; target service-age features plus compact own message backlog/last-exchange context | Directly studies service age, duplicates, retasking loss, useful communication timing, delay, and loss while preserving decentralized execution | Best fit to current completion-only request and RF III's central research question. First implement ideal links, then LOS and the prospectus's pointing requirements. Geometry-only broadcast is an intermediate validation model. |
| 3. Completion sharing with explicit link-aware peer observations | Option 2 plus a masked peer-set/attention encoder and recipient-aware communication actions; locally estimated or actually received peer relative geometry, ephemeris age/validity, predicted access, and link feasibility | Studies which peer to contact, pointing/time costs, disconnected constellations, larger agent counts, and later RF IV link degradation | Appropriate when partner selection or directed links become the scientific question. Larger sensing/communication assumptions and training burden. No peer intended targets, private battery/storage, or future actions are necessary by default. |

Option 3 can use preloaded/propagated peer ephemerides with stated uncertainty, or an explicit navigation beacon. If the only exchanged application data must be catalog completions, do not silently append live peer dynamics to that packet; derive geometry from the declared navigation model. Mask missing/stale peer information, use ages, and avoid fixed `sensor_1`, `sensor_2` slots. A Deep Sets/attention representation supports a variable number of peers, unlike the inspection paper's fixed-size input.

Completion sharing cannot prevent every pair of agents from initially choosing the same unfinished request: neither yet has a completed fact to share. That is part of the hypothesis to measure. Intent sharing may reduce those initial conflicts, but excluding it makes the finite-cost completion-sharing study interpretable; it should not be claimed universally useless.

**Upstream reuse and commit recommendation.**

Keep `multi-agent-space-imaging-2026` on its current AMOS lineage for this study. Audit/backport small relevant upstream changes rather than rebasing hundreds of commits immediately:

| Upstream commit | Relevant behavior | Recommendation |
|---|---|---|
| `5020e7ab7eb1e00b8e3e3a2d15f091dd13f1fdf4` | Fixes `ContinuePreviousAction` returning early when one agent has no action entry | Port; current branch still has the early return. |
| `5a22fb456fc449df94db6f671ca1fa69c36e6362` | Adds explicit step-start/step-end reward timing | Port and verify both the function and learner configuration wiring. |
| `0368685380189d5cae5b08c07fc7ccf5e8de4de4`, `1079ef7`, `3f4c4f1` | Final-observation/NO_ACTION condensing fix and exact sentinel regression coverage | Compare against branch-local condensing fixes; bring over missing semantics and tests, preserving accumulated `d_ts`. Do not overwrite blindly. |
| `494df0f5607bb7920fd1a730a28bdcf5a0b7786d` | Directional communication and broadcast interfaces | Reuse interface conventions as appropriate. Extend actual completion timing; retain metadata-only communication rather than generic whole-Data union. |

A later migration can start from a pinned develop revision and port the RSO imaging extensions into the modular dynamics/FSW layout. That is worthwhile for upstream maintenance, not a prerequisite for retasking. None of the reviewed feature branches supplies the missing persistent service semantics wholesale.

**Metrics and validation before training.** Use matched initial conditions, policy capacity, reward accounting, communication opportunity schedule, and simulator horizon. At minimum cross `{conflict, continuous}` with `{independent, ideal completion, finite completion}`. The ideal case should share exactly the completion fields used in the finite case, so the comparison isolates information availability. A separate full-global-state reference can address the older prospectus comparison without being conflated with the ideal completion reference. Report training/evaluation uncertainty across multiple seeds and include the same heuristic under both modes.

Log per attempt: sensor, request, chosen target, start, physical completion/failure, interruption, first competing qualifying completion, first local receipt, and end. Compute the slide's nonduplicate interrupted time and its late duplicate interval `max(0, min(end, received) - max(start, competing_completion))` with disjoint classification. If no message arrives, receipt is effectively infinity for the interval. Sum sensor-time, not merely time during which any conflict exists; specify how failed agents affect the denominator. Report unique/duplicate service per request or revisit cycle, communication duration, and interruption count separately. Persist the event log outside the compact actor catalog so merging old information cannot destroy metric evidence.

For exact physical-time discounting of an action spanning several global events, aggregate rewards as `R_i = sum_k gamma^(t_k - t_start_i) * r_i,k`, then bootstrap with `gamma^(t_end_i - t_start_i)`. If using a chosen start/end approximation instead, document it and ensure event partitioning does not silently create different objectives for the two modes. The slide's communication penalty is per second; the current rewarder applies an optional fixed penalty per broadcast action, so duration-dependent experiments require a corresponding accounting change.

Essential new regressions: unrelated completion preserves a charge deadline; received matching completion retasks only affected imagers; delayed completion is not visible before receipt; all sensors retask in continuous mode; reselected identical task preserves progress; a teammate event cannot deliver a 30-second transmission at 5 seconds; dropped/out-of-order deltas eventually reconcile; old delivery cannot overwrite fresher delivered observation; simultaneous completions receive deterministic nonduplicated credit; promotions/revisits are not blocked by old request completion; zero eligible targets stay masked; observation slot always commands its associated ID; physical-time return is invariant to insertion of irrelevant simulator boundaries.

Validation performed for this review: 11 existing focused tests passed (`test_typed_messages.py`, `test_target_intent_observation.py`, `test_async_discounting.py`). Separate small executable probes reproduced candidate eligibility disagreement, premature broadcast delivery, target-wide merge loss, and the missing reward-time argument. These checks used the actual Python classes/functions with small stub states; they were not a new end-to-end training or orbital performance study. Existing rollout figures cannot establish the revised completion-only hypothesis until these gaps are addressed.

The local prospectus still lists intent, expected value, full peer state, and image relay. The user's latest request deliberately narrows that scope. Update its RF III strategy list and the branch's `PROSPECTUS_MAPPING.md`, `ARCHITECTURE.md`, and `OBSERVATION_VALIDATION.md` when implementing completion-only experiments; preserve intent/relay as optional later comparisons if desired. This narrowing retains the core contribution: persistent catalog maintenance, asynchronous physical execution, and deciding when finite-cost information exchange is valuable.
