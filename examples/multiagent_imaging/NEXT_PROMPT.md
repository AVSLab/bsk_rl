# Suggested next prompt

Work only in `/Users/dahu1128/Repositories/bsk_rl-multi-agent-space-imaging-2026` on
branch `multi-agent-space-imaging-2026`. Do not modify, switch, reset, stash, or clean the
original `/Users/dahu1128/Repositories/bsk_rl` AMOS worktree. Do not create a branch with
`codex` in its name.

First verify that the worktree is clean and that `HEAD` exactly matches
`origin/multi-agent-space-imaging-2026`. Read `ARCHITECTURE.md`,
`OBSERVATION_VALIDATION.md`, and `TEST_RESULTS.md` before changing code.

## Goal

Implement the next bounded evaluation phase: run one frozen target-wise `imager` module
across one, two, and three sensing spacecraft and verify that the evaluation path respects
asynchronous decisions, explicit roles, local knowledge, and the revised `14 + 13K`
observation contract. Do not submit cluster jobs, run a full Monte Carlo, or start long
training.

1. Audit the existing AMOS 2026 checkpoint-loading/evaluation code and
   `examples/multiagent_imaging/evaluate.py`. List and remove evaluator assumptions about
   one sensor, `SS1`, one action stream, or one local datastore. Do not broadly rewrite the
   validated environment.
2. Add a direct RLlib `RLModule.from_checkpoint` loader for the shared module named
   `imager`. Use one loaded module instance for every explicit sensing agent. Passive RSO
   spacecraft must never reach policy mapping or inference.
3. Invoke inference only for a sensor whose `requires_retasking` flag is true. Use
   `NO_ACTION` for a sensor continuing its previous command. Preserve per-sensor elapsed
   `d_ts` and the AMOS per-second discount convention.
4. Save a checkpoint contract beside smoke checkpoints containing at least: module name,
   global feature count 14, per-target feature count 13, candidate count `K`, action count
   `4 + K`, `condition_on_spacecraft`, information case, and training sensor count. Reject
   incompatible checkpoints with a clear error; never silently reshape an AMOS
   single-sensor policy.
5. Add configuration support for an explicit ordered list of sensing-orbit specifications
   (classical elements or a documented equivalent). The environment may enumerate this
   list to construct spacecraft, but core role, reward, data, and observation logic must
   not infer behavior from a sensor name or list position. Preserve the current distinct
   two-orbit defaults.
6. Produce one tiny shared-policy smoke checkpoint locally if no compatible checkpoint is
   available. Keep training bounded to the minimum needed to exercise save/load and
   inference; do not use its return as scientific evidence.
7. Run paired frozen-policy rollouts for one, two, and three sensors using identical seed,
   target catalog, priorities, reward settings, cooldown, candidate count, and horizon.
   Start with the `independent` case so variable sensor count is isolated from
   communication. Add one two-sensor `centralized_information` rollout only as a mechanics
   check, not a performance comparison.
8. Save per-sensor actions, command durations, rewards, battery/storage/wheel histories,
   local catalogs, received messages, and product ownership. Save non-double-counted team
   acquisitions, deliveries, value, duplicates, target conflicts, broadcast time, and
   message ages from reward/evaluation diagnostics. Never expose global accounting to the
   actor.
9. Add tests for one shared module instance, explicit sensing-role-only inference,
   one/two/three-sensor rollout compatibility, deterministic paired initial conditions,
   asynchronous continuation, checkpoint-contract rejection, and unchanged single-sensor
   AMOS behavior.
10. Do not add back generic teammate position, velocity, resource, wheel, current-action,
    or team-count inputs. If the rollouts reveal a concrete coordination failure that the
    same-RSO intent/cooldown/pending fields cannot represent, document the evidence and
    propose a masked teammate-set encoder without implementing it in this phase.
11. Run focused tests, the full unit regression, the bounded integration regression, Ruff,
    `git diff --check`, and the AMOS runtime validation. Update the architecture and test
    records with exact commands and results.
12. Commit the work in sensible commits and push the same branch. Report commit SHAs,
    output/checkpoint locations, limitations of the smoke policy, and provide the next
    detailed prompt for authorizing a small local learning comparison before any cluster
    campaign.
