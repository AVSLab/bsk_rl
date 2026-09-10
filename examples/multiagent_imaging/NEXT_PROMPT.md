# Next task prompt — three-sensor baseline preflight

Continue from the current remote HEAD of `multi-agent-space-imaging-2026`. Read
`examples/multiagent_imaging/BASELINE_MONTE_CARLO.md`, `cluster/README.md`, and
`cluster/EXECUTION.md` first. Pull the reviewed branch into the existing separate
cluster checkout at
`/projects/dahu1128/bsk_rl-multi-agent-space-imaging-2026`; use
`/projects/dahu1128/.venv-completion-v2/bin/python` and the already built separate
Basilisk runtime. Preserve the AMOS checkout and environment unchanged.

I authorize preparation and submission of only the new three-sensor matched LEO
seed-zero baseline pair, array tasks 0 and 100. Do not submit the other 198 tasks,
four-worker learning, or the six-cell learned-policy study.

Use campaign schema `three-sensor-full-state-baselines-v2`: three sensing agents,
100 passive Basilisk/Vizard RSO spacecraft outside the PettingZoo agent list, ten
candidates, 45,000-second complete episodes, conflict retasking, and seeds 0–49 in
each eventual information/environment cell. Preserve the existing
`reimage_cooldown_orbits=2.0` behavior, anchored at qualified capture time. Verify
and report its actual derived seconds for the 700/800/700-km sensing team. Treat
full ground delivery as the ground-confirmed coverage boundary; do not silently
move the cooldown anchor to delivery.

The independent controller must read only each sensor's own resources, products,
catalog, and declared target ephemerides. The centralized-full-state controller
must read every live sensor's position, velocity, attitude/rate, battery, storage,
wheel state, active task/reservation, physical onboard products and owners, request
epochs, and durable time-tagged capture/completion/delivery catalog at every
asynchronous decision boundary. It must jointly assign current actions and prevent
same-target assignments or fresh/in-progress target duplication. Keep it explicitly
labeled a maximum-information coordination reference: its greedy scheduler is not
a proof of globally optimal future coverage. Neither baseline may select broadcast
or directed transmission, and neither baseline trains or restores a policy.

Generate a fresh manifest under a new `baseline-mc-3sensor-v2` results directory;
do not reuse or overwrite the completed two-sensor v1 tasks. Run the runtime and
support-data audits, submit only tasks 0 and 100, wait for both to finish, and verify
matched initial-condition hashes, all three sensors present, 100 passive targets,
horizon/resource status, zero radio actions/occupancy, and the centralized audit's
one full-team read per decision boundary.

Report constellation-union qualified capture coverage and fully ground-delivered
coverage. Report existing duplicate attempts and wasted sensor-seconds plus both
new definitions: (1) qualified ground-delivered products whose capture time is
older than a newer product of the same target also delivered by another sensor,
including the causally known-at-delivery subset; and (2) cross-sensor onboard
same-target overlap, including affected targets/products, redundant acquisition
count, and integrated excess holder sensor-seconds. Generate the partial coverage
and duplicate-work plots for this matched pair, label them as one-seed preflight
evidence with no uncertainty, record Slurm elapsed/MaxRSS/AllocCPUS, update
`cluster/EXECUTION.md`, commit and push the evidence, and recommend whether the
remaining 198 three-sensor episodes should be authorized.
