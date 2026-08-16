# AMOS 2025 provenance and architecture audit

This audit was performed before implementation. The working tree already contained
unrelated AMOS 2026 changes; they were preserved. Research Focus I code is isolated in
`examples/prospectus_rfi/` plus its corresponding tests.

## Historical code line

- Closest frozen late-summer AMOS 2025 training configuration: commit
  `d0bcc54c6610643cc946ce92f2ea30314659fe0e` on the historical
  `IA_Polaris_imaging_june10` line.
- Current local tip named `IA_Polaris_imaging_june10`:
  `2c32a6ccf95d49489c1afee31adc5a4ec460aeae`. It moved after the 2025 work and is
  therefore not itself a frozen AMOS-paper revision.
- Public space-to-space surveillance snapshot branch:
  `space-to-space-imaging-public`, commit
  `c6b9e4310a36476944cfa27b1d02e43c34362952`.
- Broader public journal snapshot: `amos-journal-public`, commit
  `38022cff03be3207066d1f29bf235ea6487ed39c` (based on
  `07e0bf71529c59f68e8be2910313659b0bdadf8e`).
- New study branch: `amos2025-architecture-comparison`, created from
  committed AMOS 2026 branch tip `94ad6cfb1a25460eb886b29bbdceb7c1ee683d7c`.
  This newer base is required for the maintained Ray 2.35 RLModule, fixed-duration
  action, lifecycle, and attention APIs. The physical environment is reconstructed
  explicitly from the AMOS 2025 values below instead of inheriting AMOS 2026 defaults.

## Historical scripts and configuration

The relevant historical files are:

- `examples/updated_train_Polaris.py`
- `examples/train_Polaris.py`
- `examples/Polaris_environment.py`
- `examples/policy_evaluation.py`
- `examples/updated_policy_evaluation.py`
- `examples/batch_heuristic_evaluation.py`
- public evaluator `examples/space_to_space_imaging_evaluation.py` in the public snapshot

At `d0bcc54...`, the trainer used 100 total targets, 10 presented candidates,
`total_time = 1.5 * 100 * 300 = 45,000 s`, and actions in this order:

1. `ImageRSO(n_ahead_image=10, duration=300)`
2. `Charge(duration=300)`
3. `Downlink(duration=300)` (some evaluation variants used 180 s)
4. `Desat(duration=150)`

The scanner was an `AccessSatellite` with `ImagingSCDynModel` and
`ImagingSCFSWModel`. Each target used `BasicTargetDynamicsModel`,
`BasicTargetFSWModel`, a scalar time observation, and one episode-length drift action.

The frozen `d0bcc54...` trainer source shows a flattened NumPy vector with:

- five spacecraft values: storage fraction, battery fraction, three wheel-speed fractions;
- seven values per presented target: elevation, relative Hill-frame position (three),
  pointing angle, distance, and shadow factor;
- two eclipse-window values;
- four ground-station window values (open/close for two opportunities).

The selected August 13 policy artifact is a nearby but distinct `obs2` run. Its saved
run parameters and RLModule constructor are the authority for policy transfer. They
record an 87-value observation: the same five spacecraft values, ten seven-value target
rows, two eclipse values, and ten ground-station values (open/close for five
opportunities). Its target/eclipsing normalization is also different from the later
source snapshot: elevation and pointing angle use norm 1, Hill position and distance use
1,596,000 m, and eclipse uses norm 1. The transfer evaluator reconstructs this exact
field order and normalization and retains the historical repeated-target padding with no
validity mask. It does not feed the new 91-value masked K=10 observation into the old
network.

Research Focus I preserves these physical fields but groups the 11 global values before
the target rows and appends one `valid_target` bit to each target row. For the new
candidate sweep, the flattened size is `11 + 8K` and the discrete action count is
`K + 3`, where `K` is 5, 10, or 20. Total catalog size N remains independent of K and
is sampled uniformly from 100 through 400 at every training reset.

Historical scanner resources were 200 Mbit storage (`50 * 4e6`), 4 Mbit/s image and
downlink rates, 1.8 MJ battery, -10/-30/-25/-80 W base/instrument/transmitter/thruster
loads, 1 m² panel, 6000 rpm wheel limit, initial wheel speeds uniform in
[-500, 500] rpm, zero disturbance torque, and 0.01 rad image attitude error. The old
trainer sampled initial battery from 10–40%; this requested study changes that factor to
20–60%.

The exact historical LEO target distribution is semi-major axis uniform from 6871 to
8371 km, eccentricity uniform from 0 to 0.02 subject to perigee >= 6771 km, inclination
uniform from 0 to 180 degrees, and RAAN, argument of periapsis, and true anomaly each
uniform from 0 to 360 degrees.

## Reward interpretation

The implemented split is

`R = (1-alpha) * illuminated unique observation value + alpha * useful downlink value + operational penalties`.

Therefore `alpha = 0` is observation-only reward. It is a scalar reward weight and has
no relationship to the AlphaZero algorithm. An infinite re-imaging cooldown reproduces
the 2025 one-image-per-target behavior.

## Architectures

### Fixed-input monolithic MLP

The AMOS 2025 RLlib PPO family used the standard `PPOTorchRLModule` with separate actor
and value networks (`vf_share_layers=False`). The later historical source snapshot used
`fcnet_hiddens=[2048, 2048]` with ReLU, which is the starting point for the new
fixed-input architecture study. It consumed all presented target features in one fixed
flattened vector and emitted one fixed logit per presented target plus the non-target
actions. In this study, its input and output widths therefore vary between the K=5,
K=10, and K=20 runs. Invalid rows are zeroed and invalid action logits are masked.

The archived public evaluator points to the August 13
`wGAE_balance0d100i_largepenalties_smallbatch_obs2` policy family; the locally archived
best checkpoint is iteration 427. Inspection of that checkpoint itself shows a
`PPOTorchRLModule` with 87 inputs, 13 actions, separate `[1024, 1024]` actor and critic
MLPs, tanh activations, and 2,293,774 trainable parameters. Its inspector
`module_state.pt` SHA-256 is
`6db5bcd4fda20205977dfab377441f625051ef9e9dfaebde5e8db5ec1ab0e2c4`.

The corresponding saved run parameters record N=100, K=10, a 300 s image action,
300 s charge, 180 s downlink, 150 s desaturation, alpha=0 observation reward, and
10--40% initial battery. Evaluating it in the Research Focus I environment deliberately
changes image duration to 100 s, downlink to 300 s, battery initialization to 20--60%,
and N to 100/200/400. Results from this campaign are therefore an out-of-distribution
transfer baseline, not evidence for a policy trained under the new configuration.

### Target-set attention policy

The newer source first appears as `examples/gat_module_complete.py` at commit
`7b09363ef210b0f7de190ec7c0b40c082aad53ef`; the full-action trainer appears at commit
`c66cf69de4bb3b4fad292664f20086263b83a68b`.

Despite its historical filename, this implementation does not construct explicit graph
nodes/edges, an adjacency structure, or a graph message-passing operator. It applies a
shared target encoder, scaled-dot-product self-attention, target/global attention and
permutation-consistent mean/max aggregation, then emits one logit per target. This study
therefore calls it a **target-set attention policy**, not a GAT or GNN. Its trainable
dimensions do not depend on K, although its learned behavior and compute cost can still
differ across K=5, 10, and 20.

## Heuristic and checkpoint format

The historical `ImageRSO` angle heuristic prioritizes visible, eligible targets and
selects the smallest current pointing error, with eligible and then known-target
fallbacks. The historical version can inspect the full eligible catalog. The study also
implements an information-matched version restricted to the same K candidates as the
learned policies. Both use the same battery/storage/wheel resource shield at evaluation.

Archived policies use RLlib checkpoint format 2.0 (Ray 2.35 locally): algorithm state,
environment/connector state, learner state, and per-module PyTorch state. The inspector
module is stored below
`learner_group/learner/rl_module/inspector/module_state.pt` and is loadable with
`RLModule.from_checkpoint(...)`.
