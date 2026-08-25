# Prospectus symbol-to-code mapping

| Prospectus concept/symbol | Code location | Operational meaning |
|---|---|---|
| Sensing-agent set $\mathcal{S}$ | `SpacecraftRole.SENSING_AGENT`; `env.sensing_satellites` | Spacecraft exposed to PettingZoo and the shared policy |
| Passive RSO set $\mathcal{T}$ | `SpacecraftRole.PASSIVE_TARGET`; `env.passive_satellites` | Propagated Basilisk spacecraft excluded from learning |
| Local catalog $K_i$ | `LocalCatalogKnowledge` | Sensor-$i$ acquisition, delivery, pending, and cooldown knowledge |
| Physical store $B_i$ | `SensorProductStore` | Image products physically onboard sensor $i$ |
| Team truth/ledger $L$ | `TeamServiceLedger` | Non-double-counted service and duplicate accounting |
| Product $p$ | `ImageProductRecord` | Provenance-preserving captured/delivered image record |
| Local observation $o_i$ | `SensorSatellite.observation_spec` | Resources, geometry, team context, and target-wise features available to sensor $i$ |
| Teammate set $\mathcal{N}_i$ | `TeammateSetSummary`; `pool_teammate_statuses` | Available peer metadata summarized with permutation-invariant mean/max/action pooling |
| Teammate context $z_i$ | `TEAMMATE_SUMMARY_KEYS` | 24 pooled features with fixed shape for any sensor count |
| Candidate set $\mathcal{C}_i(t)$ | `PolarisScTargetProperties`; `ImageRSO` | Sensor-local eligible target candidates at a decision epoch |
| Shared policy $\pi_\theta$ | `GNNModule`; `shared_sensor_policy` | One target-wise attention module used by all sensing agents |
| Variable action duration $\Delta t_i$ | `requires_retasking`, `NO_ACTION`, connector-accumulated `d_ts` | Elapsed duration attributed to sensor $i$'s prior action |
| Per-second discount $\gamma^{\Delta t_i}$ | `TimeDiscountedGAEPPOTorchLearner` with `gamma=0.999` | AMOS semi-Markov discount convention |
| Image reward weight $1-\alpha$ | `MultiSensorRSOTargetImageReward.alpha` | Priority-weighted useful acquisition term |
| Ground reward weight $\alpha$ | `MultiSensorRSOTargetImageReward.alpha` | Unique delivered-ground-value term |
| Intent/status message $m_{i\rightarrow j}$ | `IntentStatusMessage`, `DirectedMessage` | Explicit directional metadata, never full datastore data |
| Perfect metadata channel | `PerfectMetadataChannel` | First semantic validation case with no link impairment |
| Broadcast action $a_i^b$ | `BroadcastIntent` | Finite-duration opportunity cost and sender enable |
| LOS broadcast graph $G(t)$ | `IntentStatusCommunication._geometric_directional_pairs` | Initial Earth-occlusion-only directional connectivity |
| Centralized-information upper bound | `CentralizedInformationView` | Ideal read-only newest metadata from every sensor |
| Unique team value | `TeamServiceLedger.team_value` | Agent credits summed once, without replicated full-team reward |
| Unique team acquisitions | `TeamServiceLedger.unique_acquisition_count` | Non-double-counted qualifying captures before ground delivery |
| Duplicate attempt/success | `duplicate_attempt_count`; `successful_duplicate_count` | Separate acquisition-attempt and delivered-quality diagnostics |
