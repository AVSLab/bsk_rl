# Prospectus symbol-to-code mapping

| Prospectus concept/symbol | Code location | Operational meaning |
|---|---|---|
| Sensing-agent set $\mathcal{S}$ | `SpacecraftRole.SENSING_AGENT`; `env.sensing_satellites` | Spacecraft exposed to PettingZoo and the shared policy |
| Passive RSO set $\mathcal{T}$ | `SpacecraftRole.PASSIVE_TARGET`; `env.passive_satellites` | Propagated spacecraft excluded from learning |
| Local catalog $K_i$ | `sensor.data_store.catalog` | Sensor-$i$ acquisition, delivery, pending, and cooldown knowledge |
| Physical store $B_i$ | Sensor Basilisk storage; `sensor.data_store.products` metadata | Image products physically owned by sensor $i$ |
| Global team truth $L$ | `MultiSensorRSOTargetImageReward._team_accounting` | Private non-double-counted service and duplicate accounting |
| Product $p$ | `ImageProductRecord` | Provenance-preserving image record |
| Local observation $o_i$ | `SensorSatellite.observation_spec` | Own resources/environment and target-wise features available to sensor $i$ |
| Candidate set $\mathcal{C}_i(t)$ | `PolarisScTargetProperties`; `ImageRSO` | Sensor-local eligible target candidates at a decision epoch |
| Same-target coordination $c_{ij}$ | `known_teammate_intent` | Freshness-weighted fraction of known peers targeting candidate $j$ |
| Shared policy $\pi_\theta$ | `GNNModule`; RLlib module `imager` | One target-wise attention module used by every sensing agent |
| Variable action duration $\Delta t_i$ | `requires_retasking`, `NO_ACTION`, accumulated `d_ts` | Elapsed duration attributed to sensor $i$'s action |
| Per-second discount $\gamma^{\Delta t_i}$ | `TimeDiscountedGAEPPOTorchLearner`, `gamma=0.999` | AMOS semi-Markov discount convention |
| Image reward weight $1-\alpha$ | `MultiSensorRSOTargetImageReward.alpha` | Priority-weighted qualifying acquisition term |
| Ground reward weight $\alpha$ | `MultiSensorRSOTargetImageReward.alpha` | Unique delivered-ground-value term |
| Intent/status message $m_{i\rightarrow j}$ | `IntentStatusMessage`; `DirectedMessage` | Directional compact target metadata, never full datastore data |
| Perfect metadata channel | `PerfectMetadataChannel` | Semantic-validation case without link impairment |
| Broadcast action $a_i^b$ | `BroadcastIntent` | Finite-duration opportunity cost and sender enable |
| LOS broadcast graph $G(t)$ | `IntentStatusCommunication._geometric_directional_pairs` | Earth-occlusion-only directional connectivity |
| Centralized-information upper bound | `CentralizedInformationView` | Ideal read-only aggregation of sensor-local metadata |
| Unique team value | `rewarder.team_summary["team_value"]` | Agent credits summed once, without replicated full-team reward |
| Unique team acquisitions | `rewarder.team_summary["unique_acquisition_count"]` | Non-double-counted qualifying captures |
| Duplicate attempt/success | `duplicate_attempt_count`; `successful_duplicate_count` | Separate acquisition-attempt and delivered-quality diagnostics |
