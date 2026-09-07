# Prospectus mapping: completion sharing and asynchronous retasking

| Slide concept | Implementation | Meaning |
|---|---|---|
| Sensing-agent set | `SpacecraftRole.SENSING_AGENT`, `env.sensing_satellites` | Only these spacecraft appear in PettingZoo/RLlib |
| Passive RSO targets | `PASSIVE_TARGET`, `BasicTargetDynamicsModel` | Actual propagating spacecraft; no target policy |
| Local time-tagged catalog | `CompletionCatalog.records` and derived target summaries | Own completed facts plus received qualified completions |
| Completion sharing | `CompletionCommunication` | Exposure identity/source/request/capture/completion/optional delivery timestamps; no intent |
| Remove completed targets | `catalog.is_eligible`, `candidate_snapshot` | Suppress matching current request during service cooldown; pad with masked empty slots |
| New request / revisit | `request_epoch_s`, `cooldown_s` | Old service does not satisfy a newer mission request or an expired revisit interval |
| Next event `t[n+1]` | Existing Basilisk terminal events plus queued reception/heartbeat | Minimum enabled boundary, quantized to simulation ticks |
| Decision set `I[n]` | `requires_retasking` and `retasking_mode` | Conflict: own end or known active-target completion; continuous: all surviving sensors |
| Agents retaining previous action | `NO_ACTION`, or policy `ContinueTask` | Preserve FSW progress and original deadline |
| Shielded candidate policy | `GNNModule` completion mask | Eligibility/continue mask; full safety shielding remains future work |
| Own execution state | `ActiveTask`, `CompletionContext` | Mode, elapsed/remaining time, hold progress; never transmitted |
| Observation `o_i` | `26+17K+12P` completion-v2 directed vector | Own resources/progress, local target facts and declared peer contact beacons |
| Selected receiver | `TransmitCompletions`, `action_point_peer` | SimpleNav-driven slew and continuous valid hold; only the selected catalog receives the packet |
| Shared policy | RLlib `imager` module | Parameter-sharing independent PPO, with local critic inputs |
| Time discount | `discount_per_s ** elapsed_seconds` | 45,000-second reward half-life; action-start rewards and 6000-second GAE trace half-life |
| Catalog/acquisition reward | Existing AMOS mixture in `CompletionImageReward` | Unique qualified capture and separate unique ground value |
| Communication penalty | `communication_cost_per_s` | Optional cost on occupied communication task seconds (including slew), default zero |
| Duplicate counts | Private `_TeamServiceAccounting` | Capture attempts and successful delivered duplicates logged separately |
| Duplicate wasted time | `coordination_metrics().duplicate_sensor_time_s` | Work after another qualifying completion and before learning about it or ending |
| Nonduplicate interruption time | `interrupted_nonduplicate_sensor_time_s` | Elapsed image-task duration on policy switch, excluding duplicate intervals |
| Wasted-time fraction | `wasted_time_fraction` | Disjoint waste divided by sensing-agent count × elapsed episode time |
| Total constellation reward | Sum of `cumulative_reward` over sensors | Avoids multiplying each unique priority by number of agents |
| Physical spacecraft visualization | All spacecraft in Vizard `scList` | Rendering does not alter agent membership or dynamics |

The three supplied slides are requirements/context for this implementation. Completion
means a locally qualified finished exposure; ground delivery is tracked separately.
Packet lifetime and knowledge lifetime are distinct. Full exposure history is retained
for the bounded episode while actor summaries use the most recent relevant information
*per fact*, not the last packet for the entire target.

The next empirical question is whether completion sharing reduces the slide's duplicate
and interrupted imaging time enough to offset communication time and policy reevaluation.
The implementation provides the six matched experiment cells and event evidence needed
to answer it, without adding intention sharing or private peer-state observations.
