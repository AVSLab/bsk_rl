# Mixed-trained priority-response evaluation with 200 targets

## Scientific comparison

- Policy: mixed-regime-trained graph-attention policy with downlink reward weight `alpha=0.1`
- Training catalog: exactly 50 LEO, 30 MEO, and 20 GEO targets
- Evaluation catalog: exactly 100 LEO, 60 MEO, and 40 GEO targets
- Episode duration: 45,000 s
- Monte Carlo seeds: 0 through 99
- Candidate list: 10 targets per decision epoch
- Priority event: first decision after the episode midpoint
- Tracked targets: five HIOs, three SHIOs, and eight unpromoted controls

## Recommended priority construction

The initial priorities are drawn uniformly and rescaled to sum to 200. The initial mean priority is therefore exactly 1, matching the 100-target training catalog whose priorities sum to 100. At the online event, HIO priority becomes 5 and SHIO priority becomes 10. Other targets retain their initial values and the catalog is not renormalized.

This mean-normalized construction is preferred over scaling from the sampled catalog maximum. A sample maximum changes across seeds and catalog sizes. With the current initial distribution it is also close to 2, so a 5x/10x maximum rule would produce promoted priorities near 10/20. Those magnitudes were not used during training and would confound catalog-density generalization with a new priority scale.

## Submission

Run on the Mac:

```bash
cd /Users/dahu1128/Repositories/bsk_rl
AMOS2026_CLUSTER_HOST=dahu1128@login-ci5.rc.colorado.edu \
  docs/amos_2026/sync_and_submit_mixed_trained_priority_response_200.sh
```

The launcher submits ten independent jobs. Each job evaluates ten seeds sequentially. There are no Slurm dependencies.

## Required outputs

Every seed must contain:

- `mc_status.json`
- `priority_response_targets.csv`
- `verified_deliveries.csv`
- `metrics_*.json`

After all seeds complete, run `analyze_gat_priority_response_mc.py` on Alpine. The final Figure 5 should use the cumulative successful-image response and exact packet-matched delivery fields from these outputs.
