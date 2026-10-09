"""Derive safe 24h collection cutoff from approved new-source compute readiness."""
import argparse,json,math
from pathlib import Path
from examples.multiagent_imaging.cluster.study_gate import source_gate

def budget(acceptance,arm,wall_s=86400):
    if not acceptance.get('passed'):raise ValueError('Readiness did not pass')
    if acceptance['source_gate']!=source_gate():raise ValueError('Readiness source/config differs')
    timing=acceptance['measured_throughput'][arm]
    # Eight frozen missions plus a conservative update overrun/restore/shutdown
    # reserve. These are planning estimates, never proof of future throughput.
    evaluation_reserve=math.ceil(8*timing['max_heldout_wall_s']*2+600)
    update_reserve=math.ceil(timing['update_wall_s']*2+timing['startup_wall_s']+900)
    cutoff=wall_s-evaluation_reserve-update_reserve
    if cutoff<3600:raise ValueError('24h resources leave insufficient safe training budget')
    return dict(training_start_update_cutoff_s=cutoff,evaluation_reserve_s=evaluation_reserve,
        update_shutdown_reserve_s=update_reserve,wall_s=wall_s,
        soft_true_decision_target=200000,hard_new_sampled_decision_cap=260000,
        max_updates=200,max_training_episodes=600,
        matching='Common true-decision axes/milestones. Crossings have recorded overshoot, not exact matched experience.')

def parallel_budget(acceptance, arm, wall_s=86400):
    plans = {name: budget(acceptance, name, wall_s) for name in ('catalog_tx', 'geometry_only')}
    resources = acceptance.get('launcher_resources', {})
    if resources.get('num_env_runners') != 28 or resources.get('ray_cpus') != 32 or resources.get('torch_threads') != 1:
        raise ValueError('Need same-release measured 28-runner/32-CPU readiness; one-runner timing is insufficient')
    timing = acceptance['measured_throughput']
    # Common exposure budget from the slower measured arm, with conservative
    # headroom. This is a planning estimate, not guaranteed future throughput.
    rates = [timing[n]['true_decisions']/timing[n]['update_wall_s'] for n in plans]
    cutoff = min(p['training_start_update_cutoff_s'] for p in plans.values())
    target = int(.65 * min(rates) * cutoff // 50000) * 50000
    if target < 50000:
        raise ValueError('Measured resources cannot support one common 50k decision milestone safely')
    largest_batch = max(timing[n]['true_decisions'] for n in plans)
    smallest_batch = min(timing[n]['true_decisions'] for n in plans)
    hard = target + 2 * largest_batch
    updates = math.ceil(hard / smallest_batch) + 2
    episodes = updates * max(timing[n]['training_episodes'] for n in plans) * 2
    return dict(plans[arm], soft_true_decision_target=target,
                hard_new_sampled_decision_cap=hard, max_updates=updates,
                max_training_episodes=episodes, measured_launcher=resources,
                planning_safety_factor=.65, target_derivation='65% of slower observed true-decisions/s times common safe collection cutoff; rounded down to 50k',
                training_start_update_cutoff_s=cutoff)

def main():
    p=argparse.ArgumentParser();p.add_argument('--readiness',type=Path,required=True);p.add_argument('--arm',choices=['catalog_tx','geometry_only'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--parallel',action='store_true');a=p.parse_args()
    record=(parallel_budget if a.parallel else budget)(json.loads(a.readiness.read_text()),a.arm)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(record,indent=2)+'\n');print(record['training_start_update_cutoff_s'])
if __name__=='__main__':main()
