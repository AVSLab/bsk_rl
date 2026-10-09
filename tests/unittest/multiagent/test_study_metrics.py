"""No Ray initialization, mission sampling, or optimizer work."""
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

from examples.multiagent_imaging import study_metrics as m

def product(name,sensor,capture,completion,delivery=None,epoch=0,quality=1):
    return dict(record_id=name,source_sensor=sensor,storage_owner=sensor,target_id=3,
                capture_time=capture,completion_time=completion,delivery_time=delivery,
                request_epoch=epoch,quality=quality)

def task(sensor,start,end,epoch=0):
    return dict(sensor=sensor,start=start,end=end,mode='image',target_id=3,request_epoch=epoch)

def test_union_and_two_competitors_not_counted_twice():
    ps=[product('a','sensor_0',5,10),product('b','sensor_1',10,20)]
    ts=[task('sensor_2',0,90)]
    r=m.execution_redundancy(ps,ts,50,['sensor_2'],100)
    assert r['total_sensor_s']==50  # union[10,55) U[20,60)
    assert m.union_duration([(0,10),(5,15),(15,20)])==20

def test_expiry_request_epoch_quality_and_receipt_time():
    ps=[product('a','sensor_0',0,10),product('bad','sensor_0',30,40,quality=.4)]
    r=m.execution_redundancy(ps,[task('sensor_1',0,100)],50,['sensor_1'],100,{'sensor_1':{'a':25}})
    assert r['total_sensor_s']==40 and r['unaware_sensor_s']==15
    assert m.execution_redundancy(ps,[task('sensor_1',50,100)],50,['sensor_1'],100)['total_sensor_s']==0
    assert m.execution_redundancy(ps,[task('sensor_1',20,80,epoch=15)],50,['sensor_1'],100)['total_sensor_s']==0

def test_pending_duplicate_and_delivery_boundary():
    ps=[product('a','sensor_0',0,10,30),product('b','sensor_1',15,20,40)]
    r=m.pending_duplicates(ps,100,100)
    assert r['excess_completed_acquisitions']==1
    assert r['participating_product_count']==2 and r['overlap_target_s']==10
    ps[1]['completion_time']=30
    assert m.pending_duplicates(ps,100,100)['excess_completed_acquisitions']==0

def test_simultaneous_permutation_and_legal_revisit():
    ps=[product('a','sensor_0',5,10),product('b','sensor_1',5,10),product('c','sensor_2',5,10)]
    r=m.pending_duplicates(ps,20,100)
    assert r['excess_completed_acquisitions']==2
    assert r==m.pending_duplicates(list(reversed(ps)),20,100)
    legal=[product('a','sensor_0',0,10),product('b','sensor_1',25,30)]
    assert m.pending_duplicates(legal,20,100)['excess_completed_acquisitions']==0
    assert m.pending_duplicates([ps[0],dict(ps[1],request_epoch=5)],20,100)['excess_completed_acquisitions']==0

def test_physical_identity_and_episode_statelessness():
    p=product('a','sensor_0',5,10)
    delivered=dict(p,delivery_time=20)
    assert m.products_with_deliveries([p],[delivered],100)[0]['delivery_time']==20
    assert p['delivery_time'] is None
    assert m.pending_duplicates([],100,100)['excess_completed_acquisitions']==0
    with pytest.raises(ValueError):m.products_with_deliveries([p],[delivered,delivered],100)
    with pytest.raises(ValueError):m.products_with_deliveries([p,p],[],100)

def test_actual_catalog_reward_and_channel_reset_without_simulation():
    from bsk_rl.comm.completion_communication import CompletionCommunication
    from bsk_rl.data.completion_catalog import CompletionCatalog,CompletionRecord
    from bsk_rl.data.completion_reward import CompletionImageReward
    from bsk_rl.sats.roles import SpacecraftRole
    import numpy as np
    sensors=[NS(name=f'sensor_{i}',role=SpacecraftRole.SENSING_AGENT,
                data_store=NS(catalog=CompletionCatalog(f'sensor_{i}',[3],cooldown_s=100))) for i in range(2)]
    fact=CompletionRecord('old','sensor_0',3,0,10,20,True)
    sensors[0].data_store.catalog.merge_record(fact,20)
    channel=CompletionCommunication();channel.link_satellites(sensors)
    channel.reset_overwrite_previous();channel.reset_post_sim_init()
    channel.pending.append('old');channel.transmissions['sensor_0']='old'
    channel.acknowledged['link']={'old':2};channel.sequence=3
    channel.delivery_history.append('old');channel.transmission_history.append('old')
    channel.last_exchange['sensor_0']=20;channel.peer_exchange['link']=20
    for sensor in sensors: sensor.data_store.catalog=CompletionCatalog(sensor.name,[3],cooldown_s=100)
    channel.reset_overwrite_previous();channel.reset_post_sim_init()
    assert not any([channel.pending,channel.transmissions,channel.acknowledged,channel.sequence,
                    channel.delivery_history,channel.transmission_history,channel.last_exchange,channel.peer_exchange])
    assert all(not c.records and not c.received_at and not c.version_received_at and c.request_epochs=={3:0} for c in channel.catalogs.values())
    reward=CompletionImageReward();reward.cum_reward={'sensor_0':4}
    reward.per_sensor_metrics={'sensor_0':{'acquisition_credit':5}};reward.team_summary={'old':7}
    reward.old_storage_by_sensor={'sensor_0':np.array([1])};reward._team_accounting='old';reward.reimage_cooldown_s=50
    reward.reset_overwrite_previous()
    assert not reward.cum_reward and not reward.per_sensor_metrics and not reward.team_summary and not reward.old_storage_by_sensor
    assert reward._team_accounting is None and reward.reimage_cooldown_s is None

def test_actual_env_reset_clears_episode_counters_but_keeps_seed_stream(monkeypatch):
    from bsk_rl.completion_gym import CompletionConstellationTasking
    from bsk_rl.gym import SensingAgentConstellationTasking
    env=object.__new__(CompletionConstellationTasking)
    sensors=[NS(name=f'sensor_{i}',completion_task='old',completion_candidates='old',completion_peers='old',_active_completion_transmit='old') for i in range(4)]
    env._sensing_satellites=sensors;env.default_seed=100;env._episode_seed_index=5
    env.task_history=['old'];env.decision_counts={'sensor_0':100};env.communication_time={'sensor_0':30}
    seeds=[]
    monkeypatch.setattr(SensingAgentConstellationTasking,'reset',lambda self,seed=None,options=None:seeds.append(seed))
    env.reset();env.decision_counts['sensor_0']=25;env.reset()
    assert seeds==[105,106] and env._episode_seed_index==7
    assert not env.task_history and all(x==0 for x in env.decision_counts.values())
    assert all(x==0 for x in env.communication_time.values())
    assert all(not x for x in env.action_selection_counts.values())
    assert all(s.completion_task is None and s.completion_candidates is None and s.completion_peers is None and s._active_completion_transmit is None for s in sensors)

def test_existing_geometry_is_absent_without_transmit_and_without_delta():
    from bsk_rl.obs.completion_observations import peer_snapshot
    sensors=[NS(name=f'sensor_{i}',simulator=NS(sim_time=20),completion_peers=None) for i in range(2)]
    channel=NS(information_case='independent',sensors=sensors,receivers=lambda s:{'sensor_1'},delta=lambda *a:('payload',))
    sensors[0].completion_communicator=channel
    assert peer_snapshot(sensors[0]).peers==(None,)
    channel.information_case='completion';channel.delta=lambda *a:()
    sensors[0].completion_peers=None
    assert peer_snapshot(sensors[0]).peers==(None,)

def metric_row(seed=1,horizon=100):
    ps=[product('a','sensor_0',5,10,20),product('b','sensor_1',15,20)]
    return dict(simulated_seconds=horizon,policy_decisions={'sensor_0':5,'sensor_1':6},
        capture_products=ps,delivered_products=[dict(product=ps[0],credited_value=1,unique_service=True)],
        agent_live_until_s={'sensor_0':horizon,'sensor_1':50},first_receipts={'sensor_1':{'a':25}},
        coordination={'task_history':[dict(task('sensor_1',0,40),reason='action_end')]},reimage_cooldown_s=100,
        coverage={'catalog_target_count':200,'ground_delivery_coverage_fraction':.005},
        reward_by_agent={'sensor_0':3,'sensor_1':2},constellation_reward=5,
        team_summary={'unique_service_count':1},lifetime_episode_id=str(seed),resources={},
        communication_summary={'packet_outcome_counts':{}},acknowledged_versions_by_link={},action_selection_counts={})

def test_failed_agent_live_time_union_denominator_and_censoring():
    row=metric_row();v=m.mission_metrics(row,45000)
    assert not v['reached_horizon']
    assert v['execution_redundancy']['available_live_sensor_s']==150
    assert v['execution_redundancy']['total_sensor_s']==30
    assert v['execution_redundancy']['unaware_sensor_s']==15
    assert v['latency']['useful_product_median_s']==15
    assert v['latency']['censored_qualified_n']==1 and v['latency']['censored_age_s']==[85]
    assert v['ground_coverage_fraction']==.005

def test_empty_and_zero_duration_truthful_metrics():
    row=metric_row(horizon=0);row.update(capture_products=[],delivered_products=[],agent_live_until_s={'sensor_0':0,'sensor_1':0},coordination={'task_history':[]})
    row['coverage']['ground_delivery_coverage_fraction']=0;row['team_summary']['unique_service_count']=0
    v=m.mission_metrics(row,45000)
    assert v['execution_redundancy']['fraction_of_constellation_available_time'] is None
    assert v['latency']['useful_product_median_s'] is None and not v['reached_horizon']

def test_real_driver_dedup_state_resume_two_episode_resets(tmp_path):
    import json
    from examples.multiagent_imaging.wandb_logging import TrainingLogger
    logger=TrainingLogger(tmp_path,{},mode='disabled')
    row=metric_row();row['mission_metrics']=m.mission_metrics(row,100)
    assert logger.ingest_episode(row);assert not logger.ingest_episode(row)
    state=logger.checkpoint_state();checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    (checkpoint/'manifest.json').write_text(json.dumps({'lifetime_metrics':state}))
    resumed=TrainingLogger(tmp_path,{},mode='disabled',resume=checkpoint)
    assert resumed.checkpoint_state()==state
    assert not resumed.ingest_episode(row)
    second=metric_row(seed=2);second['mission_metrics']=m.mission_metrics(second,100)
    assert resumed.ingest_episode(second)
    assert resumed.lifetime.state['true_decisions']==22 and resumed.lifetime.state['reward']==10
    assert resumed.lifetime.state['complete_training_episodes']==2
    conflict=dict(row,mission_metrics=dict(row['mission_metrics'],constellation_reward=9))
    with pytest.raises(ValueError):resumed.ingest_episode(conflict)

def test_driver_only_wandb_axes_episode_rewards_and_no_heartbeat_increment(tmp_path,monkeypatch):
    import sys,json
    from examples.multiagent_imaging.wandb_logging import TrainingLogger
    calls,logs,definitions=[],[],[]
    run=NS(id='arm-id',name='stable-arm-name',project='gnc-study',entity='entity',url='https://wandb.ai/e/p/runs/id',log=logs.append,finish=lambda **kw:None)
    monkeypatch.setitem(sys.modules,'wandb',NS(init=lambda **kw:calls.append(kw) or run,define_metric=lambda *a,**kw:definitions.append((a,kw))))
    monkeypatch.delenv('BSK_RL_WANDB_KEY_PATH',raising=False)
    logger=TrainingLogger(tmp_path,{},mode='online',required=True)
    row=metric_row();row['mission_metrics']=m.mission_metrics(row,100)
    logger.ingest_episode(row);before=logger.checkpoint_state();logger.heartbeat()
    assert logger.checkpoint_state()==before
    assert logs[0]['agent/sensor_0/reward']==3 and logs[0]['mission/useful_delivered_visits']==1
    assert definitions and any(k.get('step_metric')=='lifetime/true_decisions' for a,k in definitions)
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir();(checkpoint/'manifest.json').write_text(json.dumps({'lifetime_metrics':before}));logger.save_identity(checkpoint)
    resumed=TrainingLogger(tmp_path,{},mode='online',required=True,resume=checkpoint)
    assert calls[-1]['resume']=='must' and calls[-1]['id']=='arm-id' and calls[-1]['name']=='stable-arm-name'
    resumed.driver_pid=-1
    with pytest.raises(RuntimeError):resumed.heartbeat()

def test_actual_reward_boundary_counts_failure_penalty_once(monkeypatch):
    from bsk_rl.completion_gym import CompletionConstellationTasking
    from bsk_rl.gym import SensingAgentConstellationTasking
    env=object.__new__(CompletionConstellationTasking)
    env.simulator=NS(sim_time=25);env._reward_event_index=1;env._reward_logged_index=-1
    env.episode_reward_by_agent={'sensor_0':0,'sensor_1':0};env.reward_events=[]
    monkeypatch.setattr(SensingAgentConstellationTasking,'_get_reward',lambda _:dict(sensor_0=3,sensor_1=-1))
    assert env._get_reward()==dict(sensor_0=3,sensor_1=-1)
    env._get_reward();assert env.episode_reward_by_agent==dict(sensor_0=3,sensor_1=-1)
    assert len(env.reward_events)==2

def test_study_info_has_no_global_truth(monkeypatch):
    from bsk_rl.completion_gym import CompletionConstellationTasking
    from bsk_rl.gym import SensingAgentConstellationTasking
    env=object.__new__(CompletionConstellationTasking);env._sensing_satellites=[NS(name='sensor_0',completion_study=True,requires_retasking=True,retask_reason='local_end')]
    env.latest_step_duration=5.;env.communication_cost_per_s=0
    monkeypatch.setattr(CompletionConstellationTasking,'previously_dead',property(lambda _:[]))
    monkeypatch.setattr(SensingAgentConstellationTasking,'_get_info',lambda _: (_ for _ in ()).throw(AssertionError('Read global truth')))
    info=env._get_info();assert info['sensor_0']==dict(requires_retasking=True,d_ts=5.,retask_reason='local_end',communication_cost_rate=0.)

def test_actual_callback_product_rewards_spool_and_replay(tmp_path,monkeypatch):
    from dataclasses import asdict
    import json
    import numpy as np
    from bsk_rl.data.multiagent_rso_data import ImageProductRecord
    from bsk_rl.data.multiagent_rso_reward import _ServiceEntry
    from bsk_rl.data.completion_catalog import CompletionCatalog
    from examples.multiagent_imaging.training_audit import CompletionAuditCallbacks
    row=metric_row(horizon=100)
    ps=[ImageProductRecord(**p) for p in row['capture_products']]
    services=[_ServiceEntry(ps[0],True,False,1.)]
    sensors=[NS(name=f'sensor_{i}',is_alive=lambda:True,get_obs=lambda:np.zeros(290),
        dynamics=NS(battery_charge_fraction=.9,storage_level_fraction=.2,wheel_speeds_fraction=[.01,.02,.01]),
        data_store=NS(products=(),catalog=CompletionCatalog(f'sensor_{i}',range(200)))) for i in range(2)]
    targets=[NS(rso_target=NS(id=i),target_population_regime='leo') for i in range(200)]
    env=NS(seed=10,default_seed=100,_episode_seed_index=1,simulator=NS(sim_time=100),time_limit=100,
        sensing_satellites=sensors,passive_satellites=targets,episode_reward_by_agent=row['reward_by_agent'],reward_events=[],decision_records=[],
        decision_counts=row['policy_decisions'],action_selection_counts={'sensor_0':{},'sensor_1':{}},
        agent_live_until_s=row['agent_live_until_s'],communication_cost_per_s=0,communication_time={'sensor_0':0,'sensor_1':0},
        coordination_metrics=lambda:row['coordination'],communicator=NS(information_case='catalog_tx',delivery_history=[],transmission_history=[],acknowledged={}),
        rewarder=NS(_team_accounting=NS(capture_attempts=ps),service_entries=services,quality_threshold=.5,alpha=.1,
            reimage_cooldown_s=100,team_summary=row['team_summary'],per_sensor_metrics={s.name:dict(acquisition_credit=1,delivery_credit=1) for s in sensors}))
    runner=NS(env=NS(unwrapped=NS(par_env=env)),worker_index=1)
    class Episode:
        id_='ray-id';is_terminated=False;is_truncated=True
        def __len__(self):return 7
        def agent_steps(self):return 11
        def get_return(self):return 5
    monkeypatch.setenv('BSK_STUDY_EPISODE_SPOOL',str(tmp_path))
    callbacks=CompletionAuditCallbacks();callbacks.on_episode_end(env_runner=runner,episode=Episode())
    first=runner.completion_audits[0]
    assert first['mission_metrics']['ground_coverage_fraction']==.005
    assert first['reward_by_agent']=={'sensor_0':3,'sensor_1':2}
    callbacks.on_episode_end(env_runner=runner,episode=Episode())
    assert len(list(tmp_path.glob('*.json')))==1
    assert json.loads(next(tmp_path.glob('*.json')).read_text())['mission_metrics']==first['mission_metrics']

def test_powered_radio_time_is_not_task_occupancy_and_replay_fails():
    row=metric_row()
    row['coordination']['task_history']=[dict(task('sensor_1',0,40),mode='broadcast',reason='timeout')]
    row['radio_power_intervals']=[dict(sender='sensor_1',start=0,end=40,radio_on_s=7)]
    result=m.mission_metrics(row,100)
    assert result['agents']['sensor_1']['radio_on_s']==7
    assert result['agents']['sensor_1']['radio_task_occupancy_s']==40
    state=m.LifetimeMetrics();state.add('one',result)
    assert state.state['radio_on_sensor_s']==7 and state.state['radio_task_sensor_s']==40
    row['radio_power_intervals']*=2
    with pytest.raises(ValueError,match='Repeated radio'):m.mission_metrics(row,100)

def test_radio_audit_includes_unfinished_hold_once_without_mutating():
    from bsk_rl.comm.completion_communication import CompletionCommunication
    channel=CompletionCommunication()
    action=NS(start_time=20.,radio_on_s=3.,last_time=28.,was_valid=True)
    channel.sensors=[NS(name='sensor_0',_active_completion_transmit=action)]
    channel.transmissions={'sensor_0':'pending'}
    channel.transmission_history=[dict(sender='sensor_0',start=0,end=10,radio_on_s=4),dict(sender='sensor_0',start=0,end=10,completed=True)]
    rows=channel.radio_audit(30)
    assert [r['radio_on_s'] for r in rows]==[4,5]
    assert action.radio_on_s==3 and len(channel.transmission_history)==2

def test_driver_accounting_preserves_baseline_and_does_not_double_accumulate(tmp_path,monkeypatch):
    import json
    from examples.multiagent_imaging.wandb_logging import TrainingLogger
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    (checkpoint/'manifest.json').write_text(json.dumps({'training_accounting':dict(driver_wall_s=3600,driver_allocated_cpu_h=9)}))
    monkeypatch.setenv('BSK_ALLOCATED_CPUS','9')
    monkeypatch.setattr('examples.multiagent_imaging.wandb_logging.time.monotonic',lambda:100.)
    logger=TrainingLogger(tmp_path,{},mode='disabled',resume=checkpoint)
    monkeypatch.setattr('examples.multiagent_imaging.wandb_logging.time.monotonic',lambda:200.)
    expected=dict(driver_wall_s=3700,driver_allocated_cpu_h=9.25)
    assert logger.accounting_state()==expected==logger.accounting_state()
    assert logger.lifetime.state['true_decisions']==0

def test_worker_progress_two_resets_and_driver_heartbeat_do_not_count_experience(tmp_path,monkeypatch):
    import json
    from examples.multiagent_imaging.training_audit import CompletionAuditCallbacks
    from examples.multiagent_imaging.wandb_logging import TrainingLogger
    monkeypatch.setenv('BSK_STUDY_EPISODE_SPOOL',str(tmp_path/'spool'))
    sensors=[NS(name=f'sensor_{i}',completion_study=True,action_description=['charge','image'],
        dynamics=NS(storage_level_fraction=.2,battery_charge_fraction=.8)) for i in range(4)]
    env=NS(seed=31,_episode_seed_index=1,simulator=NS(sim_time=10),sensing_satellites=sensors,
        episode_reward_by_agent={s.name:2 for s in sensors},decision_counts={s.name:3 for s in sensors},
        action_selection_counts={s.name:{'image':3} for s in sensors})
    runner=NS(worker_index=1,env=NS(unwrapped=NS(par_env=env)))
    CompletionAuditCallbacks._publish_progress(runner)
    raw=json.loads((tmp_path/'spool/progress/worker-1.json').read_text())
    assert raw['agents']['sensor_3']['actions']=={'charge':0,'image':3}
    logger=TrainingLogger(tmp_path,{},mode='disabled');logs=[];logger.run=NS(log=logs.append)
    logger.heartbeat();assert logs[-1]['progress/constellation_reward']==8
    assert logs[-1]['progress/sensor_3/action/charge']==0
    assert logger.checkpoint_state()['true_decisions']==0
    env.seed=32;env._episode_seed_index=2;env.simulator.sim_time=0
    env.episode_reward_by_agent={s.name:0 for s in sensors};env.decision_counts={s.name:0 for s in sensors}
    env.action_selection_counts={s.name:{} for s in sensors}
    CompletionAuditCallbacks._publish_progress(runner);logger.heartbeat()
    assert logs[-1]['progress/episode_index']==2 and logs[-1]['progress/constellation_reward']==0
    assert logs[-1]['progress/sensor_3/action/image']==0
    assert logger.checkpoint_state()['complete_training_episodes']==0

def test_failed_sensor_channel_cancel_retains_powered_radio_time():
    from bsk_rl.comm.completion_communication import CompletionCommunication,Transmission
    channel=CompletionCommunication()
    action=NS(start_time=20,radio_on_s=3,last_time=28,was_valid=True)
    channel.sensors=[NS(name='sensor_0',_active_completion_transmit=action)]
    channel.transmissions={'sensor_0':Transmission(20,float('inf'),(),set(),True)}
    channel.transmission_history=[]
    channel.cancel_transmission('sensor_0',30)
    rows=channel.radio_audit(30)
    assert len(rows)==1 and rows[0]['radio_on_s']==5
    assert action._audit_recorded and not channel.transmissions

def test_zero_update_driver_restore_compares_entire_lifetime_and_cost():
    from examples.multiagent_imaging.checkpoints import study_driver_restore_evidence
    manifest=dict(lifetime_metrics=dict(episode_hashes={'episode':'hash'},true_decisions=10,reward=2),
        training_accounting=dict(driver_wall_s=100,driver_allocated_cpu_h=.25))
    algorithm=NS(study_lifetime_state=manifest['lifetime_metrics'],study_accounting_state=manifest['training_accounting'])
    evidence=study_driver_restore_evidence(algorithm,manifest)
    assert evidence['lifetime_metrics_matches_checkpoint'] and evidence['training_accounting_matches_checkpoint']
    algorithm.study_lifetime_state=dict(manifest['lifetime_metrics'],reward=3)
    with pytest.raises(AssertionError,match='lifetime_metrics'):study_driver_restore_evidence(algorithm,manifest)
    assert study_driver_restore_evidence(NS(),{})=={} # historical checkpoint compatibility

def test_fresh_study_wandb_ids_ignore_historical_environment(tmp_path,monkeypatch):
    import sys
    from examples.multiagent_imaging.wandb_logging import TrainingLogger
    calls=[]
    def init(**kwargs):
        calls.append(kwargs)
        return NS(id=kwargs['id'],name=kwargs['name'],project=kwargs['project'],entity='entity',url='https://wandb.ai/test',log=lambda _:None)
    monkeypatch.setitem(sys.modules,'wandb',NS(init=init))
    monkeypatch.setenv('WANDB_RUN_ID','historical');monkeypatch.setenv('BSK_RL_WANDB_RUN_ID','historical')
    monkeypatch.setenv('WANDB_RESUME','allow');monkeypatch.delenv('BSK_RL_WANDB_KEY_PATH',raising=False)
    for arm in ('catalog_tx','geometry_only'):
        output=tmp_path/arm;output.mkdir()
        TrainingLogger(output,{'mission':{'information_case':arm}},mode='online',required=True)
    assert calls[0]['id']!=calls[1]['id'] and all(c['id']!='historical' and c['resume']=='never' for c in calls)
