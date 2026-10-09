"""Only metadata/model forward calls and stubs: no Ray init or mission stepping."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
import gymnasium as gym
from bsk_rl.data.completion_catalog import CompletionCatalog, CompletionRecord
from bsk_rl.data.multiagent_rso_data import ImageProductRecord
from bsk_rl.data.onboard_inventory import InventoryEntry,InventorySnapshot
from bsk_rl.comm.completion_communication import CompletionCommunication
from bsk_rl.obs.completion_observations import CompletionPeers, peer_snapshot
from bsk_rl.sats.roles import SpacecraftRole
from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.readiness import schema,validate_schema
from examples.multiagent_imaging.train import target_attention_config

def product(rid='a', sensor='sensor_0', target=3, quality=1., capture=10., end=20.):
    return ImageProductRecord(rid,sensor,target,capture,None,quality,sensor,0.,end)

def make_channel(case='catalog_tx', **kw):
    clock=NS(sim_time=20.)
    sensors=[NS(name=f'sensor_{i}',role=SpacecraftRole.SENSING_AGENT,simulator=clock,is_alive=lambda:True,
        dynamics=NS(r_BN_N=np.array([7e6,i*1e3,0.]),v_BN_N=np.array([0.,7500.,0.]),BN=np.eye(3)),
        fsw=NS(locPoint=NS(pHat_B=np.array([0.,1.,0.]))),completion_peers=None,
        data_store=NS(catalog=CompletionCatalog(f'sensor_{i}',[3,4],cooldown_s=100),products=())) for i in range(2)]
    comm=CompletionCommunication(case,**kw);comm.link_satellites(sensors);comm.reset_overwrite_previous();comm.reset_post_sim_init()
    return comm,sensors,clock

def receive(comm,sensors,clock,start=20,end=30):
    clock.sim_time=start; comm.begin_directed(*sensors,start)
    comm.transmissions[sensors[0].name].end=end;clock.sim_time=end;comm.communicate()

@pytest.mark.parametrize('case',['catalog_tx','geometry_only'])
def test_common_navigation_without_catalog_delta(case):
    comm,sensors,clock=make_channel(case)
    assert peer_snapshot(sensors[0]).peers==(sensors[1],)
    obs=object.__new__(CompletionPeers);obs.satellite=sensors[0];obs.simulator=clock;obs.norm_time=12000
    row=obs.get_obs()['peer_0']
    assert row.shape==(18,) and row[16]==1 and row[17]==0
    assert not comm.transmit_eligible(*sensors)
    sensors[1].dynamics.r_BN_N=np.array([-7e6,0.,0.]);sensors[0].completion_peers=None
    assert peer_snapshot(sensors[0]).peers==(None,)
    assert not obs.get_obs()['peer_0'].any()

def test_navigation_never_reads_peer_private_fields_and_ignores_truth():
    comm,sensors,clock=make_channel()
    class Forbidden:
        def __getattr__(self,key): raise AssertionError('Private read: '+key)
    sensors[1].data_store=Forbidden();sensors[1].completion_task=Forbidden();sensors[1].action=Forbidden()
    obs=object.__new__(CompletionPeers);obs.satellite=sensors[0];obs.simulator=clock;obs.norm_time=12000
    before=obs.get_obs()['peer_0'].copy()
    comm._team_accounting=Forbidden()
    sensors[0].completion_peers=None
    assert np.array_equal(before,obs.get_obs()['peer_0'])

def test_low_quality_onboard_is_transmitted_without_reward_ownership():
    comm,sensors,clock=make_channel()
    sensors[0].data_store.catalog.record_capture(product(quality=.2))
    receive(comm,sensors,clock)
    receiver=sensors[1].data_store.catalog
    assert receiver.received_inventory.status('sensor_0','a',30)=='onboard'
    assert receiver.received_inventory.current(30)[0].fact.quality==.2
    assert receiver.is_eligible(3,30) and not receiver.target(3).pending_record_ids
    assert sensors[1].data_store.products==()

def test_frozen_packet_and_delivery_revision_reordering():
    comm,sensors,clock=make_channel(delay_s=5)
    cat=sensors[0].data_store.catalog;p=product();cat.record_capture(p)
    comm.begin_directed(*sensors,20);frozen=comm.transmissions['sensor_0'].inventory_snapshot
    cat.record_delivery(p.delivered(25))
    comm.transmissions['sensor_0'].end=30;clock.sim_time=30;comm.communicate()
    assert not sensors[1].data_store.catalog.records
    clock.sim_time=35;comm.communicate()
    recv=sensors[1].data_store.catalog
    assert recv.received_inventory.status('sensor_0','a',35)=='onboard'
    receive(comm,sensors,clock,40,50);clock.sim_time=55;comm.communicate()
    assert recv.received_inventory.status('sensor_0','a',55)=='delivered'
    recv.receive_inventory(frozen,56,600)
    assert recv.received_inventory.status('sensor_0','a',56)=='delivered'
    assert recv.records['a'].delivery_time==25
    assert not recv.is_eligible(3,109) and recv.is_eligible(3,110)

def test_tombstone_expiry_preserves_cooldown_and_new_request():
    comm,sensors,clock=make_channel(ttl_s=40)
    cat=sensors[0].data_store.catalog;cat.record_capture(product())
    receive(comm,sensors,clock)
    cat.record_drop('a',35);receive(comm,sensors,clock,35,40)
    recv=sensors[1].data_store.catalog
    assert recv.received_inventory.status('sensor_0','a',40)=='dropped'
    assert recv.received_inventory.status('sensor_0','a',75)=='unknown'
    assert not recv.is_eligible(3,80)
    recv.set_request_epoch(3,80);assert recv.is_eligible(3,80)
    assert recv.records['a'].capture_time==10

def test_partial_snapshot_never_erases_unreceived_records_complete_can():
    comm,sensors,clock=make_channel(inventory_chunk_records=1)
    cat=sensors[0].data_store.catalog
    cat.record_capture(product());cat.record_capture(product('b',target=4))
    receive(comm,sensors,clock)
    first=comm.delivery_history[-1]['record_ids'][0]
    receive(comm,sensors,clock,31,40)
    recv=sensors[1].data_store.catalog
    assert len(recv.received_inventory.current(40))==2
    assert comm.delivery_history[-1]['record_ids']!=[first]
    assert not comm.transmit_eligible(*sensors)
    snapshot=InventorySnapshot('sensor_0',3,45,(),False)
    recv.receive_inventory(snapshot,45,600)
    assert len(recv.received_inventory.current(45))==2
    recv.receive_inventory(replace(snapshot,complete=True),46,600)
    assert not recv.received_inventory.current(46)
    assert not recv.is_eligible(3,46) # durable fact, even with authoritative absence

def test_packet_replay_idempotent_unknown_expiry_and_refresh():
    comm,sensors,clock=make_channel(ttl_s=40);cat=sensors[0].data_store.catalog;cat.record_capture(product())
    receive(comm,sensors,clock)
    snapshot=InventorySnapshot('sensor_0',1,20,tuple(cat.owned_inventory.values()),True)
    recv=sensors[1].data_store.catalog
    assert recv.receive_inventory(snapshot,31,40)==0
    assert recv.received_inventory.status('sensor_0','a',60)=='unknown'
    clock.sim_time=60;assert comm.transmit_eligible(*sensors)
    assert not recv.is_eligible(3,60)

@pytest.mark.parametrize('loss,delay,ttl,outcome',[(1,0,600,'lost'),(0,50,40,'expired')])
def test_loss_delay_ttl_gates(loss,delay,ttl,outcome):
    comm,sensors,clock=make_channel(loss_probability=loss,delay_s=delay,ttl_s=ttl)
    sensors[0].data_store.catalog.record_capture(product())
    receive(comm,sensors,clock);clock.sim_time=100;comm.communicate()
    assert comm.delivery_history[-1]['outcome']==outcome
    assert not sensors[1].data_store.catalog.records
    assert not comm.acknowledged

def test_bits_hold_and_geometry_control_no_transmission():
    comm,sensors,clock=make_channel();sensors[0].data_store.catalog.record_capture(product())
    snap=comm.frozen_inventory('sensor_0','sensor_1',20)
    assert snap.wire_bytes()>64
    sensors[0].completion_metadata_bitrate_bps=1
    assert not comm.transmit_eligible(*sensors)
    with pytest.raises(ValueError):comm.begin_directed(*sensors,20)
    comm.information_case='geometry_only';sensors[0].completion_metadata_bitrate_bps=64000
    with pytest.raises(ValueError):comm.begin_directed(*sensors,20)
    comm.communicate();assert not comm.pending and not comm.delivery_history

def test_snapshot_conflict_atomic_rejection():
    comm,sensors,clock=make_channel();cat=sensors[0].data_store.catalog;cat.record_capture(product())
    receive(comm,sensors,clock);recv=sensors[1].data_store.catalog
    entry=cat.owned_inventory['a'];bad=replace(entry,fact=replace(entry.fact,quality=.9))
    with pytest.raises(ValueError):recv.receive_inventory(InventorySnapshot('sensor_0',1,35,(bad,),True),35,600)
    assert recv.received_inventory.entries[('sensor_0','a')][0]==entry

@pytest.mark.parametrize('arm',['catalog_tx','geometry_only'])
def test_model_schema_private_mask_separation_and_fresh_weights(arm):
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
    from ray.rllib.core.columns import Columns
    from bsk_rl.utils.rllib.target_gnn_module import GNNModule
    cfg=MultiAgentImagingConfig.from_json(Path(__file__).parents[3]/f'examples/multiagent_imaging/configs/walker4_t200_cd1_{arm}.json')
    contract=schema(cfg);assert contract['observation']['size']==290
    model_cfg=target_attention_config(cfg)
    def build(c):
        torch.manual_seed(3105)
        return RLModuleSpec(module_class=GNNModule,catalog_class=PPOCatalog,
            observation_space=gym.spaces.Box(-np.inf,np.inf,(290,),np.float32),
            action_space=gym.spaces.Discrete(18),model_config_dict=target_attention_config(c)).build()
    model=build(cfg);other=build(replace(cfg,information_case='geometry_only' if arm=='catalog_tx' else 'catalog_tx'))
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in model.state_dict().items())
    obs=torch.zeros(4,290);obs[:,1]=.9
    peers=obs[:,236:].reshape(4,3,18);peers[:,:,16]=1
    first=model._forward_inference({Columns.OBS:obs})[Columns.ACTION_DIST_INPUTS]
    assert (first[:,15:]<-1e8).all()
    # Navigation enters actor and critic even when transmit isn't allowed.
    shifted=obs.clone();shifted[:,236]=.7
    assert not torch.equal(model.vf(obs),model.vf(shifted))
    peers[:,:,17]=1
    after=model._forward_inference({Columns.OBS:obs})[Columns.ACTION_DIST_INPUTS]
    assert bool((after[:,15:]>-1e8).all())==(arm=='catalog_tx')
    with pytest.raises(ValueError):validate_schema(schema(replace(cfg,n_targets=100)),cfg)
    assert model_cfg['dropout']==.1 and model_cfg['width_f']==256

def test_new_schema_checkpoint_logits_rng_and_aggregation_export(tmp_path,monkeypatch):
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
    from bsk_rl.utils.rllib.target_gnn_module import GNNModule
    from examples.multiagent_imaging import checkpoints
    cfg=MultiAgentImagingConfig.from_json(Path(__file__).parents[3]/'examples/multiagent_imaging/configs/walker4_t200_cd1_catalog_tx.json')
    torch.manual_seed(3105)
    module=RLModuleSpec(module_class=GNNModule,catalog_class=PPOCatalog,
        observation_space=gym.spaces.Box(-np.inf,np.inf,(290,),np.float32),action_space=gym.spaces.Discrete(18),
        model_config_dict=target_attention_config(cfg)).build()
    task=NS(_episode_seed_index=4,default_seed=99,communicator=NS(information_case='catalog_tx'))
    runner=NS(worker_index=0,env=NS(unwrapped=NS(par_env=task)),
        study_audit_state={'episode_hashes':{'a':'hash'},'true_decisions':100,'reward':7},completion_audits=[])
    algo=NS(training_iteration=2,get_module=lambda _:module,save_to_path=lambda p:Path(p).mkdir(),
        config=NS(num_env_runners=0),env_runner_group=NS(local_env_runner=runner,
        _worker_manager=NS(actor_ids=lambda:[],total_num_restarts=lambda:0)),study_lifetime_state={'version':'physical-mission-v1','true_decisions':100})
    monkeypatch.setattr(checkpoints,'optimizer_digest',lambda _: 'optimizer-test')
    probes=np.zeros((4,290),np.float32);probes[:,1]=.9
    folder=tmp_path/'new-checkpoint'
    result=checkpoints.save_checkpoint(algo,cfg,folder,probes)
    assert result['max_logit_error']==0
    import json
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['lifetime_metrics']==algo.study_lifetime_state
    before=checkpoints._runner_seed_state(runner)
    task._episode_seed_index=0;runner.study_audit_state={};torch.manual_seed(5)
    checkpoints._restore_runner_seed_state(runner,[before])
    assert checkpoints._runner_seed_state(runner)==before
    assert runner._needs_initial_reset

def test_study_initializer_space_check_never_steps_or_samples():
    from examples.multiagent_imaging.train import check_study_static_contract,build_ppo_config,PPOSettings,LauncherResources
    cfg=MultiAgentImagingConfig.from_json(Path(__file__).parents[3]/'examples/multiagent_imaging/configs/walker4_t200_cd1_catalog_tx.json')
    env=NS(_study_contract=schema(cfg),get_agent_ids=lambda:{'sensor_0'},
        observation_space={'sensor_0':NS(shape=(290,))},action_space={'sensor_0':NS(n=18)},
        step=lambda _: (_ for _ in ()).throw(AssertionError('Initialization sampled physics')))
    check_study_static_contract(env)
    ppo=build_ppo_config(cfg,train_batch_size=4200,ppo_settings=PPOSettings.from_profile('amos_best'),resources=LauncherResources(num_env_runners=1))
    assert ppo.num_env_runners==1 and ppo.num_learners==0 and ppo.train_batch_size==4200
    assert ppo.num_sgd_iter==30 and ppo.sgd_minibatch_size==128

def test_resource_caps_and_measured_long_budget(monkeypatch):
    from examples.multiagent_imaging.cluster.study_gate import allocation
    from examples.multiagent_imaging.cluster import study_long_budget as module
    assert allocation(9,14400,36)['reserved_cpu_h']==36
    assert allocation(9,86400,216)['reserved_cpu_h']==216
    with pytest.raises(ValueError):allocation(10,86400,216)
    monkeypatch.setattr(module,'source_gate',lambda:{'identity':'source'})
    accepted=dict(passed=True,source_gate={'identity':'source'},measured_throughput={'catalog_tx':dict(max_heldout_wall_s=600,update_wall_s=2000,startup_wall_s=100)})
    plan=module.budget(accepted,'catalog_tx')
    assert plan['training_start_update_cutoff_s']+plan['evaluation_reserve_s']+plan['update_shutdown_reserve_s']==86400
    with pytest.raises(ValueError):module.budget(dict(accepted,passed=False),'catalog_tx')
    with pytest.raises(ValueError):module.budget(dict(accepted,source_gate={'identity':'changed'}),'catalog_tx')

def test_checkpoint_byte_verifier_offline(tmp_path,monkeypatch):
    import json,sys
    from examples.multiagent_imaging.cluster import study_checkpoint_bytes as module
    folder=tmp_path/'checkpoint';folder.mkdir();(folder/'data').write_bytes(b'original')
    before=tmp_path/'before.json';after=tmp_path/'after.json'
    monkeypatch.setattr(sys,'argv',['bytes','--checkpoint',str(folder),'--output',str(before)]);module.main()
    monkeypatch.setattr(sys,'argv',['bytes','--checkpoint',str(folder),'--output',str(after),'--compare',str(before)]);module.main()
    assert json.loads(after.read_text())['passed']
    (folder/'data').write_bytes(b'changed')
    monkeypatch.setattr(sys,'argv',['bytes','--checkpoint',str(folder),'--output',str(tmp_path/'failed.json'),'--compare',str(before)])
    with pytest.raises(SystemExit):module.main()

def test_new_numpy_and_policy_masks_match_without_full_storage_shield():
    from bsk_rl.obs.completion_observations import private_action_mask,CandidateSnapshot
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
    from ray.rllib.core.columns import Columns
    from bsk_rl.utils.rllib.target_gnn_module import GNNModule
    for arm in ('catalog_tx','geometry_only'):
        cfg=MultiAgentImagingConfig.from_json(Path(__file__).parents[3]/f'examples/multiagent_imaging/configs/walker4_t200_cd1_{arm}.json')
        channel,sensors,clock=make_channel(arm)
        sensor=sensors[0];sensor.completion_task=None
        sensor.completion_n_candidates=10;sensor.completion_empty_storage_mask_fraction=1e-6;sensor.completion_minimum_task_battery_fraction=.05
        sensor.dynamics.storage_level_fraction=1.;sensor.dynamics.battery_charge_fraction=.9
        sensor.completion_candidates=CandidateSnapshot(20,(NS(id=4),)+(None,)*9)
        sensor.data_store.catalog.record_capture(product())
        mask=private_action_mask(sensor)
        # Two fake peers are padding; extra agents absent in this metadata stub.
        obs=torch.zeros(1,290);obs[:,0]=1.;obs[:,1]=.9;obs[:,26+16]=1
        peers=obs[:,236:].reshape(1,3,18);peers[:,0,16]=1;peers[:,0,17]=int(channel.transmit_eligible(*sensors))
        torch.manual_seed(3105)
        module=RLModuleSpec(module_class=GNNModule,catalog_class=PPOCatalog,
            observation_space=gym.spaces.Box(-np.inf,np.inf,(290,),np.float32),action_space=gym.spaces.Discrete(18),model_config_dict=target_attention_config(cfg)).build()
        logits=module._forward_inference({Columns.OBS:obs})[Columns.ACTION_DIST_INPUTS][0]
        assert torch.equal(logits[:16]>-1e8,torch.tensor(mask,dtype=torch.bool))
        assert mask[5]==1 # Full physical storage does not hide imaging.

def test_ready_only_observation_rebuild_and_true_terminal_bootstrap(monkeypatch):
    from contextlib import nullcontext
    from bsk_rl.completion_gym import CompletionConstellationTasking
    env=object.__new__(CompletionConstellationTasking)
    counts={'sensor_0':0,'sensor_1':0}
    def observation(name): counts[name]+=1;return np.ones(290)
    env._sensing_satellites=[NS(name='sensor_0',requires_retasking=True,get_obs=lambda:observation('sensor_0')),
        NS(name='sensor_1',requires_retasking=False,get_obs=lambda:observation('sensor_1'),observation_space=NS(sample=lambda:np.ones(290)))]
    env.simulator=NS(sim_time=25);env.time_limit=45000;env.generate_obs_retasking_only=True
    env.profiler=NS(section=lambda _:nullcontext())
    monkeypatch.setattr(CompletionConstellationTasking,'previously_dead',property(lambda _:[]))
    obs=env._get_obs();assert counts==dict(sensor_0=1,sensor_1=0) and not obs['sensor_1'].any()
    env.simulator.sim_time=45000
    terminal=env._get_obs();assert terminal['sensor_1'].all() and counts==dict(sensor_0=2,sensor_1=1)
    assert env.generate_obs_retasking_only

def test_hard_true_decision_cap_fails_before_any_physical_action(tmp_path,monkeypatch):
    from bsk_rl.completion_gym import CompletionConstellationTasking
    env=object.__new__(CompletionConstellationTasking)
    env.simulator=NS(sim_time=42);env.study_remaining_true_budget=1
    env.satellites=[NS(role=SpacecraftRole.SENSING_AGENT,is_alive=lambda:True),NS(role=SpacecraftRole.SENSING_AGENT,is_alive=lambda:True)]
    env.decision_counts={'sensor_0':0,'sensor_1':0};env.default_seed=1;env._episode_seed_index=2;env.task_history=[]
    monkeypatch.setenv('BSK_STUDY_EPISODE_SPOOL',str(tmp_path))
    with pytest.raises(RuntimeError,match='cap'):env._step([0,0])
    import json
    assert not json.loads(next(tmp_path.glob('*.failed')).read_text())['reached_horizon']

def test_launch_wait_propagates_fast_child_failure(tmp_path):
    import subprocess
    source=Path(__file__).parents[3]/'examples/multiagent_imaging/cluster/study_common.sh'
    code=source.read_text().split('study_run() {',1)[1]
    script='set -euo pipefail\nstudy_run() {'+code+'\nstudy_run bash -c "exit 7"\n'
    result=subprocess.run(['bash','-c',script],capture_output=True,text=True)
    assert result.returncode==7
