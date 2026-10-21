"""Local constructed causal fixtures, never representative policy evaluation.

Seed 401, target 55 and the fixed controllers are declared fixture choices. All
captures, pointing, storage, rewards and downlinks are native. No completion facts
or image products are injected. Official JSON configs and production code remain
unchanged. A fixture stops at 2400 s without claiming a complete 45000 s mission.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from examples.multiagent_imaging.update_evidence import file_sha256

SEED = 401
TARGET = 55
STOP_S = 2400.
CASES = ("delivered", "blocked", "expired", "replay", "no_comm_image",
         "no_comm_downlink", "team_delivered", "team_blocked")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def receiver_action(candidate_ids, mask):
    """Fixed private controller: prefer the declared target if locally eligible."""
    legal = [i for i, target in enumerate(candidate_ids)
             if target is not None and mask[5 + i]]
    preferred = next((i for i in legal if candidate_ids[i] == TARGET), None)
    return 5 + (preferred if preferred is not None else legal[0]) if legal else 0


def net_value_gate(communicated, blocked, no_comm_alternatives):
    """An information benefit alone cannot qualify a more costly radio action."""
    best=max(no_comm_alternatives.values())
    return dict(information_value=communicated-blocked,
                best_tested_no_communication_reward=best,
                net_value_against_best_tested_alternative=communicated-best,
                passed=communicated>blocked and communicated>best,
                scope='Only the declared controllers, state and finite window; not an optimal-policy bound')


def credit_trace(raw, gamma, lambda_, lambda_time="step"):
    """Production condensation/targets checked against a separate scalar oracle.

    Zero value predictions isolate the reward contribution. These are diagnostic
    advantages, not claimed critic predictions or standardized PPO advantages.
    Truncation is represented explicitly; the diagnostic bootstrap value is zero.
    """
    from ray.rllib.env.single_agent_episode import SingleAgentEpisode
    from bsk_rl import NO_ACTION
    from bsk_rl.utils.rllib.discounting import (
        CondenseMultiStepActions, compute_value_targets_time_discounted)

    actions = raw["actions"]
    if (not actions or actions[0] == NO_ACTION or
            len(actions) != len(raw['rewards']) or len(actions) != len(raw['durations']) or
            not np.isfinite(raw['rewards']).all() or not np.isfinite(raw['durations']).all() or
            any(dt <= 0 for dt in raw['durations'])):
        raise ValueError('Malformed primitive reward/duration/action trace')
    starts = [i for i, action in enumerate(actions) if action != NO_ACTION]
    times = np.r_[0., np.cumsum(raw["durations"])]
    episode = SingleAgentEpisode(
        observations=[np.array([i], dtype=np.float32) for i in range(len(actions)+1)],
        infos=[dict(d_ts=dt, requires_retasking=True,
                    communication_cost_rate=0.) for dt in [0., *raw["durations"]]],
        actions=actions, rewards=raw["rewards"], len_lookback_buffer=0)
    CondenseMultiStepActions(gamma=gamma)(data=None, episodes=[episode])
    condensed = np.asarray(episode.rewards.data)
    expected, durations = [], []
    for j, begin in enumerate(starts):
        end = starts[j+1] if j+1 < len(starts) else len(actions)
        expected.append(sum(float(raw["rewards"][k])*gamma**(times[k+1]-times[begin])
                            for k in range(begin, end)))
        durations.append(float(times[end]-times[begin]))
    np.testing.assert_allclose(condensed, expected, rtol=1e-12, atol=1e-12)
    # Append a zero bootstrap row, as in the learner's retained target evidence.
    rewards = np.r_[condensed, 0.]
    truncated = np.zeros(len(rewards))
    truncated[-1] = 1
    targets = compute_value_targets_time_discounted(
        np.zeros(len(rewards)), rewards, np.zeros(len(rewards)), truncated,
        np.r_[0., durations], gamma, lambda_, reward_time="step_start", lambda_time=lambda_time)
    scalar = np.zeros(len(rewards))
    for j in reversed(range(len(starts))):
        factor = lambda_ ** durations[j] if lambda_time == "second" else lambda_
        scalar[j] = expected[j] + gamma**durations[j]*factor*scalar[j+1]
    np.testing.assert_allclose(targets, scalar, rtol=2e-6, atol=1e-9)
    rows = []
    for j, begin in enumerate(starts):
        end = starts[j+1] if j+1 < len(starts) else len(actions)
        events = [dict(time_s=float(times[k+1]), primitive_reward=float(raw["rewards"][k]),
                       delay_from_action_s=float(times[k+1]-times[begin]),
                       endpoint_discount=float(gamma**(times[k+1]-times[begin])))
                  for k in range(begin,end) if raw["rewards"][k] != 0]
        rows.append(dict(action=int(actions[begin]), start_s=float(times[begin]),
                         duration_s=durations[j], condensed_reward=float(condensed[j]),
                         diagnostic_value_prediction=0., diagnostic_advantage=float(targets[j]),
                         diagnostic_value_target=float(targets[j]), reward_events=events))
    return dict(passed=True, rows=rows, gamma_per_s=gamma, lambda_per_decision=lambda_,
                lambda_time=lambda_time,
                forced_rows_removed=len(actions)-len(starts),
                oracle_max_abs_target_error=float(np.max(np.abs(targets-scalar))),
                value_scope="Zero-baseline finite-window reward contribution, not a trained critic or policy improvement")


def _private_snapshot(sensor, now):
    from bsk_rl.obs.completion_observations import candidate_snapshot, private_action_mask
    sensor.completion_candidates = None
    sensor.completion_peers = None
    sensor.observation_builder.obs_dict_cache = None
    ids = [None if t is None else int(t.id) for t in candidate_snapshot(sensor,10).targets]
    return dict(time_s=now, candidates=ids, mask=private_action_mask(sensor).tolist(),
                preferred_target_eligible=sensor.data_store.catalog.is_eligible(TARGET,now),
                preferred_target_eligible_without_received_facts=sensor.data_store.catalog.is_privately_eligible(TARGET,now),
                observation=np.asarray(sensor.get_obs()).tolist(),
                records={k:asdict(v) for k,v in sensor.data_store.catalog.records.items()},
                received_inventory=[dict(owner=owner, record_id=rid, entry=asdict(row[0]),
                                         epoch=row[1],ttl=row[2]) for (owner,rid),row in
                                    sorted(sensor.data_store.catalog.received_inventory.entries.items())],
                own_products=[asdict(p) for p in sensor.data_store.products])


def run_case(case, output, wall_seconds=240, *, experiment=None, alternative_target_id=None, receiver_priority_controller=False):
    import Basilisk
    from bsk_rl import NO_ACTION
    from bsk_rl.act.completion_actions import TransmitCompletions
    from bsk_rl.obs.completion_observations import peer_snapshot, private_action_mask
    from examples.multiagent_imaging.config import MultiAgentImagingConfig
    from examples.multiagent_imaging.environment import build_environment
    from bsk_rl.utils.coordination import earth_unoccluded

    output.mkdir(parents=True,exist_ok=False)
    config_path = Path(__file__).with_name("configs")/"walker4_t200_cd1_catalog_tx.json"
    baseline = MultiAgentImagingConfig.from_json(config_path)
    config = replace(experiment or baseline,
        information_case="geometry_only" if case.startswith("no_comm") else "catalog_tx",
        packet_loss_probability=1. if case in ("blocked","team_blocked") else 0.,
        message_delay_s=600. if case == "expired" else 0.,
        reward_credit_mode="team_mean" if case.startswith("team_") else experiment.reward_credit_mode if experiment is not None else "owned")
    native = Basilisk.getBuildInfo()
    started = time.monotonic()
    env = None
    result = dict(case=case,passed=False,kind="Constructed fixed-controller diagnostic; not a learned-policy or full-mission evaluation",
        seed=SEED,preferred_target=TARGET,diagnostic_stop_s=STOP_S,
        official_horizon_s=baseline.episode_duration_s,official_config_sha256=file_sha256(config_path),
        config=config.to_dict(),overrides={k:v for k,v in config.to_dict().items()
            if baseline.to_dict().get(k)!=v},native_build=native,
        physical_product_injection=False,forced_retasking=False,ppo_updates=0,cluster_jobs=0,
        interventions=[],decisions=[],packet_transitions=[],hold_ticks=[],steps=[])
    original = TransmitCompletions.check_hold

    def observed_hold(action):
        nonlocal frozen
        transmission = action.satellite.completion_communicator.transmissions.get(action.satellite.name)
        if transmission is not None and frozen is None:
            frozen = transmission.inventory_snapshot
            result['frozen_payload'] = asdict(frozen)
            result['frozen_payload_sha256'] = digest(asdict(frozen))
        guide = action.satellite.fsw.attGuidMsg.read()
        tick = dict(time_s=float(action.simulator.sim_time),
            los=bool(earth_unoccluded(action.satellite.dynamics.r_BN_N,action.peer.dynamics.r_BN_N)),
            attitude_error=float(np.linalg.norm(guide.sigma_BR)),
            attitude_tolerance=float(action.satellite.fsw.insControl.attErrTolerance))
        complete=original(action)
        tick.update(valid=bool(action.was_valid),held_s=action.held_s,radio_on_s=action.radio_on_s,
                    required_hold_s=action.required_hold_s,complete=bool(complete))
        result['hold_ticks'].append(tick)
        return complete

    try:
        TransmitCompletions.check_hold=observed_hold
        env=build_environment(config)
        env.reset(seed=SEED)
        sensors=env.sensing_satellites
        sender,receiver=sensors[0],sensors[2]
        initial=np.asarray([[*s.dynamics.r_BN_N,*s.dynamics.v_BN_N] for s in env.satellites],dtype='<f8')
        result['initial_orbit_sha256']=hashlib.sha256(initial.tobytes()).hexdigest()
        raw={s.name:dict(actions=[],rewards=[],durations=[]) for s in sensors}
        sent=opportunity=receiver_imaged=replayed=False
        previous_packets=0
        frozen=None
        first_product=None
        physical_trace=[]
        while env.simulator.sim_time < STOP_S-1e-6:
            if time.monotonic()-started>wall_seconds:
                raise RuntimeError('Constructed fixture wall cap exceeded')
            now=float(env.simulator.sim_time)
            actions={}
            for sensor in sensors:
                if not sensor.requires_retasking:
                    actions[sensor.name]=NO_ACTION
                    continue
                snap=_private_snapshot(sensor,now)
                ids=snap['candidates']
                action=0
                if sensor is sender:
                    if np.isclose(now,300):
                        assert TARGET in ids
                        action=5+ids.index(TARGET)
                    elif first_product is not None and not sent and not opportunity:
                        if case.startswith('no_comm'):
                            if case=='no_comm_image':
                                # A real private alternative, rather than assuming
                                # the first shortlist entry is an optimal opportunity.
                                chunks=np.asarray(snap['observation'][26:26+10*config.target_features]).reshape(10,config.target_features)
                                legal=[i for i in range(10) if snap['mask'][5+i]]
                                lit=[i for i in legal if chunks[i,9]>=.5]
                                slot=min(lit or legal,key=lambda i:chunks[i,7])
                                if alternative_target_id is not None:
                                    slot=ids.index(alternative_target_id)
                                    assert slot in legal
                                action=5+slot
                                result['alternative_controller']='Minimum current pointing angle among illuminated private candidates; no optimality claim'
                            else:
                                action={"no_comm_charge":0,"no_comm_desat":2}.get(case,1)
                            opportunity=True
                            result['counterfactual_opportunity_start_s']=now
                        else:
                            peers=peer_snapshot(sensor).peers
                            slot=next(i for i,p in enumerate(peers) if p is receiver)
                            action=15+slot
                            assert private_action_mask(sensor)[action]
                            result['sender_knowledge_before_send']=snap
                            result['receiver_before_send']=_private_snapshot(receiver,now)
                            result['transmit_start_s']=now
                            sent=True
                elif sensor is receiver and now>=600:
                    if not receiver_imaged:
                        action=receiver_action(ids,snap['mask'])
                        if receiver_priority_controller:
                            chunks=np.asarray(snap['observation'][26:26+10*config.target_features]).reshape(10,config.target_features)
                            legal=[i for i in range(10) if ids[i] is not None and snap['mask'][5+i]]
                            lit=[i for i in legal if chunks[i,9]>=.5]
                            action=5+max(lit or legal,key=lambda i:chunks[i,0]) if legal else 0
                        receiver_imaged=action>=5
                        result['receiver_first_image_decision']=dict(time_s=now,action=action,
                            target_id=ids[action-5] if action>=5 else None,private=snap)
                    elif snap['mask'][1]:
                        action=1
                assert snap['mask'][action],(case,sensor.name,now,action)
                actions[sensor.name]=action
                result['decisions'].append(dict(sensor=sensor.name,action=action,time_s=now,
                    target_id=ids[action-5] if 5<=action<15 else None,private=snap))
            # Bound only the diagnostic collection; leave the official horizon intact.
            env.simulator.max_step_duration=min(config.max_step_duration_s,STOP_S-now)
            _,rewards,_,_,_=env.step(actions)
            end=float(env.simulator.sim_time)
            dt=end-now
            owned=dict(env.rewarder.last_owned_science_reward)
            result['steps'].append(dict(start_s=now,end_s=end,actions=actions,rewards=rewards,
                owned_science_reward=owned,
                components=getattr(env.rewarder, 'last_reward_components', {})))
            for sensor in sensors:
                raw[sensor.name]['actions'].append(int(actions[sensor.name]))
                raw[sensor.name]['rewards'].append(float(rewards[sensor.name]))
                raw[sensor.name]['durations'].append(dt)
            if first_product is None:
                products=[p for p in sender.data_store.products if p.target_id==TARGET]
                if products:
                    first_product=products[0]
                    assert first_product.quality>=.5 and first_product.completion_time==end
                    result['native_source_product']=asdict(first_product)
                    assert all(not s.data_store.catalog.records for s in sensors if s is not sender)
            transmission=env.communicator.transmissions.get(sender.name)
            if transmission is not None and frozen is None:
                frozen=transmission.inventory_snapshot
                result['frozen_payload']=asdict(frozen)
                result['frozen_payload_sha256']=digest(asdict(frozen))
            if len(env.communicator.delivery_history)>previous_packets:
                packet=env.communicator.delivery_history[-1]
                previous_packets+=1
                assert frozen is not None
                assert digest(packet['inventory_snapshot'])==digest(asdict(frozen))
                snapshots={s.name:_private_snapshot(s,end) for s in sensors}
                assert all(not snapshots[s.name]['received_inventory'] for s in sensors
                           if s is not receiver)
                if packet['outcome']=='accepted':
                    assert first_product.record_id in receiver.data_store.catalog.records
                    assert TARGET not in snapshots[receiver.name]['candidates']
                    assert not receiver.data_store.products
                else:
                    assert first_product.record_id not in receiver.data_store.catalog.records
                result['packet_transitions'].append(dict(packet=packet,private_after=snapshots))
                if case=='replay' and packet['outcome']=='accepted':
                    before=_private_snapshot(receiver,end)
                    changes=receiver.data_store.catalog.receive_inventory(frozen,end,config.message_ttl_s)
                    after=_private_snapshot(receiver,end)
                    assert changes==0 and before==after
                    replayed=True
                    result['interventions'].append(dict(kind='Duplicate network delivery of the identical frozen packet at the same receipt time; no second radio task',time_s=end,changed_entries=changes,observation_mask_catalog_exact=True))
            physical_trace.append(dict(time_s=end,orbits=np.asarray([[*s.dynamics.r_BN_N,*s.dynamics.v_BN_N] for s in sensors]).tolist(),
                storage=[float(s.dynamics.storage_level_fraction) for s in sensors],
                batteries=[float(s.dynamics.battery_charge_fraction) for s in sensors]))
        assert first_product is not None and receiver_imaged
        if case=='replay':
            assert replayed
        result.update(passed=True,simulated_seconds=float(env.simulator.sim_time),
            captures=[asdict(p) for p in env.rewarder._team_accounting.capture_attempts],
            deliveries=[asdict(p) for p in env.rewarder.service_entries],
            team_summary=dict(env.rewarder.team_summary),reward_by_agent=dict(env.episode_reward_by_agent),
            constellation_reward=float(sum(env.episode_reward_by_agent.values())),
            task_history=list(env.task_history),packets=list(env.communicator.delivery_history),
            radio=env.communicator.radio_audit(STOP_S),
            physical_trace_sha256=digest(physical_trace),raw_agent_trajectories=raw,
            new_full_mission_episodes=0,all_alive=all(s.is_alive() for s in sensors))
        assert result['all_alive'] and Basilisk.getBuildInfo()==native
    except Exception as error:
        result.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        TransmitCompletions.check_hold=original
        if env is not None:
            env.close()
        result['wall_seconds']=time.monotonic()-started
        with (output/'fixture.json').open('x') as stream:
            json.dump(result,stream,indent=2,allow_nan=False)
    return result


def review(results):
    by={r['case']:r for r in results}
    assert set(by)==set(CASES) and all(r['passed'] for r in results)
    assert len({r['initial_orbit_sha256'] for r in results})==1
    assert len({digest(r['native_source_product']) for r in results})==1
    for same in ('replay','team_delivered'):
        assert by[same]['physical_trace_sha256']==by['delivered']['physical_trace_sha256']
    assert by['team_blocked']['physical_trace_sha256']==by['blocked']['physical_trace_sha256']
    assert by['delivered']['receiver_first_image_decision']['target_id']!=TARGET
    for case in ('blocked','expired','no_comm_image','no_comm_downlink'):
        assert by[case]['receiver_first_image_decision']['target_id']==TARGET
    assert by['blocked']['packets'][0]['outcome']=='lost'
    assert by['expired']['packets'][0]['outcome']=='expired'
    assert not by['no_comm_image']['packets'] and not by['no_comm_downlink']['packets']
    for case in ('delivered','blocked','expired','replay','team_delivered','team_blocked'):
        hold=[r for r in by[case]['radio'] if r.get('outcome')=='hold_complete']
        assert hold and all(r['held_s']>=r['required_hold_s'] for r in hold)
    value=by['delivered']['constellation_reward']-by['blocked']['constellation_reward']
    comparisons={case:dict(constellation_reward=r['constellation_reward'],
        delta_from_blocked=r['constellation_reward']-by['blocked']['constellation_reward'],
        receiver_target=r['receiver_first_image_decision']['target_id'],
        acquired_value=r['team_summary']['team_acquisition_value'],
        delivered_value=r['team_summary']['team_value'],
        unique_acquisitions=r['team_summary']['unique_acquisition_count'],
        unique_delivered_visits=r['team_summary']['unique_service_count']) for case,r in by.items()}
    credit={}
    for case in ('delivered','blocked','team_delivered','team_blocked','no_comm_image','no_comm_downlink'):
        r=by[case]
        credit[case]={s:credit_trace(raw,r['config']['discount_per_s'],r['config']['gae_lambda_per_decision'])
                      for s,raw in r['raw_agent_trajectories'].items()}
    attribution={}
    for mode,good,bad in (('owned','delivered','blocked'),('team_mean','team_delivered','team_blocked')):
        start=by[good]['transmit_start_s']
        pair=[]
        for case in (good,bad):
            rows=credit[case]['sensor_0']['rows']
            row=next(r for r in rows if np.isclose(r['start_s'],start))
            assert row['action']>=15
            pair.append(row)
        attribution[mode]=dict(transmit_start_s=start,delivered=pair[0],blocked=pair[1],
            sender_transmit_target_delta=pair[0]['diagnostic_value_target']-pair[1]['diagnostic_value_target'],
            receiver_undiscounted_reward_delta=by[good]['reward_by_agent']['sensor_2']-by[bad]['reward_by_agent']['sensor_2'])
        event=next(p for p in by[good]['captures'] if p['source_sensor']=='sensor_2')
        event_time=event['completion_time']
        good_step=next(s for s in by[good]['steps'] if np.isclose(s['end_s'],event_time))
        bad_step=next(s for s in by[bad]['steps'] if np.isclose(s['end_s'],event_time))
        sender_delta=good_step['rewards']['sensor_0']-bad_step['rewards']['sensor_0']
        later=sum(start<r['start_s']<event_time for r in credit[good]['sensor_0']['rows'])
        gamma=by[good]['config']['discount_per_s']
        lam=by[good]['config']['gae_lambda_per_decision']
        contribution=sender_delta*gamma**(event_time-start)*lam**later
        np.testing.assert_allclose(contribution,attribution[mode]['sender_transmit_target_delta'],rtol=2e-6,atol=1e-9)
        attribution[mode]['downstream_benefit_path']=dict(physical_receiver_capture=event,
            reward_time_s=event_time,delay_from_transmit_start_s=event_time-start,
            endpoint_discount=gamma**(event_time-start),intervening_real_decision_count=later,
            lambda_factor=lam**later,sender_primitive_reward_delta=sender_delta,
            independent_reward_contribution_to_transmit_target=contribution)
    for good,bad in (('delivered','team_delivered'),('blocked','team_blocked')):
        assert len(by[good]['steps'])==len(by[bad]['steps'])
        for owned_step,team_step in zip(by[good]['steps'],by[bad]['steps']):
            assert owned_step['end_s']==team_step['end_s']
            science=owned_step['owned_science_reward']
            for sensor,owned_reward in owned_step['rewards'].items():
                local_adjustment=owned_reward-science[sensor]
                np.testing.assert_allclose(team_step['rewards'][sensor],sum(science.values())/4+local_adjustment,
                                           rtol=1e-12,atol=1e-12)
    gate=net_value_gate(by['delivered']['constellation_reward'],by['blocked']['constellation_reward'],
        {case:by[case]['constellation_reward'] for case in ('no_comm_image','no_comm_downlink')})
    return dict(passed=True,transport_privacy_and_target_checks_passed=True,
        causal_information_value_positive=value>0,net_value_gate=gate,
        matched_transmission_information_value=value,comparisons=comparisons,
        credit=credit,attribution=attribution,ppo_permitted_by_causal_gate=gate['passed'],
        ppo_gate_rule='Require positive value of delivered information AND higher total API return than both tested no-communication alternatives; no PPO when this fails',
        interpretation='Constructed one-seed controller comparison. Physical costs are represented by actual radio tasks and explicit no-communication imaging/downlink alternatives. Failed alternative captures and zero ground delivery remain zero; no invented opportunity penalty or general communication advantage/convergence claim.',
        native_build=results[0]['native_build'],new_full_mission_episodes=0,ppo_updates=0,cluster_jobs=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',choices=CASES,default=CASES)
    args=parser.parse_args()
    if len(set(args.cases))!=len(args.cases):
        parser.error('Repeated fixture cases are not allowed')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'executed-harness.py').write_bytes(Path(__file__).read_bytes())
    from examples.multiagent_imaging.config import MultiAgentImagingConfig
    official=Path(__file__).with_name('configs')/'walker4_t200_cd1_catalog_tx.json'
    assert MultiAgentImagingConfig.from_json(official).reward_credit_mode=='owned'
    before=dict(harness_sha256=file_sha256(__file__),official_config_sha256=file_sha256(official),
                cases=list(args.cases),fixture_seed=SEED,fixture_stop_s=STOP_S,
                ppo_updates=0,cluster_jobs=0)
    with (args.output/'execution.json').open('x') as stream:
        json.dump(before,stream,indent=2)
    results=[]
    for case in args.cases:
        result=run_case(case,args.output/case)
        results.append(result)
        print(json.dumps(dict(case=case,passed=result['passed'],reward=result['constellation_reward'],
                             receiver_target=result['receiver_first_image_decision']['target_id'])),flush=True)
    if set(args.cases)==set(CASES):
        summary=review(results)
        summary['fixture_sha256']={case:file_sha256(args.output/case/'fixture.json') for case in CASES}
        summary['execution']=before
        with (args.output/'qualification.json').open('x') as stream:
            json.dump(summary,stream,indent=2,allow_nan=False)
        print(json.dumps({k:v for k,v in summary.items() if k not in ('credit','native_build')}),flush=True)


if __name__=='__main__':
    main()
