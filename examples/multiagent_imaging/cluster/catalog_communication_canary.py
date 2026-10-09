"""Bounded forced-action native fixtures; no Ray initialization or PPO.

Seeded inventory and fault interventions are diagnostic inputs, never claimed
as policy-generated acquisitions, ground delivery, or trained mission results.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from bsk_rl import NO_ACTION
from bsk_rl.act.completion_actions import TransmitCompletions
from bsk_rl.data.multiagent_rso_data import ImageProductRecord
from bsk_rl.obs.completion_observations import peer_snapshot, private_action_mask
from bsk_rl.utils.coordination import earth_unoccluded
from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.environment import build_environment

CASES = ("nominal", "delay_loss_retry", "ttl_boundary", "broken_hold", "cancel_deadline", "frozen_revisions", "control")


def run_fixture(name, output):
    started = time.monotonic()
    base = MultiAgentImagingConfig.from_json(Path(__file__).resolve().parents[1] / "configs/walker4_t200_cd1_catalog_tx.json")
    cfg = replace(base, episode_duration_s=1200, max_step_duration_s=1,
                  information_case="geometry_only" if name == "control" else "catalog_tx",
                  message_delay_s=7 if name == "delay_loss_retry" else 0,
                  packet_loss_probability=1 if name == "delay_loss_retry" else 0,
                  message_ttl_s=300 if name == "ttl_boundary" else 600,
                  transmit_hold_s=20 if name == "broken_hold" else 10)
    result = dict(fixture=name, passed=False, kind="forced-action diagnostic; seeded metadata; no policy learning",
                  config=cfg.to_dict(), overrides={k:v for k,v in cfg.to_dict().items() if base.to_dict()[k]!=v},
                  interventions=[], ticks=[], selected=[], ideal_ack_return=True,
                  new_ppo_updates=0, simulated_seconds=0)
    env = None
    original = TransmitCompletions.check_hold
    owner_products = []
    frozen = []
    disturbance = {}

    def observed_hold(action):
        now = float(action.simulator.sim_time)
        sensor, peer = action.satellite, action.peer
        if name == "broken_hold" and action.held_s >= 2 and not disturbance:
            # Actual native spacecraft state intervention, not a mocked LOS flag.
            sc = peer.dynamics.scObject
            for field in (sc.hub.nameOfHubPosition, sc.hub.nameOfHubVelocity):
                state = sc.dynManager.getStateObject(field)
                state.setState((-np.asarray(state.getState())).tolist())
            disturbance.update(restore_at=now+5, restored=False)
            own_sc=sensor.dynamics.scObject
            own_sc.dynManager.getStateObject(own_sc.hub.nameOfHubSigma).setState([[.25],[.15],[.1]])
            result["interventions"].append(dict(time_s=now,kind="native receiver position/velocity mirrored across Earth for five seconds; sender attitude disturbed"))
        if disturbance and not disturbance["restored"] and now >= disturbance["restore_at"]:
            sc=peer.dynamics.scObject
            for field in (sc.hub.nameOfHubPosition, sc.hub.nameOfHubVelocity):
                state=sc.dynManager.getStateObject(field);state.setState((-np.asarray(state.getState())).tolist())
            disturbance["restored"]=True
            result["interventions"].append(dict(time_s=now,kind="native receiver orbit mirrored back; pointing must reacquire"))
        guide=sensor.fsw.attGuidMsg.read()
        row=dict(time_s=now,sender=sensor.name,receiver=peer.name,
                 geometric_los=bool(earth_unoccluded(sensor.dynamics.r_BN_N,peer.dynamics.r_BN_N)),
                 discovered_los=peer.name in sensor.completion_communicator.receivers(sensor),
                 attitude_error_norm=float(np.linalg.norm(guide.sigma_BR)),
                 rate_error_norm=float(np.linalg.norm(guide.omega_BR_B)),
                 attitude_tolerance=float(sensor.fsw.insControl.attErrTolerance),
                 rate_tolerance=float(sensor.fsw.insControl.rateErrTolerance),
                 held_before_s=action.held_s,radio_before=int(sensor.dynamics.transmitterPowerSink.powerStatus))
        complete=original(action)
        row.update(valid=action.was_valid, held_after_s=action.held_s, radio_after=int(sensor.dynamics.transmitterPowerSink.powerStatus),
                   powered_s=action.radio_on_s,required_hold_s=action.required_hold_s,complete=bool(complete))
        result["ticks"].append(row)
        if complete and name=="ttl_boundary":
            snapshot=sensor.completion_communicator.transmissions[sensor.name].inventory_snapshot
            sensor.completion_communicator.delay_s=snapshot.epoch+cfg.message_ttl_s-now
            assert sensor.completion_communicator.delay_s>=0
            result["interventions"].append(dict(time_s=now,kind="receipt scheduled at exact frozen-snapshot TTL boundary",delay_s=sensor.completion_communicator.delay_s))
        if name=="frozen_revisions" and not frozen:
            frozen.append(sensor.completion_communicator.transmissions[sensor.name].inventory_snapshot)
            catalog=sensor.data_store.catalog
            catalog.record_delivery(owner_products[0].delivered(now))
            catalog.record_drop(owner_products[1].record_id,now)
            result["interventions"].append(dict(time_s=now,kind="owner delivery/drop revisions after payload was frozen; no ground reward is injected"))
        return complete

    try:
        TransmitCompletions.check_hold=observed_hold
        env=build_environment(cfg);env.reset(seed=401)
        sensors=env.sensing_satellites;sender=sensors[0]
        catalog=sender.data_store.catalog
        targets=list(catalog.targets)[:2]
        for index,target in enumerate(targets):
            product=ImageProductRecord(f"fixture_{name}_{index}",sender.name,target,0,None,1. if index==0 else .2,sender.name,0.,0.)
            catalog.record_capture(product);owner_products.append(product)
        result["interventions"].append(dict(time_s=0,kind="seed two completed owned inventory facts, one below quality threshold; physical stores/rewards untouched",
                                             products=[asdict(p) for p in owner_products]))

        def step(selected=None):
            assert env.simulator.sim_time < 1200, "Fixture simulated-time cap"
            return env.step({s.name:(selected if s is sender and selected is not None else 0) if s.requires_retasking else NO_ACTION for s in sensors})

        def ready_sender():
            # Explicit retasking intervention for a forced diagnostic, retained.
            if not sender.requires_retasking:
                env._interrupt(sender,float(env.simulator.sim_time),"fixture_retask")
                sender.requires_retasking=True
            sender.completion_peers=None;sender.observation_builder.obs_dict_cache=None
            env._get_obs()
            return private_action_mask(sender)

        def start_send():
            while env.simulator.sim_time<800:
                mask=ready_sender()
                peers=peer_snapshot(sender)
                assert len(peers.peers)==3
                # Stable slots include Earth-blocked peers; never invent contacts.
                for slot,peer in enumerate(peers.peers):
                    if peer is None:assert mask[15+slot]==0
                slots=np.flatnonzero(mask[15:18])
                if len(slots):
                    slot=int(slots[0]);receiver=peers.peers[slot]
                    result["selected"].append(dict(time_s=float(env.simulator.sim_time),slot=slot,receiver=receiver.name,mask=mask.tolist()))
                    step(15+slot)
                    return receiver
                step()
            raise AssertionError("No legal LOS packet before bounded fixture deadline")

        def finish_packet(previous):
            while len(env.communicator.delivery_history)==previous:
                step()
            return env.communicator.delivery_history[-1]

        if name=="control":
            # Both observation and actual action decode, without an extra rollout.
            mask=ready_sender();nav=peer_snapshot(sender)
            assert any(peer is not None for peer in nav.peers), "Control must exercise a real LOS beacon"
            assert not mask[15:18].any()
            before=float(env.simulator.sim_time)
            try:step(15)
            except ValueError:pass
            else:raise AssertionError("Geometry-only control accepted transmit")
            assert env.simulator.sim_time==before
            for _ in range(10):step()
            assert not env.communicator.pending and not env.communicator.delivery_history
            assert all(not s.data_store.catalog.received_inventory.entries for s in sensors)
            result["control_navigation_slots"]=[None if p is None else p.name for p in nav.peers]
        elif name=="cancel_deadline":
            receiver=start_send();env._interrupt(sender,float(env.simulator.sim_time),"fixture_cancel")
            sender.requires_retasking=True
            assert not env.communicator.delivery_history
            sender.fsw.insControl.attErrTolerance=0.0
            result["interventions"].append(dict(time_s=float(env.simulator.sim_time),kind="zero attitude tolerance forces native deadline without a valid hold"))
            start_send()
            while sender._active_completion_transmit is not None:step()
            assert any(h.get('outcome')=='timeout' for h in env.communicator.transmission_history)
            assert not env.communicator.delivery_history
        else:
            receiver=start_send();packet=finish_packet(0)
            if name=="delay_loss_retry":
                assert packet['outcome']=='lost' and not env.communicator.acknowledged
                assert not receiver.data_store.catalog.records
                env.communicator.loss_probability=0
                result["interventions"].append(dict(time_s=float(env.simulator.sim_time),kind="loss injection disabled for explicit retry"))
                start_send();packet=finish_packet(1)
                assert packet['outcome']=='accepted' and packet['received_at']>=packet['sent_at']+7
            elif name=="ttl_boundary":
                assert packet['outcome']=='expired'
                assert packet['received_at']==packet['inventory_snapshot']['epoch']+cfg.message_ttl_s
                assert not env.communicator.acknowledged and not receiver.data_store.catalog.records
            else:
                assert packet['outcome']=='accepted'
                assert packet['receiver']==receiver.name
                assert len(packet['record_ids'])==2
                for peer in sensors[1:]:
                    if peer is not receiver:assert not peer.data_store.catalog.records
                if name=="broken_hold":
                    ticks=result['ticks']
                    assert disturbance.get('restored') and any(not t['geometric_los'] for t in ticks)
                    assert any(t['held_before_s']>=2 and t['held_after_s']==0 for t in ticks)
                    assert ticks[-1]['complete'] and ticks[-1]['held_after_s']>=20
                if name=="frozen_revisions":
                    recv=receiver.data_store.catalog
                    assert all(recv.received_inventory.status(sender.name,p.record_id,env.simulator.sim_time)=='onboard' for p in owner_products)
                    start_send();packet=finish_packet(1)
                    assert recv.received_inventory.status(sender.name,owner_products[0].record_id,env.simulator.sim_time)=='delivered'
                    assert recv.received_inventory.status(sender.name,owner_products[1].record_id,env.simulator.sim_time)=='dropped'
                    recv.receive_inventory(frozen[0],env.simulator.sim_time,600)
                    assert recv.received_inventory.status(sender.name,owner_products[0].record_id,env.simulator.sim_time)=='delivered'
                    assert recv.is_eligible(targets[1],env.simulator.sim_time), "Low quality must not create completion cooldown"
                    assert not recv.is_eligible(targets[0],env.simulator.sim_time)
                    assert recv.is_eligible(targets[0],recv.cooldown_s)
        assert all(not s.data_store.products for s in sensors), "Metadata created a physical image product"
        assert all(abs(v)<1e-12 for v in env.episode_reward_by_agent.values()), "Metadata generated mission reward"
        holds=[h for h in env.communicator.transmission_history if 'outcome' in h]
        for packet in env.communicator.delivery_history:
            matching=[h for h in holds if h.get('outcome')=='hold_complete' and h.get('sender')==packet['sender'] and h.get('end')==packet['sent_at']]
            assert len(matching)==1 and matching[0]['held_s']>=matching[0]['required_hold_s']
        result.update(passed=True,simulated_seconds=float(env.simulator.sim_time),packets=env.communicator.delivery_history,
                      holds=holds,reward_by_agent=env.episode_reward_by_agent,decision_records=env.decision_records,
                      no_receiver_product_ownership=True, ack_revisions={str(k):v for k,v in env.communicator.acknowledged.items()})
    except Exception as error:
        result.update(error_type=type(error).__name__,error=str(error),simulated_seconds=float(env.simulator.sim_time) if env else 0)
        raise
    finally:
        TransmitCompletions.check_hold=original
        if env:env.close()
        result['wall_seconds']=time.monotonic()-started
        output.write_text(json.dumps(result,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x),allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixtures',nargs='+',choices=CASES,default=list(CASES))
    args=parser.parse_args()
    assert len(args.fixtures)<=8 and len(set(args.fixtures))==len(args.fixtures)
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for name in args.fixtures:
        try:
            records.append(run_fixture(name,args.output/(name+'.json')))
        except Exception:
            # The failed fixture retains its raw trace; a missing acceptance file
            # must never be mistaken for an interrupted but successful canary.
            failed=json.loads((args.output/(name+'.json')).read_text())
            summary=dict(passed=False,failed_fixture=name,
                         completed_fixtures=[r['fixture'] for r in records],
                         native_fixture_instances=len(records)+1,
                         simulated_seconds=failed['simulated_seconds'],
                         new_ppo_updates=0,full_mission_episodes=0,
                         error_type=failed.get('error_type'),error=failed.get('error'))
            (args.output/'acceptance.json').write_text(json.dumps(summary,indent=2)+'\n')
            raise
    summary=dict(passed=all(r['passed'] for r in records),fixtures=[dict(fixture=r['fixture'],passed=r['passed'],simulated_seconds=r['simulated_seconds']) for r in records],
                 new_ppo_updates=0, full_mission_episodes=0, native_fixture_instances=len(records), ideal_ack_return=True)
    (args.output/'acceptance.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
