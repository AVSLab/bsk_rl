"""Evaluation-only physical product/time metrics and driver checkpoint aggregation.

Derived from the reviewed offline study reference, version physical-mission-v1.
No simulator, actor, sampler, or communication decision imports are permitted.
"""
from collections import defaultdict
import hashlib
import json
import math
import statistics


def union_duration(intervals):
    """Measure a union once; adjacent boundaries use half-open intervals."""
    end, total = -math.inf, 0.0
    for start, stop in sorted(intervals):
        if not math.isfinite(start) or not math.isfinite(stop) or stop < start:
            raise ValueError('Malformed interval')
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def products_with_deliveries(captures, services, horizon):
    products = {p['record_id']: dict(p) for p in captures}
    if len(products) != len(captures):
        raise ValueError('Repeated capture identity')
    delivered = set()
    for entry in services:
        p = entry.get('product', entry)
        if p['record_id'] in delivered:
            raise ValueError('Repeated physical delivery identity')
        delivered.add(p['record_id'])
        original = products[p['record_id']]
        for name in ('target_id', 'request_epoch', 'capture_time', 'source_sensor', 'quality'):
            assert original[name] == p[name], (p['record_id'], name)
        original['delivery_time'] = p['delivery_time']
    for p in products.values():
        assert p['storage_owner'] == p['source_sensor']
        assert 0 <= p['request_epoch'] <= p['capture_time'] <= p['completion_time'] <= horizon
        if p.get('delivery_time') is not None:
            assert p['completion_time'] <= p['delivery_time'] <= horizon
    return list(products.values())


def execution_redundancy(products, tasks, cooldown, sensors, horizon, receipts=None, live_time=None):
    """Occupied imaging time after a qualified peer completion covers the request.

    Includes time after a packet receipt; union prevents double-counting peers.
    Capped at the competing image's capture-anchored cooldown expiry. This is an
    observed opportunity-cost proxy, not counterfactual recoverable mission time.
    """
    total, unknown, busy = {}, {}, {}
    for sensor in sensors:
        sensor_tasks = [t for t in tasks if t['sensor'] == sensor]
        busy[sensor] = union_duration([(t['start'], t['end']) for t in sensor_tasks])
        assert busy[sensor] <= horizon + 1e-6
        if not math.isclose(sum(t['end']-t['start'] for t in sensor_tasks),busy[sensor],abs_tol=1e-6):
            raise ValueError('Overlapping/replayed task identities')
        if live_time is not None and busy[sensor]>live_time[sensor]+1e-6:
            raise ValueError('Task extends beyond available sensor lifetime')
        all_intervals, unknown_intervals = [], []
        for task in sensor_tasks:
            if task['mode'] != 'image':
                continue
            for p in products:
                if (p['source_sensor'] == sensor or p['quality'] < .5
                        or p['target_id'] != task['target_id']
                        or p['request_epoch'] != task['request_epoch']):
                    continue
                start = max(task['start'], p['completion_time'])
                stop = min(task['end'], p['capture_time'] + cooldown, horizon)
                if stop <= start:
                    continue
                all_intervals.append((start, stop))
                if receipts is not None:
                    received = receipts.get(sensor, {}).get(p['record_id'], math.inf)
                    if min(stop, received) > start:
                        unknown_intervals.append((start, min(stop, received)))
        total[sensor] = union_duration(all_intervals)
        unknown[sensor] = union_duration(unknown_intervals) if receipts is not None else None
        assert total[sensor] <= busy[sensor] + 1e-6
    denominator = sum(live_time.values()) if live_time is not None else len(sensors)*horizon
    return dict(available_live_sensor_s=denominator, per_agent_s=total, total_sensor_s=sum(total.values()),
                fraction_of_constellation_available_time=sum(total.values())/denominator if denominator else None,
                unaware_per_agent_s=unknown,
                unaware_sensor_s=sum(unknown.values()) if receipts is not None else None,
                occupied_per_agent_s=busy)


def pending_duplicates(products, cooldown, horizon):
    """Fresh cross-agent physical holdings, distinct from valid post-cooldown revisits.

    At a completed capture, count excess sensors holding the same fresh request.
    Simultaneous first completions have k-1 excess, with no favored sensor index.
    End-before-start ordering excludes a delivery at the same timestamp.
    """
    qualified = [p for p in products if p['quality'] >= .5]
    groups = defaultdict(list)
    for p in qualified:
        groups[(p['target_id'], p['request_epoch'])].append(p)
    excess = 0
    participant_ids = set()
    overlap_target_s = 0.0
    redundant_inventory_target_sensor_s = 0.0
    for records in groups.values():
        starts, ends = defaultdict(list), defaultdict(list)
        for p in records:
            stop = min(p.get('delivery_time') if p.get('delivery_time') is not None else horizon,
                       p['capture_time'] + cooldown, horizon)
            if stop > p['completion_time']:
                starts[p['completion_time']].append(p)
                ends[stop].append(p)
        active = {}
        boundaries = sorted(set(starts) | set(ends))
        previous = boundaries[0] if boundaries else 0
        for time in boundaries:
            agents = {p['source_sensor'] for p in active.values()}
            if len(agents) >= 2:
                participant_ids.update(active)
                overlap_target_s += time - previous
                redundant_inventory_target_sensor_s += (len(agents)-1)*(time-previous)
            for p in ends[time]:
                active.pop(p['record_id'], None)
            existing = {p['source_sensor'] for p in active.values()}
            incoming = {p['source_sensor'] for p in starts[time]}
            excess += max(0, len(existing | incoming) - max(1, len(existing)))
            for p in starts[time]:
                active[p['record_id']] = p
            previous = time
    return dict(excess_completed_acquisitions=excess,
                participating_product_count=len(participant_ids),
                overlap_target_s=overlap_target_s,
                excess_inventory_target_sensor_s=redundant_inventory_target_sensor_s)



METRIC_VERSION = "physical-mission-v1"

def mission_metrics(record, horizon_limit):
    """Evaluation-only pure calculation; consumes product/task/reward event evidence."""
    horizon = record["simulated_seconds"]
    if not math.isfinite(horizon) or not 0 <= horizon <= horizon_limit:
        raise ValueError("Invalid/overshot mission horizon")
    sensors = sorted(record["policy_decisions"])
    products = products_with_deliveries(record["capture_products"], record["delivered_products"], horizon)
    live = {s: record.get("agent_live_until_s", {}).get(s, horizon) for s in sensors}
    receipts = record.get("first_receipts")
    redundant = execution_redundancy(products, record["coordination"]["task_history"],
        record["reimage_cooldown_s"], sensors, horizon, receipts, live)
    services = record["delivered_products"]
    if sum(bool(e.get("unique_service")) for e in services) != record["team_summary"]["unique_service_count"]:
        raise ValueError("Useful visit summary disagrees with physical delivery ledger")
    if set(record["reward_by_agent"]) != set(sensors):
        raise ValueError("Incomplete per-agent reward evidence")
    if not all(math.isfinite(value) for value in record["reward_by_agent"].values()):
        raise ValueError("Nonfinite agent reward")
    if any(not math.isfinite(t) or not 0 <= t <= horizon for t in live.values()):
        raise ValueError("Invalid live-agent duration")
    credited = [e["product"] for e in services if e.get("credited_value", 0) > 0]
    qualified = [p for p in products if p["quality"] >= .5]
    delivered = [p for p in qualified if p.get("delivery_time") is not None]
    useful_delays = [p["delivery_time"]-p["capture_time"] for p in credited]
    all_delays = [p["delivery_time"]-p["capture_time"] for p in delivered]
    censored = [horizon-p["capture_time"] for p in qualified if p.get("delivery_time") is None]
    count = record["coverage"]["catalog_target_count"]
    per_agent = {}
    for sensor in sensors:
        ps = [p for p in products if p["source_sensor"]==sensor]
        ds = [p for p in delivered if p["source_sensor"]==sensor]
        intervals = [(t["start"], t["end"]) for t in record["coordination"]["task_history"]
            if t["sensor"] == sensor and t["mode"] == "broadcast"]
        interruptions = [(t["start"],t["end"]) for t in record["coordination"]["task_history"]
            if t["sensor"]==sensor and t["reason"] in {"policy_switch", "known_completion_or_new_request"}]
        sensor_delays=[p["delivery_time"]-p["capture_time"] for p in credited if p["source_sensor"]==sensor]
        radio = [r for r in record.get("radio_power_intervals", []) if r["sender"]==sensor]
        if len({(r["start"],r["end"]) for r in radio}) != len(radio):
            raise ValueError("Repeated radio power audit")
        for r in radio:
            if not 0 <= r["start"] <= r["end"] <= live[sensor] or not math.isfinite(r["radio_on_s"]) or not 0 <= r["radio_on_s"] <= r["end"]-r["start"]+1e-6:
                raise ValueError("Invalid powered radio duration")
        radio_on = sum(r["radio_on_s"] for r in radio)
        if radio_on > union_duration(intervals)+1e-6:
            raise ValueError("Powered radio duration exceeds transmit task occupancy")
        per_agent[sensor] = dict(useful_delivery_delay_median_s=statistics.median(sensor_delays) if sensor_delays else None,
            useful_delivery_delay_valid_n=len(sensor_delays),
            ground_coverage_fraction=len({p["target_id"] for p in ds})/count if count else None,
            reward=record["reward_by_agent"][sensor],
            true_decisions=record["policy_decisions"][sensor], actual_images=len(ps),
            qualified_images=sum(p["quality"]>=.5 for p in ps),
            useful_delivered_products=sum(p["source_sensor"]==sensor for p in credited),
            delivered_products=len(ds), live_time_s=live[sensor],
            radio_task_occupancy_s=union_duration(intervals),
            radio_task_occupancy_fraction=union_duration(intervals)/live[sensor] if live[sensor] else None,
            radio_on_s=radio_on, radio_on_fraction=radio_on/live[sensor] if live[sensor] else None,
            interrupted_task_s=union_duration(interruptions),
            redundant_imaging_execution_s=redundant["per_agent_s"][sensor],
            unaware_redundant_imaging_execution_s=redundant["unaware_per_agent_s"][sensor])
    assert math.isclose(sum(record["reward_by_agent"].values()), record["constellation_reward"], abs_tol=1e-7)
    ground = len({p["target_id"] for p in delivered})/count if count else None
    assert ground==record["coverage"]["ground_delivery_coverage_fraction"]
    return dict(version=METRIC_VERSION, reached_horizon=math.isclose(horizon,horizon_limit,rel_tol=0,abs_tol=1e-6),
        horizon_limit_s=horizon_limit, actual_duration_s=horizon, agents=per_agent,
        constellation_reward=record["constellation_reward"],
        true_decisions=sum(record["policy_decisions"].values()),
        useful_delivered_visits=record["team_summary"]["unique_service_count"],
        capture_coverage_fraction=len({p["target_id"] for p in qualified})/count if count else None,
        ground_coverage_fraction=ground, catalog_target_count=count,
        pending_duplicates=pending_duplicates(products,record["reimage_cooldown_s"],horizon),
        physical_onboard_overlap=pending_duplicates([dict(p,quality=1.,request_epoch=0.) for p in products],float('inf'),horizon),
        execution_redundancy=redundant,
        latency=dict(useful_product_delays_s=useful_delays, qualifying_product_delays_s=all_delays,
            useful_product_n=len(useful_delays), qualifying_product_n=len(all_delays),
            useful_product_median_s=statistics.median(useful_delays) if useful_delays else None,
            qualifying_product_median_s=statistics.median(all_delays) if all_delays else None,
            censored_qualified_n=len(censored), censored_age_s=censored,
            undelivered_qualified_fraction=len(censored)/len(qualified) if qualified else None),
        semantics="Useful latency: all products with positive ground credit, including shared simultaneous credit; visits are credited requests, not product sum. Censoring reported separately.")

class LifetimeMetrics:
    """Driver-owned checkpoint state. Repeated evidence must be byte-equivalent."""
    def __init__(self, state=None):
        self.state = json.loads(json.dumps(state)) if state else dict(version=METRIC_VERSION,
            episode_hashes={}, true_decisions=0, reward=0., complete_training_episodes=0,
            failed_horizon_episodes=0, agents={}, useful_delivered_visits=0,
            useful_product_delays_s=[], censored_qualified_n=0, qualified_products=0,
            redundant_sensor_s=0., available_live_sensor_s=0., radio_task_sensor_s=0., radio_on_sensor_s=0.)
        if self.state['version'] != METRIC_VERSION: raise ValueError('Lifetime metric version mismatch')
    def add(self, episode_id, metrics):
        digest=hashlib.sha256(json.dumps(metrics,sort_keys=True,allow_nan=False).encode()).hexdigest()
        old=self.state['episode_hashes'].get(episode_id)
        if old:
            if old != digest: raise ValueError('Conflicting callback replay')
            return False
        self.state['episode_hashes'][episode_id]=digest
        self.state['true_decisions']+=metrics['true_decisions']
        self.state['reward']+=metrics['constellation_reward']
        self.state['complete_training_episodes']+=1
        self.state['failed_horizon_episodes']+=int(not metrics['reached_horizon'])
        self.state['useful_delivered_visits']+=metrics['useful_delivered_visits']
        self.state['useful_product_delays_s']+=metrics['latency']['useful_product_delays_s']
        self.state['censored_qualified_n']+=metrics['latency']['censored_qualified_n']
        self.state['qualified_products']+=metrics['latency']['qualifying_product_n']+metrics['latency']['censored_qualified_n']
        self.state['redundant_sensor_s']+=metrics['execution_redundancy']['total_sensor_s']
        self.state['available_live_sensor_s']+=metrics['execution_redundancy']['available_live_sensor_s']
        self.state['radio_task_sensor_s']+=sum(a['radio_task_occupancy_s'] for a in metrics['agents'].values())
        self.state['radio_on_sensor_s']+=sum(a['radio_on_s'] for a in metrics['agents'].values())
        for sensor, row in metrics['agents'].items():
            total=self.state['agents'].setdefault(sensor,dict(reward=0.,true_decisions=0))
            total['reward']+=row['reward'];total['true_decisions']+=row['true_decisions']
        return True
