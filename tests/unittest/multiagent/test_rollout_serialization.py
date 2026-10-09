"""Exercise the real rollout exporter using a completed fake environment.

No simulator, sampler, or PPO optimizer runs. Nonempty catalog records cover the
scope error discovered after the first full-mission cluster evaluation.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from examples.multiagent_imaging import evaluate, study_metrics
from examples.multiagent_imaging.config import MultiAgentImagingConfig


@dataclass
class Record:
    record_id: str
    target_id: int
    source_sensor: str
    quality: float


@pytest.mark.parametrize("information_case", ["catalog_tx", "geometry_only"])
def test_completed_rollout_serializes_nonempty_catalog(monkeypatch, information_case):
    config = MultiAgentImagingConfig.from_json(
        Path(evaluate.__file__).with_name("configs") /
        f"walker4_t200_cd1_{information_case}.json"
    )
    names = [f"sensor_{i}" for i in range(4)]
    sensors = []
    for name in names:
        record = Record(f"{name}-record", 0, name, .8)
        sensors.append(NS(
            name=name, dynamics=NS(r_BN_N=[1., 2., 3.], v_BN_N=[4., 5., 6.]),
            data_store=NS(products=[], catalog=NS(
                records={record.record_id: record}, targets={},
                received_at={}, version_received_at={},
            )),
        ))
    targets = [NS(
        name=f"target_{i}", dynamics=NS(r_BN_N=[7., 8., 9.], v_BN_N=[1., 0., 0.]),
        rso_target=NS(priority=1., id=i), target_population_regime="leo",
    ) for i in range(200)]
    coordination = dict(task_history=[], communication_time_s={n: 0. for n in names})
    env = NS(
        agents=[], possible_agents=names, sensing_satellites=sensors,
        passive_satellites=targets, simulator=NS(sim_time=45000.),
        decision_counts={n: 0 for n in names}, agent_live_until_s={n: 45000. for n in names},
        decision_records=[], reward_events=[],
        rewarder=NS(
            _team_accounting=NS(capture_attempts=[]), service_entries=[],
            quality_threshold=.5, reimage_cooldown_s=6000., team_summary={},
            per_sensor_metrics={n: dict(acquisition_credit=0., delivery_credit=0.) for n in names},
        ),
        communicator=NS(
            delivery_history=[], transmission_history=[], acknowledged={},
            radio_audit=lambda end: [],
        ),
        coordination_metrics=lambda: coordination,
        reset=lambda **kwargs: ({}, {}),
    )
    closed = []
    env.close = lambda: closed.append(True)
    monkeypatch.setattr(evaluate, "build_environment", lambda *a, **kw: env)
    from examples.multiagent_imaging import baseline_monte_carlo
    monkeypatch.setattr(baseline_monte_carlo, "coverage_metrics", lambda *a, **kw: {})
    monkeypatch.setattr(baseline_monte_carlo, "duplicate_product_metrics", lambda *a, **kw: {})
    inspected = []
    def metrics(evidence, horizon):
        inspected.append((evidence, horizon))
        return {"export_test": True}
    monkeypatch.setattr(study_metrics, "mission_metrics", metrics)

    result = evaluate.run_rollout(config)
    assert closed == [True]
    assert result["sim_time_s"] == 45000.
    assert result["passive_target_count"] == 200
    assert len(inspected) == 1 and inspected[0][1] == 45000.
    for name in names:
        assert result["completion_records"][name] == [dict(
            record_id=f"{name}-record", target_id=0, source_sensor=name, quality=.8,
        )]
    assert result["mission_metrics"]["export_test"]
    assert json.loads(json.dumps(result, allow_nan=False)) == result
