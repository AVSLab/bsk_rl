"""Cooperative training credit must preserve physical ownership and team value."""
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
from bsk_rl.data.multiagent_rso_reward import MultiSensorRSOTargetImageReward
from bsk_rl.sats.roles import SpacecraftRole
from examples.multiagent_imaging.config import MultiAgentImagingConfig


def fixture(mode):
    reward = MultiSensorRSOTargetImageReward(alpha=.1, reward_credit_mode=mode)
    sensors = [NS(name=f'sensor_{i}', role=SpacecraftRole.SENSING_AGENT,
                  dynamics=NS(penalties=0), _current_action_label='image') for i in range(2)]
    reward.scenario = NS(satellites=sensors, target_spacecrafts=[])
    reward.per_sensor_metrics = {s.name:dict(acquisition_credit=0., delivery_credit=0.) for s in sensors}
    # One recipient acquires 10 units; sender has no owned product this event.
    reward._capture_products = lambda data: []
    reward._downlinked_products = lambda: []
    reward._team_accounting = NS(register_acquisitions=lambda *a: {'sensor_1':10},
        register_deliveries=lambda *a: {'sensor_1':5}, successful_duplicate_count=0,
        entries=[], team_value=5, acquisition_team_value=10, unique_acquisition_count=1,
        unique_service_count=1, duplicate_attempt_count=0)
    return reward, sensors


def test_sender_receives_team_science_credit_but_owned_metrics_stay_owned():
    reward, sensors = fixture('team_mean')
    values = reward.calculate_reward({})
    assert values == {'sensor_0':4.75,'sensor_1':4.75}
    assert sum(values.values()) == 9.5
    assert reward.last_owned_science_reward == {'sensor_0':0.,'sensor_1':9.5}
    assert reward.per_sensor_metrics['sensor_0']['acquisition_credit'] == 0
    assert reward.per_sensor_metrics['sensor_1']['acquisition_credit'] == 10
    assert reward.per_sensor_metrics['sensor_1']['delivery_credit'] == 5


def test_sender_penalty_is_not_transferred_to_receiver():
    reward, sensors = fixture('team_mean')
    sensors[0]._current_action_label='action_broadcast'
    reward.per_sensor_metrics['sensor_0']['communication_actions']=0
    reward.communication_penalty=-2
    assert reward.calculate_reward({}) == {'sensor_0':2.75,'sensor_1':4.75}


def test_legacy_owned_credit_and_checkpoint_contract_remain_identical():
    reward,_=fixture('owned')
    assert reward.calculate_reward({}) == {'sensor_0':0.,'sensor_1':9.5}
    old = MultiAgentImagingConfig()
    assert 'reward_credit_mode' not in old.to_dict()
    new = replace(old,reward_credit_mode='team_mean')
    assert new.to_dict()['reward_credit_mode']=='team_mean'
    with pytest.raises(ValueError): replace(old,reward_credit_mode='other')
