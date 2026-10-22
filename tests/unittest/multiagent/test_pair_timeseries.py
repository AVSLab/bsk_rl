"""Compact extraction retains real choices and counts without product arrays."""
import io
import json
import tracemalloc

from examples.multiagent_imaging.cluster.collect_pair_comparison import project, skip_value
from examples.multiagent_imaging.cluster.collect_pair_timeseries import project_update
from examples.multiagent_imaging.update_evidence import JSONStream


def test_timeseries_projection_keeps_transmit_selections_separate_from_packets():
    episode = dict(episode_id='e', episode_seed=7, simulated_seconds=45000,
        policy_decisions={'s':10}, action_selection_counts={'s':{
            'transmit_completions_0':3, 'action_imageRSO_0':7}},
        capture_products=[{'record_id':str(i)} for i in range(4)],
        communication_summary={'packet_outcome_counts':{'accepted':2}},
        cooperative_tasking={'rejected_acquisitions':1,'unique_acquisition_requests':2},
        mission_metrics={'agents':{'s':{'radio_on_s':20}}, 'latency':{'large':[0]*100}},
        decision_records=[{'large':[0]*100}])
    reader=JSONStream(io.StringIO(json.dumps(dict(iteration=1,episodes=[episode],
        retained_learner_batch={'large':[0]*100}))))
    row=project_update(reader,skip_value,project);reader.finish()
    e=row['episodes'][0]
    assert e['physical_completed_captures']==4 and 'capture_products' not in e
    assert e['action_selection_counts']['s']['transmit_completions_0']==3
    assert e['communication_summary']['packet_outcome_counts']['accepted']==2
    assert 'latency' not in e['mission_metrics'] and 'decision_records' not in e
    assert 'retained_learner_batch' not in row


def test_skipped_training_arrays_do_not_scale_projection_memory():
    raw='{"retained_learner_batch":['+','.join(['0']*2000000)+'],"episodes":[]}'
    stream=io.StringIO(raw)
    tracemalloc.start()
    reader=JSONStream(stream)
    assert project_update(reader,skip_value,project)=={'episodes':[]}
    reader.finish()
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    assert peak<1024**2
