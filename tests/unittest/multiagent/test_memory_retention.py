"""Offline retention and host accounting; no environment/Ray initialization."""
import gzip,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from examples.multiagent_imaging.learner_batch_evidence import save_raw_episodes
from examples.multiagent_imaging.memory_profile import cgroup_directories,read_cgroup_memory,ray_memory_options,MemoryProfile


def episode(index=0):
    box=lambda x:SimpleNamespace(data=x)
    return SimpleNamespace(id_=str(index),multi_agent_episode_id='mission',agent_id='sensor_0',module_id='imager',
        observations=box(np.arange(12,dtype=np.float32).reshape(3,4)),actions=box([2,0]),rewards=box([.1,.2]),
        infos=box([{'d_ts':np.float32(0)},{'d_ts':1},{'d_ts':2}]),is_terminated=True,is_truncated=False)


def test_stream_roundtrip_and_preserve_existing(tmp_path):
    eps=[episode(i) for i in range(4)]
    save_raw_episodes(tmp_path,(e for e in eps))
    path=tmp_path/'raw-episodes.json.gz';before=path.read_bytes()
    data=json.load(gzip.open(path,'rt'))
    assert len(data)==4 and [r['id'] for r in data]==['0','1','2','3']
    for e,r in zip(eps,data):
        np.testing.assert_array_equal(r['observations'],e.observations.data)
        assert r['actions']==e.actions.data and r['rewards']==e.rewards.data
        assert r['infos']==e.infos.data and r['terminated'] and not r['truncated']
    save_raw_episodes(tmp_path,[episode(99)])
    assert path.read_bytes()==before and not list(tmp_path.glob('*.partial'))


def test_partial_capture_cannot_publish(tmp_path):
    def broken():
        yield episode()
        raise RuntimeError('failure')
    with pytest.raises(RuntimeError):save_raw_episodes(tmp_path,broken())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('v2',[True,False])
def test_process_memory_cgroup_and_job_ancestor(tmp_path,v2):
    proc=tmp_path/'proc';(proc/'self').mkdir(parents=True);mount=tmp_path/'mounted';job=mount/'job_1';step=job/'step_batch';step.mkdir(parents=True)
    (proc/'self/cgroup').write_text('0::/job_1/step_batch\n' if v2 else '5:memory:/job_1/step_batch\n')
    (proc/'self/mountinfo').write_text(f'1 2 0:1 / {mount} rw - '+('cgroup2 cgroup rw\n' if v2 else 'cgroup cgroup rw,memory\n'))
    name='memory.max' if v2 else 'memory.limit_in_bytes'
    (job/name).write_text('107374182400');(step/name).write_text('max' if v2 else '9223372036854771712')
    assert cgroup_directories(proc)==[step,job,mount]
    records=read_cgroup_memory(proc)
    assert records[1]['values'][name]=='107374182400'


def test_mount_namespace_root_translation(tmp_path):
    proc=tmp_path/'proc';(proc/'self').mkdir(parents=True)
    (proc/'self/cgroup').write_text('0::/batch/job_1\n')
    (proc/'self/mountinfo').write_text('1 2 0:1 /batch /sys/fs/cgroup rw - cgroup2 cgroup rw\n')
    assert cgroup_directories(proc)==[Path('/sys/fs/cgroup/job_1'),Path('/sys/fs/cgroup')]


def test_absolute_store_cap_fails_invalid():
    assert ray_memory_options({})=={}
    assert ray_memory_options({'BSK_RL_RAY_OBJECT_STORE_BYTES':str(8*1024**3),'BSK_MEMORY_ALLOCATION_BYTES':str(200*1024**3)})=={'object_store_memory':8*1024**3}
    for value in ['-1','0','broken']:
        with pytest.raises(ValueError):ray_memory_options({'BSK_RL_RAY_OBJECT_STORE_BYTES':value})
    with pytest.raises(ValueError):ray_memory_options({'BSK_RL_RAY_OBJECT_STORE_BYTES':str(100*1024**3),'BSK_MEMORY_ALLOCATION_BYTES':str(100*1024**3)})


def test_profile_reports_memory_without_commands_or_env(tmp_path):
    p=MemoryProfile(tmp_path,interval_s=5);p.start();p.mark('before-test');p.close()
    rows=[json.loads(s) for s in p.path.read_text().splitlines()]
    assert [r['phase'] for r in rows]==['before-ray','before-test','shutdown']
    assert rows[1]['processes'] and not rows[1]['rss_sum_is_job_memory']
    assert all(set(r)<= {'pid','name','rss_bytes','shared_bytes','pss_bytes','uss_bytes'} for row in rows for r in row['processes'])
