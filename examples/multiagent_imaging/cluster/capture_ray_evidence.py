"""Preserve bounded Ray error logs and cgroup memory counters; no imports/Ray init."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def capture(root, output):
    output.mkdir(parents=True, exist_ok=False)
    files, skipped = {}, []
    total=0
    for pattern in ('*/logs/raylet.out','*/logs/raylet.err','*/logs/gcs_server.out',
                    '*/logs/python-core-worker*.log','*/logs/worker-*.out','*/logs/worker-*.err'):
        for source in sorted(root.glob(pattern)):
            name=str(source.relative_to(root))
            if name in files:continue
            size=source.stat().st_size
            if size>50*1024**2 or total+size>256*1024**2:
                skipped.append(dict(path=name,bytes=size,reason='Declared bounded log-copy cap'));continue
            dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
            files[name]=dict(bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
            total+=size
    memory={}
    # Read only named memory counters; never process environments/command lines.
    for path in (Path('/sys/fs/cgroup/memory.events'),Path('/sys/fs/cgroup/memory.peak'),
                 Path('/sys/fs/cgroup/memory.max'),Path('/sys/fs/cgroup/memory.current')):
        if path.is_file():memory[str(path)]=path.read_text()
    record=dict(files=files,skipped=skipped,cgroup_memory=memory,total_bytes=total,
                limitation='Counters at visible cgroup root may describe the node, not the job; paths retained explicitly')
    (output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();capture(a.root,a.output)
