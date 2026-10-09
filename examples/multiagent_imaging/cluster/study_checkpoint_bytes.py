"""Hash read-only checkpoint bytes before/after a future resume gate."""
import argparse,hashlib,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--compare',type=Path);a=p.parse_args()
    hashes={str(f.relative_to(a.checkpoint)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.checkpoint.rglob('*')) if f.is_file()}
    result=dict(passed=True,checkpoint_sha256=hashes)
    if a.compare: result['passed']=hashes==json.loads(a.compare.read_text())['checkpoint_sha256']
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise SystemExit('Checkpoint bytes changed')
if __name__=='__main__':main()
