"""Measure storage allocation with a real archived mission, in fresh processes.

This isolates history retention/serialization. It is not a measured 29-worker
cluster saving. No simulator, learner, or scientific payload reduction.
"""
import argparse
import json
from pathlib import Path
import time
import tracemalloc

from examples.multiagent_imaging.update_evidence import UpdateJournal, UpdateRecords, file_sha256, peak_rss_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--mode', choices=['legacy', 'journal'], required=True)
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert 2 <= args.updates <= 64 and args.episode.stat().st_size < 32 * 1024**2
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.episode.read_text()
    started = time.monotonic()
    tracemalloc.start()
    def update(i):
        return dict(iteration=i, episodes=[json.loads(source)])
    if args.mode == 'legacy':
        records = [update(i) for i in range(1, args.updates + 1)]
        (args.output / 'updates.json').write_text(json.dumps(records, sort_keys=True,
            indent=2, allow_nan=False) + '\n')
    else:
        journal = UpdateJournal(args.output)
        for i in range(1, args.updates + 1):
            journal.append(update(i))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Verification is outside the allocation measurement and holds one update.
    rows = UpdateRecords(args.output)
    count = 0
    for item in rows:
        count += 1
        assert item == update(count)
    assert count == args.updates
    result = dict(mode=args.mode, updates=args.updates, episode_bytes=args.episode.stat().st_size,
        episode_sha256=file_sha256(args.episode), peak_python_allocation_bytes=peak,
        process_peak_rss_bytes=peak_rss_bytes(),
        all_scientific_payloads_roundtrip=True, wall_s=time.monotonic()-started,
        scope='Real archived mission storage replay; no native simulation or cluster savings claim')
    with (args.output / 'measurement.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
