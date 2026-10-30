"""Native P2-64 D side: host issue pacing of dummy forwards (lead scratch)."""
import json, sys, statistics, collections
from pathlib import Path
run = Path(sys.argv[1])
def rows(p): return [json.loads(l) for l in open(p) if l.strip()]
W = {}
for p in sorted(run.glob('nodes/*/worker_dp_wave/*.jsonl')):
    r = rows(p); s = next(x for x in r if x['event'] == 'worker_wave_start')
    if s['role'] == 'decode': W[s['dp_rank']] = r
fwd = {}
for k, r in W.items():
    for x in r:
        c = x.get('calls', {}).get('dp_cpu_all_reduce')
        if c and x.get('calls', {}).get('dp_padding_sync', {}).get('stage') == 'serving':
            fwd.setdefault(c['forward'], {})[k] = x
lanes = sorted(W)
print('lanes', lanes, 'forwards', len(fwd), min(fwd), max(fwd))
kinds = collections.Counter()
for f, d in fwd.items():
    kinds[tuple(sorted(collections.Counter(x['event'] for x in d.values()).items()))] += 1
print('forward lane-kind mixes', kinds.most_common(8))
alld = [f for f in sorted(fwd) if len(fwd[f]) == len(lanes) and all(x['event'] == 'worker_dummy_batch' for x in fwd[f].values())]
print('all-dummy forwards', len(alld))
def end(f): return max(x['end_t'] for x in fwd[f].values())
iv = [1000 * (end(f) - end(f - 1)) for f in alld if f - 1 in fwd and f - 1 in alld]
iv.sort()
if iv:
    print('all-dummy consecutive issue-end intervals: n', len(iv), 'p10', iv[len(iv)//10], 'median', statistics.median(iv), 'p90', iv[9*len(iv)//10])
    print('fast(<12ms)', sum(v < 12 for v in iv), 'paced', sum(v >= 12 for v in iv))
# Runs of consecutive all-dummy forwards
runs = []; cur = []
for f in alld:
    if cur and f == cur[-1] + 1: cur.append(f)
    else:
        if cur: runs.append(cur)
        cur = [f]
if cur: runs.append(cur)
print('all-dummy runs', len(runs), 'lengths', collections.Counter(len(r) for r in runs).most_common(10))
for r in runs[:12]:
    ivs = [round(1000 * (end(f) - end(f - 1)), 1) for f in r if f - 1 in fwd]
    print(r[0], len(r), ivs[:40])
