"""Scratch: engine iterations inside parts d and f per formal D request, and the lane's step interval there."""
import json, sys, statistics
from bisect import bisect_left, bisect_right
from pathlib import Path
sys.path.insert(0, sys.argv[3])
import analyze_lifecycle as al
run = Path(sys.argv[1]); rows = [json.loads(l) for l in open(sys.argv[2])]
cores = al.load_cores(run, [])
res = []
for (role, dp), core in cores.items():
    if role != 'decode':
        continue
    tl = al.decode_timelines(core['rows'])
    sched = [r['mono'] for r in core['rows'] if r['event'] == 'schedule']
    for row in rows:
        if row.get('d_dp_rank') != dp or row['missing']:
            continue
        t = tl[next(k for k in tl if k.startswith(f"cmpl-{row['request_id']}-"))]
        read, recv, prom, first = t['read']['mono'], t['received']['mono'], t['promotion']['mono'], al.returned(t['first'])
        d_iter = bisect_right(sched, recv) - bisect_right(sched, read)
        f_iter = bisect_right(sched, first) - bisect_right(sched, prom)
        i0, i1 = bisect_left(sched, read), bisect_right(sched, first)
        gaps = [(b - a) * 1e3 for a, b in zip(sched[i0:i1], sched[i0 + 1:i1])]
        res.append(dict(lane=row['lane_state'], d_iter=d_iter, f_iter=f_iter, step_ms=statistics.median(gaps) if gaps else None,
                        d=row['d'], f=row['f'], c_to_e_iter=row['iterations']))
def q(xs):
    xs = sorted(x for x in xs if x is not None); n = len(xs)
    return dict(n=n, mean=round(statistics.mean(xs), 2), p50=round(xs[n // 2], 2), p95=round(xs[int(0.95 * (n - 1))], 2), min=round(xs[0], 2), max=round(xs[-1], 2))
for lane in ('decoding', 'idle'):
    sub = [r for r in res if r['lane'] == lane]
    print(lane, {k: q([r[k] for r in sub]) for k in ('d_iter', 'f_iter', 'step_ms', 'c_to_e_iter')})
    from collections import Counter
    print('  d_iter counts', sorted(Counter(r['d_iter'] for r in sub).items()), 'f_iter counts', sorted(Counter(r['f_iter'] for r in sub).items()))
    for k in sorted({r['d_iter'] for r in sub}):
        s = [r['d'] for r in sub if r['d_iter'] == k]
        print('   d with', k, 'iterations: n', len(s), 'mean', round(statistics.mean(s), 1))
