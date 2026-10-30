"""Replay the native P-side prefill arrivals through a run-ahead FIFO model.

The four prefill lanes advance in lockstep (CPU all-reduce per forward). In a
wave, a forward is a prefill when a request is waiting, else a dummy. A dummy
is issued on the CPU once fewer than K forwards are in flight, costs c of CPU
time, and g of GPU time. A prefill holds the CPU until its output (P of GPU
time). Every 32nd forward of a wave is the DP sync; the wave ends there when
no request waits. Prediction: each formal request's schedule-to-update time.
"""
import json, sys, statistics
from pathlib import Path
run = Path(sys.argv[1]); g, c, K, P, d_sched, d_post = map(float, sys.argv[2:8])
K = int(K)
SYNC = 32
def rows(p): return [json.loads(l) for l in open(p) if l.strip()]
adds = {}
for p in sorted(run.glob('nodes/*/scheduler_lifecycle/*.jsonl')):
    r = rows(p); s = next(x for x in r if x['event'] == 'start')
    if s['role'] != 'prefill': continue
    for x in r:
        if x['event'] == 'add': adds[x['request_id']] = x['t']
obs = {o['rid']: o for o in json.load(open('explore_out.json')) if 'step_ms' in o}
reqs = sorted(adds.items(), key=lambda kv: kv[1])
pending = []  # request ids waiting, in arrival order
i = 0; t = None; gpu_free = 0.0; done = []; step = 0; running = False
pred = {}
while i < len(reqs) or pending:
    if not running:
        if not pending:
            rid, a = reqs[i]; i += 1; pending.append(rid); t = max(t or a, a)
        running = True; step = 0; done = []
    while i < len(reqs) and reqs[i][1] <= t:
        pending.append(reqs[i][0]); i += 1
    if pending:
        start = max(t + d_sched / 1000, gpu_free)
        end = start + P / 1000
        for rid in pending: pred[rid] = (t, end)
        pending = []
        gpu_free = end; done.append(end); t = end + d_post / 1000
    else:
        issue = t if len(done) < K else max(t, done[-K])
        start = max(issue, gpu_free); end = start + g / 1000
        gpu_free = end; done.append(end); t = issue + c / 1000
    step += 1
    if step % SYNC == 0:
        while i < len(reqs) and reqs[i][1] <= t:
            pending.append(reqs[i][0]); i += 1
        if not pending: running = False
err = []
for rid, o in obs.items():
    if rid not in pred: continue
    s, e = pred[rid]
    err.append((o['step_ms'], 1000 * (e - adds[rid]), 1000 * (o['e'] - adds[rid]), adds[rid]))
act = [x[2] for x in err]; prd = [x[1] for x in err]
print('n', len(err), 'actual add->update mean %.1f' % statistics.mean(act), 'pred mean %.1f' % statistics.mean(prd),
      'mean abs err %.1f' % statistics.mean(abs(a - b) for a, b in zip(act, prd)),
      'corr %.3f' % statistics.correlation(act, prd))
err.sort(key=lambda x: x[3])
print(' '.join('%d/%d' % (x[2], x[1]) for x in err))
