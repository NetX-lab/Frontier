"""Estimate the native P-side dummy run-ahead depth for every prefill wave.

For each formal prefill forward F, the dummies F+1.. are classified as fast
(CPU launch only) or paced (CPU waits for the GPU). The GPU is assumed busy from
the first fast dummy onward at the steady paced duration g, so the queue depth at
the first paced dummy is n_fast - elapsed / g.
"""
import json, sys, statistics
from pathlib import Path
run = Path(sys.argv[1])
def rows(p): return [json.loads(l) for l in open(p) if l.strip()]
W = {}
for p in sorted(run.glob('nodes/*/worker_dp_wave/*.jsonl')):
    r = rows(p); s = next(x for x in r if x['event'] == 'worker_wave_start')
    if s['role'] == 'prefill': W[s['dp_rank']] = r
fwd = {}
for k, r in W.items():
    for x in r:
        c = x.get('calls', {}).get('dp_cpu_all_reduce')
        if c and x.get('calls', {}).get('dp_padding_sync', {}).get('stage') == 'serving':
            fwd.setdefault(c['forward'], {})[k] = x
lanes = sorted(W)
def span(f):  # CPU span of forward f across lanes: latest start, latest end
    d = fwd[f]
    return max(x['t'] for x in d.values()), max(x['end_t'] for x in d.values())
def is_prefill(f): return any(x['event'] == 'worker_execute_model' for x in fwd[f].values())
pre = [f for f in sorted(fwd) if is_prefill(f) and len(fwd[f]) == len(lanes)]
paced = [1000 * (span(f)[1] - span(f-1)[1]) for f in sorted(fwd) if f-1 in fwd and not is_prefill(f) and not is_prefill(f-1)]
out = []
for F in pre:
    # F+1 is the rendezvous with the prefill lane's post-processing; its
    # latest start is the first dummy launch after the GPU went idle.
    first = F + 1; f = F + 2; n = 1
    if f not in fwd: continue
    while f in fwd and not is_prefill(f) and 1000 * (span(f)[1] - span(f-1)[1]) < 12: n += 1; f += 1
    if f not in fwd or is_prefill(f): continue
    g = [1000 * (span(h)[1] - span(h-1)[1]) for h in range(f + 1, f + 6) if h in fwd and not is_prefill(h) and not is_prefill(h-1)]
    if len(g) < 3: continue
    g = statistics.median(g)
    elapsed = 1000 * (span(f - 1)[1] - span(first)[0])
    out.append(dict(prefill=F, first_fast=first, n_fast=n, elapsed_ms=round(elapsed, 1), g_ms=round(g, 2),
        cpu_ms=round(elapsed / max(n, 1), 2), depth=round(n - elapsed / g, 2)))
for o in out: print(o)
d = [o['depth'] for o in out]
print('waves', len(out), 'depth median', statistics.median(d), 'min', min(d), 'max', max(d),
      'g median', statistics.median(o['g_ms'] for o in out), 'cpu median', statistics.median(o['cpu_ms'] for o in out))
