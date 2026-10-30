"""Exploratory per-prefill decomposition from the wave run copy (lead scratch)."""
import collections, json, statistics, sys
from pathlib import Path

run = Path(sys.argv[1])

def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]

engines, workers = {}, {}
for p in sorted(run.glob('nodes/*/scheduler_lifecycle/*.jsonl')):
    r = rows(p); s = next(x for x in r if x['event'] == 'start')
    if s['role'] == 'prefill':
        engines[s['dp_rank']] = r
for p in sorted(run.glob('nodes/*/worker_dp_wave/*.jsonl')):
    r = rows(p); s = next(x for x in r if x['event'] == 'worker_wave_start')
    if s['role'] == 'prefill':
        workers[s['dp_rank']] = r

# Worker forwards keyed by forward number: (kind, record, sync, reduce) with wall times.
def wall(rec, mono):
    return rec['t'] + (mono - rec['mono'])
fw = {}
for rank, r in workers.items():
    d = {}
    for x in r:
        if x['event'] in ('worker_execute_model', 'worker_dummy_batch'):
            c = x.get('calls', {})
            red = c.get('dp_cpu_all_reduce')
            if red is None:
                continue
            d[red['forward']] = dict(kind=x['event'], t=x['t'], end_t=x['end_t'],
                tokens=x.get('tokens'), ids=x.get('request_ids', []),
                red_t=wall(x, red['mono']), red_end=wall(x, red['end_mono']),
                stage=c.get('dp_padding_sync', {}).get('stage'), values=red.get('values'))
    fw[rank] = d
print('forwards per worker', {k: (min(v), max(v), len(v)) for k, v in fw.items()})

out = []
for rank, r in engines.items():
    pend = {}
    for x in r:
        if x['event'] == 'schedule' and x.get('tokens') == 6144:
            pend[x['step']] = x
        elif x['event'] == 'update' and x.get('step') in pend:
            sch = pend.pop(x['step'])
            rid = sch['new'][0][0] if sch.get('new') else None
            if rid is None or 'warmup' in rid:
                continue
            ex = [(f, v) for f, v in fw[rank].items() if rid in v['ids']]
            if len(ex) != 1:
                out.append(dict(rid=rid, lane=rank, missing='execute')); continue
            f, v = ex[0]
            peers = {p: fw[p].get(f) for p in fw if p != rank}
            ok = all(pv is not None and pv['stage'] == 'serving' for pv in peers.values())
            join = max([v['red_t']] + [pv['red_t'] for pv in peers.values() if pv]) if ok else None
            out.append(dict(rid=rid, lane=rank, forward=f, s=sch['t'], e=x['t'],
                step_ms=(x['t'] - sch['t']) * 1000,
                sched_to_reduce_ms=(v['red_t'] - sch['t']) * 1000,
                reduce_ms=(v['red_end'] - v['red_t']) * 1000,
                reduce_end_to_exec_end_ms=(v['end_t'] - v['red_end']) * 1000,
                exec_end_to_update_ms=(x['t'] - v['end_t']) * 1000,
                last_join_lane=max(peers, key=lambda p: peers[p]['red_t']) if ok else None,
                peer_kinds=collections.Counter(pv['kind'] for pv in peers.values() if pv),
                peer_tokens=[pv['tokens'] for pv in peers.values() if pv],
                values=v['values']))
out.sort(key=lambda o: o.get('s', 0))
prev = None
for o in out:
    if 's' in o:
        o['gap_ms'] = None if prev is None else (o['s'] - prev) * 1000
        prev = o['e']
good = [o for o in out if 'step_ms' in o]
print('formal', len(out), 'decomposed', len(good))
for key in ('step_ms', 'sched_to_reduce_ms', 'reduce_ms', 'reduce_end_to_exec_end_ms', 'exec_end_to_update_ms'):
    vals = [o[key] for o in good]
    print(f'{key:28s} mean {statistics.mean(vals):8.1f} min {min(vals):8.1f} max {max(vals):8.1f}')
json.dump(out, open(sys.argv[2], 'w'), indent=1, default=str)
