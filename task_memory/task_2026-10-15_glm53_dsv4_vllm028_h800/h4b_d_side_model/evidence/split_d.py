"""Scratch: split part d per formal D request into the NIXL transfer reports seen inside its READ window."""
import json, sys, statistics
from pathlib import Path
sys.path.insert(0, sys.argv[3])
import analyze_lifecycle as al

run = Path(sys.argv[1]); rows = [json.loads(l) for l in open(sys.argv[2])]
problems = []
cores = al.load_cores(run, problems)
out = []
for (role, dp), core in cores.items():
    if role != 'decode':
        continue
    timelines = al.decode_timelines(core['rows'])
    updates = [r for r in core['rows'] if r['event'] == 'update']
    windows = {rid: (t['read']['mono'], t['received']['mono']) for rid, t in timelines.items() if 'read' in t and 'received' in t}
    for row in rows:
        if row.get('d_dp_rank') != dp or row['missing']:
            continue
        rid = next(k for k in timelines if k.startswith(f"cmpl-{row['request_id']}-"))
        lo, hi = windows[rid]
        overlap = any(o != rid and o_lo < hi and o_hi > lo for o, (o_lo, o_hi) in windows.items())
        reports = [u for u in updates if lo < u['mono'] <= hi and u.get('kv_stats', {}).get('transfer_duration')]
        durations = [d for u in reports for d in u['kv_stats']['transfer_duration']]
        total_bytes = sum(b for u in reports for b in u['kv_stats'].get('bytes_transferred', ()))
        last = max(u['mono'] for u in reports) if reports else None
        out.append(dict(request_id=row['request_id'], dp=dp, lane_state=row['lane_state'], d=row['d'], overlap=overlap,
                        handles=len(durations), bytes=total_bytes, max_transfer_ms=max(durations) * 1e3 if durations else None,
                        read_to_last_report_ms=(last - lo) * 1e3 if last else None,
                        last_report_to_received_ms=(hi - last) * 1e3 if last else None))
def q(xs):
    xs = sorted(x for x in xs if x is not None); n = len(xs)
    return dict(n=n, mean=round(statistics.mean(xs), 2), p50=round(xs[n // 2], 2), p95=round(xs[int(0.95 * (n - 1))], 2), min=round(xs[0], 2), max=round(xs[-1], 2)) if n else dict(n=0)
summary = {}
for name, sel in (('all', lambda r: True), ('no_overlap', lambda r: not r['overlap']), ('overlap', lambda r: r['overlap']),
                  ('no_overlap_decoding', lambda r: not r['overlap'] and r['lane_state'] == 'decoding'),
                  ('no_overlap_idle', lambda r: not r['overlap'] and r['lane_state'] == 'idle')):
    sub = [r for r in out if sel(r)]
    summary[name] = {k: q([r[k] for r in sub]) for k in ('d', 'handles', 'bytes', 'max_transfer_ms', 'read_to_last_report_ms', 'last_report_to_received_ms')}
print(json.dumps(dict(problems=problems, summary=summary), indent=1))
Path(sys.argv[4]).write_text(''.join(json.dumps(r) + '\n' for r in out))
