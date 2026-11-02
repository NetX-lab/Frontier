"""Read-only probe of profile04 torch traces: per-step batch composition, step interval, GPU kernel span."""
import bisect, gzip, json, glob, re, statistics, sys
R = "/data/ycfeng/tmp/glm53_dsv4/profile02_results_readonly_7ba56n6z/artifacts/torch_traces"
pat = re.compile(r'execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)')
role, dp, tp = sys.argv[1], sys.argv[2], sys.argv[3]
f = sorted(glob.glob(f'{R}/{role}/{dp}/{dp}_pp0_tp{tp}_*'))[0]
ev = json.load(gzip.open(f))['traceEvents']
gpu = sorted((e for e in ev if e.get('cat') == 'gpu_user_annotation' and e['name'].startswith('execute_context')), key=lambda e: e['ts'])
cpu = sorted((e for e in ev if e.get('cat') == 'user_annotation' and e['name'].startswith('execute_context')), key=lambda e: e['ts'])
ker = sorted((e['ts'], e['ts'] + e['dur']) for e in ev if e.get('cat') == 'kernel')
ks = [k[0] for k in ker]
rows = []
for i, g in enumerate(gpu):
    end = gpu[i + 1]['ts'] if i + 1 < len(gpu) else g['ts'] + 1e9
    a, b = bisect.bisect_left(ks, g['ts']), bisect.bisect_left(ks, end)
    seg = ker[a:b]
    span = (max(k[1] for k in seg) - seg[0][0]) if seg else 0
    busy = sum(k[1] - k[0] for k in seg)
    m = pat.match(g['name'])
    rows.append(dict(ctx=int(m[1]), ctx_tok=int(m[2]), gen=int(m[3]), gen_tok=int(m[4]), ts=g['ts'],
                     interval=(gpu[i + 1]['ts'] - g['ts']) if i + 1 < len(gpu) else None, span=span, busy=busy, kernels=len(seg)))
print(f, len(cpu), 'cpu steps', len(gpu), 'gpu steps')
from collections import Counter
print('composition', Counter((r['ctx'], r['ctx_tok'], r['gen'], r['gen_tok']) for r in rows).most_common(12))
for key in sorted({(r['ctx'], r['gen']) for r in rows}):
    rs = [r for r in rows if (r['ctx'], r['gen']) == key]
    iv = [r['interval'] for r in rs if r['interval']]
    print(key, 'n', len(rs), 'interval_ms med', round(statistics.median(iv) / 1e3, 3) if iv else None,
          'span_ms med', round(statistics.median(r['span'] for r in rs) / 1e3, 3), 'busy_ms med', round(statistics.median(r['busy'] for r in rs) / 1e3, 3),
          'kernels med', statistics.median(r['kernels'] for r in rs))
if len(sys.argv) > 4:
    for r in rows[:int(sys.argv[4])]: print(r)
