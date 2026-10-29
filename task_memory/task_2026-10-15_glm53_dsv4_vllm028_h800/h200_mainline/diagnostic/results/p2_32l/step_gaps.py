"""Scratch: decode-lane schedule intervals while decoding, to look for periodic observer flush stalls."""
import json, sys, statistics
from pathlib import Path
for f in sorted(Path(sys.argv[1]).glob('nodes/*/scheduler_lifecycle/*.jsonl')):
    recs = [json.loads(l) for l in open(f)]
    if recs[0]['role'] != 'decode':
        continue
    sch = [r for r in recs if r['event'] == 'schedule']
    gaps = [((b['mono'] - a['mono']) * 1e3, a, b) for a, b in zip(sch, sch[1:]) if a.get('cached', 0) >= 4 and b.get('cached', 0) >= 4]
    g = sorted(x for x, _, _ in gaps); n = len(g)
    print(f.name, 'n', n, 'p50', round(g[n // 2], 2), 'p90', round(g[int(.9 * n)], 2), 'p99', round(g[int(.99 * n)], 2), 'max', round(g[-1], 2), 'mean', round(statistics.mean(g), 2))
    big = [(round(x, 1), round(a['t'] % 1000, 2)) for x, a, b in gaps if x > 30]
    print('  gaps > 30 ms:', len(big), big[:15])
    sd = [r['dur_ms'] for r in sch]; ud = [r['dur_ms'] for r in recs if r['event'] == 'update']
    print('  schedule dur_ms mean', round(statistics.mean(sd), 3), 'update dur_ms mean', round(statistics.mean(ud), 3))
