"""Native graph-mode census on the first node of each role, rank 0.

Usage: python3 mode_census.py <run artifacts dir>... (P0, P2-32, P2-64).
Prints the final descriptor modes per counter kind, the desired-mode
combinations that differ across DP lanes, and modes by classifier has_prefill.
"""
import collections, json, pathlib, sys

for b in sys.argv[1:]:
    base = pathlib.Path(b)
    for role in ('prefill', 'decode'):
        modes = collections.Counter(); pairs = collections.Counter(); hp = collections.Counter()
        for path in sorted(base.glob('nodes/*/serving/rank0.json')):
            d = json.loads(path.read_text())
            if d.get('role') != role:
                continue
            for r in d['counters']['rows']:
                if r.get('kind') not in ('serving', 'dummy'):
                    continue
                de = r['dispatch_evidence']; n = r.get('successful_forwards') or 0
                m = de['dispatch']['descriptor']['cg_mode']
                modes[(r['kind'], m)] += n
                desired = tuple(de['dp_post_all_reduce'][1])
                if len(set(desired)) > 1:
                    pairs[(desired, m)] += n
                c = (de.get('classifier') or {}).get('inputs') or {}
                hp[(r['kind'], c.get('has_prefill'), m)] += n
            break
        print(base.parent.name if base.name == 'artifacts' else base.name, role)
        print('  final modes (kind, mode): forwards', dict(modes))
        print('  desired-mode differences', dict(pairs.most_common(6)))
        print('  (kind, has_prefill, mode)', dict(hp))
