"""Check whether each retained pass preserves the pooled curve's LOO failure."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

from frontier.cc_backend.backends.measured_cc_backend import _Curve
from v4_attention_heldout import collective_heldout

root = Path(__file__).resolve().parent
result = json.loads((root / 'result.json').read_text())
comparisons = []
for curve in result['curves']:
    points_by_pass = defaultdict(dict)
    for point in curve['points']:
        samples = defaultdict(list)
        for row in point['source_rows']:
            samples[row['pass']].append(float(row['median_us']))
        for run, values in samples.items():
            points_by_pass[run][point['bytes']] = median(values)
    passes = []
    for run, points in sorted(points_by_pass.items()):
        sizes = tuple(sorted(points))
        assert sizes == tuple(p['bytes'] for p in curve['points'])
        individual = _Curve(sizes, tuple(points[size] for size in sizes),
                            tuple(curve['production_summary']['calls']))
        summary = collective_heldout(dict(individual.describe(), key=curve['key']))
        passes.append(dict(pass_index=run, **summary))
    comparisons.append({'key': curve['key'], 'passes': passes})
output = root / 'pass_comparison.json'
assert not output.exists()
output.write_text(json.dumps({'classification': 'DIAGNOSTIC_PASS_COMPARISON',
                              'curves': comparisons}, indent=2) + '\n')
for curve in comparisons:
    print(curve['key'], [(p['pass_index'], p['verdict'],
                         p['heldout']['median_absolute'], p['heldout']['p90_absolute'])
                        for p in curve['passes']])
