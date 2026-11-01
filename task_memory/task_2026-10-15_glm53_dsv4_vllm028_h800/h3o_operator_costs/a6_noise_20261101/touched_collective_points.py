"""Per-pass spread of the uniform collective points that the F2 P2 replays actually priced.

Joins each exact uniform-collective census row with the retained A6 collective point
(same curve key and payload bytes), checks that the census price equals the pooled
measured median, and reports how far each single pass's median lies from it. Unequal
SP rows use a separate exact table and are counted but not joined. Diagnostic only.
"""
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent.parent
curves = json.loads((root / 'a6_collectives_20261101/result.json').read_text())['curves']
points = {(tuple(curve['key']), point['bytes']): point for curve in curves for point in curve['points']}
output = {'classification': 'DIAGNOSTIC_TOUCHED_COLLECTIVE_PASS_SPREAD', 'runs': {}}
for scale in ('p2_32', 'p2_64'):
    census = json.loads((root / f'f_comparisons_20261101/f2_{scale}_collectives.json').read_text())['rows']
    rows, unequal_ms = [], 0.0
    for row in census:
        assert row['exact'], row
        if row['unequal_rows'] is not None:
            unequal_ms += row['predicted_total_ms']
            continue
        key = (tuple(row['curve_key']), row['requested_bytes'])
        point = points[key]
        assert abs(point['measured_us'] - row['lower_time_us']) < 1e-9, key
        by_pass = {}
        for source in point['source_rows']:
            by_pass.setdefault(source['pass'], []).append(float(source['median_us']))
        pooled = point['measured_us']
        passes = {name: statistics.median(values) for name, values in sorted(by_pass.items())}
        deviation = max(abs(value - pooled) / pooled for value in passes.values())
        rows.append({'role': row['role'], 'curve_key': row['curve_key'], 'bytes': row['requested_bytes'],
                     'call_count': row['call_count'], 'predicted_total_ms': row['predicted_total_ms'],
                     'pooled_median_us': pooled, 'pass_medians_us': passes,
                     'pass_range_over_pooled': (max(passes.values()) - min(passes.values())) / pooled,
                     'max_pass_deviation_from_pooled': deviation})
    uniform_ms = sum(row['predicted_total_ms'] for row in rows)
    above = [row for row in rows if row['max_pass_deviation_from_pooled'] > 0.10]
    output['runs'][scale] = {
        'uniform_points': len(rows), 'uniform_predicted_ms': uniform_ms, 'unequal_sp_predicted_ms': unequal_ms,
        'points_with_a_pass_more_than_10_percent_from_pooled': len(above),
        'predicted_ms_share_of_those_points': sum(row['predicted_total_ms'] for row in above) / uniform_ms,
        'weighted_max_pass_deviation': sum(row['predicted_total_ms'] * row['max_pass_deviation_from_pooled']
                                           for row in rows) / uniform_ms,
        'rows': sorted(rows, key=lambda row: -row['predicted_total_ms'])}
    print(scale, {k: v for k, v in output['runs'][scale].items() if k != 'rows'})
    for row in output['runs'][scale]['rows']:
        print(f"  {row['role']:8s} {'/'.join(map(str, row['curve_key'])):26s} {row['bytes']:>10d} "
              f"{row['predicted_total_ms']:>11.1f} ms  pooled {row['pooled_median_us']:9.3f} us  "
              f"range {100 * row['pass_range_over_pooled']:6.2f}%  max dev {100 * row['max_pass_deviation_from_pooled']:6.2f}%")
target = root / 'a6_noise_20261101/touched_collective_points.json'
assert not target.exists()
target.write_text(json.dumps(output, indent=2) + '\n')
