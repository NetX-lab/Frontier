"""Attribute measured collective LOO errors to the retained payload samples.

Use the production curve loader and interpolation; report all points without
changing the A6 population, thresholds, measurement tables, or simulator inputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

from frontier.cc_backend.backends.measured_cc_backend import (
    MeasuredCCBackend, _Curve, _ROW_CURVES,
)
from frontier.types import ClusterType
from v4_attention_heldout import build_p2_config, collective_heldout


def source_points(path):
    points = defaultdict(lambda: defaultdict(list))
    with path.open() as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            tokens = tuple(int(v) for v in row['tokens_per_dp_rank'].split('/') if v)
            if (row['group'] == 'dp' or 'single_active' in row['scenario']
                    or not tokens or len(set(tokens)) != 1):
                continue
            measured = _ROW_CURVES.get((row['call'], None),
                                      _ROW_CURVES.get((row['op'], row['group'])))
            key = (*measured, int(row['group_size']))
            points[key][int(row['max_input_bytes'])].append({'csv_line': line, **row})
    return points


def describe_point(size, time_us, rows):
    values = [float(row['median_us']) for row in rows]
    if statistics.median(values) != time_us:
        raise ValueError(f'Source pooling differs from production at {size} bytes')
    return {
        'bytes': size, 'measured_us': time_us, 'source_rows': rows,
        'sample_medians_us': values,
        'sample_range_over_pooled_median': (max(values) - min(values)) / time_us,
        'sample_median_cv': (statistics.stdev(values) / statistics.mean(values)
                             if len(values) > 1 else None),
        'graph_modes': sorted({row['cuda_graph'] for row in rows}),
        'kernel_signatures': sorted({row['kernels'] for row in rows}),
    }


def analyze(args):
    started = time.monotonic()
    config, config_receipt = build_p2_config(args.scratch_dir)
    cluster = config.get_clusters()[ClusterType.DECODE]
    replica = cluster.replica_config
    backend = MeasuredCCBackend(cluster.cc_backend_config, ClusterType.DECODE,
                                replica.device, replica.network_device, replica.world_size)
    path = Path(backend._source_path)
    source = source_points(path)
    previous = json.loads(args.a6_result.read_text())
    expected = next(f for f in previous['other_families'] if f['family'] == 'uniform measured collectives')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected['input_sha256']:
        raise ValueError('A6 collective input changed')
    summaries = [collective_heldout(curve) for curve in backend.describe_curves()]
    if json.loads(json.dumps(summaries)) != expected['curves']:
        raise ValueError('Production curve summaries differ from frozen A6')
    curves = []
    for key, curve in sorted(backend._curves.items()):
        if set(source[key]) != set(curve.sizes):
            raise ValueError(f'Payload population differs: {key}')
        points = [describe_point(size, latency, source[key][size])
                  for size, latency in zip(curve.sizes, curve.times_us)]
        for index in range(1, len(points) - 1):
            point, lower, upper = points[index], points[index - 1], points[index + 1]
            held = _Curve(curve.sizes[:index] + curve.sizes[index + 1:],
                          curve.times_us[:index] + curve.times_us[index + 1:], curve.calls)
            predicted = held.predict_us(point['bytes'])
            delta = predicted - point['measured_us']
            point.update(predicted_loo_us=predicted, signed_error_us=delta,
                         absolute_error_us=abs(delta), absolute_relative_error=abs(delta / point['measured_us']),
                         lower_bytes=lower['bytes'], upper_bytes=upper['bytes'],
                         lower_ratio=point['bytes'] / lower['bytes'], upper_ratio=upper['bytes'] / point['bytes'],
                         neighbor_kernel_signatures_equal=(lower['kernel_signatures'] == point['kernel_signatures']
                                                           == upper['kernel_signatures']),
                         neighbor_graph_modes_equal=(lower['graph_modes'] == point['graph_modes'] == upper['graph_modes']))
        curves.append({'key': key, 'production_summary': curve.describe(), 'points': points})
    return {
        'classification': 'DIAGNOSTIC_COLLECTIVE_CURVE_ATTRIBUTION',
        'status': 'PASS_REPRODUCED_A6', 'source_table': str(path), 'source_table_sha256': digest,
        'a6_result': str(args.a6_result.resolve()),
        'a6_result_sha256': hashlib.sha256(args.a6_result.read_bytes()).hexdigest(),
        'config_receipt': config_receipt, 'curves': curves, 'a6_curves': summaries,
        'elapsed_seconds': time.monotonic() - started,
        'limits': ('Retained medians describe three passes and existing scenarios; '
                   'kernel/mode association does not isolate causality. No new accuracy gate or input promotion.'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a6-result', type=Path, required=True)
    parser.add_argument('--scratch-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'elapsed_seconds')}))
    for curve in result['curves']:
        points = [p for p in curve['points'] if 'absolute_relative_error' in p]
        worst = max(points, key=lambda p: p['absolute_relative_error'])
        print(curve['key'], 'worst', {k: worst[k] for k in (
            'bytes', 'measured_us', 'predicted_loo_us', 'absolute_relative_error',
            'neighbor_kernel_signatures_equal', 'neighbor_graph_modes_equal')})


if __name__ == '__main__':
    main()
