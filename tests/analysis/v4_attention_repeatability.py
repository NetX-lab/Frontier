"""Analyze fixed-geometry repetitions with the production attention row reducer."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import time

from frontier.profiling.attention.dsv4_replay_rows import build_dsv4_replay_rows
from frontier.profiling.kernel_gap.dsv4_replay_gaps import ReplayRecords
from v4_attention_coverage import check_steps, plan_from_spec
from v4_attention_repeats import SCOPES, TIMING_PREFIXES, identity


def read_selected(path, case_ids):
    """Keep complete records for selected cases without loading unrelated calls."""
    groups = defaultdict(list)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            if case_ids is None or row['kind'] == 'engine' or row.get('case_id') in case_ids:
                groups[row['kind']].append(row)
    if len(groups['engine']) != 1:
        raise ValueError(f'{path}: expected one engine record')
    return ReplayRecords(str(path), groups['engine'][0], tuple(groups['step']),
                         tuple(groups['call']), tuple(groups['interval'])), {
        'path': str(path.resolve()), 'sha256': digest.hexdigest(),
        'selected_records': {key: len(rows) for key, rows in groups.items()},
    }


def error(value, reference):
    delta = value - reference
    return {'signed_difference_us': delta, 'absolute_difference_us': abs(delta),
            'relative_difference': None if reference == 0 else abs(delta / reference)}


def distribution(values):
    if len(values) < 2 or not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError('Repeated durations require at least two finite nonnegative values')
    mean = statistics.mean(values)
    cuts = statistics.quantiles(values, n=10, method='inclusive')
    split = len(values) // 2
    first, second = statistics.median(values[:split]), statistics.median(values[split:])
    return {'samples': len(values), 'values_us': values, 'mean_us': mean,
            'median_us': statistics.median(values), 'sample_stdev_us': statistics.stdev(values),
            'coefficient_of_variation': None if mean == 0 else statistics.stdev(values) / mean,
            'minimum_us': min(values), 'maximum_us': max(values), 'p10_us': cuts[0], 'p90_us': cuts[8],
            'split_first_median_us': first, 'split_second_median_us': second,
            'split_error': error(second, first)}


def row_key(row):
    return row['case_id'], row['attention_variant'], row['plans_tile_schedule']


def analyze(args):
    started = time.monotonic()
    plan = plan_from_spec(args.package_plan)
    mapping = json.loads(args.case_mapping.read_text())
    expected = defaultdict(set)
    for case_id, item in mapping.items():
        expected[item['original']].add(item['repeat'])
    if set(mapping) != {case.case_id for case in plan.cases}:
        raise ValueError('Mapping and plan have different case identities')
    originals = set(expected)
    reference_rows, reference_inputs = [], []
    for path in args.reference_raw:
        record, receipt = read_selected(path, originals)
        reference_rows.extend(build_dsv4_replay_rows(plan, [record]))
        reference_inputs.append(receipt)
    reference = {row_key(row): row for row in reference_rows}
    if len(reference) != len(reference_rows) or {key[0] for key in reference} != originals:
        raise ValueError('Reference rows have duplicate or missing case identities')
    with args.published_table.open() as stream:
        published = {(r['case_id'], r['attention_variant'], r['plans_tile_schedule']): r
                     for r in csv.DictReader(stream) if r['case_id'] in originals}
    reconstructed = 0
    for key, row in reference.items():
        table = published[(key[0], key[1], str(key[2]))]
        for column, value in row.items():
            if column.startswith('time_stats.') and column.endswith('.median'):
                if abs(float(table[column]) - value) > 1e-12:
                    raise ValueError(f'Published reference median differs: {key} {column}')
                reconstructed += 1
    measured_rows, measured_inputs, steps = [], [], []
    for path in args.raw:
        record, receipt = read_selected(path, None)
        measured_rows.extend(build_dsv4_replay_rows(plan, [record]))
        measured_inputs.append(receipt)
        steps.extend((str(path), i, step) for i, step in enumerate(record.steps, 1))
    geometry = check_steps(plan, steps)
    geometry = {key: value for key, value in geometry.items() if key not in ('windows', 'classes')}
    result = {'classification': 'DIAGNOSTIC_MEASUREMENT_REPEATABILITY',
              'reference_inputs': reference_inputs, 'measured_inputs': measured_inputs,
              'package_plan': str(args.package_plan.resolve()), 'geometry': geometry,
              'reconstructed_reference_medians': reconstructed,
              'published_table_sha256': hashlib.sha256(args.published_table.read_bytes()).hexdigest(),
              'identity_mismatches': [], 'count_mismatches': [], 'groups': [], 'csa_contrasts': []}
    groups = defaultdict(dict)
    for row in measured_rows:
        item = mapping[row['case_id']]
        renamed = {**row, 'case_id': item['original']}
        key = row_key(renamed)
        original = reference[key]
        if identity(renamed) != identity(original):
            left, right = json.loads(identity(renamed)), json.loads(identity(original))
            result['identity_mismatches'].append({'case_id': row['case_id'], 'group': list(key[1:]),
                'fields': {k: {'measured': left.get(k), 'reference': right.get(k)}
                           for k in left.keys() | right.keys() if left.get(k) != right.get(k)}})
        for field, value in original.items():
            if field.startswith(TIMING_PREFIXES[0]) and field.endswith(('.count', '.kernel_count')):
                if row.get(field) != value:
                    result['count_mismatches'].append({'case_id': row['case_id'], 'field': field,
                                                       'reference': value, 'measured': row.get(field)})
        if item['repeat'] in groups[key]:
            raise ValueError(f'Duplicate repeat: {key} {item["repeat"]}')
        groups[key][item['repeat']] = row
    if set(groups) != set(reference):
        raise ValueError('Reference and measured layer groups differ')
    for key, repeats in sorted(groups.items()):
        if set(repeats) != expected[key[0]]:
            raise ValueError(f'Missing repeat: {key}')
        for scope in SCOPES:
            column = f'time_stats.{scope}.median'
            original = reference[key]
            item = {'original_case_id': key[0], 'variant': key[1], 'plans_tile_schedule': key[2],
                    'scope': scope, 'layer_ids': json.loads(original['global_layer_ids'])}
            if column not in original:
                item['status'] = 'SCOPE_NOT_EXECUTED_IN_THIS_LAYER_GROUP'
            else:
                values = [1000 * repeats[i][column] for i in sorted(repeats)]
                ref = 1000 * original[column]
                item.update(status='MEASURED', reference_us=ref, **distribution(values))
                item['median_error'] = error(item['median_us'], ref)
                item['individual_errors'] = [error(value, ref) for value in values]
            result['groups'].append(item)
    for case_id in sorted(originals):
        first, reuse = (groups[(case_id, 'csa', planner)] for planner in (True, False))
        for scope in SCOPES:
            column = f'time_stats.{scope}.median'
            ref_first, ref_reuse = (reference[(case_id, 'csa', p)] for p in (True, False))
            if column not in ref_first or column not in ref_reuse:
                continue
            differences = [1000 * (first[i][column] - reuse[i][column]) for i in sorted(first)]
            reference_difference = 1000 * (ref_first[column] - ref_reuse[column])
            result['csa_contrasts'].append({'case_id': case_id, 'scope': scope,
                'reference_signed_difference_us': reference_difference,
                'repeat_signed_differences_us': differences,
                'same_sign_as_reference': sum(d * reference_difference > 0 for d in differences),
                'median_signed_difference_us': statistics.median(differences)})
    result['status'] = ('PASS_IDENTITY_AND_GEOMETRY' if geometry['prediction_matches']
                        and not result['identity_mismatches'] and not result['count_mismatches']
                        else 'FAIL_IDENTITY_OR_GEOMETRY')
    result['measured_rows'] = len(measured_rows)
    result['elapsed_seconds'] = time.monotonic() - started
    result['limits'] = ('Fixed request geometry with varying seeded prompt contents and persistent GPU state. '
                        'No A6 qualification or causal cache/clock/instrumentation claim.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-plan', type=Path, required=True)
    parser.add_argument('--case-mapping', type=Path, required=True)
    parser.add_argument('--reference-raw', type=Path, nargs='+', required=True)
    parser.add_argument('--raw', type=Path, nargs='+', required=True)
    parser.add_argument('--published-table', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3))
    result = analyze(args)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'measured_rows', 'elapsed_seconds')}))
    raise SystemExit(0 if result['status'] == 'PASS_IDENTITY_AND_GEOMETRY' else 1)


if __name__ == '__main__':
    main()
