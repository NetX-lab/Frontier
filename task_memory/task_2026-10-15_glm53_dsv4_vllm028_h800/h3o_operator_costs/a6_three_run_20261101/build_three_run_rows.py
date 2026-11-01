"""Write the attention rows of three runs for every published workload.

The 989 workloads shared by G1h-02, G2h-01 and G2h-04 already have three runs as
rows.json files. The 432 supplement-only workloads have G1h-s01 plus the two
supplement passes, whose case identities map back to the originals through the
package's case mapping. This script rebuilds the supplement rows from the raw
records with the production row reducer, renames each pass row to its original
workload, and writes three combined rows.json files (reference, pass 1, pass 2)
in the format that ``v4_attention_multipass_table.py`` reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource

from frontier.profiling.attention.dsv4_replay_rows import build_dsv4_replay_rows
from v4_attention_coverage import plan_from_spec
from v4_attention_repeatability import read_selected


def row_key(row):
    return row['case_id'], row['attention_variant'], row['plans_tile_schedule']


def keyed(rows, label):
    result = {row_key(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f'{label}: duplicate workload/layer rows')
    return result


def write(path, rows):
    payload = json.dumps(rows).encode()
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    return {'path': str(path), 'rows': len(rows), 'sha256': hashlib.sha256(payload).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-plan', type=Path, required=True)
    parser.add_argument('--case-mapping', type=Path, required=True)
    parser.add_argument('--reference-raw', type=Path, nargs='+', required=True)
    parser.add_argument('--raw', type=Path, nargs='+', required=True)
    parser.add_argument('--shared-reference', type=Path, required=True)
    parser.add_argument('--shared-comparison', type=Path, nargs=2, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3))
    plan = plan_from_spec(args.package_plan)
    mapping = json.loads(args.case_mapping.read_text())
    if set(mapping) != {case.case_id for case in plan.cases}:
        raise ValueError('Mapping and plan have different case identities')
    originals = {item['original'] for item in mapping.values()}
    repeats = sorted({item['repeat'] for item in mapping.values()})
    if repeats != [1, 2]:
        raise ValueError(f'Expected repeats 1 and 2, got {repeats}')

    inputs = {'reference_raw': [], 'raw': []}
    reference_rows = []
    for path in args.reference_raw:
        record, receipt = read_selected(path, originals)
        reference_rows.extend(build_dsv4_replay_rows(plan, [record]))
        inputs['reference_raw'].append(receipt)
    reference = keyed(reference_rows, 'supplement reference')
    if {key[0] for key in reference} != originals:
        raise ValueError('Supplement reference rows miss original workloads')

    passes = {repeat: [] for repeat in repeats}
    for path in args.raw:
        record, receipt = read_selected(path, None)
        for row in build_dsv4_replay_rows(plan, [record]):
            item = mapping[row['case_id']]
            passes[item['repeat']].append({**row, 'case_id': item['original']})
        inputs['raw'].append(receipt)
    for repeat, rows in passes.items():
        if set(keyed(rows, f'pass {repeat}')) != set(reference):
            raise ValueError(f'Pass {repeat} rows differ from the supplement reference rows')

    shared = []
    for path in (args.shared_reference, *args.shared_comparison):
        payload = path.read_bytes()
        rows = json.loads(payload)
        shared.append(rows)
        inputs.setdefault('shared_rows', []).append(
            {'path': str(path.resolve()), 'sha256': hashlib.sha256(payload).hexdigest(), 'rows': len(rows)})
    shared_keys = set(keyed(shared[0], 'shared reference'))
    for rows, label in zip(shared[1:], ('shared comparison 1', 'shared comparison 2')):
        if set(keyed(rows, label)) != shared_keys:
            raise ValueError(f'{label} rows differ from the shared reference rows')
    if {key[0] for key in shared_keys} & originals:
        raise ValueError('Shared and supplement workloads overlap')

    supplement = [reference_rows, passes[1], passes[2]]
    outputs = {}
    for index, name in enumerate(('reference', 'pass1', 'pass2')):
        outputs[f'supplement_{name}'] = write(args.output_dir / f'supplement_{name}' / 'rows.json',
                                              supplement[index])
        outputs[f'combined_{name}'] = write(args.output_dir / f'combined_{name}' / 'rows.json',
                                            shared[index] + supplement[index])
    manifest = {'classification': 'DIAGNOSTIC_THREE_RUN_ROWS', 'inputs': inputs,
                'package_plan': str(args.package_plan.resolve()),
                'case_mapping': str(args.case_mapping.resolve()),
                'supplement_workloads': len(originals), 'supplement_rows': len(reference_rows),
                'shared_workloads': len({key[0] for key in shared_keys}), 'shared_rows': len(shared_keys),
                'combined_order': ['G1h-02 + G1h-s01', 'G2h-01 + supplement pass 1',
                                   'G2h-04 + supplement pass 2'],
                'outputs': outputs}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({name: (item['rows'], item['sha256'][:12]) for name, item in outputs.items()}))


if __name__ == '__main__':
    main()
