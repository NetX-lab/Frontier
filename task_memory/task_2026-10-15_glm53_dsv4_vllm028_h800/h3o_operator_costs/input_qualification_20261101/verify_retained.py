"""Reuse retained qualification checkers and verify their concrete input bindings."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_inputs import digest, load, module, TASK, RUNS, B29
from frontier.profiling.collectives.nixl_kv_read_benchmark import (
    KVRegionLayout, read_cases, summarize_case)
from v4_collective_exact_repeatability import measured_points, check_replay


def check_nixl():
    original = load(TASK / 'h3o_operator_costs/b30b/f3_nixl02_20261101/verification.json')
    sources = original['download']
    for item in sources:
        assert digest(item['local_path']) == item['sha256']
        assert Path(item['local_path']).stat().st_size == item['bytes']
    root = Path(sources[0]['local_path']).parent
    plan = load(root / 'nixl_plan.json')['config']
    layout = KVRegionLayout.from_config(plan['kv_cache'])
    cases = read_cases(plan['grid'], layout, plan['tensor_parallel_size'])
    expected = {(p, c.name): c for p in range(plan['passes']) for c in cases}
    raw = [json.loads(line) for line in (root / 'nixl_reads.jsonl').read_text().splitlines()]
    csv_rows = list(csv.DictReader((root / 'nixl_summary.csv').open(newline='')))
    assert len(raw) == len(csv_rows) == len(expected) == 240
    actual = {(r['pass'], r['case']): r for r in raw}
    assert set(actual) == set(expected)
    by_csv = {(int(r['pass']), r['case']): r for r in csv_rows}
    trials = notifications = 0
    tails = defaultdict(list)
    for key, case in expected.items():
        record = actual[key]
        assert record['bytes_per_reader'] == case.blocks_per_request * case.requests_in_flight * layout.block_bytes
        assert record['descriptors_per_request'] == case.blocks_per_request * layout.num_regions
        assert len(record['readers']) == case.active_pairs
        assert {r['rank'] for r in record['readers']} == set(range(8, 8 + case.active_pairs))
        assert record['notifications_expected'] == case.active_pairs * case.requests_in_flight * (plan['warmup'] + plan['repeats'])
        assert record['notifications_received'] == record['notifications_expected']
        notifications += record['notifications_received']
        for reader in record['readers']:
            assert reader['data_verified'] is True and len(reader['timings']) == plan['repeats']
            trials += len(reader['timings'])
            for timing in reader['timings']:
                assert all(math.isfinite(v) and v >= 0 for v in timing.values())
                tails[(case.active_pairs, case.requests_in_flight)].append(timing['complete_us'] / 1000)
        rebuilt = summarize_case(record)
        for column, value in rebuilt.items():
            stored = by_csv[key][column]
            assert (float(stored) == value if type(value) in (int, float) else stored == str(value)), (key, column)
    assert trials == 32400 and notifications == 214272
    return dict(status='PASS_INTEGRITY_NOT_TIMING_QUALIFICATION', sources=sources,
        cases=80, passes=3, raw_records=240, reader_trials=trials, notifications=notifications,
        csv_exactly_rebuilt=True, exact_formal_prompt_present=6144 in {c.prompt_tokens for c in cases},
        complete_batch_tail_ms=[dict(active_pairs=k[0], requests_in_flight=k[1], count=len(v),
                                   minimum=min(v), maximum=max(v)) for k,v in sorted(tails.items())],
        tail_limit='Observed full in-flight batch completion maxima, not per-request service bounds or serving-fabric guarantees.')


def check_collectives():
    frozen = load(TASK / 'h3o_operator_costs/collective_repeatability_20261101/result.json')
    uniform, exact = measured_points(Path(frozen['summary']['path']))
    assert digest(frozen['summary']['path']) == frozen['summary']['sha256']
    result = {}
    for case, previous in frozen['replays'].items():
        current = check_replay(case, Path(previous['census']), Path(previous['a8']), uniform, exact)
        assert current == previous
        assert current['verdict'] == 'PASS'
        result[case] = {k:current[k] for k in ['admission','repeatability','verdict']}
        result[case]['sources'] = [{ 'path': str(p), 'sha256': digest(p)} for p in
            [Path(previous['census']), Path(previous['a8']), Path(frozen['summary']['path'])]]
    return result


def check_f4():
    prior = load(TASK / 'h3o_operator_costs/qualification_codex_20261031/evidence/qualification_evidence.json')
    result = []
    for scale in ['32','64']:
        source = prior['sources'][f'P{scale}_F4_B28']
        assert digest(source['path']) == source['sha256']
        comparison = load(source['path'])
        roots = [Path(comparison[key]) for key in ['base','variant']]
        receipts = [load(root / 'receipt.json') for root in roots]
        assert all(r['status'] == 'FINISHED' for r in receipts)
        metric_roots = [Path(r['metrics_dir']) for r in receipts]
        rows = []
        for name, expected in comparison['metrics'].items():
            paths = [root / name for root in metric_roots]
            hashes = [digest(p) for p in paths]
            assert expected['equal'] and hashes[0] == hashes[1]
            assert [p.stat().st_size for p in paths] == expected['sizes']
            rows.append(dict(name=name, sha256=hashes[0], paths=list(map(str,paths))))
        result.append(dict(case=f'p2_{scale}', status='PASS_BYTE_IDENTICAL_F4_B28', metrics=rows))
    return result


def check_warm():
    comparison = load(TASK / 'h3o_operator_costs/two_fix_20261101/comparison/comparison.json')
    result = {}
    for case, record in comparison['cases'].items():
        for generation, root in [('b29', B29 / f'{case}_warm_k19b29'), ('new', RUNS / case)]:
            assert digest(root / 'receipt.json') == record['receipt_sha256'][generation]
        files = record['sources']['readings']['warm']
        old, new = load(files['b29']), load(files['new'])
        assert old['warm'] == new['warm'] and old['openers'] == new['openers']
        rows = list(new['warm']['metrics'])
        for repeat in new['warm']['repetitions'].values():
            rows.extend(repeat['metrics'])
        assert len(rows) == 16 and all(r['status'] == 'PASS' and math.isfinite(r['relative_error'])
                                     and r['relative_error'] <= 0.1 for r in rows)
        result[case] = dict(status='PASS_RETAINED_WARM_READING', metric_checks=16,
            max_absolute_relative_error=max(r['relative_error'] for r in rows),
            excluded_by_event=sum(len(v) for v in new['openers'].values()),
            openers=new['openers'], retained_request_repeat_count=new['warm']['retained_request_repeat_count'],
            all_request_verdict=record['verdicts']['all_request'],
            sources=[dict(path=p, sha256=digest(p)) for p in files.values()])
    return result


def check_reference_receipts():
    prior = load(TASK / 'h3o_operator_costs/qualification_codex_20261031/evidence/qualification_evidence.json')
    output = []
    for scale in ['32', '64']:
        records = {name:prior['sources'][f'P{scale}_{name}'] for name in
                   ['run_manifest', 'platform_terminal_receipt', 'run_check_status.groundtruth_clean']}
        for record in records.values():
            assert digest(record['path']) == record['sha256']
        manifest = load(records['run_manifest']['path'])
        terminal = load(records['platform_terminal_receipt']['path'])
        r4 = load(records['run_check_status.groundtruth_clean']['path'])['runs'][manifest['run_tag']]
        resources = manifest['resources']
        assert resources['gpu_type'] == 'H200' and resources['gpus'] == int(scale)
        assert resources['gpu'] * resources['replica'] == int(scale)
        assert len(r4['checks']) == 8 and all(v.startswith('PASS:') for v in r4['checks'].values())
        assert r4['rjob_name'] == terminal['job'] and terminal['status']['phase'] == 'Succeeded'
        replicas = terminal['replicas']
        assert len(replicas) == resources['replica']
        assert len({r['node'] for r in replicas}) == len(replicas)
        for replica in replicas:
            assert replica['phase'] == 'Succeeded'
            for container in replica['containers']:
                assert container['restartCount'] == 0 and container['state']['terminated']['exitCode'] == 0
                assert container['imageID'] == resources['image']
        output.append(dict(case=f'p2_{scale}', status='PASS_ORIGINAL_H200_R4_RECEIPT_BINDING',
            original_job=terminal['job'], gpus=resources['gpus'], nodes=resources['replica'],
            r4_checks=8, groundtruth_commit=manifest['groundtruth_commit'], image=resources['image'],
            container_gpu_hours=terminal['container_gpu_hours'],
            original_authorization=manifest['authorization'], source_records=records,
            limitation='Historical clean-reference/basic scope; not a future prelaunch promotion receipt.'))
    return output


def main(output):
    window = module('q1_nixl_window', TASK / 'h3o_operator_costs/two_fix_20261101/nixl_window_check.py')
    windows = [window.check_run(RUNS / case) for case in ['p2_32','p2_64']]
    assert all(r['status'] == 'NO_COMPLETION_CAN_MOVE' and r['kv_transfers'] == 128 for r in windows)
    result = dict(status='PASS_RETAINED_CPU_CHECKS_NOT_FORMAL_ACCEPTANCE',
        nixl_integrity=check_nixl(), nixl_completion_windows=windows,
        collectives=check_collectives(), measured_capacity_f4=check_f4(), warm_readings=check_warm(),
        original_native_reference_receipts=check_reference_receipts())
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['collectives','measured_capacity_f4']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
