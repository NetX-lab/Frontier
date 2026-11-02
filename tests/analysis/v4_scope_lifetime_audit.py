#!/usr/bin/env python3
"""Reproduce the frozen scope probe predicates using retained scalar evidence only."""

import ast
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(scope, output):
    seal = json.loads((scope / 'package_primary/root_seal.json').read_text())
    package = Path(seal['package'])
    assert sha(package / 'manifest.json') == seal['manifest_sha256']
    manifest = json.loads((package / 'manifest.json').read_text())
    assert manifest['files'] == seal['inputs']
    for path, expected in manifest['files'].items():
        assert sha(package / path) == expected, path
    source = scope / 'probe_scope_lifetime.py'
    assert sha(source) == seal['source_sha256']
    spec = importlib.util.spec_from_file_location('frozen_scope_source', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = scope / 'results/raw'
    inventory = json.loads((scope / 'results/cloud_inventory.json').read_text())
    for row in inventory['files']:
        path = raw / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    report = json.loads((raw / 'probe/probe_report.json').read_text())
    stages = [json.loads(line) for line in (raw / 'diagnostics/stages.jsonl').read_text().splitlines()]
    checks = []

    def check(name, value):
        checks.append({'name': name, 'pass': bool(value)})
        assert value, name

    check('no_errors_or_fault', not report['errors'] and not (raw / 'diagnostics/fault.log').read_bytes())
    check('stage_sequence', [s['sequence'] for s in stages] == list(range(1, len(stages) + 1)))
    check('stage_clock_order', all(a['monotonic_ns'] < b['monotonic_ns'] for a, b in zip(stages, stages[1:])))
    observe = report['observe']
    check('observe_errors_empty', not observe['errors'])
    capture = observe['callback_capture']
    rows = [s['result'] for s in stages if s['event'] == 'callback']
    check('callback_rows_match_stage_sink', rows == capture['events'])
    check('subscription_closed', capture['subscribed'] and capture['unsubscribed'] and
          not capture['subscription_active'] and not capture['errors'] and not observe['profile_errors'])
    created = [r for r in rows if r['callback_id'] == 13]
    clones = [r for r in rows if r['callback_id'] == 20]
    execs = [r for r in rows if r['callback_id'] == 18]
    count = len(module.NAMES)
    check('callback_kind_closure', len(rows) == len(created) + len(clones) + len(execs) and
          len(clones) == count and len(execs) == 1)
    relation = execs[0]
    observed_stages = [s for s in stages if s['sequence'] < next(s['sequence'] for s in stages
                       if s['event'] == 'control.capture.start')]
    windows = [s['result'] for s in observed_stages if s['event'] == 'op.end']
    check('window_sink_identity', windows == observe['windows'])
    check('window_set', len(windows) == count and {w['scope'] for w in windows} == set(module.NAMES) and
          {w['slot'] for w in windows} == set(range(count)))
    starts = [s for s in observed_stages if s['event'] == 'op.start']
    check('op_start_count', len(starts) == count)
    for window in windows:
        expected_start = {k: v for k, v in window.items() if k != 'end'}
        matches = [s for s in starts if s['result'] == expected_start]
        end = next(s for s in observed_stages if s['event'] == 'op.end' and s['result'] == window)
        check('op_start_end_pair_' + str(window['slot']), len(matches) == 1 and matches[0]['sequence'] < end['sequence'])
    links, joins, threads = [], [], []
    for clone in clones:
        selected = [r for r in created if (r['node'], r['node_id']) ==
                    (clone['original_node'], clone['original_node_id'])]
        check('full_original_tuple_' + str(clone['node_id']), len(selected) == 1)
        original = selected[0]
        check('kernel_type_and_source_graph_' + str(original['node_id']),
              original['node_type'] == original['queried_type'] == 0 and
              (original['graph'], original['graph_id']) == (relation['graph'], relation['graph_id']))
        check('payload_scalar_identity_' + str(clone['node_id']),
              original['payload']['node'] == original['node'] and original['payload']['graph'] == original['graph'] and
              clone['payload']['node'] == clone['node'] and clone['payload']['originalNode'] == original['node'] and
              clone['payload']['originalGraph'] == original['graph'])
        matches = [w for w in windows if w['start'] <= original['monotonic_ns'] < w['end']]
        check('unique_fresh_scope_' + str(original['node_id']), len(matches) == 1 and
              matches[0]['scope'] == original['scope'])
        threads.append({key: original[key] == observe['capture_frame'][key]
                        for key in ('native_id', 'python_ident')})
        check('separate_thread_observations_' + str(original['node_id']),
              all(isinstance(original[key], int) and original[key] > 0 for key in ('native_id', 'python_ident')))
        links.append({'source': original, 'clone': clone})
        joins.append({'scope': original['scope'], 'source_node': original['node'], 'source_full_id': original['node_id'],
                      'clone_node': clone['node'], 'clone_full_id': clone['node_id'],
                      'callback_ns': original['monotonic_ns'], 'window_start': matches[0]['start'], 'window_end': matches[0]['end']})
    check('source_clone_unique', all(len({r[kind][field] for r in links}) == count and
          all(r[kind][field] > 0 for r in links) for kind in ('source', 'clone') for field in ('node', 'node_id')))
    check('source_scope_coverage', {r['source']['scope'] for r in links} == set(module.NAMES))
    check('producer_derived_rows', links == observe['links'] and threads == observe['thread_observations'])
    unmatched = sum((r['node'], r['node_id']) not in {(c['original_node'], c['original_node_id']) for c in clones}
                    for r in created)
    check('unmatched_count_only', unmatched == observe['unmatched_created_count'])
    check('callbacks_before_capture_end', all(r['monotonic_ns'] < observe['capture_end']['monotonic_ns'] for r in rows))
    tree = ast.parse(source.read_text())
    lifetime = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'lifetime')
    source_queries = [n for n in ast.walk(lifetime) if isinstance(n, ast.Call) and
                      isinstance(n.func, ast.Attribute) and n.func.attr == 'raw_cuda_graph']
    check('single_negative_source_api_in_frozen_source', len(source_queries) == 1)
    check('default_false_lifetime', observe['default_constructor'] and observe['capture_complete'] and
          observe['raw_source_runtime_error'] is True and observe['raw_exec'] == relation['graph_exec'])
    topologies = []
    for mode in ('observe', 'control'):
        phase = report[mode]
        trace = json.loads((raw / f'probe/{mode}_trace.json').read_text())
        launch, kernels = module.probe.target_kernels(trace['traceEvents'])
        ids = [int(k['args']['graph node id']) for k in kernels]
        check(mode + '_raw_trace_join', launch == phase['target_launch'] and kernels == phase['target_kernels'])
        check(mode + '_complete_topology', len(kernels) == len(set(ids)) == count and
              len({int(k['args']['graph id']) for k in kernels}) == 1 and phase['replay_complete'])
        if mode == 'observe':
            check('direct_full_id_replay', set(ids) == {c['node_id'] for c in clones} and
                  {int(k['args']['graph id']) for k in kernels} == {relation['exec_graph_id']})
        closes = [s for s in stages if s['event'] == mode + '.close']
        kineto = [s for s in stages if s['event'] == mode + '.kineto.start']
        check(mode + '_close_before_kineto', len(closes) == len(kineto) == 1 and closes[0]['result'] is True and
              closes[0]['sequence'] < kineto[0]['sequence'])
        check(mode + '_trace_source_path', trace['traceName'] == phase['trace_path'] ==
              inventory['worker_root'] + f'/probe/{mode}_trace.json')
        topologies.append([(k['name'], k['args']['grid'], k['args']['block']) for k in kernels])
    control = report['control']
    check('unsubscribed_control', control['callback_capture'] is None and not control['errors'] and not control['profile_errors'])
    check('topology_identity', topologies[0] == topologies[1] and control['topology'])
    check('numeric_identity', observe['numeric_identity'] and control['numeric_identity'] and
          report['control_identity'] and observe['outputs'] == control['outputs'])
    check('shape_and_expected_outputs', all(len(values) == report['elements'] for values in observe['outputs']) and
          observe['outputs'] == [[value] * report['elements'] for value in (2.0, 2.0, 3.0, 3.0, 2.0)])
    result = {'status': 'PASS_SCOPED_RAW_REPRODUCTION', 'observed_utc': datetime.now(timezone.utc).isoformat(),
              'predicates': {f'P{i}': 'PASS' for i in range(1, 7)}, 'checks': checks, 'joins': joins,
              'thread_equalities_observed_only': threads, 'created_rows': len(created), 'unmatched_created_count': unmatched,
              'control': 'PASS_TOPOLOGY_OUTPUT_IDENTITY_ONLY', 'producer_control_verdict_preserved': control['verdict'],
              'manifest_sha256': seal['manifest_sha256'], 'source_sha256': seal['source_sha256'],
              'audit_source_sha256': sha(Path(__file__)), 'raw_files': inventory['files'],
              'limits': 'One image/driver/shape live-TLS diagnostic only. No native ownership, timing/non-perturbation, dependency, cost or reference admission.'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'checks': len(checks), 'joins': len(joins), 'unmatched': unmatched}))


if __name__ == '__main__':
    audit(Path(sys.argv[1]), Path(sys.argv[2]))
