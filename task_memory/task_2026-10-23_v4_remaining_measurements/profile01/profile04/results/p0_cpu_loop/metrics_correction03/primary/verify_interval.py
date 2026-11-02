"""Verify the allocated interval correction without executing spent harnesses."""
import ast
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
PRIMARY = Path(__file__).resolve().parent
MODE = sys.argv[1]
PLAN = json.loads((PRIMARY / (MODE + '_allocation.json')).read_text())
CHECKS = []
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
def exact(value):
    if isinstance(value, float):
        return ('float', value.hex())
    if isinstance(value, dict):
        return [(key, exact(item)) for key, item in value.items()]
    if isinstance(value, (list, tuple)):
        return [exact(item) for item in value]
    return value
def check(label, actual, expected):
    CHECKS.append(dict(label=label, actual=copy.deepcopy(actual), expected=copy.deepcopy(expected), passed=actual == expected))
    assert actual == expected, label
def definition(path, name):
    return next(node for node in ast.parse(Path(path).read_text()).body if isinstance(node, ast.FunctionDef) and node.name == name)
def pure(path, name):
    namespace = {}
    exec(compile(ast.Module(body=[definition(path, name)], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]
def metrics_check():
    spec = importlib.util.spec_from_file_location('interval_candidate', PLAN['source'])
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    frozen = pure(PLAN['auditor'], 'metrics')
    old = pure(adapter.ADAPTER, 'batched_metrics')
    if MODE == 'unit':
        event = {'ts': 5.125, 'dur': 0.375}
        fixtures = [[], [event, dict(event), event, {'ts': -1., 'dur': .25}],
                    [{'ts': 100., 'dur': 1e16}, {'ts': 0., 'dur': 1.}, {'ts': 1., 'dur': 1.}],
                    [{'ts': 8., 'dur': .25}, {'ts': 8., 'dur': .125}, {'ts': 3., 'dur': -.5}, {'ts': 2., 'dur': 0.}]]
        for index, events in enumerate(fixtures):
            groups = {('total', None): events, 'copy': list(events), None: events[::-1], 17: events * 2, 'empty': []}
            before = exact([events, groups])
            identities = [id(e) for e in events]
            expected = {key: frozen(members) for key, members in groups.items()}
            check(f'fixture{index}:all-metrics-floathex-order', exact(adapter.batched_metrics(events, groups)), exact(expected))
            check(f'fixture{index}:input-values', exact([events, groups]), before)
            check(f'fixture{index}:event-identities', [id(e) for e in events], identities)
        new_loop = next(n for n in ast.walk(definition(PLAN['source'], 'batched_metrics')) if isinstance(n, ast.For) and ast.unparse(n.target) == 'position')
        old_loop = next(n for n in ast.walk(definition(PLAN['auditor'], 'metrics')) if isinstance(n, ast.For))
        check('exact-union-recurrence-AST', [ast.dump(n) for n in new_loop.body[1:]], [ast.dump(n) for n in old_loop.body])
        stage = type('Stage', (), {'instrument': staticmethod(ast.parse)})()
        adapter.ACCEPTED.compose_metrics(stage)
        original = stage.instrument(Path(PLAN['auditor']).read_text())
        original_dump = ast.dump(original)
        changed = adapter.instrument_intervals(original)
        check('original-composed-AST-unmodified', ast.dump(original), original_dump)
        for tree in (original, changed):
            tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == 'batched_metrics')]
        check('all-composed-AST-outside-metrics', ast.dump(changed), ast.dump(original))
        return dict(fixtures=len(fixtures), source_gross_lines=len(Path(PLAN['source']).read_text().splitlines()))
    n = PLAN['events']
    events = [{'ts': ((i * 104729) % n) / 8, 'dur': ((i % 23) + 1) / 64} for i in range(n)]
    groups = {('total', None): events, ('status', 0): list(events)}
    groups.update({('stream', k): events[k::155] for k in range(155)})
    groups.update({('hierarchy', k): events[k::41] for k in range(41)})
    groups.update({('leaf', k): events[k:PLAN['leaf_memberships']:42] for k in range(42)})
    check('synthetic-group-count', len(groups), PLAN['groups'])
    check('synthetic-memberships', sum(map(len, groups.values())), PLAN['memberships'])
    started = time.monotonic()
    expected = old(events, groups)
    old_seconds = time.monotonic() - started
    started = time.monotonic()
    actual = adapter.batched_metrics(events, groups)
    new_seconds = time.monotonic() - started
    check('synthetic-exact-all-count-sum-union-floathex', exact(actual), exact(expected))
    return dict(events=n, groups=len(groups), memberships=PLAN['memberships'], old_seconds=old_seconds,
                new_seconds=new_seconds, relative_reduction=1 - new_seconds / old_seconds, order=['accepted-pure', 'candidate-pure'])
def control_check():
    control = PLAN['control']
    prefix = Path(PLAN['prefix'])
    command = ['timeout', '60s', '/usr/bin/python3', '-u', '-B', PLAN['source'], control['input_bindings']['trace_path'], str(prefix)]
    with prefix.with_suffix('.stdout').open('x') as output, prefix.with_suffix('.stderr').open('x') as errors:
        started = time.monotonic()
        result = subprocess.run(command, stdout=output, stderr=errors, cwd=PLAN['repo'])
    check('control-returncode', result.returncode, 0)
    new = json.loads(prefix.with_suffix('.json').read_text())
    old = json.loads(Path(control['retained_output_path']).read_text())
    check('child-actual30GiB', json.loads(prefix.with_suffix('.cgroup.json').read_text())['limits']['memory.limit_in_bytes'], '32212254720')
    index = prefix.with_suffix('.kernels.jsonl.gz')
    check('fresh-index-path', new['kernel_sample_index']['path'], str(index))
    check('fresh-index-sha256', new['kernel_sample_index']['sha256'], sha(index))
    check('fresh-elapsed-valid', math.isfinite(new['elapsed_seconds']) and 0 < new['elapsed_seconds'] < 60, True)
    check('fresh-rss-valid', new['maxrss_kib'] > 0, True)
    check('useful-steps', sum(row['useful'] for row in new['steps']), control['expected_useful_steps'])
    for payload in (new, old):
        del payload['elapsed_seconds'], payload['maxrss_kib']
        del payload['kernel_sample_index']['path'], payload['kernel_sample_index']['sha256']
    check('all-non-clock-JSON-floathex-order', exact(new), exact(old))
    digest, rows, size = hashlib.sha256(), 0, 0
    with gzip.open(index, 'rb') as fresh, gzip.open(control['retained_kernel_index_path'], 'rb') as retained:
        while True:
            a, b = fresh.read(1024 * 1024), retained.read(1024 * 1024)
            assert a == b, 'Every decompressed index byte must match'
            if not a:
                break
            digest.update(a)
            rows += a.count(b'\n')
            size += len(a)
    check('index-row-count', rows, new['kernel_sample_index']['rows'])
    return dict(command=command, enclosing_seconds=time.monotonic() - started, kernel_total=new['kernel_total'],
                index_rows=rows, index_bytes=size, decompressed_sha256=digest.hexdigest(), clean_gzip_eof=True)
started = time.monotonic()
receipt = dict(mode=MODE, used=1, ceiling=1, started_utc=datetime.now(timezone.utc).isoformat(), python=sys.version)
with (PRIMARY / (MODE + '_started.json')).open('x') as stream:
    json.dump(receipt, stream, indent=2)
try:
    for pin in PLAN['pins']:
        check('pin:' + pin['path'], sha(pin['path']), pin['sha256'])
    cgroup = Path('/proc/self/cgroup').read_text()
    memory_path = next(line.split(':', 2)[2] for line in cgroup.splitlines() if 'memory' in line.split(':', 2)[1].split(','))
    mount = next(line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines() if ' - cgroup ' in line and 'memory' in line.split(' - ', 1)[1].split()[2].split(','))
    group = Path(mount[4]) / Path(memory_path).relative_to(mount[3])
    receipt['cgroup'] = dict(group=str(group), actual_bytes=int((group / 'memory.limit_in_bytes').read_text()))
    check('parent-actual30GiB', receipt['cgroup']['actual_bytes'], 32212254720)
    check('original-python-version', sys.version, PLAN['python'])
    check('original-allocator-environment', {k: os.environ.get(k) for k in PLAN['environment']}, PLAN['environment'])
    receipt['result'] = control_check() if MODE.startswith('control') else metrics_check()
    receipt['status'] = 'PASS'
except BaseException as error:
    receipt.update(status='FAIL', error=repr(error))
    raise
finally:
    receipt.update(seconds=time.monotonic() - started, checks=CHECKS)
    with (PRIMARY / (MODE + '_result.json')).open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'checks'}), flush=True)
