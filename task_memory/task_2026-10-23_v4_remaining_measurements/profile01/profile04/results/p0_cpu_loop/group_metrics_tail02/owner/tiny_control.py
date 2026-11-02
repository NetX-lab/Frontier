"""Run the sole new AST/fixture control; never import or execute old drivers."""
import ast
import copy
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO, OWNER = Path.cwd(), Path(__file__).resolve().parent
TIMER = REPO / 'tests/analysis/v4_trace_audit_group_metrics_timing.py'
INPUTS = json.loads((OWNER / 'tiny_inputs.json').read_text())
START, checks = time.monotonic(), []
result = {'command': ['sudo', '-n', 'systemd-run', '--scope', '-p', 'MemoryMax=30G', 'timeout', '60s', '/usr/bin/python3', '-I', '-B', str(Path(__file__).resolve())],
          'cwd': str(REPO), 'python': sys.version, 'executable': sys.executable,
          'conda': 'not used; /usr/bin/python3', 'tiny_invocations': 1, 'full_invocations': 0, 'checks': checks}


def check(name, expected, actual):
    row = copy.deepcopy(json.loads(json.dumps(dict(name=name, expected=expected, actual=actual))))
    row['status'] = 'PASS' if row['expected'] == row['actual'] else 'FAIL'
    checks.append(row)
    assert row['status'] == 'PASS', name


def definition(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def function(node, namespace):
    tree = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]))
    exec(compile(tree, '<tiny-copied-function>', 'exec'), namespace)
    return namespace[node.name]


try:
    result['execution_hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (TIMER, Path(__file__), OWNER / 'tiny_inputs.json')}
    group_path = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0:'))
    result['actual_memory_limit_bytes'] = int((Path('/sys/fs/cgroup') / group_path.lstrip('/') / 'memory.max').read_text())
    check('actual_memory_limit_bytes', 32212254720, result['actual_memory_limit_bytes'])
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in INPUTS['pins']}
    check('immutable_pins_before', INPUTS['pins'], hashes)
    ns = {'__file__': str(TIMER), '__name__': 'tiny_ast_only'}
    exec(compile(TIMER.read_bytes(), str(TIMER), 'exec'), ns)
    prefix, wrappers = ns['PREFIX'] + ':', (ns['GLOBAL_SORT'], ns['GROUP_SORT'])
    constructors = {k: v for k, v in vars(ast).items() if isinstance(v, type) and issubclass(v, ast.AST)}
    accepted = eval(json.loads(Path(INPUTS['baseline_path']).read_text())[INPUTS['baseline_key']], {'__builtins__': {}, **constructors})
    stage = SimpleNamespace(instrument=lambda source: copy.deepcopy(accepted))
    ns['compose_timing'](stage)
    timed = stage.instrument('tiny AST copy')
    stripped = copy.deepcopy(timed)
    restored = definition(stripped, 'batched_metrics')
    for body in (restored.body, next(n for n in restored.body if isinstance(n, ast.For)).body):
        body[:] = [n for n in body if not (isinstance(n, ast.FunctionDef) and n.name in wrappers)
                   and not ast.unparse(n).startswith(f"_obs('{prefix}")]
    for call in ast.walk(restored):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in wrappers:
            call.func, call.args = call.args[0], call.args[1:]
    check('stripped_complete_module_AST', ast.dump(accepted), ast.dump(stripped))
    restored.body.pop(-2)
    restored.body.pop(1)
    frozen = definition(ast.parse(ns['ADAPTER'].read_bytes()), 'batched_metrics')
    check('stripped_frozen_source_AST', ast.dump(frozen), ast.dump(restored))
    check('unchanged_union_AST', ast.dump(frozen.body[3].body[2]), ast.dump(restored.body[3].body[2]))
    pipeline = ast.parse(Path(INPUTS['pipeline_path']).read_bytes())
    index = next(i for i, n in enumerate(pipeline.body) if ast.unparse(n) == 'observer.stage._obs = observation')
    pipeline.body.insert(index + 1, ast.parse('compose_metrics(observer.stage)').body[0])
    composed = ns['pipeline_tree'](SimpleNamespace(pipeline_tree=lambda: copy.deepcopy(pipeline)))
    check('single_adjacent_timing_composition', 'compose_timing(observer.stage)', ast.unparse(composed.body[index + 2]))
    del composed.body[index + 2]
    check('restored_pipeline_CLI_AST', ast.dump(pipeline), ast.dump(composed))
    observations, orders = [], []
    def sorted_spy(*args, **kwargs):
        ordered = sorted(*args, **kwargs)
        orders.append([e['label'] for e in ordered])
        return ordered
    observe = lambda name, **values: observations.append([name, values])
    timed_function = function(definition(timed, 'batched_metrics'), {'_obs': observe, 'sorted': sorted_spy})
    frozen_function = function(frozen, {})
    for fixture in INPUTS['fixtures']:
        events = [dict(ts=t, dur=d, label=i) for i, (t, d) in enumerate(fixture['events'])]
        groups = {k: [events[i] for i in indices] for k, indices in fixture['groups'].items()}
        actual, expected = timed_function(events, groups), frozen_function(events, groups)
        values = lambda rows: [[k, v, v['sum_ms'].hex(), v['union_ms'].hex()] for k, v in rows.items()]
        check(fixture['name'] + ':exact_values', values(expected), values(actual))
        check(fixture['name'] + ':stable_duplicate_orders_and_sort_count', fixture['sorted_orders'], orders)
        check(fixture['name'] + ':sparse_boundaries', 4 + 6 * len(groups), sum(n.startswith(prefix) for n, _ in observations))
        check(fixture['name'] + ':group_membership_telemetry', True, all(v['members'] == len(groups[k]) for n, v in observations if n.startswith(prefix + 'group') for k in groups if repr(k) == v['group']))
        observations.clear()
        orders.clear()
    check('original_float_accumulation_witness', True, actual['original']['sum_ms'].hex() != actual['reverse']['sum_ms'].hex())
    for name in wrappers:
        captured, iterable, key, returned = [], [], object(), []
        wrapped = function(definition(definition(timed, 'batched_metrics'), name), {'_obs': observe, 'key': 'fixture', 'members': []})
        original = lambda *a, **k: captured.append([a, k]) or returned
        output = wrapped(original, iterable, key=key, reverse=False)
        check(name + ':same_list_and_exact_forwarding', [True, 1, True, True, False],
              [output is returned, len(captured), captured[0][0][0] is iterable, captured[0][1]['key'] is key, captured[0][1]['reverse']])
    check('immutable_pins_after', hashes, {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in INPUTS['pins']})
    check('serialized_evidence_after_list_mutation', True, all(r['expected'] == r['actual'] for r in checks))
    result['status'] = 'PASS'
except Exception:
    result.update(status='FAIL', error=traceback.format_exc())
finally:
    result['elapsed_seconds'] = time.monotonic() - START
    with (OWNER / 'tiny_result.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'elapsed_seconds', 'tiny_invocations', 'full_invocations')}))
    sys.exit(0 if result['status'] == 'PASS' else 1)
