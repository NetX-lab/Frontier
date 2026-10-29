"""One allocated standard-library control; never execute a trace or pipeline."""

import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO = Path.cwd()
OWNER = Path(__file__).resolve().parent
AUDITOR = Path('/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE02-DATA/audit.py')
STAGE = Path('/data/ycfeng/tmp/glm53_dsv4/p0_audit_cpu_20261024/stage_audit.py')
GC_ROOT = Path('/data/ycfeng/tmp/glm53_dsv4/p0_gc_observation_20261024')
PINS = {
    REPO / 'tests/analysis/v4_trace_audit_metrics.py':
        '6dff04f36dbcd8a66a298e4f6ae88d382bb29813c8a98f8fea91372b271b788a',
    AUDITOR: '7ed0ea45a61d5150e2f45bbc1cdf245f2331e40a351c235228b209b064ede0c8',
    STAGE: 'f7f1253ce9ebae5dcb3b60e2aeb97d7f17f58f34a7d68af56611918f809b9fe4',
    GC_ROOT / 'load_without_gc.py':
        '6b91e25fc592d4f1340ef22e6d6e73c4dd8ed6e7ce6239f7a113d946a74edd02',
    GC_ROOT / 'observe_gc.py':
        '3ae64dce5e1498418c3489d48bb74393f08c4dbbcb95231bab4e11a414a5a83c',
    REPO / 'tests/comparison/calibration/forward_device_timeline.py':
        'fe4c3f82500df79ce097ce762808bcaf2b78040e41cbab942ccc746a723b3a52',
}
TIMER = REPO / 'tests/analysis/v4_trace_audit_metrics_timing.py'
COMMAND = f'timeout 60s /usr/bin/python3 -B {Path(__file__).relative_to(REPO)}'
START = time.monotonic()
RECEIPT = {'command': COMMAND, 'cwd': str(REPO), 'python': sys.version,
           'executable': sys.executable, 'platform': platform.platform(),
           'conda': 'not used; /usr/bin/python3', 'tiny_invocations_used': 1,
           'complete_trace_invocations': 0, 'pipeline_invocations': 0,
           'fixture_results': [], 'checks': []}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, expected, actual):
    passed = actual == expected
    RECEIPT['checks'].append(dict(name=name, expected=expected, actual=actual,
                                  status='PASS' if passed else 'FAIL'))
    assert passed, name


def definition(tree, name):
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def dump(tree):
    return ast.dump(tree, indent=2, include_attributes=False)


def execute_function(node, namespace, filename):
    tree = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]))
    exec(compile(tree, filename, 'exec'), namespace)
    return namespace[node.name]


def exact_results(results):
    return [dict(key=repr(key), count=value['count'], sum_ms=value['sum_ms'],
                 union_ms=value['union_ms'], sum_hex=value['sum_ms'].hex(),
                 union_hex=value['union_ms'].hex()) for key, value in results.items()]


try:
    RECEIPT['source_hashes'] = {str(path): sha(path) for path in [*PINS, TIMER, Path(__file__)]}
    for path, expected in PINS.items():
        check('pin:' + str(path), expected, sha(path))
    check('timer_gross_lines_at_most_140', True, len(TIMER.read_text().splitlines()) <= 140)
    RECEIPT['timer_gross_lines'] = len(TIMER.read_text().splitlines())
    spec = importlib.util.spec_from_file_location('metrics_tail_timer', TIMER)
    timer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(timer)
    adapter = timer.load_adapter()
    auditor_source = AUDITOR.read_text()
    auditor_tree = ast.parse(auditor_source)
    frozen = execute_function(definition(auditor_tree, 'metrics'), {}, str(AUDITOR))
    # Extract only the stage's AST transformer and literal rules; do not import its runner.
    stage_tree = ast.parse(STAGE.read_text())
    literals = {node.targets[0].id: ast.literal_eval(node.value) for node in stage_tree.body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ('RULES', 'LOOPS')}
    stage_namespace = {'ast': ast, 'HITS': [], **literals}
    stage_instrument = execute_function(definition(stage_tree, 'instrument'), stage_namespace, str(STAGE))
    observer, runner = object(), object()
    stage = SimpleNamespace(instrument=stage_instrument, _obs=observer, run=runner)
    adapter.compose_metrics(stage)
    accepted = stage.instrument(auditor_source)
    accepted_dump = dump(accepted)
    accepted_function = definition(accepted, 'batched_metrics')
    check('accepted_function_source_closure',
          dump(definition(ast.parse(timer.ADAPTER.read_text()), 'batched_metrics')),
          dump(accepted_function))
    timer.compose_timing(stage)
    transformed = stage.instrument(auditor_source)
    check('composition_keeps_observer_identity', True, stage._obs is observer)
    check('composition_keeps_runner_identity', True, stage.run is runner)
    check('composition_equals_direct_timing', dump(timer.instrument_timing(accepted)), dump(transformed))
    check('input_AST_unchanged', accepted_dump, dump(accepted))
    check('other_module_statements_unchanged',
          [dump(node) for node in accepted.body if node is not accepted_function],
          [dump(node) for node in transformed.body
           if node is not definition(transformed, 'batched_metrics')])

    class StripTimer(ast.NodeTransformer):
        observations = wrappers = sorts = 0

        def visit_FunctionDef(self, node):
            if node.name == timer.SORT_WRAPPER:
                self.wrappers += 1
                return None
            return self.generic_visit(node)

        def visit_Expr(self, node):
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == '_obs' and isinstance(call.args[0], ast.Constant)
                    and str(call.args[0].value).startswith(timer.PREFIX + ':')):
                self.observations += 1
                return None
            return self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == timer.SORT_WRAPPER:
                self.sorts += 1
                assert dump(node.args[0]) == dump(ast.Name(id='sorted', ctx=ast.Load()))
                return ast.Call(func=node.args[0], args=node.args[1:], keywords=node.keywords)
            return self.generic_visit(node)

    stripper = StripTimer()
    stripped = ast.fix_missing_locations(stripper.visit(copy.deepcopy(transformed)))
    check('removed_observations_wrapper_sort', [5, 1, 1],
          [stripper.observations, stripper.wrappers, stripper.sorts])
    check('stripped_function_exact_AST', dump(accepted_function),
          dump(definition(stripped, 'batched_metrics')))
    check('stripped_full_generated_module_exact_AST', accepted_dump, dump(stripped))
    restored_sort = definition(stripped, 'batched_metrics').body[3].iter
    accepted_pipeline = adapter.pipeline_tree()
    transformed_pipeline = timer.pipeline_tree(adapter)
    stripped_pipeline = copy.deepcopy(transformed_pipeline)
    insertion = dump(ast.parse('compose_timing(observer.stage)').body[0])
    positions = [i for i, node in enumerate(stripped_pipeline.body) if dump(node) == insertion]
    check('single_pipeline_composition', 1, len(positions))
    check('compose_after_accepted_adapter', dump(ast.parse('compose_metrics(observer.stage)').body[0]),
          dump(stripped_pipeline.body[positions[0] - 1]))
    del stripped_pipeline.body[positions[0]]
    check('pipeline_AST_equals_accepted', dump(accepted_pipeline), dump(stripped_pipeline))
    original_pipeline = copy.deepcopy(accepted_pipeline)
    accepted_insertion = dump(ast.parse('compose_metrics(observer.stage)').body[0])
    positions = [i for i, node in enumerate(original_pipeline.body) if dump(node) == accepted_insertion]
    check('single_accepted_pipeline_composition', 1, len(positions))
    del original_pipeline.body[positions[0]]
    check('original_loading_GC_pipeline_AST_unchanged', dump(ast.parse(adapter.PIPELINE.read_bytes())),
          dump(original_pipeline))
    closure = dict(accepted_function=dump(accepted_function),
                   transformed_function=dump(definition(transformed, 'batched_metrics')),
                   stripped_function=dump(definition(stripped, 'batched_metrics')),
                   restored_sort_expression=dump(restored_sort),
                   accepted_generated_module=accepted_dump, transformed_generated_module=dump(transformed),
                   stripped_generated_module=dump(stripped), accepted_pipeline=dump(accepted_pipeline),
                   transformed_pipeline=dump(transformed_pipeline), stripped_pipeline=dump(stripped_pipeline))
    closure_path = OWNER / 'ast_closure.json'
    with closure_path.open('x') as stream:
        json.dump(closure, stream, indent=2)
        stream.write('\n')
    RECEIPT['ast_closure'] = dict(path=str(closure_path), sha256=sha(closure_path))
    # Keep receipt checks compact; the complete compared ASTs remain in ast_closure.json.
    for row in RECEIPT['checks']:
        for key in ('expected', 'actual'):
            if isinstance(row[key], (str, list)) and len(json.dumps(row[key])) > 2000:
                row[key] = {'sha256': hashlib.sha256(json.dumps(row[key]).encode()).hexdigest(),
                            'detail': 'Full AST retained in ast_closure.json'}

    observations, sort_calls, orders = [], [], []

    def observe(name, **values):
        observations.append(dict(stage=name, seconds=time.monotonic() - START, **values))

    def original_sorted(*args, **kwargs):
        sort_calls.append(1)
        ordered = builtins.sorted(*args, **kwargs)
        orders.append([event['label'] for event in ordered])
        return ordered

    timed = execute_function(definition(transformed, 'batched_metrics'),
                             {'_obs': observe, 'sorted': original_sorted}, str(TIMER))
    events = [dict(ts=t, dur=d, label=i) for i, (t, d) in enumerate(
        [(8.0, 2.0), (0.0, 10.0), (0.0, 1.0), (0.0, 1.0),
         (12.0, 0.5), (12.5, 0.0), (11.0, 1.5)])]
    ties = [dict(ts=t, dur=d, label=i) for i, (t, d) in enumerate(
        [(0.0, 2.0), (0.0, 1.0), (0.0, 1.0), (3.0, 1.0)])]
    floats = [dict(ts=t, dur=d, label=i) for i, (t, d) in enumerate([(2, 0.1), (1, 0.2), (0, 0.3)])]
    fixtures = [
        ('empty_no_groups', [], {}),
        ('empty_groups', [], {('total', None): [], ('hierarchy', 'empty'): []}),
        ('stable_ties', ties, {('total', None): ties, ('per_stream', '7'): ties[1:3]}),
        ('repeated_membership', events, {('total', None): events, ('per_stream', '7'): events[::2],
         ('status', 'unique_leaf'): events[1::2], ('hierarchy', 'same'): [events[i] for i in (0, 1, 1, 3, 6)],
         ('leaf', 'eager'): events[:3], ('hierarchy', 'empty'): [], None: events, 7: events[:1], '7': events[-1:]}),
        ('original_order_floats', floats, {'total': floats, 'reverse': floats[::-1]}),
    ]
    suffixes = ['batch_start', 'group_construction_start', 'group_construction_done', 'sort_start',
                'sort_done', 'union_start', 'union_done', 'batch_done']
    for name, fixture_events, groups in fixtures:
        observations.clear()
        sort_calls.clear()
        orders.clear()
        before = copy.deepcopy((fixture_events, groups))
        identities = [(key, id(members), [id(event) for event in members]) for key, members in groups.items()]
        expected = {key: frozen(members) for key, members in groups.items()}
        accepted_output = adapter.batched_metrics(fixture_events, groups)
        actual = timed(fixture_events, groups)
        check(name + ':accepted_equals_frozen', exact_results(expected), exact_results(accepted_output))
        check(name + ':timed_exact_results', exact_results(expected), exact_results(actual))
        check(name + ':values_unchanged', True, before == (fixture_events, groups))
        check(name + ':group_event_identities_unchanged', identities,
              [(key, id(members), [id(event) for event in members]) for key, members in groups.items()])
        check(name + ':original_sorted_delegated_once', 1, len(sort_calls))
        check(name + ':sparse_stage_order', [timer.PREFIX + ':' + suffix for suffix in suffixes],
              [row['stage'] for row in observations])
        check(name + ':group_count', len(groups), observations[0]['groups'])
        check(name + ':membership_count_from_group_lengths', sum(len(members) for members in groups.values()),
              observations[0]['memberships'])
        if name == 'stable_ties':
            check('stable_tie_order', [[1, 2, 0, 3]], orders)
        RECEIPT['fixture_results'].append(dict(name=name, expected=exact_results(expected),
            actual=exact_results(actual), groups=len(groups), memberships=observations[0]['memberships'],
            observations=copy.deepcopy(observations), sort_calls=len(sort_calls), sorted_labels=copy.deepcopy(orders)))
    wrong = frozen(builtins.sorted(floats, key=lambda event: (event['ts'], event['ts'] + event['dur'])))
    correct = frozen(floats)
    check('float_order_negative_witness', True, wrong['sum_ms'].hex() != correct['sum_ms'].hex())
    RECEIPT['float_order_witness'] = dict(original=correct['sum_ms'].hex(), reordered=wrong['sum_ms'].hex())
    RECEIPT['status'] = 'PASS'
except Exception:
    RECEIPT['status'] = 'FAIL'
    RECEIPT['error'] = traceback.format_exc()
finally:
    RECEIPT['elapsed_seconds'] = time.monotonic() - START
    with (OWNER / 'tiny_control_result.json').open('x') as stream:
        json.dump(RECEIPT, stream, indent=2, default=str)
        stream.write('\n')
    print(json.dumps({key: RECEIPT[key] for key in
                     ('status', 'elapsed_seconds', 'tiny_invocations_used', 'complete_trace_invocations',
                      'pipeline_invocations')}, sort_keys=True))
    if RECEIPT['status'] != 'PASS':
        print(RECEIPT.get('error', 'Control failed'), file=sys.stderr)
        sys.exit(1)
