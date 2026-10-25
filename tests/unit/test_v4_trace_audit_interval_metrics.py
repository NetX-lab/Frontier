#!/usr/bin/env python3
"""Check the new adapter using isolated frozen functions, never outer harnesses."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'tests/analysis/v4_trace_audit_interval_metrics.py'
OWNER = ROOT / ('task_memory/task_2026-10-23_v4_remaining_measurements/profile01/'
                'profile04/results/p0_cpu_loop/metrics_correction03/owner')
ALLOCATION = OWNER / 'unit_execution_allocation.json'
ASSERTIONS = []


def definition(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def isolated(path, name):
    node = copy.deepcopy(definition(ast.parse(path.read_text()), name))
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def values(value):
    if isinstance(value, float):
        return {'floathex': value.hex()}
    if isinstance(value, dict):
        return [[values(k), values(v)] for k, v in value.items()]
    if isinstance(value, (list, tuple)):
        return [values(v) for v in value]
    return value


class IntervalMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('interval_metrics_adapter', SOURCE)
        cls.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.adapter)
        cls.accepted_path = cls.adapter.ADAPTER
        cls.audit_path = Path('/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE02-DATA/audit.py')
        cls.original = staticmethod(isolated(cls.audit_path, 'metrics'))
        cls.accepted = staticmethod(isolated(cls.accepted_path, 'batched_metrics'))
        cls.source_tree = ast.parse(SOURCE.read_text())
        cls.accepted_tree = ast.parse(cls.accepted_path.read_text())

    def check(self, label, actual, expected):
        actual, expected = copy.deepcopy(actual), copy.deepcopy(expected)
        ASSERTIONS.append(dict(label=label, actual=actual, expected=expected,
                               passed=actual == expected))
        self.assertEqual(actual, expected, label)

    def compare(self, events, groups):
        snapshot = values([events, groups])
        identities = [id(events), id(groups), [id(e) for e in events],
                      [(id(m), [id(e) for e in m]) for m in groups.values()]]
        expected = {key: self.original(members) for key, members in groups.items()}
        for name, function in [('new', self.adapter.batched_metrics), ('accepted', self.accepted)]:
            result = function(events, groups)
            self.check(name + ':floathex/count/key-order', values(result), values(expected))
            self.check(name + ':input-values', values([events, groups]), snapshot)
            self.check(name + ':input-object-identities',
                       [id(events), id(groups), [id(e) for e in events],
                        [(id(m), [id(e) for e in m]) for m in groups.values()]], identities)

    def test_pins_and_caps(self):
        allocation = json.loads(ALLOCATION.read_text())
        for pin in allocation['reviewed_files']:
            self.check('sha256:' + pin['path'],
                       hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest(), pin['sha256'])
        for path, cap in [(SOURCE, 160), (Path(__file__), 260), (OWNER / 'source_notes.md', 80)]:
            self.check('gross-cap:' + path.name, len(path.read_text().splitlines()) <= cap, True)
        self.check('original-python-version', sys.version, allocation['python'])
        self.check('original-environment', {key: os.environ.get(key) for key in allocation['environment']},
                   allocation['environment'])
        self.check('accepted-pin', self.adapter.ADAPTER_SHA256,
                   '4dec265e20426a20e437588a4e44828e257247c82897cc8188621f93369a5251')

    def test_empty_and_mixed_keys(self):
        events = []
        self.compare(events, {})
        self.compare(events, {('total', None): events, None: [], 17: [], 'empty': []})
        events = [{'ts': 2.5, 'dur': 0.125}, {'ts': -3.0, 'dur': 1.75}]
        self.compare(events, {('total', None): events, None: events[:1], 17: [], 'text': events[::-1]})

    def test_ties_reversed_groups_and_float_sum_order(self):
        events = [{'ts': 8.0, 'dur': 0.25}, {'ts': 2.0, 'dur': 0.0},
                  {'ts': 8.0, 'dur': 0.25}, {'ts': 8.0, 'dur': 0.125},
                  {'ts': -4.0, 'dur': 2.0}, {'ts': 3.0, 'dur': -0.5}]
        self.compare(events, {'total': events, 'reversed': events[::-1], 'ties': events[::2], 'empty': []})
        events = [{'ts': 100.0, 'dur': 1e16}, {'ts': 0.0, 'dur': 1.0}, {'ts': 1.0, 'dur': 1.0}]
        self.compare(events, {'total': events, 'reverse': events[::-1]})
        self.check('fixture-sum-order-is-observable',
                   self.original(events)['sum_ms'].hex() != self.original(events[::-1])['sum_ms'].hex(), True)

    def test_repeated_members_and_global_duplicate_identities(self):
        event = {'ts': 5.125, 'dur': 0.375, 'args': {'stream': 7}}
        tie = copy.deepcopy(event)
        other = {'ts': -1.0, 'dur': 0.25}
        events = [event, tie, event, other, event]
        self.compare(events, {('total', None): events, 'equal-copy': list(events),
                              'repeated': [other, event, event, other, event],
                              'reverse': events[::-1], 'tie': [tie, event, tie], 'empty': []})

    def test_exact_recurrence_and_representation_ast(self):
        new = definition(self.source_tree, 'batched_metrics')
        old = definition(ast.parse(self.audit_path.read_text()), 'metrics')
        loop = next(n for n in ast.walk(new) if isinstance(n, ast.For) and ast.unparse(n.target) == 'position')
        frozen_loop = next(n for n in ast.walk(old) if isinstance(n, ast.For))
        self.check('exact-two-frozen-recurrence-statements',
                   [ast.dump(n) for n in loop.body[1:]], [ast.dump(n) for n in frozen_loop.body])
        accepted = definition(self.accepted_tree, 'batched_metrics')
        result = lambda tree: next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                                  and ast.unparse(n.targets[0]) == 'result')
        self.check('count-sum-result-expression', ast.dump(result(new)), ast.dump(result(accepted)))
        expressions = ["ordered = sorted(events, key=lambda e: (e['ts'], e['ts'] + e['dur']))",
                       "starts = [event['ts'] for event in ordered]",
                       "ends = [event['ts'] + event['dur'] for event in ordered]",
                       "order = {id(event): position for position, event in enumerate(ordered)}",
                       'positions = range(len(ordered)) if members is events else sorted(order[id(event)] for event in members)',
                       'a, b = starts[position], ends[position]']
        assignments = [ast.dump(n) for n in ast.walk(new) if isinstance(n, ast.Assign)]
        for expression in expressions:
            self.check('one-exact-expression:' + expression,
                       assignments.count(ast.dump(ast.parse(expression).body[0])), 1)
        self.check('no-mutable-membership-operations',
                   [n.attr for n in ast.walk(new) if isinstance(n, ast.Attribute)], ['items'])

    def test_composed_ast_grouping_pipeline_and_entry_unchanged(self):
        source = self.audit_path.read_text()
        old_stage = types.SimpleNamespace(instrument=ast.parse)
        self.adapter.ACCEPTED.compose_metrics(old_stage)
        original = old_stage.instrument(source)
        original_dump = ast.dump(original)
        changed = self.adapter.instrument_intervals(original)
        self.check('input-ast-not-mutated', ast.dump(original), original_dump)
        self.check('new-ast-is-copy', changed is not original, True)
        strip = lambda tree: ast.Module(body=[n for n in tree.body if not
            (isinstance(n, ast.FunctionDef) and n.name == 'batched_metrics')], type_ignores=tree.type_ignores)
        self.check('all-ast-outside-metrics', ast.dump(strip(changed)), ast.dump(strip(original)))
        new_stage = types.SimpleNamespace(instrument=ast.parse)
        self.adapter.compose_metrics(new_stage)
        self.check('actual-composition-matches-only-metrics-replacement', ast.dump(new_stage.instrument(source)), ast.dump(changed))
        observations = lambda tree: [ast.dump(n) for n in definition(tree, 'batched_metrics').body
                                    if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
        self.check('unchanged-sparse-start-end', observations(changed), observations(original))
        self.check('only-two-metrics-observations', len(observations(changed)), 2)
        self.check('pipeline-callable-identity', self.adapter.pipeline_tree is self.adapter.ACCEPTED.pipeline_tree, True)
        self.check('pipeline-loading-gc-ast', ast.dump(self.adapter.pipeline_tree()), ast.dump(self.adapter.ACCEPTED.pipeline_tree()))
        self.check('entry-function-ast', ast.dump(definition(self.source_tree, 'main')),
                   ast.dump(definition(self.accepted_tree, 'main')))
        self.check('entry-guard-ast', ast.dump(self.source_tree.body[-1]), ast.dump(self.accepted_tree.body[-1]))

    def test_actual_composed_function_and_sparse_observations(self):
        stage = types.SimpleNamespace(instrument=ast.parse)
        self.adapter.compose_metrics(stage)
        tree = stage.instrument(self.audit_path.read_text())
        observations = []
        namespace = {'_obs': lambda name, **fields: observations.append(dict(name=name, **fields))}
        exec(compile(ast.Module(body=[definition(tree, 'batched_metrics')], type_ignores=[]),
                     str(SOURCE), 'exec'), namespace)
        event = {'ts': 0.125, 'dur': 0.375}
        events = [event, event]
        groups = {'total': events, 'repeated': [event, event, event], 'empty': []}
        self.check('composed-function-floathex', values(namespace['batched_metrics'](events, groups)),
                   values({key: self.original(members) for key, members in groups.items()}))
        self.check('sparse-observations-actual-values', observations,
                   [dict(name=self.adapter.ACCEPTED.PREFIX + ':start', events=2, groups=3, memberships=5),
                    dict(name=self.adapter.ACCEPTED.PREFIX + ':end')])

    def test_cgroup_sensor_expressions_are_frozen(self):
        sensor_path = Path('/data/ycfeng/tmp/glm53_dsv4/p0_gc_observation_20261024/observe_gc.py')
        old = definition(ast.parse(sensor_path.read_text()), 'run')
        new = definition(ast.parse(Path(__file__).read_text()), 'main')
        for name in ['cgroup', 'memory_path', 'mount', 'group', 'limits']:
            assignments = lambda tree: [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                                        and ast.unparse(n.targets[0]) == name]
            self.check('frozen-cgroup-sensor:' + name, assignments(new), assignments(old))

    def test_exact_source_guard_rejects_mismatched_missing_duplicate_definitions(self):
        stage = types.SimpleNamespace(instrument=ast.parse)
        self.adapter.ACCEPTED.compose_metrics(stage)
        tree = stage.instrument(self.audit_path.read_text())
        mismatched = copy.deepcopy(tree)
        definition(mismatched, 'batched_metrics').body[0].value.value = 'Different source'
        missing = copy.deepcopy(tree)
        missing.body.remove(definition(missing, 'batched_metrics'))
        duplicate = copy.deepcopy(tree)
        duplicate.body.append(copy.deepcopy(definition(duplicate, 'batched_metrics')))
        for label, candidate in [('mismatched', mismatched), ('missing', missing), ('duplicate', duplicate)]:
            before = ast.dump(candidate)
            with self.assertRaises(AssertionError):
                self.adapter.instrument_intervals(candidate)
            self.check('guard-keeps-input:' + label, ast.dump(candidate), before)


def write(name, payload):
    with (OWNER / name).open('x') as stream:
        json.dump(payload, stream, indent=2)
        stream.write('\n')


def main():
    allocation = json.loads(ALLOCATION.read_text())
    write('unit_invocation.json', dict(allocation_sha256=hashlib.sha256(ALLOCATION.read_bytes()).hexdigest(),
                                     command=allocation['command'], pid=os.getpid(), used=1, ceiling=1))
    started, result, error = time.monotonic(), None, None
    try:
        cgroup = Path('/proc/self/cgroup').read_text()
        memory_path = next(line.split(':', 2)[2] for line in cgroup.splitlines() if 'memory' in line.split(':', 2)[1].split(','))
        mount = next(line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines() if ' - cgroup ' in line and 'memory' in line.split(' - ', 1)[1].split()[2].split(','))
        group = Path(mount[4]) / Path(memory_path).relative_to(mount[3])
        limits = {name: (group / name).read_text().strip() for name in ['memory.limit_in_bytes', 'memory.usage_in_bytes', 'memory.failcnt']}
        write('unit_cgroup.json', dict(cgroup=cgroup, memory_mount_root=mount[3], group=str(group), limits=limits))
        assert limits['memory.limit_in_bytes'] == '32212254720', limits
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IntervalMetricsTests))
    except BaseException as caught:
        error = repr(caught)
        raise
    finally:
        write('unit_assertions.json', ASSERTIONS)
        write('unit_result.json', dict(tests_run=result.testsRun if result else 0,
              failures=[dict(test=str(test), traceback=trace) for test, trace in result.failures] if result else [],
              errors=[dict(test=str(test), traceback=trace) for test, trace in result.errors] if result else [],
              success=result.wasSuccessful() if result else False, capture_error=error,
              seconds=time.monotonic() - started, python=sys.version, executable=sys.executable,
              environment={key: os.environ.get(key) for key in allocation['environment']},
              assertion_count=len(ASSERTIONS), used=1, ceiling=1))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
