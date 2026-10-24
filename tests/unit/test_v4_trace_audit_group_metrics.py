"""Exact frozen metrics and AST checks for the numeric group-order adapter."""

import ast
import builtins
import copy
import hashlib
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
AUDITOR = Path('/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE02-DATA/audit.py')
STAGE = Path('/data/ycfeng/tmp/glm53_dsv4/p0_audit_cpu_20261024/stage_audit.py')
GC_ROOT = Path('/data/ycfeng/tmp/glm53_dsv4/p0_gc_observation_20261024')
PINS = {
    REPO / 'tests/analysis/v4_trace_audit_metrics.py': '6dff04f36dbcd8a66a298e4f6ae88d382bb29813c8a98f8fea91372b271b788a',
    AUDITOR: '7ed0ea45a61d5150e2f45bbc1cdf245f2331e40a351c235228b209b064ede0c8',
    STAGE: 'f7f1253ce9ebae5dcb3b60e2aeb97d7f17f58f34a7d68af56611918f809b9fe4',
    GC_ROOT / 'observe_gc.py': '3ae64dce5e1498418c3489d48bb74393f08c4dbbcb95231bab4e11a414a5a83c',
    GC_ROOT / 'load_without_gc.py': '6b91e25fc592d4f1340ef22e6d6e73c4dd8ed6e7ce6239f7a113d946a74edd02',
    REPO / 'tests/comparison/calibration/forward_device_timeline.py': 'fe4c3f82500df79ce097ce762808bcaf2b78040e41cbab942ccc746a723b3a52',
}
EVIDENCE, AST_EVIDENCE = [], {}


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def execute(node, namespace):
    tree = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]))
    exec(compile(tree, '<isolated_function>', 'exec'), namespace)
    return namespace[node.name]


def stage_transformer():
    tree = ast.parse(STAGE.read_text())
    literals = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ('RULES', 'LOOPS')}
    return execute(function(tree, 'instrument'), {'ast': ast, 'HITS': [], **literals})


def fields(results):
    return [dict(key=repr(k), count=v['count'], sum_ms=v['sum_ms'], union_ms=v['union_ms'],
                 sum_hex=v['sum_ms'].hex(), union_hex=v['union_ms'].hex()) for k, v in results.items()]


class GroupMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load(REPO / 'tests/analysis/v4_trace_audit_group_metrics.py')
        cls.source = AUDITOR.read_text()
        cls.frozen_node = function(ast.parse(cls.source), 'metrics')
        cls.frozen = staticmethod(execute(cls.frozen_node, {}))

    def equal(self, name, expected, actual):
        EVIDENCE.append(dict(name=name, expected=copy.deepcopy(expected), actual=copy.deepcopy(actual),
                             status='PASS' if actual == expected else 'FAIL'))
        self.assertEqual(actual, expected)

    def exact(self, name, events, groups):
        before = copy.deepcopy((events, groups))
        identities = [(repr(k), id(v), [id(e) for e in v]) for k, v in groups.items()]
        expected = {k: self.frozen(v) for k, v in groups.items()}
        actual = self.adapter.batched_metrics(events, groups)
        self.equal(name, fields(expected), fields(actual))
        self.assertEqual(actual, expected)
        self.equal(name + ':inputs', True, before == (events, groups))
        self.equal(name + ':identities', identities,
                   [(repr(k), id(v), [id(e) for e in v]) for k, v in groups.items()])
        return actual

    def test_source_pins(self):
        for path, expected in PINS.items():
            self.equal('pin:' + str(path), expected, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_empty_groups(self):
        self.exact('empty_no_groups', [], {})
        self.exact('empty_groups', [], {('total', None): [], ('status', 'unowned'): [], 7: []})

    def test_overlap_repeated_memberships_and_keys(self):
        events = [dict(ts=a, dur=b, traceEvents_index=i) for i, (a, b) in enumerate(
            [(8.0, 2.0), (0.0, 10.0), (0.0, 1.0), (0.0, 1.0), (12.0, 0.5), (12.5, 0.0), (11.0, 1.5)])]
        groups = {('total', None): events, ('per_stream', '7'): events[::2],
                  ('status', 'unique_leaf'): events[1::2], ('hierarchy', 'same'): [events[i] for i in (0, 1, 1, 3, 6)],
                  ('leaf', 'eager'): events[:3], ('hierarchy', 'empty'): [], None: events, 7: events[:1], '7': events[-1:]}
        self.exact('overlap_repeated_heterogeneous_keys', events, groups)

    def test_stable_global_order_and_integer_group_keys(self):
        events = [dict(ts=0, dur=d, label=i) for i, d in enumerate([2, 1, 1, 3])]
        groups = {'reverse_ties': [events[i] for i in (2, 1, 0, 3)], 'repeat': [events[i] for i in (2, 2, 1)]}
        captures = []
        def sorter(items, *, key):
            keys = []
            def observed(event):
                value = key(event)
                keys.append(copy.deepcopy(value))
                return value
            ordered = builtins.sorted(items, key=observed)
            captures.append(dict(keys=copy.deepcopy(keys), labels=[e['label'] for e in ordered]))
            return ordered
        with mock.patch.object(self.adapter, 'sorted', side_effect=sorter, create=True):
            self.exact('stable_ties', events, groups)
        self.equal('sort_count', 1 + len(groups), len(captures))
        self.equal('stable_global_order', [1, 2, 0, 3], captures[0]['labels'])
        self.equal('integer_group_positions', True, all(type(k) is int for row in captures[1:] for k in row['keys']))
        self.equal('group_order_and_duplicates', [[1, 2, 0, 3], [1, 2, 2]], [row['labels'] for row in captures[1:]])

    def test_duplicate_global_identity_with_equal_endpoints(self):
        first, second = dict(ts=1.0, dur=2.0), dict(ts=1.0, dur=2.0)
        self.exact('duplicate_identity_equal_endpoints', [first, second, first],
                   {None: [second, first, first], 'opposite': [first, first, second]})

    def test_original_float_accumulation(self):
        events = [dict(ts=t, dur=d) for t, d in [(2, 0.1), (1, 0.2), (0, 0.3)]]
        actual = self.exact('original_order_floats', events, {'total': events, 'reverse': events[::-1]})
        wrong = self.frozen(sorted(events, key=lambda e: (e['ts'], e['ts'] + e['dur'])))
        self.equal('float_order_negative_witness', True, actual['total']['sum_ms'].hex() != wrong['sum_ms'].hex())

    def test_generated_ast_and_composition(self):
        staged = stage_transformer()(self.source)
        accepted = self.adapter.ACCEPTED.instrument_metrics(copy.deepcopy(staged))
        before = ast.dump(accepted)
        corrected = self.adapter.instrument_groups(accepted)
        original, replacement = function(accepted, 'batched_metrics'), function(corrected, 'batched_metrics')
        self.equal('accepted_input_AST_unchanged', before, ast.dump(accepted))
        self.equal('outside_metrics_definition_AST', [ast.dump(n) for n in accepted.body if n is not original],
                   [ast.dump(n) for n in corrected.body if n is not replacement])
        trace = function(corrected, 'trace')
        insertions = ast.parse(self.adapter.ACCEPTED.BATCH_INSERTION).body
        pos = next(i for i, n in enumerate(trace.body) if ast.dump(n) == ast.dump(insertions[0]))
        self.equal('original_grouping_expressions_and_calls', [ast.dump(n) for n in insertions],
                   [ast.dump(n) for n in trace.body[pos:pos + 2]])
        stripped = copy.deepcopy(replacement)
        del stripped.body[1]
        del stripped.body[-2]
        source_node = function(ast.parse(Path(self.adapter.__file__).read_text()), 'batched_metrics')
        self.equal('telemetry_stripped_source_AST', ast.dump(source_node), ast.dump(stripped))
        recurrence = next(n for n in ast.walk(source_node) if isinstance(n, ast.For) and isinstance(n.target, ast.Name))
        frozen_loop = next(n for n in self.frozen_node.body if isinstance(n, ast.For))
        self.equal('exact_frozen_union_recurrence_AST', [ast.dump(n) for n in frozen_loop.body],
                   [ast.dump(n) for n in recurrence.body[-2:]])
        self.equal('no_mutable_membership_lists', False, any(isinstance(n, ast.List) for n in ast.walk(source_node)))
        observed = []
        injected = execute(replacement, {'_obs': lambda name, **values: observed.append(copy.deepcopy(dict(stage=name, **values)))})
        event = dict(ts=0.0, dur=1.0)
        self.equal('injected_exact_metrics', fields({'g': self.frozen([event, event])}), fields(injected([event], {'g': [event, event]})))
        self.equal('sparse_start_end', [dict(stage=self.adapter.PREFIX + ':start', events=1, groups=1, memberships=2),
                                      dict(stage=self.adapter.PREFIX + ':end')], observed)
        stage = SimpleNamespace(instrument=stage_transformer(), _obs=object(), run=object())
        observer, runner = stage._obs, stage.run
        self.adapter.compose_metrics(stage)
        self.equal('composition_AST', ast.dump(corrected), ast.dump(stage.instrument(self.source)))
        self.assertIs(stage._obs, observer)
        self.assertIs(stage.run, runner)
        AST_EVIDENCE.update(accepted_module=ast.dump(accepted, indent=2), corrected_module=ast.dump(corrected, indent=2),
                            stripped_metrics=ast.dump(stripped, indent=2), frozen_union=ast.dump(frozen_loop, indent=2))

    def test_pipeline_and_entry_closure(self):
        self.assertIs(self.adapter.pipeline_tree, self.adapter.ACCEPTED.pipeline_tree)
        self.assertEqual(self.adapter.PIPELINE, self.adapter.ACCEPTED.PIPELINE)
        pipeline = self.adapter.pipeline_tree()
        original = ast.parse(self.adapter.PIPELINE.read_bytes())
        insertion = ast.dump(ast.parse('compose_metrics(observer.stage)').body[0])
        positions = [i for i, n in enumerate(pipeline.body) if ast.dump(n) == insertion]
        self.equal('original_single_composition_boundary', 1, len(positions))
        del pipeline.body[positions[0]]
        self.equal('original_loading_GC_pipeline_AST', ast.dump(original), ast.dump(pipeline))
        source = ast.parse(Path(self.adapter.__file__).read_text())
        accepted = ast.parse(self.adapter.ADAPTER.read_text())
        self.equal('unchanged_entry_main_AST', ast.dump(function(accepted, 'main')), ast.dump(function(source, 'main')))
        AST_EVIDENCE.update(original_pipeline=ast.dump(original, indent=2), stripped_pipeline=ast.dump(pipeline, indent=2))

    def test_receipt_values_are_copied(self):
        expected, actual = [[1, 2, 0, 3]], [[1, 2, 0, 3]]
        self.equal('receipt_snapshot', expected, actual)
        actual.clear()
        expected[0].reverse()
        self.assertEqual(EVIDENCE[-1]['expected'], [[1, 2, 0, 3]])
        self.assertEqual(EVIDENCE[-1]['actual'], [[1, 2, 0, 3]])


if __name__ == '__main__':
    unittest.main()
