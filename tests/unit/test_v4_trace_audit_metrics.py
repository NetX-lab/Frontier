"""Exact CPU and AST controls for the allocated frozen metrics adapter."""

import ast
import copy
import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
AUDITOR = Path('/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE02-DATA/audit.py')
STAGE = Path('/data/ycfeng/tmp/glm53_dsv4/p0_audit_cpu_20261024/stage_audit.py')
GC_ROOT = Path('/data/ycfeng/tmp/glm53_dsv4/p0_gc_observation_20261024')
PINS = {
    AUDITOR: '7ed0ea45a61d5150e2f45bbc1cdf245f2331e40a351c235228b209b064ede0c8',
    REPO / 'tests/comparison/calibration/forward_device_timeline.py': 'fe4c3f82500df79ce097ce762808bcaf2b78040e41cbab942ccc746a723b3a52',
    STAGE: 'f7f1253ce9ebae5dcb3b60e2aeb97d7f17f58f34a7d68af56611918f809b9fe4',
    GC_ROOT / 'observe_gc.py': '3ae64dce5e1498418c3489d48bb74393f08c4dbbcb95231bab4e11a414a5a83c',
    GC_ROOT / 'load_without_gc.py': '6b91e25fc592d4f1340ef22e6d6e73c4dd8ed6e7ce6239f7a113d946a74edd02',
}


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load(REPO / 'tests/analysis/v4_trace_audit_metrics.py')
        cls.source = AUDITOR.read_text()
        definition = next(n for n in ast.parse(cls.source).body
                          if isinstance(n, ast.FunctionDef) and n.name == 'metrics')
        namespace = {}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(AUDITOR), 'exec'), namespace)
        cls.frozen = staticmethod(namespace['metrics'])

    def test_source_pins(self):
        for path, expected in PINS.items():
            with self.subTest(path=str(path)):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_exact_groups_and_repeated_memberships(self):
        events = [dict(ts=a, dur=b) for a, b in
                  [(8.0, 2.0), (0.0, 10.0), (0.0, 1.0), (0.0, 1.0),
                   (12.0, 0.5), (12.5, 0.0), (11.0, 1.5)]]
        groups = {('total', None): events, ('per_stream', '7'): events[::2],
                  ('status', 'unique_leaf'): events[1::2],
                  ('hierarchy', 'layer|same'): [events[i] for i in (0, 1, 1, 3, 6)],
                  ('leaf', 'layer|same|eager'): events[:3], ('hierarchy', 'empty'): [],
                  None: events, 7: events[:1], '7': events[-1:]}
        before = copy.deepcopy((events, groups))
        actual = self.adapter.batched_metrics(events, groups)
        self.assertEqual(actual, {k: self.frozen(v) for k, v in groups.items()})
        self.assertEqual(list(actual), list(groups))
        self.assertEqual((events, groups), before)

    def test_empty_groups(self):
        for groups in ({}, {('total', None): [], ('status', 'unowned'): [],
                           ('per_stream', 'None'): [], ('leaf', 'empty'): []}):
            self.assertEqual(self.adapter.batched_metrics([], groups),
                             {k: self.frozen(v) for k, v in groups.items()})

    def test_one_stable_sort(self):
        events = [dict(ts=0, dur=d, label=i) for i, d in enumerate([2, 1, 1, 3])]
        with mock.patch('builtins.sorted', wraps=sorted) as sorter:
            self.adapter.batched_metrics(events, {k: events for k in range(9)})
        self.assertEqual(sorter.call_count, 1)
        ordered = sorted(*sorter.call_args.args, **sorter.call_args.kwargs)
        self.assertEqual([e['label'] for e in ordered], [1, 2, 0, 3])

    def test_original_fractional_sum_negative_control(self):
        events = [dict(ts=t, dur=d) for t, d in [(2, 0.1), (1, 0.2), (0, 0.3)]]
        expected = self.frozen(events)
        actual = self.adapter.batched_metrics(events, {'total': events})['total']
        wrong = self.frozen(sorted(events, key=lambda e: (e['ts'], e['ts'] + e['dur'])))
        self.assertEqual(actual, expected)
        self.assertNotEqual(wrong['sum_ms'], expected['sum_ms'])
        self.assertEqual(wrong['count'], expected['count'])
        self.assertEqual(wrong['union_ms'], expected['union_ms'])

    def test_full_frozen_ast_and_predicate_negative_control(self):
        stage = load(STAGE)
        staged = stage.instrument(self.source)
        corrected = self.adapter.instrument_metrics(copy.deepcopy(staged))
        trace = next(n for n in corrected.body if isinstance(n, ast.FunctionDef) and n.name == 'trace')
        expected_insertions = ast.parse("""
_metric_events = [e for _, e in kernels]
_metric_results = batched_metrics(_metric_events, {
    ('total', None): _metric_events,
    **{('per_stream', k): [e for _, e, _ in v] for k, v in streams.items()},
    **{('status', k): v for k, v in statuses.items()},
    **{('hierarchy', k): v for k, v in hierarchy.items()},
    **{('leaf', k): v for k, v in leaf.items()},
})
""").body
        pos = next(i for i, n in enumerate(trace.body) if ast.dump(n) == ast.dump(expected_insertions[0]))
        self.assertEqual([ast.dump(n) for n in trace.body[pos:pos+2]],
                         [ast.dump(n) for n in expected_insertions])
        self.assertEqual(ast.dump(trace.body[pos+2]), ast.dump(ast.parse('per_stream = {}').body[0]))
        del trace.body[pos:pos+2]
        definition = next(n for n in corrected.body if isinstance(n, ast.FunctionDef) and n.name == 'batched_metrics')
        namespace = {}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), '<injected>', 'exec'), namespace)
        self.assertEqual(namespace['batched_metrics']([], {'empty': []}), {'empty': self.frozen([])})
        corrected.body.remove(definition)
        reverse = {ast.dump(ast.parse(new, mode='eval').body): ast.parse(old, mode='eval').body
                   for new, old in [("_metric_results[('per_stream', stream)]", 'metrics([e for _, e, _ in items])'),
                                    ("_metric_results[('total', None)]", 'metrics([e for _, e in kernels])'),
                                    *[(f"_metric_results[('{kind}', k)]", 'metrics(v)')
                                      for kind in ('status', 'hierarchy', 'leaf')]]}
        hits = []

        class Restore(ast.NodeTransformer):
            def visit_Subscript(self, node):
                shape = ast.dump(node)
                if shape in reverse:
                    hits.append(shape)
                    return copy.deepcopy(reverse[shape])
                return self.generic_visit(node)

        corrected = Restore().visit(corrected)
        self.assertEqual(len(hits), 5)
        self.assertEqual(len(set(hits)), 5)
        self.assertEqual(ast.dump(corrected), ast.dump(staged))
        observations = {ast.dump(ast.parse(code).body[0]) for _, before, after in stage.RULES
                        for code in (before, after) if code}
        observations.update(ast.dump(ast.parse(code).body[0]) for _, code in stage.LOOPS)

        class RemoveStage(ast.NodeTransformer):
            def visit_Expr(self, node):
                return None if ast.dump(node) in observations else self.generic_visit(node)
            visit_If = visit_Expr

        restored = RemoveStage().visit(corrected)
        self.assertEqual(ast.dump(restored), ast.dump(ast.parse(self.source)))
        predicate = next(n for n in ast.walk(restored) if isinstance(n, ast.Compare)
                         and ast.unparse(n) == "row['local_num_tokens'] > 0")
        predicate.ops[0] = ast.GtE()
        with self.assertRaises(AssertionError):
            self.assertEqual(ast.dump(restored), ast.dump(ast.parse(self.source)))

    def test_pipeline_ast_and_original_file_binding(self):
        tree = self.adapter.pipeline_tree()
        original = ast.parse(self.adapter.PIPELINE.read_bytes())
        insertion = ast.parse('compose_metrics(observer.stage)').body[0]
        pos = next(i for i, n in enumerate(tree.body) if ast.dump(n) == ast.dump(insertion))
        self.assertEqual(ast.dump(tree.body[pos-1]), ast.dump(ast.parse('observer.stage._obs = observation').body[0]))
        del tree.body[pos]
        self.assertEqual(ast.dump(tree), ast.dump(original))
        argv = ['adapter.py', 'traces/prefill/dp0/input.trace.json.gz', 'fresh_output_prefix']
        with mock.patch.object(sys, 'argv', argv), mock.patch('builtins.exec') as execute:
            self.adapter.main()
            self.assertEqual(sys.argv, argv)
        code, namespace = execute.call_args.args
        self.assertEqual(code.co_filename, str(self.adapter.PIPELINE))
        self.assertEqual(namespace['__file__'], str(self.adapter.PIPELINE))

    def test_composition_preserves_observer_and_runner(self):
        stage = load(STAGE)
        observation, run = stage._obs, stage.run
        self.adapter.compose_metrics(stage)
        self.assertIs(stage._obs, observation)
        self.assertIs(stage.run, run)
        self.assertIn('batched_metrics', [n.name for n in stage.instrument(self.source).body
                                       if isinstance(n, ast.FunctionDef)])


if __name__ == '__main__':
    unittest.main()
