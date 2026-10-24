#!/usr/bin/env python3
"""Batch only the frozen auditor's metrics interval sorts.

Usage: /usr/bin/python3 -B v4_trace_audit_metrics.py RELATIVE_TRACE FRESH_PREFIX
The original loading-GC pipeline supplies loading, observation and output policy.
"""

import ast
import copy
import hashlib
import inspect
from pathlib import Path

PIPELINE = Path('/data/ycfeng/tmp/glm53_dsv4/p0_gc_observation_20261024/load_without_gc.py')
PIPELINE_SHA256 = '6b91e25fc592d4f1340ef22e6d6e73c4dd8ed6e7ce6239f7a113d946a74edd02'
SUBSTITUTIONS = (
    ('metrics([e for _, e, _ in items])', "_metric_results[('per_stream', stream)]"),
    ('metrics([e for _, e in kernels])', "_metric_results[('total', None)]"),
    ('{k: metrics(v) for k, v in statuses.items()}',
     "{k: _metric_results[('status', k)] for k, v in statuses.items()}"),
    ('{k: metrics(v) for k, v in hierarchy.items()}',
     "{k: _metric_results[('hierarchy', k)] for k, v in hierarchy.items()}"),
    ('{k: metrics(v) for k, v in leaf.items()}',
     "{k: _metric_results[('leaf', k)] for k, v in leaf.items()}"),
)
BATCH_INSERTION = """
_metric_events = [e for _, e in kernels]
_metric_results = batched_metrics(_metric_events, {
    ('total', None): _metric_events,
    **{('per_stream', k): [e for _, e, _ in v] for k, v in streams.items()},
    **{('status', k): v for k, v in statuses.items()},
    **{('hierarchy', k): v for k, v in hierarchy.items()},
    **{('leaf', k): v for k, v in leaf.items()},
})
"""


def batched_metrics(events, groups):
    """Preserve group count/sum order and merge intervals in one stable order."""
    results, states, memberships = {}, {}, {}
    for key, members in groups.items():
        results[key] = {'count': len(members),
                        'sum_ms': sum(e['dur'] for e in members) / 1000,
                        'union_ms': 0.0}
        state = states[key] = [0.0, float('-inf')]
        # Frozen groups retain original event objects and repeated memberships.
        for event in members:
            memberships.setdefault(id(event), []).append(state)
    for event in sorted(events, key=lambda e: (e['ts'], e['ts'] + e['dur'])):
        a, b = event['ts'], event['ts'] + event['dur']
        for state in memberships[id(event)]:
            state[0] += max(0.0, b - max(a, state[1]))
            state[1] = max(state[1], b)
    for key, result in results.items():
        result['union_ms'] = states[key][0] / 1000
    return results


def instrument_metrics(tree):
    """Compose after stage.instrument; match the five complete AST expressions."""
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == 'trace']
    assert len(functions) == 1, 'Expected the frozen trace function'
    trace = functions[0]
    replacements = {ast.dump(ast.parse(old, mode='eval').body):
                    ast.parse(new, mode='eval').body for old, new in SUBSTITUTIONS}
    hits = {shape: 0 for shape in replacements}

    class MetricsCalls(ast.NodeTransformer):
        def substitute(self, node):
            shape = ast.dump(node)
            if shape in replacements:
                hits[shape] += 1
                return copy.deepcopy(replacements[shape])
            return self.generic_visit(node)

        visit_Call = substitute
        visit_DictComp = substitute

    MetricsCalls().visit(trace)
    assert all(count == 1 for count in hits.values()), hits
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == 'metrics' for node in ast.walk(trace))
    anchor = ast.dump(ast.parse('per_stream = {}').body[0])
    positions = [i for i, node in enumerate(trace.body) if ast.dump(node) == anchor]
    assert len(positions) == 1, 'Expected the frozen per_stream initializer'
    trace.body[positions[0]:positions[0]] = ast.parse(BATCH_INSERTION).body
    definition = ast.parse(inspect.getsource(batched_metrics)).body[0]
    tree.body.insert(tree.body.index(trace), definition)
    return ast.fix_missing_locations(tree)


def compose_metrics(stage):
    original = stage.instrument

    def instrument(source):
        return instrument_metrics(original(source))

    stage.instrument = instrument


def pipeline_tree():
    source = PIPELINE.read_bytes()
    assert hashlib.sha256(source).hexdigest() == PIPELINE_SHA256, 'Pipeline pin mismatch'
    tree = ast.parse(source, filename=str(PIPELINE))
    anchor = ast.dump(ast.parse('observer.stage._obs = observation').body[0])
    positions = [i for i, node in enumerate(tree.body) if ast.dump(node) == anchor]
    assert len(positions) == 1, 'Expected the original observation assignment'
    tree.body.insert(positions[0] + 1, ast.parse('compose_metrics(observer.stage)').body[0])
    return ast.fix_missing_locations(tree)


def main():
    namespace = {'__name__': 'frozen_loading_gc_pipeline', '__file__': str(PIPELINE),
                 'compose_metrics': compose_metrics}
    exec(compile(pipeline_tree(), str(PIPELINE), 'exec'), namespace)


if __name__ == '__main__':
    main()
