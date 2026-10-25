#!/usr/bin/env python3
"""Reuse interval columns while preserving the accepted group audit pipeline.

Usage: /usr/bin/python3 -B v4_trace_audit_interval_metrics.py RELATIVE_TRACE FRESH_PREFIX
"""

import ast
import copy
import hashlib
import importlib.util
import inspect
from pathlib import Path

ADAPTER = Path(__file__).with_name('v4_trace_audit_group_metrics.py')
ADAPTER_SHA256 = '4dec265e20426a20e437588a4e44828e257247c82897cc8188621f93369a5251'
assert hashlib.sha256(ADAPTER.read_bytes()).hexdigest() == ADAPTER_SHA256, 'Accepted group adapter pin mismatch'
SPEC = importlib.util.spec_from_file_location('accepted_group_metrics_adapter', ADAPTER)
ACCEPTED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACCEPTED)
PIPELINE = ACCEPTED.PIPELINE
pipeline_tree = ACCEPTED.pipeline_tree


def batched_metrics(events, groups):
    """Keep member sums unchanged and merge cached intervals by integer position."""
    ordered = sorted(events, key=lambda e: (e['ts'], e['ts'] + e['dur']))
    starts = [event['ts'] for event in ordered]
    ends = [event['ts'] + event['dur'] for event in ordered]
    order = {id(event): position for position, event in enumerate(ordered)}
    results = {}
    for key, members in groups.items():
        result = {'count': len(members),
                  'sum_ms': sum(e['dur'] for e in members) / 1000,
                  'union_ms': 0.0}
        union, end = 0.0, float('-inf')
        positions = (range(len(ordered)) if members is events else
                     sorted(order[id(event)] for event in members))
        for position in positions:
            a, b = starts[position], ends[position]
            union += max(0.0, b - max(a, end))
            end = max(end, b)
        result['union_ms'] = union / 1000
        results[key] = result
    return results


def instrument_intervals(tree):
    """Replace only the exact accepted metrics AST in a copied composed tree."""
    tree = copy.deepcopy(tree)
    positions = [i for i, node in enumerate(tree.body)
                 if isinstance(node, ast.FunctionDef) and node.name == 'batched_metrics']
    assert len(positions) == 1, 'Expected the accepted group metrics definition'
    original = ACCEPTED.instrument_groups(ast.parse(
        inspect.getsource(ACCEPTED.ACCEPTED.batched_metrics))).body[0]
    assert ast.dump(tree.body[positions[0]]) == ast.dump(original), 'Accepted group metrics AST mismatch'
    replacement = ast.parse(inspect.getsource(batched_metrics)).body[0]
    replacement.body.insert(1, copy.deepcopy(original.body[1]))
    replacement.body.insert(-1, copy.deepcopy(original.body[-2]))
    tree.body[positions[0]] = replacement
    return ast.fix_missing_locations(tree)


def compose_metrics(stage):
    """Reuse the accepted composition before replacing its metrics definition."""
    ACCEPTED.compose_metrics(stage)
    original = stage.instrument

    def instrument(source):
        return instrument_intervals(original(source))

    stage.instrument = instrument


def main():
    namespace = {'__name__': 'frozen_loading_gc_pipeline', '__file__': str(PIPELINE),
                 'compose_metrics': compose_metrics}
    exec(compile(pipeline_tree(), str(PIPELINE), 'exec'), namespace)


if __name__ == '__main__':
    main()
