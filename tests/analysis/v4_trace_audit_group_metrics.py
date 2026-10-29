#!/usr/bin/env python3
"""Compute frozen group metrics using numeric positions in one stable order.

Usage: /usr/bin/python3 -B v4_trace_audit_group_metrics.py RELATIVE_TRACE FRESH_PREFIX
The accepted adapter supplies every trace substitution and pipeline policy.
"""

import ast
import copy
import hashlib
import importlib.util
import inspect
from pathlib import Path

ADAPTER = Path(__file__).with_name('v4_trace_audit_metrics.py')
ADAPTER_SHA256 = '6dff04f36dbcd8a66a298e4f6ae88d382bb29813c8a98f8fea91372b271b788a'
assert hashlib.sha256(ADAPTER.read_bytes()).hexdigest() == ADAPTER_SHA256, 'Accepted adapter pin mismatch'
SPEC = importlib.util.spec_from_file_location('accepted_metrics_adapter', ADAPTER)
ACCEPTED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACCEPTED)
PIPELINE = ACCEPTED.PIPELINE
pipeline_tree = ACCEPTED.pipeline_tree
PREFIX = 'DEBUG-V4-METRICS-GROUP-02'


def batched_metrics(events, groups):
    """Keep original group sums and merge each group's ordered intervals."""
    order = {id(event): position for position, event in enumerate(
        sorted(events, key=lambda e: (e['ts'], e['ts'] + e['dur'])))}
    results = {}
    for key, members in groups.items():
        result = {'count': len(members),
                  'sum_ms': sum(e['dur'] for e in members) / 1000,
                  'union_ms': 0.0}
        union, end = 0.0, float('-inf')
        for event in sorted(members, key=lambda e: order[id(e)]):
            a, b = event['ts'], event['ts'] + event['dur']
            union += max(0.0, b - max(a, end))
            end = max(end, b)
        result['union_ms'] = union / 1000
        results[key] = result
    return results


def instrument_groups(tree):
    """Replace only the accepted metrics definition in a copy of its AST."""
    tree = copy.deepcopy(tree)
    positions = [i for i, node in enumerate(tree.body)
                 if isinstance(node, ast.FunctionDef) and node.name == 'batched_metrics']
    assert len(positions) == 1, 'Expected the accepted metrics definition'
    original = ast.parse(inspect.getsource(ACCEPTED.batched_metrics)).body[0]
    assert ast.dump(tree.body[positions[0]]) == ast.dump(original), 'Accepted metrics AST mismatch'
    replacement = ast.parse(inspect.getsource(batched_metrics)).body[0]
    replacement.body.insert(1, ast.parse(
        f"_obs('{PREFIX}:start', events=len(events), groups=len(groups), "
        "memberships=sum(len(members) for members in groups.values()))").body[0])
    replacement.body.insert(-1, ast.parse(f"_obs('{PREFIX}:end')").body[0])
    tree.body[positions[0]] = replacement
    return ast.fix_missing_locations(tree)


def compose_metrics(stage):
    """Reuse the accepted substitutions before replacing the metrics function."""
    ACCEPTED.compose_metrics(stage)
    original = stage.instrument

    def instrument(source):
        return instrument_groups(original(source))

    stage.instrument = instrument


def main():
    namespace = {'__name__': 'frozen_loading_gc_pipeline', '__file__': str(PIPELINE),
                 'compose_metrics': compose_metrics}
    exec(compile(pipeline_tree(), str(PIPELINE), 'exec'), namespace)


if __name__ == '__main__':
    main()
