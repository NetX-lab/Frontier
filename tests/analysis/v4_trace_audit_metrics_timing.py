#!/usr/bin/env python3
"""Observe the accepted metrics batch without replacing its calculations.

Usage: /usr/bin/python3 -B v4_trace_audit_metrics_timing.py RELATIVE_TRACE FRESH_PREFIX
The original loading-GC pipeline retains its observation and output policies.
"""

import ast
import copy
import hashlib
import importlib.util
from pathlib import Path

ADAPTER = Path(__file__).with_name('v4_trace_audit_metrics.py')
ADAPTER_SHA256 = '6dff04f36dbcd8a66a298e4f6ae88d382bb29813c8a98f8fea91372b271b788a'
PREFIX = 'DEBUG-V4-METRICS-TAIL-01'
SORT_WRAPPER = '_metrics_tail_sorted'
SETUP = f"""
_obs('{PREFIX}:batch_start', events=len(events), groups=len(groups),
     memberships=sum(len(members) for members in groups.values()))
def {SORT_WRAPPER}(original_sorted, *args, **kwargs):
    _obs('{PREFIX}:sort_start')
    ordered = original_sorted(*args, **kwargs)
    _obs('{PREFIX}:sort_done')
    _obs('{PREFIX}:union_start')
    return ordered
_obs('{PREFIX}:group_construction_start')
"""


def boundary(name, **expressions):
    arguments = ', '.join(f'{key}={value}' for key, value in expressions.items())
    return ast.parse(f"_obs('{PREFIX}:{name}'{', ' if arguments else ''}{arguments})").body[0]


def instrument_timing(tree):
    """Copy the generated AST and observe only its batched_metrics definition."""
    tree = copy.deepcopy(tree)
    definitions = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == 'batched_metrics']
    assert len(definitions) == 1, 'Expected the accepted batched_metrics definition'
    definition = definitions[0]
    assert len(definition.body) == 6, 'Expected the accepted six-statement function'
    docstring, initialization, construction, recurrence, finalization, result = definition.body
    assert isinstance(construction, ast.For) and isinstance(recurrence, ast.For)
    assert ast.dump(construction.iter) == ast.dump(ast.parse('groups.items()', mode='eval').body)
    expected_sort = ast.parse("sorted(events, key=lambda e: (e['ts'], e['ts'] + e['dur']))",
                              mode='eval').body
    assert ast.dump(recurrence.iter) == ast.dump(expected_sort), 'Expected one original sort'
    original_sort = recurrence.iter
    recurrence.iter = ast.Call(func=ast.Name(id=SORT_WRAPPER, ctx=ast.Load()),
                               args=[original_sort.func, *original_sort.args],
                               keywords=original_sort.keywords)
    definition.body = [docstring, *ast.parse(SETUP).body, initialization, construction,
                       boundary('group_construction_done', unique_events='len(memberships)'),
                       recurrence, boundary('union_done'), finalization,
                       boundary('batch_done'), result]
    return ast.fix_missing_locations(tree)


def compose_timing(stage):
    """Wrap the already accepted stage.instrument without changing stage._obs."""
    original = stage.instrument

    def instrument(source):
        return instrument_timing(original(source))

    stage.instrument = instrument


def load_adapter():
    source = ADAPTER.read_bytes()
    assert hashlib.sha256(source).hexdigest() == ADAPTER_SHA256, 'Accepted adapter pin mismatch'
    spec = importlib.util.spec_from_file_location('accepted_metrics_adapter', ADAPTER)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def pipeline_tree(adapter):
    """Compose immediately after the accepted adapter's pipeline insertion."""
    tree = adapter.pipeline_tree()
    anchor = ast.dump(ast.parse('compose_metrics(observer.stage)').body[0])
    positions = [i for i, node in enumerate(tree.body) if ast.dump(node) == anchor]
    assert len(positions) == 1, 'Expected the accepted metrics composition'
    tree.body.insert(positions[0] + 1, ast.parse('compose_timing(observer.stage)').body[0])
    return ast.fix_missing_locations(tree)


def main():
    adapter = load_adapter()
    namespace = {'__name__': 'frozen_loading_gc_pipeline', '__file__': str(adapter.PIPELINE),
                 'compose_metrics': adapter.compose_metrics, 'compose_timing': compose_timing}
    exec(compile(pipeline_tree(adapter), str(adapter.PIPELINE), 'exec'), namespace)


if __name__ == '__main__':
    main()
