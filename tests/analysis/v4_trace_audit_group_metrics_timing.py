#!/usr/bin/env python3
"""Observe the frozen group metrics through sparse segment boundaries.

Usage: /usr/bin/python3 -B v4_trace_audit_group_metrics_timing.py RELATIVE_TRACE FRESH_PREFIX
The accepted group adapter retains every CLI substitution and pipeline policy.
"""

import ast
import copy
import hashlib
import importlib.util
from pathlib import Path

ADAPTER = Path(__file__).with_name('v4_trace_audit_group_metrics.py')
ADAPTER_SHA256 = '4dec265e20426a20e437588a4e44828e257247c82897cc8188621f93369a5251'
PREFIX = 'DEBUG-V4-GROUP-METRICS-SEGMENTS-01'
GLOBAL_SORT = '_group_metrics_global_sorted'
GROUP_SORT = '_group_metrics_group_sorted'
WRAPPERS = f"""
def {GLOBAL_SORT}(original_sorted, *args, **kwargs):
    _obs('{PREFIX}:global_sort_start')
    ordered = original_sorted(*args, **kwargs)
    _obs('{PREFIX}:global_sort_done')
    _obs('{PREFIX}:global_order_start')
    return ordered
def {GROUP_SORT}(original_sorted, *args, **kwargs):
    _obs('{PREFIX}:group_sort_start', group=repr(key), members=len(members))
    ordered = original_sorted(*args, **kwargs)
    _obs('{PREFIX}:group_sort_done', group=repr(key), members=len(members))
    _obs('{PREFIX}:group_union_start', group=repr(key), members=len(members))
    return ordered
"""


def boundary(name, group=False):
    values = ', group=repr(key), members=len(members)' if group else ''
    return ast.parse(f"_obs('{PREFIX}:{name}'{values})").body[0]


def wrap_sorted(original, wrapper):
    return ast.Call(func=ast.Name(id=wrapper, ctx=ast.Load()),
                    args=[original.func, *original.args], keywords=original.keywords)


def instrument_timing(tree):
    """Copy and exactly match the accepted composed function before insertion."""
    source = ADAPTER.read_bytes()
    assert hashlib.sha256(source).hexdigest() == ADAPTER_SHA256, 'Accepted group pin mismatch'
    reference = next(node for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef) and node.name == 'batched_metrics')
    reference.body.insert(1, ast.parse(
        "_obs('DEBUG-V4-METRICS-GROUP-02:start', events=len(events), groups=len(groups), "
        "memberships=sum(len(members) for members in groups.values()))").body[0])
    reference.body.insert(-1, ast.parse("_obs('DEBUG-V4-METRICS-GROUP-02:end')").body[0])
    tree = copy.deepcopy(tree)
    definitions = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == 'batched_metrics']
    assert len(definitions) == 1, 'Expected the accepted group metrics definition'
    definition = definitions[0]
    assert ast.dump(definition) == ast.dump(reference), 'Accepted composed group AST mismatch'
    docstring, start, order, results, groups, done, returned = definition.body
    original_sort = order.value.generators[0].iter.args[0]
    order.value.generators[0].iter.args[0] = wrap_sorted(original_sort, GLOBAL_SORT)
    count_sum, initialization, recurrence, union_result, group_result = groups.body
    recurrence.iter = wrap_sorted(recurrence.iter, GROUP_SORT)
    groups.body = [boundary('group_count_sum_start', group=True), count_sum,
                   boundary('group_count_sum_done', group=True), initialization,
                   recurrence, boundary('group_union_done', group=True),
                   union_result, group_result]
    definition.body = [docstring, *ast.parse(WRAPPERS).body, start, order,
                       boundary('global_order_done'), results, groups, done, returned]
    return ast.fix_missing_locations(tree)


def compose_timing(stage):
    """Observe after the accepted composition without replacing its observer."""
    original = stage.instrument

    def instrument(source):
        return instrument_timing(original(source))

    stage.instrument = instrument


def load_adapter():
    source = ADAPTER.read_bytes()
    assert hashlib.sha256(source).hexdigest() == ADAPTER_SHA256, 'Accepted group pin mismatch'
    spec = importlib.util.spec_from_file_location('accepted_group_metrics_adapter', ADAPTER)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def pipeline_tree(adapter):
    """Insert timing immediately after the accepted group adapter's composition."""
    tree = copy.deepcopy(adapter.pipeline_tree())
    anchor = ast.dump(ast.parse('compose_metrics(observer.stage)').body[0])
    positions = [i for i, node in enumerate(tree.body) if ast.dump(node) == anchor]
    assert len(positions) == 1, 'Expected the accepted group composition'
    tree.body.insert(positions[0] + 1, ast.parse('compose_timing(observer.stage)').body[0])
    return ast.fix_missing_locations(tree)


def main():
    adapter = load_adapter()
    namespace = {'__name__': 'frozen_loading_gc_pipeline', '__file__': str(adapter.PIPELINE),
                 'compose_metrics': adapter.compose_metrics, 'compose_timing': compose_timing}
    exec(compile(pipeline_tree(adapter), str(adapter.PIPELINE), 'exec'), namespace)


if __name__ == '__main__':
    main()
