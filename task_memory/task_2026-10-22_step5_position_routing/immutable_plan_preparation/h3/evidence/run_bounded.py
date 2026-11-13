"""Complete the same one-variable pair with scoped actual-point checks."""
import ast
from pathlib import Path

ORIGINAL = Path(__file__).resolve().with_name('run.py')
original_nodes = ast.parse(ORIGINAL.read_text()).body
cut = next(index for index, node in enumerate(original_nodes) if isinstance(node, ast.Assign) and
           any(isinstance(target, ast.Name) and target.id == 'pricing_out' for target in node.targets))
exec(compile(ast.Module(body=original_nodes[:cut + 1], type_ignores=[]), str(ORIGINAL), 'exec'), globals())

from types import MethodType, SimpleNamespace
from dataclasses import replace
from copy import deepcopy

predictor = namespace['predictor']
manager = namespace['manager']
lane = namespace['workload'].lane(0)
checks = []
Path.read_bytes = fixed_file_byte_adapter

def expect_error(name, function):
    try:
        function()
    except ValueError as error:
        checks.append({'name': name, 'error_type': type(error).__name__, 'error': str(error)})
    else:
        raise AssertionError(f'{name} unexpectedly succeeded')

with predictor._step5_measured_layer(60):
    context, queries, plans = predictor._step5_position_queries(lane)
    model, precision = predictor._step5_ffn_model('moe_grouped_gemm', context=context)
    def point():
        return predictor._step5_position_queries(lane)
    def price():
        value = predictor._predict_step5_ffn_operator('moe_grouped_gemm', queries['moe_grouped_gemm'], context=context)
        return value, predictor._step5_ffn_costs['moe_grouped_gemm']['kernel_count']
    original_point = context, queries, plans
    baseline_value, baseline_count = price()
    rows, identity = model._frontier_moe_physical_rows, model._frontier_measurement_identity
    model._frontier_moe_physical_rows = tuple(list(rows))
    assert point() == original_point
    checks.append({'name': 'actual physical-row tuple replacement', 'equal': True})
    model._frontier_moe_physical_rows = ()
    expect_error('empty actual rows', point)
    model._frontier_moe_physical_rows = rows
    assert point() == original_point
    model._frontier_measurement_identity = identity + ' '
    expect_error('duration/count identity replacement', price)
    model._frontier_measurement_identity = identity
    assert price() == (baseline_value, baseline_count)
    key = tuple(float(queries['moe_grouped_gemm'][name]) for name in model._frontier_feature_names)
    duration = model._frontier_exact_lookup[key]
    count_model = model._frontier_kernel_count_model
    count = count_model._frontier_exact_lookup[key]
    model._frontier_exact_lookup[key] = duration * 1.1
    assert price() == (duration * 1.1, baseline_count)
    checks.append({'name': 'live duration lookup', 'before': duration, 'after': duration * 1.1})
    model._frontier_exact_lookup[key] = duration
    count_model._frontier_exact_lookup[key] = count + 1
    assert price() == (baseline_value, count + 1)
    checks.append({'name': 'live count lookup', 'before': count, 'after': count + 1})
    count_model._frontier_exact_lookup[key] = count
    replacement_model = deepcopy(model)
    previous_family = manager._active_measurement_type
    manager._active_measurement_type = predictor._active_measurement_type
    manager._store_model_precision('moe_grouped_gemm', precision.name, replacement_model)
    assert point() == original_point
    selected, _ = predictor._step5_ffn_model('moe_grouped_gemm', context=context)
    assert selected is replacement_model and price() == (baseline_value, baseline_count)
    checks.append({'name': 'live estimator replacement', 'equal': True})
    manager._store_model_precision('moe_grouped_gemm', precision.name, model)
    manager._active_measurement_type = previous_family

old_layer, old_workload = namespace['layer_id'], namespace['workload']
namespace['layer_id'] = 61
namespace['workload'] = replace(old_workload, global_layer_id=61)
exec(wave_code, namespace)
layer61 = json.loads(payload(namespace['phases']))
assert all(proof['global_layer_id'] == 61 for row in layer61 for proof in row['proofs'].values())
checks.append({'name': 'fresh actual layer61 proofs', 'proofs': sum(len(row['proofs']) for row in layer61)})
namespace['layer_id'], namespace['workload'] = old_layer, old_workload

# Execute the unchanged default-data/custom-method differential guard block
# from the first harness. All six observations must match the original resolver.
guard_start = next(index for index, node in enumerate(original_nodes) if isinstance(node, ast.Assign) and
                   any(isinstance(target, ast.Name) and target.id == 'cfg' for target in node.targets))
guard_end = next(index for index, node in enumerate(original_nodes) if index > guard_start and
                 isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'stats' for target in node.targets))
exec(compile(ast.Module(body=original_nodes[guard_start:guard_end], type_ignores=[]), str(ORIGINAL), 'exec'), globals())

stats = pstats.Stats(str(pricing_out / 'warm.prof'))
def calls(name):
    return sum(values[1] for (_, _, function), values in stats.stats.items() if function == name)
assert calls('_get_on_demand_prediction') == 288 and calls('_predict_step5_ffn_operator') == 144
assert calls('loads') == 1136 and calls('dumps') == 800
assert calls('_measurement_identity') == 240
assert calls('_resolve_moe_layer_ids') == 144 and calls('is_moe_layer') == 16
Path.read_bytes = real_read_bytes
assert before == state(fixture)
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE).strip()
final = {'task_id': 'D29-IMMUTABLE-PLAN-PREPARATION-H3', 'session_id': 'ses_e963098ecffebqCOtdysKGY6Xe',
         'label': requested_label, 'pricing': pricing, 'live_checks': checks, 'guard_checks': guards,
         'custom_predicate_scan_calls': len(custom_calls),
         'warm_counts': {name: calls(name) for name in ('loads', 'dumps', '_get_on_demand_prediction',
              '_predict_step5_ffn_operator', '_measurement_identity', '_resolve_moe_layer_ids',
              'is_moe_layer', 'get_moe_layer_ids', 'profiling_name', 'bind_operator_query')},
         'source_clean': True, 'fixture_unchanged': True, 'no_new_cache': True}
(ROOT / 'default-contract-result.json').write_text(json.dumps(final, indent=2) + '\n')
print(json.dumps({key: value for key, value in final.items() if key not in ('pricing', 'live_checks', 'guard_checks')}, indent=2))
