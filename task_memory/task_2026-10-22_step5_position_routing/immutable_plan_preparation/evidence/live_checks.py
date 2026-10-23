"""Real-wave live point/model/numeric checks for the derived-key-only control."""
import ast
import sys
from pathlib import Path

label = sys.argv[1]
assert label in ('live-current4', 'live-key')
driver = Path(__file__).resolve().with_name('probe.py')
prefix = []
for node in ast.parse(driver.read_text()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'timings' for t in node.targets):
        break
    if isinstance(node, ast.Assert) and ast.unparse(node.test) == "LABEL == 'red'":
        continue
    prefix.append(node)
exec(compile(ast.Module(body=prefix, type_ignores=[]), str(driver), 'exec'), globals())

from copy import deepcopy
from dataclasses import fields, replace
from frontier.cc_backend.collective_plan import CollectiveTensor, CollectivePlanCost

def key_once(plan):
    if '_d29_derived_key' not in plan.__dict__:
        object.__setattr__(plan, '_d29_derived_key', real_key(plan))
    return plan._d29_derived_key

CollectivePlan.cache_key = key_once if LABEL == 'live-key' else real_key
checks = []

def run_wave():
    exec(wave_code, namespace)
    return json.loads(payload(namespace['phases']))

def expect_error(name, function):
    try:
        function()
    except (ValueError, KeyError) as error:
        checks.append({'name': name, 'error_type': type(error).__name__, 'error': str(error)})
    else:
        raise AssertionError(f'{name} unexpectedly succeeded')

constructors = Counter()
original_init = {}
for cls in (CollectivePlan, CollectiveTensor, CollectivePlanCost):
    original_init[cls] = cls.__init__
    def wrapper(self, *args, _cls=cls, **kwargs):
        constructors[_cls.__name__] += 1
        return original_init[_cls](self, *args, **kwargs)
    cls.__init__ = wrapper
baseline = run_wave()
for cls, initializer in original_init.items():
    cls.__init__ = initializer
predictor = namespace['predictor']
manager = namespace['manager']
lane = namespace['workload'].lane(0)
with predictor._step5_measured_layer(60):
    context, queries, plans = predictor._step5_position_queries(lane)
    model, precision = predictor._step5_ffn_model('moe_grouped_gemm', context=context)
    routed_key = tuple(float(queries['moe_grouped_gemm'][name]) for name in model._frontier_feature_names)
rows = model._frontier_moe_physical_rows
identity = model._frontier_measurement_identity
duration = model._frontier_exact_lookup[routed_key]
count_model = model._frontier_kernel_count_model
count = count_model._frontier_exact_lookup[routed_key]

def assert_restored():
    assert run_wave() == baseline

model._frontier_moe_physical_rows = tuple(list(rows))
assert_restored()
checks.append({'name': 'same-content physical-row tuple replacement', 'equal': True})
model._frontier_moe_physical_rows = ()
expect_error('empty physical-row tuple', run_wave)
model._frontier_moe_physical_rows = rows
assert_restored()

model._frontier_measurement_identity = identity + ' '
expect_error('duration/count measurement identity replacement', run_wave)
model._frontier_measurement_identity = identity
assert_restored()

model._frontier_exact_lookup[routed_key] = duration * 1.1
changed = run_wave()
assert changed != baseline
assert changed[0]['moe']['moe_grouped_gemm'] == duration * 1.1
checks.append({'name': 'live exact duration point', 'before': duration, 'after': changed[0]['moe']['moe_grouped_gemm']})
model._frontier_exact_lookup[routed_key] = duration
assert_restored()
count_model._frontier_exact_lookup[routed_key] = count + 1
changed = run_wave()
assert changed[0]['proofs']['moe_grouped_gemm']['kernel_count'] == count + 1
checks.append({'name': 'live exact kernel count point', 'before': count, 'after': changed[0]['proofs']['moe_grouped_gemm']['kernel_count']})
count_model._frontier_exact_lookup[routed_key] = count
assert_restored()

replacement_model = deepcopy(model)
previous_manager_measurement = manager._active_measurement_type
manager._active_measurement_type = predictor._active_measurement_type
manager._store_model_precision('moe_grouped_gemm', precision.name, replacement_model)
assert_restored()
with predictor._step5_measured_layer(60):
    selected, _ = predictor._step5_ffn_model('moe_grouped_gemm', context=context)
    assert selected is replacement_model
checks.append({'name': 'actual estimator replacement selected', 'equal': True})
manager._store_model_precision('moe_grouped_gemm', precision.name, model)
manager._active_measurement_type = previous_manager_measurement
assert_restored()

for field_name in ('context_json', 'buffer_json'):
    sample = plans['step5_moe_dispatch']
    value = json.loads(getattr(sample, field_name))
    value['d29_live_point_probe'] = True
    changed_plan = replace(sample, **{field_name: json.dumps(value)})
    assert CollectivePlan.cache_key(changed_plan) == real_key(changed_plan) != CollectivePlan.cache_key(sample)
    expect_error(f'complete plan {field_name} replacement', lambda: predictor._cc_backend.predict_collective_plan(changed_plan))

old_layer, old_workload = namespace['layer_id'], namespace['workload']
namespace['layer_id'] = 61
namespace['workload'] = replace(old_workload, global_layer_id=61)
layer61 = run_wave()
assert all(proof['global_layer_id'] == 61 for row in layer61 for proof in row['proofs'].values())
assert all(proof['global_layer_id'] == 60 for row in baseline for proof in row['proofs'].values())
checks.append({'name': 'actual layer61 proofs fresh', 'proofs': sum(len(row['proofs']) for row in layer61)})
namespace['layer_id'], namespace['workload'] = old_layer, old_workload
assert_restored()

source = old_workload.routing_source
first_lane = source.source_lanes[0]
changed_lane = replace(first_lane, intervals=tuple(replace(interval, start=7000) for interval in first_lane.intervals))
changed_source = replace(source, source_lanes=(changed_lane, *source.source_lanes[1:]))
namespace['workload'] = namespace['materialize_layer_ep_workload'](policy_id='position_cyclic', routing_source=changed_source,
    target_replica_id=1, global_layer_id=60, routing_token_count=changed_source.gathered_physical_rows,
    router_topk=8, total_expert_num=352, moe_expert_parallel_size=16,
    expert_to_ep=namespace['build_contiguous_expert_ownership'](352, 16))
source7000 = run_wave()
for record in namespace['phases'].lane_records:
    frame = record.execution_time.layer_execution_times[0].moe_or_mlp_time_component.operator_times.physical_frame
    assert frame['routing_source'] == changed_source and frame['routing_source'] != source
checks.append({'name': 'different measured source intervals at7000 selected live',
               'source_start': 7000, 'payload_sha256': hashlib.sha256(json.dumps(source7000, sort_keys=True).encode()).hexdigest()})
namespace['workload'] = old_workload
assert_restored()

retained = list(predictor._step5_collective_receipts[4].values())
for plan, _, _ in retained:
    assert type(plan.phase) is str and type(plan.graph_mode) is str
    assert type(plan.context_json) is str and type(plan.buffer_json) is str
    assert all(type(v) is int for v in plan.sizes)
    assert all(type(v) is int for group in plan.groups for v in group)
    assert all(type(tensor) is CollectiveTensor and all(type(getattr(tensor, field.name)) in (str, int)
               for field in fields(tensor)) for tensor in plan.tensors)

Path.read_bytes = real_read_bytes
assert before == state(fixture)
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE).strip()
result = {'task_id': 'D29-IMMUTABLE-PLAN-PREPARATION', 'label': LABEL, 'revision': REV,
          'constructors_cold': dict(constructors), 'checks': checks, 'fixture_unchanged': True,
          'all_restored_waves_equal': True, 'real_retained_plans_immutable_fields': True}
(OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key != 'checks'}, indent=2))
