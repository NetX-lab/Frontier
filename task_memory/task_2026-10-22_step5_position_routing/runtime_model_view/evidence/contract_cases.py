"""Probe selected-model failures and the explicit unrelated-model error change."""
from contextlib import contextmanager
from copy import copy
import hashlib
import json
from pathlib import Path
import sys
from fixture_wiring import setup, bind_production_owner

ROOT = Path(__file__).resolve().parent
output = ROOT / 'contracts'
output.mkdir()
frozen = json.loads((ROOT / 'freeze.json').read_text())
raw = Path(frozen['collective_original']).read_bytes()
assert hashlib.sha256(raw).hexdigest() == frozen['collective_sha256']
target = output / 'baseline/collective-fixture/h800/collective_local.json'
target.parent.mkdir(parents=True)
target.write_bytes(raw)
ns, wave, tree, node, setup_path = setup(frozen, 'baseline', output)
bind_production_owner(ns, output, 'baseline')
import frontier.execution_time_predictor.prediction_model_registry as registry
import frontier.execution_time_predictor.moe_routing_workload as workload
from frontier.types import ClusterType, MeasurementType

original_view = registry.PredictionModelRegistry._models_view_for_family
original_point = workload.MoeRoutingWorkload._step5_position_queries
view_globals, point_globals = dict(vars(registry)), dict(vars(workload))
exec(compile((ROOT / 'candidate/candidate_view.py').read_bytes(), 'candidate_view.py', 'exec'), view_globals)
exec(compile((ROOT / 'candidate/candidate_point.py').read_bytes(), 'candidate_point.py', 'exec'), point_globals)
candidate_view, candidate_point = view_globals['_models_view_for_family'], point_globals['_step5_position_queries']
predictor = ns['predictor']
manager = predictor._model_manager
role = predictor._cluster_type
family = predictor._measurement_family_name(predictor._active_measurement_type)
manager._active_measurement_type = predictor._active_measurement_type
exec(wave, ns)
record = ns['phases'].lane_records[0]
lane = record.batch.lane_workload
layer = record.execution_time.layer_execution_times[0].global_layer_id
records = []


def capture(call):
    try:
        return {'ok': True}, call()
    except Exception as error:
        return {'ok': False, 'type': type(error).__name__, 'message': str(error)}, None


def view(function, name=None, current_role=role):
    return function(manager, family, current_role, **({'model_name': name} if name is not None else {}))


def point(candidate):
    registry.PredictionModelRegistry._models_view_for_family = candidate_view if candidate else original_view
    function = candidate_point if candidate else original_point
    try:
        with predictor._step5_measured_layer(layer):
            return function(predictor, lane)
    finally:
        registry.PredictionModelRegistry._models_view_for_family = original_view


@contextmanager
def restore_registries():
    mappings = [manager._legacy_model_registry(family), manager._contract_model_registry(family)]
    nested = [manager._legacy_precision_registry(family), manager._contract_precision_registry(family)]
    snapshots = [(mapping, dict(mapping)) for mapping in mappings]
    snapshots += [(mapping, {key: dict(value) for key, value in mapping.items()}) for mapping in nested]
    try:
        yield
    finally:
        for mapping, snapshot in snapshots:
            mapping.clear()
            mapping.update(snapshot)


def collision(name):
    original = next(model for (key, identity), model in manager._contract_model_registry(family).items()
                    if key == name and isinstance(identity, tuple))
    alternate = copy(original)
    identity = json.loads(original._frontier_measurement_identity)
    identity['native_source'] = {'source_receipt': 'D29_CONTROL_ALTERNATE_SOURCE'}
    alternate._frontier_measurement_identity = json.dumps(identity, sort_keys=True)
    precision = next(key for key, bucket in manager._contract_precision_registry(family).items()
                     if any(model is original for model in bucket.values()))
    manager._store_model_precision(name, precision, alternate)


full = view(original_view)
assert list(view(candidate_view)) == list(full)
assert all(view(candidate_view)[key] is model for key, model in full.items())
expected = {key: model for key, model in full.items()
            if getattr(model, '_frontier_cache_model_name', None) == 'moe_grouped_gemm'}
selected = view(candidate_view, 'moe_grouped_gemm')
assert list(selected) == list(expected) and all(selected[key] is model for key, model in expected.items())
assert view(candidate_view, 'D29_ABSENT_MODEL') == {}
assert all(model._frontier_cache_model_name != 'moe_grouped_gemm_kernel_count' for model in selected.values())
assert all(name == model._frontier_cache_model_name
           for (name, identity), model in manager._contract_model_registry(family).items()
           if isinstance(identity, tuple))
records.append({'case': 'unfiltered order/objects, selected set, absent name, count exclusion and writer names', 'pass': True,
                'full_models': len(full), 'selected_models': len(selected)})
normal = point(False)
assert point(True) == normal
records.append({'case': 'normal actual point result', 'pass': True})

for name in ('moe_grouped_gemm', 'moe_gating_linear'):
    with restore_registries():
        collision(name)
        before, _ = capture(lambda: view(original_view))
        default, _ = capture(lambda: view(candidate_view))
        old_point, _ = capture(lambda: point(False))
        new_point, result = capture(lambda: point(True))
        assert before == default == old_point and not before['ok']
        assert 'Conflicting native model artifacts' in before['message']
        if name == 'moe_grouped_gemm':
            assert new_point == before
        else:
            assert new_point['ok'] and result == normal
        records.append({'case': 'registered native label collision: ' + name,
            'old_full_and_point': before, 'new_unfiltered': default, 'new_point': new_point,
            'equivalent': name == 'moe_grouped_gemm'})

with predictor._step5_measured_layer(layer):
    model, precision = predictor._step5_ffn_model('moe_grouped_gemm', context=normal[0])
rows, marker = model._frontier_moe_physical_rows, model._frontier_measurement_identity
for change in ('rows', 'identity', 'empty_rows', 'malformed_rows'):
    try:
        if change == 'rows': model._frontier_moe_physical_rows = tuple(list(rows))
        if change == 'identity': model._frontier_measurement_identity = marker + ' '
        if change == 'empty_rows': model._frontier_moe_physical_rows = ()
        if change == 'malformed_rows': model._frontier_moe_physical_rows = ({**rows[0], 'routing_input': '{'},)
        before, result_before = capture(lambda: point(False))
        after, result_after = capture(lambda: point(True))
        assert before == after
        if change in ('rows', 'identity'):
            assert before['ok'] and result_before == result_after == normal
        else:
            assert not before['ok']
        records.append({'case': change, 'old': before, 'new': after, 'pass': True})
    finally:
        model._frontier_moe_physical_rows, model._frontier_measurement_identity = rows, marker

with restore_registries():
    replacement = copy(model)
    manager._store_model_precision('moe_grouped_gemm', precision.name, replacement)
    assert any(value is replacement for value in view(candidate_view, 'moe_grouped_gemm').values())
    assert point(False) == point(True) == normal
    records.append({'case': 'live estimator replacement via production registrar', 'pass': True})

original_type = predictor._active_measurement_type
for measurement in (MeasurementType.KERNEL_ONLY, MeasurementType.CUDA_EVENT):
    predictor._active_measurement_type = measurement
    before, result_before = capture(lambda: point(False))
    after, result_after = capture(lambda: point(True))
    assert before == after and (not before['ok'] or result_before == result_after)
    records.append({'case': 'live family ' + measurement.value, 'old': before, 'new': after, 'pass': True})
predictor._active_measurement_type = original_type
assert point(False) == point(True) == normal
assert view(original_view) == full
for relative, expected_hash in frozen['sources']['baseline']['hashes'].items():
    assert hashlib.sha256((Path(frozen['sources']['baseline']['root']) / relative).read_bytes()).hexdigest() == expected_hash
receipt = {'task_id': 'D29-NARROW-MODEL-VIEW-CONTRACT', 'cases': records,
    'status': 'NORMAL_AND_SELECTED_FAILURES_MATCH_UNRELATED_ERROR_TIMING_DIFFERS',
    'limits': 'Cached-manager real point; this does not test every public registry state or establish full CLI/native acceptance.'}
(ROOT / 'contract_cases.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
