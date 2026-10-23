"""Actual-source full-wave checks retained from the accepted residual probes."""
import cProfile
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import pstats
import statistics
import time
from unittest.mock import patch

store, plan = setup_namespace['store'], setup_namespace['plan']
trace = setup_namespace['bound_trace_source']
import frontier.metrics.ep_wave_metrics as ep
import frontier.metrics.op_trace_utils as op
from frontier.cc_backend.collective_plan import CollectivePlanCost
output = ROOT / LABEL
output.mkdir(exist_ok=False)
with Path('/data/ycfeng/tmp/d29_trace_serialization_repair/wave-candidate02/op_traces.jsonl').open('rb') as stream:
    stream.readline()
    reference = b''.join(stream.readline() for _ in range(208))
assert len(reference) == 2889655
assert hashlib.sha256(reference).hexdigest() == '18d320438ee1fb5f077b12aaa83da8055b47d8cf256d88aedcca159a5d7b819c'


def emit(path):
    current = trace.TraceStore(str(path), enabled=True)
    with patch.object(trace.time, 'time', return_value=0.0):
        current.initialize({'evidence': 'D29-IMMUTABLE-COST-FIELDS'})
    store._trace_store = current
    store._frontier_ep_wave_lane_ledger_rows = []
    begin = time.perf_counter()
    ep.record_ep_wave(store, plan, time=setup_namespace['captured']['meta']['wave_start_time_s'], replica_id=1, stage_id=0, cluster_type=setup_namespace['role'])
    seconds = time.perf_counter() - begin
    assert len(current._buffer) == 208
    return current, seconds


measurements = []
whole_reference = None
ledger_reference = None
for iteration in range(15):
    current, seconds = emit(output / f'iteration-{iteration}')
    # Ledger serialization/checks occur strictly outside emission timing.
    ledger = json.dumps(store._frontier_ep_wave_lane_ledger_rows).encode()
    if ledger_reference is None:
        ledger_reference = ledger
        (output / 'ledger.json').write_bytes(ledger)
    assert ledger == ledger_reference
    current.flush()
    whole = Path(current._filepath).read_bytes()
    assert whole.split(b'\n', 1)[1] == reference
    if whole_reference is None:
        whole_reference = whole
    assert whole == whole_reference
    measurements.append(seconds)

current, _ = emit(output / 'ownership')
def pending_bytes():
    return b''.join((json.dumps({field.name: getattr(event, field.name) for field in dataclasses.fields(event)
                                if getattr(event, field.name) is not None}) + '\n').encode() for event in current._buffer)
before = pending_bytes()
layer = plan.phase_times.lane_records[0].execution_time.layer_execution_times[0]
assert layer._finalized
source_native = layer._moe_or_mlp_time.operator_times.physical_frame['native_context']
snapshot = copy.deepcopy(source_native)
returned = layer.moe_or_mlp_time_component
returned.operator_times.physical_frame['native_context']['activation_layout']['shape'][0] = -1
assert source_native == snapshot and pending_bytes() == before
prior = source_native['activation_layout']['shape'][0]
source_native['activation_layout']['shape'][0] = -2
assert pending_bytes() == before
source_native['activation_layout']['shape'][0] = prior
native_event = next(event for event in current._buffer if 'native_context' in event.meta)
event_native = native_event.meta['native_context']
prior = event_native['activation_layout']['shape'][0]
event_native['activation_layout']['shape'][0] = -3
assert source_native == snapshot
event_native['activation_layout']['shape'][0] = prior
assert pending_bytes() == before
event = next(event for event in current._buffer if 'collective_plan' in event.meta)
row = next(row for row in store._frontier_ep_wave_lane_ledger_rows if row['ep_id'] == event.meta['ep_id'])
proof = row['phases'][event.meta['phase']]['collective_proofs'][event.name]
assert proof['plan'] is not event.meta['collective_plan']
assert proof['cost'] is not event.meta['collective_cost']
proof_snapshot = json.dumps(proof)
event.meta['collective_plan']['context']['mutation_probe'] = True
event.meta['collective_cost']['kernel_counts'] = (999,) * 16
assert json.dumps(proof) == proof_snapshot

costs = [cost for record in plan.phase_times.lane_records
         for cost in record.execution_time.layer_execution_times[0].communication_time_component.operator_times.costs.values()]
def immutable(value):
    return value is None or type(value) in (str, int, float, bool) or (type(value) is tuple and all(immutable(item) for item in value))
assert len(costs) == 64
assert all(type(cost) is CollectivePlanCost and all(immutable(getattr(cost, field.name)) for field in dataclasses.fields(cost)) for cost in costs)
cost_domain = sorted({(field.name, type(getattr(cost, field.name)).__name__) for cost in costs for field in dataclasses.fields(cost)})

contexts = []
original_meta = ep.compute_op_trace_meta
def observe(name, kind, context):
    if kind == 'COMM' and context.collective_plans is not None and name in context.collective_plans:
        contexts.append((name, kind, context))
    return original_meta(name, kind, context)
with patch.object(ep, 'compute_op_trace_meta', observe):
    current, _ = emit(output / 'contexts')
assert len(contexts) == 64
for name, kind, context in contexts:
    cost = context.collective_costs[name]
    changed = object.__new__(CollectivePlanCost)
    for field in dataclasses.fields(cost):
        object.__setattr__(changed, field.name, getattr(cost, field.name))
    object.__setattr__(changed, 'plan_key', 'invalid_identity')
    replacement = dataclasses.replace(context, collective_costs={**context.collective_costs, name: changed})
    try:
        op.compute_op_trace_meta(name, kind, replacement)
    except ValueError as error:
        assert str(error) == 'Trace collective plan and cost identities disagree'
    else:
        raise AssertionError('invalid plan key accepted')

profile_store = trace.TraceStore(str(output / 'profile'), enabled=True)
profile_store.initialize({})
store._trace_store = profile_store
store._frontier_ep_wave_lane_ledger_rows = []
profiler = cProfile.Profile()
profiler.enable()
ep.record_ep_wave(store, plan, time=setup_namespace['captured']['meta']['wave_start_time_s'], replica_id=1, stage_id=0, cluster_type=setup_namespace['role'])
profiler.disable()
profiler.dump_stats(str(output / 'emission.pstats'))
counts = {name: sum(values[1] for (path, _, symbol), values in pstats.Stats(profiler).stats.items()
                    if symbol == name and (name != 'deepcopy' or path.endswith('/copy.py')))
          for name in ('asdict', 'deepcopy', 'to_dict', 'cache_key', 'compute_op_trace_meta')}

# Retain the original per-row write boundary and its late-JSON prefix on error.
failure_store = trace.TraceStore(str(output / 'late-json-failure'), enabled=True)
failure_store.initialize({})
first = trace.TraceEvent(**json.loads(reference.splitlines()[0]))
invalid = trace.TraceEvent(**{**json.loads(reference.splitlines()[1]), 'meta': {'invalid': object()}})
failure_store.log_event(first)
failure_store.log_event(invalid)
try:
    failure_store.flush()
except TypeError:
    assert failure_store._buffer == [first, invalid]
else:
    raise AssertionError('serialization failure suppressed')
prefix = Path(failure_store._filepath).read_bytes().split(b'\n', 1)[1]
assert prefix == reference.splitlines(keepends=True)[0] and len(prefix) == 3902

result = {'task_id': 'D29-IMMUTABLE-COST-FIELDS', 'label': LABEL, 'source_revision_during_measurement': REV,
          'source_root': str(SOURCE), 'source_hashes': SOURCE_HASHES,
          'events': 208, 'collective_events_and_cost_objects': 64, 'bytes': len(reference),
          'sha256': hashlib.sha256(reference).hexdigest(), 'all_event_metadata_proofs_and_order_equal': True,
          'whole_jsonl_sha256': hashlib.sha256(whole_reference).hexdigest(),
          'ledger_bytes': len(ledger_reference), 'ledger_sha256': hashlib.sha256(ledger_reference).hexdigest(),
          'emission_seconds': measurements, 'median_emission_s': statistics.median(measurements),
          'emission_only_counters': counts, 'admitted_cost_field_types': cost_domain,
          'public_getter_and_pending_snapshot_mutations_isolated': True,
          'pending_meta_mutation_does_not_change_source': True,
          'fresh_trace_ledger_mappings_independent': True, 'wrong_plan_key_negatives': len(contexts),
          'wrong_plan_key_exact_error': 'Trace collective plan and cost identities disagree',
          'late_json_error_restores_buffer_and_prefix_bytes': len(prefix),
          'limits': 'Actual trace+prepared source; pricing/setup, ledger encoding, flush and verification excluded from emission timing. No diagnostic reuse control or unallocated source behavior.15 sequential CPU measurements, not whole CLI admission.'}
(output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key != 'emission_seconds'}, indent=2))
