"""One-variable real-wave controls using the RED driver's identical setup."""
import ast
import sys
from pathlib import Path

label = sys.argv[1]
assert label in ('current-a3', 'key-a', 'current-b', 'key-b')
driver = Path(__file__).resolve().with_name('probe.py')
prefix = []
for node in ast.parse(driver.read_text()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'timings' for t in node.targets):
        break
    if isinstance(node, ast.Assert) and ast.unparse(node.test) == "LABEL == 'red'":
        continue
    prefix.append(node)
exec(compile(ast.Module(body=prefix, type_ignores=[]), str(driver), 'exec'), globals())

import copy
from dataclasses import fields, FrozenInstanceError, replace
from unittest.mock import patch
from frontier.cc_backend.collective_plan import CollectiveTensor, CollectivePlanCost
from frontier.metrics.metrics_store import MetricsStore
from frontier.metrics.trace_store import TraceStore
import frontier.metrics.trace_store as trace_module
from frontier.metrics.ep_wave_metrics import record_ep_wave

def serialized_key_once(plan):
    """Cache only a derived string on the immutable plan, not a result/proof."""
    if '_d29_derived_key' not in plan.__dict__:
        object.__setattr__(plan, '_d29_derived_key', real_key(plan))
    return plan._d29_derived_key

CollectivePlan.cache_key = serialized_key_once if LABEL.startswith('key-') else real_key
timings = []
previous_proofs = None
previous_plans = None
plan_objects = {}
payloads = []
for index in range(13):
    read_counts.clear()
    start = time.perf_counter()
    exec(wave_code, namespace)
    seconds = time.perf_counter() - start
    timings.append({'index': index, 'pricing_s': seconds, **read_counts})
    phases = namespace['phases']
    if index == 0:
        first_phases = phases
    payloads.append(payload(phases))
    proofs = []
    plans = []
    for record in phases.lane_records:
        layer = record.execution_time.layer_execution_times[0]
        moe = layer.moe_or_mlp_time_component.operator_times
        comm = layer.communication_time_component.operator_times
        proofs.extend(moe.physical_frame['operator_costs'].values())
        plans.extend(comm.plans.values())
        for plan in comm.plans.values():
            plan_objects[id(plan)] = plan
    retained_plans = [entry[0] for entry in namespace['predictor']._step5_collective_receipts[4].values()]
    if previous_proofs is not None:
        assert len(proofs) == len(previous_proofs) == 144
        assert all(new is not old for new, old in zip(proofs, previous_proofs))
        assert all(new is old for new, old in zip(retained_plans, previous_plans))
    previous_proofs, previous_plans = proofs, retained_plans
    plan_objects = {id(plan): plan for plan in retained_plans}
assert len(plan_objects) == 8
assert len(set(payloads)) == 1
assert hashlib.sha256(payloads[0]).hexdigest() == '3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83'
(OUT / 'pricing_payload.json').write_bytes(payloads[0])

# Narrow warm pricing profile: excluded from the unprofiled measurements above.
constructors = Counter()
original_init = {}
for cls in (CollectivePlan, CollectiveTensor, CollectivePlanCost):
    original_init[cls] = cls.__init__
    def wrapper(self, *args, _cls=cls, **kwargs):
        constructors[_cls.__name__] += 1
        return original_init[_cls](self, *args, **kwargs)
    cls.__init__ = wrapper
prof = cProfile.Profile()
prof.enable()
exec(wave_code, namespace)
prof.disable()
for cls, initializer in original_init.items():
    cls.__init__ = initializer
prof.dump_stats(str(OUT / 'warm.prof'))
stats = pstats.Stats(prof)
profile = [{'file': file, 'line': line, 'function': name, 'primitive': cc,
            'calls': nc, 'self_s': tt, 'cumulative_s': ct}
           for (file, line, name), (cc, nc, tt, ct, callers) in stats.stats.items()
           if file.startswith(str(SOURCE / 'frontier')) or
              ((name in ('dumps', 'loads', 'asdict', 'predict')) and
               ('json/' in file or 'dataclasses.py' in file or 'sklearn/' in file)) or
              name == 'serialized_key_once']
profile.sort(key=lambda row: row['cumulative_s'], reverse=True)
counts = {name: sum(row['calls'] for row in profile if row['function'] == name)
          for name in ('cache_key', 'serialized_key_once', 'to_dict', 'dumps', 'loads', 'asdict',
                       'predict', '_predict_step5_ffn_operator', '_step5_position_queries',
                       '_step5_model_points', '_step5_plan_candidates', '_lookup_step5_ffn_operator',
                       '_native_measurement_binding', '_measurement_identity', 'predict_collective_plan')}

# Black-box output acceptance only: no emission profiling or encoder controls.
namespace['phases'] = first_phases  # Preserve the first wave's generated lane/request IDs.
timing_node = next(node for node in tree.body if node.lineno == 138)
plan_node = next(node for node in tree.body if node.lineno == 139)
exec(compile(ast.Module(body=[timing_node, plan_node], type_ignores=[]), str(OUT / 'setup.py'), 'exec'), namespace)
store = MetricsStore.__new__(MetricsStore)
store._config = namespace['config'].metrics_config
store._cluster_configs = namespace['clusters']
store._frontier_ep_wave_lane_ledger_rows = []
store._trace_store = TraceStore(str(OUT / 'output-equality'), enabled=True)
with patch.object(trace_module.time, 'time', return_value=0.0):
    store._trace_store.initialize({'evidence': 'D29-IMMUTABLE-COST-FIELDS'})
record_ep_wave(store, namespace['plan'], time=namespace['captured']['meta']['wave_start_time_s'],
               replica_id=1, stage_id=0, cluster_type=namespace['role'])
assert len(store._trace_store._buffer) == 208
ledger = json.dumps(store._frontier_ep_wave_lane_ledger_rows).encode()
store._trace_store.flush()
events = Path(store._trace_store._filepath).read_bytes().split(b'\n', 1)[1]
gold = Path('/data/ycfeng/tmp/d29_artifact_bytes_repair/verified03/candidate')
gold_events = (gold / 'iteration-0/op_traces.jsonl').read_bytes().split(b'\n', 1)[1]
assert events == gold_events
assert ledger == (gold / 'ledger.json').read_bytes()
(OUT / 'ledger.json').write_bytes(ledger)

# Reachable replacements get new keys; public dictionaries remain detached.
mutation_checks = []
sample = next(iter(plan_objects.values()))
key_before = CollectivePlan.cache_key(sample)
public = sample.to_dict()
public['context']['d29_mutation_probe'] = True
public['buffers']['d29_mutation_probe'] = True
assert CollectivePlan.cache_key(sample) == key_before
mutation_checks.append('public to_dict mutation detached from immutable plan/key')
try:
    sample.context_json = '{}'
except FrozenInstanceError:
    mutation_checks.append('normal plan field mutation rejected')
else:
    raise AssertionError('immutable plan field accepted mutation')
changed_context = json.loads(sample.context_json)
changed_context['d29_mutation_probe'] = True
replacement = replace(sample, context_json=json.dumps(changed_context))
assert CollectivePlan.cache_key(replacement) == real_key(replacement) != key_before
mutation_checks.append('dataclasses.replace creates distinct full key, never inherits derived key')
try:
    namespace['predictor']._cc_backend.predict_collective_plan(replacement)
except KeyError as error:
    assert error.args == (real_key(replacement),)
    mutation_checks.append('changed full plan preserves exact missing-key error')
else:
    raise AssertionError('changed full plan accepted')

Path.read_bytes = real_read_bytes
assert before == state(fixture)
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE).strip()
result = {'task_id': 'D29-IMMUTABLE-PLAN-PREPARATION', 'label': LABEL, 'source': str(SOURCE),
          'revision': REV, 'adapter': 'ONE SINGLE-ROLE DIAGNOSTIC FIXED-FILE-BYTE ADAPTER, not P/D/run snapshot',
          'timings': timings, 'cold_s': timings[0]['pricing_s'],
          'warm_median_s': statistics.median(row['pricing_s'] for row in timings[1:]),
          'warm_min_s': min(row['pricing_s'] for row in timings[1:]),
          'warm_max_s': max(row['pricing_s'] for row in timings[1:]),
          'constructors_warm': {cls.__name__: constructors[cls.__name__] for cls in original_init},
          'profile_counts': counts, 'profile': profile, 'plan_objects': len(plan_objects),
          'fresh_operator_proofs': 144, 'payload_sha256': hashlib.sha256(payloads[0]).hexdigest(),
          'events': 208, 'event_bytes': len(events), 'event_sha256': hashlib.sha256(events).hexdigest(),
          'ledger_bytes': len(ledger), 'ledger_sha256': hashlib.sha256(ledger).hexdigest(),
          'mutation_checks': mutation_checks, 'fixture_unchanged': True,
          'excluded': 'setup/import/model loading; payload/output equality; cProfile/counters; mutation checks; no trace encoding investigation'}
(OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key not in ('profile', 'timings')}, indent=2))
