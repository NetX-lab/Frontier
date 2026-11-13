"""Actual full-wave snapshot verification with delegated-read observation only."""
import ast
from collections import Counter
import cProfile
import dataclasses
import hashlib
import json
from pathlib import Path
import pstats
import statistics
import subprocess
import sys
import time
from fixture_wiring import setup, bind_production_owner

ROOT = Path(__file__).resolve().parent
LABEL = sys.argv[1]
assert LABEL in ('baseline', 'candidate')
frozen = json.loads((ROOT / 'freeze.json').read_text())
SOURCE = Path(frozen['sources'][LABEL]['root'])
REV = frozen['sources'][LABEL]['head']
SOURCE_HASHES = frozen['sources'][LABEL]['hashes']
for path,expected in SOURCE_HASHES.items():
    assert hashlib.sha256((SOURCE/path).read_bytes()).hexdigest() == expected, path
reads = []
phase = 'setup'
real_bytes = Path.read_bytes
real_text = Path.read_text

def observe_bytes(path, *args, **kwargs):
    if path.name == 'collective_local.json':
        reads.append({'phase': phase, 'method': 'read_bytes', 'path': str(path),
                      'caller': sys._getframe(1).f_code.co_filename, 'success': False})
        item = reads[-1]
        value = real_bytes(path, *args, **kwargs)
        item['success'] = True
        return value
    return real_bytes(path, *args, **kwargs)

def observe_text(path, *args, **kwargs):
    if path.name == 'collective_local.json':
        reads.append({'phase': phase, 'method': 'read_text', 'path': str(path),
                      'caller': sys._getframe(1).f_code.co_filename, 'success': False})
        item = reads[-1]
        value = real_text(path, *args, **kwargs)
        item['success'] = True
        return value
    return real_text(path, *args, **kwargs)

Path.read_bytes, Path.read_text = observe_bytes, observe_text
setup_namespace, wave_code, setup_tree, wave_node, setup_path = setup(frozen, LABEL, ROOT)
assert not reads
phase = 'binding'
start = time.perf_counter()
binding_details = None
if LABEL == 'candidate':
    p_backend, d_backend, bound_raw, collective_path = bind_production_owner(setup_namespace, ROOT, LABEL)
    assert hashlib.sha256(bound_raw).hexdigest() == frozen['collective_sha256']
    assert len(reads) == 1 and reads[0]['success']
    binding_details = {'same_owner': True, 'both_role_backends_real': True,
        'between_role_file_changed_to_invalid_json': True, 'bound_sha256': hashlib.sha256(bound_raw).hexdigest()}
binding_s = time.perf_counter() - start

def payload(phases):
    rows = []
    for record in phases.lane_records:
        layer = record.execution_time.layer_execution_times[0]
        moe = layer.moe_or_mlp_time_component.operator_times
        comm = layer.communication_time_component.operator_times
        rows.append({'ep': record.ep_id, 'moe': moe.op_times, 'comm': comm.op_times,
            'plans': {name: value.cache_key() for name,value in comm.plans.items()},
            'proofs': moe.physical_frame['operator_costs']})
    assert len(rows) == 16 and sum(len(row['moe']) + len(row['comm']) for row in rows) == 208
    assert sum(len(row['proofs']) for row in rows) == 144
    return json.dumps(rows, sort_keys=True).encode()

samples = []
payloads = []
for index in range(5):
    phase = 'cold' if index == 0 else ('profile' if index == 4 else 'warm')
    before_reads = len(reads)
    profiler = cProfile.Profile() if index == 4 else None
    if profiler: profiler.enable()
    begin = time.perf_counter()
    exec(wave_code, setup_namespace)
    seconds = time.perf_counter() - begin
    if profiler:
        profiler.disable()
        profiler.dump_stats(str(ROOT / LABEL / 'pricing.prof'))
    current_reads = len(reads) - before_reads
    assert current_reads == (0 if LABEL == 'candidate' else 80)
    samples.append({'phase': phase, 'pricing_s': seconds, 'physical_collective_reads': current_reads,
                    'profiled': bool(profiler)})
    payloads.append(payload(setup_namespace['phases']))
assert len(set(payloads)) == 1
assert hashlib.sha256(payloads[0]).hexdigest() == '3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83'
assert len(reads) == (1 if LABEL == 'candidate' else 400)
receipts = setup_namespace['predictor']._step5_collective_receipts
assert len(receipts) == (4 if LABEL == 'candidate' else 5)
(ROOT / LABEL / 'pricing_payload.json').write_bytes(payloads[0])
stats = pstats.Stats(str(ROOT / LABEL / 'pricing.prof'))
count = lambda symbol: sum(values[1] for (_,_,name),values in stats.stats.items() if name == symbol)
counts = {symbol: count(symbol) for symbol in ('_get_on_demand_prediction', '_predict_step5_ffn_operator',
    'predict_collective_plan', '_step5_position_queries', 'cache_key', 'loads', 'dumps')}
assert counts['_get_on_demand_prediction'] == 288 and counts['_predict_step5_ffn_operator'] == 144
assert counts['predict_collective_plan'] == 64 and counts['_step5_position_queries'] == 16
pricing_counts = counts.copy()
phase = 'output_verification'
tail = [node for node in setup_tree.body if node.lineno > wave_node.lineno]
exec(compile(ast.Module(body=tail,type_ignores=[]),str(setup_path),'exec'),setup_namespace)
# Execute every accepted cost/output/mutation/error assertion unchanged.
body = Path(frozen['protected_body'])
assert hashlib.sha256(body.read_bytes()).hexdigest() == frozen['protected_body_sha256']
run_root = ROOT
ROOT = run_root / LABEL / 'accepted-output'
ROOT.mkdir(exist_ok=False)
exec(compile(body.read_bytes(),str(body),'exec'),globals())
output_result = result
ROOT = run_root

lifetimes = []
if LABEL == 'candidate':
    phase = 'new_run_invalid_json'
    from frontier.cc_backend.collective_profile import CollectiveProfileInputs
    from frontier.execution_time_predictor.shared_prediction_model_manager import ExecutionTimePredictionModelManager
    def new_backend():
        backend = setup_namespace['Cluster'](setup_namespace['clusters'][setup_namespace['role']],
            setup_namespace['config'].metrics_config,setup_namespace['config'].request_generator_config).cc_backend
        backend._config = dataclasses.replace(backend._config,profiling_data_dir=str(ROOT / LABEL / 'collective-fixture'))
        return backend
    new_manager = ExecutionTimePredictionModelManager.__new__(ExecutionTimePredictionModelManager)
    new_manager._collective_profile_inputs = CollectiveProfileInputs()
    new = new_backend()
    failed_start = len(reads)
    for _ in range(2):
        try: new.bind_collective_profile_inputs(new_manager.collective_profile_inputs)
        except json.JSONDecodeError as error:
            lifetimes.append({'case': 'fresh owner repeated invalid JSON', 'error': type(error).__name__, 'message': str(error)})
        else: raise AssertionError('new-run invalid JSON accepted')
    assert len(reads) - failed_start == 2
    collective_path.write_bytes(bound_raw)
    phase = 'new_run_valid_retry'
    new.bind_collective_profile_inputs(new_manager.collective_profile_inputs)
    assert new.collective_profile_bytes() == bound_raw
    assert len(reads) - failed_start == 3
    lifetimes.append({'case': 'failed owner later successful retry', 'bound_sha256': hashlib.sha256(new.collective_profile_bytes()).hexdigest()})
    collective_path.write_bytes(b'\xff')
    phase = 'new_run_invalid_utf8'
    utf8 = new_backend()
    utf8_owner = CollectiveProfileInputs()
    utf8_start = len(reads)
    for _ in range(2):
        try: utf8.bind_collective_profile_inputs(utf8_owner)
        except UnicodeDecodeError as error:
            lifetimes.append({'case': 'fresh owner repeated invalid UTF8', 'error': type(error).__name__})
        else: raise AssertionError('invalid UTF8 accepted')
    assert len(reads) - utf8_start == 2
    assert p_backend.collective_profile_bytes() == d_backend.collective_profile_bytes() == bound_raw
    assert new.collective_profile_bytes() == bound_raw
    collective_path.write_bytes(bound_raw)
    phase = 'standalone_new_run'
    standalone = new_backend()
    standalone_start = len(reads)
    first_plan = next(iter(setup_namespace['phases'].lane_records[0].execution_time.layer_execution_times[0].communication_time_component.operator_times.plans.values()))
    standalone_cost = standalone.predict_collective_plan(first_plan)
    standalone.predict_collective_plan(first_plan)
    assert len(reads) - standalone_start == 1
    lifetimes.append({'case':'standalone actual backend retains call shape','duration_ms':standalone_cost.duration_ms,'reads':1})

Path.read_bytes, Path.read_text = real_bytes, real_text
fixture = Path(frozen['fixture'])
metadata = {str(path.relative_to(fixture)): [path.stat().st_size,path.stat().st_mtime_ns]
            for path in fixture.rglob('*') if path.is_file()}
assert metadata == frozen['fixture_metadata']
assert hashlib.sha256(Path(frozen['collective_original']).read_bytes()).hexdigest() == frozen['collective_sha256']
for path,expected in SOURCE_HASHES.items():
    assert hashlib.sha256((SOURCE/path).read_bytes()).hexdigest() == expected, path
for path,expected in frozen['input_hashes'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected,path
report = {'task_id':'D29-RUN-SNAPSHOT-WAVE','session_id':frozen['session_id'], 'label':LABEL,
    'source':str(SOURCE),'source_hashes_before_and_after':SOURCE_HASHES,'source_unchanged':True,
    'protected_setup_sha256':frozen['setup_sha256'],'derived_setup_sha256':hashlib.sha256(setup_path.read_bytes()).hexdigest(),
    'fixture_wiring_sha256':hashlib.sha256((ROOT/'fixture_wiring.py').read_bytes()).hexdigest(),
    'wave_ast_sha256':hashlib.sha256(ast.dump(wave_node,include_attributes=False).encode()).hexdigest(),
    'protected_body_sha256':frozen['protected_body_sha256'], 'binding_s':binding_s,'binding':binding_details,
    'samples':samples,'warm_median_s':statistics.median(sample['pricing_s'] for sample in samples if sample['phase']=='warm'),
    'operation_counts':pricing_counts,'collective_receipt_tuple_length':len(receipts),
    'main_run_successful_file_reads':sum(item['success'] for item in reads if item['phase'] in ('setup','binding','cold','warm','profile')),
    'reads':reads,'payload_sha256':hashlib.sha256(payloads[0]).hexdigest(),'output':output_result,
    'new_run_lifetimes':lifetimes,'fixture_metadata_unchanged':True,'loaded_models':setup_namespace['loaded'],
    'limits':'Reconstructed cached manager/predictor, explicit real owner and P/D backend binding; no full Simulator/CLI/native qualification. Constructor coverage is separate82PASS evidence. No diagnostic byte adapter or numeric/result/proof cache.'}
(ROOT / LABEL / 'wave_result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({key:report[key] for key in ('task_id','label','binding_s','samples','operation_counts','main_run_successful_file_reads','payload_sha256','fixture_metadata_unchanged')},indent=2))
