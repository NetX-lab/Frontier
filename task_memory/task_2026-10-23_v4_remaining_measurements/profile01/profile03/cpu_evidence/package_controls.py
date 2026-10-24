"""CPU-only package integrity and actual worker pre-output command controls."""
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path('/data/ycfeng/Frontier/.worktrees/v4-native-profile03-run')
ROOT = Path(__file__).resolve().parent
TASK = REPO / 'task_memory/task_2026-10-23_v4_remaining_measurements'
INITIAL = json.loads((TASK / 'profile01/initial_delivery.json').read_text())
OLD = Path(INITIAL['package'])
NEW = Path('/data/ycfeng/Frontier/.worktrees/v4-native-profile03-run/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_profile_03')
NATIVE = Path('/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mapsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def files(path):
    return {str(p.relative_to(path)): sha(p) for p in path.rglob('*') if p.is_file()}


assert files(OLD) == INITIAL['package_hashes']
config = json.loads((NEW / 'config.json').read_text())
lock = json.loads((NEW / 'source_lock.json').read_text())
engine = config['engine']
assert config == json.loads((NEW / 'case/inputs/native_config.json').read_text())
assert engine == json.loads((NEW / 'engine.json').read_text())
assert engine == json.loads((NEW / 'case/inputs/engine.json').read_text())
runtime = {}
for name, target in config['harness_files'].items():
    content = (NEW / target).read_bytes()
    for revision in (lock['accepted_runtime_commit'], lock['harness_commit']):
        assert content == subprocess.check_output(['git', '-C', str(REPO), 'show', f'{revision}:{name}'])
    assert lock['committed_files'][target]['sha256'] == sha(NEW / target)
    runtime[name] = sha(NEW / target)
for name, expected in config['expected_source_hashes'].items():
    assert sha(NEW / 'overlay_model.py' if name == 'vllm/models/deepseek_v4/nvidia/model.py'
               else NATIVE / name) == expected
assert config['expected_source_hashes'] == json.loads((OLD / 'config.json').read_text())['expected_source_hashes']
for name, expected in config['tool_hashes'].items():
    assert sha(Path(name)) == expected
for name, expected in config['retained_files'].items():
    assert sha(NEW / name) == expected
for name, row in lock['retained_files'].items():
    assert sha(NEW / name) == row['sha256']
    if 'source_sha256' in row:
        assert sha(Path(row['source'])) == row['source_sha256']
    if 'staged_sha256' in row:
        assert row['staged_sha256'] == row['sha256']
for name in ('native_dummy_bootstrap.py', 'launch.py'):
    row = lock['retained_files'][name]
    assert row['source_sha256'] != row['staged_sha256']
    assert config['retained_files'][name] == config['tool_hashes'][str(NEW / name)] == row['staged_sha256']
for key in ('packaging_source', 'profile_plan'):
    row = lock[key]
    assert sha(REPO / row['path']) == row['sha256']
    assert (REPO / row['path']).read_bytes() == subprocess.check_output(
        ['git', '-C', str(REPO), 'show', f"{row['commit']}:{row['path']}"])
for row in lock['profile_inputs'].values():
    assert sha(REPO / row['path']) == row['sha256']
for name, expected in INITIAL['model_hashes'].items():
    assert sha(NEW / 'model' / name) == expected
inputs = json.loads((TASK / 'profile01/profile03/proposal_inputs.json').read_text())
freeze = json.loads((REPO / inputs['freeze_path']).read_text())
assert sha(REPO / inputs['freeze_path']) == inputs['freeze_sha256']
for name, row in inputs['files'].items():
    assert sha(Path(row['source_path'])) == row['sha256']
for name, expected in INITIAL['workload_hashes'].items():
    row = inputs['files']['trace/' + name]
    assert sha(NEW / 'trace' / name) == sha(NEW / 'case/inputs/trace' / name) == expected
    assert expected == row['sha256'] == freeze['original_files'][row['source_path']]
for name, expected in freeze['original_files'].items():
    assert sha(Path(name)) == expected
checkers = dict(freeze['dependencies'], **{freeze['checker_path']: freeze['checker_sha256']})
for name, expected in checkers.items():
    assert sha(Path(name)) == expected
for key, value in inputs['preserved_resource_template'].items():
    assert config[key] == value
original_engine = json.loads((OLD / 'engine.json').read_text())
assert {k: v for k, v in original_engine.items() if k != 'profiler_config'} == {
    k: v for k, v in engine.items() if k != 'profiler_config'}
assert {k: v for k, v in original_engine['profiler_config'].items() if k != 'torch_profiler_dir'} == {
    k: v for k, v in engine['profiler_config'].items() if k != 'torch_profiler_dir'}
for name in ('config.json', 'source_lock.json', 'evidence_readiness.json', 'profile_plan.json'):
    value = json.loads((NEW / name).read_text())
    assert value['diagnostic_only'] and not value['e2e_reference_admission']
assert 'all_four_metrics_required' not in (NEW / 'case/manifest.yaml').read_text()
assert config['run_tag'] == 'dsv4-h800-p0-profile-20261024-03'
assert len(files(NEW)) == 45
for package in (OLD, NEW):
    assert not (package / 'manifest.json').exists()
    assert not (package / 'case/runs').exists()
    assert not list(package.rglob('__pycache__'))
matrix = TASK / INITIAL['matrix']['binding_path'].split('v4_remaining_measurements/')[1]
assert sha(matrix) == INITIAL['matrix']['binding_sha256']

# Run the generated launcher with a test-only manifest, stopping before the body.
# Then enter the real worker until its own imports complete, before cloud path use.
# No test-added sys.path or preload is used at either entry.
class BeforeManifest(Exception):
    pass


class BeforeOutput(Exception):
    pass


def launcher_trace(frame, event, arg):
    if event == 'call' and frame.f_code.co_name == 'write_run_manifest':
        assert 'pd_replay' not in sys.modules
        raise BeforeManifest
    return launcher_trace


os.environ.update(STEPMIND_BACKEND='rjob', AUTHORIZATION='CPU_ONLY_SENTINEL_NEVER_SUBMIT')
launch = ROOT / 'new_launcher_green/launch.py'
sys.argv = [str(launch)]
sys.settrace(launcher_trace)
try:
    runpy.run_path(str(launch), run_name='__main__')
except BeforeManifest:
    pass
else:
    raise AssertionError('Actual launcher sentinel missed')
sys.settrace(None)
worker = sys.modules['native_pd_worker']
output_line = next(node.lineno for node in ast.walk(ast.parse(Path(worker.__file__).read_text()))
                   if isinstance(node, ast.Assign) and ast.unparse(node).startswith("root = Path(config['output_dir'])"))


def worker_trace(frame, event, arg):
    if frame.f_code.co_name == 'run' and event == 'line' and frame.f_lineno == output_line:
        raise BeforeOutput
    return worker_trace


sys.settrace(worker_trace)
try:
    worker.run(config)
except BeforeOutput:
    pass
else:
    raise AssertionError('Worker crossed output boundary')
sys.settrace(None)
replay = sys.modules['pd_replay']
driver = sys.modules['vllm_replay']
import native_serve_profile

peers = [dict(host=f'cpu-node-{i}', ip=f'192.0.2.{i+1}') for i in range(4)]
observed = []
for peer in peers:
    plan = worker.node_plan(config, peers[::-1], peer['host'], [str(i) for i in range(8)])
    role, lane, planned_engine = plan['role'], plan['lane'], plan['engine']
    assert config['role_profiler_max_iterations'] == {'prefill': 512, 'decode': 64}
    assert config['engine']['profiler_config']['max_iterations'] == 64
    env, changed = replay.instance_env('clean', planned_engine, NEW, {'NCCL_DEBUG': 'WARN'},
                                       '0,1,2,3,4,5,6,7', observer=None, role=role)
    command = driver.server_command(planned_engine, NEW / 'model', config['http_ports'][role])
    expected = engine['profiler_config'] | {'torch_profiler_dir':
        str(Path(engine['profiler_config']['torch_profiler_dir']) / role / f'dp{lane}'),
        'max_iterations': config['role_profiler_max_iterations'][role]}
    assert json.loads(command[command.index('--profiler-config') + 1]) == expected
    assert command[command.index('--worker-extension-cls') + 1] == 'native_serve_profile.NativeServeProfile'
    assert ('--headless' in command) == (lane == 1)
    assert ('--data-parallel-start-rank' in command) == (lane == 1)
    assert planned_engine['data_parallel_size_local'] == 1
    assert env['VLLM_MOE_ROUTING_SIMULATION_STRATEGY'] == 'balanced_topk'
    assert env['NCCL_DEBUG'] == 'WARN'
    assert env['PYTHONPATH'].split(os.pathsep)[0] == str(NEW / 'tests/comparison/calibration')
    statement = next(node for node in ast.walk(ast.parse((NEW / 'native_pd_worker.py').read_text()))
                     if isinstance(node, ast.If) and ast.unparse(node.test) == "'serving_profile_limits' in config")
    exec(compile(ast.Module(body=[statement], type_ignores=[]), 'actual_staged_limit_forwarding', 'exec'),
         dict(config=config, env=env, changed=changed, json=json))
    with patch.dict(os.environ, {'FRONTIER_NATIVE_PROFILE_LIMITS': env['FRONTIER_NATIVE_PROFILE_LIMITS']}):
        assert native_serve_profile.ServingProfile(SimpleNamespace()).limits == {'records': 600000, 'bytes': 268435456}
    observed.append(dict(role=role, dp=lane, headless=lane == 1, profiler=expected, command=command,
                         balanced_routing=env['VLLM_MOE_ROUTING_SIMULATION_STRATEGY']))
manifest, formal, warmup = replay.canonical_workload(NEW / 'trace')
assert (len(formal), len(warmup), manifest['repetitions']['clean_repetitions']) == (128, 16, 3)
native_file_map = config['expected_source_hashes']
maps = dict(native_source_map_sha256=mapsha(native_file_map), runtime_map_sha256=mapsha(runtime),
            tool_map_sha256=mapsha(config['tool_hashes']), model_map_sha256=mapsha(config['model_hashes']),
            workload_map_sha256=mapsha(INITIAL['workload_hashes']), package_map_sha256=mapsha(files(NEW)))
counts = dict(package_files=len(files(NEW)), package_bytes=sum(p.stat().st_size for p in NEW.rglob('*') if p.is_file()),
              runtime_modules=len(runtime), native_source_pins=len(native_file_map), tool_hashes=len(config['tool_hashes']),
              retained_files=len(config['retained_files']), model_files=len(config['model_hashes']),
              workload_files=len(INITIAL['workload_hashes']), original_frozen_files=len(freeze['original_files']),
              frozen_checkers=len(checkers), role_dp_paths=len(observed))
receipt = dict(status='PASS', counts=counts, hash_maps=maps, cpu_paths=observed, package_hashes=files(NEW),
               retained_provenance={name: lock['retained_files'][name] for name in ('launch.py', 'native_dummy_bootstrap.py')},
               no_test_path_injection=True, original_package_unchanged=files(OLD) == INITIAL['package_hashes'],
               actual_entry_worker_imports=True, worker_stopped_before_output_path=True, submitted=False)
(ROOT / 'package_controls.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(status='PACKAGE_HASHES_ACTUAL_ENTRY_COMMANDS_PASS', counts=counts, hash_maps=maps)))
