"""DS-V4 staged reader admission; CPU artifacts do not qualify a native run."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[2]
TASK = REPO / 'task_memory/task_2026-10-22_v4_native_binding'
FROZEN = REPO / 'task_memory/task_2026-10-21_native_cold_start/check_groundtruth_run_successor.py'
PREVIOUS = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/'
    'task_2026-10-22_v4_native_binding/check_groundtruth_run_successor.py')
R2 = PREVIOUS.with_name('check_groundtruth_run_successor_r2.py')
CHECKER = Path(os.environ.get('NATIVE_READER_BINDING_CHECKER', R2))
PACKAGE = Path('/data/ycfeng/tmp/glm53_dsv4/v4_successor04_actual_binding')
CHECKS = {'checkout', 'inputs_unchanged', 'harness_pinned_and_client_rows', 'requests_and_idle_lanes',
          'native_workers', 'native_lifecycle', 'rank_runtime_graph', 'terminal_status'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def load(path):
    spec = importlib.util.spec_from_file_location('reader_binding_check', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def v4_package(tmp_path):
    """Copy the retained candidate04 full-checker control, never its sealed package."""
    return copy_cpu_package(PACKAGE, tmp_path)


def copy_cpu_package(source, tmp_path):
    if not source.is_dir():
        pytest.skip('Retained DS-V4 CPU package is unavailable')
    stage = tmp_path / 'v4'
    shutil.copytree(source, stage)
    for path in stage.rglob('*.json'):
        path.write_text(path.read_text().replace(str(source), str(stage)))
    config = json.loads((stage / 'config.json').read_text())
    # The actual reviewed bootstrap overwrites inherited PYTHONPATH with staged_dir.
    for name in config['tool_hashes']:
        config['tool_hashes'][name] = digest(Path(name))
    for name in config['retained_files']:
        config['retained_files'][name] = digest(stage / name)
    write(stage / 'config.json', config)
    case = stage / 'case'
    write(case / 'inputs/native_config.json', config)
    lock = json.loads((stage / 'source_lock.json').read_text())
    lock['runtime_overlay'] = json.loads((stage / 'overlay.json').read_text())
    for name, row in lock['retained_files'].items():
        row['sha256'] = digest(stage / name)
    write(stage / 'source_lock.json', lock)
    seal = json.loads((stage / 'manifest.json').read_text())
    seal['files'] = {name: digest(stage / name) for name in seal['files']}
    write(stage / 'manifest.json', seal)
    run = case / 'runs/groundtruth_clean' / config['run_tag']
    manifest = json.loads((run / 'run_manifest.json').read_text())
    manifest['inputs_sha256'] = {name: digest(case / name) for name in manifest['inputs_sha256']}
    write(run / 'run_manifest.json', manifest)
    manifest_hash = digest(run / 'run_manifest.json')
    for path in (run / 'nodes').glob('*/result.json'):
        node = json.loads(path.read_text())
        node['run_manifest_sha256'] = manifest_hash
        node['server_env_set']['PYTHONPATH'] = config['harness_dir'] + ':' + str(stage)
        write(path, node)
        ready_path = path.with_name('server_ready.json')
        if ready_path.exists():
            ready = json.loads(ready_path.read_text())
            ready['run_manifest_sha256'] = manifest_hash
            write(ready_path, ready)
        # Preserve the real drain-before-stop ordering after copying.
        for name in ('ready.json', 'drained.json', 'native_stopped.json'):
            path.with_name(name).touch()
    repetitions_path = run / 'run/canonical_repetitions.json'
    repetitions = json.loads(repetitions_path.read_text())
    repetitions['idle_boundaries'][0]['fresh_services']['run_manifest_sha256'] = manifest_hash
    for ready in repetitions['idle_boundaries'][0]['fresh_services']['services'].values():
        ready['run_manifest_sha256'] = manifest_hash
    write(repetitions_path, repetitions)
    summary_path = run / 'run/replay_summary.json'
    summary = json.loads(summary_path.read_text())
    summary['canonical_repetitions'] = repetitions
    write(summary_path, summary)
    (run / 'run/client_requests.jsonl').touch()
    (run / 'replay_finished.json').touch()
    return stage, case, run, config, manifest


def check_package(checker, fixture, label):
    stage, case, run, config, manifest = fixture
    result = checker.check(case, config['run_tag'], 'clean')
    write(stage.parent / f'{label}_predicates.json', result)
    assert set(result['checks']) == CHECKS
    assert result['status'] == 'FAIL'
    assert result['checks']['rank_runtime_graph'].startswith('FAIL: UNVERIFIED:')
    assert result['checks']['terminal_status'].startswith('FAIL: UNVERIFIED: FileNotFoundError:')
    return result


def test_v4_full_checker_binds_staged_readers_and_bootstrap_inheritance(v4_package):
    old = check_package(load(FROZEN), v4_package, 'frozen_red')
    assert old['checks']['harness_pinned_and_client_rows'].startswith('FAIL: Actual clean pinned')
    result = check_package(load(CHECKER), v4_package, 'successor_control')
    assert old['checks']['rank_runtime_graph'] == result['checks']['rank_runtime_graph'] == (
        "FAIL: UNVERIFIED: KeyError: 'vllm/compilation/breakable_cudagraph.py'")
    assert result['checks']['harness_pinned_and_client_rows'].startswith('PASS: Actual clean pinned')
    assert {name for name, value in result['checks'].items() if value.startswith('FAIL')} == {
        'rank_runtime_graph', 'terminal_status'}, result['checks']


@pytest.mark.parametrize('fault', ['wrong_path', 'extra_entry', 'missing_inheritance', 'reordered',
    'checkout_path', 'staged_bytes', 'checkout_digest', 'input_digest', 'harness_dir', 'stage_escape',
    'bootstrap_env', 'overlay_stage', 'reader_alias', 'dependency_alias'])
def test_v4_full_checker_rejects_unbound_reader_environment(v4_package, fault):
    stage, case, run, config, manifest = v4_package
    result_path = run / 'nodes/cpu-v4-0/result.json'
    node = json.loads(result_path.read_text())
    module = 'tests/comparison/calibration/pd_replay.py'
    if fault in ('wrong_path', 'extra_entry', 'missing_inheritance', 'reordered', 'checkout_path'):
        path = node['server_env_set']['PYTHONPATH']
        node['server_env_set']['PYTHONPATH'] = {
            'wrong_path': str(stage.parent / 'unverified/tests/comparison/calibration') + ':' + str(stage),
            'extra_entry': path + ':/unverified',
            'missing_inheritance': config['harness_dir'],
            'reordered': str(stage) + ':' + config['harness_dir'],
            'checkout_path': str(Path(manifest['harness_worktree']) / 'tests/comparison/calibration')
                             + ':' + str(stage),
        }[fault]
        write(result_path, node)
    elif fault == 'staged_bytes':
        path = stage / config['harness_files'][module]
        path.write_text(path.read_text() + '\n# Altered CPU fixture.\n')
    elif fault == 'checkout_digest':
        manifest['harness_sha256'][module] = '0' * 64
    elif fault == 'input_digest':
        manifest['inputs_sha256'][str(stage / config['harness_files'][module])] = '0' * 64
    elif fault == 'harness_dir':
        config['harness_dir'] = str(stage.parent / 'unverified')
    elif fault == 'stage_escape':
        config['harness_files'][module] = str(Path(manifest['harness_worktree']) / module)
    elif fault in ('reader_alias', 'dependency_alias'):
        target = module if fault == 'reader_alias' else 'tests/comparison/calibration/client_metrics.py'
        path = stage / config['harness_files'][target]
        alias = path.with_name('verified_copy.py')
        shutil.copyfile(path, alias)
        path.write_text(path.read_text() + '\n# Actual imported file is no longer reviewed.\n')
        config['harness_files'][target] = str(alias.relative_to(stage))
        manifest['inputs_sha256'][str(alias)] = digest(alias)
        manifest['inputs_sha256'][str(path)] = digest(path)
    elif fault == 'bootstrap_env':
        path = Path(config['observer_bootstrap'])
        path.write_text(path.read_text().replace('PYTHONPATH=str(staged)', "PYTHONPATH='/unverified'"))
        # Consistent local hashes do not authorize a new inheritance producer.
        config['retained_files'][path.name] = digest(path)
        manifest['inputs_sha256'][str(path)] = digest(path)
    elif fault == 'overlay_stage':
        path = Path(config['overlay_manifest'])
        overlay = json.loads(path.read_text())
        overlay['staged_dir'] = str(stage.parent / 'unverified')
        write(path, overlay)
        config['retained_files'][path.name] = digest(path)
        manifest['inputs_sha256'][str(path)] = digest(path)
    if fault in ('harness_dir', 'stage_escape', 'bootstrap_env', 'overlay_stage',
                 'reader_alias', 'dependency_alias'):
        write(case / manifest['execution_config'], config)
        manifest['inputs_sha256'][manifest['execution_config']] = digest(case / manifest['execution_config'])
    # Keep the original prelaunch mtime; negatives must not become stale-output tests.
    stamp = (run / 'run_manifest.json').stat()
    write(run / 'run_manifest.json', manifest)
    os.utime(run / 'run_manifest.json', ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    result = check_package(load(CHECKER), v4_package, fault)
    assert result['checks']['harness_pinned_and_client_rows'].startswith('FAIL'), result['checks']


def test_v4_no_extension_preserves_existing_strict_reader_behavior(v4_package):
    stage, case, run, config, manifest = v4_package
    config['engine'].pop('worker_extension_cls')
    write(case / manifest['engine_config'], config['engine'])
    write(case / manifest['execution_config'], config)
    for name in (manifest['engine_config'], manifest['execution_config']):
        manifest['inputs_sha256'][name] = digest(case / name)
    stamp = (run / 'run_manifest.json').stat()
    write(run / 'run_manifest.json', manifest)
    os.utime(run / 'run_manifest.json', ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    for path in (run / 'nodes').glob('*/result.json'):
        node = json.loads(path.read_text())
        node['plan']['engine'].pop('worker_extension_cls')
        for name in ('PYTHONPATH', 'FRONTIER_NATIVE_SERVING_EVIDENCE_DIR',
                     'FRONTIER_NATIVE_RUN_GENERATION', 'FRONTIER_NATIVE_ROLE'):
            node['server_env_set'].pop(name)
        argv = node['server_command']
        index = argv.index('--worker-extension-cls')
        del argv[index:index + 2]
        write(path, node)
    for extra in (False, True):
        path = run / 'nodes/cpu-v4-0/result.json'
        node = json.loads(path.read_text())
        if extra:
            node['server_env_set']['PYTHONPATH'] = config['harness_dir']
        write(path, node)
        old = load(FROZEN).check(case, config['run_tag'], 'clean')
        new = load(CHECKER).check(case, config['run_tag'], 'clean')
        assert set(old['checks']) == set(new['checks']) == CHECKS
        assert old['checks'] == new['checks']
        assert new['checks']['harness_pinned_and_client_rows'].startswith('FAIL' if extra else 'PASS')
        assert old['status'] == new['status'] == 'FAIL'
        write(stage.parent / f'legacy_{extra}_predicates.json', new)


def import_origins(stage, phase, modules):
    """Observe real CPU startup paths; resolve specs without importing their bodies."""
    code = '''import importlib.machinery, json, pathlib, sys
stage = pathlib.Path(sys.argv[1])
if sys.argv[2] == "worker":
    sys.path[:0] = [str(stage), str(stage / "tests/comparison/calibration")]
print(json.dumps({name: importlib.machinery.PathFinder.find_spec(name, sys.path).origin
                  for name in json.loads(sys.argv[3])}))
assert not any(name.split(".")[0] in ("torch", "vllm") for name in sys.modules)
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=(str(stage) if phase == 'worker'
                           else str(stage / 'tests/comparison/calibration') + ':' + str(stage)))
    result = subprocess.run([sys.executable, '-B', '-c', code, str(stage), phase, json.dumps(modules)],
                            cwd=stage, env=env, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize('module,kind', [('native_full_structure', 'module'),
    ('native_full_structure', 'package'), ('native_dummy_initialization', 'module'),
    ('native_dummy_initialization', 'package'), ('pd_replay', 'module'), ('pd_replay', 'package')])
def test_v4_real_import_shadow_is_rejected_with_unchanged_nested_pins(v4_package, module, kind):
    stage, case, run, config, manifest = v4_package
    nested = stage / f'tests/comparison/calibration/{module}.py'
    pinned = digest(nested)
    shadow = stage / (f'{module}.py' if kind == 'module' else f'{module}/__init__.py')
    shadow.parent.mkdir(exist_ok=True)
    shadow.write_bytes(nested.read_bytes() + b'\n# Root shadow fixture.\n')
    phase = 'worker' if module == 'pd_replay' else 'server'
    assert import_origins(stage, phase, [module])[module] == str(shadow)
    assert digest(nested) == pinned == manifest['harness_sha256'][f'tests/comparison/calibration/{module}.py']
    previous = check_package(load(PREVIOUS), v4_package, f'r2_previous_{module}_{kind}')
    assert previous['checks']['harness_pinned_and_client_rows'].startswith('PASS')
    current = check_package(load(CHECKER), v4_package, f'r2_{module}_{kind}')
    assert current['checks']['harness_pinned_and_client_rows'].startswith('FAIL'), current['checks']
    assert 'Staged import resolves outside its verified owner:' in current['checks']['harness_pinned_and_client_rows']


@pytest.mark.parametrize('location,module', [
    ('tests/comparison/calibration', 'pd_replay'),
    ('tests/comparison/calibration', 'pd_metrics'),
    ('tests/comparison/calibration', 'pd_proxy'),
    ('tests/comparison/calibration', 'vllm_replay'),
    ('tests/comparison/stage_admission_pp', 'vllm_burst_driver'),
    ('tests/comparison/calibration', 'native_full_structure'),
    ('tests/comparison/calibration', 'native_dummy_initialization'),
    ('', 'probe_nixl_read'), ('', 'probe_runtime_inventory')])
def test_v4_package_before_verified_file_is_rejected(v4_package, location, module):
    stage, case, run, config, manifest = v4_package
    shadow = stage / location / module / '__init__.py'
    shadow.parent.mkdir()
    shadow.write_text('# Import package takes precedence over the verified .py file.\n')
    result = check_package(load(CHECKER), v4_package, f'r2_package_{module}')
    assert 'Staged import resolves outside its verified owner:' in result['checks']['harness_pinned_and_client_rows']


@pytest.mark.parametrize('collector_shadow', [None, 'module', 'package'])
def test_v4_actual_worker_and_server_spawn_resolve_verified_owners(v4_package, collector_shadow):
    stage, case, run, config, manifest = v4_package
    code = '''import importlib, json, pathlib, sys
stage = pathlib.Path(sys.argv[1])
sys.path[:0] = [str(stage), str(stage / "tests/comparison/calibration")]
modules = ["probe_nixl_read", "probe_runtime_inventory", "pd_replay", "pd_metrics", "pd_proxy",
           "vllm_replay", "vllm_burst_driver"]
origins = {name: importlib.import_module(name).__file__ for name in modules}
assert not any(name.split(".")[0] in ("torch", "vllm") for name in sys.modules)
print(json.dumps({"origins": origins, "path": sys.path}))
'''
    result = subprocess.run([sys.executable, '-B', '-c', code, str(stage)], cwd=stage,
        env=dict(os.environ, PYTHONPATH=str(stage), PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True, text=True, check=True)
    worker = json.loads(result.stdout)
    expected = {
        'probe_nixl_read': str(stage / 'probe_nixl_read.py'),
        'probe_runtime_inventory': str(stage / 'probe_runtime_inventory.py'),
        'pd_replay': str(stage / 'tests/comparison/calibration/pd_replay.py'),
        'pd_metrics': str(stage / 'tests/comparison/calibration/pd_metrics.py'),
        'pd_proxy': str(stage / 'tests/comparison/calibration/pd_proxy.py'),
        'vllm_replay': str(stage / 'tests/comparison/dp_placement_pp/vllm_replay.py'),
        'vllm_burst_driver': str(stage / 'tests/comparison/stage_admission_pp/vllm_burst_driver.py'),
    }
    assert worker['origins'] == expected
    if collector_shadow:
        nested = stage / 'tests/comparison/calibration/native_full_structure.py'
        shadow = stage / ('native_full_structure.py' if collector_shadow == 'module'
                          else 'native_full_structure/__init__.py')
        shadow.parent.mkdir(exist_ok=True)
        shadow.write_bytes(nested.read_bytes() + b'\n# Server and spawn shadow.\n')
    # A non-isolated -m parent and real multiprocessing spawn inherit cwd and path.
    script = stage / 'reader_resolution_probe.py'
    script.write_text('''import importlib.machinery, json, multiprocessing, sys
def observe():
    names = ["native_full_structure", "native_dummy_initialization"]
    assert not any(name.split(".")[0] in ("torch", "vllm") for name in sys.modules)
    return {"origins": {name: importlib.machinery.PathFinder.find_spec(name, sys.path).origin
                        for name in names}, "path": sys.path}
def child(pipe):
    pipe.send(observe())
    pipe.close()
if __name__ == "__main__":
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=child, args=(send,))
    process.start()
    assert receive.poll(30), "Spawn resolver did not return"
    spawned = receive.recv()
    process.join(30)
    assert process.exitcode == 0
    print(json.dumps({"server": observe(), "spawn": spawned}))
''')
    result = subprocess.run([sys.executable, '-B', '-m', 'reader_resolution_probe'], cwd=stage,
        env=dict(os.environ, PYTHONPATH=config['harness_dir'] + ':' + str(stage),
                 PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, check=True)
    serving = json.loads(result.stdout)
    expected = {name: str(stage / 'tests/comparison/calibration' / (name + '.py'))
                for name in ('native_full_structure', 'native_dummy_initialization')}
    if collector_shadow:
        expected['native_full_structure'] = str(shadow)
    assert serving['server']['origins'] == serving['spawn']['origins'] == expected
    assert serving['server']['path'][:3] == serving['spawn']['path'][:3] == [
        str(stage), config['harness_dir'], str(stage)]
    write(stage.parent / 'r2_actual_resolution.json', {'worker': worker, **serving})
    result = check_package(load(CHECKER), v4_package, 'r2_actual_resolution_control')
    if collector_shadow:
        assert 'Staged import resolves outside its verified owner:' in result['checks']['harness_pinned_and_client_rows']
    else:
        assert {name for name, value in result['checks'].items() if value.startswith('FAIL')} == {
            'rank_runtime_graph', 'terminal_status'}


def test_v4_nonwinning_and_unrelated_root_files_remain_allowed(v4_package):
    stage, case, run, config, manifest = v4_package
    for name in ('pd_metrics', 'pd_proxy', 'vllm_replay', 'vllm_burst_driver', 'unrelated_notes'):
        (stage / (name + '.py')).write_text('# Does not win any reachable import.\n')
    (stage / 'native_full_structure').mkdir()  # A namespace portion loses to the verified module.
    (stage / 'vllm').mkdir()  # A namespace portion also loses to installed vLLM's regular package.
    result = check_package(load(CHECKER), v4_package, 'r2_nonwinning_control')
    assert {name for name, value in result['checks'].items() if value.startswith('FAIL')} == {
        'rank_runtime_graph', 'terminal_status'}


def test_v4_current_generation05_layout_keeps_only_missing_native_evidence_failures(tmp_path):
    source = Path('/data/ycfeng/tmp/glm53_dsv4/v4_generation05_joint_d06f')
    fixture = copy_cpu_package(source, tmp_path)
    stage, case, run, config, manifest = fixture
    assert len(config['expected_source_hashes']) == 67
    result = check_package(load(CHECKER), fixture, 'r2_generation05_control')
    assert {name for name, value in result['checks'].items() if value.startswith('FAIL')} == {
        'rank_runtime_graph', 'terminal_status'}, result['checks']
    expected = ("FAIL: UNVERIFIED: KeyError: 'vllm/v1/worker/gpu/dp_utils.py'"
        if CHECKER.name.endswith(('_r3.py', '_r4.py')) else
        'FAIL: UNVERIFIED: ValueError: Missing or unexpected physical-rank serving files: cpu-v4-0')
    assert result['checks']['rank_runtime_graph'] == expected


@pytest.mark.parametrize('fault', ['replaced', 'overridden'])
def test_v4_bootstrap_cwd_must_match_observed_server_search_order(v4_package, fault):
    stage, case, run, config, manifest = v4_package
    path = Path(config['observer_bootstrap'])
    replacement = ('os.chdir(staged.parent)' if fault == 'replaced'
                   else 'os.chdir(staged)\n    os.chdir(staged.parent)')
    path.write_text(path.read_text().replace('os.chdir(staged)', replacement))
    config['retained_files'][path.name] = digest(path)
    manifest['inputs_sha256'][str(path)] = digest(path)
    write(case / manifest['execution_config'], config)
    manifest['inputs_sha256'][manifest['execution_config']] = digest(case / manifest['execution_config'])
    stamp = (run / 'run_manifest.json').stat()
    write(run / 'run_manifest.json', manifest)
    os.utime(run / 'run_manifest.json', ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    result = check_package(load(CHECKER), v4_package, 'r2_bootstrap_cwd')
    assert 'Bootstrap no longer declares the reviewed server working directory' in (
        result['checks']['harness_pinned_and_client_rows'])


@pytest.mark.parametrize('location,kind', [('', 'module'), ('', 'package'),
    ('tests/comparison/calibration', 'module'), ('tests/comparison/calibration', 'package')])
def test_v4_server_entrypoint_cannot_resolve_stage_code_as_installed_vllm(v4_package, location, kind):
    stage, case, run, config, manifest = v4_package
    shadow = stage / location / ('vllm.py' if kind == 'module' else 'vllm/__init__.py')
    shadow.parent.mkdir(exist_ok=True)
    shadow.write_text('# This reachable server -m entrypoint is not the installed vLLM package.\n')
    assert import_origins(stage, 'server', ['vllm'])['vllm'] == str(shadow)
    previous = check_package(load(PREVIOUS), v4_package, f'r2_previous_vllm_{kind}')
    assert previous['checks']['harness_pinned_and_client_rows'].startswith('PASS')
    result = check_package(load(CHECKER), v4_package, f'r2_vllm_{kind}')
    assert 'Staged server import shadows the installed vLLM package:' in (
        result['checks']['harness_pinned_and_client_rows'])


def test_v4_collector_dependency_cannot_be_omitted_from_checkout_binding(v4_package):
    stage, case, run, config, manifest = v4_package
    module = 'tests/comparison/calibration/native_dummy_initialization.py'
    manifest['harness_sha256'].pop(module)
    stamp = (run / 'run_manifest.json').stat()
    write(run / 'run_manifest.json', manifest)
    os.utime(run / 'run_manifest.json', ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    result = check_package(load(CHECKER), v4_package, 'r2_missing_checkout_owner')
    assert 'Staged import has no verified checkout owner:' in result['checks']['harness_pinned_and_client_rows']


@pytest.mark.parametrize('shadow', [None, 'module', 'package', 'native_origin', 'native_bytes', 'callable_origin', 'missing_env'])
@pytest.mark.parametrize('quantization', ['mxfp4', 'fp8'])
def test_balanced_strategy_actual_import_and_missing_identity(tmp_path, monkeypatch, shadow, quantization):
    import ast
    import inspect
    import torch
    from types import SimpleNamespace, ModuleType
    from tests.comparison.calibration.native_full_structure import balanced_routing_inventory, FullModelInventory, identity
    from tests.comparison.calibration.native_dummy_initialization import inspect_model
    from tests.unit.test_native_dummy_initialization import config_fixture
    from tests.comparison.dp_placement_pp.vllm_replay import server_env
    frozen = load(REPO / 'task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py')
    checker = load(REPO / 'task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r4.py')
    path = tmp_path / 'balanced_routing.py'
    shutil.copyfile(REPO / checker.BALANCED_MODULE, path)
    module = load(path)
    native = Path('/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69')
    sources = {}
    bodies = ('def __getattr__(name): return "balanced_topk"\nenvironment_variables = {"VLLM_MOE_ROUTING_SIMULATION_STRATEGY": __getattr__}',
        'def create_fused_moe_router(): pass',
        'class RoutingSimulator:\n _routing_strategies = {}\n register_strategy = classmethod(lambda cls, n, s: cls._routing_strategies.update({n: s}))\n simulate_routing = staticmethod(lambda: None)\nclass RoutingSimulatorRouter:\n def _compute_routing(self): pass',
        'from torch.nn import Module\nclass MoERunner(Module):\n def _forward_impl(self): pass')
    for name, body in zip(checker.ROUTING_PINS, bodies):
        key = name[:-3].replace('/', '.')
        stub = ModuleType(key)
        stub.__file__ = str(native / name)
        exec(compile(body, stub.__file__, 'exec'), stub.__dict__)
        monkeypatch.setitem(sys.modules, key, stub)
        sources[name] = digest(native / name)
    simulator = sys.modules[checker.ROUTING_PINS[2][:-3].replace('/', '.')]
    registry = simulator.RoutingSimulator
    monkeypatch.setitem(sys.modules, 'vllm.distributed', SimpleNamespace(get_ep_group=lambda: None))
    monkeypatch.setitem(sys.modules, 'vllm.forward_context', SimpleNamespace(get_forward_context=lambda: None))
    simulator.RoutingStrategy = object
    module.__name__ = 'balanced_routing'
    monkeypatch.setitem(sys.modules, 'balanced_routing', module)
    module.register_balanced_topk()
    envs = sys.modules['vllm.envs']
    factory = sys.modules[checker.ROUTING_PINS[1][:-3].replace('/', '.')]
    factory.envs = simulator.envs = envs
    monkeypatch.setitem(sys.modules, 'vllm.model_executor.layers.fused_moe.layer', factory)
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(envs=envs))
    monkeypatch.setitem(sys.modules, 'vllm.envs', envs)
    monkeypatch.setenv('VLLM_MOE_ROUTING_SIMULATION_STRATEGY', 'balanced_topk')
    runner = sys.modules[checker.ROUTING_PINS[3][:-3].replace('/', '.')].MoERunner()
    runner.router = simulator.RoutingSimulatorRouter()
    child = type('RoutedExperts', (torch.nn.Module,), {'__module__':
        'vllm.model_executor.layers.fused_moe.routed_experts'})()
    method = type('Mxfp4MoEMethod', (), {})()
    experts = type('MarlinExperts', (), {})()
    method.mxfp4_backend, method.experts_cls = SimpleNamespace(name='MARLIN'), type(experts)
    method.num_experts, method.intermediate_size, method.hidden_size = 1, 32, 32
    method.moe_quant_config = SimpleNamespace(use_mxfp4_w4a16=True, a1_scale=None, a2_scale=None)
    experts.quant_config, experts.input_dtype = method.moe_quant_config, None
    method.moe_kernel = SimpleNamespace(fused_experts=experts,
        prepare_finalize=type('MoEPrepareAndFinalizeNaiveDPEPModular', (), {})())
    if quantization == 'fp8':
        method = type('Fp8MoEMethod', (), {})()
        method.fp8_backend, method.block_quant = SimpleNamespace(name='TRITON'), True
        method.moe_quant_config = SimpleNamespace(block_shape=[128, 128],
            w1_scale=None, w2_scale=None, a1_scale=None, a2_scale=None)
        method.moe_kernel = SimpleNamespace(fused_experts=type('TritonExperts', (), {})(),
            prepare_finalize=type('MoEPrepareAndFinalizeNaiveDPEPModular', (), {})())
        child.w13_weight_scale_inv = child.w2_weight_scale_inv = None
        child.weight_block_size = [128, 128]
    child.quant_method, runner.routed_experts = method, child
    model = torch.nn.Module()
    model.experts = runner
    row = balanced_routing_inventory(model)
    inventory = dict(moe_routing=row, initialization=inspect_model(model, config_fixture()))
    assert digest(Path(inspect.getfile(FullModelInventory))) == 'e6a9025c9af92237cd0d25dd937ecd00177a36a0cddf512bb9aa5d81176378aa'
    assert digest(Path(inspect.getfile(inspect_model))) == '118b5feb8b3fb15f72f3736fbf0af166eaeebe99cba46245b2ddf5ddc2b6a8ca'
    tree = ast.parse(inspect.getsource(FullModelInventory))
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Subscript) and isinstance(n.targets[0].slice, ast.Constant)
        and n.targets[0].slice.value == 'moe_backends')
    # Execute the unchanged real comprehension, not a hand-filled backend map.
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), inspect.getfile(FullModelInventory), 'exec'),
        {'self': SimpleNamespace(get_model=lambda: model), 'result': inventory, 'identity': identity})
    assert set(inventory['moe_backends']) == (set() if quantization == 'mxfp4' else {'experts.routed_experts'})
    assert checker.independent_routing_owners(inventory) == {'experts'}
    config = dict(script=str(tmp_path / 'worker.py'), harness_files={checker.BALANCED_MODULE: path.name},
        installed_package=str(native / 'vllm'), expected_source_hashes=sources)
    manifest = dict(harness_sha256={checker.BALANCED_MODULE: checker.BALANCED_SHA256}, groundtruth_checkout_path=str(native))
    engine, env = dict(moe_gating_routing_runtime_path='balanced_topk'), {'VLLM_MOE_ROUTING_SIMULATION_STRATEGY': 'balanced_topk'}
    monkeypatch.setattr(sys.modules[server_env.__module__], 'runtime_version', lambda engine: '0.28.0')
    assert server_env('clean', engine | {'worker_extension_cls': 'native_full_structure.FullModelInventory'}, tmp_path, {})[1] == env
    assert server_env('clean', {}, tmp_path, env)[0] == {}
    if quantization == 'mxfp4':
        with pytest.raises(ValueError, match='Actual balanced routing strategy/source/environment identity mismatch'):
            frozen.verify_balanced_runtime(inventory, env, engine, config, manifest)
    else:
        frozen.verify_balanced_runtime(inventory, env, engine, config, manifest)
    checker.verify_balanced_runtime(inventory, env, engine, config, manifest)
    if shadow in ('module', 'package'):
        target = tmp_path / ('balanced_routing.py' if shadow == 'module' else 'balanced_routing/__init__.py')
        target.parent.mkdir(exist_ok=True)
        target.write_text('# Same-name unapproved strategy.\n')
        inventory['moe_routing'] = row | {'origin': importlib.machinery.PathFinder.find_spec('balanced_routing', [str(tmp_path)]).origin, 'sha256': digest(target)}
    elif shadow in ('native_origin', 'native_bytes'):
        row['native_sources']['vllm/envs.py']['origin' if shadow == 'native_origin' else 'sha256'] = 'unverified'
    elif shadow == 'callable_origin':
        factory.create_fused_moe_router.__code__ = factory.create_fused_moe_router.__code__.replace(co_filename=str(tmp_path / 'shadow.py'))
        with pytest.raises(ValueError, match='callable'):
            balanced_routing_inventory(model)
        return
    elif shadow == 'missing_env':
        env.clear()
    else:
        inventory.pop('moe_routing')
    with pytest.raises((ValueError, KeyError)):
        checker.verify_balanced_runtime(inventory, env, engine, config, manifest)


@pytest.mark.parametrize('fault', [None, 'missing_hash', 'mismatch', 'selector', 'model', 'role', 'format', 'dp', 'ambient', 'source_bytes'])
def test_balanced_real_frontier_input_binding(tmp_path, monkeypatch, fault):
    checker = load(REPO / 'task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py')
    path = tmp_path / 'case/inputs/frontier.json'
    path.parent.mkdir(parents=True)
    shutil.copyfile(R2.parent.parent / 'task_2026-10-22_v4_balanced_routing/balanced_frontier_input/frontier.json', path)
    config = dict(script=str(tmp_path / 'worker.py'), gpu_type='H800',
        harness_worktree='/data/ycfeng/Frontier/.worktrees/v4-balanced-callers',
        retained_files={'case/inputs/frontier.json': digest(path)},
        tool_hashes={str(path): digest(path)})
    engine = dict(model_name='deepseek-ai/DeepSeek-V4-Flash', tensor_parallel_size=8, data_parallel_size=2,
        pipeline_parallel_size=1, moe_gating_routing_runtime_path='balanced_topk')
    manifest = dict(inputs_sha256={str(path): digest(path)}, harness_worktree=config['harness_worktree'], harness_commit='186dfef5503ecb5329bd0fa96f9e978cc813276a')
    if fault in ('selector', 'model', 'role', 'format'):
        flat = json.loads(path.read_text())
        key = {'selector': 'replica_config_moe_gating_routing_runtime_path', 'model': 'replica_config_model_name',
            'role': 'cluster_config_decode_replica_config_moe_expert_parallel_size', 'format': 'trace_request_generator_config_trace_file'}[fault]
        flat.pop(key)
        write(path, flat)
        config['retained_files']['case/inputs/frontier.json'] = config['tool_hashes'][str(path)] = manifest['inputs_sha256'][str(path)] = digest(path)
    elif fault == 'dp':
        flat = json.loads(path.read_text())
        flat['replica_config_attn_dp'] = 1
        write(path, flat)
        config['retained_files']['case/inputs/frontier.json'] = config['tool_hashes'][str(path)] = manifest['inputs_sha256'][str(path)] = digest(path)
    elif fault == 'ambient':
        monkeypatch.setitem(sys.modules, 'frontier.config', None)
        monkeypatch.setitem(sys.modules, 'frontier.config.flat_dataclass', None)
    elif fault == 'source_bytes':
        manifest['harness_commit'] = '9d4467552d8a9bc701b2ef699d6c529cd5f46514'
    elif fault:
        manifest['inputs_sha256'][str(path)] = '0' * 64
        if fault == 'missing_hash':
            manifest['inputs_sha256'].clear()
    if fault not in (None, 'ambient'):
        with pytest.raises((KeyError, ValueError), match='Actual imported Frontier source differs' if fault == 'source_bytes' else None):
            checker.verify_balanced_frontier(config, engine, manifest)
    else:
        receipt = checker.verify_balanced_frontier(config, engine, manifest)
        assert receipt['sources']['frontier/config/config.py']['origin'] == config['harness_worktree'] + '/frontier/config/config.py'


R4 = REPO / 'task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r4.py'
NATIVE06 = Path('/data/ycfeng/tmp/glm53_dsv4/native06_case_20261023')
NATIVE06_RANKS = sorted(NATIVE06.glob('runs/groundtruth_clean/*/nodes/*/serving/rank*.json'))


def native06_operands(path):
    run = path.parents[3]
    manifest = json.loads((run / 'run_manifest.json').read_text())
    config = json.loads((NATIVE06 / manifest['execution_config']).read_text())
    engine = json.loads((NATIVE06 / manifest['engine_config']).read_text())
    node = json.loads((path.parents[1] / 'result.json').read_text())
    return json.loads(path.read_text())['inventory'], node['server_env_set'], engine, config, manifest


@pytest.mark.parametrize('path', NATIVE06_RANKS, ids=lambda p: p.parent.parent.name + '-' + p.stem)
def test_r4_actual_native06_rank_owner_identity(path):
    checker = load(R4)
    operands = native06_operands(path)
    assert len(NATIVE06_RANKS) == 32
    assert operands[0]['moe_backends'] == {}
    assert len(checker.independent_routing_owners(operands[0])) == 43
    checker.verify_balanced_runtime(*operands)
    with pytest.raises(ValueError, match='Actual balanced routing strategy/source/environment identity mismatch'):
        load(R4.with_name('check_groundtruth_run_successor_r3.py')).verify_balanced_runtime(*operands)


@pytest.mark.parametrize('kind', ['owner', 'child', 'backend', 'router', 'runner'])
@pytest.mark.parametrize('fault', ['missing', 'extra', 'wrong'])
def test_r4_rejects_incomplete_or_wrong_independent_identity(kind, fault):
    if not NATIVE06_RANKS:
        pytest.skip('Retained native06 ranks are unavailable')
    checker = load(R4)
    operands = native06_operands(NATIVE06_RANKS[0])
    inventory = operands[0]
    owner = sorted(checker.independent_routing_owners(inventory))[0]
    if kind in ('owner', 'child'):
        mapping = inventory['initialization']['modules']
    elif kind == 'backend':
        mapping = inventory['initialization']['backends']
    else:
        mapping = inventory['moe_routing'][kind + 's']
    name = owner + (checker.CHILD_SUFFIX if kind in ('child', 'backend') else '')
    if fault == 'missing':
        del mapping[name]
    elif fault == 'extra':
        mapping['unexpected' + (checker.CHILD_SUFFIX if kind in ('child', 'backend') else '')] = mapping[name]
    else:
        mapping[name] = mapping[name] | {'experts': 'WrongExperts'} if kind == 'backend' else 'wrong.Class'
    with pytest.raises((ValueError, KeyError)):
        checker.verify_balanced_runtime(*operands)


@pytest.mark.parametrize('fault', ['empty', 'quant_method', 'backend', 'experts', 'prepare_finalize',
    'input_dtype', 'strategy', 'origin', 'sha256', 'native_origin', 'native_hash', 'selector',
    'environment', 'runtime_path', 'manifest_hash', 'missing_initialization', 'wrong_owner_name', 'wrong_child_name'])
def test_r4_rejects_empty_coverage_consumer_and_source_mismatch(fault):
    if not NATIVE06_RANKS:
        pytest.skip('Retained native06 ranks are unavailable')
    checker = load(R4)
    operands = native06_operands(NATIVE06_RANKS[0])
    inventory, env, engine, config, manifest = operands
    initialization, actual = inventory['initialization'], inventory['moe_routing']
    if fault == 'empty':
        initialization['modules'].clear()
        initialization['backends'].clear()
        actual['routers'].clear()
        actual['runners'].clear()
    elif fault in checker.BACKEND_RECORDS[0]:
        name = sorted(checker.independent_routing_owners(inventory))[0] + checker.CHILD_SUFFIX
        initialization['backends'][name][fault] = None if fault == 'input_dtype' else 'unverified'
    elif fault in ('native_origin', 'native_hash'):
        actual['native_sources'][checker.ROUTING_PINS[0]]['origin' if fault == 'native_origin' else 'sha256'] = 'unverified'
    elif fault == 'selector':
        engine['moe_gating_routing_runtime_path'] = 'uniform_topk'
    elif fault == 'manifest_hash':
        manifest['harness_sha256'][checker.BALANCED_MODULE] = '0' * 64
    elif fault == 'missing_initialization':
        del inventory['initialization']
    elif fault in ('wrong_owner_name', 'wrong_child_name'):
        name = sorted(checker.independent_routing_owners(inventory))[0]
        name += checker.CHILD_SUFFIX if fault == 'wrong_child_name' else ''
        initialization['modules']['wrong.' + name] = initialization['modules'].pop(name)
    else:
        actual[fault] = 'unverified'
    with pytest.raises((ValueError, KeyError)):
        checker.verify_balanced_runtime(*operands)


def test_r4_only_replaces_owner_seam_and_preserves_legacy_controls():
    checker = load(R4)
    frozen = load(checker.R3_PATH)
    text = checker.routing_successor_source()
    assert text.replace('    routers = independent_routing_owners(inventory)\n',
        "    routers = {name.removesuffix('.routed_experts') for name in inventory['moe_backends']}\n") == checker.R3_PATH.read_text()
    assert checker.successor_source() == frozen.successor_source()
    checker.verify_balanced_runtime({}, {}, {}, {}, {})
    frozen.verify_balanced_runtime({}, {}, {}, {}, {})
    for verify in (checker.verify_balanced_runtime, frozen.verify_balanced_runtime):
        with pytest.raises(KeyError):
            verify({}, {'VLLM_MOE_ROUTING_SIMULATION_STRATEGY': 'balanced_topk'}, {}, {}, {})


@pytest.mark.parametrize('generation', ['r3', 'r2'])
def test_r4_rejects_frozen_dependency_tamper(tmp_path, generation):
    checker = load(R4)
    original = checker.R3_PATH if generation == 'r3' else checker.R2_PATH
    changed = tmp_path / original.name
    changed.write_bytes(original.read_bytes() + b'\n# Unapproved dependency bytes.\n')
    source = checker.routing_successor_source if generation == 'r3' else checker.successor_source
    with pytest.raises(ValueError, match='Frozen ' + generation.upper() + ' dependency hash mismatch'):
        source(changed)
