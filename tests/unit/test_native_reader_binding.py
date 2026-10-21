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
PREVIOUS = TASK / 'check_groundtruth_run_successor.py'
R2 = TASK / 'check_groundtruth_run_successor_r2.py'
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
    current = check_package(load(R2), v4_package, f'r2_{module}_{kind}')
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
    result = check_package(load(R2), v4_package, f'r2_package_{module}')
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
    result = check_package(load(R2), v4_package, 'r2_actual_resolution_control')
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
    result = check_package(load(R2), v4_package, 'r2_nonwinning_control')
    assert {name for name, value in result['checks'].items() if value.startswith('FAIL')} == {
        'rank_runtime_graph', 'terminal_status'}


def test_v4_current_generation05_layout_keeps_only_missing_native_evidence_failures(tmp_path):
    source = Path('/data/ycfeng/tmp/glm53_dsv4/v4_generation05_joint_d06f')
    fixture = copy_cpu_package(source, tmp_path)
    stage, case, run, config, manifest = fixture
    assert len(config['expected_source_hashes']) == 67
    result = check_package(load(R2), fixture, 'r2_generation05_control')
    assert {name for name, value in result['checks'].items() if value.startswith('FAIL')} == {
        'rank_runtime_graph', 'terminal_status'}, result['checks']
    assert result['checks']['rank_runtime_graph'] == (
        'FAIL: UNVERIFIED: ValueError: Missing or unexpected physical-rank serving files: cpu-v4-0')


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
    result = check_package(load(R2), v4_package, 'r2_bootstrap_cwd')
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
    result = check_package(load(R2), v4_package, f'r2_vllm_{kind}')
    assert 'Staged server import shadows the installed vLLM package:' in (
        result['checks']['harness_pinned_and_client_rows'])


def test_v4_collector_dependency_cannot_be_omitted_from_checkout_binding(v4_package):
    stage, case, run, config, manifest = v4_package
    module = 'tests/comparison/calibration/native_dummy_initialization.py'
    manifest['harness_sha256'].pop(module)
    stamp = (run / 'run_manifest.json').stat()
    write(run / 'run_manifest.json', manifest)
    os.utime(run / 'run_manifest.json', ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    result = check_package(load(R2), v4_package, 'r2_missing_checkout_owner')
    assert 'Staged import has no verified checkout owner:' in result['checks']['harness_pinned_and_client_rows']
