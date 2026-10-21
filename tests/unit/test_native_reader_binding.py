"""DS-V4 staged reader admission; CPU artifacts do not qualify a native run."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

import pytest


REPO = Path(__file__).resolve().parents[2]
TASK = REPO / 'task_memory/task_2026-10-22_v4_native_binding'
FROZEN = REPO / 'task_memory/task_2026-10-21_native_cold_start/check_groundtruth_run_successor.py'
CHECKER = Path(os.environ.get('NATIVE_READER_BINDING_CHECKER', TASK / 'check_groundtruth_run_successor.py'))
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
    if not PACKAGE.is_dir():
        pytest.skip('Retained DS-V4 CPU package is unavailable')
    stage = tmp_path / 'v4'
    shutil.copytree(PACKAGE, stage)
    for path in stage.rglob('*.json'):
        path.write_text(path.read_text().replace(str(PACKAGE), str(stage)))
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
