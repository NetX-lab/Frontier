"""Exercise native pre-launch provenance using real local Git repositories."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from tests.comparison.calibration import native_pd_worker as worker


REPO = Path(__file__).resolve().parents[2]
TRACE = REPO / 'data/processed_traces/canonical_fixed_6144_1024_req128_qps1'
MODULES = (
    'tests/comparison/calibration/native_pd_worker.py',
    'tests/comparison/calibration/pd_replay.py',
    'tests/comparison/calibration/pd_proxy.py',
    'tests/comparison/calibration/pd_metrics.py',
    'tests/comparison/calibration/client_metrics.py',
    'tests/comparison/calibration/e2e_metrics_gap.py',
    'tests/comparison/dp_placement_pp/vllm_replay.py',
    'tests/comparison/stage_admission_pp/vllm_burst_driver.py',
)


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def commit(path):
    git(path, 'add', '.')
    git(path, '-c', 'user.name=Native CPU fixture', '-c', 'user.email=fixture@invalid',
        '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Explicit CPU fixture source')
    return git(path, 'rev-parse', 'HEAD')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_case(temporary):
    """Create real clean fixture sources; never claim an actual native run."""
    stage = temporary / 'package'
    case_dir = stage / 'case'
    for child in ('inputs', 'inputs/trace'):
        (case_dir / child).mkdir(parents=True, exist_ok=True)
    source, harness = temporary / 'native-source', temporary / 'harness-source'
    for directory in (source, harness):
        directory.mkdir()
        git(directory, 'init', '-q')
    (source / 'source.py').write_text('# CPU provenance fixture, not vLLM runtime.\n')
    source_commit = commit(source)
    mapping = {}
    for name in MODULES:
        destination = harness / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, destination)
        target = 'native_pd_worker.py' if name == MODULES[0] else name
        staged = stage / target
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, staged)
        mapping[name] = target
    commit(harness)
    (stage / 'trace').mkdir()
    for name in ('trace.csv', 'request_ids.json', 'warmup_pool.csv', 'manifest.json', 'SHA256SUMS'):
        shutil.copy2(TRACE / name, stage / 'trace' / name)
        shutil.copy2(TRACE / name, case_dir / 'inputs/trace' / name)
    model = stage / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{"vocab_size":128}\n')
    (stage / 'retained.py').write_text('# Explicit retained fixture dependency.\n')
    engine = dict(vllm_version='0.28.0', data_parallel_size=4, tensor_parallel_size=8,
                  pipeline_parallel_size=1, native_reference={'source_sha': source_commit})
    config = dict(case_dir=str(case_dir), run_tag='cpu-http-fixture', mode='clean',
        engine=engine, script=str(stage / 'native_pd_worker.py'), model_dir=str(model),
        model_hashes={'config.json': sha(model / 'config.json')}, harness_worktree=str(harness),
        harness_files=mapping, retained_files={'retained.py': sha(stage / 'retained.py')},
        gpu_type='H200', replica=8, gpu=8, charged_group='step_main', image='fixture-no-native-image',
        cpu=64, mem_gb=512, timeout_seconds=12339, replay_timeout_s=2469)
    case = dict(case_id='explicit-cpu-http-fixture', fixture_only=True, run_generation='CPU fixture',
        groundtruth_checkout_path=str(source), groundtruth_commit=source_commit,
        workload_sha256={p.name: sha(p) for p in (stage / 'trace').iterdir()})
    (case_dir / 'manifest.yaml').write_text(yaml.safe_dump(case))
    (case_dir / 'inputs/engine.json').write_text(json.dumps(engine))
    (case_dir / 'inputs/native_config.json').write_text(json.dumps(config))
    return config, source, harness


def test_actual_clean_provenance_and_prelaunch_snapshot_are_preserved(tmp_path):
    config, source, harness = native_case(tmp_path)
    path = worker.write_run_manifest(config, 'CPU fixture authorization only; no native job')
    manifest = json.loads(path.read_text())
    assert manifest['mode'] == 'clean' and manifest['groundtruth_tree_dirty'] is False
    assert manifest['groundtruth_commit'] == git(source, 'rev-parse', 'HEAD')
    assert manifest['harness_commit'] == git(harness, 'rev-parse', 'HEAD')
    assert manifest['harness_tree_dirty_paths'] == []
    assert 'selected_checkout' not in manifest
    assert manifest['resources']['gpus'] == 64
    for name, digest in manifest['inputs_sha256'].items():
        target = Path(name) if Path(name).is_absolute() else Path(config['case_dir']) / name
        assert sha(target) == digest
    output = tmp_path / 'retained/case/runs/groundtruth_clean/cpu-http-fixture'
    output.mkdir(parents=True)
    execution_config = Path(config['case_dir']) / 'inputs/native_config.json'
    receipt = worker.write_launcher_receipt(config, 'CPU_FIXTURE_ONLY', execution_config)
    submission = json.loads(receipt.read_text())
    assert submission['run_manifest_sha256'] == sha(path)
    assert submission['execution_config_sha256'] == sha(execution_config)
    assert submission['run_generation'] == manifest['run_generation']
    with pytest.raises(ValueError, match='actual submitted RJob name'):
        worker.write_launcher_receipt(config, '', execution_config)
    worker.stage_run_manifest(config, output)
    assert (output / 'run_manifest.json').read_bytes() == path.read_bytes()
    assert (output / 'run_manifest.json').stat().st_mtime_ns == path.stat().st_mtime_ns
    assert (output / 'launcher_receipt.json').read_bytes() == receipt.read_bytes()
    assert (output / 'launcher_receipt.json').stat().st_mtime_ns == receipt.stat().st_mtime_ns
    assert (output.parents[2] / 'inputs/trace/request_ids.json').read_bytes() == (TRACE / 'request_ids.json').read_bytes()
    with pytest.raises(FileExistsError):
        worker.write_run_manifest(config, 'Do not overwrite an existing run')


@pytest.mark.parametrize('fault', [
    'empty_authorization', 'native_commit', 'native_dirty', 'native_untracked',
    'harness_dirty', 'harness_untracked', 'staged_module', 'case_trace',
    'runner_trace', 'model', 'retained', 'mode', 'execution_config',
])
def test_prelaunch_rejects_actual_source_and_input_mutations(tmp_path, fault):
    config, source, harness = native_case(tmp_path)
    case_dir = Path(config['case_dir'])
    stage = Path(config['script']).parent
    authorization = 'CPU fixture authorization only'
    if fault == 'empty_authorization':
        authorization = '  '
    elif fault == 'native_commit':
        case = yaml.safe_load((case_dir / 'manifest.yaml').read_text())
        case['groundtruth_commit'] = '0' * 40
        (case_dir / 'manifest.yaml').write_text(yaml.safe_dump(case))
    elif fault in ('native_dirty', 'native_untracked'):
        (source / ('source.py' if fault.endswith('dirty') else 'untracked.py')).write_text('# changed\n')
    elif fault in ('harness_dirty', 'harness_untracked'):
        path = harness / (MODULES[0] if fault.endswith('dirty') else 'untracked.py')
        path.write_text('# changed\n')
    elif fault == 'staged_module':
        (stage / 'native_pd_worker.py').write_text('# changed\n')
    elif fault == 'case_trace':
        (case_dir / 'inputs/trace/trace.csv').write_text('changed\n')
    elif fault == 'runner_trace':
        (stage / 'trace/trace.csv').write_text('changed\n')
    elif fault == 'model':
        (Path(config['model_dir']) / 'config.json').write_text('{"changed":true}')
    elif fault == 'retained':
        (stage / 'retained.py').write_text('# changed\n')
    elif fault == 'mode':
        config['mode'] = 'light'
        (case_dir / 'inputs/native_config.json').write_text(json.dumps(config))
    else:
        config['cpu'] += 1
    with pytest.raises(ValueError):
        worker.write_run_manifest(config, authorization)
    assert not (case_dir / 'runs').exists()
