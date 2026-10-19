"""Verify the disposable-worker overlay at the filesystem and exec boundaries."""
import hashlib
import json
import sys
from pathlib import Path
import pytest
from tests.comparison.calibration.native_observer_bootstrap import bootstrap

FILES = ('v1/frontier_trace.py', 'v1/worker/gpu_worker.py',
    'v1/worker/gpu_model_runner.py', 'v1/engine/core.py', 'v1/core/sched/output.py',
    'compilation/cuda_graph.py')
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
@pytest.fixture
def inputs(tmp_path, monkeypatch):
    package, staged = tmp_path / 'installed/vllm', tmp_path / 'inputs'
    staged.mkdir()
    files = {}
    for index, name in enumerate(FILES):
        target, source = package / name, staged / f'overlay{index}.py'
        target.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f'# Reviewed overlay {index}\n')
        if index:
            target.write_text(f'# Base {index}\n')
        files[name] = dict(source=str(source), sha256=digest(source),
                          base_sha256=digest(target) if index else None)
    (package / '__init__.py').write_text('raise AssertionError("Package executed")')
    monkeypatch.syspath_prepend(str(package.parent))
    monkeypatch.chdir(tmp_path)
    binaries = {}
    for name in ('_C_stable_libtorch.abi3.so', '_moe_C_stable_libtorch.abi3.so'):
        path = package / name
        path.write_bytes(b'locked image extension')
        binaries[str(path)] = digest(path)
    overlay = dict(installed_package=str(package), staged_dir=str(staged),
                   files=files, binary_sha256=binaries)
    observer = dict(run_id='startup-01', source_sha='2cf0a69' + '1' * 33,
                    instrumentation_sha='b' * 40, config_id='full-glm',
                    replica_id='pod', role='MONOLITHIC', expected_ranks=list(range(16)),
                    output_dir='/mnt/codesign-exp/ycfeng/startup-01/trace')
    script = staged / 'native_full_structure.py'
    script.write_text('raise AssertionError("Native diagnostic executed")')
    config = dict(script=str(script), overlay_manifest=str(staged / 'overlay.json'),
                  observer_manifest=str(staged / 'observer.json'),
                  tool_hashes={str(script): digest(script)})
    return package, staged, config, overlay, observer
@pytest.mark.parametrize('fault', ['normal', 'base', 'origin', 'torch', 'vllm',
    'binary_manifest', 'unwritable', 'observer', 'source_hash', 'recorder_present',
    'binary_hash', 'allowlist', 'missing_input', 'binary_removed',
    'missing_graph', 'missing_graph_source', 'missing_graph_base'])
def test_bootstrap_validates_before_writes_and_launch(inputs, monkeypatch, fault):
    package, staged, config, overlay, observer = inputs
    last, binaries = package / FILES[-1], overlay['binary_sha256']
    binary, row = next(iter(binaries)), overlay['files'][FILES[-1]]
    faults = {
        'base': lambda: last.write_text('# Wrong base\n'),
        'origin': lambda: monkeypatch.setattr(sys, 'path', []),
        'torch': lambda: monkeypatch.setitem(sys.modules, 'torch', object()),
        'vllm': lambda: monkeypatch.setitem(sys.modules, 'vllm', object()),
        'binary_manifest': lambda: binaries.pop(binary),
        'binary_removed': lambda: Path(binary).unlink(),
        'missing_graph': lambda: overlay['files'].pop('compilation/cuda_graph.py'),
        'missing_graph_source': lambda: Path(row['source']).unlink(),
        'missing_graph_base': lambda: row.pop('base_sha256'),
        'missing_input': lambda: config.pop('overlay_manifest'),
        'unwritable': lambda: last.chmod(0o444),
        'observer': lambda: observer.pop('run_id'),
        'source_hash': lambda: row.update(sha256='0' * 64),
        'recorder_present': lambda: (package / FILES[0]).write_text('# Unexpected\n'),
        'binary_hash': lambda: binaries.update({binary: '0' * 64}),
        'allowlist': lambda: overlay['files'].update({'unapproved.py': row})}
    if fault != 'normal':
        faults[fault]()
    for name, value in dict(overlay=overlay, observer=observer, config=config).items():
        (staged / f'{name}.json').write_text(json.dumps(value))
    before = {p: p.read_bytes() for p in package.rglob('*') if p.is_file()}
    calls = []
    monkeypatch.setenv('PYTHONPATH', '/unapproved/full/native/source')
    if fault != 'normal':
        expected_error = 'Six approved files required' if fault == 'missing_graph' else None
        with pytest.raises((ValueError, OSError, KeyError), match=expected_error):
            bootstrap(staged / 'config.json', package=package,
                      launch=lambda *args: calls.append(args))
        assert before == {p: p.read_bytes() for p in package.rglob('*') if p.is_file()}
        assert not calls
        return
    bootstrap(staged / 'config.json', package=package,
              launch=lambda *args: calls.append(args))
    executable, argv, env = calls[0]
    assert executable == sys.executable
    assert argv == [executable, '-B', config['script'], '--config',
                    str(staged / 'config.json')]
    assert Path.cwd() == staged and env['PYTHONPATH'] == str(staged)
    observer.update(mode='light', capacity_bytes=4194304,
                    startup_memory={'records': 12, 'bytes': 65536})
    assert json.loads(env['VLLM_FRONTIER_OBSERVER']) == observer
    assert all(digest(package / name) == row['sha256']
               for name, row in overlay['files'].items())
    assert all(digest(Path(path)) == value for path, value in binaries.items())


@pytest.mark.parametrize('fault', ['normal', 'base', 'source', 'model_pin', 'unapproved'])
def test_v4_repair_requires_exact_source_and_original_base_before_writes(inputs, fault):
    package, staged, config, overlay, observer = inputs
    name = 'models/deepseek_v4/nvidia/model.py'
    target, source = package / name, staged / 'v4_model.py'
    target.parent.mkdir(parents=True)
    target.write_text('# Original V4 model\n')
    source.write_text('# Reviewed V4 post-load repair\n')
    overlay['files'][name] = dict(source=str(source), sha256=digest(source),
                                 base_sha256=digest(target))
    config['expected_source_hashes'] = {'vllm/' + name: digest(source)}
    if fault == 'base':
        target.write_text('# Other V4 source\n')
    elif fault == 'source':
        source.write_text('# Unreviewed change\n')
    elif fault == 'model_pin':
        config['expected_source_hashes']['vllm/' + name] = '0' * 64
    elif fault == 'unapproved':
        overlay['files']['model_executor/model_loader/dummy_loader.py'] = overlay['files'][name]
    for filename, value in dict(overlay=overlay, observer=observer, config=config).items():
        (staged / f'{filename}.json').write_text(json.dumps(value))
    before = {p: p.read_bytes() for p in package.rglob('*') if p.is_file()}
    calls = []
    if fault != 'normal':
        with pytest.raises((ValueError, OSError, KeyError)):
            bootstrap(staged / 'config.json', package=package,
                      launch=lambda *args: calls.append(args))
        assert before == {p: p.read_bytes() for p in package.rglob('*') if p.is_file()}
        assert calls == []
    else:
        bootstrap(staged / 'config.json', package=package,
                  launch=lambda *args: calls.append(args))
        assert len(calls) == 1
        assert digest(target) == config['expected_source_hashes']['vllm/' + name]
        assert all(digest(package / path) == row['sha256']
                   for path, row in overlay['files'].items())


@pytest.mark.parametrize('name', [
    'distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py',
    'distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py',
    'v1/worker/kv_connector_model_runner_mixin.py'])
@pytest.mark.parametrize('fault', ['normal', 'missing', 'base', 'source', 'pin'])
def test_nixl_requires_complete_pinned_sources_before_writes(inputs, name, fault):
    package, staged, config, overlay, observer = inputs
    config['expected_source_hashes'] = {}
    for index, path in enumerate([
        'distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py',
        'distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py',
        'v1/worker/kv_connector_model_runner_mixin.py']):
        target, source = package / path, staged / f'nixl_{index}.py'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f'# Original NIXL source {index}\n')
        source.write_text(f'# Reviewed NIXL observation {index}\n')
        overlay['files'][path] = dict(source=str(source), sha256=digest(source),
                                     base_sha256=digest(target))
        config['expected_source_hashes']['vllm/' + path] = digest(source)
    if fault == 'missing':
        overlay['files'].pop(name)
    elif fault == 'base':
        (package / name).write_text('# Different installed source\n')
    elif fault == 'source':
        Path(overlay['files'][name]['source']).write_text('# Unreviewed source\n')
    elif fault == 'pin':
        config['expected_source_hashes']['vllm/' + name] = '0' * 64
    for filename, value in dict(overlay=overlay, observer=observer, config=config).items():
        (staged / f'{filename}.json').write_text(json.dumps(value))
    before = {p: p.read_bytes() for p in package.rglob('*') if p.is_file()}
    calls = []
    if fault != 'normal':
        with pytest.raises((ValueError, OSError, KeyError)):
            bootstrap(staged / 'config.json', package=package,
                      launch=lambda *args: calls.append(args))
        assert before == {p: p.read_bytes() for p in package.rglob('*') if p.is_file()}
        assert calls == []
    else:
        bootstrap(staged / 'config.json', package=package,
                  launch=lambda *args: calls.append(args))
        assert len(calls) == 1
        assert all(digest(package / path) == row['sha256']
                   for path, row in overlay['files'].items())
