"""Verify the disposable-worker overlay at the filesystem and exec boundaries."""
import hashlib
import json
import sys
from pathlib import Path
import pytest
from tests.comparison.calibration.native_observer_bootstrap import bootstrap

FILES = ('v1/frontier_trace.py', 'v1/worker/gpu_worker.py',
    'v1/worker/gpu_model_runner.py', 'v1/engine/core.py', 'v1/core/sched/output.py')
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
    'binary_hash', 'allowlist', 'missing_input', 'binary_removed'])
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
        with pytest.raises((ValueError, OSError, KeyError)):
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
