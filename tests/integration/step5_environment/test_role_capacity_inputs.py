"""Validate the capacity launch trust boundary without SDK or native imports."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest

PATH = Path(__file__).with_name('check_role_capacity.py')
spec = importlib.util.spec_from_file_location('role_capacity', PATH)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


@pytest.mark.parametrize('task,output', [(Path('/untrusted'), module.TASK/'new'),
    (module.TASK, Path('/outside')), (module.TASK, module.TASK)])
def test_untrusted_paths_are_rejected_before_loading(monkeypatch, task, output):
    monkeypatch.setattr(sys, 'argv', [str(PATH), 'submit', '--task-root', str(task), '--output-dir', str(output)])
    monkeypatch.setattr(module, 'load', lambda path: pytest.fail('External module load reached'))
    with pytest.raises(ValueError): module.main()


def test_duplicate_option_uses_actual_last_value(monkeypatch):
    monkeypatch.setattr(sys, 'argv', [str(PATH), 'submit', '--task-root', str(module.TASK),
        '--output-dir', str(module.TASK/'new'), '--task-root', '/untrusted'])
    monkeypatch.setattr(module, 'load', lambda path: pytest.fail('External module load reached'))
    with pytest.raises(ValueError): module.main()


def test_missing_actual_cloud_mount_rejected_before_loading(monkeypatch):
    monkeypatch.setattr(sys, 'argv', [str(PATH), 'run', '--task-root', str(module.TASK),
                                    '--output-dir', str(module.TASK/'new')])
    monkeypatch.setattr(module.os.path, 'ismount', lambda path: False)
    monkeypatch.setattr(module, 'load', lambda path: pytest.fail('External module load reached'))
    with pytest.raises(AssertionError):
        module.main()


@pytest.mark.parametrize('native_failure', [True, False])
def test_evidence_write_failure_does_not_change_native_outcome(monkeypatch, tmp_path, native_failure):
    failure = ValueError('original native failure')
    calls = []
    def llm(**kwargs):
        calls.append(kwargs)
        if native_failure:
            raise failure
        return object()
    monkeypatch.setitem(sys.modules, 'vllm', types.SimpleNamespace(LLM=llm))
    monkeypatch.setattr(module.importlib.metadata, 'distribution', lambda name: types.SimpleNamespace(version='locked'))
    monkeypatch.setattr(module, 'SOURCE_MANIFESTS', {})
    def write(path, value):
        if path.name != 'invocation.json':
            raise OSError(28, 'No space left on device')
    monkeypatch.setattr(module, 'save', write)
    def closed(*args, **kwargs):
        raise BrokenPipeError('closed stdout')
    monkeypatch.setattr('builtins.print', closed)
    launcher = types.SimpleNamespace(EXPECTED_VLLM_VERSION='locked')
    if native_failure:
        with pytest.raises(ValueError) as caught:
            module.execute(module.TASK, tmp_path, launcher)
        assert caught.value is failure
    else:
        module.execute(module.TASK, tmp_path, launcher)
    assert len(calls) == 1 and calls[0]['skip_tokenizer_init'] is False
