"""Check new quantizer observations reject paths outside the active task."""
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).with_name('inspect_quantization_binding.py')
TASK = SCRIPT.parents[3]/'task_memory/task_2026-10-15_step5_preview_h800'


def worker():
    spec = importlib.util.spec_from_file_location('quantization_binding', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mode', ['submit', 'run'])
def test_output_confinement_precedes_sdk_and_writes(monkeypatch, tmp_path, mode):
    module = worker()
    output = tmp_path/'not_written'
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), mode, '--task-root', str(TASK), '--output-dir', str(output)])
    with pytest.raises(AssertionError):
        module.main()
    assert not output.exists()


@pytest.mark.parametrize('mode', ['submit', 'run'])
def test_wrong_task_rejected_before_loading_modules(monkeypatch, tmp_path, mode):
    module = worker()
    monkeypatch.setattr(module, 'load', lambda path: pytest.fail('Unvalidated task reached module execution'))
    output = tmp_path/'observed'
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), mode, '--task-root', str(tmp_path), '--output-dir', str(output)])
    with pytest.raises(AssertionError):
        module.main()
    assert not output.exists()


def test_missing_cloud_mount_rejected(monkeypatch):
    module = worker()
    monkeypatch.setattr(module.os.path, 'ismount', lambda path: False)
    with pytest.raises(AssertionError, match='Required cloud volume is not mounted'):
        module.require_cloud_mount()
