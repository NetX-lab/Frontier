"""Regression checks for sparse decode input locks and submission paths."""
import importlib.util
import io
import os
from pathlib import Path
import sys
import types

import pytest


WORKTREE = Path(__file__).resolve().parents[3]
TASK = WORKTREE/'task_memory/task_2026-10-15_step5_preview_h800'
SCRIPT = Path(__file__).with_name('check_sparse_decode.py')


def module():
    spec = importlib.util.spec_from_file_location('sparse_decode_check', SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize('relative', [
    'g1_preparation.json',
    'native_source_export/observed/vllm/models/step5/nvidia/ops/sparse_decode_attention.py',
])
def test_changed_input_rejected_before_submission(monkeypatch, relative):
    target = TASK/relative
    original_open = Path.open

    def open_file(path, mode='r', *args, **kwargs):
        if path == target and mode == 'rb':
            with original_open(path, mode) as stream:
                return io.BytesIO(stream.read()+b'\n')
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_file)
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), 'submit', '--task-root', str(TASK),
                                    '--output-dir', str(TASK/'never_submitted')])
    with pytest.raises((SystemExit, AssertionError)):
        module().main()


def test_relative_paths_resolved_before_submission(monkeypatch, tmp_path):
    observed = {}

    class ResourceConfig:
        def __init__(self, **kwargs):
            observed['resource'] = kwargs

    class Workers:
        rjob = types.SimpleNamespace(meta=types.SimpleNamespace(name='INERT_STUB'))

        def poll(self):
            return 'STUB_ONLY'

    def spawn_tasks(config, **kwargs):
        observed['spawn'] = kwargs
        assert os.environ['EXP_ID'] == 'step5-g1-sparse-decode'
        return Workers()

    for name in ('steptron', 'steptron.exp', 'steptron.exp.base_exp',
                 'steptron.utils', 'steptron.utils.stepmind'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules['steptron.exp.base_exp'].ResourceConfig = ResourceConfig
    sys.modules['steptron.utils.stepmind'].spawn_tasks = spawn_tasks
    monkeypatch.setenv('STEPMIND_BACKEND', 'rjob')
    monkeypatch.chdir(tmp_path)
    output = tmp_path/'never_submitted'
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), 'submit', '--task-root',
                                    os.path.relpath(TASK, tmp_path), '--output-dir', output.name])
    module().main()
    command = observed['spawn']['command']
    assert str(TASK) in command and str(output) in command
    assert observed['spawn']['code_mount_point'] == str(WORKTREE)
    assert observed['spawn']['charged_group'] == 'codesign'
    assert observed['resource']['positive_tags'] == ['H800']
    assert not output.exists()
