#!/usr/bin/env python3
"""CPU-only regressions for the isolated local FFN launch trust boundary."""
import ast
import errno
import importlib.util
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT/'check_local_ffn.py'

def driver():
    spec = importlib.util.spec_from_file_location('local_ffn_driver', SCRIPT)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mode,case', [('child', None), ('run', 'dense')])
def test_invalid_case_mode_fails_before_task_loading(monkeypatch, mode, case):
    module = driver()
    monkeypatch.setattr(module, 'load', lambda p: pytest.fail('No task loading before argument admission'))
    argv = [str(SCRIPT), mode, '--task-root', str(module.TASK), '--output-dir', str(module.TASK/'unused_ffn_output')]
    if case: argv += ['--case', case]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit): module.main()


def test_actual_mount_checked_before_task_loading(monkeypatch):
    module = driver()
    monkeypatch.setattr(module.os.path, 'ismount', lambda p: False)
    monkeypatch.setattr(module, 'load', lambda p: pytest.fail('Missing mount must fail first'))
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), 'run', '--task-root', str(module.TASK),
                                    '--output-dir', str(module.TASK/'unused_ffn_output')])
    with pytest.raises(AssertionError): module.main()


def test_submit_preserves_one_gpu_locked_launcher(monkeypatch):
    module = driver(); seen = {}
    launcher = types.SimpleNamespace(inspection=types.SimpleNamespace(locked_inputs=lambda: None),
        resource_config_arguments=lambda: dict(gpu=1, cpu=8, mem_gb=32, positive_tags=['H800']),
        spawn_arguments=lambda command: dict(command=command, charged_group='codesign'))
    monkeypatch.setattr(module, 'load', lambda p: launcher)
    def spawn(resources, **kwargs):
        seen.update(kwargs); seen['resources'] = resources
        return types.SimpleNamespace(rjob=types.SimpleNamespace(meta=types.SimpleNamespace(name='stub')), poll=lambda: 'stub')
    for name, values in [('steptron.exp.base_exp', dict(ResourceConfig=lambda **k: k)),
                         ('steptron.utils.stepmind', dict(spawn_tasks=spawn))]:
        stub = types.ModuleType(name); stub.__dict__.update(values); monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setenv('STEPMIND_BACKEND', 'rjob')
    monkeypatch.setattr(module.os, 'chdir', lambda p: None)
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), 'submit', '--task-root', str(module.TASK),
                                    '--output-dir', str(module.TASK/'unused_ffn_output')])
    module.main()
    assert seen['resources']['gpu'] == 1 and seen['charged_group'] == 'codesign'
    assert 'timeout 2400 ' in seen['command'] and 'VLLM_ENABLE_CUDA_COMPATIBILITY=1' in seen['command']


@pytest.mark.parametrize('native_exit', [0, 1])
def test_final_status_failure_preserves_child_exit(native_exit, tmp_path):
    module = driver()
    fn = next(n for n in ast.parse(SCRIPT.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    def full_disk(*args): raise OSError(errno.ENOSPC, 'evidence volume full')
    with pytest.raises(SystemExit) as caught:
        exec(compile(ast.Module(body=fn.body[-2:], type_ignores=[]), str(SCRIPT), 'exec'),
             dict(diagnostic=module.diagnostic, save=full_disk, output=tmp_path,
                  statuses=[dict(native_exit=native_exit)], CLOUD=module.CLOUD))
    assert caught.value.code == native_exit
