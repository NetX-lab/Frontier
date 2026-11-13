"""Regression checks for attention observation confinement and child isolation."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest

SCRIPT = Path(__file__).with_name('check_attention_state.py')
TASK = SCRIPT.parents[3]/'task_memory/task_2026-10-15_step5_preview_h800'


def worker():
    spec = importlib.util.spec_from_file_location('attention_state_check', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mode', ['submit', 'run'])
def test_outside_task_rejected_before_sdk_or_writes(monkeypatch, tmp_path, mode):
    output = tmp_path/'not_written'
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), mode, '--task-root', str(TASK), '--output-dir', str(output)])
    with pytest.raises(AssertionError):
        worker().main()
    assert not output.exists()


def test_failed_case_retained_without_skipping_independent_cases(monkeypatch, tmp_path):
    module = worker()
    written, calls = {}, []
    launcher = types.SimpleNamespace(inspection=types.SimpleNamespace(TASK_ROOT=tmp_path, locked_inputs=lambda: None))
    collector = types.SimpleNamespace(verify_inputs=lambda p: (None, {'files': []}, None), verify_installed=lambda m: None,
        record_environment=lambda p: None, sha256_file=lambda p: '7cb9acf7b030fd61dd545abe9dc989c1934b148684c5f8775f7625c6c9cabeb7',
        dump_json=lambda p, value: written.update({p.name: value}))
    monkeypatch.setattr(module, 'load', lambda p: launcher if p.name == 'submit_export.py' else collector)
    monkeypatch.setattr(module, 'SOURCE_BATCHES', {})
    monkeypatch.setattr(module, 'CLOUD_ROOT', tmp_path/'cloud')
    monkeypatch.setattr(module.os.path, 'ismount', lambda p: True)
    monkeypatch.setattr(module, 'memory', lambda torch: {'host_peak_rss_kb': 1})
    monkeypatch.setitem(sys.modules, 'torch', types.ModuleType('torch'))
    def run(arguments):
        calls.append(arguments)
        return types.SimpleNamespace(returncode=1 if arguments[-1] == 'dsa_short' else 0)
    monkeypatch.setattr(module.subprocess, 'run', run)
    output = tmp_path/'observations'
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), 'run', '--task-root', str(tmp_path), '--output-dir', str(output)])
    with pytest.raises(AssertionError):
        module.main()
    assert [a[-1] for a in calls] == list(module.CASES)
    assert all(a[0] == sys.executable and a[1] == '-B' for a in calls)
    assert len({a[a.index('--output-dir')+1] for a in calls}) == len(module.CASES)
    assert written['case_statuses.json']['cases'][0]['exit_code'] == 1
    assert 'failure.json' in written and 'resources.json' in written
