"""Regression checks for native precision output confinement and process isolation."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest


WORKTREE = Path(__file__).resolve().parents[3]
TASK = WORKTREE/'task_memory/task_2026-10-15_step5_preview_h800'
SCRIPT = Path(__file__).with_name('check_precision_ffn.py')


def module():
    spec = importlib.util.spec_from_file_location('precision_ffn_check', SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize('mode', ['submit', 'run'])
def test_output_outside_task_rejected_before_writes(monkeypatch, tmp_path, mode):
    output = tmp_path/'never_written'
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), mode, '--task-root', str(TASK), '--output-dir', str(output)])
    with pytest.raises(AssertionError, match='Output must be'):
        module().main()
    assert not output.exists()


def test_each_case_uses_independent_process_and_retains_failures(monkeypatch, tmp_path):
    worker = module()
    output = tmp_path/'local_results'
    calls, written = [], {}
    fake_launcher = types.SimpleNamespace(inspection=types.SimpleNamespace(TASK_ROOT=tmp_path,
        locked_inputs=lambda: None))
    fake_collector = types.SimpleNamespace(verify_inputs=lambda path: (None, {'files': []}, None),
        verify_installed=lambda manifest: None, record_environment=lambda path: None,
        sha256_file=lambda path: 'worker_stub',
        dump_json=lambda path, value: written.update({path.name: value}))
    monkeypatch.setattr(worker, 'load', lambda path: fake_launcher if path.name == 'submit_export.py' else fake_collector)
    monkeypatch.setattr(worker, 'SOURCE_BATCHES', {})
    monkeypatch.setattr(worker, 'CLOUD_ROOT', tmp_path/'cloud')
    monkeypatch.setattr(worker.os.path, 'ismount', lambda path: True)

    def run(arguments):
        calls.append(arguments)
        return types.SimpleNamespace(returncode=1 if arguments[-1] == 'moe_fp8' else 0)

    monkeypatch.setattr(worker.subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), 'run', '--task-root', str(tmp_path), '--output-dir', str(output)])
    with pytest.raises(AssertionError, match='Native failures retained'):
        worker.main()
    assert len(calls) == len(worker.CASES)
    assert [arguments[-1] for arguments in calls] == list(worker.CASES)
    assert all(arguments[0] == sys.executable and arguments[1] == '-B' for arguments in calls)
    assert len({arguments[arguments.index('--output-dir')+1] for arguments in calls}) == len(calls)
    assert written['case_statuses.json']['cases'][1]['exit_code'] == 1
