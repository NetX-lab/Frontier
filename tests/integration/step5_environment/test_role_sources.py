"""Reject untrusted full-role source outputs before loading task code or SDK."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest

SCRIPT = Path(__file__).with_name('inspect_role_sources.py')
TASK = SCRIPT.parents[3]/'task_memory/task_2026-10-15_step5_preview_h800'


@pytest.mark.parametrize('mode', ['submit', 'run'])
@pytest.mark.parametrize('wrong_task', [False, True])
def test_untrusted_paths_do_not_load_collector(monkeypatch, tmp_path, mode, wrong_task):
    spec = importlib.util.spec_from_file_location('role_sources', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.importlib.util, 'spec_from_file_location',
                        lambda *args: pytest.fail('Untrusted path reached module loading'))
    output = tmp_path/'observed'
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), mode, '--task-root',
                                    str(tmp_path if wrong_task else TASK), '--output-dir', str(output)])
    with pytest.raises(AssertionError):
        module.main()
    assert not output.exists()


@pytest.mark.parametrize('argument', ['--task-root', '--output-dir'])
def test_duplicate_paths_validate_the_value_used_by_collector(monkeypatch, tmp_path, argument):
    spec = importlib.util.spec_from_file_location('role_sources', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.importlib.util, 'spec_from_file_location',
                        lambda *args: pytest.fail('Untrusted repeated path reached module loading'))
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), 'submit', '--task-root', str(TASK),
                                    '--output-dir', str(TASK/'native_role_sources/not_written'),
                                    argument, str(tmp_path/'outside')])
    with pytest.raises(AssertionError):
        module.main()


def test_source_set_reaches_worker_without_loading_sdk(monkeypatch):
    spec = importlib.util.spec_from_file_location('role_sources', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen = {}
    sdk = types.ModuleType('steptron.exp.base_exp')
    sdk.ResourceConfig = lambda **kwargs: kwargs
    backend = types.ModuleType('steptron.utils.stepmind')

    def spawn_tasks(resource, **kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(rjob=types.SimpleNamespace(meta=types.SimpleNamespace(name='STUB')),
                                     poll=lambda: 'STUB_ONLY')

    backend.spawn_tasks = spawn_tasks
    monkeypatch.setitem(sys.modules, 'steptron.exp.base_exp', sdk)
    monkeypatch.setitem(sys.modules, 'steptron.utils.stepmind', backend)
    monkeypatch.setenv('STEPMIND_BACKEND', 'rjob')
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), 'submit', '--task-root', str(TASK),
                                    '--output-dir', str(TASK/'native_role_sources/not_written'),
                                    '--source-set', 'mrv2'])
    module.main()
    assert ' run --source-set mrv2 --task-root ' in seen['command']
    assert seen['charged_group'] == 'codesign'
