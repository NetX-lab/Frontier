"""Fault-injection checks for transparent native capacity observations."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).with_name('role_capacity_observation.py')
spec = importlib.util.spec_from_file_location('capacity_observation', PATH)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


@pytest.mark.parametrize('stage', module.STAGES)
def test_native_return_and_arguments_are_preserved(stage):
    calls, records, result = [], [], object()
    def native(self, *args, **kwargs):
        calls.append((args, kwargs)); return result
    base = type('Native', (), {name: native for name in module.STAGES})
    worker = module.observed_worker(base, lambda *row: records.append(row))()
    assert getattr(worker, stage)(1, enabled=False) is result
    assert calls == [((1,), {'enabled': False})]
    assert [row[2] for row in records] == ['before', 'after']


@pytest.mark.parametrize('stage', module.STAGES)
def test_observer_failure_cannot_mask_native_exception(stage):
    failure, calls = ValueError('original native failure'), []
    def native(self):
        calls.append(stage); raise failure
    def broken(*args):
        raise OSError('observer recording failure')
    base = type('Native', (), {name: native for name in module.STAGES})
    with pytest.raises(ValueError) as caught:
        getattr(module.observed_worker(base, broken)(), stage)()
    assert caught.value is failure and calls == [stage]


def test_closed_stdout_cannot_change_native_outcome(monkeypatch):
    def closed(*args, **kwargs): raise BrokenPipeError('closed stdout')
    monkeypatch.setattr('builtins.print', closed)
    def broken(*args): raise OSError('recording failed')
    result, failure = object(), ValueError('native failure')
    base = type('Native', (), {name: lambda *a: result for name in module.STAGES})
    assert module.observed_worker(base, broken)().load_model() is result
    def fail(*args): raise failure
    base.load_model = fail
    with pytest.raises(ValueError) as caught: module.observed_worker(base, broken)().load_model()
    assert caught.value is failure


def test_module_import_does_not_load_torch_or_vllm():
    import subprocess, sys
    code = "import runpy,sys; runpy.run_path(sys.argv[1]); assert 'torch' not in sys.modules and 'vllm' not in sys.modules"
    subprocess.run([sys.executable, '-B', '-c', code, str(PATH)], check=True)
