"""CPU checks for transparent resource observation and trusted generation paths."""
import importlib.util
from pathlib import Path
import sys
import types
import weakref
import gc

import pytest

SCRIPT = Path(__file__).with_name('check_precision_resources.py')


def worker():
    spec = importlib.util.spec_from_file_location('precision_resources', SCRIPT)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mode', ['submit', 'run'])
def test_untrusted_paths_precede_module_loading(monkeypatch, tmp_path, mode):
    module = worker()
    monkeypatch.setattr(module, 'load', lambda path: pytest.fail('Untrusted module loaded'))
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), mode, '--task-root', str(tmp_path), '--output-dir', str(tmp_path/'new')])
    with pytest.raises(AssertionError):
        module.main()
    assert not (tmp_path/'new').exists()


def test_transparent_call_returns_identity_and_propagates_failure():
    module = worker(); calls = []; output = object(); sentinel = object()
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return output
    owner = types.SimpleNamespace(call=original)
    phases = []
    module.observe(owner, 'call', 'load', lambda *args: phases.append(args))
    assert owner.call(sentinel, key=sentinel) is output
    assert calls == [((sentinel,), {'key': sentinel})]
    assert phases == [('load', 'before'), ('load', 'after')]
    def fail():
        raise ValueError('native failure')
    owner.call = fail; module.observe(owner, 'call', 'failure', lambda *args: phases.append(args))
    with pytest.raises(ValueError, match='native failure'):
        owner.call()
    assert phases[-2:] == [('failure', 'before'), ('failure', 'after')]


def test_class_method_wrapper_does_not_retain_instance():
    module = worker()
    class Loader:
        def load(self, value):
            return value
    module.observe(Loader, 'load', 'dummy', lambda *args: None)
    instance = Loader(); ref = weakref.ref(instance); value = object()
    assert instance.load(value) is value
    del instance; gc.collect()
    assert ref() is None


def test_uninitialized_cuda_snapshot_does_not_initialize():
    module = worker()
    cuda = types.SimpleNamespace(is_initialized=lambda: False)
    result = module.snapshot(types.SimpleNamespace(cuda=cuda))
    assert result['cuda_allocated_bytes'] is None
    assert result['cuda_lifetime_peak_reserved_bytes'] is None
    assert result['host_current_and_peak_rss_kib']['VmRSS'] > 0


def test_observer_failure_does_not_mask_native_error(capsys):
    module = worker()
    def fail():
        raise ValueError('original native failure')
    def broken_record(*args):
        raise OSError('observer write failed')
    owner = types.SimpleNamespace(call=fail)
    module.observe(owner, 'call', 'load', broken_record)
    with pytest.raises(ValueError, match='original native failure'):
        owner.call()
    assert 'resource_evidence_verified' in capsys.readouterr().err


@pytest.mark.parametrize('version', [1, 2])
def test_cgroup_subtree_mount_root_and_namespace_root(tmp_path, version):
    module = worker(); root = tmp_path/'memory'; root.mkdir()
    name = 'memory.current' if version == 2 else 'memory.usage_in_bytes'
    (root/name).write_text('123')
    filesystem = 'cgroup2 cgroup rw' if version == 2 else 'cgroup cgroup rw,memory'
    mounts = f'7 1 0:5 /pod/container {root} rw - {filesystem}\n'
    controllers = '' if version == 2 else 'memory'
    for path in ('/pod/container', '/'):
        values = module.read_cgroup_memory(f'0:{controllers}:{path}\n', mounts)
        assert values[str(root/name)] == '123'


def test_unresolved_cgroup_rejects_missing_records():
    module = worker()
    with pytest.raises(AssertionError, match='No actual cgroup'):
        module.read_cgroup_memory('0::/\n', '')


def test_parent_process_tree_includes_live_process():
    module = worker(); rows = module.process_tree(module.os.getpid())
    assert any(row['pid'] == module.os.getpid() and row['rss_kib']['VmRSS'] > 0 for row in rows)
