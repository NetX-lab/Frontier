#!/usr/bin/env python3
"""CPU-only checks for the local attention controller and native caller boundary."""
import builtins
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT/'check_local_attention.py'


def driver():
    spec = importlib.util.spec_from_file_location('attention_driver_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    module = driver()
    task = tmp_path/'worktree'/'task_memory'/'task'
    task.mkdir(parents=True)
    monkeypatch.setattr(module, 'TASK', task)
    monkeypatch.setattr(module, 'WORKTREE', task.parent.parent)
    cloud = tmp_path/'cloud'/'ycfeng'
    cloud.mkdir(parents=True)
    monkeypatch.setattr(module, 'PERSONAL_CLOUD', cloud)
    args = SimpleNamespace(mode='run', kind=None, task_root=task, output_dir=task/'generation',
        cloud_root=cloud/'attention_new', worker_tmp_root=tmp_path/'worker'/'new',
        tensor_copy_bytes=67108864, evidence_bytes=4*1024**3, driver_sha256=module.sha256(SCRIPT))
    return module, args


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fake_collector():
    return SimpleNamespace(sha256_file=lambda path: hashlib.sha256(path.read_bytes()).hexdigest())


def test_import_has_no_native_or_platform_side_effects(monkeypatch):
    imported = builtins.__import__
    def forbid_native(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'vllm', 'triton', 'steptron', 'stepmind'):
            pytest.fail('Import crossed the inert controller boundary: '+name)
        return imported(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', forbid_native)
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('Import spawned a child'))
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('Import executed a command'))
    module = driver()
    assert module.KINDS == ('swa', 'dsa')


def test_actual_local_preparation_preserves_the_37_case_30_shape_plan():
    module = driver()
    before = (module.TASK/'local_scope_sampling_01.json').read_bytes()
    _, _, attention, files, plan = module.prepare(module.TASK, module.sha256(SCRIPT))
    selected = attention.samples(plan)
    assert len(plan['cases']) == 37
    assert len(selected) == 30
    assert [row['label'] for row in selected if row['partition'] == 'holdout'] == [
        'holdout_subsequent_19', 'holdout_subsequent_515', 'holdout_subsequent_4099']
    assert all(row['graph'] == 'eager' and row['rows'] == 1 and row['precision'] == 'bf16' for row in selected)
    assert len({row['relative_path'] for row in files}) == len(files) == 73
    assert (module.TASK/'local_scope_sampling_01.json').read_bytes() == before


@pytest.mark.parametrize('field,change', [
    ('task_root', 'foreign'), ('output_dir', 'task'), ('output_dir', 'outside'),
    ('output_dir', 'existing'), ('cloud_root', 'outside'), ('cloud_root', 'root'),
    ('cloud_root', 'existing'), ('worker_tmp_root', 'existing'), ('worker_tmp_root', 'task'),
    ('worker_tmp_root', 'cloud'), ('worker_tmp_root', 'root'), ('worker_tmp_root', 'volume'),
    ('worker_tmp_root', 'worktree'), ('tensor_copy_bytes', 0),
    ('kind', 'swa')])
def test_path_and_generation_rejections_before_loading(inputs, monkeypatch, field, change):
    module, args = inputs
    values = dict(foreign=args.task_root.parent/'foreign', task=args.task_root,
        outside=args.task_root.parent/'outside', root=Path('/') if field == 'worker_tmp_root' else module.PERSONAL_CLOUD,
        cloud=args.cloud_root, volume=Path('/mnt/codesign-exp/ycfeng/native_cache'), worktree=module.WORKTREE)
    if change == 'existing':
        path = getattr(args, field)
        path.mkdir(parents=True)
        values['existing'] = path
    setattr(args, field, values.get(change, change))
    monkeypatch.setattr(module, 'prepare', lambda *a: pytest.fail('Rejected paths reached task imports'))
    with pytest.raises(ValueError):
        module.validate_paths(args)


def test_symlink_escape_and_existing_symlink_outputs_are_rejected(inputs):
    module, args = inputs
    link = args.task_root/'escape'
    link.symlink_to(args.task_root.parent, target_is_directory=True)
    args.output_dir = link/'outside'
    with pytest.raises(ValueError):
        module.validate_paths(args)
    args.output_dir = args.task_root/'generation'
    link = module.PERSONAL_CLOUD/'escape'
    link.symlink_to(args.task_root.parent, target_is_directory=True)
    args.cloud_root = link/'outside'
    with pytest.raises(ValueError):
        module.validate_paths(args)


@pytest.mark.parametrize('kind', [None, 'swa', 'dsa'])
def test_child_requires_parent_generation_and_unused_kind_output(inputs, kind):
    module, args = inputs
    args.mode, args.kind = 'child', kind
    with pytest.raises(ValueError):
        module.validate_paths(args)
    args.cloud_root.mkdir(parents=True); args.worker_tmp_root.mkdir(parents=True)
    if kind is None:
        with pytest.raises(ValueError):
            module.validate_paths(args)
        return
    assert module.validate_paths(args)[2] == args.cloud_root
    (args.cloud_root/kind).mkdir()
    with pytest.raises(ValueError):
        module.validate_paths(args)


def test_missing_actual_mount_fails_before_preparation(inputs, monkeypatch):
    module, args = inputs
    monkeypatch.setattr(module.os.path, 'ismount', lambda path: False)
    monkeypatch.setattr(module, 'prepare', lambda *a: pytest.fail('No imports before actual-mount validation'))
    monkeypatch.setattr(sys, 'argv', ['driver', *module.invocation_argv('run', args.task_root,
        args.output_dir, args.cloud_root, args.worker_tmp_root, args)[1:]])
    with pytest.raises(ValueError, match='mounted'):
        module.main()


def test_freeze_rejects_driver_and_dependency_drift(inputs, monkeypatch):
    module, args = inputs
    with pytest.raises(ValueError, match='driver'):
        module.verify_frozen('0'*64)
    frozen = module.WORKTREE/'dependency.py'
    frozen.write_text('original\n')
    monkeypatch.setattr(module, 'FROZEN_FILES', {'dependency.py': module.sha256(frozen)})
    module.verify_frozen(args.driver_sha256)
    frozen.write_text('changed\n')
    with pytest.raises(ValueError, match='dependency'):
        module.verify_frozen(args.driver_sha256)


def test_freeze_rejects_wrong_loaded_module_path(inputs, monkeypatch):
    module, args = inputs
    frozen = module.WORKTREE/'loaded_dependency.py'
    foreign = module.WORKTREE/'other'/'loaded_dependency.py'
    frozen.write_text('original\n'); foreign.parent.mkdir(); foreign.write_bytes(frozen.read_bytes())
    monkeypatch.setattr(module, 'FROZEN_FILES', {'loaded_dependency.py': module.sha256(frozen)})
    loaded = ModuleType('loaded_dependency'); loaded.__file__ = str(foreign)
    monkeypatch.setitem(sys.modules, 'loaded_dependency', loaded)
    with pytest.raises(ValueError, match='Loaded'):
        module.verify_frozen(args.driver_sha256)


def test_inert_payload_is_exact_and_uses_unchanged_locked_launcher(inputs):
    module, args = inputs
    actual = driver()
    launcher = actual.load(actual.TASK/'native_source_export/submit_export.py')
    before = set(sys.modules)
    config, kwargs = module.submit_payload(launcher, args.task_root, args.output_dir,
                                          args.cloud_root, args.worker_tmp_root, args)
    assert config == dict(cpu=8, gpu=1, mem_gb=32, replica=1,
        image='a.z01.normandyfunc.com/docker-public/vllm/vllm-openai@sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20',
        positive_tags=['H800'], extra_requirements=[],
        mounts=['juicefs+s3://oss.i.shaipower.com/codesign-exp:/mnt/codesign-exp'],
        custom_resources=[], envs={}, command='{COMMAND}', task_specs={'default': {'is_critical': True}})
    shell = ('export VLLM_ENABLE_CUDA_COMPATIBILITY=1 LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH '
             'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '
             'PYTHONPYCACHEPREFIX='+shlex.quote(str(args.worker_tmp_root/'cache'/'pycache'))+'; exec '+shlex.join([
                'python3', '-B', str(SCRIPT), 'run', '--task-root', str(args.task_root),
                '--output-dir', str(args.output_dir), '--cloud-root', str(args.cloud_root),
                '--worker-tmp-root', str(args.worker_tmp_root), '--driver-sha256', args.driver_sha256,
                '--tensor-copy-bytes', '67108864', '--evidence-bytes', str(args.evidence_bytes)]))
    assert kwargs == dict(command='timeout 2400 bash -c '+shlex.quote(shell), charged_group='codesign',
                          use_image=True, code_mount_point=str(actual.WORKTREE))
    assert not any(name.split('.')[0] in ('steptron', 'stepmind', 'torch', 'vllm') for name in set(sys.modules)-before)
    assert not args.cloud_root.exists() and not args.output_dir.exists() and not args.worker_tmp_root.exists()


def test_fake_submit_passes_payload_once_and_waits_without_stop(inputs, monkeypatch):
    module, args = inputs
    args.mode = 'submit'
    calls = []
    launcher = SimpleNamespace(resource_config_arguments=lambda: dict(gpu=1),
        spawn_arguments=lambda command: dict(command=command, charged_group='codesign'))
    monkeypatch.setattr(module, 'prepare', lambda *a: (launcher, None, None, None, None))
    def spawn(config, **kwargs):
        calls.append(('spawn', config, kwargs))
        def wait(): calls.append(('poll',)); return 'fake_terminal'
        return SimpleNamespace(rjob=SimpleNamespace(meta=SimpleNamespace(name='fake')), poll=wait,
                               stop=lambda: pytest.fail('No platform stop'))
    for name, exports in [('steptron.exp.base_exp', dict(ResourceConfig=lambda **k: k)),
                          ('steptron.utils.stepmind', dict(spawn_tasks=spawn))]:
        fake = ModuleType(name); fake.__dict__.update(exports); monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.setenv('STEPMIND_BACKEND', 'rjob')
    monkeypatch.setattr(module.os, 'chdir', lambda path: calls.append(('cwd', path)))
    monkeypatch.setattr(sys, 'argv', ['driver', *module.invocation_argv('submit', args.task_root,
        args.output_dir, args.cloud_root, args.worker_tmp_root, args)[1:]])
    assert module.main() == 0
    assert [row[0] for row in calls] == ['cwd', 'spawn', 'poll']
    assert not args.cloud_root.exists() and not args.output_dir.exists()


class FakeChild:
    def __init__(self, code, pid):
        self.code, self.pid, self.polls = code, pid, 0
    def poll(self):
        self.polls += 1
        return None if self.polls == 1 else self.code


@pytest.mark.parametrize('codes,observer_fails,archive_fails,status_fails', [
    ((0, 0), False, False, False), ((9, 0), False, False, False),
    ((0, -15), False, False, False), ((0, 0), True, False, False),
    ((9, 0), True, False, False), ((0, 0), False, True, False),
    ((9, 0), False, True, False), ((0, 0), False, False, True),
    ((9, 0), True, True, True)])
def test_parent_runs_two_sequential_kinds_and_preserves_raw_exit_and_evidence_errors(
        inputs, monkeypatch, codes, observer_fails, archive_fails, status_fails, capsys):
    module, args = inputs
    spawned, children, nvidia = [], [], []
    def spawn(argv, stdout, stderr, env):
        if children:
            assert children[-1].polls == 2, 'Second kind cannot overlap the first'
        kind = argv[argv.index('--kind')+1]
        assert kind == ('swa', 'dsa')[len(spawned)]
        assert env['TMPDIR'] == str(args.worker_tmp_root)
        assert all(Path(env[key]).is_relative_to(args.worker_tmp_root) for key in
            ('XDG_CACHE_HOME', 'HF_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'VLLM_CACHE_ROOT', 'PYTHONPYCACHEPREFIX'))
        assert Path(stdout.name).is_relative_to(args.worker_tmp_root)
        stdout.write(b'raw child stdout\n'); stderr.write(b'raw child stderr\n')
        cell = args.worker_tmp_root/'observations'/kind/'shape'
        write_json(cell/'observation.json', dict(kind=kind, status='raw_eager_observation_not_admitted'))
        child = FakeChild(codes[len(spawned)], 800+len(spawned))
        children.append(child); spawned.append(argv)
        return child
    def sample(*args):
        if observer_fails: raise OSError('fake observer unavailable')
        return [dict(pid=123, rss_kib={'VmRSS': 2})]
    resources = SimpleNamespace(process_tree=sample, cgroup_memory=lambda: {'memory.current': '2'}, SAMPLE_SECONDS=1)
    monkeypatch.setattr(module, 'load', lambda path: resources)
    monkeypatch.setattr(module.subprocess, 'Popen', spawn)
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(module, 'nic_counters', lambda: {'port': 8})
    def query(argv, **kwargs):
        nvidia.append(argv)
        return SimpleNamespace(returncode=0, stdout='fake GPU evidence', stderr='')
    monkeypatch.setattr(module.subprocess, 'run', query)
    native_save = module.save
    def save(path, value):
        if status_fails and path.name.endswith('status.json'): raise OSError('fake full evidence disk')
        native_save(path, value)
    monkeypatch.setattr(module, 'save', save)
    if archive_fails:
        monkeypatch.setattr(module, 'archive', lambda *a: (_ for _ in ()).throw(OSError('fake archive failure')))
    code = module.run_parent(args.task_root, args.output_dir, args.cloud_root, args.worker_tmp_root,
                             args, fake_collector())
    assert code == next((value for value in codes if value != 0), 0)
    assert len(spawned) == 2 and all(child.polls == 2 for child in children)
    assert not list(args.cloud_root.rglob('*cache*'))
    if not status_fails:
        status = json.loads((args.output_dir/'status.json').read_text())
        assert [row['native_exit'] for row in status['cases']] == list(codes)
        assert status['native_exit'] == code and status['status'] == 'raw_unadmitted'
        assert status['evidence_status'] == ('UNVERIFIED' if observer_fails or archive_fails else 'recorded_not_validated')
        assert bool(status['evidence_errors']) == (observer_fails or archive_fails)
    log = capsys.readouterr().out
    assert 'ATTENTION_CHILD_EXIT' in log
    if observer_fails or archive_fails or status_fails:
        assert 'CAPACITY_EVIDENCE_UNVERIFIED' in log
    if not archive_fails:
        mapping = json.loads((args.output_dir/'archive_manifest.json').read_text())
        assert len(mapping['files']) == 7  # Two observations and five completed log files.
        assert all(Path(row['cloud_path']).is_relative_to(args.cloud_root) for row in mapping['files'])
        assert all(Path(row['worker_path']).is_relative_to(args.worker_tmp_root) for row in mapping['files'])
        assert (args.cloud_root/'swa'/'shape'/'observation.json').read_bytes() == (
            args.worker_tmp_root/'observations'/'swa'/'shape'/'observation.json').read_bytes()


def install_fake_modules(monkeypatch, exports):
    modules = {}
    for name, attributes in exports.items():
        parts = name.split('.')
        for end in range(1, len(parts)+1):
            key = '.'.join(parts[:end])
            if key not in modules:
                fake = ModuleType(key); fake.__path__ = []; modules[key] = fake
                monkeypatch.setitem(sys.modules, key, fake)
                if end > 1:
                    setattr(modules['.'.join(parts[:end-1])], parts[end-1], fake)
        modules[name].__dict__.update(attributes)
    return modules


def fake_runtime(inputs, monkeypatch, kind, failure=None):
    module, args = inputs
    events = []
    write_json(args.task_root/'local_scope_sampling_01.json', dict(status='fake plan'))
    text = SimpleNamespace(num_hidden_layers=92, layer_types_with_mtp=tuple(range(95)), moe_num_experts=352)
    native = SimpleNamespace(model_config=SimpleNamespace(hf_config=text, dtype='fake BF16'),
        quant_config=SimpleNamespace(ignored_layers=tuple(range(1406))),
        compilation_config=SimpleNamespace(cudagraph_mode='native unchanged'))
    config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=text),
                             with_hf_config=lambda hf: native)
    class Args:
        def __init__(self, **kwargs): events.append(('engine_args', kwargs))
        def create_engine_config(self): return config
    torch = dict(cuda=SimpleNamespace(device_count=lambda: 1, set_device=lambda index: events.append(('device', index)),
        get_device_properties=lambda index: SimpleNamespace(uuid='GPU-fake')),
        set_default_dtype=lambda value: events.append(('dtype', value)), device=lambda name: nullcontext(),
        no_grad=lambda: nullcontext())
    install_fake_modules(monkeypatch, {
        'torch': torch, 'vllm.config': {'set_current_vllm_config': lambda value: nullcontext()},
        'vllm.distributed.parallel_state': {
            'init_distributed_environment': lambda **kw: events.append(('distributed', kw)),
            'initialize_model_parallel': lambda *a: events.append(('parallel', a))},
        'vllm.engine.arg_utils': {'EngineArgs': Args},
        'vllm.model_executor.model_loader.utils': {'configure_quant_config': lambda *a: events.append(('quant', a))},
        'vllm.models.step5.nvidia.model': {'Step5ForCausalLM': object},
        'vllm.v1.worker.workspace': {'init_workspace_manager': lambda device: events.append(('workspace',))}})
    identity = dict(hardware='NVIDIA H800', torch='fake torch', triton='fake triton', vllm='fake vllm',
                    native_python='fake python', kernel_driver='570.124.06', user_driver='compat libcuda')
    environment = dict(device_name=identity['hardware'], torch=identity['torch'], triton=identity['triton'],
        vllm=identity['vllm'], python=identity['native_python'], kernel_module_label=identity['kernel_driver'],
        mapped_libcuda=[identity['user_driver']])
    collector = fake_collector()
    collector.verify_installed = lambda files: events.append(('installed', files))
    collector.dump_json = write_json
    def record(output):
        collector.dump_json(output/'environment.json', environment)
        return environment
    collector.record_environment = record
    def measure(task, output, actual_native, actual_kind, **kwargs):
        events.append(('measure', task, output, actual_native, actual_kind, kwargs))
        assert actual_kind == kind and actual_native is native
        if failure is not None:
            raise failure
        output.mkdir(parents=True)
        return [dict(kind=actual_kind, sample=index) for index in range(30)]
    attention = SimpleNamespace(measure=measure, __file__=str(ROOT/'local_attention_measurement.py'))
    monkeypatch.setattr(module, 'verify_frozen', lambda value: events.append(('freeze', value)))
    monkeypatch.setattr(module, 'loaded_sources', lambda files, c: [dict(module='fake actually loaded native')])
    monkeypatch.setattr(module, 'loaded_extensions', lambda c: dict(modules=['fake extension']))
    monkeypatch.setattr(module, 'nic_counters', lambda: {'fake port': 4})
    args.worker_tmp_root.mkdir(parents=True)
    return events, collector, attention, dict(identity=identity), environment


@pytest.mark.parametrize('kind', ['swa', 'dsa'])
def test_native_caller_initializes_unchanged_g1_and_calls_actual_kind(inputs, monkeypatch, kind):
    module, args = inputs
    events, collector, attention, plan, _ = fake_runtime(inputs, monkeypatch, kind)
    output = args.output_dir/kind
    module.execute(args.task_root, output, args.worker_tmp_root, kind, args, collector, attention, [], plan)
    engine = next(row[1] for row in events if row[0] == 'engine_args')
    assert engine == dict(model=str(args.task_root/'reference_inputs'), load_format='dummy', skip_tokenizer_init=True,
        enable_prefix_caching=False, enable_chunked_prefill=True, max_num_seqs=4,
        max_num_batched_tokens=512, max_model_len=8192)
    distributed = next(row[1] for row in events if row[0] == 'distributed')
    assert distributed == dict(world_size=1, rank=0, local_rank=0,
        distributed_init_method='file://'+str(args.worker_tmp_root/('distributed_'+kind)), backend='nccl')
    measure = next(row for row in events if row[0] == 'measure')
    assert measure[2] == args.worker_tmp_root/'observations'/kind
    assert measure[5] == dict(warmups=3, repeats=5, tensor_copy_bytes=67108864)
    status = json.loads((output/'native_status.json').read_text())
    assert status['measurement_returned'] and status['workload_error'] is None and status['trust_error'] is None
    observations = json.loads((output/'measurements.json').read_text())
    assert len(observations['rows']) == 30 and observations['status'] == 'raw_unadmitted'


@pytest.mark.parametrize('native_fails,observer_fails', [(True, False), (True, True), (False, True)])
def test_native_exception_is_not_replaced_by_auxiliary_evidence_failure(inputs, monkeypatch, native_fails, observer_fails):
    module, args = inputs
    original = RuntimeError('raw native error') if native_fails else None
    _, collector, attention, plan, _ = fake_runtime(inputs, monkeypatch, 'swa', original)
    if observer_fails:
        monkeypatch.setattr(module, 'loaded_extensions', lambda *a: (_ for _ in ()).throw(OSError('fake identity observation failed')))
        original_save = module.save
        def save(path, value):
            if path.name in ('device.json', 'failure.json'): raise OSError('fake evidence disk full')
            original_save(path, value)
        monkeypatch.setattr(module, 'save', save)
    output = args.output_dir/'swa'
    if native_fails:
        with pytest.raises(RuntimeError) as caught:
            module.execute(args.task_root, output, args.worker_tmp_root, 'swa', args, collector, attention, [], plan)
        assert caught.value is original
    else:
        module.execute(args.task_root, output, args.worker_tmp_root, 'swa', args, collector, attention, [], plan)
    status = json.loads((output/'native_status.json').read_text())
    assert status['measurement_returned'] == (not native_fails)
    assert (status['workload_error'] is not None) == native_fails
    assert bool(status['evidence_errors']) == observer_fails
    if native_fails:
        assert 'raw native error' in status['workload_error']['traceback']


@pytest.mark.parametrize('field', ['device_name', 'torch', 'triton', 'vllm', 'python', 'kernel_module_label', 'mapped_libcuda'])
def test_actual_runtime_identity_drift_rejects_before_measurement(inputs, monkeypatch, field):
    module, args = inputs
    events, collector, attention, plan, environment = fake_runtime(inputs, monkeypatch, 'swa')
    environment[field] = ['wrong libcuda'] if field == 'mapped_libcuda' else 'wrong identity'
    output = args.output_dir/'swa'
    with pytest.raises(ValueError, match='drift'):
        module.execute(args.task_root, output, args.worker_tmp_root, 'swa', args, collector, attention, [], plan)
    assert not any(row[0] == 'measure' for row in events)
    status = json.loads((output/'native_status.json').read_text())
    assert not status['measurement_returned'] and status['trust_error'] is not None


@pytest.mark.parametrize('fault', ['path', 'bytes', 'none'])
def test_loaded_native_source_identity_uses_actual_module_file(inputs, monkeypatch, fault):
    module, args = inputs
    root = args.worker_tmp_root/'package'
    native = root/'vllm'/'fixture.py'
    native.parent.mkdir(parents=True); native.write_text('frozen native\n')
    expected = dict(relative_path='vllm/fixture.py', source_sha256=module.sha256(native), source_size_bytes=native.stat().st_size)
    loaded = ModuleType('vllm.fixture'); loaded.__file__ = str(native)
    if fault == 'path':
        foreign = root/'foreign.py'; foreign.write_bytes(native.read_bytes()); loaded.__file__ = str(foreign)
    elif fault == 'bytes':
        native.write_text('changed native')
    monkeypatch.setitem(sys.modules, 'vllm.fixture', loaded)
    monkeypatch.setattr(module.importlib.util, 'find_spec', lambda name: SimpleNamespace(origin=str(root/'vllm'/'__init__.py')))
    if fault == 'none':
        rows = module.loaded_sources([expected], fake_collector())
        assert rows == [dict(module='vllm.fixture', path=str(native), sha256=expected['source_sha256'],
                             size_bytes=expected['source_size_bytes'], verification='matched_frozen_export')]
    else:
        with pytest.raises(ValueError, match='Loaded native source'):
            module.loaded_sources([expected], fake_collector())


def test_environment_evidence_disk_failure_does_not_block_native_measurement(inputs, monkeypatch):
    module, args = inputs
    events, collector, attention, plan, _ = fake_runtime(inputs, monkeypatch, 'swa')
    def full_disk(path, value):
        raise OSError('fake environment diagnostic disk full')
    collector.dump_json = full_disk
    output = args.output_dir/'swa'
    module.execute(args.task_root, output, args.worker_tmp_root, 'swa', args, collector, attention, [], plan)
    status = json.loads((output/'native_status.json').read_text())
    assert status['measurement_returned'] and status['workload_error'] is None
    assert any(row['evidence'] == 'environment_write' for row in status['evidence_errors'])
    assert collector.dump_json is full_disk
    assert any(row[0] == 'measure' for row in events)


def test_after_measurement_source_drift_is_not_reported_as_native_failure_or_success(inputs, monkeypatch):
    module, args = inputs
    _, collector, attention, plan, _ = fake_runtime(inputs, monkeypatch, 'dsa')
    calls = []
    def frozen(value):
        calls.append(value)
        if len(calls) == 2:
            raise ValueError('source drift after native measurement')
    monkeypatch.setattr(module, 'verify_frozen', frozen)
    output = args.output_dir/'dsa'
    with pytest.raises(ValueError, match='source drift'):
        module.execute(args.task_root, output, args.worker_tmp_root, 'dsa', args, collector, attention, [], plan)
    status = json.loads((output/'native_status.json').read_text())
    assert status['measurement_returned'] and status['workload_error'] is None
    assert status['trust_error']['stage'] == 'source_verification_after'
    assert status['status'] == 'raw_unadmitted'


def test_loaded_identity_diagnostics_include_actual_package_extension_and_library_bytes(inputs, monkeypatch):
    module, args = inputs
    root = args.worker_tmp_root
    root.mkdir(parents=True)
    package = root/'torch.py'; package.write_bytes(b'fake package')
    extension = root/'native_extension.so'; extension.write_bytes(b'fake extension')
    library = root/'libcuda.so.580.82.07'; library.write_bytes(b'fake libcuda')
    torch = ModuleType('torch'); torch.__file__ = str(package)
    native = ModuleType('vllm._test_extension'); native.__file__ = str(extension)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'vllm._test_extension', native)
    original_read = Path.read_text
    def read(path, *args, **kwargs):
        if str(path) == '/proc/self/maps':
            return '00-01 r-xp 00000000 00:01 1 '+str(library)+'\n'
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    observed = module.loaded_extensions(fake_collector())
    rows = {row['module']: row for row in observed['modules']}
    assert rows['torch']['path'] == str(package)
    assert rows['vllm._test_extension']['sha256'] == hashlib.sha256(b'fake extension').hexdigest()
    assert observed['mapped_gpu_libraries'] == [dict(path=str(library), size_bytes=12,
        sha256=hashlib.sha256(b'fake libcuda').hexdigest())]


@pytest.mark.parametrize('mode,kind', [('child', None), ('run', 'swa'), ('submit', 'dsa')])
def test_invalid_kind_mode_is_rejected_before_loading(inputs, monkeypatch, mode, kind):
    module, args = inputs
    monkeypatch.setattr(module, 'prepare', lambda *a: pytest.fail('Invalid mode must not import runtime tooling'))
    argv = ['driver', *module.invocation_argv(mode, args.task_root, args.output_dir,
        args.cloud_root, args.worker_tmp_root, args, kind)[1:]]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 2


def test_nic_counter_units_and_missing_counters_remain_raw_diagnostics(monkeypatch, tmp_path):
    module = driver()
    counters = tmp_path/'counters'
    counters.mkdir()
    (counters/'port_xmit_data').write_text('17\n')
    (counters/'port_rcv_data').write_text('0\n')
    original = Path.glob
    def glob(path, pattern):
        if str(path) == '/sys/class/infiniband':
            return counters.glob('port_*_data')
        return original(path, pattern)
    monkeypatch.setattr(Path, 'glob', glob)
    assert module.nic_counters() == {str(counters/'port_rcv_data'): 0, str(counters/'port_xmit_data'): 17}
    # Missing sysfs counter files are not proof of no NIC traffic.
    monkeypatch.setattr(Path, 'glob', lambda *a: iter(()))
    assert module.nic_counters() == {}


def test_loaded_package_init_is_checked_using_actual_package_module(inputs, monkeypatch):
    module, args = inputs
    root = args.worker_tmp_root/'package'
    package = root/'vllm'/'config'/'__init__.py'
    package.parent.mkdir(parents=True); package.write_text('fake package init\n')
    loaded = ModuleType('vllm.config'); loaded.__file__ = str(package)
    monkeypatch.setitem(sys.modules, 'vllm.config', loaded)
    monkeypatch.setattr(module.importlib.util, 'find_spec', lambda name: SimpleNamespace(origin=str(root/'vllm'/'__init__.py')))
    expected = dict(relative_path='vllm/config/__init__.py', source_sha256=module.sha256(package),
                    source_size_bytes=package.stat().st_size)
    rows = module.loaded_sources([expected], fake_collector())
    assert rows[0]['module'] == 'vllm.config' and rows[0]['verification'] == 'matched_frozen_export'
    package.write_text('changed package init\n')
    with pytest.raises(ValueError, match='Loaded native source'):
        module.loaded_sources([expected], fake_collector())


@pytest.fixture(autouse=True)
def no_real_child_or_command(monkeypatch):
    imported = builtins.__import__
    def fake_native_only(name, *a, **kw):
        if name.split('.')[0] in ('torch', 'vllm', 'triton', 'steptron', 'stepmind') and name not in sys.modules:
            pytest.fail('Real native or SDK import prohibited: '+name)
        return imported(name, *a, **kw)
    monkeypatch.setattr(builtins, '__import__', fake_native_only)
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('Real child prohibited'))
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('Real command prohibited'))


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '1024'])
def test_generation_evidence_cap_requires_positive_integer(inputs, value):
    module, args = inputs
    args.evidence_bytes = value
    with pytest.raises(ValueError, match='evidence'):
        module.validate_paths(args)


def test_generation_evidence_cap_is_required_and_propagated(inputs, monkeypatch):
    module, args = inputs
    for mode, kind in [('submit', None), ('run', None), ('child', 'swa')]:
        argv = module.invocation_argv(mode, args.task_root, args.output_dir,
            args.cloud_root, args.worker_tmp_root, args, kind)
        assert argv[argv.index('--evidence-bytes')+1] == str(args.evidence_bytes)
    argv = module.invocation_argv('run', args.task_root, args.output_dir,
        args.cloud_root, args.worker_tmp_root, args)[1:]
    del argv[argv.index('--evidence-bytes'):argv.index('--evidence-bytes')+2]
    monkeypatch.setattr(sys, 'argv', ['driver', *argv])
    monkeypatch.setattr(module, 'prepare', lambda *a: pytest.fail('Missing cap reached preparation'))
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 2


def fake_parent(inputs, monkeypatch, payload_bytes=32, second_failure=None, escape=None, failure_kind='dsa'):
    module, args = inputs
    calls, children = [], []
    original_open = Path.open
    if second_failure == 'log_open':
        def open_log(path, *a, **kw):
            if path == args.worker_tmp_root/'logs'/(failure_kind+'.stderr'):
                raise OSError('fake second log open failed')
            return original_open(path, *a, **kw)
        monkeypatch.setattr(Path, 'open', open_log)
    def spawn(argv, stdout, stderr, env):
        kind = argv[argv.index('--kind')+1]
        calls.append(kind)
        if kind == failure_kind and second_failure == 'popen':
            raise OSError('fake second Popen failed')
        if children:
            assert children[-1].polls == 2
        stdout.write(b'original stdout\n'); stderr.write(b'original stderr\n')
        cell = args.worker_tmp_root/'observations'/kind
        cell.mkdir(parents=True)
        (cell/'tensor.bin').write_bytes(b'R'*payload_bytes)
        if escape is not None and kind == 'swa':
            (cell/'escape').symlink_to(escape, target_is_directory=escape.is_dir())
        diagnostics = args.output_dir/kind
        diagnostics.mkdir()
        (diagnostics/'native_status.json').write_bytes(b'original child diagnostic\n')
        cache = args.worker_tmp_root/'cache'/'ignored.bin'
        cache.write_bytes(b'C'*16384)
        child = FakeChild(7 if kind == 'swa' else 0, 900+len(children))
        children.append(child)
        return child
    monkeypatch.setattr(module, 'load', lambda path: SimpleNamespace(
        process_tree=lambda pid: [], cgroup_memory=lambda: {}, SAMPLE_SECONDS=1))
    monkeypatch.setattr(module.subprocess, 'Popen', spawn)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw:
        SimpleNamespace(returncode=0, stdout='fake GPU', stderr=''))
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(module, 'nic_counters', lambda: {})
    return calls, children


def evidence_size(output, scratch):
    return sum(path.stat().st_size for root in (output, scratch/'observations', scratch/'logs')
               for path in root.rglob('*') if path.is_file())


@pytest.mark.parametrize('limit', [1, 2400])
def test_generation_cap_checks_whole_generation_before_each_tree_copy(inputs, monkeypatch, limit):
    module, args = inputs
    args.evidence_bytes = limit
    calls, _ = fake_parent(inputs, monkeypatch, payload_bytes=1024)
    copied, original_copytree = [], module.shutil.copytree
    def checked_copy(source, destination, *a, **kw):
        usage = evidence_size(args.output_dir, args.worker_tmp_root)
        archived = sum(path.stat().st_size for path in args.cloud_root.rglob('*') if path.is_file())
        source_bytes = sum(path.stat().st_size for path in source.rglob('*') if path.is_file())
        assert usage <= limit, 'Over-budget entire generation was copied'
        assert archived+source_bytes <= limit, 'Cumulative archived bytes exceeded generation cap'
        copied.append(source)
        return original_copytree(source, destination, *a, **kw)
    monkeypatch.setattr(module.shutil, 'copytree', checked_copy)
    assert module.run_parent(args.task_root, args.output_dir, args.cloud_root, args.worker_tmp_root,
                             args, fake_collector()) == 7
    assert calls == ['swa', 'dsa']
    status = json.loads((args.output_dir/'status.json').read_text())
    assert status['native_exit'] == 7 and status['evidence_status'] == 'UNVERIFIED'
    assert status['evidence_usage_bytes'] == evidence_size(args.output_dir, args.worker_tmp_root)
    assert status['evidence_usage_bytes'] > limit
    assert any('evidence byte cap' in row['error'] for row in status['evidence_errors'])
    assert len(copied) == (0 if limit == 1 else 1)
    assert (args.worker_tmp_root/'observations'/'dsa'/'tensor.bin').read_bytes() == b'R'*1024


def test_generation_cap_counts_diagnostics_but_excludes_caches(inputs, monkeypatch):
    module, args = inputs
    args.evidence_bytes = 8192
    fake_parent(inputs, monkeypatch)
    assert module.run_parent(args.task_root, args.output_dir, args.cloud_root, args.worker_tmp_root,
                             args, fake_collector()) == 7
    status = json.loads((args.output_dir/'status.json').read_text())
    assert status['evidence_status'] == 'recorded_not_validated'
    manifest = json.loads((args.output_dir/'archive_manifest.json').read_text())
    assert manifest['evidence_bytes'] == args.evidence_bytes
    assert manifest['archived_bytes'] == sum(row['size_bytes'] for row in manifest['files'])
    assert status['evidence_usage_bytes'] == evidence_size(args.output_dir, args.worker_tmp_root)
    assert status['evidence_usage_bytes'] < args.evidence_bytes
    assert (args.cloud_root/'diagnostics'/'swa.stdout').read_bytes() == b'original stdout\n'
    assert not list(args.cloud_root.rglob('*cache*'))


@pytest.mark.parametrize('directory', [False, True])
def test_archive_rejects_symlink_escape_without_reading_or_copying_target(inputs, monkeypatch, directory):
    module, args = inputs
    outside = args.task_root.parent/'outside'
    if directory:
        outside.mkdir(); target = outside/'secret.bin'
    else:
        target = outside
    target.write_bytes(b'outside tree bytes')
    fake_parent(inputs, monkeypatch, escape=outside)
    original_open = Path.open
    def no_external_read(path, *a, **kw):
        if path.resolve() == target:
            pytest.fail('Archive read an escaped symlink target')
        return original_open(path, *a, **kw)
    monkeypatch.setattr(Path, 'open', no_external_read)
    assert module.run_parent(args.task_root, args.output_dir, args.cloud_root, args.worker_tmp_root,
                             args, fake_collector()) == 7
    assert not (args.cloud_root/'swa').exists()
    assert target.stat().st_size == len(b'outside tree bytes')
    status = json.loads((args.output_dir/'status.json').read_text())
    assert status['evidence_status'] == 'UNVERIFIED'
    assert any('symlink' in row['error'].lower() for row in status['evidence_errors'])


@pytest.mark.parametrize('fault', ['installed', 'torch', 'vllm'])
def test_early_child_verification_and_import_failure_write_structured_original_failure(inputs, monkeypatch, fault):
    module, args = inputs
    events, collector, attention, plan, _ = fake_runtime(inputs, monkeypatch, 'swa')
    original = SystemExit(23) if fault == 'installed' else ImportError('fake '+fault+' import failed')
    if fault == 'installed':
        collector.verify_installed = lambda files: (_ for _ in ()).throw(original)
    else:
        imported = builtins.__import__
        def fail_import(name, *a, **kw):
            if name == fault or (fault == 'vllm' and name.startswith('vllm.')):
                raise original
            return imported(name, *a, **kw)
        monkeypatch.setattr(builtins, '__import__', fail_import)
    # Finalization must not rely on a successfully imported runtime.
    monkeypatch.setattr(module, 'loaded_sources', lambda *a: (_ for _ in ()).throw(ImportError('no runtime')))
    monkeypatch.setattr(module, 'loaded_extensions', lambda *a: (_ for _ in ()).throw(ImportError('no runtime')))
    output = args.output_dir/'swa'
    with pytest.raises(type(original)) as caught:
        module.execute(args.task_root, output, args.worker_tmp_root, 'swa', args, collector, attention, [], plan)
    assert caught.value is original
    assert not any(row[0] == 'measure' for row in events)
    status = json.loads((output/'native_status.json').read_text())
    failure = json.loads((output/'failure.json').read_text())
    assert not status['measurement_returned'] and status['workload_error'] is None
    stage = 'installed_source_verification' if fault == 'installed' else 'runtime_imports'
    assert status['trust_error']['stage'] == failure['trust_error']['stage'] == stage
    assert type(original).__name__ in status['trust_error']['traceback']


def test_closed_stdout_does_not_change_known_exits_archive_or_next_kind(inputs, monkeypatch):
    module, args = inputs
    calls, _ = fake_parent(inputs, monkeypatch)
    reader, writer = os.pipe()
    os.close(reader)
    # A real block-buffered pipe keeps unflushed bytes, as the worker log stream does.
    stdout = open(writer, 'w')
    monkeypatch.setattr(sys, 'stdout', stdout)
    assert module.run_parent(args.task_root, args.output_dir, args.cloud_root, args.worker_tmp_root,
                             args, fake_collector()) == 7
    # CPython exits with 120 instead of the known exit when this final flush fails.
    stdout.close()
    assert calls == ['swa', 'dsa']
    status = json.loads((args.output_dir/'status.json').read_text())
    assert [row['native_exit'] for row in status['cases']] == [7, 0]
    assert status['evidence_status'] == 'UNVERIFIED'
    assert any('BrokenPipeError' in row['error'] for row in status['evidence_errors'])
    assert (args.cloud_root/'swa'/'tensor.bin').read_bytes() == b'R'*32
    assert (args.output_dir/'archive_manifest.json').is_file()


@pytest.mark.parametrize('fault', ['log_open', 'popen'])
def test_second_child_startup_failure_finalizes_completed_evidence_without_fake_exit(inputs, monkeypatch, fault):
    module, args = inputs
    calls, children = fake_parent(inputs, monkeypatch, second_failure=fault)
    with pytest.raises(OSError, match='fake second'):
        module.run_parent(args.task_root, args.output_dir, args.cloud_root, args.worker_tmp_root,
                          args, fake_collector())
    assert calls == (['swa'] if fault == 'log_open' else ['swa', 'dsa'])
    assert len(children) == 1 and children[0].polls == 2
    status = json.loads((args.output_dir/'status.json').read_text())
    assert [row['native_exit'] for row in status['cases']] == [7]
    assert status['native_exit'] == 7
    assert status['startup_error']['kind'] == 'dsa'
    assert status['startup_error']['stage'] == ('child_logs' if fault == 'log_open' else 'child_spawn')
    assert 'OSError' in status['startup_error']['traceback']
    assert (args.cloud_root/'swa'/'tensor.bin').read_bytes() == b'R'*32
    assert (args.cloud_root/'diagnostics'/'swa.stdout').read_bytes() == b'original stdout\n'
    assert (args.cloud_root/'diagnostics'/'dsa.stdout').read_bytes() == b''
    assert (args.output_dir/'archive_manifest.json').is_file()


def test_generation_cap_never_copies_when_tree_usage_cannot_be_read(inputs, monkeypatch):
    module, args = inputs
    fake_parent(inputs, monkeypatch)
    original_scandir = module.os.scandir
    def unreadable(path):
        if Path(path) == args.worker_tmp_root/'observations'/'swa':
            raise PermissionError('fake unreadable raw evidence tree')
        return original_scandir(path)
    monkeypatch.setattr(module.os, 'scandir', unreadable)
    assert module.run_parent(args.task_root, args.output_dir, args.cloud_root, args.worker_tmp_root,
                             args, fake_collector()) == 7
    assert not (args.cloud_root/'diagnostics').exists()
    assert not (args.cloud_root/'dsa').exists()
    status = json.loads((args.output_dir/'status.json').read_text())
    assert status['evidence_status'] == 'UNVERIFIED'
    assert status['evidence_usage_bytes'] is None
    assert any('PermissionError' in row['error'] for row in status['evidence_errors'])


def test_stdout_closing_before_the_final_cap_diagnostic_cannot_change_the_exit(inputs, monkeypatch):
    module, args = inputs
    # No child exit line is printed, so the final cap diagnostic is the first failed stdout write.
    args.evidence_bytes = 1
    fake_parent(inputs, monkeypatch, second_failure='popen', failure_kind='swa')
    reader, writer = os.pipe()
    os.close(reader)
    stdout = open(writer, 'w')
    monkeypatch.setattr(sys, 'stdout', stdout)
    with pytest.raises(OSError, match='fake second Popen failed'):
        module.run_parent(args.task_root, args.output_dir, args.cloud_root, args.worker_tmp_root,
                          args, fake_collector())
    status = json.loads((args.output_dir/'status.json').read_text())
    assert [row['evidence'] for row in status['evidence_errors']] == ['parent_evidence_bytes']
    stdout.close()


@pytest.mark.parametrize('fault', ['log_open', 'popen'])
def test_first_child_startup_failure_has_no_invented_native_exit(inputs, monkeypatch, fault):
    module, args = inputs
    calls, children = fake_parent(inputs, monkeypatch, second_failure=fault, failure_kind='swa')
    with pytest.raises(OSError, match='fake second'):
        module.run_parent(args.task_root, args.output_dir, args.cloud_root, args.worker_tmp_root,
                          args, fake_collector())
    assert calls == ([] if fault == 'log_open' else ['swa']) and children == []
    status = json.loads((args.output_dir/'status.json').read_text())
    assert status['cases'] == [] and status['native_exit'] is None
    assert status['startup_error']['kind'] == 'swa'
    assert status['evidence_status'] == 'UNVERIFIED'
    assert (args.cloud_root/'diagnostics'/'swa.stdout').read_bytes() == b''


def test_cumulative_archive_cap_counts_existing_partial_copy_bytes(inputs, monkeypatch):
    module, args = inputs
    args.evidence_bytes = 2400
    calls, _ = fake_parent(inputs, monkeypatch)
    original_archive = module.archive
    def partial_copy(source, destination, collector):
        if destination.name == 'swa':
            destination.mkdir()
            (destination/'partial.bin').write_bytes(b'P'*2390)
            raise OSError('fake partially copied archive')
        return original_archive(source, destination, collector)
    monkeypatch.setattr(module, 'archive', partial_copy)
    assert module.run_parent(args.task_root, args.output_dir, args.cloud_root, args.worker_tmp_root,
                             args, fake_collector()) == 7
    assert calls == ['swa', 'dsa']
    assert (args.cloud_root/'swa'/'partial.bin').read_bytes() == b'P'*2390
    assert not (args.cloud_root/'dsa').exists() and not (args.cloud_root/'diagnostics').exists()
    manifest = json.loads((args.output_dir/'archive_manifest.json').read_text())
    assert manifest['archived_bytes'] == 2390 and manifest['files'] == []
    status = json.loads((args.output_dir/'status.json').read_text())
    assert status['native_exit'] == 7 and status['evidence_status'] == 'UNVERIFIED'
    assert any('2422 > 2400' in row['error'] for row in status['evidence_errors'])
