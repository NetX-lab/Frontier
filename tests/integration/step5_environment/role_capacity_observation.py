"""Record native full-role lifecycle boundaries without changing execution."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import traceback

STAGES = ('init_device', 'load_model', 'determine_available_memory',
          'initialize_from_config', 'compile_or_warm_up_model')
CLOUD = Path('/mnt/codesign-exp/ycfeng/step5-preview')


def report_error(value):
    try:
        print(value, flush=True)
    except Exception:
        pass


def worker_identity(worker, proc_root=Path('/proc')):
    """Capture identity in the worker's live PID namespace before native startup."""
    stat = (proc_root/'self/stat').read_text()
    pid = os.getpid()
    if int(stat.split(' ', 1)[0]) != pid:
        raise ValueError('Worker /proc identity does not match os.getpid()')
    start = int(stat.rsplit(')', 1)[1].split()[19])
    parent = os.getppid()
    parent_stat = (proc_root/str(parent)/'stat').read_text()
    row = dict(pid=pid, start_time=start, rank=worker.rank, local_rank=worker.local_rank,
               parent_pid=parent, parent_start_time=int(parent_stat.rsplit(')', 1)[1].split()[19]),
               boot_id=(proc_root/'sys/kernel/random/boot_id').read_text().strip(),
               pid_namespace=os.readlink(proc_root/'self/ns/pid'))
    node_file = os.environ.get('STEP5_CAPACITY_NODE_FILE')
    if node_file is not None:
        node = json.loads(Path(node_file).read_text())
        if (row['boot_id'], row['pid_namespace']) != (node['boot_id'], node['pid_namespace']):
            raise ValueError('Worker and supervisor must share boot and PID namespace')
        if row['rank'] != node['node_rank']*8+row['local_rank'] or not 0 <= row['local_rank'] < 8:
            raise ValueError('Worker rank is outside its allocated node')
        row.update(generation=node['generation'], node_rank=node['node_rank'],
                   observer_sha256=node['observer_sha256'])
    return row


def identity_record(worker):
    """Publish a small local record before imports, CUDA setup, or initialization."""
    row = worker_identity(worker)
    worker._capacity_identity = row
    worker._capacity_completed_stages = []
    worker._capacity_evidence_errors = []
    if 'STEP5_CAPACITY_NODE_FILE' not in os.environ:
        return
    node = json.loads(Path(os.environ['STEP5_CAPACITY_NODE_FILE']).read_text())
    output = Path(node['observer_output'])
    fd, temporary = tempfile.mkstemp(dir=output, prefix='.', suffix='.json')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(row, stream); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, output/f"worker_{row['rank']}_{row['pid']}.json")
    finally:
        os.unlink(temporary)


def observation_output():
    node_file = os.environ.get('STEP5_CAPACITY_NODE_FILE')
    if node_file is not None:
        node = json.loads(Path(node_file).read_text())
        output = Path(node['observer_output']).resolve()
        if not output.is_dir() or output.is_relative_to(Path('/mnt/codesign-exp')):
            raise ValueError('TP16 lifecycle observations require existing worker-local storage')
        return output, node['record_bytes']
    output = Path(os.environ['STEP5_CAPACITY_OUTPUT']).resolve()
    assert output.is_relative_to(CLOUD) and output != CLOUD
    assert os.path.ismount('/mnt/codesign-exp') and output.is_dir()
    return output, 64*1024**2


def write_record(worker, phase, boundary, error=None):
    """Resource observation failures must not replace native returns or errors."""
    try:
        output, record_bytes = observation_output()
        import torch
        row = dict(phase=phase, boundary=boundary, native_error=error, pid=os.getpid(),
                   rank=worker.rank, local_rank=worker.local_rank, time_ns=time.time_ns())
        row.update(getattr(worker, '_capacity_identity', {}))
        row['rss_kib'] = {k: int(line.split()[1]) for line in Path('/proc/self/status').read_text().splitlines()
                          for k in ('VmRSS', 'VmHWM') if line.startswith(k+':')}
        if torch.cuda.is_initialized() and hasattr(worker, 'device'):
            row['cuda'] = dict(allocated=torch.cuda.memory_allocated(worker.device),
                reserved=torch.cuda.memory_reserved(worker.device),
                allocator_peak_since_last_native_reset_allocated=torch.cuda.max_memory_allocated(worker.device),
                allocator_peak_since_last_native_reset_reserved=torch.cuda.max_memory_reserved(worker.device),
                total=torch.cuda.get_device_properties(worker.device).total_memory)
            row['mapped_libcuda'] = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                           if '/libcuda.so' in line})
            row['kernel_driver'] = Path('/proc/driver/nvidia/version').read_text()
            row['device'] = str(worker.device)
            row['gpu_uuid'] = str(torch.cuda.get_device_properties(worker.device).uuid)
        row['observer_sha256'] = os.environ['STEP5_CAPACITY_OBSERVER_SHA']
        config = worker.vllm_config
        row['parallel_config'] = {n: getattr(config.parallel_config, n) for n in
            ('tensor_parallel_size', 'data_parallel_size', 'pipeline_parallel_size', 'enable_expert_parallel')}
        row['configuration'] = dict(model_dtype=str(config.model_config.dtype),
            gpu_memory_utilization=config.cache_config.gpu_memory_utilization,
            speculative=config.speculative_config is not None,
            prefix_caching=config.cache_config.enable_prefix_caching,
            graph_mode=str(config.compilation_config.cudagraph_mode),
            capture_sizes=config.compilation_config.cudagraph_capture_sizes)
        row['memory_accounting'] = {n: getattr(worker, n) for n in
            ('requested_memory', 'available_kv_cache_memory_bytes', 'total_consumed',
             'peak_activation_memory', 'cudagraph_memory_estimate') if hasattr(worker, n)}
        runner = getattr(worker, 'model_runner', None)
        if runner is not None:
            row['runner'] = type(runner).__module__+'.'+type(runner).__qualname__
            row['model_memory_usage'] = getattr(runner, 'model_memory_usage', None)
            if boundary == 'after' and phase in ('load_model', 'initialize_from_config', 'compile_or_warm_up_model'):
                row['inventory'] = inventory(runner, torch)
            if error is not None:
                row['inventory_status'] = 'Partial native construction is not retained; do not infer missing tensors'
        path = output/f'rank_{worker.rank}_pid_{os.getpid()}.jsonl'
        payload = json.dumps(row)+'\n'
        rank_limit = record_bytes//8 if 'STEP5_CAPACITY_NODE_FILE' in os.environ else record_bytes
        used = path.stat().st_size if path.exists() else 0
        if 'STEP5_CAPACITY_NODE_FILE' not in os.environ:
            used = sum(p.stat().st_size for p in output.glob('rank_*.jsonl'))
        assert used+len(payload.encode()) <= rank_limit, 'Rank evidence exceeds record budget'
        with path.open('a') as stream:
            stream.write(payload)
    except Exception:
        failure = traceback.format_exc()
        if hasattr(worker, '_capacity_evidence_errors'):
            worker._capacity_evidence_errors.append(dict(phase=phase, boundary=boundary, error=failure))
        report_error(json.dumps(dict(phase=phase, boundary=boundary, resource_evidence_verified=False,
                                     observer_error=failure)))


def inventory(runner, torch):
    """Metadata only: no state_dict, content copies, GPU checks or extra kernels."""
    model = getattr(runner, 'model', None)
    if model is None:
        return dict(status='model_not_observable')
    tensors, modules, storages = [], [], {}
    for name, module in model.named_modules():
        record = dict(name=name, cls=type(module).__module__+'.'+type(module).__qualname__)
        if hasattr(module, 'quant_method'):
            method = module.quant_method
            record['quant_method'] = type(method).__name__
            record['backend'] = str(getattr(method, 'fp8_backend', getattr(method, 'unquantized_backend', None)))
        for attr in ('num_heads', 'num_kv_heads', 'total_num_heads', 'total_num_kv_heads',
                     'num_local_experts', 'num_experts', 'n_routed_experts'):
            if hasattr(module, attr):
                record[attr] = getattr(module, attr)
        if hasattr(module, 'moe_config'):
            parallel = module.moe_config.moe_parallel_config
            record['moe_parallel'] = {n: getattr(parallel, n) for n in ('tp_size', 'ep_size', 'use_ep')}
        modules.append(record)
        for kind, values in (('parameter', module._parameters), ('buffer', module._buffers)):
            for local, value in values.items():
                if not isinstance(value, torch.Tensor):
                    continue
                storage = value.untyped_storage()
                key = f'{value.device}:{storage.data_ptr()}'
                storages[key] = storage.nbytes()
                tensors.append(dict(name=name+'.'+local, kind=kind, shape=list(value.shape),
                    dtype=str(value.dtype), device=str(value.device), stride=list(value.stride()),
                    storage=key, storage_bytes=storage.nbytes(), storage_offset=value.storage_offset(),
                    persistent=kind == 'parameter' or local not in module._non_persistent_buffers_set))
    return dict(status='metadata_only_not_finite_or_content_qualification', modules=modules,
                tensors=tensors, unique_storage_bytes=sum(storages.values()), unique_storages=storages)


def capacity_snapshot(worker):
    """Return small census and identity control records through the supported RPC API."""
    try:
        import torch
        row = dict(worker._capacity_identity)
        if worker_identity(worker) != row:
            raise ValueError('Worker identity changed after initialization')
        row.update(gpu_uuid=str(torch.cuda.get_device_properties(worker.device).uuid),
                   completed_stages=list(worker._capacity_completed_stages))
        config = worker.vllm_config
        row['parallel_config'] = {n: getattr(config.parallel_config, n) for n in
            ('tensor_parallel_size', 'data_parallel_size', 'pipeline_parallel_size',
             'enable_expert_parallel', 'nnodes', 'node_rank')}
        row['configuration'] = dict(model_dtype=str(config.model_config.dtype),
            gpu_memory_utilization=config.cache_config.gpu_memory_utilization,
            speculative=config.speculative_config is not None,
            prefix_caching=config.cache_config.enable_prefix_caching)
        model = worker.model_runner.model
        classes, experts = {}, []
        for name, module in model.named_modules():
            cls = type(module).__module__+'.'+type(module).__qualname__
            classes[cls] = classes.get(cls, 0)+1
            if hasattr(module, 'n_routed_experts'):
                experts.append(dict(name=name, count=module.n_routed_experts))
        row.update(status='metadata_only_not_G2', module_classes=classes, routed_experts=experts,
                   model_memory_usage=worker.model_runner.model_memory_usage,
                   resource_evidence_errors=list(getattr(worker, '_capacity_evidence_errors', [])))
        return row
    except Exception:
        return dict(status='UNVERIFIED', observer_error=traceback.format_exc(), pid=os.getpid(),
                    rank=worker.rank)


def observed_worker(base, record=write_record):
    """Use a supported worker subclass; call each inherited native method once."""
    def observe(name):
        original = getattr(base, name)
        def call(self, *args, **kwargs):
            def boundary(where, error=None):
                try:
                    record(self, name, where, error)
                except Exception:
                    report_error('CAPACITY_OBSERVER_UNVERIFIED '+name+' '+where+' '+traceback.format_exc())
            if name == 'init_device':
                try:
                    identity_record(self)
                except Exception:
                    report_error('CAPACITY_IDENTITY_UNVERIFIED '+traceback.format_exc())
            boundary('before')
            try:
                result = original(self, *args, **kwargs)
            except BaseException:
                boundary('error', traceback.format_exc())
                raise
            if hasattr(self, '_capacity_completed_stages'):
                try:
                    self._capacity_completed_stages.append(name)
                except Exception:
                    report_error('CAPACITY_STAGES_UNVERIFIED '+traceback.format_exc())
            boundary('after')
            return result
        return call
    methods = {name: observe(name) for name in STAGES}
    methods['capacity_snapshot'] = capacity_snapshot
    return type('ObservedWorker', (base,), methods)


def __getattr__(name):
    if name != 'CapacityWorker':
        raise AttributeError(name)
    task = Path(os.environ['STEP5_CAPACITY_TASK']).resolve()
    assert task == Path(__file__).resolve().parents[3]/'task_memory/task_2026-10-15_step5_preview_h800'
    observation_output()
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == os.environ['STEP5_CAPACITY_OBSERVER_SHA']
    if 'STEP5_CAPACITY_NODE_FILE' in os.environ:
        node = json.loads(Path(os.environ['STEP5_CAPACITY_NODE_FILE']).read_text())
        assert node['observer_sha256'] == os.environ['STEP5_CAPACITY_OBSERVER_SHA']
    from vllm.v1.worker.gpu_worker import Worker
    return observed_worker(Worker)
