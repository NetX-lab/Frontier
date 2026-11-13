"""Record native full-role lifecycle boundaries without changing execution."""
import hashlib
import json
import os
from pathlib import Path
import sys
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
        reserved = 0
        if 'STEP5_CAPACITY_NODE_FILE' in os.environ and hasattr(worker, '_request_observation'):
            from role_execution_observation import pending_profiler_bytes, rank_record_bytes
            state = worker._request_observation
            used = rank_record_bytes(output, worker.rank)
            reserved = state.device_owned_bytes+pending_profiler_bytes(state)
        assert used+len(payload.encode())+reserved <= rank_limit, 'Rank evidence exceeds record budget'
        with path.open('a') as stream:
            stream.write(payload)
    except Exception:
        failure = traceback.format_exc()
        if hasattr(worker, '_capacity_evidence_errors'):
            worker._capacity_evidence_errors.append(dict(phase=phase, boundary=boundary, error=failure))
        report_error(json.dumps(dict(phase=phase, boundary=boundary, resource_evidence_verified=False,
                                     observer_error=failure)))


def register_storage(value, storages):
    """Deduplicate storage references and reject contradictory alias metadata."""
    storage = value.untyped_storage()
    key = f'{value.device}:{storage.data_ptr()}'
    size = storage.nbytes()
    if key in storages and storages[key] != size:
        raise ValueError('Aliased storage metadata has inconsistent sizes')
    storages[key] = size
    return key, size


def storage_record(name, kind, value, storages, *, persistent=None):
    """Use the same physical storage identity for resident and known-state views."""
    key, size = register_storage(value, storages)
    return dict(name=name, kind=kind, shape=list(value.shape), dtype=str(value.dtype),
                device=str(value.device), stride=list(value.stride()), storage=key,
                storage_bytes=size, storage_offset=value.storage_offset(), persistent=persistent)


def module_tensors(name, module, tensor_type):
    """Enumerate actual registered resident tensors for metadata and content observers."""
    for kind, values in (('parameter', module._parameters), ('buffer', module._buffers)):
        for local, value in values.items():
            if isinstance(value, tensor_type):
                yield name+'.'+local, kind, local, value, (
                    kind == 'parameter' or local not in module._non_persistent_buffers_set)


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
        for path, kind, _, value, persistent in module_tensors(name, module, torch.Tensor):
            tensors.append(storage_record(path, kind, value, storages, persistent=persistent))
    return dict(status='metadata_only_not_finite_or_content_qualification', modules=modules,
                tensors=tensors, unique_storage_bytes=sum(storages.values()), unique_storages=storages)


def allocator_observation(runner, torch, known_storages, limit):
    """Read native totals and registered references without claiming allocation ownership."""
    row = dict(status='UNVERIFIED', device=str(getattr(runner, 'device', None)),
        storage_aliases_are_not_additive=True,
        allocation_membership='UNMEASURED', allocation_history='UNMEASURED',
        peaks='SINCE_LAST_NATIVE_RESET_NOT_THIS_SCOPE',
        global_memory='DEVICE_WIDE_NOT_PROCESS_EXCLUSIVE',
        sampling='SEQUENTIAL_HOST_READS_NOT_ATOMIC_OR_GPU_COMPLETION',
        observer_allocations='INCLUDED_IN_NATIVE_TOTALS_NOT_SUBTRACTED', errors=[])
    try:
        row['observation_started_ns'] = time.monotonic_ns()
        if not torch.cuda.is_initialized():
            raise ValueError('CUDA must already be initialized by the native worker')
        if torch.cuda.is_current_stream_capturing():
            raise ValueError('Allocator observations are skipped during native CUDA capture')
        device = str(runner.device)
        if not device.startswith('cuda:') or not device[5:].isdigit():
            raise ValueError('Allocator observation requires the explicit native CUDA device')
        row['backend'] = torch.cuda.memory.get_allocator_backend()
        stats = torch.cuda.memory_stats(runner.device)
        families = ('allocated_bytes', 'reserved_bytes', 'active_bytes',
                    'requested_bytes', 'inactive_split_bytes')
        keys = tuple(f'{family}.all.{metric}' for family in families for metric in ('current', 'peak'))
        row['statistics'] = {key: stats[key] for key in keys if key in stats}
        if any(type(value) is not int or value < 0 for value in row['statistics'].values()):
            raise ValueError('Native allocator statistics must be nonnegative integers')
        missing = [key for key in keys if key not in stats]
        if missing:
            row['errors'].append(dict(reason='MISSING_NATIVE_ALLOCATOR_STATISTICS', keys=missing))
        free, total = torch.cuda.mem_get_info(runner.device)
        if any(type(value) is not int or value < 0 for value in (free, total)) or not 0 <= free <= total or not total:
            raise ValueError('Native device memory totals are invalid')
        row['global_free_bytes'], row['global_total_bytes'] = free, total
        row['global_used_bytes'] = total-free
        storages = dict(known_storages)
        model = getattr(runner, 'model', None)
        entries = 0
        if model is None:
            row['errors'].append(dict(reason='REGISTERED_RESIDENT_MODEL_UNAVAILABLE'))
        else:
            for module_index, (name, module) in enumerate(model.named_modules()):
                if module_index >= limit:
                    raise ValueError('Resident modules exceed the existing entry cap')
                for _, _, _, value, _ in module_tensors(name, module, torch.Tensor):
                    if entries >= limit:
                        raise ValueError('Resident tensor references exceed the existing entry cap')
                    entries += 1
                    register_storage(value, storages)
        if any(type(size) is not int or size < 0 for size in storages.values()):
            raise ValueError('Referenced storage sizes must be nonnegative integers')
        current = {key: size for key, size in storages.items() if key.startswith(device+':')}
        row['known_cuda_storage_bytes'] = sum(current.values())
        row['known_cuda_storages'] = len(current)
        row['other_device_storage_bytes'] = sum(size for key, size in storages.items() if key not in current)
        if row['backend'] != 'native':
            row['errors'].append(dict(reason='ALLOCATOR_STATISTICS_SEMANTICS_UNVERIFIED_FOR_BACKEND'))
        elif not missing:
            allocated = stats['allocated_bytes.all.current']
            reserved = stats['reserved_bytes.all.current']
            row['allocated_minus_known_cuda_storage_bytes'] = allocated-row['known_cuda_storage_bytes']
            row['reserved_minus_allocated_bytes'] = reserved-allocated
            if (allocated > reserved or any(stats[f'{family}.all.peak'] < stats[f'{family}.all.current']
                                           for family in families)):
                row['errors'].append(dict(reason='INCONSISTENT_NATIVE_ALLOCATOR_TOTALS'))
            if row['allocated_minus_known_cuda_storage_bytes'] < 0:
                row['errors'].append(dict(reason='REFERENCED_STORAGES_EXCEED_ALLOCATOR_TOTAL_MEMBERSHIP_UNMEASURED'))
        if not row['errors']:
            row['status'] = 'OBSERVED_NATIVE_TOTALS_AND_STORAGE_REFERENCES_NOT_ALLOCATION_CENSUS'
    except BaseException as failure:
        row['errors'].append(dict(reason='ALLOCATOR_OBSERVATION_FAILED',
            error_type=type(failure).__module__+'.'+type(failure).__qualname__, error=str(failure)))
    try:
        row['observation_finished_ns'] = time.monotonic_ns()
    except BaseException as failure:
        row['status'] = 'UNVERIFIED'
        row['errors'].append(dict(reason='ALLOCATOR_CLOCK_FAILED', error=str(failure)))
    return row


def known_state_inventory(runner, torch, limit):
    """Inspect named existing owners; never traverse arbitrary native resources or allocate."""
    if torch is None:
        return dict(status='UNVERIFIED_ACTUAL_TORCH_UNAVAILABLE', tensors=[], unique_storages={},
                    unique_storage_bytes=0, gaps=['actual Tensor type unavailable'])
    tensors, storages, gaps, groups = [], {}, [], []
    entries = 0

    def claim_entry():
        nonlocal entries
        if entries >= limit:
            raise ValueError('Known-state inventory exceeds the existing entry cap')
        entries += 1

    def add(name, value, kind):
        if value is None:
            return
        claim_entry()
        if isinstance(value, torch.Tensor):
            tensors.append(storage_record(name, kind, value, storages))
        elif value is not None:
            gaps.append(dict(name=name, type=type(value).__module__+'.'+type(value).__qualname__,
                             reason='UNCLASSIFIED_NATIVE_STORAGE_OWNER'))

    def fields(path, owner, names, kind):
        for name in names:
            if hasattr(owner, name):
                add(path+'.'+name, getattr(owner, name), kind)

    config = getattr(runner, 'kv_cache_config', None)
    context = getattr(getattr(runner, 'compilation_config', None), 'static_forward_context', None)
    if config is None or context is None:
        claim_entry()
        gaps.append(dict(name='runner.kv_cache_config/static_forward_context', reason='NATIVE_OWNER_UNAVAILABLE'))
    else:
        for index, group in enumerate(config.kv_cache_groups):
            claim_entry()
            spec = group.kv_cache_spec
            specs = getattr(spec, 'kv_cache_specs', None)
            rows = []
            for name in group.layer_names:
                claim_entry()
                layer_spec = specs[name] if specs is not None else spec
                rows.append(dict(layer=name, spec_type=type(layer_spec).__name__,
                    block_size=layer_spec.block_size, sliding_window=getattr(layer_spec, 'sliding_window', None),
                    dtype=str(getattr(layer_spec, 'dtype', None)),
                    kernel_block_size=runner.kernel_block_sizes[index]))
                layer = context.get(name)
                if layer is None:
                    claim_entry()
                    gaps.append(dict(name=name, reason='KV_LAYER_NOT_IN_FORWARD_CONTEXT'))
                    continue
                fields('forward_context.'+name, layer, ('kv_cache', '_k_scale', '_v_scale'), 'kv_layer_view')
            groups.append(dict(group_id=index, layers=rows))
        for name, layer in context.items():
            cache = getattr(layer, '_cache', None)
            if cache is None or not hasattr(layer, 'main_layer_name'):
                continue
            fields('forward_context.'+name+'._cache', cache,
                ('_mean', 'mean', 'numerator', 'denominator', 'maximum', 'last_slot',
                 'request_journal_values', 'request_journal_last', 'request_journal_tag'), 'sparse_side_storage')
            fields('forward_context.'+name, layer, ('kv_cache',), 'sparse_transfer_alias')
            adapter = getattr(getattr(layer, '_target_impl', None), 'adapter', None)
            if adapter is not None:
                fields('forward_context.'+name+'._target_impl.adapter', adapter, ('_workspace',), 'sparse_workspace')

    for index, value in enumerate(getattr(runner, 'kv_caches', ())):
        add('runner.kv_caches.'+str(index), value, 'kv_allocation_alias')
    fields('runner.model_state', getattr(runner, 'model_state', None), ('request_state',), 'model_request_state')
    tables = getattr(runner, 'block_tables', None)
    if tables is not None:
        for index, value in enumerate(tables.input_block_tables):
            add('runner.block_tables.input_block_tables.'+str(index), value, 'batch_block_table')
        for index, owner in enumerate(tables.block_tables):
            fields('runner.block_tables.block_tables.'+str(index), owner, ('gpu',), 'persistent_block_table')
        fields('runner.block_tables.num_blocks', tables.num_blocks, ('cpu', 'gpu'), 'block_counts')
        fields('runner.block_tables', tables,
            ('slot_mappings', 'dcp_sharded', 'block_table_ptrs', 'block_table_strides',
             'block_sizes_tensor', 'slot_mapping_enabled', 'input_block_table_ptrs'), 'block_table_layout')
    workspace = sys.modules.get('vllm.v1.worker.workspace')
    manager = getattr(workspace, '_manager', None)
    if manager is not None:
        for slot, value in enumerate(manager._current_workspaces):
            add('workspace._manager._current_workspaces.'+str(slot), value, 'shared_workspace')
        for slot, resources in enumerate(manager._persistent_resources):
            for index, (key, value) in enumerate(resources.items()):
                path = f'workspace._manager._persistent_resources.{slot}.{index}'
                add(path, value, 'persistent_workspace_resource')
                if tensors and tensors[-1]['name'] == path:
                    tensors[-1]['resource_key_type'] = type(key).__module__+'.'+type(key).__qualname__
    graph = getattr(runner, 'cudagraph_manager', None)
    pool = getattr(graph, 'pool', None)
    if pool is not None and (type(pool) is not tuple or any(type(part) is not int for part in pool)):
        claim_entry()
        gaps.append(dict(name='runner.cudagraph_manager.pool', reason='UNCLASSIFIED_GRAPH_POOL_HANDLE'))
        pool = None
    return dict(status='KNOWN_OWNER_METADATA_ONLY_NOT_COMPLETE_ALLOCATION_OR_CONTENT_CENSUS',
        tensors=tensors, unique_storages=storages, unique_storage_bytes=sum(storages.values()),
        kv_groups=groups, graph_pool_handle=list(pool) if pool is not None else None,
        gaps=gaps, referenced_live_bytes='UNMEASURED', unclassified_allocator_bytes='UNMEASURED',
        allocator_peaks='UNMEASURED', storage_aliases_are_not_additive=True,
        allocator_occupancy=allocator_observation(runner, torch, storages, limit))


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
