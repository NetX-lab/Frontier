"""Opt-in, soft-failing witnesses of the locked native NIXL pull path.

Content-selected execution is diagnostic, not a timing probe. CUDA calls cannot
be interrupted by this observer: elapsed caps are checked only after they return.
Native-only requests have no observer CUDA fence or tensor content reads.
Frozen storage/digest primitives are imported
from the stage without modifying or composing the single-request TP16 observer.
No native return, exception, argument, descriptor or notification is replaced.
"""
import copy
import functools
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import inspect
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
import weakref


ENV = 'STEP5_PD_STATE_OBSERVATION'
NIXL = 'vllm.distributed.kv_transfer.kv_connector.v1.nixl.'
# Native instrumentation and retention-policy implementations are independently pinned.
NATIVE_SOURCE_PATHS = (
    'vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py',
    'vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py',
    'vllm/v1/worker/gpu_worker.py',
    'vllm/v1/worker/gpu_model_runner.py',
    'vllm/v1/worker/gpu/model_runner.py',
    'vllm/v1/worker/utils.py',
    'vllm/v1/core/sched/scheduler.py',
    'vllm/v1/core/kv_cache_manager.py',
    'vllm/v1/core/kv_cache_coordinator.py',
    'vllm/v1/core/single_type_kv_cache_manager.py',
    'vllm/v1/core/block_pool.py',
    'vllm/v1/engine/core_client.py',
    'vllm/v1/engine/core.py',
)
MODULES = {path[:-3].replace('/', '.') for path in NATIVE_SOURCE_PATHS}
_installed = None


class ObservationBudgetExceeded(RuntimeError):
    """The diagnostic budget, not the native workload, has been exhausted."""


def block_lists(groups):
    return [list(map(int, group)) for group in groups]


class Observation:
    def __init__(self, config):
        self.config = config
        if config['role'] not in ('prefill', 'decode'):
            raise ValueError('role must be prefill or decode')
        self.role = config['role']
        self.output = Path(config['output_dir']).resolve()
        if not self.output.is_absolute() or self.output.is_relative_to('/mnt/codesign-exp'):
            raise ValueError('Observation output must be worker-local scratch')
        self.output.mkdir(parents=True, exist_ok=True)
        for key in ('expected_tp_ranks', 'expected_layer_count',
                    'expected_summary_layer_count', 'region_size', 'block_size',
                    'max_record_bytes', 'max_capture_bytes_per_request', 'max_events'):
            if type(config[key]) is not int or config[key] <= 0:
                raise ValueError(f'{key} must be a positive integer')
        self.selected = tuple(config['selected_client_request_ids'])
        if not self.selected or any(not isinstance(s, str) or not s for s in self.selected):
            raise ValueError('selected_client_request_ids must be nonempty strings')
        self.content_ids = tuple(config.get('content_client_request_ids', self.selected))
        if any(value not in self.selected for value in self.content_ids):
            raise ValueError('content_client_request_ids must be a subset of selected IDs')
        self.lock = threading.RLock()
        self.local = threading.local()
        self.pid = os.getpid()
        self.events = 0
        path = self.output/f'{self.role}_pid_{self.pid}.jsonl'
        self.written = path.stat().st_size if path.exists() else 0
        self.disabled = False
        self.captured = {}
        self.capture_elapsed = {}
        self.provenance = set()
        self.runners = []
        self.runner_reported = weakref.WeakSet()
        self.connectors = []
        self.managers = weakref.WeakKeyDictionary()
        self.manager_schedulers = weakref.WeakKeyDictionary()
        self.worker_states = weakref.WeakKeyDictionary()
        self.connector_states = weakref.WeakKeyDictionary()
        self.freed = weakref.WeakKeyDictionary()
        self.deferred = weakref.WeakKeyDictionary()

    def selected_id(self, request_id):
        if not isinstance(request_id, str):
            return None
        for client_id in self.selected:
            # Locked completion serving adds cmpl- and a prompt index;
            # InputProcessor adds an eight-hex internal uniqueness suffix.
            pattern = (r'(?:cmpl-|chatcmpl-)?' + re.escape(client_id)
                       + r'(?:-\d+)?(?:-[0-9a-f]{8})?')
            if request_id == client_id or re.fullmatch(pattern, request_id):
                return client_id
        return None

    def content_selected(self, request_id):
        return self.selected_id(request_id) in self.content_ids

    def verify_module(self, module, expected, kind):
        path = Path(module.__file__).resolve()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        witness = (kind, module.__name__, str(path), expected, actual)
        if witness not in self.provenance:
            self.emit('provenance', kind=kind, module=module.__name__, module_file=str(path),
                      expected_sha256=expected, actual_sha256=actual, verified=actual == expected)
            self.provenance.add(witness)
        if not isinstance(expected, str) or actual != expected:
            raise ValueError(f'{kind} loaded source hash mismatch: {module.__name__}: {path}')

    def frozen_helpers(self):
        modules = []
        for name in ('role_capacity_observation', 'role_state_observation'):
            module = importlib.import_module(name)
            self.verify_module(module, self.config['frozen_environment_hashes'][name+'.py'],
                               'frozen_environment')
            modules.append(module)
        return modules[0], modules[1]

    def identity(self, owner):
        row = {}
        config = getattr(owner, 'vllm_config', None)
        if config is not None:
            row['dp_rank'] = config.parallel_config.data_parallel_rank
            transfer = getattr(config, 'kv_transfer_config', None)
            if transfer is not None:
                row['configured_engine_id'] = transfer.engine_id
        for name in ('tp_rank', 'engine_id'):
            if hasattr(owner, name):
                row[name] = getattr(owner, name)
        connector = getattr(owner, 'connector', None)
        if connector is not None and hasattr(connector, 'engine_id'):
            row['engine_id'] = connector.engine_id
        if hasattr(owner, 'model_runner') or type(owner).__name__ == 'GPUModelRunner':
            live = [ref() for ref in self.connectors if ref() is not None]
            if len(live) == 1:
                row.update(tp_rank=live[0].tp_rank, engine_id=live[0].engine_id)
        if owner in self.managers:
            row.update(self.managers[owner])
        return row

    def _process(self):
        if self.pid != os.getpid():
            self.pid = os.getpid()
            self.events = self.written = 0
            self.disabled = False
            self.captured = {}

    def emit(self, event, owner=None, request_id=None, **fields):
        with self.lock:
            self._process()
            if self.disabled:
                return
            row = dict(event=event, role=self.role, host=socket.gethostname(),
                       pid=self.pid, monotonic_ns=time.monotonic_ns(),
                       wall_time_ns=time.time_ns(), observer_fence=False)
            if owner is not None:
                row.update(self.identity(owner))
            if request_id is not None:
                row.update(request_id=request_id,
                           selected_client_request_id=self.selected_id(request_id),
                           content_selected=self.content_selected(request_id))
            row.update(fields)
            payload = (json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n').encode()
            # Reserve room for one terminal diagnostic record within the same cap.
            if (self.events >= self.config['max_events']-1 or
                    self.written+len(payload)+2048 > self.config['max_record_bytes']):
                raise ObservationBudgetExceeded('JSONL byte/event cap reached')
            self._write(payload)

    def _write(self, payload):
        path = self.output/f'{self.role}_pid_{self.pid}.jsonl'
        with path.open('ab') as stream:
            stream.write(payload)
        self.events += 1
        self.written += len(payload)

    def error(self, seam, error):
        """Report immediately, with no retry and no purported replacement data."""
        try:
            with self.lock:
                self._process()
                budget = isinstance(error, ObservationBudgetExceeded)
                event = 'observer_budget_exceeded' if budget else 'observer_error'
                row = dict(event=event, role=self.role, host=socket.gethostname(),
                           pid=self.pid, monotonic_ns=time.monotonic_ns(),
                           wall_time_ns=time.time_ns(), seam=seam,
                           error=repr(error)[:1000], verified=False)
                print('STEP5_PD_STATE_OBSERVATION UNVERIFIED '+json.dumps(row),
                      file=sys.stderr, flush=True)
                payload = (json.dumps(row)+'\n').encode()
                if (not self.disabled and self.events < self.config['max_events'] and
                        self.written+len(payload) <= self.config['max_record_bytes']):
                    self._write(payload)
                self.disabled = True
        except Exception:
            self.disabled = True

    def safe(self, seam, callback, *args):
        if self.disabled:
            return None
        try:
            return callback(*args)
        except Exception as error:
            self.error(seam, error)
            return None

    def synchronize(self):
        import torch

        torch.cuda.synchronize()

    def reserve_capture(self, request_id, size):
        with self.lock:
            used = self.captured.get(request_id, 0)
            if used+size > self.config['max_capture_bytes_per_request']:
                raise ObservationBudgetExceeded(f'CPU capture bytes for {request_id}')
            self.captured[request_id] = used+size

    def diagnostic(self, owner, request_id, boundary, callback):
        if not self.content_selected(request_id):
            return
        cap = self.config['max_capture_seconds_per_request']
        if (not isinstance(cap, (int, float)) or isinstance(cap, bool) or
                not math.isfinite(cap) or cap <= 0):
            raise ValueError('max_capture_seconds_per_request must be positive')
        started = time.monotonic_ns()
        try:
            return callback()
        finally:
            elapsed = time.monotonic_ns()-started
            total = self.capture_elapsed.get(request_id, 0)+elapsed
            self.capture_elapsed[request_id] = total
            self.emit('capture_elapsed', owner, request_id, boundary=boundary,
                      elapsed_ns=elapsed, cumulative_elapsed_ns=total,
                      cap_seconds=cap, cap_mode='post_return_soft',
                      cuda_calls_interruptible=False, observer_fence=True)
            if total > cap*1e9:
                raise ObservationBudgetExceeded(f'Post-return capture elapsed cap: {request_id}')

    def register(self, connector, result, arguments, token):
        capacity, _ = self.frozen_helpers()

        self.connectors.append(weakref.ref(connector))
        self.connector_states[connector] = dict(reads={}, captured_sources=set(), tail_checkpoints={})
        config = connector.kv_cache_config
        context = connector.vllm_config.compilation_config.static_forward_context
        expected_raw = dict(config.transfer_group_index_by_layer)
        expected_side = {name: layer.main_layer_name for name, layer in context.items()
                         if hasattr(layer, 'main_layer_name') and
                          hasattr(layer, 'kv_cache_side_storage_transfer_view')}
        group_specs, semantics = [], []
        for group_id, scheduler_group in enumerate(config.transfer_group_ids):
            group = config.kv_cache_groups[scheduler_group]
            spec = group.kv_cache_spec
            layer_specs = getattr(spec, 'kv_cache_specs', None)
            records = {}
            for name in group.layer_names:
                layer_spec = layer_specs[name] if layer_specs is not None else spec
                entry = dict(spec_type=type(layer_spec).__name__, block_size=int(layer_spec.block_size))
                for field in ('sliding_window', 'extra_retained_tokens'):
                    if hasattr(layer_spec, field):
                        entry[field] = getattr(layer_spec, field)
                records[name] = entry
            group_specs.append(dict(transfer_group_id=group_id, scheduler_group_id=int(scheduler_group),
                                    layer_names=list(group.layer_names), layer_specs=records))
            kinds = {(spec['block_size'], spec.get('sliding_window')) for spec in records.values()}
            if len(kinds) != 1:
                raise ValueError('Transfer group has differing native block/window semantics')
            block_size, window = kinds.pop()
            semantics.append(dict(group_id=group_id, native_group_id=int(scheduler_group),
                                  block_size=block_size, sliding_window=window))
        storages, registrations = {}, []
        bases = connector.kv_caches_base_addr[connector.engine_id][connector.tp_rank]
        for name, tensor in connector.device_kv_caches.items():
            main = expected_side.get(name, name)
            component = 'summary_mean' if name in expected_side else 'raw_kv'
            group = expected_raw[main]
            # Native registration deduplicates equal page addresses across groups.
            # A backing allocation alone is not identity: sibling rows have offsets.
            matches = [index for index, base in enumerate(bases)
                       if base == tensor.data_ptr()]
            if len(matches) != 1:
                raise ValueError(f'Cache view has no unique native physical region: {name}')
            index = matches[0]
            region_name = connector.region_names[index]
            region_tensor = connector.device_kv_caches[region_name]
            if (tensor.shape != region_tensor.shape or tensor.stride() != region_tensor.stride()
                    or tensor.dtype != region_tensor.dtype):
                raise ValueError(f'Cache alias differs from its registered physical view: {name}')
            metadata = capacity.storage_record(name, component, tensor, storages)
            wire = connector.block_len_per_layer[index]
            stride = connector.block_stride_per_layer[index]
            row_bytes = tensor[0].numel()*tensor.element_size()
            if (not tensor[0].is_contiguous() or wire != row_bytes or
                    tensor.stride(0)*tensor.element_size() != stride or stride != wire or
                    connector.region_num_blocks[index] != tensor.shape[0] or
                    connector.region_group_ids[index] not in (group, -1)):
                raise ValueError(f'Registered wire region differs from complete physical rows: {name}')
            registrations.append(dict(layer=main, registration_name=name, group_id=group,
                scheduler_group_id=int(config.transfer_group_ids[group]),
                component=component, region_index=index, tensor=metadata,
                region_registration_name=region_name,
                region_group_id=connector.region_group_ids[index],
                wire_row_storage_offset_bytes=tensor.storage_offset()*tensor.element_size(),
                wire_block_stride_bytes=stride,
                resident_view_bytes=tensor.numel()*tensor.element_size(), wire_block_bytes=wire,
                physical_stride_bytes=tensor.stride(0)*tensor.element_size(),
                alias_storage=metadata['storage'], resident_and_wire_are_not_additive=True))
        self.emit('registration', connector, expected_raw_layers=expected_raw,
                  expected_summary_layers=expected_side, registrations=registrations,
                  registered_region_names=list(connector.region_names),
                  registered_region_groups=list(connector.region_group_ids),
                   transfer_group_specs=group_specs,
                   transfer_group_to_scheduler_group=list(config.transfer_group_ids),
                   group_specs=semantics,
                   transfer_group_ids=list(config.transfer_group_ids),
                  physical_blocks_per_logical=int(connector._physical_blocks_per_logical_kv_block),
                  unique_resident_storage_bytes=sum(storages.values()),
                  expected_tp_ranks=self.config['expected_tp_ranks'],
                  expected_layer_count=self.config['expected_layer_count'],
                  expected_summary_layer_count=self.config['expected_summary_layer_count'],
                  region_size=self.config['region_size'], block_size=self.config['block_size'])
        if len(expected_raw) != self.config['expected_layer_count']:
            raise ValueError('Native raw layer inventory differs from activation contract')
        if len(expected_side) != self.config['expected_summary_layer_count']:
            raise ValueError('Native summary layer inventory differs from activation contract')
        if set(connector.device_kv_caches) != set(expected_raw) | set(expected_side):
            raise ValueError('Missing or unexpected registered native cache tensors')
        self.connector_states[connector]['registrations'] = registrations
        self.connector_states[connector]['group_specs'] = semantics
        if 'resident_identity' in self.config:
            module_name = 'vllm.models.step5.nvidia.sparse_summary_cache'
            self.verify_module(sys.modules[module_name], self.config['schema_source_hashes'][module_name], 'native_schema')
            self.reference_module().components(self, connector)

    def capture(self, connector, request_id, groups, event, remote_groups=None, **join):
        if not self.content_selected(request_id):
            return
        import torch
        _, state_module = self.frozen_helpers()

        state = self.connector_states[connector]
        for registration in state['registrations']:
            group = registration['group_id']
            ids = groups[group]
            if not ids:
                continue
            tensor = connector.device_kv_caches[registration['registration_name']]
            wire = registration['wire_block_bytes']
            row_bytes = state_module.tensor_bytes(torch, tensor[0])
            if (wire != row_bytes or tensor.stride(0)*tensor.element_size() != wire or
                    registration['region_index'] is None):
                raise ValueError('Registered wire row does not equal the complete physical tensor row')
            self.reserve_capture(request_id, len(ids)*row_bytes)
            hashes = []
            for block in ids:
                if not 0 <= block < tensor.shape[0]:
                    raise ValueError('Physical block ID outside native registered tensor')
                raw = tensor[block].contiguous().view(torch.uint8).reshape(-1).cpu()
                sample = state_module.StateObservation.raw_sample(raw, 0, row_bytes,
                                                     tensor[block].shape, str(tensor.dtype))
                hashes.append(sample['sha256'])
                del sample, raw
            fields = dict(group_id=group, layer=registration['layer'],
                          component=registration['component'],
                          registration_name=registration['registration_name'],
                          region_index=registration['region_index'],
                          region_registration_name=registration['region_registration_name'],
                          wire_row_storage_offset_bytes=registration['wire_row_storage_offset_bytes'],
                          local_block_ids=list(ids), block_hashes=hashes,
                          block_bytes=row_bytes, digest_scope='complete_physical_wire_row',
                          observer_fence=True, **join)
            if remote_groups is not None:
                if len(remote_groups[group]) != len(ids):
                    raise ValueError('Actual local/remote descriptor block mapping length differs')
                fields['remote_block_ids'] = list(remote_groups[group])
            self.emit(event, connector, request_id, **fields)

    def read_request(self, connector, arguments):
        req, meta = arguments['req_id'], arguments['meta']
        if not self.selected_id(req):
            return
        remote = meta.remote
        snapshot = dict(remote_engine_id=remote.engine_id,
                        remote_request_id=remote.request_id,
                        remote_host=remote.host, remote_port=remote.port,
                        remote_num_tokens=remote.num_tokens,
                        advertised_remote_block_ids=block_lists(remote.block_ids),
                        reserved_local_block_ids=block_lists(meta.local_block_ids),
                        local_physical_block_ids=block_lists(meta.local_physical_block_ids))
        snapshot['block_id_group_order'] = 'transfer_group'
        reads = self.connector_states[connector]['reads']
        if req in reads:
            if reads[req]['metadata'] != snapshot:
                raise ValueError('Native handshake reentry changed canonical READ metadata')
            return
        reads[req] = dict(metadata=snapshot, plans=[])
        self.emit('read_metadata', connector, req, **snapshot)

    def read_context(self, connector, arguments):
        req = arguments['request_id']
        if not self.selected_id(req):
            return None
        spec = arguments['read_spec']
        if spec.block_ids_by_region:
            raise ValueError('Region-reordered pull needs a region-to-layer content join')
        context = dict(request_id=req, remote_request_id=arguments['remote_request_id'],
                       remote_engine_id=arguments['dst_engine_id'], remote_tp_rank=int(spec.remote_rank),
                       requested_local_block_ids=block_lists(spec.local_block_ids),
                       requested_remote_block_ids=block_lists(spec.remote_block_ids), descriptors=[])
        self.local.read = context
        return context

    def descriptors(self, connector, result, arguments, token):
        context = getattr(self.local, 'read', None)
        if context is not None:
            context['descriptors'].append(dict(block_ids=block_lists(arguments['block_ids']),
                descriptor_ids=result.tolist(),
                region_group_ids=copy.deepcopy(arguments.get('region_group_ids'))))

    def read_submitted(self, connector, result, arguments, context):
        if context is None or not result:
            return
        descs = context.pop('descriptors')
        if len(descs) != 2:
            raise ValueError('Selected READ has no exact remote/local descriptor pair')
        context.update(remote_block_ids=descs[0]['block_ids'],
                       local_block_ids=descs[1]['block_ids'],
                       remote_descriptor_ids=descs[0]['descriptor_ids'],
                       local_descriptor_ids=descs[1]['descriptor_ids'])
        req = context['request_id']
        self.connector_states[connector]['reads'][req]['plans'].append(context)
        self.emit('read_submit', connector, **context, boundary='after_native_submit')

    def network_done(self, connector, result, arguments, token):
        if arguments['transfers'] is not connector._recving_transfers:
            return
        done, failed = result
        for req in sorted(done - failed - connector._recv_failures):
            if self.selected_id(req):
                self.emit('read_network_done', connector, req,
                          boundary='native_pop_done_transfers',
                          **self.connector_states[connector]['reads'][req]['metadata'])

    def transfer_results(self, connector, result, arguments, token):
        ready = [req for req in sorted(result.finished_recving-result.failed_recving) if self.selected_id(req)]
        # Emit every native readiness witness before any diagnostic CUDA action,
        # including when a completion batch contains more than one request.
        for req in ready:
            state = self.connector_states[connector]['reads'][req]
            self.emit('native_read_ready', connector, req,
                       boundary='native_get_transfer_results_finished_recving',
                      observer_fence=False, **state['metadata'])
        for req in ready:
            if self.content_selected(req):
                self.diagnostic(connector, req, 'after_read',
                                lambda: self.read_content(connector, req))

    def read_content(self, connector, req):
        self.synchronize()
        state = self.connector_states[connector]['reads'][req]
        self.emit('read_gpu_visible', connector, req,
                  boundary='native_postprocessing_then_diagnostic_cuda_synchronize',
                  observer_fence=True, **state['metadata'])
        for plan in state['plans']:
            self.capture(connector, req, plan['local_block_ids'], 'target_content',
                         plan['remote_block_ids'], remote_engine_id=plan['remote_engine_id'],
                         remote_request_id=plan['remote_request_id'],
                         remote_tp_rank=plan['remote_tp_rank'])
        self.tail(connector, req, state['metadata']['local_physical_block_ids'], 'after_read')

    def notified(self, connector, result, arguments, token):
        for req in sorted(result):
            if self.selected_id(req):
                self.emit('source_notified', connector, req,
                          boundary='native_all_consumer_notifications_received',
                          cause='READ_completion_notification_not_lease_expiry')

    def worker_before(self, worker, arguments):
        if getattr(self.local, 'worker_startup', False):
            return []
        output = arguments['scheduler_output']
        self.runner_info(worker)
        self.local.worker_identity = self.identity(worker)
        self.local.worker_execution_kind = 'scheduled'
        self.local.scheduled_request_ids = [req for req in output.num_scheduled_tokens if self.selected_id(req)]
        states = self.worker_states.setdefault(worker, {})
        if self.role == 'prefill':
            for req in output.finished_req_ids:
                if self.content_selected(req) and req in states:
                    for reference in self.connectors:
                        connector = reference()
                        if connector is None:
                            continue
                        state = self.connector_states[connector]
                        if req in state['captured_sources']:
                            continue
                        self.diagnostic(connector, req, 'source_held',
                                        lambda: self.source_content(connector, req, states[req], 'source_held'))
                        state['captured_sources'].add(req)
        for req in output.finished_req_ids:
            states.pop(req, None)
        for new in output.scheduled_new_reqs:
            if self.selected_id(new.req_id):
                states[new.req_id] = dict(blocks=block_lists(new.block_ids),
                                          computed=int(new.num_computed_tokens),
                                          prompt_tokens=len(new.prompt_token_ids))
                client = self.selected_id(new.req_id)
                if client in self.config.get('local_reference_client_request_ids', ()):
                    self.emit('reference_start', worker, new.req_id,
                              actual_prompt_tokens=len(new.prompt_token_ids),
                              actual_prompt_sha256=hashlib.sha256(json.dumps(new.prompt_token_ids).encode()).hexdigest(),
                              initial_computed_tokens=int(new.num_computed_tokens))
                    helper = self.reference_module()
                    self.diagnostic(worker, new.req_id, 'reference_resident_identity',
                        lambda: helper.resident_identity(self, worker, new.req_id,
                            'reference:'+client, self.identity(worker)['tp_rank']))
        cached = output.scheduled_cached_reqs
        for index, req in enumerate(cached.req_ids):
            if not self.selected_id(req):
                continue
            state = states[req]
            added = cached.new_block_ids[index]
            if added is not None:
                if req in cached.resumed_req_ids:
                    state['blocks'] = block_lists(added)
                else:
                    for group, blocks in zip(state['blocks'], added):
                        group.extend(map(int, blocks))
            state['computed'] = int(cached.num_computed_tokens[index])
        scheduled = []
        for req, count in output.num_scheduled_tokens.items():
            if not self.selected_id(req):
                continue
            state = states[req]
            start = state['computed']
            positions = list(range(start, start+int(count)))
            fields = {} if self.selected_id(req) in self.config.get('pressure_client_request_ids', ()) else dict(
                local_block_ids=copy.deepcopy(state['blocks']))
            self.emit('target_compute' if self.role == 'decode' else 'source_compute', worker, req,
                       num_computed_tokens=start, num_scheduled_tokens=int(count),
                       positions=positions, **fields,
                      boundary='before_native_execute')
            state['computed'] += int(count)
            scheduled.append(req)
        return scheduled

    def worker_after(self, worker, result, arguments, scheduled):
        if not scheduled:
            return
        for reference in self.connectors:
            connector = reference()
            if connector is None:
                continue
            for req in scheduled:
                if not self.content_selected(req):
                    continue
                request = self.worker_states[worker][req]
                if self.role == 'prefill':
                    captured = self.connector_states[connector]['captured_sources']
                    if request['computed'] >= request['prompt_tokens'] and req not in captured:
                        # Before returning the final prefill execution: the scheduler
                        # cannot have exposed this step's handoff response yet.
                        self.diagnostic(connector, req, 'source_final_prefill',
                            lambda: self.source_content(connector, req, request,
                                                        'after_final_prefill_before_worker_return'))
                        captured.add(req)
                else:
                    checkpoints = self.connector_states[connector]['tail_checkpoints'].setdefault(req, {})
                    boundary = None
                    if 'first_local_tail_complete' not in checkpoints and request['computed'] >= request['prompt_tokens']:
                        boundary = 'first_local_tail_complete'
                    elif 'first_local_tail_complete' in checkpoints and 'first_decode_region_crossing' not in checkpoints:
                        end = checkpoints['first_local_tail_complete']
                        position = ((end-1)//self.config['region_size']+1)*self.config['region_size']
                        if request['computed'] > position:
                            boundary = 'first_decode_region_crossing'
                    elif 'resident_identity' in self.config and 'final_decode_input_complete' not in checkpoints:
                        spec = self.config['request_specs'][self.selected_id(req)]
                        if request['computed'] == request['prompt_tokens']+spec['num_decode_tokens']-1:
                            boundary = 'final_decode_input_complete'
                    if boundary is not None:
                        self.diagnostic(connector, req, boundary,
                            lambda: self.local_tail(connector, req, request, boundary))
                        if 'resident_identity' in self.config:
                            self.diagnostic(connector, req, boundary+'.reference_state',
                                lambda: self.reference_module().checkpoint(self, connector, req, request, boundary))
                        checkpoints[boundary] = request['computed']

    def reference_module(self):
        module = importlib.import_module('r2_reference')
        self.verify_module(module, self.config['reference_helper_sha256'], 'reference_helper')
        return module

    def model_loaded(self, worker, result, arguments, token):
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        name = 'vllm.distributed.parallel_state'
        self.verify_module(sys.modules[name], self.config['schema_source_hashes'][name], 'native_schema')
        self.runtime_module().bind_model_state(worker, self, wrap)
        self.reference_module().resident_identity(self, worker, None, 'startup',
                                                  get_tensor_model_parallel_rank())

    def runtime_module(self):
        module = importlib.import_module('r2_runtime')
        self.verify_module(module, self.config['runtime_helper_sha256'], 'runtime_helper')
        return module

    def dummy_before(self, worker, arguments):
        self.runner_info(worker)
        self.local.worker_identity = self.identity(worker)
        self.local.worker_execution_kind = 'idle_dp'
        self.local.scheduled_request_ids = ()

    def source_content(self, connector, req, request, boundary):
        physical = self.source_blocks(connector, request)
        self.synchronize()
        self.capture(connector, req, physical, 'source_content', boundary=boundary,
                     source_block_scope='native_exchange_clipped_rows')
        self.tail(connector, req, physical, boundary)

    def source_blocks(self, connector, request):
        selected = [request['blocks'][g] for g in connector.kv_cache_config.transfer_group_ids]
        groups = self.physical_blocks(connector, selected)
        for spec in self.connector_states[connector]['group_specs']:
            window = spec['sliding_window']
            if window is not None:
                # Match native get_exchange_clipped_blocks: old HMA table slots
                # are historical allocations, not retained transferable pages.
                count = (window+spec['block_size']-1)//spec['block_size']+1
                group = spec['group_id']
                groups[group] = groups[group][-count:]
        return groups

    def local_tail(self, connector, req, request, boundary):
        self.synchronize()
        selected = [request['blocks'][g] for g in connector.kv_cache_config.transfer_group_ids]
        self.tail(connector, req, self.physical_blocks(connector, selected), boundary,
                  num_computed_tokens=request['computed'])

    def runner_created(self, runner, result, arguments, token):
        self.runners.append(weakref.ref(runner))
        self.emit('runner_created', runner, runner_module=type(runner).__module__,
                  runner_class=type(runner).__qualname__, boundary='native_runner_constructor_returned')

    def runner_info(self, worker):
        if worker in self.runner_reported or not hasattr(worker, 'model_runner'):
            return
        runner = worker.model_runner
        module = type(runner).__module__
        zeroer = None
        if module == 'vllm.v1.worker.gpu.model_runner':
            zeroer = getattr(runner, 'kv_block_zeroer', None)
        elif module == 'vllm.v1.worker.gpu_model_runner':
            zeroer = getattr(runner, '_kv_block_zeroer', None)
        supported = zeroer is not None and getattr(getattr(type(zeroer), 'zero_block_ids', None), '_step5_r2', False)
        self.emit('runner_abi', worker, runner_module=module, runner_class=type(runner).__qualname__,
                  configured_use_v2_model_runner=getattr(worker, 'use_v2_model_runner', None),
                  reset_method='vllm.v1.worker.utils.KVBlockZeroer.zero_block_ids' if supported else None,
                  reset_abi_status='OBSERVED' if supported else 'UNVERIFIED_RESET_ABI',
                  boundary='before_first_native_worker_execute')
        self.runner_reported.add(worker)

    def physical_blocks(self, connector, groups):
        if connector._physical_blocks_per_logical_kv_block != 1:
            raise ValueError('Locked observation requires one physical page per logical block')
        return block_lists(groups)

    def tail(self, connector, req, groups, boundary, **fields):
        if not self.content_selected(req):
            return
        import torch
        capacity, state_module = self.frozen_helpers()

        context = connector.vllm_config.compilation_config.static_forward_context
        for row in self.connector_states[connector]['registrations']:
            if row['component'] != 'summary_mean' or not groups[row['group_id']]:
                continue
            cache = context[row['registration_name']]._cache
            assigned = groups[row['group_id']]
            if 'num_computed_tokens' in fields:
                index = (fields['num_computed_tokens']-1)//self.config['block_size']
                pages = assigned[max(0, index-1):index+1]
                selection = 'computed_token_page_and_preceding_page'
            else:
                # At read readiness the final assigned page may be allocation
                # padding. Preserve that distinction rather than inventing a tail.
                pages = assigned[-2:]
                selection = 'last_two_assigned_pages_may_include_padding'
            records = []
            for page in pages:
                numeric_tensors = {name: getattr(cache, name)[page]
                                   for name in ('numerator', 'denominator', 'maximum')}
                marker = cache.last_slot[page]
                numeric, exact = {}, {}
                for name, tensor in numeric_tensors.items():
                    exact[name], host = self.tensor_sample(req, name, tensor, capacity, state_module)
                    numeric[name] = dict(elements=host.numel(), finite=int(torch.isfinite(host).sum().item()))
                exact['last_slot'], host_marker = self.tensor_sample(req, 'last_slot', marker, capacity, state_module)
                mean = connector.device_kv_caches[row['registration_name']][page]
                mean_sample, _ = self.tensor_sample(req, 'summary_mean', mean, capacity, state_module)
                if boundary != 'first_local_tail_complete':
                    mean_sample.pop('raw_hex')
                records.append(dict(block_id=page, last_slot=int(host_marker.item()),
                                    accumulator_counts=numeric, exact_tensors=exact,
                                    summary_mean_wire_row=mean_sample))
            journal = self.journal(connector, req, cache, capacity, state_module)
            self.emit('tail_state', connector, req, layer=row['layer'], group_id=row['group_id'],
                      boundary=boundary, page_selection=selection, pages=records, journal=journal,
                      observer_fence=True, continuity_verdict='UNVERIFIED_REQUIRES_REFERENCE_REVIEW', **fields)

    def tensor_sample(self, req, name, tensor, capacity, state_module):
        import torch

        size = tensor.numel()*tensor.element_size()
        self.reserve_capture(req, size)
        metadata = capacity.storage_record(name, 'native_tail_tensor', tensor, {})
        host = tensor.detach().contiguous().cpu()
        # Flatten first: dtype-changing view of a zero-dimensional last_slot is
        # otherwise rejected by torch. No numeric conversion is performed.
        raw = host.reshape(-1).view(torch.uint8).reshape(-1)
        sample = state_module.StateObservation.raw_sample(raw, 0, size, tensor.shape, str(tensor.dtype))
        sample['tensor'] = metadata
        return sample, host

    def journal(self, connector, req, cache, capacity, state_module):
        width = cache.request_journal_width
        if width == 0:
            return dict(width=0, status='NOT_ALLOCATED_NATIVE_ABI')
        for reference in self.runners:
            runner = reference()
            if runner is None or runner.vllm_config is not connector.vllm_config:
                continue
            states = getattr(runner, 'req_states', None)
            if states is None or req not in states.req_id_to_index:
                continue
            index = int(states.req_id_to_index[req])
            tensors = {}
            for name in ('request_journal_values', 'request_journal_last', 'request_journal_tag'):
                tensor = getattr(cache, name)[index*width:(index+1)*width]
                tensors[name], _ = self.tensor_sample(req, name, tensor, capacity, state_module)
            return dict(width=int(width), request_state_index=index, status='OBSERVED_EXACT_BYTES', tensors=tensors)
        return dict(width=int(width), status='UNVERIFIED_REQUEST_INDEX_UNAVAILABLE')

    def scheduler_ready(self, scheduler, result, arguments, token):
        self.managers[scheduler.kv_cache_manager] = self.identity(scheduler)
        self.manager_schedulers[scheduler.kv_cache_manager] = weakref.ref(scheduler)
        coordinator = scheduler.kv_cache_manager.coordinator
        self.managers[coordinator] = self.identity(scheduler)
        self.manager_schedulers[coordinator] = weakref.ref(scheduler)
        if 'resident_identity' in self.config:
            pool = scheduler.kv_cache_manager.block_pool
            self.emit('pool_snapshot', scheduler, total_blocks=pool.num_gpu_blocks,
                      null_block_id=pool.null_block.block_id,
                      pool_free_blocks=pool.get_num_free_blocks(),
                      boundary='native_scheduler_constructor_return')

    def release_guard(self, scheduler, request=None):
        fields = {}
        if scheduler is not None:
            for name in ('processed_step_seq', 'defer_block_free'):
                if hasattr(scheduler, name):
                    fields[name] = getattr(scheduler, name)
        if request is not None:
            if hasattr(request, 'last_sched_seq'):
                fields['last_sched_seq'] = request.last_sched_seq
            fields['status'] = getattr(request.status, 'name', str(request.status))
        return fields

    def hold(self, scheduler, result, arguments, token):
        request = arguments['request']
        if self.role == 'prefill' and result[0] and self.selected_id(request.request_id):
            join = {}
            if result[1] is not None and 'remote_engine_id' in result[1]:
                join['engine_id'] = result[1]['remote_engine_id']
            self.emit('source_hold', scheduler, request.request_id,
                      blocks=block_lists(scheduler.kv_cache_manager.get_block_ids(request.request_id)),
                      pool_free_blocks=scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
                      num_computed_tokens=request.num_computed_tokens,
                      kv_xfer_params=result[1], pending_owner='native_connector',
                      block_id_group_order='scheduler_group',
                      boundary='connector_finished_delay_free', **join)

    def pool_before(self, manager, arguments):
        req = arguments['request']
        if not self.selected_id(req.request_id):
            return None
        return dict(blocks=block_lists(manager.get_block_ids(req.request_id)),
                    pool_free_blocks=manager.block_pool.get_num_free_blocks())

    def allocation_before(self, manager, arguments):
        before = self.pool_before(manager, arguments)
        if before is not None:
            self.local.allocation = dict(
                num_new_computed_tokens=arguments.get('num_new_computed_tokens', 0),
                num_external_computed_tokens=arguments.get('num_external_computed_tokens', 0),
                max_model_len=manager.max_model_len)
            if 'resident_identity' in self.config:
                before['native_demands'] = []
                self.local.allocation.update(request_id=arguments['request'].request_id,
                    demands=before['native_demands'], reserved_blocks=arguments.get('reserved_blocks', 0),
                    has_scheduled_reqs=arguments.get('has_scheduled_reqs', True),
                    request_status=getattr(arguments['request'].status, 'name', str(arguments['request'].status)),
                    native_watermark_blocks=manager.watermark_blocks)
        return before

    def allocation_demand(self, coordinator, result, arguments, token):
        context = getattr(self.local, 'allocation', None)
        if context is None:
            return
        manager = self.manager_schedulers[coordinator]().kv_cache_manager
        context['demands'].append(dict(native_blocks_to_allocate=int(result),
            pool_free_blocks=manager.block_pool.get_num_free_blocks(),
            apply_admission_cap=arguments.get('apply_admission_cap', False),
            **{name: context[name] for name in ('reserved_blocks', 'has_scheduled_reqs',
                                                'request_status', 'native_watermark_blocks')}))

    def allocated(self, manager, result, arguments, before):
        req = arguments['request'].request_id
        if before is None:
            return
        free = manager.block_pool.get_num_free_blocks()
        if result is None:
            self.emit('pressure', manager, req, before=before,
                      pool_free_blocks_after=free, cause='native_allocate_slots_returned_none',
                      num_external_computed_tokens=arguments.get('num_external_computed_tokens', 0))
            return
        blocks = block_lists(manager.get_block_ids(req))
        fields = dict(blocks=blocks, before=before, pool_free_blocks_before=before['pool_free_blocks'],
                      pool_free_blocks_after=free,
                      num_external_computed_tokens=arguments.get('num_external_computed_tokens', 0))
        if hasattr(arguments['request'], 'num_prompt_tokens'):
            fields['num_prompt_tokens'] = int(arguments['request'].num_prompt_tokens)
        if 'resident_identity' in self.config:
            fields.update(initial_computed_tokens=int(arguments['request'].num_computed_tokens),
                          num_local_cache_tokens=arguments.get('num_new_computed_tokens', 0),
                          kv_transfer_params_present=getattr(arguments['request'], 'kv_transfer_params', None) is not None)
        params = getattr(arguments['request'], 'kv_transfer_params', None)
        if params is not None:
            for key in ('remote_request_id', 'remote_engine_id'):
                if key in params:
                    fields[key] = params[key]
        if self.role == 'decode' and (arguments.get('num_external_computed_tokens', 0) or
                                      arguments.get('delay_cache_blocks', False)):
            self.emit('target_reserve', manager, req, **fields)
        elif (self.selected_id(req) not in self.config.get('pressure_client_request_ids', ())
              or before['blocks'] != blocks):
            self.emit('block_allocate', manager, req, **fields)
        freed = self.freed.setdefault(manager, {})
        reused = []
        for group, ids in enumerate(blocks):
            for block in ids:
                previous = freed.pop((group, block), None)
                if previous is not None:
                    reused.append(dict(group_id=group, block_id=block, previous_request_id=previous))
        if reused:
            self.emit('reuse', manager, req, blocks=reused, boundary='native_allocate_slots')

    def skipped_before(self, coordinator, arguments):
        req = arguments['request_id']
        if self.selected_id(req):
            scheduler = self.manager_schedulers[coordinator]()
            manager = scheduler.kv_cache_manager
            request = scheduler.requests[req]
            native = sys.modules['vllm.v1.core.single_type_kv_cache_manager']
            policies = []
            for group, single in enumerate(manager.coordinator.single_type_managers):
                prefix = (type(single).remove_skipped_blocks
                          is native.SingleTypeKVCacheManager.remove_skipped_blocks)
                spec = single.kv_cache_spec
                window = getattr(spec, 'sliding_window', None)
                retained = (spec.retained_window_size(
                    scheduler_block_size=single.scheduler_block_size,
                    retain_eagle_proof=single.enable_caching and single.use_eagle)
                    if prefix and window is not None else None)
                policies.append(dict(native_group_id=group, block_size=single.block_size,
                                     prefix_retirement=prefix, sliding_window=window,
                                     retained_window=retained,
                                     num_skipped_tokens=single.get_num_skipped_tokens(
                                         arguments['processed_computed_tokens']) if prefix else None))
            return dict(blocks=block_lists(manager.get_block_ids(req)),
                        pool_free_blocks=manager.block_pool.get_num_free_blocks(),
                        request_num_computed_tokens=int(request.num_computed_tokens),
                        request_num_in_flight_tokens=int(request.num_in_flight_tokens),
                        allocation_prefix=copy.deepcopy(getattr(self.local, 'allocation', None)),
                        retirement_group_specs=policies,
                        **self.release_guard(scheduler, request))

    def skipped(self, coordinator, result, arguments, before):
        if before is None:
            return
        req = arguments['request_id']
        manager = self.manager_schedulers[coordinator]().kv_cache_manager
        blocks = block_lists(manager.get_block_ids(req))
        if blocks != before['blocks']:
            self.emit('block_remove_skipped', coordinator, req, blocks=blocks, before=before,
                      pool_free_blocks_before=before['pool_free_blocks'],
                      pool_free_blocks_after=manager.block_pool.get_num_free_blocks(),
                      processed_computed_tokens=arguments['processed_computed_tokens'],
                      num_prompt_tokens=arguments.get('num_prompt_tokens'),
                      counts_scope='skipped_blocks_release',
                      boundary='native_remove_skipped_blocks_return')

    def freed_blocks(self, manager, result, arguments, before):
        if before is None:
            return
        req = arguments['request'].request_id
        after = manager.block_pool.get_num_free_blocks()
        freed = self.freed.setdefault(manager, {})
        for group, ids in enumerate(before['blocks']):
            for block in ids:
                freed[group, block] = req
        reference = self.manager_schedulers.get(manager)
        scheduler = reference() if reference is not None else None
        self.emit('source_free' if self.role == 'prefill' else 'target_free', manager, req,
                  blocks=before['blocks'], pool_free_blocks_before=before['pool_free_blocks'],
                  pool_free_blocks_after=after, boundary='native_kv_cache_manager_free',
                  counts_scope='request_release', status_observed_at='native_manager_free_return',
                  **self.release_guard(scheduler, arguments['request']))

    def scheduler_free_before(self, scheduler, arguments):
        request = arguments['request']
        if not self.selected_id(request.request_id):
            return None
        return dict(blocks=block_lists(scheduler.kv_cache_manager.get_block_ids(request.request_id)),
                    pool_free_blocks=scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
                    queued={id(blocks) for _, blocks in scheduler.deferred_frees})

    def scheduler_free_after(self, scheduler, result, arguments, before):
        if before is None:
            return
        req = arguments['request'].request_id
        for fence, blocks in scheduler.deferred_frees:
            if id(blocks) not in before['queued']:
                guard = self.release_guard(scheduler, arguments['request'])
                self.deferred.setdefault(scheduler, {})[id(blocks)] = dict(
                    request_id=req, blocks=before['blocks'], fence=int(fence),
                    deferred_fence_seq=int(fence),
                    **{key: guard[key] for key in ('last_sched_seq', 'status') if key in guard},
                    status_observed_at='deferred_free_submission')
                self.emit('free_deferred', scheduler, req, blocks=before['blocks'],
                          fence=int(fence), deferred_fence_seq=int(fence),
                          pool_free_blocks_before=before['pool_free_blocks'],
                          **guard,
                          pool_free_blocks_after=scheduler.kv_cache_manager.block_pool.get_num_free_blocks())

    def drain_before(self, scheduler, arguments):
        if not scheduler.deferred_frees:
            return None
        queued = {id(blocks) for _, blocks in scheduler.deferred_frees}
        context = dict(scheduler=scheduler, queued=queued, initial_queued=queued.copy())
        self.local.drain = context
        return context

    def pool_free_before(self, pool, arguments):
        context = getattr(self.local, 'drain', None)
        if context is None:
            return None
        scheduler = context['scheduler']
        if pool is not scheduler.kv_cache_manager.block_pool:
            return None
        remaining = {id(blocks) for _, blocks in scheduler.deferred_frees}
        removed = context['queued']-remaining
        context['queued'] = remaining
        record = None
        reason = 'deferred_queue_pop_not_unique'
        if len(removed) == 1:
            key = next(iter(removed))
            # Remove before calling native code, so a native exception cannot
            # leave a stale object ID available for a later false association.
            record = self.deferred.get(scheduler, {}).pop(key, None)
            reason = 'deferred_entry_has_no_selected_request_record'
        # Do not inspect, consume, copy or replace arguments['ordered_blocks'].
        return dict(scheduler=scheduler, record=record, reason=reason,
                    pool_free_blocks=pool.get_num_free_blocks())

    def pool_freed(self, pool, result, arguments, before):
        if before is None:
            return
        scheduler, record = before['scheduler'], before['record']
        after = pool.get_num_free_blocks()
        if record is None:
            self.emit('release_unverified', scheduler, reason=before['reason'],
                      counts_scope='unattributed_native_pool_call',
                      pool_free_blocks_before=before['pool_free_blocks'], pool_free_blocks_after=after,
                      native_pool_free_boundary='BlockPool.free_blocks_return', verified=False)
            return
        manager = scheduler.kv_cache_manager
        freed = self.freed.setdefault(manager, {})
        for group, ids in enumerate(record['blocks']):
            for block in ids:
                freed[group, block] = record['request_id']
        self.emit('source_free' if self.role == 'prefill' else 'target_free', scheduler,
                  **record, pool_free_blocks_before=before['pool_free_blocks'], pool_free_blocks_after=after,
                  counts_scope='request_release', boundary='native_deferred_request_pool_free',
                  native_pool_free_boundary='BlockPool.free_blocks_return',
                  **self.release_guard(scheduler))

    def drained(self, scheduler, result, arguments, before):
        if before is None:
            return
        remaining = {id(blocks) for _, blocks in scheduler.deferred_frees}
        records = self.deferred.get(scheduler, {})
        for key in before['initial_queued'] - remaining:
            record = records.pop(key, None)
            if record is None:
                continue
            self.emit('release_unverified', scheduler, **record, verified=False,
                      reason='selected_deferred_release_missing_per_call_pool_boundary')

    def frontend_before(self, client, arguments):
        if not self.selected_id(arguments['request'].request_id):
            return None
        return dict(lb_engines_before=copy.deepcopy(client.lb_engines),
                    engine_inflight_before=[int(client.engine_inflight[e]) for e in client.core_engines])

    def frontend(self, client, result, arguments, before):
        if before is None:
            return
        index = client.core_engines.index(result)
        self.emit('frontend_select', client, arguments['request'].request_id,
                  dp_rank=client.engine_ranks_managed[index], selected_dp_rank=client.engine_ranks_managed[index],
                  frontend_index=client.client_index, selected_engine_index=index,
                  engine_identity_hex=result.hex(), lb_engines=copy.deepcopy(client.lb_engines),
                  engine_inflight=[int(client.engine_inflight[e]) for e in client.core_engines], **before)

    def reset(self, zeroer, result, arguments, token):
        # This common native zeroer is called by both runner versions, including
        # startup warmup. Recording its return does NOT prove GPU completion.
        self.emit('reset', boundary='native_zero_block_ids_return_no_observer_fence',
                  **getattr(self.local, 'worker_identity', {}),
                  scheduled_request_ids=list(getattr(self.local, 'scheduled_request_ids', ())),
                  native_method=type(zeroer).__module__+'.'+type(zeroer).__qualname__+'.zero_block_ids',
                  observer_fence=False, gpu_completion='UNVERIFIED_NO_DIAGNOSTIC_FENCE',
                  block_ids=copy.deepcopy(arguments['block_ids']),
                   block_ids_by_group=copy.deepcopy(arguments.get('block_ids_by_group')))
        if 'resident_identity' in self.config:
            self.reference_module().reset_content(self, zeroer, arguments)

    def abort_before(self, scheduler, arguments):
        # Do not consume the native request_ids iterable. Snapshot selected live
        # requests and join only the actual aborted results after the native call.
        return {req: dict(request_status=getattr(request.status, 'name', str(request.status)),
                          native_receive_completed=req in scheduler.finished_recving_kv_req_ids)
                for req, request in scheduler.requests.items() if self.selected_id(req)}

    def abort(self, scheduler, result, arguments, token):
        for request in result:
            status = getattr(request.status, 'name', str(request.status))
            if self.selected_id(request.request_id) and 'ABORT' in status:
                self.emit('abort', scheduler, request.request_id, status=status,
                          before=(token or {}).get(request.request_id),
                          boundary='native_finish_requests_returned',
                          pool_free_blocks=scheduler.kv_cache_manager.block_pool.get_num_free_blocks())


def wrap(cls, name, observer, before=None, after=None, read_scope=False, worker_scope=False,
         drain_scope=False, allocation_scope=False, runtime_scope=False, startup_scope=False):
    """Only callbacks are guarded; the original is invoked exactly once."""
    original = getattr(cls, name)
    if getattr(original, '_step5_r2', False):
        return
    signature = inspect.signature(original)

    @functools.wraps(original)
    def observed(owner, *args, **kwargs):
        arguments = None
        token = None
        if not observer.disabled:
            def prepare():
                bound = signature.bind(owner, *args, **kwargs)
                return dict(bound.arguments)
            arguments = observer.safe(name+'.arguments', prepare)
            if arguments is not None and before is not None:
                token = observer.safe(name+'.before', before, owner, arguments)
        if startup_scope:
            observer.local.worker_startup = True
        try:
            result = original(owner, *args, **kwargs)
        finally:
            if startup_scope:
                observer.local.worker_startup = False
            if read_scope:
                observer.local.read = None
            if worker_scope:
                observer.local.worker_identity = {}
                observer.local.worker_execution_kind = None
                observer.local.scheduled_request_ids = ()
            if runtime_scope:
                observer.local.runtime = None
            if drain_scope:
                observer.local.drain = None
            if allocation_scope:
                observer.local.allocation = None
        if arguments is not None and after is not None:
            observer.safe(name+'.after', after, owner, result, arguments, token)
        return result

    observed._step5_r2 = True
    setattr(cls, name, observed)


def instrument(module, observer):
    name = module.__name__
    observer.verify_module(module, observer.config['source_hashes'][name], 'native_source')
    if name == NIXL+'base_worker':
        cls = module.NixlBaseConnectorWorker
        wrap(cls, 'register_kv_caches', observer, after=observer.register)
        wrap(cls, '_compute_desc_ids', observer, after=observer.descriptors)
        wrap(cls, '_pop_done_transfers', observer, after=observer.network_done)
        wrap(cls, 'get_transfer_results', observer, after=observer.transfer_results)
    elif name == NIXL+'pull_worker':
        cls = module.NixlPullConnectorWorker
        wrap(cls, '_read_blocks_for_req', observer, before=observer.read_request)
        wrap(cls, '_read_blocks', observer, before=observer.read_context,
             after=observer.read_submitted, read_scope=True)
        wrap(cls, '_get_new_notifs', observer, after=observer.notified)
    elif name == 'vllm.v1.worker.gpu_worker':
        if 'resident_identity' in observer.config:
            wrap(module.Worker, 'compile_or_warm_up_model', observer, startup_scope=True)
            wrap(module.Worker, 'load_model', observer, after=observer.model_loaded)
            wrap(module.Worker, 'execute_dummy_batch', observer,
                 before=observer.dummy_before, worker_scope=True)
        wrap(module.Worker, 'execute_model', observer,
             before=observer.worker_before, after=observer.worker_after, worker_scope=True)
    elif name in ('vllm.v1.worker.gpu_model_runner', 'vllm.v1.worker.gpu.model_runner'):
        wrap(module.GPUModelRunner, '__init__', observer, after=observer.runner_created)
        if name == 'vllm.v1.worker.gpu.model_runner' and 'resident_identity' in observer.config:
            observer.runtime_module().install(module, observer, wrap)
    elif name == 'vllm.v1.worker.utils':
        wrap(module.KVBlockZeroer, 'zero_block_ids', observer, after=observer.reset)
    elif name == 'vllm.v1.core.sched.scheduler':
        wrap(module.Scheduler, '__init__', observer, after=observer.scheduler_ready)
        wrap(module.Scheduler, '_connector_finished', observer, after=observer.hold)
        wrap(module.Scheduler, '_free_request_blocks', observer,
             before=observer.scheduler_free_before, after=observer.scheduler_free_after)
        wrap(module.Scheduler, '_drain_deferred_frees', observer,
             before=observer.drain_before, after=observer.drained, drain_scope=True)
        wrap(module.Scheduler, 'finish_requests', observer,
             before=observer.abort_before, after=observer.abort)
    elif name == 'vllm.v1.core.kv_cache_manager':
        wrap(module.KVCacheManager, 'allocate_slots', observer,
              before=observer.allocation_before, after=observer.allocated, allocation_scope=True)
        wrap(module.KVCacheManager, 'free', observer,
               before=observer.pool_before, after=observer.freed_blocks)
    elif name == 'vllm.v1.core.kv_cache_coordinator':
        if 'resident_identity' in observer.config:
            wrap(module.KVCacheCoordinator, 'get_num_blocks_to_allocate', observer, after=observer.allocation_demand)
        wrap(module.KVCacheCoordinator, 'remove_skipped_blocks', observer,
              before=observer.skipped_before, after=observer.skipped)
    elif name == 'vllm.v1.core.block_pool':
        wrap(module.BlockPool, 'free_blocks', observer,
             before=observer.pool_free_before, after=observer.pool_freed)
    elif name == 'vllm.v1.engine.core_client':
        wrap(module.DPLBAsyncMPClient, 'get_core_engine_for_request', observer,
             before=observer.frontend_before, after=observer.frontend)
    elif name == 'vllm.v1.engine.core':
        zero_waves = weakref.WeakKeyDictionary()

        def counter_fields(engine, arguments):
            fields = {key: getattr(engine, key) for key in (
                'dp_rank', 'current_wave', 'step_counter', 'dp_sync_interval',
                'engines_running', 'pending_pause', 'ignore_start_dp_wave')}
            for key in ('dp_rank', 'current_wave', 'step_counter', 'dp_sync_interval'):
                if type(fields[key]) is not int:
                    raise ValueError(f'Native DP counter is not an integer: {key}')
            for key in ('engines_running', 'pending_pause', 'ignore_start_dp_wave'):
                if type(fields[key]) is not bool:
                    raise ValueError(f'Native DP state is not a bool: {key}')
            return fields

        def checked(engine, result, arguments, before):
            fields = counter_fields(engine, arguments)
            if type(result) is not bool or type(arguments['local_unfinished']) is not bool:
                raise ValueError('Native finish-check input/return must be bool scalars')
            observer.emit('native_dp_finish_check', **fields,
                step_before=before['step_counter'], local_unfinished=arguments['local_unfinished'],
                native_has_unfinished=result,
                source_sync_boundary=(fields['step_counter'] == 1
                                      or fields['step_counter'] % fields['dp_sync_interval'] == 0),
                boundary='native_has_global_unfinished_reqs_return_no_observer_collective')

        def counter_zero(engine, arguments):
            fields = counter_fields(engine, arguments)
            wave = fields['current_wave']
            if (not fields['engines_running'] and fields['step_counter'] == 0
                    and zero_waves.get(engine) != wave):
                observer.emit('native_dp_wave_counter_zero', **fields,
                    boundary='native_process_input_queue_entry_after_wave_reset_or_startup')
                zero_waves[engine] = wave

        wrap(module.DPEngineCoreProc, '_has_global_unfinished_reqs', observer,
             before=counter_fields, after=checked)
        wrap(module.DPEngineCoreProc, '_process_input_queue', observer, before=counter_zero)
    observer.emit('hook_installed', module=name, source_path=name.replace('.', '/')+'.py')


class ObservationLoader(importlib.abc.Loader):
    def __init__(self, loader, observer):
        self.loader, self.observer = loader, observer

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        # A native import failure propagates once. Never retry module execution.
        self.loader.exec_module(module)
        self.observer.safe('instrument.'+module.__name__, instrument, module, self.observer)

    def __getattr__(self, name):
        return getattr(self.loader, name)


class ObservationFinder(importlib.abc.MetaPathFinder):
    def __init__(self, observer):
        self.observer = observer

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = ObservationLoader(spec.loader, self.observer)
        return spec


def install():
    """Activate only through the explicit per-server JSON environment contract."""
    global _installed
    if not os.environ.get(ENV) or _installed is not None:
        return _installed
    observer = Observation(json.loads(os.environ[ENV]))
    _installed = observer
    sys.meta_path.insert(0, ObservationFinder(observer))
    for name in sorted(MODULES & sys.modules.keys()):
        observer.safe('instrument.'+name, instrument, sys.modules[name], observer)
    observer.safe('install', lambda: observer.emit('observer_started',
                  instrumented_sources=list(NATIVE_SOURCE_PATHS),
                  selected_client_request_ids=list(observer.selected),
                  content_client_request_ids=list(observer.content_ids),
                  elapsed_cap_mode='post_return_soft_cuda_calls_not_interruptible'))
    return observer
