"""CPU fault checks for actual native table and known-state witness boundaries."""
import ast
import copy
from dataclasses import replace
import dis
import hashlib
import json
import time
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace
import weakref

import numpy as np
import pytest

from test_role_model_input_observation import (DeviceCpuApi, DeviceCpuTensor, TASK, device_case, fake_only,
    native, observation, runner, events, record, work)
from test_role_execution_driver import retained_methods
import role_capacity_observation as capacity
import role_execution_observation as execution
import role_state_observation as state_observation


def test_added_native_source_pins_and_observed_arguments_match_retained_owners():
    paths = {
        'runner': 'native_role_sources/observed_generation2/vllm/v1/worker/gpu/model_runner.py',
        'request_state': 'native_role_sources/observed_generation2/vllm/v1/worker/gpu/states.py',
        'block_table': 'native_state_20261019_01/native_sources/vllm/v1/worker/gpu/block_table.py',
        'summary': 'native_source_export/observed/vllm/models/step5/nvidia/sparse_summary_cache.py',
        'worker_utils': 'native_role_sources/observed/vllm/v1/worker/utils.py',
        'workspace': 'native_workspace_sources/observed/vllm/v1/worker/workspace.py',
        'prepare_no_dp_ep': 'native_state_20261019_01/native_sources/vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py',
    }
    for name, (source, owner, arguments, _) in observation.CENSUS_CALLS.items():
        if source not in paths:
            continue
        path = TASK/paths[source]
        body = path.read_bytes()
        assert hashlib.sha256(body).hexdigest() == observation.CENSUS_SOURCES[source][1]
        ast.parse(body)
        code = observation.nested_code(compile(body, str(path), 'exec', dont_inherit=True), *owner)
        args = code.co_varnames[:code.co_argcount+code.co_kwonlyargcount]
        assert all(argument in args for argument in arguments), name
        assert list(dis.findlinestarts(code))


def test_actual_table_prefix_and_stale_suffix_are_copied_before_native_reuse(device_case):
    case = device_case
    before = len(case.api.operations)
    index = case.observed.inputs(case.values, case.cpu, 1, 1)
    assert index == 0
    assert not any(row[0] == 'allocate' for row in case.api.operations[before:])
    case.observed.output(index, None, [])
    case.api.complete()
    case.values['block_tables'][0].values[0] = [17, 18, 19]
    case.runner.block_tables.num_blocks.gpu.values[0, 2] = 3
    result = case.observed.drain(lambda row: None)
    step = result['steps'][0]
    assert step['values']['block_table_0'] == [[0, 7, 8], [0, 0, 0]]
    assert step['values']['block_counts'] == [[0, 0, 1]]
    assert step['values']['idx_mapping'] == [2]
    assert step['block_table_semantics'].endswith('SUFFIX_NOT_OWNERSHIP')
    assert step['status'] == 'OBSERVED_DIAGNOSTIC_DEVICE_WITNESS_NOT_G2'
    assert result['G2'] == 'UNVERIFIED'


def test_group_specific_widths_come_from_resident_tables_before_one_reservation(device_case):
    case = device_case
    group2 = SimpleNamespace(layer_names=['layer.2'])
    case.runner.kv_cache_config.kv_cache_groups.append(group2)
    case.runner.block_tables.input_block_tables.append(case.tensor([[1]*5]*3, case.api.int32))
    case.runner.block_tables.block_sizes.append(8)
    counts = case.tensor([[0, 0, 1], [0, 0, 1]], case.api.int32)
    case.runner.block_tables.num_blocks = SimpleNamespace(gpu=counts, np=counts.values)
    allocations = len(case.api.operations)
    reservations = []
    def reserve(owned):
        assert len(case.api.operations) == allocations
        reservations.append(owned)
    current = observation.DeviceObservation(case.runner, case.observed.sites, case.observed.limits,
        case.observed.requests, case.observed.sampling, 65536, reserve=reserve)
    assert current.layout['block_table_0'][0] == (3, 3)
    assert current.layout['block_table_1'][0] == (3, 5)
    assert current.layout['block_counts'][0] == (2, 3)
    assert reservations == [current.owned_bytes]
    current.registrations(SimpleNamespace(scheduled_new_reqs=[case.req]))
    case.values['block_tables'] += (case.runner.block_tables.input_block_tables[1][:2],)
    case.values['slot_mappings_by_layer']['layer.2'] = case.values['slot_mappings_by_layer']['layer.0']
    index = current.inputs(case.values, case.cpu, 1, 1)
    current.output(index, None, [])
    case.api.complete()
    step = current.drain(lambda row: None)['steps'][0]
    assert step['block_sizes'] == [4, 8]
    assert step['values']['block_table_1'] == [[1]*5]*2


@pytest.mark.parametrize('fault', ['groups', 'dtype', 'width', 'storage', 'offset'])
def test_final_table_binding_fault_stops_only_device_evidence(device_case, fault):
    case = device_case
    table = case.values['block_tables'][0]
    if fault == 'groups':
        case.values['block_tables'] = ()
    elif fault == 'dtype':
        table.dtype = case.api.int64
    elif fault == 'width':
        case.values['block_tables'] = (table[:, :2],)
    elif fault == 'storage':
        case.values['block_tables'] = (case.tensor(table.values.copy(), case.api.int32),)
    else:
        case.values['block_tables'] = (case.runner.block_tables.input_block_tables[0][1:],)
    assert case.observed.inputs(case.values, case.cpu, 1, 1) is None
    assert case.observed.stopped
    assert case.observed.errors[0]['kind'] == 'finalized_device_inputs'
    assert not any(row[0] in ('query', 'synchronize') for row in case.api.operations)


@pytest.mark.parametrize('fault', ['cp', 'dtype', 'width', 'count_layout', 'allowance'])
def test_resident_table_readiness_is_checked_before_reservation_or_allocation(device_case, fault):
    case = device_case
    tables = case.runner.block_tables
    remaining = 65536
    if fault == 'cp':
        tables.cp_size = 2
    elif fault == 'dtype':
        tables.input_block_tables[0].dtype = case.api.int64
    elif fault == 'width':
        tables.input_block_tables[0] = tables.input_block_tables[0][:, :0]
    elif fault == 'count_layout':
        tables.num_blocks.gpu = case.tensor([[1]], case.api.int32)
    else:
        remaining = case.observed.owned_bytes-1
    allocations, reservations = len(case.api.operations), []
    with pytest.raises(ValueError):
        observation.DeviceObservation(case.runner, case.observed.sites, case.observed.limits,
            case.observed.requests, case.observed.sampling, remaining, reserve=reservations.append)
    assert len(case.api.operations) == allocations
    assert reservations == []


def known_runner(case, monkeypatch):
    torch = SimpleNamespace(Tensor=DeviceCpuTensor)
    raw = case.tensor(np.zeros((4, 2, 4, 8)), case.api.float32)
    mean = case.tensor(np.zeros((5, 1, 8)), case.api.uint8)
    numerator = case.tensor(np.zeros((5, 8)), case.api.float32)
    summary = SimpleNamespace(_mean=mean, mean=mean[:4], numerator=numerator,
        denominator=case.tensor(np.ones((5, 8)), case.api.float32),
        maximum=case.tensor(np.full((5, 1), -np.inf), case.api.float32),
        last_slot=case.tensor([-1]*5, case.api.int64))
    workspace = case.tensor(np.zeros(32), case.api.uint8)
    adapter = SimpleNamespace(_workspace=workspace)
    context = {'layer.0': SimpleNamespace(kv_cache=raw), 'layer.1': SimpleNamespace(kv_cache=raw[:]),
        'summary.0': SimpleNamespace(main_layer_name='layer.0', kv_cache=summary.mean,
            _cache=summary, _target_impl=SimpleNamespace(adapter=adapter))}
    specs = {'layer.0': SimpleNamespace(block_size=4, sliding_window=None, dtype=case.api.float32),
             'layer.1': SimpleNamespace(block_size=4, sliding_window=3, dtype=case.api.float32)}
    case.runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec = SimpleNamespace(kv_cache_specs=specs)
    case.runner.kernel_block_sizes = [4]
    case.runner.compilation_config = SimpleNamespace(static_forward_context=context)
    case.runner.kv_caches = [raw, raw[:, :, :2]]
    case.runner.block_tables.block_tables = [SimpleNamespace(gpu=case.runner.block_tables.input_block_tables[0])]
    case.runner.cudagraph_manager = SimpleNamespace(pool=(17, 29))
    module = ModuleType('vllm.v1.worker.workspace')
    module._manager = SimpleNamespace(_current_workspaces=[workspace, None],
        _persistent_resources=[{('tensor', 'alias'): workspace[2:4], ('opaque', 0): object()}, {}])
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return torch, context, module, workspace


def test_known_storage_reuses_physical_identity_and_keeps_unknown_resources_explicit(device_case, monkeypatch):
    case = device_case
    torch, context, module, workspace = known_runner(case, monkeypatch)
    before = len(case.api.operations)
    result = capacity.known_state_inventory(case.runner, torch, 100)
    assert len(case.api.operations) == before
    raw = [row for row in result['tensors'] if row['kind'] in ('kv_layer_view', 'kv_allocation_alias')]
    assert len({row['storage'] for row in raw}) == 1
    summary = [row for row in result['tensors'] if row['name'].endswith(('._mean', '.mean'))]
    assert len(summary) == 2 and summary[0]['storage'] == summary[1]['storage']
    aliases = [row for row in result['tensors'] if row['storage_bytes'] == 32]
    assert len(aliases) == 3 and len({row['storage'] for row in aliases}) == 1
    assert result['unique_storage_bytes'] == sum(result['unique_storages'].values())
    assert result['kv_groups'][0]['layers'][1]['sliding_window'] == 3
    assert result['graph_pool_handle'] == [17, 29]
    assert any(row['reason'] == 'UNCLASSIFIED_NATIVE_STORAGE_OWNER' for row in result['gaps'])
    assert result['referenced_live_bytes'] == result['allocator_peaks'] == 'UNMEASURED'
    assert not case.api.host_reads_allowed


def test_known_storage_and_missing_owner_gaps_obey_existing_entry_cap(device_case, monkeypatch):
    case = device_case
    torch, _, module, _ = known_runner(case, monkeypatch)
    with pytest.raises(ValueError, match='entry cap'):
        capacity.known_state_inventory(case.runner, torch, 1)
    module._manager._current_workspaces = []
    module._manager._persistent_resources = [{i: object() for i in range(200)}]
    with pytest.raises(ValueError, match='entry cap'):
        capacity.known_state_inventory(SimpleNamespace(), torch, 20)


def test_existing_resident_inventory_output_schema_is_preserved(device_case):
    case = device_case
    tensor = case.tensor([1, 2, 3, 4], case.api.float32)
    model = SimpleNamespace(_parameters={'a': tensor, 'b': tensor[1:]}, _buffers={},
        _non_persistent_buffers_set=set(), named_modules=lambda: [('', model)])
    result = capacity.inventory(SimpleNamespace(model=model), SimpleNamespace(Tensor=DeviceCpuTensor))
    assert result['unique_storage_bytes'] == tensor.values.nbytes
    assert [row['persistent'] for row in result['tensors']] == [True, True]
    assert set(result['tensors'][0]) == {'name','kind','shape','dtype','device','stride','storage',
                                       'storage_bytes','storage_offset','persistent'}
    assert result['tensors'][1]['storage_offset'] == 1


LIFECYCLE_SOURCE = """class RequestState:
    def __init__(self):
        self.req_id_to_index = {}
    def add_request(self, req_id, prompt_len, all_token_ids, num_computed_tokens, max_tokens):
        self.req_id_to_index[req_id] = 2
    def remove_request(self, req_id):
        return self.req_id_to_index.pop(req_id, None)
class BlockTables:
    block_sizes = [4]
    redirect_writes_to_null_block = False
    def __init__(self):
        self.num_blocks = SimpleNamespace(np=np.zeros((1, 3), dtype=np.int32))
    def append_block_ids(self, req_index, new_block_ids, overwrite):
        self.num_blocks.np[0, req_index] = len(new_block_ids[0])
    def apply_staged_writes(self):
        return None
    def gather_block_tables(self, idx_mapping, num_reqs_padded):
        return None
class GPUModelRunner:
    def _remove_request(self, req_id):
        return self.req_states.remove_request(req_id)
    def finish_requests(self, scheduler_output):
        for req_id in sorted(scheduler_output.finished_req_ids):
            self._remove_request(req_id)
    def add_requests(self, scheduler_output):
        for req in scheduler_output.scheduled_new_reqs:
            self._remove_request(req.req_id)
            self.req_states.add_request(req.req_id, req.prompt_len, [], 0, 9)
            self.block_tables.append_block_ids(2, req.block_ids, True)
    def update_requests(self, scheduler_output):
        for req, count, ids in zip(scheduler_output.scheduled_cached_reqs.req_ids,
                scheduler_output.scheduled_cached_reqs.num_computed_tokens,
                scheduler_output.scheduled_cached_reqs.new_block_ids):
            self.block_tables.append_block_ids(self.req_states.req_id_to_index[req], ids, False)
"""


@pytest.fixture
def lifecycle(native, monkeypatch, tmp_path):
    path = tmp_path/'cpu_state_lifecycle_fixture.py'
    path.write_text(LIFECYCLE_SOURCE)
    module = ModuleType('cpu_state_lifecycle_fixture')
    module.__file__ = str(path)
    module.SimpleNamespace, module.np = SimpleNamespace, np
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(LIFECYCLE_SOURCE, str(path), 'exec', dont_inherit=True), module.__dict__)
    calls = {name: row for name, row in observation.CENSUS_CALLS.items()
             if name in observation.CENSUS_REQUEST_SITES}
    monkeypatch.setattr(observation, 'CENSUS_CALLS', calls)
    sources = {key: (module.__name__, hashlib.sha256(path.read_bytes()).hexdigest())
               for key in observation.CENSUS_SOURCES}
    monkeypatch.setattr(observation, 'CENSUS_SOURCES', sources)
    class Model:
        def named_modules(self):
            return []
    native.sites.bind_census(SimpleNamespace(model=Model()), observation.ObservationLimits())
    owner = module.GPUModelRunner()
    owner.req_states, owner.block_tables = module.RequestState(), module.BlockTables()
    rows, gaps = native.sites.refresh_state_bindings(owner, observation.ObservationLimits())
    # Bind the actual source-matched runner into the same finite owner table.
    native.sites.bind_census_owner('runner', owner, native.sites.matching_census_sites(owner),
                                 None, observation.ObservationLimits(), gaps)
    state = observation.ModelInputObservation(native.sites, observation.ObservationLimits(),
                                              lambda value: {}, lambda row: None)
    scope = observation.ObservationScope(owner, state, 'execute')
    scope.active = True
    scheduler = SimpleNamespace(total_num_scheduled_tokens=4, finished_req_ids=set(), preempted_req_ids=set(),
        scheduled_new_reqs=[SimpleNamespace(req_id='first', prompt_len=4, num_computed_tokens=0, block_ids=([17],))],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[],num_computed_tokens=[],new_block_ids=[]),
        new_block_ids_to_zero=[],new_block_ids_to_zero_by_group=(),
        kv_cache_block_copies=[],kv_cache_block_copies_by_group=())
    return SimpleNamespace(native=native, module=module, owner=owner, state=state, scope=scope,
                           scheduler=scheduler, rows=rows, gaps=gaps)


def observed_lifecycle(case, fn):
    previous = sys.getprofile()
    try:
        sys.setprofile(case.scope.profile_hook)
        fn()
    finally:
        sys.setprofile(previous)


def test_observed_assignment_removal_and_index_reuse_keep_original_request_intervals(lifecycle):
    case = lifecycle
    observed_lifecycle(case, lambda: case.owner.add_requests(case.scheduler))
    first = copy.deepcopy(case.scope.events)
    appends = [row for row in first if row['site'] == 'block_append']
    assert appends[0]['arguments']['new_block_ids'] == [[17]]
    assert appends[0]['arguments']['overwrite'] is True
    assert appends[0]['lifecycle_after']['block_counts_cpu'] == [[0, 0, 1]]
    assert appends[0]['lifecycle_after']['request_indices'] == [dict(req_id='first', persistent_index=2)]
    case.scheduler.finished_req_ids = {'first'}
    case.scheduler.scheduled_new_reqs[0].req_id = 'second'
    observed_lifecycle(case, lambda: (case.owner.finish_requests(case.scheduler), case.owner.add_requests(case.scheduler)))
    assert case.owner.req_states.req_id_to_index == {'second': 2}
    assert first[0]['arguments']['lifecycle']['scheduler']['new_requests'][0]['req_id'] == 'first'
    assert all(row['gpu_completion'] == 'UNMEASURED' for row in case.scope.events)
    assert not case.scope.errors


def test_zero_work_still_records_native_removal_and_absence_of_device_completion(lifecycle):
    case = lifecycle
    case.owner.req_states.req_id_to_index['ended'] = 2
    case.scheduler.total_num_scheduled_tokens = 0
    case.scheduler.finished_req_ids = {'ended'}
    observed_lifecycle(case, lambda: case.owner.finish_requests(case.scheduler))
    calls = case.scope.events
    assert [row['site'] for row in calls] == ['request_finish', 'request_remove', 'request_state_remove']
    assert calls[-1]['result'] == 2
    assert calls[-1]['lifecycle_after']['request_indices'] == []
    assert calls[0]['arguments']['lifecycle']['scheduler']['total_num_scheduled_tokens'] == 0
    assert all(row['outcome'] == 'RETURNED_CPU_CALL_NOT_GPU_COMPLETION' for row in calls)


def test_replacement_of_profiling_table_rebinds_new_owner_without_rewriting_old_record(lifecycle):
    case = lifecycle
    old = case.owner.block_tables
    recorded = copy.deepcopy(case.rows)
    case.owner.block_tables = case.module.BlockTables()
    rows, gaps = case.native.sites.refresh_state_bindings(case.owner, case.state.limits)
    current = next(row for row in rows if row['module_path'] == 'runner.block_tables')
    assert current['owner_id'] == id(case.owner.block_tables)
    assert current['previous_owner_id'] == id(old)
    assert id(old) not in case.native.sites.census_owners
    assert recorded == case.rows
    assert not gaps


def test_incomplete_cached_assignment_does_not_escape_into_native_call(lifecycle):
    case = lifecycle
    case.scheduler.scheduled_cached_reqs.req_ids = ['incomplete']
    observed_lifecycle(case, lambda: case.owner.update_requests(case.scheduler))
    assert case.scope.stopped and case.state.stopped
    assert case.scope.errors[0]['kind'] == 'profile_observation'
    assert case.owner.req_states.req_id_to_index == {}


class StateCpuTensor(DeviceCpuTensor):
    """Guarded async storage fixture with an explicit FP8 bit interpretation."""
    def __getitem__(self, key):
        values = self.values[key]
        if not isinstance(values, np.ndarray):
            indices = key if type(key) is tuple else (key,)
            slices = tuple(slice(index if index >= 0 else self.values.shape[axis]+index,
                                 (index if index >= 0 else self.values.shape[axis]+index)+1)
                           for axis, index in enumerate(indices))
            values = self.values[slices].reshape(())
        return StateCpuTensor(values, self.api, self.dtype, self.device, pinned=self.pinned)

    def view(self, shape):
        if shape is self.api.uint8:
            values, dtype = self.values.view(np.uint8), self.api.uint8
        elif shape in (self.api.float32, self.api.bfloat16, self.api.float16, self.api.float8_e4m3fn):
            assert self.dtype is self.api.uint8
            kind = {self.api.float32: np.float32, self.api.bfloat16: np.uint16,
                    self.api.float16: np.float16, self.api.float8_e4m3fn: np.uint8}[shape]
            values, dtype = self.values.view(kind), shape
        else:
            values, dtype = self.values.reshape((shape,) if type(shape) is int else shape), self.dtype
        return StateCpuTensor(values, self.api, dtype, self.device, pinned=self.pinned)

    def numerical(self):
        if self.dtype is not self.api.float8_e4m3fn:
            return self.values
        bits = self.values
        exponent, mantissa = (bits >> 3) & 15, bits & 7
        values = np.where(exponent == 0, mantissa/8*2**-6,
                          (1+mantissa/8)*np.power(2.0, exponent.astype(np.int32)-7))
        values = np.where(bits & 128, -values, values)
        return np.where((exponent == 15) & (mantissa == 7), np.nan, values)

    def copy_(self, source, *, non_blocking):
        assert non_blocking is True
        if self.device == 'cpu':
            assert self.pinned
        self.api.queued.append(lambda: np.copyto(self.values,
            source.values if self.dtype is source.dtype else source.numerical()))
        self.api.operations.append(('copy', self.device, tuple(self.shape), self.api.stream.cuda_stream))
        return self

    def zero_(self):
        self.api.queued.append(lambda: self.values.fill(0))
        self.api.operations.append(('zero', tuple(self.shape)))
        return self



class StateCpuApi(DeviceCpuApi):
    def finfo(self, dtype):
        return SimpleNamespace(bits={self.bfloat16: 16, self.float16: 16,
                                     self.float32: 32, self.float8_e4m3fn: 8}[dtype])

    def empty(self, shape, *, dtype, device, pin_memory=False):
        if dtype is self.float8_e4m3fn:
            result = super().empty(shape, dtype=self.uint8, device=device, pin_memory=pin_memory)
        else:
            result = super().empty(shape, dtype=dtype, device=device, pin_memory=pin_memory)
        return StateCpuTensor(result.values, self, dtype, device, pinned=pin_memory)

    def abs(self, tensor, *, out):
        self.queued.append(lambda: np.abs(tensor.numerical(), out=out.values))
        self.operations.append(('abs', tuple(tensor.shape)))

    def eq(self, tensor, value, *, out):
        self.queued.append(lambda: np.equal(tensor.numerical(), value, out=out.values))
        self.operations.append(('eq', tuple(tensor.shape)))

    def add(self, first, second, *, out):
        self.queued.append(lambda: np.add(first.numerical(), second.numerical(), out=out.values))
        self.operations.append(('add', tuple(first.shape)))

    def amin(self, tensor, *, out):
        self.queued.append(lambda: np.copyto(out.values, np.min(tensor.numerical())))
        self.operations.append(('amin', tuple(tensor.shape)))

    def amax(self, tensor, *, out):
        self.queued.append(lambda: np.copyto(out.values, np.max(tensor.numerical())))
        self.operations.append(('amax', tuple(tensor.shape)))


@pytest.fixture
def content_case(device_case, monkeypatch):
    case = device_case
    api = StateCpuApi()
    def tensor(values, dtype):
        result = api.empty(np.array(values).shape, dtype=dtype, device='cuda:0')
        result.values[...] = values
        return result
    case.api, case.tensor = api, tensor
    case.runner.main_stream = api.stream
    # Use numerical FP32 hidden/cache tensors; BF16 input bytes remain original.
    case.runner.model_config.dtype = api.bfloat16
    case.runner.model_config.hf_text_config.fp32_residual_connection = True
    for name, value in vars(case.values['input_batch']).items():
        dtype = getattr(api, value.dtype.name)
        setattr(case.values['input_batch'], name, tensor(value.values, dtype))
    batch = case.values['input_batch']
    case.embed, case.hidden = tensor(case.embed.values, api.bfloat16), tensor(case.hidden.values, api.float32)
    case.runner.execute_model_state = SimpleNamespace(input_batch=batch, hidden_states=case.hidden)
    case.runner.model_state.request_state = tensor([2, -1], api.int64)
    table = tensor([[2, 4, 99], [0, 0, 0], [0, 0, 0]], api.int32)
    counts = tensor([[0, 0, 1]], api.int32)
    case.runner.block_tables = SimpleNamespace(cp_size=1, cp_rank=0, block_sizes=[8],
        input_block_tables=[table], num_blocks=SimpleNamespace(gpu=counts, np=counts.values))
    slots = tensor(list(range(16, 20))+[-1]*4, api.int64)
    case.values.update(block_tables=(table[:2],), model_inputs=dict(input_ids=None, positions=batch.positions,
        inputs_embeds=case.embed, intermediate_tensors=None),
        slot_mappings_by_layer={'layer.0': slots, 'layer.1': slots})
    raw = tensor(np.full((5, 2, 8, 6), np.nan), api.float32)
    raw.values[2, :, :4] = np.arange(48).reshape(2, 4, 6)
    mean = tensor(np.full((6, 1, 4), 127), api.uint8)
    mean.values[2, 0] = [56, 64, 0, 184]
    numerator, denominator = tensor(np.full((6, 4), np.nan), api.float32), tensor(np.full((6, 4), np.nan), api.float32)
    maximum = tensor(np.full((6, 1), -np.inf), api.float32)
    numerator.values[2], denominator.values[2], maximum.values[2] = [-1, 0, 3, 2], [0, 2, 1, 3], [4]
    summary = SimpleNamespace(_mean=mean, mean=mean[:5], numerator=numerator, denominator=denominator,
        maximum=maximum, last_slot=tensor([-1, -1, 19, -1, -1, -1], api.int64),
        request_journal_width=0, block_size=8, pages=5)
    context = {'layer.0': SimpleNamespace(kv_cache=raw), 'layer.1': SimpleNamespace(kv_cache=raw[:]),
        'summary.0': SimpleNamespace(main_layer_name='layer.0', kv_cache=summary.mean, _cache=summary,
            _target_impl=SimpleNamespace(summary_cache=summary))}
    spec = SimpleNamespace(block_size=8, sliding_window=3, dtype=api.float32)
    case.runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec = spec
    case.runner.compilation_config = SimpleNamespace(static_forward_context=context)
    sites = SimpleNamespace(torch=api, tensor_type=StateCpuTensor, intermediate_type=None, padding_mask_used=True,
                            census_owners={})
    class Routed:
        pass
    routed = Routed()
    routed.w13_weight_scale_inv = tensor([[0, -1], [2, 3], [4, 5]], api.float32)
    # Noncontiguous expert rows exercise copy into already owned contiguous scratch.
    routed.w2_weight_scale_inv = tensor(np.arange(12).reshape(3, 4), api.float32)[:, ::2]
    binding = dict(module_path='actual.layer.moe.experts', layer_id=7, sites=['routed_modular'])
    sites.census_owners[id(routed)] = weakref.ref(routed), binding
    routed._parameters = {name: getattr(routed, name) for name in state_observation.SCALE_PARAMETERS}
    routed._buffers, routed._non_persistent_buffers_set = {}, set()
    model = SimpleNamespace(_parameters={}, _buffers={}, _non_persistent_buffers_set=set())
    model.named_modules = lambda: [('', model), (binding['module_path'], routed)]
    case.runner.model = model
    limits = replace(case.observed.limits, scan_rows=8)
    reservations = []
    case.observed = observation.DeviceObservation(case.runner, sites, limits, case.observed.requests,
        case.observed.sampling, 65536, reserve=reservations.append, deadline=time.monotonic()+60)
    case.observed.registrations(SimpleNamespace(scheduled_new_reqs=[case.req]))
    case.observed.state.assignments[case.req.req_id] = [[2]]
    case.raw, case.summary, case.routed, case.reservations = raw, summary, routed, reservations
    case.context, case.slots = context, slots
    return case


def finish_content(case):
    index = case.observed.inputs(case.values, case.cpu, 1, 1)
    assert index is not None
    before = len(case.api.operations)
    case.observed.output(index, None, [])
    assert not any(row[0] == 'allocate' for row in case.api.operations[before:])
    assert not any(row[0] in ('query', 'synchronize') for row in case.api.operations)
    case.api.complete()
    records = []
    result = case.observed.drain(records.append)
    return result, records[0]['rows'][0]['state_content']


def test_all_applicable_cache_ranges_and_full_scale_rows_use_one_owned_event(content_case):
    case = content_case
    result, content = finish_content(case)
    dsa, swa = content['rows']
    assert dsa['expected'] == dsa['observed'] == dict(raw=48, closed_summary=0, partial_summary=4, live_tail=9)
    assert swa['logical_raw_range'] == [1, 4] and swa['observed']['raw'] == 36
    assert dsa['observed_last_slot'] == dsa['expected_last_slot'] == 19
    assert content['status'].startswith('OBSERVED_APPLICABLE_CONTENT_ONLY')
    assert content['raw_writer_and_capture_to_replay_physical_join'] == 'UNMEASURED'
    assert case.reservations == [case.observed.owned_bytes]
    scales = content['resident_scales']
    assert [row['observed'] for row in scales] == [dict(finite=6, zero=1, negative=1), dict(finite=6, zero=1, negative=0)]
    assert scales[0]['minimum'] == -1 and scales[0]['maximum'] == 5
    assert all(row['sample_policy'] == 'FIRST_RESIDENT_EXPERT_NOT_ALL_EXECUTED_EXPERTS' for row in scales)
    assert all('raw_hex' not in row['sample'] for row in result['steps'][0]['state_content']['resident_scales'])
    assert all('raw_hex' not in row for row in result['steps'][0]['state_content']['samples'])
    assert content['transient_activation_scales'] == 'UNMEASURED'
    assert not case.observed.stopped


@pytest.mark.parametrize('fault', ['raw', 'summary', 'tail', 'marker', 'position', 'seq_len', 'table', 'slot', 'scale'])
def test_actual_content_or_device_join_fault_is_retained_without_changing_native_outputs(content_case, fault):
    case = content_case
    if fault == 'raw':
        case.raw.values[2, 0, 0, 0] = np.inf
    elif fault == 'summary':
        case.summary.mean.values[2, 0, 1] = 127
    elif fault == 'tail':
        case.summary.denominator.values[2, 0] = np.nan
    elif fault == 'marker':
        case.summary.last_slot.values[2] = -1
    elif fault == 'position':
        case.values['input_batch'].positions.values[1] = 77
    elif fault == 'seq_len':
        case.values['input_batch'].seq_lens.values[0] = 5
    elif fault == 'table':
        case.values['block_tables'][0].values[0, 0] = 4
    elif fault == 'slot':
        case.slots.values[0] = 31
    else:
        # Inject before the fixture executes the already queued preparation reads.
        case.routed.w13_weight_scale_inv.values[0, 1] = np.inf
    _, content = finish_content(case)
    assert content['status'] == 'UNVERIFIED'
    assert any(row['status'].startswith('UNVERIFIED') for row in content['rows']+content['resident_scales'])
    assert not case.observed.stopped
    if fault == 'scale':
        scale = content['resident_scales'][0]
        assert scale['observed']['finite'] == 5
        assert scale['minimum'] is scale['maximum'] is None and scale['extrema_raw_hex']


def test_summary_closure_excludes_retained_accumulators_and_checks_fp8_closed_rows(content_case):
    case = content_case
    case.cpu.update(num_tokens=8, num_tokens_after_padding=8, num_scheduled_tokens=[8], query_start_loc_np=[0, 8, 8])
    batch = case.values['input_batch']
    batch.is_padding.values[:] = False
    batch.seq_lens.values[:] = [8, 0]
    batch.query_start_loc.values[:] = [0, 8, 8]
    case.slots.values[:] = list(range(16, 24))
    case.raw.values[2] = 3
    case.hidden.values[:] = 1
    case.summary.last_slot.values[2] = -1
    case.summary.numerator.values[2] = np.nan
    case.summary.maximum.values[2] = -np.inf
    _, content = finish_content(case)
    dsa = content['rows'][0]
    assert dsa['closed_summary_rows'] == 1 and dsa['open_tail_tokens'] == 0
    assert dsa['expected'] == dsa['observed'] == dict(raw=96, closed_summary=4, partial_summary=0, live_tail=0)
    assert dsa['status'].startswith('OBSERVED_FINITE_CANDIDATE_CONTENT')
    assert [row['kind'] for row in content['samples'] if row['layer'] == 'layer.0'] == ['raw_current_tail','last_closed_summary']


def test_state_samples_are_detached_before_native_page_and_scale_storage_reuse(content_case):
    case = content_case
    index = case.observed.inputs(case.values, case.cpu, 1, 1)
    case.observed.output(index, None, [])
    case.api.complete()
    case.raw.values[:] = -99
    case.summary.mean.values[:] = 0
    case.routed.w13_weight_scale_inv.values[:] = 123
    records = []
    case.observed.drain(records.append)
    content = records[0]['rows'][0]['state_content']
    raw = next(row for row in content['samples'] if row['layer'] == 'layer.0' and row['kind'] == 'raw_current_tail')
    assert np.frombuffer(bytes.fromhex(raw['raw_hex']), dtype=np.float32).tolist() == [18,19,20,21,22,23,42,43,44,45,46,47]
    assert content['resident_scales'][0]['minimum'] == -1
    assert np.frombuffer(bytes.fromhex(content['resident_scales'][0]['sample']['raw_hex']), dtype=np.float32).tolist() == [0,-1]


def test_missing_assignment_is_explicit_and_never_scans_guessed_pages(content_case):
    case = content_case
    case.observed.state.assignments.clear()
    _, content = finish_content(case)
    assert content['status'] == 'UNVERIFIED' and content['rows'] == [] and content['samples'] == []
    assert case.observed.entries[0]['state_selection'][0]['status'] == 'UNVERIFIED_NO_OBSERVED_BLOCK_ASSIGNMENT'


def test_changed_native_cache_storage_and_byte_exhaustion_stop_only_evidence(content_case):
    case = content_case
    allocations, reservations = len(case.api.operations), []
    with pytest.raises(ValueError, match='remaining rank'):
        observation.DeviceObservation(case.runner, case.observed.sites, replace(case.observed.limits, owned_bytes=1),
            case.observed.requests, case.observed.sampling, 65536, reserve=reservations.append,
            deadline=time.monotonic()+60)
    assert len(case.api.operations) == allocations and reservations == []
    index = case.observed.inputs(case.values, case.cpu, 1, 1)
    case.context['layer.0'].kv_cache = case.tensor(case.raw.values.copy(), case.api.float32)
    with pytest.raises(ValueError, match='owner changed'):
        case.observed.output(index, None, [])
    assert not any(row[0] in ('query','synchronize') for row in case.api.operations)


@pytest.mark.parametrize('fault', [None, 'dtype', 'shape'])
def test_actual_uva_count_rotation_copies_current_native_view_without_pinning_first_pointer(device_case, fault):
    case = device_case
    relative = 'native_role_sources/observed_generation2/vllm/v1/worker/gpu/buffer_utils.py'
    original = retained_methods(relative, 'UvaBackedTensor', ['copy_to_uva'], {})['copy_to_uva']
    first = case.runner.block_tables.num_blocks.gpu
    rotated = case.tensor([[0, 0, 1]], case.api.int32)
    owner = case.runner.block_tables.num_blocks
    owner.pool = SimpleNamespace(copy_to_uva=lambda values: rotated)
    owner.copy_to_uva = MethodType(original, owner)
    owner.copy_to_uva()
    assert owner.gpu is rotated and rotated.data_ptr() != first.data_ptr()
    if fault == 'dtype':
        rotated.dtype = case.api.int64
    elif fault == 'shape':
        owner.gpu = rotated[:, :2]
    index = case.observed.inputs(case.values, case.cpu, 1, 1)
    if fault is not None:
        assert index is None and case.observed.stopped
        assert case.observed.errors[0]['kind'] == 'finalized_device_inputs'
        return
    case.observed.output(index, None, [])
    case.api.complete()
    result = case.observed.drain(lambda row: None)
    assert result['steps'][0]['values']['block_counts'] == [[0, 0, 1]]
    assert result['steps'][0]['tensors']['block_counts']['source']['data_ptr'] == rotated.data_ptr()
    assert result['G2'] == 'UNVERIFIED'


@pytest.mark.parametrize('replacement', ['raw_layer', 'summary_layer', 'cache', 'target', 'transfer',
                                         *state_observation.SUMMARY_TENSORS])
def test_replaced_native_state_owner_is_rejected_before_any_content_scan(content_case, replacement):
    case = content_case
    if replacement == 'raw_layer':
        case.context['layer.0'] = SimpleNamespace(kv_cache=case.raw)
    elif replacement == 'summary_layer':
        case.context['summary.0'] = copy.copy(case.context['summary.0'])
    elif replacement == 'cache':
        case.context['summary.0']._cache = copy.copy(case.summary)
    elif replacement == 'target':
        case.context['summary.0']._target_impl.summary_cache = copy.copy(case.summary)
    elif replacement == 'transfer':
        alias = case.context['summary.0'].kv_cache
        case.context['summary.0'].kv_cache = case.tensor(alias.values.copy(), alias.dtype)
    else:
        tensor = getattr(case.summary, replacement)
        setattr(case.summary, replacement, case.tensor(tensor.values.copy(), tensor.dtype))
    index = case.observed.inputs(case.values, case.cpu, 1, 1)
    before = len(case.api.operations)
    with pytest.raises(ValueError, match='owner changed'):
        case.observed.output(index, None, [])
    # Only the existing hidden-output diagnostic can precede owner rejection.
    assert not any(row[0] == 'add' for row in case.api.operations[before:])
    assert not any(row[0] in ('query', 'synchronize') for row in case.api.operations)


@pytest.mark.parametrize('fault', ['missing_index', 'duplicate_index', 'request_state', 'query_start'])
def test_state_device_join_rejects_wrong_index_spaces_without_losing_raw_witnesses(content_case, fault):
    case = content_case
    index = case.observed.inputs(case.values, case.cpu, 1, 1)
    case.observed.output(index, None, [])
    case.api.complete()
    destinations = case.observed.destinations[index]
    if fault == 'missing_index':
        destinations['idx_mapping'].values[0] = 1
    elif fault == 'duplicate_index':
        destinations['idx_mapping'].values[:] = [2, 2, 1]
        case.observed.entries[index]['tensors']['idx_mapping']['rows'] = 3
    elif fault == 'request_state':
        destinations['request_state'].values[0] = 1
    else:
        destinations['query_start_loc'].values[0] = 1
    records = []
    case.observed.drain(records.append)
    content = records[0]['rows'][0]['state_content']
    assert content['status'] == 'UNVERIFIED' and content['samples']
    assert all(not row['device_inputs_match_candidate_range'] for row in content['rows'])
    assert content['completion'] == 'OWNED_COPY_EVENT_QUERY_READY_AFTER_NATIVE_GENERATE'


def test_group_and_layer_metadata_are_included_in_known_owner_entry_cap(device_case, monkeypatch):
    case = device_case
    torch, context, _, _ = known_runner(case, monkeypatch)
    context.clear()
    group = case.runner.kv_cache_config.kv_cache_groups[0]
    group.layer_names = ['missing.'+str(i) for i in range(100)]
    spec = group.kv_cache_spec.kv_cache_specs['layer.0']
    group.kv_cache_spec.kv_cache_specs = {name: spec for name in group.layer_names}
    with pytest.raises(ValueError, match='entry cap'):
        capacity.known_state_inventory(case.runner, torch, 10)


def test_enlarged_native_state_destination_bytes_match_all_actual_allocations(content_case):
    case = content_case
    owned = case.observed.destinations
    host = [value for row in owned for value in row.values()] + list(case.observed.state.once.values())
    device = [case.observed.absolute, case.observed.finite, case.observed.counts]
    device.extend(case.observed.state.device.values())
    def unique_bytes(tensors):
        storages = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes() for value in tensors}
        return sum(storages.values())
    assert case.observed.host_bytes == unique_bytes(host)
    assert case.observed.device_bytes == unique_bytes(device)
    assert case.observed.owned_bytes == unique_bytes(host+device)
    assert case.observed.state.device['state_absolute'].data_ptr() == case.observed.absolute.data_ptr()
    assert case.observed.state.device['state_finite'].data_ptr() == case.observed.finite.data_ptr()


def resident_case(case, parameters, buffers=None, *, limits=None, deadline=None, reserve=None):
    """Prepare actual fixture registrations through the same resident/device observer."""
    model = SimpleNamespace(_parameters=parameters, _buffers=buffers or {},
        _non_persistent_buffers_set={'temporary'})
    model.named_modules = lambda: [('', model)]
    case.runner.model = model
    sites = SimpleNamespace(torch=case.api, tensor_type=StateCpuTensor, intermediate_type=None,
                            padding_mask_used=True, census_owners={})
    current = observation.DeviceObservation(case.runner, sites, limits or case.observed.limits,
        case.observed.requests, case.observed.sampling, 65536, reserve=reserve,
        deadline=deadline if deadline is not None else time.monotonic()+60)
    current.registrations(SimpleNamespace(scheduled_new_reqs=[case.req]))
    current.state.assignments[case.req.req_id] = [[2]]
    case.observed = current
    return model


def test_registered_parameter_aliases_and_overlapping_views_have_complete_finite_coverage(content_case):
    case = content_case
    weight = case.tensor(np.arange(60).reshape(5, 12), case.api.float32)
    alias, transpose, strided = weight[:], weight[:, ::2], weight[:, ::2]
    # Aliases of one exact view are deduplicated; overlapping layouts are independently scanned.
    resident_case(case, {'weight': weight, 'alias': alias, 'strided': strided, 'strided_alias': transpose})
    result, content = finish_content(case)
    census = content['resident_parameters']
    assert census['registered_tensors'] == 4 and census['unique_tensor_views'] == 2
    assert census['unique_registered_storage_bytes'] == weight.values.nbytes
    assert [row['elements'] for row in census['rows']] == [60, 30]
    assert [row['finite_elements'] for row in census['rows']] == [60, 30]
    assert [len(row['aliases']) for row in census['rows']] == [2, 2]
    assert census['status'] == 'OBSERVED_REGISTERED_RESIDENT_FINITE_CENSUS_AT_PREPARATION_ONLY'
    assert all(row['finite_source'] == 'COMPLETE_REGISTERED_TENSOR_VIEW' for row in census['rows'])
    assert all(row['sample']['raw_hex'] == np.array([0], np.float32).tobytes().hex() for row in census['rows'])
    assert all('raw_hex' not in row['sample'] for row in result['steps'][0]['state_content']['resident_parameters']['rows'])
    assert result['G2'] == 'UNVERIFIED'


def test_resident_scalar_empty_buffer_and_fp8_bytes_preserve_their_numeric_meaning(content_case):
    case = content_case
    scalar = case.tensor(7, case.api.float32)
    fp8 = case.tensor([56, 127, 255, 184], case.api.uint8).view(case.api.float8_e4m3fn)
    empty = case.tensor(np.empty((2, 0, 3)), case.api.float32)
    resident_case(case, {'scalar': scalar, 'fp8': fp8}, {'temporary': empty})
    _, content = finish_content(case)
    rows = content['resident_parameters']['rows']
    assert [row['elements'] for row in rows] == [1, 4, 0]
    assert [row['finite_elements'] for row in rows] == [1, 2, 0]
    assert rows[0]['shape'] == [] and rows[0]['sample']['raw_hex'] == np.array([7], np.float32).tobytes().hex()
    assert rows[1]['sample']['raw_hex'] == '38'
    assert rows[1]['status'] == 'UNVERIFIED_NONFINITE_RESIDENT_TENSOR'
    assert rows[2]['chunks'] == 0 and rows[2]['sample'] is None and rows[2]['persistent'] is False
    assert content['full_resident_parameter_finite_census'] == content['status'] == 'UNVERIFIED'
    assert not case.observed.stopped


@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16', 'float8_e4m3fn'])
def test_first_resident_sample_preserves_source_bytes_with_mixed_dtype_alignment(content_case, dtype):
    case = content_case
    first = case.tensor([56], case.api.uint8).view(case.api.float8_e4m3fn)
    value = case.tensor([3, 4], getattr(case.api, dtype))
    if dtype == 'float8_e4m3fn':
        value.values[:] = [56, 64]
    resident_case(case, {'first': first, 'value': value})
    _, content = finish_content(case)
    row = content['resident_parameters']['rows'][1]
    assert row['sample_offset'] % row['sample_bytes'] == 0
    assert row['sample']['raw_hex'] == value.values[:1].tobytes().hex()
    assert row['finite_elements'] == 2


@pytest.mark.parametrize('shape,strides', [((5, 7), (1, 5)), ((3, 5, 7), (35, 1, 5)),
                                          ((1, 5, 7), (1, 14, 2)), ((97,), (2,))])
def test_noncontiguous_parameter_chunks_cover_each_logical_element_without_allocations(content_case, shape, strides):
    case = content_case
    size = sum((n-1)*stride for n, stride in zip(shape, strides))+1
    storage = np.arange(size, dtype=np.float32)
    array = np.lib.stride_tricks.as_strided(storage, shape=shape, strides=tuple(s*4 for s in strides))
    value = StateCpuTensor(array, case.api, case.api.float32, case.runner.device)
    resident_case(case, {'value': value})
    prepared = case.observed.state.parameters[0]
    before = len(case.api.operations)
    chunks = list(state_observation.parameter_chunks(value, prepared['layout']))
    assert len(case.api.operations) == before
    assert len(chunks) == prepared['layout']['chunks']
    assert all(chunk.values.size <= case.observed.limits.scan_rows*case.observed.width for chunk in chunks)
    assert np.concatenate([chunk.values.reshape(-1) for chunk in chunks]).tolist() == value.values.reshape(-1).tolist()
    _, content = finish_content(case)
    row = content['resident_parameters']['rows'][0]
    assert row['finite_elements'] == row['elements'] == value.values.size


def test_opaque_integer_parameter_storage_is_never_promoted_to_fp8_finite_qualification(content_case):
    case = content_case
    opaque = case.tensor([127, 255], case.api.uint8)
    weight = case.tensor([1, 2], case.api.float32)
    resident_case(case, {'weight': weight, 'opaque': opaque})
    _, content = finish_content(case)
    census = content['resident_parameters']
    assert census['registered_tensors'] == 2 and census['unique_tensor_views'] == 1
    assert census['status'] == 'UNVERIFIED' and content['status'] == 'UNVERIFIED'
    assert census['gaps'][0]['dtype'] == 'torch.uint8'
    assert census['gaps'][0]['reason'] == 'UNVERIFIED_RESIDENT_NUMERIC_INTERPRETATION_OR_DEVICE'


def test_resident_inverse_scale_counts_are_reused_without_a_duplicate_complete_scan(content_case):
    case = content_case
    _, content = finish_content(case)
    rows = content['resident_parameters']['rows']
    assert len(rows) == 2 and all(row['finite_source'] == 'EXISTING_RESIDENT_SCALE_REDUCTION' for row in rows)
    assert [row['finite_elements'] for row in rows] == [6, 6]
    assert [row['finite_elements'] for row in rows] == [row['observed']['finite'] for row in content['resident_scales']]
    assert not any(row[0] == 'add' for row in case.api.operations[:next(i for i, row in enumerate(case.api.operations)
                                                                             if row[0] == 'event')])


def test_resident_scan_is_submitted_only_at_preparation_and_detached_before_storage_reuse(content_case):
    case = content_case
    weight = case.tensor([1, 2, 3, 4], case.api.float32)
    resident_case(case, {'weight': weight})
    preparation = case.observed.state.resident_preparation
    assert preparation['started'] <= preparation['enqueue_finished'] < preparation['deadline']
    assert 'query' not in [row[0] for row in case.api.operations]
    case.api.complete()
    weight.values[:] = np.nan
    _, content = finish_content(case)
    row = content['resident_parameters']['rows'][0]
    assert row['finite_elements'] == 4 and row['sample']['raw_hex'] == np.array([1], np.float32).tobytes().hex()
    case.observed.registrations(SimpleNamespace(scheduled_new_reqs=[]))
    second = case.observed.inputs(case.values, case.cpu, 2, 2)
    case.observed.output(second, None, [])
    case.api.complete()
    records = []
    result = case.observed.drain(records.append)
    later = records[0]['rows'][0]['state_content']['resident_parameters']
    assert later['status'] == 'PREVIOUS_PREPARATION_WITNESS_NOT_RESCANNED' and not later['rows']
    assert result['G2'] == 'UNVERIFIED'


@pytest.mark.parametrize('deadline', [None, float('nan'), float('inf'), -1])
def test_resident_missing_or_expired_deadline_is_rejected_before_reservation(content_case, deadline):
    case = content_case
    reservations, before = [], len(case.api.operations)
    with pytest.raises((ValueError, TimeoutError), match='deadline'):
        observation.DeviceObservation(case.runner, case.observed.sites, case.observed.limits,
            case.observed.requests, case.observed.sampling, 65536, reserve=reservations.append, deadline=deadline)
    assert reservations == [] and len(case.api.operations) == before


@pytest.mark.parametrize('fault', ['model', 'module', 'tensor', 'added', 'removed', 'shape', 'persistent',
                                 'alias_shape', 'alias_storage', 'alias_dtype'])
def test_resident_owner_census_rejects_replacement_before_complete_scans(content_case, fault, monkeypatch):
    case = content_case
    weight = case.tensor(np.arange(30), case.api.float32)
    alias = weight[:]
    resident_case(case, {'weight': weight, 'alias': alias}, {'temporary': weight[:2]})
    current, model = case.observed.state, case.runner.model
    if fault == 'model':
        case.runner.model = SimpleNamespace()
    elif fault == 'module':
        model.named_modules = lambda: [('', SimpleNamespace(_parameters=model._parameters,
            _buffers=model._buffers, _non_persistent_buffers_set={'temporary'}))]
    elif fault == 'tensor':
        model._parameters['weight'] = case.tensor(np.arange(30), case.api.float32)
    elif fault == 'added':
        model._parameters['added'] = weight
    elif fault == 'removed':
        del model._parameters['weight']
    elif fault == 'shape':
        weight.values = weight.values.reshape(5, 6)
    elif fault == 'alias_shape':
        alias.values = alias.values.reshape(5, 6)
    elif fault == 'alias_storage':
        alias.values = alias.values.copy()
    elif fault == 'alias_dtype':
        alias.dtype = case.api.float8_e4m3fn
    else:
        model._non_persistent_buffers_set.clear()
    before = len(case.api.operations)
    with pytest.raises(ValueError, match='changed after'):
        current.check_parameters()
    assert len(case.api.operations) == before


def test_complete_resident_scan_deadline_expiry_stops_enqueues_without_host_wait(content_case, monkeypatch):
    case = content_case
    weight = case.tensor(np.arange(60), case.api.float32)
    calls = []
    def clock():
        calls.append(None)
        return 1 if len(calls) < 6 else 100
    monkeypatch.setattr(state_observation.time, 'monotonic', clock)
    before = len(case.api.operations)
    with pytest.raises(TimeoutError, match='original workload deadline'):
        resident_case(case, {'weight': weight}, deadline=10)
    assert any(row[0] == 'add' for row in case.api.operations[before:])
    assert not any(row[0] in ('query', 'synchronize') for row in case.api.operations[before:])


def test_parameter_destinations_and_samples_are_charged_to_single_existing_reservation(content_case):
    case = content_case
    reservations, before = [], len(case.api.operations)
    current = observation.DeviceObservation(case.runner, case.observed.sites, case.observed.limits,
        case.observed.requests, case.observed.sampling, 65536, deadline=time.monotonic()+60,
        reserve=reservations.append)
    assert reservations == [current.owned_bytes]
    host = [value for row in current.destinations for value in row.values()] + list(current.state.once.values())
    device = [current.absolute, current.finite, current.counts, *current.state.device.values()]
    def unique(values):
        return sum({value.untyped_storage().data_ptr(): value.untyped_storage().nbytes() for value in values}.values())
    assert unique(host) == current.host_bytes and unique(device) == current.device_bytes
    assert unique(host+device) == current.owned_bytes
    assert not any(row[0] in ('query', 'synchronize') for row in case.api.operations[before:])


def test_repeated_resident_submission_cannot_overwrite_preparation_witness(content_case):
    case = content_case
    current = case.observed.state
    witness = copy.deepcopy(current.resident_preparation)
    before = len(case.api.operations)
    with pytest.raises(ValueError, match='only once'):
        current.submit_resident(time.monotonic()+60)
    assert current.resident_preparation == witness
    assert len(case.api.operations) == before


def worker_preparation(case, tmp_path, monkeypatch, deadline, *, result=None, failure=None):
    from test_role_execution_observation import make_worker
    records = tmp_path/'records'
    records.mkdir()
    node = tmp_path/'node.json'
    node.write_text(json.dumps(dict(observer_output=str(records), record_bytes=8*65536,
                                   workload_deadline=deadline)))
    monkeypatch.setenv('STEP5_CAPACITY_NODE_FILE', str(node))
    worker, calls, _ = make_worker((records, node), result=result, failure=failure)
    worker.model_runner = case.runner
    observed = SimpleNamespace(device=None, sites=case.observed.sites,
        limits=case.observed.cpu_limits, stopped=False, last_record=None)
    case.runner._model_input_observation = observed
    preparation = dict(limits=vars(case.observed.limits), requests=case.observed.requests,
                       sampling_fields=case.observed.sampling)
    return worker, calls, observed, preparation


def test_worker_resident_preparation_uses_original_node_workload_deadline(content_case, tmp_path, monkeypatch):
    case = content_case
    deadline = time.monotonic()+60
    worker, _, observed, preparation = worker_preparation(case, tmp_path, monkeypatch, deadline)
    result = execution.prepare_device_observation(worker, preparation)
    assert result['status'] == 'PREPARED_DIAGNOSTIC_DEVICE_WITNESSES_NOT_G2'
    assert observed.device.state.resident_preparation['deadline'] == deadline
    assert result['preparation'] == preparation
    assert worker._request_observation.device_owned_bytes == observed.device.owned_bytes
    assert not worker._request_observation.collection_stopped


@pytest.mark.parametrize('fault', ['missing_deadline', 'expired_deadline', 'scan_failure'])
@pytest.mark.parametrize('native_failure', [False, True])
def test_resident_preparation_failure_preserves_original_native_result_or_exception(
        content_case, tmp_path, monkeypatch, fault, native_failure):
    from test_role_execution_observation import scheduler
    case = content_case
    deadline = None if fault == 'missing_deadline' else -1 if fault == 'expired_deadline' else time.monotonic()+60
    sentinel = SystemExit('Original native failure') if native_failure else object()
    worker, calls, observed, preparation = worker_preparation(case, tmp_path, monkeypatch, deadline,
        result=None if native_failure else sentinel, failure=sentinel if native_failure else None)
    if fault == 'scan_failure':
        def fail_scan(self, entry, original_deadline):
            assert original_deadline == deadline
            raise KeyboardInterrupt('Injected resident enqueue failure')
        monkeypatch.setattr(state_observation.StateObservation, 'scan_parameters', fail_scan)
    before = len(case.api.operations)
    result = execution.prepare_device_observation(worker, preparation)
    state = worker._request_observation
    assert result['status'] == 'UNVERIFIED'
    assert result['error_type'].endswith('KeyboardInterrupt' if fault == 'scan_failure' else
                                        'TimeoutError' if fault == 'expired_deadline' else 'ValueError')
    assert observed.device is None and observed.stopped and state.collection_stopped
    if fault == 'scan_failure':
        assert state.device_owned_bytes > 0
    else:
        assert state.device_owned_bytes == 0 and len(case.api.operations) == before
    assert not any(row[0] in ('query', 'synchronize') for row in case.api.operations[before:])
    work = scheduler()
    if native_failure:
        with pytest.raises(SystemExit) as caught:
            worker.execute_model(work)
        assert caught.value is sentinel and state.native_failures == 1
    else:
        assert worker.execute_model(work) is sentinel and state.native_failures == 0
    assert len(calls) == 1 and calls[0] == ((work,), {})
