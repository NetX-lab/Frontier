"""CPU seam tests only: these do not qualify a native PD transfer."""
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import struct
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest


ROOT = Path(__file__).resolve().parents[2]
PD = ROOT/'tests/integration/step5_pd'
SPEC = importlib.util.spec_from_file_location('step5_r2_observation', PD/'r2_observation.py')
r2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r2)


def configuration(path, role='decode', **changes):
    result = dict(role=role, output_dir=str(path), selected_client_request_ids=['case'],
                  expected_tp_ranks=8, expected_layer_count=2,
                  expected_summary_layer_count=1, region_size=8, block_size=16,
                  max_record_bytes=64*1024**2, max_capture_bytes_per_request=64*1024**2,
                  max_events=50000, max_capture_seconds_per_request=60,
                  source_hashes={}, frozen_environment_hashes={
                      name: hashlib.sha256((ROOT/'tests/integration/step5_environment'/name).read_bytes()).hexdigest()
                      for name in ('role_state_observation.py', 'role_capacity_observation.py')})
    result.update(changes)
    return result


def events(observer):
    return [json.loads(line) for path in observer.output.glob('*.jsonl')
            for line in path.read_text().splitlines()]


def backend_sources(root, monkeypatch, config, modules):
    """Host-only imported-source fixtures with the native object relationships."""
    from tests.integration.step5_pd import r2_runtime
    sources = {
        0: ('class Fp8Config: pass\nclass Fp8MoEMethod:\n    def apply(self): pass\n'
            '    @property\n    def is_monolithic(self): return self.moe_kernel.is_monolithic\n'
            'from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped\n'),
        1: 'def is_layer_skipped(prefix, ignored_layers, **kw): return prefix in ignored_layers\n',
        2: ('class UnquantizedFusedMoEMethod:\n    def apply(self): pass\n'
            '    @property\n    def is_monolithic(self): return self.moe_kernel.is_monolithic\n'),
        3: ('class FusedMoEExpertsModular:\n    @staticmethod\n    def is_monolithic(): return False\n'
            'class FusedMoEPrepareAndFinalizeModular: pass\n'
            'class FusedMoEKernelModularImpl:\n    def apply(self): pass\n'
            'class FusedMoEKernelMonolithicImpl: pass\n'
            'class FusedMoEKernel:\n    def apply(self): pass\n'
            '    @property\n    def is_monolithic(self): return isinstance(self.impl, FusedMoEKernelMonolithicImpl)\n'
            '    @property\n    def fused_experts(self): return self.impl.fused_experts\n'
            '    @property\n    def prepare_finalize(self): return self.impl.prepare_finalize\n'),
        5: ('from enum import Enum\nclass Fp8MoeBackend(Enum): DEEPGEMM = 1\n'
            'def backend_to_kernel_cls(backend):\n    from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import TritonOrDeepGemmExperts\n    return [TritonOrDeepGemmExperts]\n'),
        6: ('from enum import Enum\nclass UnquantizedMoeBackend(Enum): TRITON = 1\n'
            'def backend_to_kernel_cls(backend):\n    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts\n    return [TritonExperts]\n'),
        7: ('from vllm.model_executor.layers.fused_moe.experts.fallback import FallbackExperts\n'
            'class TritonOrDeepGemmExperts(FallbackExperts):\n'
            '    @staticmethod\n    def get_clses():\n'
            '        from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import DeepGemmExperts\n'
            '        from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts\n'
            '        return DeepGemmExperts, TritonExperts\n'),
        8: 'class DeepGemmExperts(FusedMoEExpertsModular):\n    def apply(self): pass\n',
        9: 'class TritonExperts(FusedMoEExpertsModular):\n    def apply(self): pass\n',
        10: 'class FallbackExperts(FusedMoEExpertsModular):\n    def apply(self): pass\n',
        11: 'class MoEPrepareAndFinalizeNaiveDPEPModular(FusedMoEPrepareAndFinalizeModular): pass\n',
    }
    for index in (1, 0, 2, 3, 4, 5, 6, 8, 9, 10, 7, 11):
        name = r2_runtime.BACKEND_MODULES[index]
        source = sources.get(index, '')
        if index in (8, 9, 10, 11):
            source = 'from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEExpertsModular, FusedMoEPrepareAndFinalizeModular\n'+source
        path = root/(name.removeprefix('vllm.').replace('.', '/')+'.py')
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(source)
        module = ModuleType(name); module.__file__, module.__spec__ = str(path), NS(origin=str(path))
        monkeypatch.setitem(sys.modules, name, module); modules[name] = module
        exec(compile(source, str(path), 'exec'), module.__dict__)
        config['schema_source_hashes'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return [modules[name] for name in r2_runtime.BACKEND_MODULES]


@pytest.mark.parametrize('fault', ['none', 'file', 'spec', 'hash', 'missing', 'forward',
                                  'instance_forward', 'quant', 'kernel', 'monolithic',
                                  'same_name', 'backend', 'apply', 'backend_source'])
@pytest.mark.parametrize('native_fails', [False, True])
@pytest.mark.parametrize('module_index', [-4, -3, -2, -1])
def test_derived_actual_import_model_provenance_remains_soft(tmp_path, monkeypatch, fault, native_fails, module_index):
    from tests.integration.step5_pd import r2_runtime
    root = tmp_path/'installed/vllm'
    config = configuration(tmp_path, routing_strategy='position_cyclic',
                           installed_native_root=str(root), expected_layer_count=1)
    config['schema_source_hashes'] = {}
    modules = {}
    for name in r2_runtime.SOURCE_MODULES:
        path = root/(name.removeprefix('vllm.').replace('.', '/')+'.py')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('class Step5ForCausalLM:\n    def forward(self):\n        return 1\n')
        module = ModuleType(name)
        module.__file__, module.__spec__ = str(path), NS(origin=str(path))
        monkeypatch.setitem(sys.modules, name, module)
        exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
        modules[name] = module
        config['schema_source_hashes'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    name = 'vllm.models.step5.nvidia.model'
    model = modules[name]
    model._get_step5_moe_layer_indices = lambda config: {3}
    broken = r2_runtime.SOURCE_MODULES[module_index]
    selected = modules[broken]
    if fault in ('file', 'spec'):
        foreign = tmp_path/'shadow.py'; foreign.write_bytes(Path(selected.__file__).read_bytes())
        if fault == 'file': selected.__file__ = str(foreign)
        else: selected.__spec__.origin = str(foreign)
    elif fault == 'hash': config['schema_source_hashes'][broken] = '0'*64
    elif fault == 'missing': monkeypatch.delitem(sys.modules, broken)
    elif fault == 'forward': model.Step5ForCausalLM.forward = lambda self: 1
    env = ModuleType('vllm.envs')
    env.VLLM_MOE_ROUTING_SIMULATION_STRATEGY, env.VLLM_MOE_SKIP_PADDING = 'position_cyclic', True
    parent = ModuleType('vllm'); parent.envs = env
    monkeypatch.setitem(sys.modules, 'vllm', parent); monkeypatch.setitem(sys.modules, 'vllm.envs', env)
    backend = backend_sources(root, monkeypatch, config, modules)
    quant_config = backend[0].Fp8Config()
    quant_config.ignored_layers, quant_config.packed_modules_mapping = ['layer.bf16'], {}
    quant_config.ignored_layers_match_mode = 'exact'
    router = type('RoutingSimulatorRouter', (), {})()
    modules[r2_runtime.SOURCE_MODULES[-1]].RoutingSimulatorRouter = type(router)
    router.top_k, router.global_num_experts, router.renormalize = 8, 352, True
    router.routing_method_type, router.routed_scaling_factor = 'Simulated', 1.25
    layers = []
    for skip in (False, True):
        quant = (backend[2].UnquantizedFusedMoEMethod if skip else backend[0].Fp8MoEMethod)()
        cls = backend[9].TritonExperts if skip else backend[7].TritonOrDeepGemmExperts
        quant.experts_cls = cls
        if skip: quant.unquantized_backend = backend[6].UnquantizedMoeBackend.TRITON
        else: quant.fp8_backend = backend[5].Fp8MoeBackend.DEEPGEMM
        kernel = backend[3].FusedMoEKernel(); kernel.impl = backend[3].FusedMoEKernelModularImpl()
        kernel.impl.fused_experts = cls()
        kernel.impl.prepare_finalize = backend[11].MoEPrepareAndFinalizeNaiveDPEPModular()
        if not skip:
            kernel.fused_experts.experts, kernel.fused_experts.fallback_experts = backend[8].DeepGemmExperts(), backend[9].TritonExperts()
        quant.moe_kernel = kernel
        experts = NS(layer_name='layer.bf16' if skip else 'layer.fp8', router=router, routed_experts=NS(quant_method=quant))
        layers.append(NS(experts=experts, use_sequence_parallel=False))
    state = type('State', (), {'prepare_attn': lambda self: None})()
    text = model.Step5ForCausalLM()
    text.config, text.num_moe_layers = NS(moe_router_activation='sigmoid', moe_top_k=8,
        moe_num_experts=352, norm_expert_weight=True, moe_router_scaling_factor=1.25), 2
    model._get_step5_moe_layer_indices = lambda config: {3, 88}
    if fault == 'instance_forward': text.forward = lambda: 1
    elif fault == 'quant': experts.routed_experts.quant_method = object()
    elif fault == 'kernel': experts.routed_experts.quant_method.moe_kernel.impl.fused_experts = object()
    elif fault == 'monolithic': experts.routed_experts.quant_method.moe_kernel.impl = backend[3].FusedMoEKernelMonolithicImpl()
    elif fault == 'same_name':
        cls = type(experts.routed_experts.quant_method)
        replacement = type(cls.__name__, (), {'__module__': cls.__module__})()
        replacement.__dict__.update(experts.routed_experts.quant_method.__dict__)
        experts.routed_experts.quant_method = replacement
    elif fault == 'backend': experts.routed_experts.quant_method.unquantized_backend = 'TRITON'
    elif fault == 'apply': experts.routed_experts.quant_method.moe_kernel.apply = lambda *a, **kw: None
    elif fault == 'backend_source':
        selected = backend[9]; foreign = tmp_path/'foreign_backend.py'
        foreign.write_bytes(Path(selected.__file__).read_bytes()); selected.__file__ = str(foreign)
    worker = NS(model_runner=NS(model_state=state, vllm_config=NS(kernel_config=NS(moe_backend='auto'),
        quant_config=quant_config, parallel_config=NS(enable_eplb=False)), model=NS(modules=lambda: [text]+layers)))
    observer = r2.Observation(config)
    monkeypatch.setattr(observer, 'identity', lambda *args: dict(role='decode', dp_rank=0, tp_rank=0))
    calls = []
    class Native:
        def load(self):
            calls.append(1)
            if native_fails: raise LookupError('native failure')
            return 17
    r2.wrap(Native, 'load', observer, after=lambda *args: r2_runtime.bind_model_state(worker, observer, r2.wrap))
    if native_fails:
        with pytest.raises(LookupError, match='native failure'): Native().load()
        assert not events(observer)
    else:
        assert Native().load() == 17
        assert observer.disabled == (fault != 'none')
        assert events(observer)[-1]['event'] == ('observer_error' if fault != 'none' else 'provenance')
        if fault == 'none':
            census = events(observer)[-1]['layers']
            assert [row['precision'] for row in census] == ['fp8', 'bf16']
            assert all(row['sources'] and not row['monolithic'] for row in census)
    assert calls == [1]


class Owner:
    pass


def release_runtime(observer, pool_hook=True):
    """CPU originals with the locked scheduler's queue/pop/iterator call shape."""
    from collections import deque

    class BlockPool:
        def __init__(self):
            self.free_count = 6
            self.calls, self.iterators, self.consumed = 0, [], []
            self.fail = False
            self.blocks = {}

        def get_num_free_blocks(self):
            return self.free_count

        def free_blocks(self, ordered_blocks):
            self.calls += 1
            self.iterators.append(ordered_blocks)
            if self.fail:
                raise LookupError('native pool free failure')
            consumed = []
            for block in ordered_blocks:
                consumed.append(block.block_id)
                block.ref_cnt -= 1
                if not block.is_null and block.ref_cnt == 0:
                    self.free_count += 1
            self.consumed.append(consumed)

    class Manager:
        def __init__(self):
            self.block_pool = BlockPool()
            self.coordinator = Owner()
            self.groups = {}
            self.calls = 0

        def get_block_ids(self, request_id):
            return [[block.block_id for block in group] for group in self.groups[request_id]]

        def pop_blocks_for_free(self, request):
            return [block for group in self.groups.pop(request.request_id) for block in group]

        def free(self, request):
            self.calls += 1
            self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request)))

    class Scheduler:
        def __init__(self):
            self.kv_cache_manager = Manager()
            self.deferred_frees = deque()
            self.sched_step_seq, self.processed_step_seq, self.defer_block_free = 10, 9, True
            self.iterators = []
            self.drain_calls = 0

        def _free_request_blocks(self, request):
            if not self.defer_block_free or request.last_sched_seq <= self.processed_step_seq:
                self.kv_cache_manager.free(request)
                return
            blocks = self.kv_cache_manager.pop_blocks_for_free(request)
            if blocks:
                self.deferred_frees.append((self.sched_step_seq, blocks))

        def _drain_deferred_frees(self):
            self.drain_calls += 1
            while self.deferred_frees:
                fence, _ = self.deferred_frees[0]
                if fence > self.processed_step_seq:
                    break
                _, blocks = self.deferred_frees.popleft()
                iterator = reversed(blocks)
                self.iterators.append(iterator)
                self.kv_cache_manager.block_pool.free_blocks(iterator)

    r2.wrap(Scheduler, '_free_request_blocks', observer,
            before=observer.scheduler_free_before, after=observer.scheduler_free_after)
    r2.wrap(Scheduler, '_drain_deferred_frees', observer,
            before=observer.drain_before, after=observer.drained, drain_scope=True)
    r2.wrap(Manager, 'free', observer, before=observer.pool_before, after=observer.freed_blocks)
    if pool_hook:
        r2.wrap(BlockPool, 'free_blocks', observer, before=observer.pool_free_before, after=observer.pool_freed)
    scheduler = Scheduler()
    observer.scheduler_ready(scheduler, None, {}, None)

    def request(request_id, groups, status='FINISHED_STOPPED'):
        native_groups = []
        for group in groups:
            native = []
            for block_id in group:
                block = scheduler.kv_cache_manager.block_pool.blocks.setdefault(
                    block_id, NS(block_id=block_id, is_null=block_id == 0, ref_cnt=0))
                block.ref_cnt += 1
                native.append(block)
            native_groups.append(native)
        scheduler.kv_cache_manager.groups[request_id] = native_groups
        return NS(request_id=request_id, status=status, last_sched_seq=10)

    return scheduler, request


class Bytes:
    def __init__(self, value, shape=None):
        self.value = bytes(value)
        self.shape = tuple(shape or (len(self.value),))
        self.dtype, self.device = 'uint8', 'cuda:0'

    def __getitem__(self, key):
        return Bytes(self.value[key])

    def contiguous(self):
        return self

    def is_contiguous(self):
        return True

    def detach(self):
        return self

    def numel(self):
        return len(self.value)

    def element_size(self):
        return 1

    def untyped_storage(self):
        return NS(data_ptr=lambda: id(self), nbytes=lambda: len(self.value))

    def storage_offset(self):
        return 0

    def stride(self):
        values, stride = [], 1
        for size in reversed(self.shape):
            values.insert(0, stride)
            stride *= size
        return tuple(values)

    def view(self, dtype):
        return self

    def reshape(self, *shape):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self

    def tobytes(self):
        return self.value


class Tensor:
    def __init__(self, rows, dtype, packed=True):
        self.rows, self.dtype = rows, dtype
        self.shape = (len(rows), 1, 2, 2) if packed else (len(rows), 4)
        self.device = 'cuda:0'

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Tensor(self.rows[index], self.dtype, len(self.shape) == 4)
        row = Bytes(self.rows[index], self.shape[1:])
        row.dtype = self.dtype
        return row

    def numel(self):
        return len(self.rows)*4

    def data_ptr(self):
        return id(self)

    def stride(self, index=None):
        values = (4, 4, 2, 1) if len(self.shape) == 4 else (4, 1)
        return values if index is None else values[index]

    def element_size(self):
        return 1

    def untyped_storage(self):
        return NS(data_ptr=lambda: id(self), nbytes=self.numel)

    def storage_offset(self):
        return 0

    def detach(self):
        return self

    def contiguous(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *shape):
        return Bytes(b''.join(self.rows))


class Numeric:
    def __init__(self, value=1):
        self.value = value
        self.shape, self.dtype, self.device = (1,), 'float32', 'cuda:0'

    def __getitem__(self, index):
        return self

    def detach(self):
        return self

    def contiguous(self):
        return self

    def reshape(self, *shape):
        return self

    def view(self, dtype):
        return Bytes(struct.pack('<f', self.value))

    def stride(self):
        return (1,)

    def storage_offset(self):
        return 0

    def untyped_storage(self):
        return NS(data_ptr=lambda: id(self), nbytes=lambda: 4)

    def cpu(self):
        return self

    def numel(self):
        return 1

    def element_size(self):
        return 4

    def sum(self):
        return self

    def item(self):
        return self.value


@pytest.fixture
def torch(monkeypatch):
    module = ModuleType('torch')
    module.uint8 = type('Dtype', (), {'is_floating_point': False, '__str__': lambda self: 'uint8'})()
    for name in ('int32', 'int64', 'bool'):
        setattr(module, name, object())
    module.syncs = []
    module.cuda = NS(synchronize=lambda: module.syncs.append('sync'))
    module.isfinite = lambda tensor: Numeric()
    monkeypatch.setitem(sys.modules, 'torch', module)
    # These are the unchanged stage primitives, not reimplemented digest/storage functions.
    monkeypatch.syspath_prepend(str(ROOT/'tests/integration/step5_environment'))
    return module


def connector(torch):
    value = Owner()
    cache = NS(last_slot=Numeric(-1), numerator=Numeric(), denominator=Numeric(), maximum=Numeric(),
               request_journal_width=0)
    context = {'summary': NS(main_layer_name='full', _cache=cache,
                            kv_cache_side_storage_transfer_view=lambda: None)}
    value.vllm_config = NS(parallel_config=NS(data_parallel_rank=1), kv_transfer_config=None,
                          compilation_config=NS(static_forward_context=context))
    value.engine_id, value.tp_rank = 'decode_dp1', 3
    value.kv_cache_config = NS(transfer_group_index_by_layer={'full': 0, 'swa': 1},
                               transfer_group_ids=(0, 1), kv_cache_groups=[
                                   NS(layer_names=['full'], kv_cache_spec=NS(block_size=16, sliding_window=None)),
                                   NS(layer_names=['swa'], kv_cache_spec=NS(block_size=16, sliding_window=512))])
    value.region_names = ['full', 'swa', 'summary']
    value.region_group_ids = [0, 1, 0]
    value.block_len_per_layer = [4, 4, 4]
    value.block_stride_per_layer = [4, 4, 4]
    value.region_num_blocks = [4, 4, 4]
    value._physical_blocks_per_logical_kv_block = 1
    value.device_kv_caches = {
        'full': Tensor([b'ABCD', b'EFGH', b'IJKL', b'MNOP'], torch.uint8),
        'swa': Tensor([b'QRST', b'UVWX', b'YZ12', b'3456'], torch.uint8),
        'summary': Tensor([b'abcd', b'efgh', b'ijkl', b'mnop'], torch.uint8, False)}
    value.kv_caches_base_addr = {value.engine_id: {value.tp_rank: [
        value.device_kv_caches[name].data_ptr() for name in value.region_names]}}
    value._recving_transfers = {}
    value._recv_failures = set()
    return value


@pytest.mark.parametrize('role', ['prefill', 'decode'])
@pytest.mark.parametrize('outcome', ['value', 'none', 'raise'])
@pytest.mark.parametrize('observation', ['success', 'error', 'budget'])
@pytest.mark.parametrize('phase', ['before', 'after'])
def test_observation_preserves_original_once(tmp_path, capsys, role, outcome, observation, phase):
    observer = r2.Observation(configuration(tmp_path, role=role, max_events=2))
    sentinel = object()
    calls = []

    class Native:
        def run(self, value):
            calls.append(value)
            if outcome == 'raise':
                raise LookupError('native failure')
            return None if outcome == 'none' else sentinel

    def observe(*args):
        if observation == 'error':
            raise ValueError('observer failure')
        observer.emit('observation')
        if observation == 'budget':
            observer.emit('over_budget')

    r2.wrap(Native, 'run', observer, **{phase: observe})
    if outcome == 'raise':
        with pytest.raises(LookupError, match='native failure'):
            Native().run(7)
    else:
        assert Native().run(7) is (None if outcome == 'none' else sentinel)
    assert calls == [7]
    if phase == 'before' or outcome != 'raise':
        expected = {'error': 'observer_error', 'budget': 'observer_budget_exceeded',
                    'success': 'observation'}[observation]
        assert events(observer)[-1]['event'] == expected
        if observation != 'success':
            assert 'UNVERIFIED' in capsys.readouterr().err


@pytest.mark.parametrize('limit,changes', [
    ('event', {'max_events': 2}), ('bytes', {'max_record_bytes': 2400}),
])
def test_budget_is_terminal_and_never_stops_native(tmp_path, capsys, limit, changes):
    observer = r2.Observation(configuration(tmp_path, **changes))
    calls = []

    class Native:
        def run(self):
            calls.append(1)
            return 17

    def after(*args):
        observer.emit('record', payload='x'*1000)

    r2.wrap(Native, 'run', observer, after=after)
    for _ in range(4):
        assert Native().run() == 17
    assert len(calls) == 4 and observer.disabled
    assert events(observer)[-1]['event'] == 'observer_budget_exceeded'
    assert observer.written <= observer.config['max_record_bytes']
    assert observer.events <= observer.config['max_events']
    assert 'UNVERIFIED' in capsys.readouterr().err


def test_registration_and_full_wire_row_sha256_use_frozen_primitives(tmp_path, torch):
    observer = r2.Observation(configuration(tmp_path))
    source, target = connector(torch), connector(torch)
    source.engine_id = 'prefill_dp0'
    source.kv_caches_base_addr[source.engine_id] = source.kv_caches_base_addr['decode_dp1']
    observer.register(source, None, {}, None)
    observer.register(target, None, {}, None)
    target.device_kv_caches['full'].rows[2] = source.device_kv_caches['full'].rows[0]
    target.device_kv_caches['swa'].rows[3] = source.device_kv_caches['swa'].rows[1]
    target.device_kv_caches['summary'].rows[2] = source.device_kv_caches['summary'].rows[0]
    observer.capture(source, 'case', [[0, 1], [0, 1]], 'source_content')
    observer.capture(target, 'case', [[2], [3]], 'target_content', [[0], [1]],
                     remote_engine_id=source.engine_id, remote_request_id='case')
    rows = events(observer)
    inventory = next(r for r in rows if r['event'] == 'registration')
    assert inventory['expected_raw_layers'] == {'full': 0, 'swa': 1}
    assert inventory['expected_summary_layers'] == {'summary': 'full'}
    assert inventory['expected_layer_count'] == 2
    assert inventory['expected_summary_layer_count'] == 1
    assert inventory['group_specs'] == [
        dict(group_id=0, native_group_id=0, block_size=16, sliding_window=None),
        dict(group_id=1, native_group_id=1, block_size=16, sliding_window=512)]
    assert all(row['tensor']['storage'] == row['alias_storage'] for row in inventory['registrations'])
    assert inventory['unique_resident_storage_bytes'] == 48
    for row in (r for r in rows if r['event'] == 'target_content'):
        src = next(r for r in rows if r['event'] == 'source_content' and
                   (r['layer'], r['component']) == (row['layer'], row['component']))
        source_index = src['local_block_ids'].index(row['remote_block_ids'][0])
        assert row['block_hashes'][0] == src['block_hashes'][source_index]
        assert row['local_block_ids'] != row['remote_block_ids']
        assert row['block_bytes'] == 4 and row['digest_scope'] == 'complete_physical_wire_row'
    assert next(r for r in rows if r['event'] == 'source_content')['block_hashes'][0] == hashlib.sha256(b'ABCD').hexdigest()
    assert 'raw_hex' not in json.dumps(rows)
    # A byte beyond a hypothetical small sample still changes the complete-row witness.
    target.device_kv_caches['full'].rows[2] = b'ABC!'
    observer.capture(target, 'case', [[2], []], 'target_content', [[0], []])
    assert [r for r in events(observer) if r['event'] == 'target_content'][-2]['block_hashes'][0] != hashlib.sha256(b'ABCD').hexdigest()


def test_source_capture_retains_full_sparse_history_but_clips_native_swa(tmp_path, torch):
    observer = r2.Observation(configuration(tmp_path, role='prefill'))
    value = connector(torch)
    observer.register(value, None, {}, None)
    request = dict(blocks=[list(range(1, 98)), list(range(101, 198))])
    groups = observer.source_blocks(value, request)
    assert groups[0] == request['blocks'][0]
    assert groups[1] == request['blocks'][1][-33:]
    assert len(request['blocks'][1]) == 97
    assert not torch.syncs


def test_grouped_native_raw_alias_capture_has_a_registered_physical_row(tmp_path, torch):
    """R2 stage02: SWA layer views alias a named full-history registration."""
    observer = r2.Observation(configuration(tmp_path))
    value = connector(torch)
    value.device_kv_caches['swa'] = value.device_kv_caches['full']
    value.region_names = ['full', 'summary']
    value.region_group_ids = [-1, 0]
    value.block_len_per_layer = [4, 4]
    value.block_stride_per_layer = [4, 4]
    value.region_num_blocks = [4, 4]
    value.kv_caches_base_addr[value.engine_id][value.tp_rank] = [
        value.device_kv_caches[name].data_ptr() for name in value.region_names]
    observer.register(value, None, {}, None)
    observer.capture(value, 'case', [[1], [2]], 'source_content')
    raw = [row for row in events(observer) if row['event'] == 'source_content' and row['component'] == 'raw_kv']
    assert {row['layer'] for row in raw} == {'full', 'swa'}
    assert raw[1]['block_hashes'] == [hashlib.sha256(b'IJKL').hexdigest()]


def test_capture_budget_prevents_copy_and_preserves_native(tmp_path, torch):
    observer = r2.Observation(configuration(tmp_path, max_capture_bytes_per_request=3))
    value = connector(torch)
    observer.register(value, None, {}, None)
    calls = []

    class Native:
        def run(self):
            calls.append('native')
            return 99

    def after(*args):
        observer.capture(value, 'case', [[0], []], 'source_content')

    r2.wrap(Native, 'run', observer, after=after)
    assert Native().run() == 99 and calls == ['native']
    assert events(observer)[-1]['event'] == 'observer_budget_exceeded'
    assert not any(r['event'] == 'source_content' for r in events(observer))


def test_exact_descriptor_mapping_keeps_native_swa_suffix(tmp_path, torch):
    observer = r2.Observation(configuration(tmp_path))
    value = connector(torch)
    observer.register(value, None, {}, None)
    req = 'cmpl-case-0-1234abcd'
    remote = NS(engine_id='prefill_dp0', request_id='cmpl-case-0-abcdef12',
                host='p-host', port=123, num_tokens=256, block_ids=[[0, 1, 2], [0, 1, 2]])
    meta = NS(remote=remote, local_block_ids=[[2, 3], [3]], local_physical_block_ids=[[2, 3], [3]])
    observer.read_request(value, dict(req_id=req, meta=meta))
    native_calls = []

    class Pull:
        def _compute_desc_ids(self, block_ids):
            native_calls.append(('desc', r2.block_lists(block_ids)))
            return NS(tolist=lambda: [101, 102])

        def _read_blocks(self, read_spec, dst_engine_id, request_id, remote_request_id):
            native_calls.append(('read', request_id))
            # This models the locked method's actual clipping before descriptor creation.
            self._compute_desc_ids([[1, 2], [2]])
            self._compute_desc_ids([[2, 3], [3]])
            return True

    # Reuse the registered fake as an instance of the instrumented native class.
    value.__class__ = Pull
    r2.wrap(Pull, '_compute_desc_ids', observer, after=observer.descriptors)
    r2.wrap(Pull, '_read_blocks', observer, before=observer.read_context,
            after=observer.read_submitted, read_scope=True)
    spec = NS(remote_rank=3, local_block_ids=[[2, 3], [3]],
              remote_block_ids=[[0, 1, 2], [0, 1, 2]], block_ids_by_region=False)
    assert value._read_blocks(spec, 'prefill_dp0', req, remote.request_id) is True
    assert len(native_calls) == 3
    row = events(observer)[-1]
    assert row['event'] == 'read_submit'
    assert row['remote_block_ids'] == [[1, 2], [2]]
    assert row['local_block_ids'] == [[2, 3], [3]]
    assert row['remote_request_id'] == remote.request_id and row['remote_tp_rank'] == 3
    assert observer.connector_states[value]['reads'][req]['metadata']['advertised_remote_block_ids'] == [[0, 1, 2], [0, 1, 2]]
    assert observer.local.read is None


def test_network_gpu_visibility_and_notification_are_distinct(tmp_path, torch):
    observer = r2.Observation(configuration(tmp_path))
    value = connector(torch)
    observer.register(value, None, {}, None)
    observer.connector_states[value]['reads']['case'] = dict(
        metadata=dict(remote_request_id='source-case', remote_engine_id='prefill_dp0',
                      local_physical_block_ids=[[], []]), plans=[])
    observer.network_done(value, ({'case'}, set()), {'transfers': value._recving_transfers}, None)
    assert not torch.syncs
    observer.notified(value, {'case'}, {}, None)
    assert not torch.syncs
    observer.transfer_results(value, NS(finished_recving={'case'}, failed_recving=set()), {}, None)
    assert torch.syncs == ['sync']
    names = [r['event'] for r in events(observer)]
    assert names[-5:] == ['read_network_done', 'source_notified', 'native_read_ready',
                         'read_gpu_visible', 'capture_elapsed']
    native_ready = next(r for r in events(observer) if r['event'] == 'native_read_ready')
    fenced = next(r for r in events(observer) if r['event'] == 'read_gpu_visible')
    assert native_ready['observer_fence'] is False and fenced['observer_fence'] is True
    # Native expiry appearing in finished_sending alone is NOT a notification.
    observer.transfer_results(value, NS(finished_sending={'case'}, finished_recving=set(), failed_recving=set()), {}, None)
    assert [r['event'] for r in events(observer)].count('source_notified') == 1


def test_actual_collector_native_ready_records_satisfy_offline_verdict(tmp_path, torch):
    from tests.integration.step5_pd import r2_verdict

    observer = r2.Observation(configuration(tmp_path, content_client_request_ids=[]))
    for rank in range(8):
        value = connector(torch)
        value.tp_rank = rank
        value.kv_caches_base_addr[value.engine_id][rank] = value.kv_caches_base_addr[value.engine_id][3]
        observer.register(value, None, {}, None)
        observer.connector_states[value]['reads']['case'] = dict(metadata={}, plans=[])
        observer.transfer_results(value, NS(finished_recving={'case'}, failed_recving=set()), {}, None)
    ready = [row for row in events(observer) if row['event'] == 'native_read_ready']
    assert len(ready) == 8 and not torch.syncs
    target = []
    for row in ready:
        target.extend([
            dict(row, event='read_network_done', monotonic_ns=row['monotonic_ns']-1),
            row,
            dict(row, event='target_compute', monotonic_ns=row['monotonic_ns']+1)])
    check = r2_verdict.Check()
    assert r2_verdict.check_native_readiness([], target, check, no_fence=True), check.result()


@pytest.mark.parametrize('conflict', [None, 'tokens', 'local_blocks', 'engine'])
def test_initial_handshake_reentry_has_one_metadata_and_one_read(tmp_path, torch, conflict):
    observer = r2.Observation(configuration(tmp_path, content_client_request_ids=[]))
    value = connector(torch)
    observer.register(value, None, {}, None)
    remote = NS(engine_id='prefill_dp0', request_id='source-case', host='p-host', port=123,
                num_tokens=256, block_ids=[[0, 1], [0, 1]])
    meta = NS(remote=remote, local_block_ids=[[2], [3]], local_physical_block_ids=[[2], [3]])

    class Pull:
        def _compute_desc_ids(self, block_ids):
            return NS(tolist=lambda: [101, 102])

        def _read_blocks(self, read_spec, dst_engine_id, request_id, remote_request_id):
            self._compute_desc_ids([[1], [1]])
            self._compute_desc_ids([[2], [3]])
            return True

        def _read_blocks_for_req(self, req_id, meta):
            self.calls += 1
            if self.calls == 1:
                return False
            spec = NS(remote_rank=3, local_block_ids=meta.local_physical_block_ids,
                      remote_block_ids=meta.remote.block_ids, block_ids_by_region=False)
            return self._read_blocks(spec, meta.remote.engine_id, req_id, meta.remote.request_id)

    value.__class__ = Pull
    value.calls = 0
    r2.wrap(Pull, '_read_blocks_for_req', observer, before=observer.read_request)
    r2.wrap(Pull, '_compute_desc_ids', observer, after=observer.descriptors)
    r2.wrap(Pull, '_read_blocks', observer, before=observer.read_context,
            after=observer.read_submitted, read_scope=True)
    assert value._read_blocks_for_req('case', meta) is False
    assert value._read_blocks_for_req('case', meta) is True
    assert value.calls == 2
    rows = events(observer)
    assert sum(row['event'] == 'read_metadata' for row in rows) == 1
    assert sum(row['event'] == 'read_submit' for row in rows) == 1
    if conflict:
        if conflict == 'tokens':
            remote.num_tokens += 16
        elif conflict == 'local_blocks':
            meta.local_physical_block_ids[0][0] = 1
        else:
            remote.engine_id = 'different-prefill-engine'
        assert value._read_blocks_for_req('case', meta) is True
        assert value.calls == 3 and observer.disabled
        assert events(observer)[-1]['event'] == 'observer_error'
        assert 'changed canonical READ metadata' in events(observer)[-1]['error']


def test_worker_positions_and_source_snapshot_precede_return(tmp_path, torch):
    observer = r2.Observation(configuration(tmp_path, role='prefill'))
    value = connector(torch)
    observer.register(value, None, {}, None)
    calls = []

    class Worker:
        def execute_model(self, scheduler_output):
            calls.append('native')
            return 42

    worker = Worker()
    worker.model_runner = object()
    worker.vllm_config = value.vllm_config
    output = NS(finished_req_ids=set(),
                scheduled_new_reqs=[NS(req_id='case', block_ids=[[0], [1]],
                                       num_computed_tokens=240, prompt_token_ids=[0]*257)],
                scheduled_cached_reqs=NS(req_ids=[]), num_scheduled_tokens={'case': 17})
    r2.wrap(Worker, 'execute_model', observer, before=observer.worker_before, after=observer.worker_after)
    assert worker.execute_model(output) == 42 and calls == ['native']
    rows = events(observer)
    compute = next(r for r in rows if r['event'] == 'source_compute')
    assert compute['positions'] == list(range(240, 257))
    assert compute['tp_rank'] == 3 and compute['engine_id'] == 'decode_dp1'
    content = [r for r in rows if r['event'] == 'source_content']
    assert len(content) == 3
    assert all(r['boundary'] == 'after_final_prefill_before_worker_return' for r in content)
    tail = next(r for r in rows if r['event'] == 'tail_state')
    assert tail['pages'][0]['last_slot'] == -1
    assert tail['pages'][0]['accumulator_counts']['numerator'] == {'elements': 1, 'finite': 1}


def test_decode_tail_positions_include_next_region_crossing(tmp_path, torch):
    observer = r2.Observation(configuration(tmp_path))
    worker = Owner()
    output = NS(finished_req_ids=set(),
                scheduled_new_reqs=[NS(req_id='case', block_ids=[[0], [1]],
                                       num_computed_tokens=240, prompt_token_ids=[0]*257)],
                scheduled_cached_reqs=NS(req_ids=[]), num_scheduled_tokens={'case': 17})
    observer.worker_before(worker, {'scheduler_output': output})
    output.scheduled_new_reqs = []
    output.scheduled_cached_reqs = NS(req_ids=['case'], new_block_ids=[None],
                                     resumed_req_ids=set(), num_computed_tokens=[257])
    output.num_scheduled_tokens = {'case': 9}
    observer.worker_before(worker, {'scheduler_output': output})
    records = events(observer)
    assert records[0]['event'] == records[1]['event'] == 'target_compute'
    assert records[0]['num_scheduled_tokens'] == 17
    assert records[1]['positions'] == list(range(257, 266))
    assert 264 in records[1]['positions']


def test_actual_pool_pressure_reserve_free_and_reuse(tmp_path):
    observer = r2.Observation(configuration(tmp_path))

    class Manager:
        def __init__(self):
            self.ids, self.free_count, self.fail = [], 8, False
            self.block_pool = NS(get_num_free_blocks=lambda: self.free_count)

        def get_block_ids(self, req):
            return [self.ids[:]]

        def allocate_slots(self, request, num_external_computed_tokens=0):
            if self.fail:
                return None
            self.ids, self.free_count = [2, 3], 6
            return object()

        def free(self, request):
            self.ids, self.free_count = [], 8

    manager = Manager()
    request = NS(request_id='case', status='FINISHED_ABORTED',
                 kv_transfer_params={'remote_request_id': 'source-case', 'remote_engine_id': 'prefill_dp0'})
    r2.wrap(Manager, 'allocate_slots', observer, before=observer.pool_before, after=observer.allocated)
    r2.wrap(Manager, 'free', observer, before=observer.pool_before, after=observer.freed_blocks)
    assert manager.allocate_slots(request, num_external_computed_tokens=240) is not None
    manager.free(request)
    manager.fail = True
    assert manager.allocate_slots(request) is None
    manager.fail = False
    manager.allocate_slots(request, 240)
    records = events(observer)
    assert [r['event'] for r in records] == ['target_reserve', 'target_free', 'pressure', 'target_reserve', 'reuse']
    assert records[0]['pool_free_blocks_before'] == 8 and records[0]['pool_free_blocks_after'] == 6
    assert records[1]['blocks'] == [[2, 3]] and records[1]['pool_free_blocks_after'] == 8
    assert records[-1]['blocks'][0]['previous_request_id'] == 'case'
    assert records[0]['remote_request_id'] == 'source-case'


def test_frontend_records_actual_lane_and_load_snapshots(tmp_path):
    observer = r2.Observation(configuration(tmp_path))

    class Client:
        def get_core_engine_for_request(self, request):
            self.engine_inflight[b'b'] += 1
            return b'b'

    client = Client()
    client.core_engines, client.engine_ranks_managed = [b'a', b'b'], [0, 1]
    client.lb_engines, client.engine_inflight = [[1, 0, 0.1], [0, 1, 0.5]], {b'a': 1, b'b': 0}
    client.client_index = 2
    r2.wrap(Client, 'get_core_engine_for_request', observer,
            before=observer.frontend_before, after=observer.frontend)
    assert client.get_core_engine_for_request(NS(request_id='case')) == b'b'
    row = events(observer)[0]
    assert row['event'] == 'frontend_select' and row['selected_dp_rank'] == 1
    assert row['frontend_index'] == 2
    assert row['engine_inflight_before'] == [1, 0] and row['engine_inflight'] == [1, 1]
    assert row['lb_engines'] == [[1, 0, 0.1], [0, 1, 0.5]]


def test_deferred_free_is_not_reported_until_native_pool_return(tmp_path):
    observer = r2.Observation(configuration(tmp_path, role='prefill'))
    scheduler, request = release_runtime(observer)
    scheduler._free_request_blocks(request('case', [[2, 3]]))
    assert [r['event'] for r in events(observer)] == ['free_deferred']
    scheduler._drain_deferred_frees()
    assert [r['event'] for r in events(observer)] == ['free_deferred']
    scheduler.processed_step_seq = 10
    scheduler._drain_deferred_frees()
    row = events(observer)[-1]
    assert row['event'] == 'source_free' and row['fence'] == 10
    assert row['pool_free_blocks_before'] == 6 and row['pool_free_blocks_after'] == 8
    assert row['blocks'] == [[2, 3]]
    assert row['counts_scope'] == 'request_release'
    assert row['last_sched_seq'] == row['deferred_fence_seq'] == row['processed_step_seq'] == 10
    assert row['defer_block_free'] is True
    assert row['native_pool_free_boundary'] == 'BlockPool.free_blocks_return'
    assert observer.local.drain is None


def test_reset_requires_actual_zeroer_and_abort_uses_native_status_name(tmp_path, torch):
    from enum import IntEnum

    class Status(IntEnum):
        FINISHED_ABORTED = 8

    observer = r2.Observation(configuration(tmp_path))
    calls = []

    class Zeroer:
        def zero_block_ids(self, block_ids, block_ids_by_group=None):
            calls.append(block_ids_by_group)

    r2.wrap(Zeroer, 'zero_block_ids', observer, after=observer.reset)
    Zeroer().zero_block_ids([], [[3], []])
    assert calls == [[[3], []]] and torch.syncs == []
    assert events(observer)[-1]['block_ids_by_group'] == [[3], []]
    assert events(observer)[-1]['observer_fence'] is False
    scheduler = Owner()
    scheduler.kv_cache_manager = NS(block_pool=NS(get_num_free_blocks=lambda: 8))
    observer.abort(scheduler, [NS(request_id='case', status=Status.FINISHED_ABORTED)], {}, None)
    assert events(observer)[-1]['event'] == 'abort'
    assert events(observer)[-1]['status'] == 'FINISHED_ABORTED'


@pytest.mark.parametrize('defect', ['missing_summary', 'wire_stride'])
def test_missing_or_partial_wire_evidence_is_error_not_success(tmp_path, torch, defect):
    observer = r2.Observation(configuration(tmp_path))
    value = connector(torch)
    if defect == 'missing_summary':
        del value.device_kv_caches['summary']
        observer.safe('register', observer.register, value, None, {}, None)
    else:
        value.block_len_per_layer[0] = 3
        observer.safe('register', observer.register, value, None, {}, None)
    assert observer.disabled
    assert events(observer)[-1]['event'] == 'observer_error'
    assert not any(r['event'] == 'source_content' for r in events(observer))


@pytest.mark.parametrize('native_fails', [False, True])
def test_loader_does_not_swallow_or_repeat_native_import(tmp_path, native_fails):
    observer = r2.Observation(configuration(tmp_path))
    calls = []

    class Loader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            calls.append('native import')
            if native_fails:
                raise ImportError('native import failed')

    module = ModuleType('vllm.v1.worker.gpu_model_runner')
    module.__file__ = str(tmp_path/'native.py')
    Path(module.__file__).write_text('# native source\n')
    module.GPUModelRunner = type('GPUModelRunner', (), {})
    observer.config['source_hashes'][module.__name__] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    loader = r2.ObservationLoader(Loader(), observer)
    if native_fails:
        with pytest.raises(ImportError, match='native import failed'):
            loader.exec_module(module)
        assert events(observer) == []
    else:
        loader.exec_module(module)
        assert events(observer)[0]['event'] == 'provenance'
        assert events(observer)[-1]['event'] == 'hook_installed'
    assert calls == ['native import']


def test_spec_loader_hook_executes_then_instruments_once(tmp_path, monkeypatch):
    observer = r2.Observation(configuration(tmp_path))
    calls = []
    loader = NS(create_module=lambda spec: None,
                exec_module=lambda module: calls.append('native'))
    spec = NS(loader=loader)
    monkeypatch.setattr(r2.importlib.machinery.PathFinder, 'find_spec', lambda *args: spec)
    monkeypatch.setattr(r2, 'instrument', lambda *args: calls.append('observation'))
    found = r2.ObservationFinder(observer).find_spec('vllm.v1.worker.gpu_worker')
    assert isinstance(found.loader, r2.ObservationLoader)
    found.loader.exec_module(ModuleType('vllm.v1.worker.gpu_worker'))
    assert calls == ['native', 'observation']


def test_activation_off_by_default_and_bootstrap_errors_are_soft(monkeypatch, capsys):
    monkeypatch.delenv(r2.ENV, raising=False)
    before = sys.meta_path[:]
    assert r2.install() is None
    runpy.run_path(str(PD/'r2_bootstrap/sitecustomize.py'))
    assert sys.meta_path == before
    fake = ModuleType('r2_observation')
    fake.install = lambda: (_ for _ in ()).throw(ValueError('bad configuration'))
    monkeypatch.setitem(sys.modules, 'r2_observation', fake)
    monkeypatch.setenv(r2.ENV, '{}')
    runpy.run_path(str(PD/'r2_bootstrap/sitecustomize.py'))
    assert 'UNVERIFIED bootstrap' in capsys.readouterr().err


def test_enabled_install_is_idempotent_and_emits_common_clock_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(r2, '_installed', None)
    monkeypatch.setattr(sys, 'meta_path', sys.meta_path[:])
    monkeypatch.setenv(r2.ENV, json.dumps(configuration(tmp_path)))
    observer = r2.install()
    assert r2.install() is observer
    assert sum(isinstance(f, r2.ObservationFinder) for f in sys.meta_path) == 1
    row = events(observer)[0]
    assert row['event'] == 'observer_started'
    assert row['role'] == 'decode' and row['host'] and row['pid'] > 0
    assert row['monotonic_ns'] > 0 and row['wall_time_ns'] > 0


@pytest.mark.parametrize('request_id,selected', [
    ('case', True), ('cmpl-case-0', True), ('cmpl-case-0-0123abcd', True),
    ('case-0123abcd', True), ('other-case', False), ('case-other', False),
])
def test_client_to_native_request_id_join(tmp_path, request_id, selected):
    observer = r2.Observation(configuration(tmp_path))
    assert bool(observer.selected_id(request_id)) is selected


@pytest.mark.parametrize('defect', ['wrong_pin', 'wrong_loaded_file'])
def test_loaded_source_mismatch_does_not_patch_or_repeat_native(tmp_path, defect):
    observer = r2.Observation(configuration(tmp_path))
    path = tmp_path/'expected.py'
    path.write_text('# locked native bytes\n')
    module = ModuleType('vllm.v1.worker.gpu_model_runner')
    module.__file__ = str(path)
    calls = []

    class NativeRunner:
        def __init__(self):
            calls.append('original constructor')

    module.GPUModelRunner = NativeRunner
    pin = hashlib.sha256(path.read_bytes()).hexdigest()
    if defect == 'wrong_pin':
        pin = '0'*64
    else:
        other = tmp_path/'loaded.py'
        other.write_text('# different bytes under the same module name\n')
        module.__file__ = str(other)
    observer.config['source_hashes'][module.__name__] = pin
    loader = NS(exec_module=lambda value: calls.append('original import'), create_module=lambda spec: None)
    r2.ObservationLoader(loader, observer).exec_module(module)
    NativeRunner()
    assert calls == ['original import', 'original constructor']
    assert not getattr(NativeRunner.__init__, '_step5_r2', False)
    rows = events(observer)
    assert rows[0]['event'] == 'provenance' and rows[0]['verified'] is False
    assert rows[0]['module_file'] == str(Path(module.__file__).resolve())
    assert rows[-1]['event'] == 'observer_error' and observer.disabled


@pytest.mark.parametrize('filename', ['role_capacity_observation.py', 'role_state_observation.py'])
def test_frozen_imported_helper_bytes_checked_before_use(tmp_path, torch, filename):
    config = configuration(tmp_path)
    config['frozen_environment_hashes'][filename] = '0'*64
    observer = r2.Observation(config)
    value = connector(torch)
    calls = []

    class Native:
        def run(self):
            calls.append('original')
            return 41

    r2.wrap(Native, 'run', observer,
            after=lambda *args: observer.register(value, None, {}, None))
    assert Native().run() == 41
    assert calls == ['original'] and observer.disabled
    witnesses = [r for r in events(observer) if r['event'] == 'provenance' and not r['verified']]
    assert len(witnesses) == 1 and witnesses[0]['module'].replace('.', '/')+'.py' == filename
    assert events(observer)[-1]['event'] == 'observer_error'
    assert not any(r['event'] == 'registration' for r in events(observer))


def test_instrumented_source_inventory_includes_both_runners_and_common_zeroer():
    assert r2.MODULES == {path[:-3].replace('/', '.') for path in r2.NATIVE_SOURCE_PATHS}
    assert {'vllm.v1.worker.gpu_model_runner', 'vllm.v1.worker.gpu.model_runner',
            'vllm.v1.worker.utils', 'vllm.v1.core.kv_cache_manager'} <= r2.MODULES
    assert 'vllm.v1.core.block_pool' in r2.MODULES
    assert 'vllm.v1.engine.core' in r2.MODULES
    assert len(r2.NATIVE_SOURCE_PATHS) == 13


@pytest.mark.parametrize('role', ['prefill', 'decode'])
def test_native_only_requests_have_all_events_and_no_fence_or_tensor_snapshot(tmp_path, torch, monkeypatch, role):
    observer = r2.Observation(configuration(tmp_path, role=role, content_client_request_ids=[]))
    value = connector(torch)
    observer.register(value, None, {}, None)

    def forbidden(*args, **kwargs):
        raise AssertionError('native-only path attempted diagnostic content action')

    monkeypatch.setattr(observer, 'synchronize', forbidden)
    monkeypatch.setattr(observer, 'capture', forbidden)
    monkeypatch.setattr(observer, 'tail', forbidden)
    monkeypatch.setattr(Bytes, 'cpu', forbidden)
    monkeypatch.setattr(Numeric, 'cpu', forbidden)
    monkeypatch.setattr(Numeric, 'item', forbidden)
    remote = NS(engine_id='prefill_dp1', request_id='case-abcdef12', host='p', port=123,
                num_tokens=257, block_ids=[[0, 1], [1]])
    observer.read_request(value, dict(req_id='case', meta=NS(
        remote=remote, local_block_ids=[[2], [3]], local_physical_block_ids=[[2], [3]])))
    context = observer.read_context(value, dict(request_id='case', remote_request_id=remote.request_id,
        dst_engine_id=remote.engine_id, read_spec=NS(remote_rank=3, block_ids_by_region=False,
            local_block_ids=[[2], [3]], remote_block_ids=[[0, 1], [1]])))
    for ids in ([[1], [1]], [[2], [3]]):
        observer.descriptors(value, NS(tolist=lambda: [7, 9]), {'block_ids': ids}, None)
    observer.local.read = None
    observer.read_submitted(value, True, {}, context)
    observer.network_done(value, ({'case'}, set()), {'transfers': value._recving_transfers}, None)
    observer.transfer_results(value, NS(finished_recving={'case'}, failed_recving=set()), {}, None)
    worker = Owner()
    output = NS(finished_req_ids=set(), scheduled_cached_reqs=NS(req_ids=[]),
                scheduled_new_reqs=[NS(req_id='case', block_ids=[[2], [3]],
                    num_computed_tokens=240, prompt_token_ids=[0]*257)], num_scheduled_tokens={'case': 17})
    scheduled = observer.worker_before(worker, {'scheduler_output': output})
    observer.worker_after(worker, None, {}, scheduled)
    output.finished_req_ids = {'case'}
    output.scheduled_new_reqs, output.num_scheduled_tokens = [], {}
    observer.worker_before(worker, {'scheduler_output': output})
    observer.reset(Owner(), None, {'block_ids': [2], 'block_ids_by_group': [[2], [3]]}, None)
    manager = Owner()
    manager.get_block_ids = lambda req: [[2], [3]]
    manager.block_pool = NS(get_num_free_blocks=lambda: 8)
    request = NS(request_id='case', num_computed_tokens=257, status='FINISHED', num_prompt_tokens=257)
    before = observer.pool_before(manager, {'request': request})
    observer.allocated(manager, object(), {'request': request, 'num_external_computed_tokens': 240}, before)
    observer.freed_blocks(manager, None, {'request': request}, before)
    if role == 'prefill':
        scheduler = Owner()
        scheduler.kv_cache_manager = manager
        observer.hold(scheduler, (True, {'remote_engine_id': 'prefill_dp1'}), {'request': request}, None)
        observer.notified(value, {'case'}, {}, None)
    client = Owner()
    client.core_engines, client.engine_ranks_managed = [b'a', b'b'], [0, 1]
    client.lb_engines, client.engine_inflight, client.client_index = [[0, 0, 0], [0, 0, 0]], {b'a': 0, b'b': 1}, 0
    before = observer.frontend_before(client, {'request': request})
    observer.frontend(client, b'b', {'request': request}, before)
    names = {r['event'] for r in events(observer)}
    assert {'read_metadata', 'read_submit', 'read_network_done', 'native_read_ready',
            'reset', 'frontend_select'} <= names
    assert {'source_hold', 'source_notified', 'source_compute', 'source_free'} <= names if role == 'prefill' else {
        'target_reserve', 'target_compute', 'target_free'} <= names
    assert not names & {'read_gpu_visible', 'source_content', 'target_content', 'tail_state', 'capture_elapsed'}
    assert torch.syncs == [] and not observer.disabled
    assert all(r['observer_fence'] is False for r in events(observer))


def test_all_native_rx_ready_records_precede_first_diagnostic_fence(tmp_path, torch):
    observer = r2.Observation(configuration(tmp_path,
        selected_client_request_ids=['content', 'native'], content_client_request_ids=['content']))
    value = connector(torch)
    observer.register(value, None, {}, None)
    for req in ('content', 'native'):
        observer.connector_states[value]['reads'][req] = dict(metadata=dict(
            remote_request_id=req, remote_engine_id='prefill_dp1', local_physical_block_ids=[[], []]), plans=[])
    calls = []

    class Native:
        def get_transfer_results(self):
            calls.append('native')
            return NS(finished_recving={'content', 'native'}, failed_recving=set())

    # Native return is obtained exactly once before the callback; synchronizing
    # is allowed only after both returned completion records have been emitted.
    def synchronize():
        ready = [r for r in events(observer) if r['event'] == 'native_read_ready']
        assert {r['request_id'] for r in ready} == {'content', 'native'}
        assert all(not r['observer_fence'] for r in ready)
        calls.append('observer fence')

    torch.cuda.synchronize = synchronize
    r2.wrap(Native, 'get_transfer_results', observer,
            after=lambda native, result, args, token: observer.transfer_results(value, result, args, token))
    result = Native().get_transfer_results()
    assert result.finished_recving == {'content', 'native'}
    assert calls == ['native', 'observer fence']
    assert [r['request_id'] for r in events(observer) if r['event'] == 'read_gpu_visible'] == ['content']


def test_group_specs_retain_nonidentity_scheduler_transfer_mapping(tmp_path, torch):
    observer = r2.Observation(configuration(tmp_path))
    value = connector(torch)
    groups = value.kv_cache_config.kv_cache_groups
    value.kv_cache_config.kv_cache_groups = [groups[1], groups[0]]
    value.kv_cache_config.transfer_group_ids = (1, 0)
    observer.register(value, None, {}, None)
    registration = next(r for r in events(observer) if r['event'] == 'registration')
    assert registration['transfer_group_to_scheduler_group'] == [1, 0]
    full, swa = registration['transfer_group_specs']
    assert (full['transfer_group_id'], full['scheduler_group_id']) == (0, 1)
    assert full['layer_specs']['full']['block_size'] == 16
    assert full['layer_specs']['full']['sliding_window'] is None
    assert (swa['transfer_group_id'], swa['scheduler_group_id']) == (1, 0)
    assert swa['layer_specs']['swa']['sliding_window'] == 512
    assert [r['scheduler_group_id'] for r in registration['registrations']] == [1, 0, 1]


def test_tail_exact_bytes_are_limited_to_three_decode_checkpoints(tmp_path, torch):
    observer = r2.Observation(configuration(tmp_path))
    value = connector(torch)
    for name in ('full', 'summary'):
        value.device_kv_caches[name].rows = [bytes([index])*4 for index in range(20)]
        tensor = value.device_kv_caches[name]
        tensor.shape = (20, *tensor.shape[1:])
        value.region_num_blocks[value.region_names.index(name)] = 20
    observer.register(value, None, {}, None)
    observer.connector_states[value]['reads']['case'] = dict(metadata=dict(
        remote_request_id='source-case', remote_engine_id='prefill_dp1',
        local_physical_block_ids=[list(range(17)), [1]]), plans=[])
    observer.transfer_results(value, NS(finished_recving={'case'}, failed_recving=set()), {}, None)
    worker = Owner()
    output = NS(finished_req_ids=set(), scheduled_cached_reqs=NS(req_ids=[]),
        scheduled_new_reqs=[NS(req_id='case', block_ids=[list(range(17)), [1]],
            num_computed_tokens=240, prompt_token_ids=[0]*257)], num_scheduled_tokens={'case': 17})
    scheduled = observer.worker_before(worker, {'scheduler_output': output})
    observer.worker_after(worker, None, {}, scheduled)
    for computed in range(257, 385):
        output.scheduled_new_reqs = []
        output.scheduled_cached_reqs = NS(req_ids=['case'], new_block_ids=[None],
            resumed_req_ids=set(), num_computed_tokens=[computed])
        output.num_scheduled_tokens = {'case': 1}
        scheduled = observer.worker_before(worker, {'scheduler_output': output})
        observer.worker_after(worker, None, {}, scheduled)
    rows = events(observer)
    checkpoints = [r for r in rows if r['event'] == 'tail_state']
    assert [r['boundary'] for r in checkpoints] == [
        'after_read', 'first_local_tail_complete', 'first_decode_region_crossing']
    assert checkpoints[-1]['num_computed_tokens'] == 265
    assert torch.syncs == ['sync']*3
    assert len([r for r in rows if r['event'] == 'target_compute']) == 129
    for checkpoint in checkpoints:
        assert checkpoint['continuity_verdict'].startswith('UNVERIFIED')
        assert checkpoint['journal'] == {'width': 0, 'status': 'NOT_ALLOCATED_NATIVE_ABI'}
        for page in checkpoint['pages']:
            assert set(page['exact_tensors']) == {'numerator', 'denominator', 'maximum', 'last_slot'}
            for sample in page['exact_tensors'].values():
                assert hashlib.sha256(bytes.fromhex(sample['raw_hex'])).hexdigest() == sample['sha256']
                assert 'storage' in sample['tensor']
            assert len(page['summary_mean_wire_row']['sha256']) == 64
            assert ('raw_hex' in page['summary_mean_wire_row']) is (
                checkpoint['boundary'] == 'first_local_tail_complete')


def test_actual_request_journal_abi_is_captured_with_native_request_index(tmp_path, torch):
    observer = r2.Observation(configuration(tmp_path))
    value = connector(torch)
    cache = value.vllm_config.compilation_config.static_forward_context['summary']._cache
    cache.request_journal_width = 2
    for name in ('request_journal_values', 'request_journal_last', 'request_journal_tag'):
        setattr(cache, name, Tensor([b'abcd', b'efgh', b'ijkl', b'mnop'], torch.uint8, False))
    runner = Owner()
    runner.vllm_config, runner.req_states = value.vllm_config, NS(req_id_to_index={'case': 1})
    observer.runners.append(r2.weakref.ref(runner))
    observer.register(value, None, {}, None)
    observer.tail(value, 'case', [[1], []], 'after_read')
    journal = events(observer)[-1]['journal']
    assert journal['status'] == 'OBSERVED_EXACT_BYTES' and journal['request_state_index'] == 1
    assert set(journal['tensors']) == {'request_journal_values', 'request_journal_last', 'request_journal_tag'}
    for sample in journal['tensors'].values():
        assert sample['raw_hex'] == b'ijklmnop'.hex()


def test_elapsed_cap_is_post_return_soft_and_preserves_native_result(tmp_path, torch, monkeypatch):
    observer = r2.Observation(configuration(tmp_path, max_capture_seconds_per_request=0.5e-9))
    clock = iter(range(10000))
    monkeypatch.setattr(r2.time, 'monotonic_ns', lambda: next(clock))
    calls = []

    class Native:
        def run(self):
            calls.append('original returned')
            return 41

    r2.wrap(Native, 'run', observer, after=lambda *args: observer.diagnostic(
        None, 'case', 'test_capture', observer.synchronize))
    assert Native().run() == 41
    assert Native().run() == 41
    assert calls == ['original returned']*2 and torch.syncs == ['sync']
    row = next(r for r in events(observer) if r['event'] == 'capture_elapsed')
    assert row['cap_mode'] == 'post_return_soft' and row['cuda_calls_interruptible'] is False
    assert events(observer)[-1]['event'] == 'observer_budget_exceeded' and observer.disabled


def test_prefill_nonfinal_chunk_has_no_diagnostic_fence(tmp_path, torch):
    observer = r2.Observation(configuration(tmp_path, role='prefill'))
    value = connector(torch)
    observer.register(value, None, {}, None)
    worker = Owner()
    output = NS(finished_req_ids=set(), scheduled_cached_reqs=NS(req_ids=[]),
        scheduled_new_reqs=[NS(req_id='case', block_ids=[[1], [1]],
            num_computed_tokens=0, prompt_token_ids=[0]*257)], num_scheduled_tokens={'case': 128})
    scheduled = observer.worker_before(worker, {'scheduler_output': output})
    observer.worker_after(worker, None, {}, scheduled)
    assert torch.syncs == []
    assert not any(r['event'] == 'source_content' for r in events(observer))
    output.scheduled_new_reqs = []
    output.scheduled_cached_reqs = NS(req_ids=['case'], new_block_ids=[None],
        resumed_req_ids=set(), num_computed_tokens=[128])
    output.num_scheduled_tokens = {'case': 129}
    scheduled = observer.worker_before(worker, {'scheduler_output': output})
    observer.worker_after(worker, None, {}, scheduled)
    assert torch.syncs == ['sync']
    assert len([r for r in events(observer) if r['event'] == 'source_content']) == 3


@pytest.mark.parametrize('module,zeroer_field', [
    ('vllm.v1.worker.gpu.model_runner', 'kv_block_zeroer'),
    ('vllm.v1.worker.gpu_model_runner', '_kv_block_zeroer'),
])
@pytest.mark.parametrize('supported', [False, True])
def test_actual_runner_module_and_reset_abi_are_reported(tmp_path, module, zeroer_field, supported):
    observer = r2.Observation(configuration(tmp_path))

    class Zeroer:
        def zero_block_ids(self, block_ids):
            pass

    if supported:
        r2.wrap(Zeroer, 'zero_block_ids', observer, after=observer.reset)
    cls = type('GPUModelRunner', (), {'__module__': module})
    runner = cls()
    setattr(runner, zeroer_field, Zeroer())
    worker = Owner()
    worker.model_runner, worker.use_v2_model_runner = runner, module.endswith('gpu.model_runner')
    observer.runner_info(worker)
    row = events(observer)[0]
    assert row['runner_module'] == module
    assert row['reset_abi_status'] == ('OBSERVED' if supported else 'UNVERIFIED_RESET_ABI')
    assert row['configured_use_v2_model_runner'] == module.endswith('gpu.model_runner')
    assert row['observer_fence'] is False


def test_inaccessible_independent_group_spec_is_unverified_not_guessed(tmp_path, torch):
    observer = r2.Observation(configuration(tmp_path))
    value = connector(torch)
    del value.kv_cache_config.kv_cache_groups
    observer.safe('register', observer.register, value, None, {}, None)
    assert events(observer)[-1]['event'] == 'observer_error' and observer.disabled
    assert not any(r['event'] == 'registration' for r in events(observer))


def test_scalar_last_slot_is_flattened_before_dtype_view(tmp_path, torch):
    class Scalar(Numeric):
        def __init__(self):
            super().__init__(-1)
            self.shape, self.dtype = (), 'int64'

        def element_size(self):
            return 8

        def stride(self):
            return ()

        def untyped_storage(self):
            return NS(data_ptr=lambda: id(self), nbytes=lambda: 8)

        def view(self, dtype):
            raise RuntimeError('torch disallows changing dtype of a zero-dimensional tensor')

        def reshape(self, *shape):
            assert shape == (-1,)
            return NS(view=lambda dtype: Bytes(struct.pack('<q', self.value)))

    observer = r2.Observation(configuration(tmp_path))
    capacity, state = observer.frozen_helpers()
    sample, host = observer.tensor_sample('case', 'last_slot', Scalar(), capacity, state)
    assert sample['shape'] == [] and sample['dtype'] == 'int64'
    assert sample['raw_hex'] == struct.pack('<q', -1).hex()
    assert sample['sha256'] == hashlib.sha256(struct.pack('<q', -1)).hexdigest()
    assert host.item() == -1


@pytest.mark.parametrize('role', ['prefill', 'decode'])
def test_one_drain_emits_two_exact_request_deltas_and_preserves_reversed_iterators(tmp_path, role):
    observer = r2.Observation(configuration(tmp_path, role=role, selected_client_request_ids=['one', 'two']))
    scheduler, request = release_runtime(observer)
    scheduler._free_request_blocks(request('one', [[2, 3]]))
    scheduler._free_request_blocks(request('two', [[4, 5, 6]]))
    pool = scheduler.kv_cache_manager.block_pool
    scheduler.processed_step_seq = 10
    scheduler._drain_deferred_frees()
    records = [r for r in events(observer) if r['event'] in ('source_free', 'target_free')]
    assert [(r['request_id'], r['pool_free_blocks_before'], r['pool_free_blocks_after']) for r in records] == [
        ('one', 6, 8), ('two', 8, 11)]
    assert all(r['counts_scope'] == 'request_release' for r in records)
    assert all(r['event'] == ('source_free' if role == 'prefill' else 'target_free') for r in records)
    assert all(r['last_sched_seq'] <= r['deferred_fence_seq'] <= r['processed_step_seq'] for r in records)
    assert scheduler.drain_calls == 1 and pool.calls == 2
    assert pool.consumed == [[3, 2], [6, 5, 4]]
    assert len(scheduler.iterators) == len(pool.iterators) == 2
    assert all(original is received for original, received in zip(scheduler.iterators, pool.iterators))
    assert all(list(iterator) == [] for iterator in scheduler.iterators)
    assert observer.deferred[scheduler] == {} and observer.local.drain is None
    assert not any(r.get('counts_scope') == 'entire_native_deferred_drain_call' for r in events(observer))
    scheduler._drain_deferred_frees()
    assert len([r for r in events(observer) if r['event'] in ('source_free', 'target_free')]) == 2
    assert pool.calls == 2


@pytest.mark.parametrize('role', ['prefill', 'decode'])
def test_immediate_manager_free_preserves_real_preemption_status_and_scheduler_guard(tmp_path, role):
    from enum import IntEnum

    class Status(IntEnum):
        RUNNING = 4

    observer = r2.Observation(configuration(tmp_path, role=role))
    scheduler, request = release_runtime(observer)
    native = request('case', [[2, 3]], status=Status.RUNNING)
    scheduler.processed_step_seq = 10
    scheduler.kv_cache_manager.free(native)
    records = [r for r in events(observer) if r['event'] in ('source_free', 'target_free')]
    assert len(records) == 1
    row = records[0]
    assert row['counts_scope'] == 'request_release'
    assert row['status'] == 'RUNNING' and row['status_observed_at'] == 'native_manager_free_return'
    assert row['last_sched_seq'] == row['processed_step_seq'] == 10 and row['defer_block_free'] is True
    assert row['pool_free_blocks_before'] == 6 and row['pool_free_blocks_after'] == 8
    assert scheduler.kv_cache_manager.calls == 1 and scheduler.kv_cache_manager.block_pool.calls == 1
    assert not any(r['event'] == 'free_deferred' for r in events(observer))


@pytest.mark.parametrize('unknown_kind', ['cow', 'unselected_request'])
def test_unattributed_queue_entry_is_unverified_without_guessing_request_or_delta(tmp_path, unknown_kind):
    observer = r2.Observation(configuration(tmp_path))
    scheduler, request = release_runtime(observer)
    unknown = request('other', [[8, 9]])
    if unknown_kind == 'cow':
        blocks = scheduler.kv_cache_manager.pop_blocks_for_free(unknown)
        scheduler.deferred_frees.append((10, blocks[::-1]))
    else:
        scheduler._free_request_blocks(unknown)
    scheduler._free_request_blocks(request('case', [[2, 3]]))
    scheduler.processed_step_seq = 10
    scheduler._drain_deferred_frees()
    rows = events(observer)
    unknown_rows = [r for r in rows if r['event'] == 'release_unverified']
    assert len(unknown_rows) == 1
    assert 'request_id' not in unknown_rows[0] and unknown_rows[0]['verified'] is False
    assert unknown_rows[0]['counts_scope'] == 'unattributed_native_pool_call'
    assert (unknown_rows[0]['pool_free_blocks_before'], unknown_rows[0]['pool_free_blocks_after']) == (6, 8)
    releases = [r for r in rows if r['event'] == 'target_free']
    assert [(r['request_id'], r['pool_free_blocks_before'], r['pool_free_blocks_after']) for r in releases] == [
        ('case', 8, 10)]
    assert scheduler.kv_cache_manager.block_pool.calls == 2 and not observer.disabled


@pytest.mark.parametrize('phase', ['before', 'after'])
def test_pool_release_observer_error_never_changes_native_drain_or_iterator(tmp_path, monkeypatch, phase):
    observer = r2.Observation(configuration(tmp_path, selected_client_request_ids=['one', 'two']))

    def fail(*args):
        raise ValueError('pool release observation failed')

    monkeypatch.setattr(observer, 'pool_free_before' if phase == 'before' else 'pool_freed', fail)
    scheduler, request = release_runtime(observer)
    scheduler._free_request_blocks(request('one', [[2, 3]]))
    scheduler._free_request_blocks(request('two', [[4, 5, 6]]))
    scheduler.processed_step_seq = 10
    assert scheduler._drain_deferred_frees() is None
    pool = scheduler.kv_cache_manager.block_pool
    assert pool.calls == 2 and pool.free_count == 11 and scheduler.drain_calls == 1
    assert pool.consumed == [[3, 2], [6, 5, 4]]
    assert all(original is received for original, received in zip(scheduler.iterators, pool.iterators))
    assert events(observer)[-1]['event'] == 'observer_error' and observer.disabled
    assert not any(r['event'] in ('source_free', 'target_free') for r in events(observer))
    assert observer.local.drain is None


def test_native_pool_free_error_propagates_once_without_release_claim(tmp_path):
    observer = r2.Observation(configuration(tmp_path))
    scheduler, request = release_runtime(observer)
    scheduler._free_request_blocks(request('case', [[2, 3]]))
    scheduler.processed_step_seq = 10
    scheduler.kv_cache_manager.block_pool.fail = True
    with pytest.raises(LookupError, match='native pool free failure'):
        scheduler._drain_deferred_frees()
    assert scheduler.kv_cache_manager.block_pool.calls == 1 and scheduler.drain_calls == 1
    assert observer.local.drain is None and not observer.disabled
    assert observer.deferred[scheduler] == {}
    assert not any(r['event'] in ('source_free', 'target_free', 'observer_error') for r in events(observer))


def test_missing_per_call_pool_hook_is_not_replaced_by_aggregate_release(tmp_path):
    observer = r2.Observation(configuration(tmp_path))
    scheduler, request = release_runtime(observer, pool_hook=False)
    scheduler._free_request_blocks(request('case', [[2, 3]]))
    scheduler.processed_step_seq = 10
    scheduler._drain_deferred_frees()
    assert scheduler.kv_cache_manager.block_pool.free_count == 8
    assert not any(r['event'] in ('source_free', 'target_free') for r in events(observer))
    row = events(observer)[-1]
    assert row['event'] == 'release_unverified' and row['request_id'] == 'case'
    assert row['reason'] == 'selected_deferred_release_missing_per_call_pool_boundary'
    assert 'pool_free_blocks_before' not in row and 'pool_free_blocks_after' not in row


def test_release_delta_preserves_native_null_and_cross_group_alias_accounting(tmp_path):
    observer = r2.Observation(configuration(tmp_path))
    scheduler, request = release_runtime(observer)
    scheduler._free_request_blocks(request('case', [[0, 2, 3], [2]]))
    scheduler.processed_step_seq = 10
    scheduler._drain_deferred_frees()
    row = next(r for r in events(observer) if r['event'] == 'target_free')
    assert row['blocks'] == [[0, 2, 3], [2]]
    assert row['pool_free_blocks_after']-row['pool_free_blocks_before'] == 2
    assert row['counts_scope'] == 'request_release'
    assert scheduler.kv_cache_manager.block_pool.consumed == [[2, 3, 2, 0]]
