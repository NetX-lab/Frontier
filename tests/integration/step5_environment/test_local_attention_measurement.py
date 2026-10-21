"""CPU-only contracts for independent native attention-owner measurements.

Native forward ASTs run with small boundary fakes; constructors and GPU kernels
are not executed, and no real Torch or vLLM package is imported.
"""
import ast
import builtins
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import typing

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import local_attention_measurement as module

TASK = Path(__file__).resolve().parents[3] / 'task_memory/task_2026-10-15_step5_preview_h800'
NATIVE = TASK / 'native_source_export/observed/vllm/models/step5/nvidia/model.py'
NATIVE_TREE = ast.parse(NATIVE.read_text())
PLAN = json.loads((TASK / 'local_scope_sampling_01.json').read_text())
SELECTED = module.samples(PLAN)


def native_class(name, namespace, methods=None):
    """Compile exported classes or selected unchanged methods without constructors."""
    cls = next(node for node in NATIVE_TREE.body
               if isinstance(node, ast.ClassDef) and node.name == name)
    if methods is not None:
        cls = copy.deepcopy(cls)
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    source = 'from __future__ import annotations\n' + ast.unparse(cls)
    exec(compile(source, str(NATIVE), 'exec'), namespace)
    return namespace[name]


def fake_vllm(monkeypatch, exports):
    """Install only the module paths used by the tested native boundaries."""
    modules = {}
    for path, attributes in exports.items():
        names = path.split('.')
        for end in range(1, len(names) + 1):
            name = '.'.join(names[:end])
            if name not in modules:
                modules[name] = ModuleType(name)
                modules[name].__package__ = name.rpartition('.')[0]
                modules[name].__path__ = []
                monkeypatch.setitem(sys.modules, name, modules[name])
                if end > 1:
                    setattr(modules['.'.join(names[:end - 1])], names[end - 1], modules[name])
        modules[path].__dict__.update(attributes)
    return modules


@contextmanager
def record_scope(events, name):
    events.append(('enter', name))
    try:
        yield
    finally:
        events.append(('exit', name))


def test_selected_cases_preserve_all_eligible_holdouts_and_prefill_queries():
    # The frozen 37-case plan excludes two replay, one multi-row and four precision cases.
    assert len(PLAN['cases']) == 37
    assert len(SELECTED) == 30
    assert all(row['rows'] == 1 and row['graph'] == 'eager' and row['precision'] == 'bf16'
               and row['query'] > 1 and row['measurement_status'] == 'unmeasured' for row in SELECTED)
    assert {row['label'] for row in SELECTED if row['partition'] == 'holdout'} == {
        'holdout_subsequent_19', 'holdout_subsequent_515', 'holdout_subsequent_4099'}
    excluded = {row['label'] for row in PLAN['cases']} - {row['label'] for row in SELECTED}
    assert excluded == {'local_graph_rows8_valid7', 'local_graph_rows8_valid8', 'local_graph_rows9_edge',
                        'precision_fp8_routed', 'precision_bf16_excluded',
                        'precision_fp32_norm_residual', 'precision_fp32_router'}
    assert all(any(row is original for original in PLAN['cases']) for row in SELECTED)


@pytest.mark.parametrize('sample', SELECTED, ids=lambda row: row['label'])
def test_prefix_chunks_fit_window_and_preserve_total(sample):
    schedule = module.chunks(sample, PLAN['geometry']['W'])
    assert all(0 < count <= 512 for count in schedule)
    assert sum(schedule[:-1]) == sample['prefix']
    assert schedule[-1] == sample['query']
    assert sum(schedule) == sample['total']


def test_boundary_513_keeps_the_one_token_prefix():
    sample = next(row for row in SELECTED if row['label'] == 'boundary_new_513')
    assert module.chunks(sample, 512) == [1, 512]


@pytest.mark.parametrize('query', [-1, 0, 513])
def test_invalid_query_cannot_be_scheduled(query):
    with pytest.raises(ValueError, match='sampled query'):
        module.chunks({'prefix': 512, 'query': query}, 512)


def owner_stats(kind):
    return {scope: {'mean': index / 4, 'kernel_count': index, 'count': 2}
            for index, scope in enumerate(module.OWNERS[kind], 1)}


def test_dsa_preparation_parts_are_one_physical_charge():
    stats = owner_stats('dsa')
    owners = module.physical_owners(stats, 'dsa')
    assert list(owners) == ['dsa_projection', 'dsa_prefill_prepare', 'indexer_w',
                            'dsa_prefill_root', 'output_projection']
    preparation = owners['dsa_prefill_prepare']
    assert preparation['parts'] == ['dsa_prefill_prepare.main_norm_rope_raw_kv',
                                     'dsa_prefill_prepare.indexer_norm_rope_cast']
    assert preparation['device_duration_mean_ms'] == 1.5
    assert preparation['kernel_count'] == 6
    assert preparation['part_stats'] == {name: stats[name] for name in preparation['parts']}
    assert sum(row['kernel_count'] for row in owners.values()) == 21
    assert sum(row['device_duration_mean_ms'] for row in owners.values()) == 5.25
    assert 'dsa_decode_prepare' not in owners


@pytest.mark.parametrize('kind,owned,device_ms', [('swa', 15, 3.75), ('dsa', 21, 5.25)])
@pytest.mark.parametrize('difference', [-2, 0, 3])
def test_reconciliation_reports_missing_and_unowned_kernels(kind, owned, device_ms, difference):
    stats = owner_stats(kind)
    stats[module.WHOLE_FORWARD] = {'mean': 9.5, 'kernel_count': owned + difference}
    assert module.reconcile(stats, kind) == {
        'owner_kernel_count': owned, 'forward_kernel_count': owned + difference,
        'unowned_kernel_count': difference, 'owner_device_mean_ms': device_ms,
        'forward_device_mean_ms': 9.5,
        'status': 'KERNELS_RECONCILED' if difference == 0 else 'UNOWNED_OR_MISSING_KERNELS'}


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('raises', [False, True])
def test_owner_wrapper_preserves_arguments_return_and_exception(enabled, raises):
    events, received = [], []
    positional, keyword, result = object(), object(), object()
    error = ValueError('native failure')
    def original(*args, **kwargs):
        received.append((args, kwargs))
        if raises:
            raise error
        return result
    target = SimpleNamespace(forward=original)
    with module.OwnerAnnotations(lambda name: record_scope(events, name)) as owners:
        owners.wrap(target, 'forward', 'projection')
        owners.enabled = enabled
        if raises:
            with pytest.raises(ValueError) as caught:
                target.forward(positional, option=keyword)
            assert caught.value is error
        else:
            assert target.forward(positional, option=keyword) is result
        assert received == [((positional,), {'option': keyword})]
        assert owners.active is None
        assert owners.calls == (['projection'] if enabled else [])
    assert target.forward is original and owners.enabled is False
    assert events == ([('enter', 'vidur_projection'), ('exit', 'vidur_projection')] if enabled else [])


def test_nested_owner_is_rejected_and_active_owner_resets():
    events, inner_calls = [], []
    inner = SimpleNamespace(forward=lambda: inner_calls.append('native') or 7)
    outer = SimpleNamespace(forward=lambda: inner.forward())
    originals = (inner.forward, outer.forward)
    with module.OwnerAnnotations(lambda name: record_scope(events, name)) as owners:
        owners.wrap(inner, 'forward', 'inner')
        owners.wrap(outer, 'forward', 'outer')
        owners.enabled = True
        with pytest.raises(RuntimeError, match='Owner inner was called inside owner outer'):
            outer.forward()
        assert owners.active is None and inner_calls == []
        assert inner.forward() == 7
        assert owners.calls == ['outer', 'inner']
    assert (inner.forward, outer.forward) == originals


def test_owner_context_restores_all_attributes_when_native_raises():
    error = RuntimeError('native failure')
    def fail():
        raise error
    targets = [SimpleNamespace(forward=fail), SimpleNamespace(forward=lambda: 1)]
    originals = [target.forward for target in targets]
    with pytest.raises(RuntimeError) as caught:
        with module.OwnerAnnotations(lambda name: record_scope([], name)) as owners:
            for index, target in enumerate(targets):
                owners.wrap(target, 'forward', str(index))
            owners.enabled = True
            targets[0].forward()
    assert caught.value is error
    assert owners.active is None and owners.enabled is False
    assert all(target.forward is original for target, original in zip(targets, originals))


class NativeModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Tensor:
    """Shape-only values for native split/cache control flow, not tensor arithmetic."""
    dtype = 'torch.bfloat16'
    device = 'cuda'
    def __init__(self, shape=(2, 8)):
        self.shape = shape
    def split(self, sizes, dim=-1):
        assert dim == -1
        return tuple(Tensor((self.shape[0], size)) for size in sizes)
    def new_empty(self, size):
        assert size == 0
        return Tensor((0,))
    def numel(self):
        return 1
    def view(self, *shape):
        return self
    def reshape(self, *shape):
        return self
    def stride(self):
        return (self.shape[-1], 1)
    def __getitem__(self, key):
        return key


class UnquantizedLinearMethod:
    @property
    def _gemm_impl(self):
        raise AssertionError('The direct native forward must bypass the quantization GEMM')


@pytest.mark.parametrize('name', ['Step5FusedQKVIndexerLinear', 'Step5SparseIndexerIndexTPLinear'])
@pytest.mark.parametrize('batch_invariant', [False, True])
def test_direct_native_projection_forward_bypasses_quant_method(name, batch_invariant):
    calls, combined = [], Tensor()
    def linear(*args):
        calls.append(('functional', args))
        return combined
    def invariant(*args):
        calls.append(('batch_invariant', args))
        return combined
    namespace = {'__name__': 'vllm.models.step5.nvidia.model', 'QKVParallelLinear': NativeModule,
        'nn': SimpleNamespace(Module=NativeModule), 'envs': SimpleNamespace(VLLM_BATCH_INVARIANT=batch_invariant),
        'current_platform': SimpleNamespace(is_cuda_alike=lambda: True), 'linear_batch_invariant': invariant,
        'torch': SimpleNamespace(nn=SimpleNamespace(functional=SimpleNamespace(linear=linear)))}
    cls = native_class(name, namespace)
    projection = cls.__new__(cls)
    projection.weight, projection.qkv_size = Tensor((12, 8)), 4
    if name == 'Step5FusedQKVIndexerLinear':
        projection.quant_method = UnquantizedLinearMethod()
    inputs = object()
    result = projection.forward(inputs)
    assert len(calls) == 1 and calls[0][0] == ('batch_invariant' if batch_invariant else 'functional')
    assert calls[0][1][:2] == (inputs, projection.weight)
    if name == 'Step5FusedQKVIndexerLinear':
        assert result == ((Ellipsis, slice(None, 4)), (Ellipsis, slice(4, None)))
    else:
        assert result == (combined, None)
        assert not hasattr(projection, 'quant_method')
    record = module.projection_dispatch(projection)
    assert record['class_name'] == 'vllm.models.step5.nvidia.model.' + name
    assert record['forward'] == 'vllm.models.step5.nvidia.model.' + name + '.forward'
    assert record['execution'] == 'native_forward_functional_linear_or_batch_invariant'
    assert record['method'] == ('UnquantizedLinearMethod' if name == 'Step5FusedQKVIndexerLinear' else None)
    assert record['gemm'] is None and record['dtype'] == 'torch.bfloat16'
    assert record['shape'] == [12, 8] and record['stride'] == [8, 1]


class ConventionalLinear(NativeModule):
    def forward(self, value):
        return value


def conventional_projection(dtype='torch.bfloat16'):
    def gemm():
        pass
    method = type('UnquantizedLinearMethod', (), {'_gemm_impl': staticmethod(gemm)})()
    projection = ConventionalLinear()
    projection.weight, projection.quant_method = Tensor((4, 8)), method
    projection.weight.dtype = dtype
    return projection


@pytest.fixture
def linear_native(monkeypatch):
    def invariant(*args):
        return args
    modules = fake_vllm(monkeypatch, {
        'vllm.envs': {'VLLM_BATCH_INVARIANT': False},
        'vllm.platforms': {'current_platform': SimpleNamespace(is_cuda_alike=lambda: True)},
        'vllm.model_executor.layers.linear': {'linear_batch_invariant': invariant}})
    return modules


def test_conventional_projection_records_native_unquantized_gemm(linear_native):
    record = module.projection_dispatch(conventional_projection())
    assert record['execution'] == 'quant_method_gemm' and record['method'] == 'UnquantizedLinearMethod'
    assert record['gemm'].endswith('.gemm')
    assert record['dtype'] == 'torch.bfloat16' and record['shape'] == [4, 8]


@pytest.mark.parametrize('batch_invariant', [False, True])
def test_conventional_dispatch_matches_unchanged_native_apply(linear_native, batch_invariant):
    linear_path = TASK / 'native_runtime_sources/observed/vllm/model_executor/layers/linear.py'
    tree = ast.parse(linear_path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'UnquantizedLinearMethod')
    apply = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'apply')
    native = linear_native['vllm.model_executor.layers.linear']
    calls, result = [], object()
    def invariant(*args):
        calls.append('batch_invariant')
        return result
    def gemm(*args):
        calls.append('gemm')
        return result
    native.linear_batch_invariant = invariant
    linear_native['vllm.envs'].VLLM_BATCH_INVARIANT = batch_invariant
    namespace = {'envs': linear_native['vllm.envs'],
        'current_platform': linear_native['vllm.platforms'].current_platform,
        'linear_batch_invariant': invariant}
    exec(compile('from __future__ import annotations\n' + ast.unparse(apply), str(linear_path), 'exec'), namespace)
    projection = conventional_projection()
    projection.quant_method._gemm_impl = gemm
    value = object()
    assert namespace['apply'](projection.quant_method, projection, value) is result
    record = module.projection_dispatch(projection)
    expected = invariant if batch_invariant else gemm
    assert calls == ['batch_invariant' if batch_invariant else 'gemm']
    assert record['execution'] == ('quant_method_batch_invariant' if batch_invariant else 'quant_method_gemm')
    assert record['gemm'] == expected.__module__ + '.' + expected.__qualname__


@pytest.mark.parametrize('kind', ['conventional', 'Step5FusedQKVIndexerLinear'])
def test_quantized_projection_is_rejected(kind, linear_native):
    projection = conventional_projection()
    if kind != 'conventional':
        projection = type(kind, (ConventionalLinear,), {})()
        projection.weight = Tensor()
    projection.quant_method = SimpleNamespace()
    with pytest.raises(ValueError, match='unquantized|BF16 precision'):
        module.projection_dispatch(projection)


@pytest.mark.parametrize('name', ['conventional', 'Step5FusedQKVIndexerLinear', 'Step5SparseIndexerIndexTPLinear'])
def test_projection_weights_must_remain_bf16(name, linear_native):
    projection = conventional_projection('torch.float32')
    if name != 'conventional':
        projection = type(name, (ConventionalLinear,), {})()
        projection.weight = Tensor()
        projection.weight.dtype = 'torch.float32'
        projection.quant_method = UnquantizedLinearMethod()
    with pytest.raises(ValueError, match='weights must be BF16'):
        module.projection_dispatch(projection)


@pytest.mark.parametrize('kwargs,match', [
    ({'kind': 'unsupported'}, 'preregistered'),
    ({'warmups': 0}, 'warmups'), ({'repeats': 1}, 'repeats'),
    ({'tensor_copy_bytes': None}, 'byte budget'),
    ({'tensor_copy_bytes': 0}, 'byte budget'), ({'tensor_copy_bytes': -1}, 'byte budget')])
def test_invalid_measurement_rejected_before_imports_or_writes(monkeypatch, tmp_path, kwargs, match):
    arguments = {'kind': 'swa', 'warmups': 1, 'repeats': 2, 'tensor_copy_bytes': 1024}
    arguments.update(kwargs)
    output = tmp_path / 'not_written'
    def reject_import(*args, **options):
        raise AssertionError('Invalid measurement attempted an import')
    with monkeypatch.context() as patch:
        patch.setattr(builtins, '__import__', reject_import)
        with pytest.raises(ValueError, match=match):
            module.measure(tmp_path / 'missing_task', output, None, **arguments)
    assert not output.exists()


def test_import_does_not_load_gpu_or_platform_packages():
    code = '''import sys
import importlib.abc
class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'vllm', 'steptron'}:
            raise AssertionError('Forbidden import: ' + fullname)
sys.meta_path.insert(0, RejectNative())
sys.path.insert(0, sys.argv[1])
import local_attention_measurement
assert not any(name.split('.')[0] in {'torch', 'vllm', 'steptron'} for name in sys.modules)
print('NO_NATIVE_IMPORTS')
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR='/data/ycfeng/tmp')
    result = subprocess.run(['/data/ycfeng/envs/frontier-py310/bin/python', '-B', '-c', code,
                             str(Path(module.__file__).parent)], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'NO_NATIVE_IMPORTS'


@pytest.mark.parametrize('kind', ['swa', 'dsa'])
def test_annotation_order_matches_unchanged_native_prefill_forward(monkeypatch, kind):
    events, roots = [], []
    def root(*args):
        roots.append(args)
    def forbidden(*args, **kwargs):
        raise AssertionError('Decode preparation or extra gate projection was reached')
    modules = fake_vllm(monkeypatch, {
        'vllm._custom_ops': {'fused_qknorm_rope_cache': lambda *args: (Tensor(), Tensor(), Tensor())},
        'vllm.models.step5.nvidia.model': {'headwise_sigmoid_gate': lambda output, *args, **kwargs: output,
                                          'step5_dsa_attention_with_output': root},
        'vllm.models.step5.nvidia.ops.indexer_norm_rope': {'indexer_norm_rope': lambda q, k, w, *args, **kwargs: (q, k, w)},
        'vllm.models.step5.nvidia.ops.qkv_indexer_norm_rope': {'qkv_indexer_norm_rope': forbidden}})
    namespace = modules['vllm.models.step5.nvidia.model'].__dict__
    torch = SimpleNamespace(empty=lambda shape, **kwargs: Tensor(shape),
                            profiler=SimpleNamespace(record_function=lambda name: record_scope(events, name)))
    namespace.update(nn=SimpleNamespace(Module=NativeModule), torch=torch, typing=typing, Any=typing.Any,
        ops=modules['vllm._custom_ops'], get_attention_context=lambda name: (None, None, Tensor(), object()),
        get_forward_context=lambda: SimpleNamespace(cudagraph_runtime_mode='eager'),
        _can_fuse_decode_aux=lambda *args: False)
    cls = native_class('Step5Attention', namespace, methods={'forward'})
    attention = cls.__new__(cls)
    attention.use_dsa_backend, attention.fuse_qkv_gate = kind == 'dsa', kind == 'swa'
    attention.fuse_swa_gate, attention.use_head_wise_attn_gate = False, True
    attention.q_size, attention.kv_size, attention.head_dim = 4, 2, 2
    attention.num_heads, attention.num_kv_heads = 2, 1
    attention.rotary_dim, attention.use_rope, attention.zero_centered = 2, True, True
    attention.q_norm = attention.k_norm = SimpleNamespace(weight=object(), variance_epsilon=1e-6)
    attention.rotary_emb = SimpleNamespace(cos_sin_cache=object())
    def layer(forward):
        value = NativeModule()
        value.forward = forward
        return value
    output = Tensor()
    attention.o_proj = layer(lambda value: (output, None))
    attention.g_proj = layer(forbidden)
    attention.attn = layer(lambda *args, **kwargs: Tensor())
    attention.attn.__dict__.update(layer_name='model.layers.0.self_attn', kv_cache_dtype='auto',
        _k_scale=1, _v_scale=1, num_heads=2, num_kv_heads=1, head_size=2, head_size_v=2,
        impl=SimpleNamespace(num_speculative_tokens=0, _split_kv_cache=lambda cache: (object(), object())))
    attention.sparse_attn = None
    if kind == 'swa':
        attention.qkvg_proj = layer(lambda value: (Tensor(), None))
        expected = ['swa_projection', 'swa_prepare', 'swa_attention', 'swa_gate', 'output_projection']
    else:
        attention.qkv_indexer_proj = layer(lambda value: (Tensor(), Tensor()))
        attention.qkv_indexer_proj.split_indexer = lambda value: (Tensor(), Tensor(), Tensor(), Tensor())
        attention.sparse_indexer_w = layer(lambda value: (Tensor(), None))
        attention.sparse_indexer_q_norm = attention.q_norm
        attention.sparse_indexer_k_norm = SimpleNamespace(weight=object(), bias=object())
        sparse_cls = native_class('Step5SparseAttention', namespace,
                                  methods={'forward', '_project_sparse_indexer', '_forward_dsa_attention'})
        sparse = sparse_cls.__new__(sparse_cls)
        sparse.owner, sparse.fuse_qkv_preparation, sparse._indexer_native_rope = attention, True, False
        sparse.sparse_rotary_emb = SimpleNamespace(cos_sin_cache=object())
        attention.sparse_attn = sparse
        expected = ['dsa_projection', 'dsa_prefill_prepare.main_norm_rope_raw_kv', 'indexer_w',
                    'dsa_prefill_prepare.indexer_norm_rope_cast', 'dsa_prefill_root', 'output_projection']
    with module.annotate(torch, attention, kind) as owners:
        originals = list(owners.originals)
        owners.enabled = True
        assert attention(Tensor(), Tensor()) is output
        assert owners.calls == expected and tuple(expected) == module.OWNERS[kind]
        assert owners.active is None
        assert events == [event for scope in expected
                          for event in [('enter', 'vidur_' + scope), ('exit', 'vidur_' + scope)]]
    assert all(getattr(owner, name) is original for owner, name, original in originals)
    if kind == 'dsa':
        assert len(roots) == 1
        assert roots[0][9:11] == (False, False)  # No decode summary update or inactive-row clear.
        assert roots[0][11] is not None  # Native DSA owns its headwise gate.


@pytest.mark.parametrize('kind,index', [('swa', 0), ('dsa', 3)])
@pytest.mark.parametrize('optional_fields', [False, True])
def test_construct_matches_accepted_g1_attention_state_kwargs(monkeypatch, kind, index, optional_fields):
    text = SimpleNamespace(**json.loads((TASK / 'reference_inputs/config.json').read_text())['text_config'])
    text.zero_centered = True
    if optional_fields:
        text.attention_bias, text.use_rope_layers, text.swa_num_attention_heads = False, [0, 3], 16
    native = SimpleNamespace(model_config=SimpleNamespace(hf_config=text),
                             cache_config=SimpleNamespace(sliding_window=512, block_size=16), quant_config=object())
    sparse, layer_types, norm_dtype = object(), object(), object()
    def capture(**kwargs):
        return kwargs
    def types_for(config):
        assert config is text
        return layer_types
    def norm_for(config):
        assert config is text
        return norm_dtype
    fake_vllm(monkeypatch, {
        'vllm.models.step5.nvidia.model': {'Step5Attention': capture, '_get_step5_layer_types': types_for},
        'vllm.models.step5.nvidia.layernorm': {'get_step5_norm_dtype': norm_for},
        'vllm.models.step5.nvidia.config': {'get_step5_sparse_config': lambda config: sparse}})
    actual = module.construct(None, kind, native)
    g1_tree = ast.parse(Path(module.__file__).with_name('check_attention_state.py').read_text())
    call = next(node for node in ast.walk(g1_tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == 'Step5Attention')
    cache_config = copy.copy(native.cache_config)
    cache_config.sliding_window = None
    expected = eval(compile(ast.Expression(call), '<accepted G1 constructor>', 'eval'), {
        'Step5Attention': capture, 'text': text, 'index': index, 'case': kind,
        'cache_config': cache_config, 'native': native, 'sparse': sparse,
        '_get_step5_layer_types': types_for, 'get_step5_norm_dtype': norm_for})
    assert actual == expected
    assert actual['cache_config'] is not native.cache_config and native.cache_config.sliding_window == 512
    assert actual['prefix'] == f'model.layers.{index}.self_attn'
    assert actual['sparse_config'] is (None if kind == 'swa' else sparse)
    native_cls = next(node for node in NATIVE_TREE.body if isinstance(node, ast.ClassDef)
                      and node.name == 'Step5Attention')
    constructor = next(node for node in native_cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    assert set(actual) <= {arg.arg for arg in constructor.args.args if arg.arg != 'self'}


@pytest.mark.parametrize('kind', ['swa', 'dsa'])
def test_measure_samples_persists_every_observation_before_attestation(monkeypatch, tmp_path, kind):
    """Exercise the actual orchestration and JSON writes with native/device boundaries faked."""
    from contextlib import nullcontext
    owner_calls, attested = [], []
    class Value:
        dtype = 'torch.bfloat16'
        shape = (2, 8)
        def clone(self): return Value()
        def copy_(self, value): owner_calls.append('restore')
        def cpu(self): return self
        def cuda(self): return self
        def long(self): return self
        def stride(self): return (8, 1)
        def __getitem__(self, key): return self
        def __floordiv__(self, value): return self
        def __mod__(self, value): return self
        def __mul__(self, value): return self
        def __add__(self, value): return self
        def all(self): return self
        def item(self): return True
    class Generator:
        def __init__(self, **kwargs): pass
        def manual_seed(self, seed): return self
    class Tracer:
        def __init__(self, path): self.trace_path = str(Path(path)/'fake_trace.json')
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get_operation_time_stats(self):
            scopes = module.OWNERS[kind]
            stats = {scope: dict(mean=1.0, kernel_count=1, count=2) for scope in scopes}
            stats[module.WHOLE_FORWARD] = dict(mean=float(len(scopes)), kernel_count=len(scopes), count=2)
            return stats
    tracer_module = ModuleType('frontier.profiling.utils.record_function_tracer')
    tracer_module.RecordFunctionTracer = Tracer
    monkeypatch.setitem(sys.modules, tracer_module.__name__, tracer_module)
    fake_vllm(monkeypatch, {
        'vllm.forward_context': {'set_forward_context': lambda *a, **k: nullcontext()},
        'vllm.v1.attention.backend': {'CommonAttentionMetadata': lambda **k: SimpleNamespace(**k)}})
    torch = SimpleNamespace(Generator=Generator, bfloat16='bf16', int32='int32',
        arange=lambda *a, **k: Value(), tensor=lambda *a, **k: Value(),
        randn=lambda *a, **k: Value(), device=lambda *a: nullcontext(),
        cuda=SimpleNamespace(synchronize=lambda: None), isfinite=lambda value: Value(),
        equal=lambda *a: True, save=lambda content, stream: stream.write(b'FAKE_CONTENT_NOT_NATIVE'),
        profiler=SimpleNamespace(record_function=lambda *a: nullcontext()))
    owners = SimpleNamespace(enabled=False, calls=[])
    def forward(*args):
        if owners.enabled: owners.calls.extend(module.OWNERS[kind])
        return Value()
    native = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(hidden_size=8, moe_num_experts=352)))
    attention = SimpleNamespace(attn=SimpleNamespace(layer_name='fake.attention'), __call__=forward)
    class Attention:
        attn = attention.attn
        def __call__(self, *args): return forward(*args)
    builder = SimpleNamespace(build=lambda *a, **k: SimpleNamespace(step5_runtime=None))
    def attest(*args):
        assert len(list((tmp_path/'observations').glob('*/observation.json'))) == len(SELECTED)
        attested.append('after_all_observations')
        return {'status': 'FAKE_ATTESTATION_NOT_NATIVE'}
    attestation_module = ModuleType('compare_precision_ffn')
    attestation_module.weight_attestation = attest
    monkeypatch.setitem(sys.modules, 'compare_precision_ffn', attestation_module)
    rows = module.measure_samples(torch, tmp_path/'observations', native, kind, SELECTED, PLAN,
        Attention(), SimpleNamespace(block_size=16), [Value()], [Value()], builder, Value(),
        {'backend': 'fake'}, {}, owners, 1, 2, 1024)
    assert len(rows) == 30
    for row in rows:
        path = tmp_path/'observations'/row['sample']['label']/'observation.json'
        saved = json.loads(path.read_text())
        assert saved['kind'] == kind and saved['sample'] == row['sample']
        assert saved['measurement_status'] == 'raw_eager_observation_not_admitted'
        assert saved['reconciliation']['status'] == 'KERNELS_RECONCILED'
        assert saved['repeat_identical'] is True and Path(saved['content']).read_bytes() == b'FAKE_CONTENT_NOT_NATIVE'
    assert attested == ['after_all_observations']
    assert json.loads((tmp_path/'observations'/'weight_attestation.json').read_text())['status'] == 'FAKE_ATTESTATION_NOT_NATIVE'
