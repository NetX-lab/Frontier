"""Measure independent eager attention owners inside the unchanged native Step5Attention forward.

Each preregistered owner is annotated where the native forward calls it, so owners are
siblings and never nest. Whole-forward repeats in the same profiler session reconcile
the owners' kernels. Decode-branch and graph-replay cases remain unmeasured.
"""
import copy
import json
import time

from local_scope_manifest import admit
from local_ffn_measurement import validate_topology

LAYER_INDEX = {'swa': 0, 'dsa': 3}
# Native call order of each kind's prefill forward; a dotted name is one part of a physical owner.
OWNERS = {
    'swa': ('swa_projection', 'swa_prepare', 'swa_attention', 'swa_gate', 'output_projection'),
    'dsa': ('dsa_projection', 'dsa_prefill_prepare.main_norm_rope_raw_kv', 'indexer_w',
            'dsa_prefill_prepare.indexer_norm_rope_cast', 'dsa_prefill_root', 'output_projection'),
}
PROJECTIONS = {'swa': ('qkvg_proj', 'o_proj'), 'dsa': ('qkv_indexer_proj', 'sparse_indexer_w', 'o_proj')}
WHOLE_FORWARD = 'attention_forward'


def samples(plan):
    """Single-request eager BF16 cases; each takes the prefill branch of both attention kinds."""
    return [row for row in plan['cases']
            if row['graph'] == 'eager' and row['rows'] == 1 and row['precision'] == 'bf16']


def chunks(sample, window):
    """Prior chunks of at most one window build the prefix; the sampled query is the last chunk."""
    if not 0 < sample['query'] <= window:
        raise ValueError('The sampled query must fit one scheduled chunk')
    full, rest = divmod(sample['prefix'], window)
    return [window]*full + ([rest] if rest else []) + [sample['query']]


def physical_owners(stats, kind):
    """Charge each physical owner once: its parts' means and kernel counts add."""
    parts = {}
    for scope in OWNERS[kind]:
        parts.setdefault(scope.split('.')[0], []).append(scope)
    return {owner: dict(parts=names, device_duration_mean_ms=sum(stats[n]['mean'] for n in names),
                        kernel_count=sum(stats[n]['kernel_count'] for n in names),
                        part_stats={n: stats[n] for n in names})
            for owner, names in parts.items()}


def reconcile(stats, kind):
    owned = sum(stats[scope]['kernel_count'] for scope in OWNERS[kind])
    forward = stats[WHOLE_FORWARD]
    return dict(owner_kernel_count=owned, forward_kernel_count=forward['kernel_count'],
                unowned_kernel_count=forward['kernel_count']-owned,
                owner_device_mean_ms=sum(stats[scope]['mean'] for scope in OWNERS[kind]),
                forward_device_mean_ms=forward['mean'],
                status='KERNELS_RECONCILED' if forward['kernel_count'] == owned else 'UNOWNED_OR_MISSING_KERNELS')


class OwnerAnnotations:
    """Annotate owners at their native call sites; an owner called inside another is rejected."""

    def __init__(self, record_function):
        self.record_function, self.enabled, self.active, self.calls = record_function, False, None, []
        self.originals = []

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.enabled = False
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)

    def wrap(self, owner, name, scope):
        original = getattr(owner, name)
        self.originals.append((owner, name, original))
        def call(*args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)
            if self.active is not None:
                raise RuntimeError(f'Owner {scope} was called inside owner {self.active}')
            self.calls.append(scope)
            self.active = scope
            try:
                with self.record_function('vidur_'+scope):
                    return original(*args, **kwargs)
            finally:
                self.active = None
        setattr(owner, name, call)


def annotate(torch, module, kind):
    import vllm._custom_ops as ops
    import vllm.models.step5.nvidia.model as model
    import vllm.models.step5.nvidia.ops.indexer_norm_rope as indexer
    import vllm.models.step5.nvidia.ops.qkv_indexer_norm_rope as joint
    owners = OwnerAnnotations(torch.profiler.record_function)
    owners.wrap(module.o_proj, 'forward', 'output_projection')
    if kind == 'swa':
        owners.wrap(module.qkvg_proj, 'forward', 'swa_projection')
        owners.wrap(ops, 'fused_qknorm_rope_cache', 'swa_prepare')
        owners.wrap(module.attn, 'forward', 'swa_attention')
        owners.wrap(model, 'headwise_sigmoid_gate', 'swa_gate')
    else:
        owners.wrap(module.qkv_indexer_proj, 'forward', 'dsa_projection')
        owners.wrap(module.sparse_indexer_w, 'forward', 'indexer_w')
        owners.wrap(ops, 'fused_qknorm_rope_cache', 'dsa_prefill_prepare.main_norm_rope_raw_kv')
        owners.wrap(indexer, 'indexer_norm_rope', 'dsa_prefill_prepare.indexer_norm_rope_cast')
        # The fused decode preparation is unreachable for single-request prefill chunks.
        owners.wrap(joint, 'qkv_indexer_norm_rope', 'dsa_decode_prepare')
        owners.wrap(model, 'step5_dsa_attention_with_output', 'dsa_prefill_root')
    return owners


def construct(torch, kind, native):
    """The accepted G1 attention-state Step5Attention arguments for this kind's layer."""
    from vllm.models.step5.nvidia.model import Step5Attention, _get_step5_layer_types
    from vllm.models.step5.nvidia.layernorm import get_step5_norm_dtype
    from vllm.models.step5.nvidia.config import get_step5_sparse_config
    text, index = native.model_config.hf_config, LAYER_INDEX[kind]
    sparse = get_step5_sparse_config(text)
    cache_config = copy.copy(native.cache_config)
    cache_config.sliding_window = None
    return Step5Attention(hidden_size=text.hidden_size, num_heads=text.num_attention_heads,
        num_kv_heads=text.num_attention_groups, max_position=text.max_position_embeddings, head_dim=text.head_dim,
        rms_norm_eps=text.rms_norm_eps, qkv_bias=getattr(text, 'attention_bias', False), rope_theta=text.rope_theta,
        cache_config=cache_config, quant_config=native.quant_config, rope_scaling=getattr(text, 'rope_scaling', None),
        prefix=f'model.layers.{index}.self_attn', sliding_window=text.sliding_window,
        use_head_wise_attn_gate=text.use_head_wise_attn_gate, layer_types=_get_step5_layer_types(text),
        use_rope_layers=getattr(text, 'use_rope_layers', []), yarn_only_types=getattr(text, 'yarn_only_types', []),
        swa_num_attention_heads=getattr(text, 'swa_num_attention_heads', None), partial_rotary_factor=text.partial_rotary_factors[index],
        zero_centered=text.zero_centered, vllm_config=native, sparse_config=None if kind == 'swa' else sparse,
        norm_dtype=get_step5_norm_dtype(text))


def dispatch(module, kind, builder):
    layer = module.attn
    record = dict(backend=layer.attn_backend.get_name(), builder=type(builder).__name__,
        impl=type(layer.impl).__name__, flash_attn_version=getattr(layer.impl, 'vllm_flash_attn_version', None),
        use_dsa_backend=module.use_dsa_backend, fuse_qkv_gate=module.fuse_qkv_gate, fuse_swa_gate=module.fuse_swa_gate,
        fuse_qkv_preparation=None if module.sparse_attn is None else module.sparse_attn.fuse_qkv_preparation,
        kv_cache=dict(dtype=str(layer.kv_cache.dtype), shape=list(layer.kv_cache.shape)), projections={})
    for name in PROJECTIONS[kind]:
        projection = getattr(module, name)
        record['projections'][name] = projection_dispatch(projection)
    return record


def projection_dispatch(projection):
    """The DSA projection classes call functional linear directly, bypassing quant_method."""
    cls = type(projection)
    direct = cls.__name__ in ('Step5FusedQKVIndexerLinear', 'Step5SparseIndexerIndexTPLinear')
    if direct:
        method = projection.quant_method if cls.__name__ == 'Step5FusedQKVIndexerLinear' else None
        if method is not None and type(method).__name__ != 'UnquantizedLinearMethod':
            raise ValueError('Native fused QKV/indexer construction requires an unquantized method')
        gemm = None
    else:
        from vllm import envs
        from vllm.platforms import current_platform
        from vllm.model_executor.layers.linear import linear_batch_invariant
        method = projection.quant_method
        if type(method).__name__ != 'UnquantizedLinearMethod':
            raise ValueError('Attention projections must preserve native excluded BF16 precision')
        batch_invariant = envs.VLLM_BATCH_INVARIANT and current_platform.is_cuda_alike()
        gemm = linear_batch_invariant if batch_invariant else method._gemm_impl
    if str(projection.weight.dtype) != 'torch.bfloat16':
        raise ValueError('Native attention projection weights must be BF16')
    return dict(class_name=cls.__module__+'.'+cls.__qualname__, forward=cls.forward.__module__+'.'+cls.forward.__qualname__,
        execution=('native_forward_functional_linear_or_batch_invariant' if direct else
                   'quant_method_batch_invariant' if batch_invariant else 'quant_method_gemm'),
        method=None if method is None else type(method).__name__,
        gemm=None if gemm is None else gemm.__module__+'.'+gemm.__qualname__,
        dtype=str(projection.weight.dtype), shape=list(projection.weight.shape), stride=list(projection.weight.stride()))


def bind_cache(torch, native, module, pages):
    """The G1 attention-state cache binding, sized for the largest selected request."""
    from vllm.v1.kv_cache_interface import KVCacheTensor, create_kv_cache_views
    layer = module.attn
    spec = layer.get_kv_cache_spec(native)
    raw = torch.zeros(pages*spec.page_size_bytes, dtype=torch.int8, device='cuda')
    layer.kv_cache = create_kv_cache_views(raw, spec, pages, layer.attn_backend.supported_kv_cache_layouts()[0],
        KVCacheTensor(size=raw.numel(), layers=[layer.layer_name], layer_stride=raw.numel(),
                      block_stride=spec.page_size_bytes))[0]
    state = [layer.kv_cache]
    if module.sparse_summary_cache is not None:
        module.sparse_summary_cache.bind_kv_cache_side_storage(native.compilation_config.static_forward_context)
        cache = module.sparse_summary_cache._cache
        state += [cache._mean, cache.numerator, cache.denominator, cache.maximum, cache.last_slot]
        if cache.request_journal_width:
            state += [cache.request_journal_values, cache.request_journal_last, cache.request_journal_tag]
    assert layer.kv_cache.dtype == torch.bfloat16 and not module.fuse_swa_gate
    return spec, state


def measure(task, output, native, kind, *, warmups, repeats, tensor_copy_bytes=None):
    if kind not in OWNERS:
        raise ValueError('Only preregistered SWA and DSA attention owners are supported')
    if warmups < 1 or repeats < 2:
        raise ValueError('Positive warmups and at least two measured repeats required')
    if tensor_copy_bytes is None or tensor_copy_bytes <= 0:
        raise ValueError('A positive tensor-copy byte budget is required')
    plan = json.loads((task/'local_scope_sampling_01.json').read_text())
    admit(plan, task)
    selected = samples(plan)
    import torch
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
    from vllm.model_executor.model_loader.reload.layerwise import record_metadata_for_reloading
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.distributed.parallel_state import get_world_group, get_tp_group, get_dp_group, get_pp_group, get_ep_group
    groups = dict(world=get_world_group(), tp=get_tp_group(), dp=get_dp_group(), pp=get_pp_group(), ep=get_ep_group())
    validate_topology(native, groups.values())
    groups = {name: dict(world_size=group.world_size, rank=group.rank,
                         backend=torch.distributed.get_backend(group.device_group)) for name, group in groups.items()}
    module = construct(torch, kind, native)
    record_metadata_for_reloading(module)
    DummyModelLoader(native.load_config).load_weights(module, native.model_config)
    process_weights_after_loading(module, native.model_config, torch.device('cuda'))
    pages = max(sample['table_pages'] for sample in selected)
    spec, state = bind_cache(torch, native, module, pages)
    initial = [tensor.clone() for tensor in state]
    layer = module.attn
    builder = layer.attn_backend.get_builder_cls()(kv_cache_spec=spec, layer_names=[layer.layer_name],
                                                   vllm_config=native, device=torch.device('cuda'))
    table = torch.arange(pages-1, -1, -1, dtype=torch.int32, device='cuda')[None]
    record = dispatch(module, kind, builder)
    required = {s['id']: s['required_backend'] for s in plan['scopes']}
    root = 'swa_attention' if kind == 'swa' else 'dsa_prefill_root'
    assert record['backend'] == required[root].split('/')[0]
    assert kind == 'dsa' or record['flash_attn_version'] == 3
    assert record['use_dsa_backend'] == (kind == 'dsa')
    assert kind == 'dsa' or record['fuse_qkv_gate'], 'SWA requires its native packed projection'
    with annotate(torch, module, kind) as owners:
        return measure_samples(torch, output, native, kind, selected, plan, module, spec, state,
                               initial, builder, table, record, groups, owners, warmups, repeats, tensor_copy_bytes)


def measure_samples(torch, output, native, kind, selected, plan, module, spec, state,
                    initial, builder, table, record, groups, owners, warmups, repeats, tensor_copy_bytes):
    from vllm.forward_context import set_forward_context
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from frontier.profiling.utils.record_function_tracer import RecordFunctionTracer
    layer, window, hidden = module.attn, plan['geometry']['W'], native.model_config.hf_config.hidden_size

    def step(previous, count):
        positions = torch.arange(previous, previous+count, device='cuda')
        slots = table[0, positions//spec.block_size].long()*spec.block_size+positions%spec.block_size
        starts = torch.tensor([0, count], dtype=torch.int32, device='cpu')
        lengths = torch.tensor([previous+count], dtype=torch.int32, device='cpu')
        common = CommonAttentionMetadata(query_start_loc=starts.cuda(), query_start_loc_cpu=starts,
            seq_lens=lengths.cuda(), num_reqs=1, num_actual_tokens=count, max_query_len=count,
            max_seq_len=previous+count, block_table_tensor=table, slot_mapping=slots, positions=positions,
            seq_lens_cpu_upper_bound=lengths)
        start = time.perf_counter_ns()
        with torch.device('cpu'):
            metadata = builder.build(0, common, fast_build=False)
        return positions, slots, metadata, (time.perf_counter_ns()-start)/1e6

    def forward(positions, x, slots, metadata):
        with set_forward_context({layer.layer_name: metadata}, native, num_tokens=x.shape[0],
                                 slot_mapping={layer.layer_name: slots}):
            return module(positions, x)

    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for sample in selected:
        cell = output/sample['label']; (cell/'profiler_traces').mkdir(parents=True, exist_ok=False)
        for tensor, value in zip(state, initial):
            tensor.copy_(value)
        generator = torch.Generator(device='cuda').manual_seed(sample['content_seed'])
        schedule, previous, prefix_inputs = chunks(sample, window), 0, []
        for count in schedule[:-1]:
            x = torch.randn((count, hidden), generator=generator, device='cuda', dtype=torch.bfloat16)
            prefix_inputs.append(x)
            positions, slots, metadata, _ = step(previous, count)
            forward(positions, x, slots, metadata)
            previous += count
        query = schedule[-1]
        x = torch.randn((query, hidden), generator=generator, device='cuda', dtype=torch.bfloat16)
        prefix_state = [tensor.clone() for tensor in state]
        host = dict(owner_pass=[], forward_pass=[])
        launches = dict(owner_pass=[], forward_pass=[])
        outputs = []

        def run(annotated, scope=None):
            for tensor, value in zip(state, prefix_state):
                tensor.copy_(value)
            positions, slots, metadata, host_ms = step(previous, query)
            torch.cuda.synchronize()
            owners.enabled, owners.calls = annotated, []
            start = time.perf_counter_ns()
            if scope is None:
                y = forward(positions, x, slots, metadata)
            else:
                with torch.profiler.record_function('vidur_'+scope):
                    y = forward(positions, x, slots, metadata)
            elapsed = (time.perf_counter_ns()-start)/1e6
            owners.enabled = False
            if annotated and tuple(owners.calls) != OWNERS[kind]:
                raise RuntimeError(f'Native {kind} prefill called owners {owners.calls}, expected {OWNERS[kind]}')
            plan = getattr(metadata, 'step5_runtime', None)
            layout = dict(input_dtype=str(x.dtype), input_shape=list(x.shape), input_stride=list(x.stride()),
                positions=positions, slots=slots, block_table=table,
                native_plan=plan is not None, staged=plan is not None and plan[7] is not None,
                paged_prefill=plan is not None and plan[5] is not None)
            return y, host_ms, elapsed, layout

        for _ in range(warmups):
            reference, _, _, layout = run(True)
        torch.cuda.synchronize()
        assert torch.isfinite(reference).all().item(), 'Independent shape outputs must be finite'
        reference = reference.clone()
        with RecordFunctionTracer(str(cell)) as tracer:
            for _ in range(repeats):
                for annotated, scope, name in ((True, None, 'owner_pass'), (False, WHOLE_FORWARD, 'forward_pass')):
                    y, host_ms, elapsed, layout = run(annotated, scope)
                    outputs.append(y.clone())
                    host[name].append(host_ms); launches[name].append(elapsed)
        stats = tracer.get_operation_time_stats()
        assert all(stats[scope]['count'] == repeats for scope in OWNERS[kind]+(WHOLE_FORWARD,))
        assert set(stats) == set(OWNERS[kind]+(WHOLE_FORWARD,)), 'Unexpected annotated scope'
        with (cell/'content.pt').open('xb') as stream:
            torch.save(dict(input=x.cpu(), prefix_inputs=[value.cpu() for value in prefix_inputs],
                            schedule=schedule, output=reference.cpu(),
                            **{name: layout[name].cpu() for name in ('positions', 'slots', 'block_table')}), stream)
        row = dict(kind=kind, layer=layer.layer_name, sample=sample, schedule=schedule, branch='prefill',
                   layout={name: value for name, value in layout.items() if name not in ('positions', 'slots', 'block_table')},
                   groups=groups, dispatch=record, owners=physical_owners(stats, kind), forward=stats[WHOLE_FORWARD],
                   reconciliation=reconcile(stats, kind), host_metadata_build_ms=host, host_profiled_launch_ms=launches,
                   repeat_identical=all(torch.equal(value, reference) for value in outputs),
                   trace=tracer.trace_path, content=str(cell/'content.pt'), charges=1,
                   graph_status='unmeasured', gap_status='unmeasured_not_pure_device_gap',
                   content_status='finite_and_repeat_identity_only_reference_comparison_pending',
                   measurement_status='raw_eager_observation_not_admitted')
        rows.append(row)
        with (cell/'observation.json').open('x') as stream:
            json.dump(row, stream, indent=2, default=float); stream.write('\n')
    from compare_precision_ffn import weight_attestation
    from check_role_capacity import diagnostic, save
    def attest():
        torch.cuda.synchronize()
        snapshot = weight_attestation(module, tensor_copy_bytes, native.model_config.hf_config.moe_num_experts)
        snapshot['time_ns'] = time.time_ns()
        save(output/'weight_attestation.json', snapshot)
    diagnostic(attest)
    return rows
