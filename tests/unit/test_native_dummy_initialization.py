"""Validate native tensor roles through the existing model-inventory boundary."""
from types import SimpleNamespace

import pytest
import torch

from tests.comparison.calibration.native_dummy_initialization import inspect_model


def config_fixture():
    return SimpleNamespace(
        model_config=SimpleNamespace(architecture='GlmMoeDsaForCausalLM', quantization='fp8', dtype=torch.bfloat16,
            hf_config=SimpleNamespace(to_dict=lambda: {})),
        load_config=SimpleNamespace(load_format='dummy'),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, enable_expert_parallel=False,
                                        all2all_backend='allgather_reducescatter'),
        scheduler_config=SimpleNamespace(async_scheduling=True, max_num_batched_tokens=256, max_num_seqs=2),
        kernel_config=SimpleNamespace(moe_backend='auto'),
        cache_config=SimpleNamespace(cache_dtype='auto', block_size=64),
    )


def test_fp32_triton_block_scales_preserve_signed_native_state():
    model = torch.nn.Module()
    model.register_parameter('w13_weight_scale_inv', torch.nn.Parameter(torch.tensor([[[-0.001], [0.001]]])))
    model.register_parameter('w2_weight_scale_inv', torch.nn.Parameter(torch.tensor([[[0.002]]])))
    model.register_parameter('w13_weight', torch.nn.Parameter(torch.ones(1, 256, 128).to(torch.float8_e4m3fn)))
    model.register_parameter('w2_weight', torch.nn.Parameter(torch.ones(1, 128, 128).to(torch.float8_e4m3fn)))
    model.weight_block_size = [128, 128]
    model.quant_method = type('Fp8MoEMethod', (), {})()
    model.quant_method.fp8_backend = SimpleNamespace(name='TRITON')
    model.quant_method.block_quant = True
    model.quant_method.moe_kernel = SimpleNamespace(fused_experts=SimpleNamespace(), prepare_finalize=SimpleNamespace())
    model.quant_method.moe_quant_config = SimpleNamespace(
        w1_scale=model.w13_weight_scale_inv, w2_scale=model.w2_weight_scale_inv,
        a1_scale=None, a2_scale=None, block_shape=[128, 128])
    config = config_fixture()
    rows = inspect_model(model, config)['tensors']
    assert [row['scale_valid'] for row in rows[:2]] == [True, True]
    assert model.w13_weight_scale_inv.flatten().tolist() == [-0.0010000000474974513, 0.0010000000474974513]
    model.w13_weight_scale_inv.data[0, 0, 0] = float('nan')
    assert inspect_model(model, config)['tensors'][0]['scale_valid'] is False
    model.w13_weight_scale_inv.data[0, 0, 0] = 0
    assert inspect_model(model, config)['tensors'][0]['scale_valid'] is True
    model.w2_weight = torch.nn.Parameter(model.w2_weight.float())
    assert inspect_model(model, config)['tensors'][1]['scale_valid'] is False
    model.quant_method.moe_quant_config.block_shape = [64, 128]
    assert inspect_model(model, config)['auxiliary'][0]['valid'] is False


def test_attention_host_copies_are_read_and_mismatch_is_invalid():
    model = type('MLAAttention', (torch.nn.Module,), {})()
    model.kv_cache_dtype = 'auto'
    model.dcp_q_replicate = False
    model.W_UK_T_dcp_qrep = None
    model.num_heads, model.qk_nope_head_dim, model.kv_lora_rank, model.v_head_dim = 2, 4, 8, 4
    model.register_parameter('W_UK_T', torch.nn.Parameter(torch.ones(2, 4, 8, dtype=torch.bfloat16)))
    model.register_parameter('W_UV', torch.nn.Parameter(torch.ones(2, 8, 4, dtype=torch.bfloat16)))
    model.attn_backend = type('AttentionBackend', (), {})
    model.impl = SimpleNamespace()
    model.prefill_backend = SimpleNamespace()
    for field in ('q', 'k', 'v', 'prob'):
        model.register_buffer(f'_{field}_scale', torch.tensor(1.0))
        setattr(model, f'_{field}_scale_float', 1.0)
    for field in ('k', 'v'):
        setattr(model, f'_{field}_scale_cpu', torch.tensor(1.0))
    rows = inspect_model(model, config_fixture())['auxiliary']
    assert rows[0]['valid'] is True
    model._k_scale_cpu.fill_(2.0)
    rows = inspect_model(model, config_fixture())['auxiliary']
    assert rows[0]['valid'] is False
    assert model._k_scale_cpu.item() == 2.0
    model.W_UK_T = torch.nn.Parameter(torch.ones(2, 4, 7, dtype=torch.bfloat16))
    assert inspect_model(model, config_fixture())['auxiliary'][1]['valid'] is False


@pytest.mark.parametrize('scale', [-0.001, 0.0, 2 ** -22, 0.001, float('nan'), float('inf')])
def test_deepgemm_scale_domain_is_not_the_triton_scale_domain(scale):
    model = torch.nn.Module()
    model.register_parameter('weight_scale_inv', torch.nn.Parameter(torch.tensor([[scale]])))
    model.register_parameter('weight', torch.nn.Parameter(torch.ones(128, 128).to(torch.float8_e4m3fn)))
    model.weight_block_size = [128, 128]
    model.quant_method = type('Fp8LinearMethod', (), {})()
    model.quant_method.block_quant = True
    kernel = type('FlashInferFp8DeepGEMMDynamicBlockScaledKernel', (), {})()
    kernel.fallback = SimpleNamespace(use_deep_gemm_e8m0=True)
    model.quant_method.fp8_linear = kernel
    row = inspect_model(model, config_fixture())['tensors'][0]
    assert row['scale_valid'] is (scale == 2 ** -22)
    model.weight = torch.nn.Parameter(model.weight.float())
    assert inspect_model(model, config_fixture())['tensors'][0]['scale_valid'] is False
    model.quant_method = None
    assert inspect_model(model, config_fixture())['tensors'][0]['scale_valid'] is False


@pytest.mark.parametrize('value', [torch.empty(0), torch.ones(2, 1), torch.ones(1, 1, dtype=torch.float64)])
def test_deepgemm_scale_shape_dtype_and_empty_state_fail_closed(value):
    model = torch.nn.Module()
    model.register_parameter('weight_scale_inv', torch.nn.Parameter(value))
    model.register_parameter('weight', torch.nn.Parameter(torch.ones(128, 128).to(torch.float8_e4m3fn)))
    model.weight_block_size = [128, 128]
    kernel = type('FlashInferFp8DeepGEMMDynamicBlockScaledKernel', (), {})()
    kernel.fallback = SimpleNamespace(use_deep_gemm_e8m0=True)
    model.quant_method = type('Fp8LinearMethod', (), {})()
    model.quant_method.block_quant, model.quant_method.fp8_linear = True, kernel
    assert inspect_model(model, config_fixture())['tensors'][0]['scale_valid'] is False


def test_deepgemm_batched_scale_preserves_the_leading_dimension():
    model = torch.nn.Module()
    model.register_parameter('weight_scale_inv', torch.nn.Parameter(torch.ones(2, 1, 2)))
    model.register_parameter('weight', torch.nn.Parameter(
        torch.ones(2, 128, 256).to(torch.float8_e4m3fn)))
    model.weight_block_size = [128, 128]
    model.is_bmm, model.bmm_batch_size = True, 2
    model.quant_method = type('Fp8LinearMethod', (), {})()
    model.quant_method.block_quant = True
    kernel = type('FlashInferFp8DeepGEMMDynamicBlockScaledKernel', (), {})()
    kernel.fallback = SimpleNamespace(use_deep_gemm_e8m0=True)
    model.quant_method.fp8_linear = kernel
    assert inspect_model(model, config_fixture())['tensors'][0]['scale_valid'] is True
    weight, scales = model.weight, model.weight_scale_inv
    for invalid in (torch.ones(1, 2), torch.ones(1, 1, 2), torch.ones(2, 2, 1),
                    torch.ones(2, 1, 2, dtype=torch.float64), torch.full((2, 1, 2), 0.3)):
        model.weight_scale_inv = torch.nn.Parameter(invalid)
        assert inspect_model(model, config_fixture())['tensors'][0]['scale_valid'] is False
    model.weight_scale_inv = scales
    model.bmm_batch_size = 1
    assert inspect_model(model, config_fixture())['tensors'][0]['scale_valid'] is False
    model.bmm_batch_size, model.is_bmm = 2, False
    assert inspect_model(model, config_fixture())['tensors'][0]['scale_valid'] is False
    assert model.weight is weight


def v4_fixture():
    model = type('DeepseekV4ForCausalLM', (torch.nn.Module,), {})()
    model.model = type('DeepseekV4Model', (torch.nn.Module,), {})()
    model.model.start_layer, model.model.end_layer = 0, 1
    layer = type('DeepseekV4DecoderLayer', (torch.nn.Module,), {})()
    layer.hc_mult, layer.hidden_size = 2, 2
    layer.hc_attn_fn = torch.nn.Parameter(torch.arange(32.).reshape(8, 4), requires_grad=False)
    layer.hc_attn_fn_broadcast = torch.tensor(
        [[2., 4.], [10., 12.], [18., 20.], [26., 28.],
         [34., 36.], [42., 44.], [50., 52.], [58., 60.]])
    layer.ffn = type('DeepseekV4MoE', (torch.nn.Module,), {})()
    layer.ffn.gate = torch.nn.Module()
    layer.ffn.hash_indices_dtype = torch.int32
    layer.ffn.use_mega_moe = False
    layer.ffn.gate.tid2eid = torch.nn.Parameter(torch.zeros(4, 2, dtype=torch.int32), requires_grad=False)
    model.model.layers = torch.nn.ModuleList([layer])
    config = config_fixture()
    config.model_config.hf_config = SimpleNamespace(
        n_routed_experts=3, num_hash_layers=1, vocab_size=4,
        num_experts_per_tok=2, to_dict=lambda: {})
    return model, config, layer


def test_v4_plain_mhc_broadcast_is_checked_without_mutation():
    model, config, layer = v4_fixture()
    source = layer.hc_attn_fn.clone()
    broadcast = layer.hc_attn_fn_broadcast
    rows = inspect_model(model, config)['auxiliary']
    row = next(row for row in rows if row['role'] == 'mhc_first_layer_broadcast')
    assert row['valid'] is True
    assert row['shape'] == [8, 2] and row['dtype'] == 'torch.float32'
    assert layer.hc_attn_fn_broadcast is broadcast
    assert torch.equal(layer.hc_attn_fn, source)
    for invalid in (None, torch.ones(8, 2), broadcast.double(),
                    broadcast.clone().requires_grad_(), broadcast[:, :1],
                    torch.full_like(broadcast, float('nan'))):
        layer.hc_attn_fn_broadcast = invalid
        row = next(row for row in inspect_model(model, config)['auxiliary']
                   if row['role'] == 'mhc_first_layer_broadcast')
        assert row['valid'] is False


def test_v4_hash_table_checks_expected_layers_and_integer_contract():
    model, config, layer = v4_fixture()
    def hashes():
        return next(row for row in inspect_model(model, config)['auxiliary']
                    if row['role'] == 'v4_hash_tables')
    assert hashes()['valid'] is True
    assert hashes()['expected_count'] == hashes()['observed_count'] == 1
    for invalid in (None, torch.zeros(4, 1, dtype=torch.int32),
                    torch.zeros(4, 2), torch.zeros(4, 2, dtype=torch.int64),
                    torch.full((4, 2), -1, dtype=torch.int32),
                    torch.full((4, 2), 3, dtype=torch.int32)):
        layer.ffn.gate.tid2eid = (None if invalid is None else
                                torch.nn.Parameter(invalid, requires_grad=False))
        assert hashes()['valid'] is False
    layer.ffn.use_mega_moe = True
    layer.ffn.gate.tid2eid = torch.nn.Parameter(torch.zeros(4, 2, dtype=torch.int64), requires_grad=False)
    assert hashes()['valid'] is True
    config.model_config.hf_config.num_hash_layers = 0
    assert hashes()['valid'] is False
    layer.ffn.gate.tid2eid = None
    assert hashes()['valid'] is True


def test_v4_mhc_coefficients_are_signed_native_parameters_not_quant_scales():
    model, config, layer = v4_fixture()
    layer.hc_attn_scale = torch.nn.Parameter(torch.tensor([-0.001, 0., 0.001]))
    layer.hc_ffn_scale = torch.nn.Parameter(torch.zeros(3))
    model.model.hc_head_scale = torch.nn.Parameter(torch.zeros(1))
    rows = [row for row in inspect_model(model, config)['tensors'] if 'scale' in row['name']]
    assert len(rows) == 3
    assert all(row['scale_role'] == 'mhc_native_coefficients' and row['scale_valid'] for row in rows)
    layer.hc_attn_scale = torch.nn.Parameter(torch.zeros(2))
    row = next(row for row in inspect_model(model, config)['tensors']
               if row['name'].endswith('hc_attn_scale'))
    assert row['scale_valid'] is False


def test_v4_mxfp4_backend_is_reported_without_qualifying_unknown_packed_scales():
    model = torch.nn.Module()
    model.quant_method = type('Mxfp4MoEMethod', (), {})()
    model.quant_method.mxfp4_backend = SimpleNamespace(name='MARLIN')
    model.quant_method.experts_cls = type('MarlinExperts', (), {})
    model.register_parameter('w13_weight_scale', torch.nn.Parameter(
        torch.zeros(1, 2, 1, dtype=torch.uint8), requires_grad=False))
    inventory = inspect_model(model, config_fixture())
    assert inventory['backends']['']['backend'] == 'MARLIN'
    assert inventory['tensors'][0]['scale_role'] == 'marlin_mxfp4_e8m0_multiplier'
    assert inventory['tensors'][0]['scale_valid'] is False
    del model.w13_weight_scale
    model.quant_method.mxfp4_backend = SimpleNamespace(name='TRITON')
    inventory = inspect_model(model, config_fixture())
    assert inventory['tensors'] == []
    coverage = next(row for row in inventory['auxiliary']
                    if row['role'] == 'mxfp4_quantization_inventory')
    assert coverage['valid'] is False


def marlin_fixture():
    model = torch.nn.Module()
    for name, shape, dtype in (
        ('w13_weight', (2, 8, 256), torch.int32), ('w2_weight', (2, 4, 256), torch.int32),
        ('w13_weight_scale', (2, 4, 128), torch.uint8), ('w2_weight_scale', (2, 2, 128), torch.uint8),
    ):
        value = torch.zeros(shape, dtype=dtype)
        if 'scale' in name:
            value = value.view(torch.float8_e8m0fnu)
        model.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
    method = type('Mxfp4MoEMethod', (), {})()
    method.num_experts, method.intermediate_size, method.hidden_size = 2, 64, 128
    method.mxfp4_backend = SimpleNamespace(name='MARLIN')
    method.experts_cls = type('MarlinExperts', (), {})
    quant = SimpleNamespace(use_mxfp4_w4a16=True, a1_scale=None, a2_scale=None,
        w1_scale=model.w13_weight_scale, w2_scale=model.w2_weight_scale)
    experts = method.experts_cls()
    experts.quant_config, experts.input_dtype = quant, None
    method.moe_quant_config = quant
    method.moe_kernel = SimpleNamespace(fused_experts=experts, prepare_finalize=SimpleNamespace())
    model.quant_method = method
    return model


def test_marlin_post_load_scales_and_packed_weights_are_checked_without_mutation():
    model = marlin_fixture()
    before = {name: (value, value.view(torch.uint8).clone()) for name, value in model.named_parameters()}
    inventory = inspect_model(model, config_fixture())
    assert inventory['auxiliary'][0]['valid'] is True
    assert all(row['scale_valid'] for row in inventory['tensors'] if 'scale' in row['name'])
    assert inventory['tensors'][2]['minimum'] == 2 ** -127
    for name, value in model.named_parameters():
        assert value is before[name][0]
        assert torch.equal(value.view(torch.uint8), before[name][1])


@pytest.mark.parametrize('fault', ['weight_dtype', 'weight_shape', 'scale_dtype', 'scale_shape',
    'scale_stride', 'scale_nan', 'alias', 'consumer', 'input_dtype', 'quantization', 'missing_weight',
    'missing_scale', 'unknown_backend'])
def test_marlin_rejects_incompatible_post_load_state(fault):
    model = marlin_fixture()
    method = model.quant_method
    if fault == 'weight_dtype':
        model.w13_weight = torch.nn.Parameter(model.w13_weight.float(), requires_grad=False)
    elif fault == 'weight_shape':
        model.w13_weight = torch.nn.Parameter(torch.zeros(2, 8, 128, dtype=torch.int32), requires_grad=False)
    elif fault in ('scale_dtype', 'scale_shape', 'scale_stride', 'scale_nan'):
        value = model.w13_weight_scale
        if fault == 'scale_dtype':
            value = value.view(torch.uint8)
        elif fault == 'scale_shape':
            value = value[:, :2]
        elif fault == 'scale_stride':
            value = value.transpose(1, 2).contiguous().transpose(1, 2)
        else:
            value.view(torch.uint8)[0, 0, 0] = 255
        model.w13_weight_scale = torch.nn.Parameter(value, requires_grad=False)
        method.moe_quant_config.w1_scale = model.w13_weight_scale
    elif fault == 'alias':
        method.moe_quant_config.w1_scale = model.w13_weight_scale.clone()
    elif fault == 'consumer':
        method.moe_kernel.fused_experts = SimpleNamespace()
    elif fault == 'input_dtype':
        method.moe_kernel.fused_experts.input_dtype = torch.float8_e4m3fn
    elif fault == 'quantization':
        method.moe_quant_config.use_mxfp4_w4a16 = False
    elif fault == 'missing_weight':
        del model.w13_weight
    elif fault == 'missing_scale':
        del model.w13_weight_scale
    else:
        method.mxfp4_backend.name = 'TRITON'
    assert inspect_model(model, config_fixture())['auxiliary'][0]['valid'] is False


def v4_attention_fixture(compress_ratio):
    model, config, layer = v4_fixture()
    hf = config.model_config.hf_config
    hf.head_dim, hf.qk_rope_head_dim, hf.num_attention_heads = 512, 64, 64
    hf.index_n_heads, hf.index_head_dim, hf.index_topk = 4, 128, 8
    hf.compress_ratios = [compress_ratio]
    config.scheduler_config.max_num_batched_tokens = 4
    core = model.model
    core.topk_indices_buffer = torch.empty(4, 8, dtype=torch.int32)
    pool = type('DeepseekV4EagerScratchPool', (), {})()
    pool._ALIGNMENT, pool._q = 256, torch.empty(4, 64, 512, dtype=torch.bfloat16)
    pool._storage = torch.empty(8192, dtype=torch.uint8)
    pool._fp4_template = (pool._storage[:1024].view(4, 4, 64),
        pool._storage[1024:1088].view(4, 4, 4), pool._storage[1280:1344].view(torch.float32).view(4, 4))
    pool._global_template = (pool._storage[:128].view(torch.int32).view(4, 8),
        pool._storage[256:272].view(torch.int32))
    pool._compressor_template = pool._storage.view(torch.float32).view(4, 512)
    core.eager_scratch_pool = pool
    attn = type('DeepseekV4FlashMLAAttention', (torch.nn.Module,), {})()
    layer.attn = attn
    attn.prefix, attn.compress_ratio = 'model.layers.0.attn', max(1, compress_ratio)
    attn.kv_cache_dtype, attn.kv_cache_torch_dtype = 'fp8_ds_mla', torch.uint8
    attn.topk_indices_buffer, attn.eager_scratch_pool = core.topk_indices_buffer, pool
    attn.backend_cls = type('DeepseekV4FlashMLABackend', (), {})
    attn.swa_backend_cls = type('DeepseekSparseSWAFlashMLABackend', (), {})
    attn._einsum_recipe, attn._tma_aligned_scales = (1, 128, 128), False
    attn.padded_heads = 64
    attn.attn_sink = torch.nn.Parameter(torch.zeros(64))
    attn.n_local_groups, attn.wo_a = 8, torch.nn.Module()
    attn.wo_a.is_bmm, attn.wo_a.bmm_batch_size = True, 8
    attn.kv_cache = torch.empty(2, 64, 584, dtype=torch.uint8) if compress_ratio > 1 else torch.tensor([])
    attn.swa_cache_layer = type('DeepseekV4SWACache', (torch.nn.Module,), {})()
    attn.swa_cache_layer.kv_cache = torch.empty(2, 64, 584, dtype=torch.uint8)
    attn.indexer, attn.compressor = None, None

    def compressor(target, head_dim):
        comp = type('DeepseekCompressor', (torch.nn.Module,), {})()
        comp.compress_ratio, comp.coff = compress_ratio, 1 + (compress_ratio == 4)
        comp.head_dim, comp.max_num_batched_tokens = head_dim, 4
        comp.k_cache_prefix = target.prefix
        comp._static_forward_context = {target.prefix: target}
        comp.eager_scratch_pool = pool
        comp._compress_scratch, comp._use_two_stage_fused_compressor = None, False
        comp.ape = torch.nn.Parameter(torch.zeros(compress_ratio, comp.coff * head_dim))
        state = type('CompressorStateCache', (torch.nn.Module,), {})()
        state.state_dim, state.dtype = 2 * comp.coff * head_dim, torch.float32
        state.kv_cache = torch.empty(2, 4, state.state_dim)
        comp.state_cache = state
        return comp

    if compress_ratio > 1:
        attn.compressor = compressor(attn, 512)
    if compress_ratio == 4:
        ix = type('DeepseekV4Indexer', (torch.nn.Module,), {})()
        ix.topk_indices_buffer, ix.eager_scratch_pool = core.topk_indices_buffer, pool
        ix.scale_fmt, ix.quant_block_size, ix.use_fp4_kv, ix.head_dim = 'ue8m0', 128, False, 128
        ix.k_cache = type('DeepseekV4IndexerCache', (torch.nn.Module,), {})()
        ix.k_cache.head_dim, ix.k_cache.prefix = 132, 'model.layers.0.attn.indexer.k_cache'
        ix.k_cache.kv_cache = torch.empty(2, 64, 132, dtype=torch.uint8)
        ix.compressor = compressor(ix.k_cache, 128)
        ix.indexer_op = SimpleNamespace(k_cache=ix.k_cache)
        attn.indexer = ix
    return model, config, attn


@pytest.mark.parametrize('compress_ratio', [0, 4, 128])
def test_v4_attention_auxiliary_layouts_have_explicit_content_limits(compress_ratio):
    model, config, attn = v4_attention_fixture(compress_ratio)
    model.model.eager_scratch_pool._q.fill_(float('nan'))
    if attn.compressor is not None:
        attn.compressor.state_cache.kv_cache.fill_(float('nan'))
    tensors = {name: (tensor, tensor.clone()) for name, tensor in model.named_parameters()}
    inventory = inspect_model(model, config)
    assert all(row['valid'] for row in inventory['auxiliary'])
    allocations = [row for row in inventory['auxiliary'] if row['role'] == 'v4_cache_allocation']
    assert allocations and all(row['content_checked'] is False for row in allocations)
    assert inventory['backends']['model.layers.0.attn']['einsum_recipe'] == (1, 128, 128)
    for name, tensor in model.named_parameters():
        assert tensor is tensors[name][0] and torch.equal(tensor, tensors[name][1])


@pytest.mark.parametrize('fault', ['scratch_alias', 'scratch_shape', 'topk_alias', 'indexer_alias',
    'indexer_format', 'missing_indexer', 'cache_dtype', 'cache_width', 'state_dim', 'compressor_target'])
def test_v4_attention_rejects_missing_or_incompatible_auxiliary_state(fault):
    model, config, attn = v4_attention_fixture(4)
    if fault == 'scratch_alias':
        pool = model.model.eager_scratch_pool
        pool._global_template = (pool._global_template[0].clone(), pool._global_template[1])
    elif fault == 'scratch_shape':
        model.model.eager_scratch_pool._q = torch.empty(4, 32, 512, dtype=torch.bfloat16)
    elif fault == 'topk_alias':
        attn.topk_indices_buffer = attn.topk_indices_buffer.clone()
    elif fault == 'indexer_alias':
        attn.indexer.indexer_op.k_cache = torch.nn.Module()
    elif fault == 'indexer_format':
        attn.indexer.use_fp4_kv = True
    elif fault == 'missing_indexer':
        attn.indexer = None
    elif fault == 'cache_dtype':
        attn.swa_cache_layer.kv_cache = attn.swa_cache_layer.kv_cache.float()
    elif fault == 'cache_width':
        attn.kv_cache = torch.empty(2, 64, 576, dtype=torch.uint8)
    elif fault == 'state_dim':
        attn.compressor.state_cache.state_dim -= 1
    else:
        attn.compressor._static_forward_context = {}
    assert any(row['valid'] is False for row in inspect_model(model, config)['auxiliary'])
