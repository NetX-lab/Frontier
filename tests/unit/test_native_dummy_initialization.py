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
    assert inventory['tensors'][0]['scale_role'] == 'unverified'
    assert inventory['tensors'][0]['scale_valid'] is False
    del model.w13_weight_scale
    model.quant_method.mxfp4_backend = SimpleNamespace(name='TRITON')
    inventory = inspect_model(model, config_fixture())
    assert inventory['tensors'] == []
    coverage = next(row for row in inventory['auxiliary']
                    if row['role'] == 'mxfp4_quantization_inventory')
    assert coverage['valid'] is False
