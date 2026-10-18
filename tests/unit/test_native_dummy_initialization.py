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
