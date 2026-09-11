from types import SimpleNamespace

from frontier.config.model_config import QuantizationConfig
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.utils.confirmation import build_moe_config_sections
from frontier.types import ActivationType, NormType


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        disable_ray=True,
        num_gpus=8,
        device="mi355x",
        output_dir="out",
        profile_method="cuda_event",
        num_tensor_parallel_workers=[1],
        expert_parallel_sizes=[8],
        use_fp8=False,
        per_channel_quant=False,
        block_shape=None,
        enable_load_imbalance=False,
        load_distributions=["uniform"],
        num_samples_per_distribution=1,
        max_tokens=1,
    )


def _model(quantization_config: QuantizationConfig | None) -> ModelConfig:
    return ModelConfig(
        name="moe-confirmation-test",
        num_layers=1,
        num_q_heads=8,
        num_kv_heads=1,
        embedding_dim=1024,
        mlp_hidden_dim=256,
        max_position_embeddings=128,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=False,
        vocab_size=1024,
        is_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        dtype="bfloat16",
        quantization_config=quantization_config,
    )


def test_moe_confirmation_reports_selective_mxfp4_grouped_gemm() -> None:
    model = _model(
        QuantizationConfig(
            quant_method="fp4",
            activation_scheme="dynamic",
            quantized_operations=["moe_grouped_gemm"],
        )
    )

    sections = build_moe_config_sections(
        args=_args(),
        model_config=model,
        num_tokens_count=1,
        use_vllm_kernel=True,
        precision_str="BF16",
        torch_dtype="torch.bfloat16",
    )

    precision_section = dict(dict(sections)["Precision & Quantization"])
    operations = dict(dict(sections)["MoE Operations by Parallelism"])[""]

    assert precision_section["Base Precision (dtype)"] == "BF16 (torch.bfloat16)"
    assert precision_section["Grouped GEMM Precision"] == "FP4"
    assert "method=fp4" in precision_section["Quantization"]
    assert "moe_gating_linear          : BF16" in operations
    assert "moe_grouped_gemm           : FP4" in operations


def test_moe_confirmation_keeps_unquantized_grouped_gemm_at_base_precision() -> None:
    sections = build_moe_config_sections(
        args=_args(),
        model_config=_model(None),
        num_tokens_count=1,
        use_vllm_kernel=True,
        precision_str="BF16",
        torch_dtype="torch.bfloat16",
    )

    precision_section = dict(dict(sections)["Precision & Quantization"])

    assert precision_section["Grouped GEMM Precision"] == "BF16"
    assert precision_section["Quantization"] == "Disabled"
