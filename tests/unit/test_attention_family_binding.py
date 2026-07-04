from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.attention.model_binding import (
    AttentionFamilyBinding,
    bind_attention_family,
)
from frontier.config.model_config import BaseModelConfig
from frontier.profiling.common.model_config import ModelConfig as ProfilingModelConfig
from frontier.types import ActivationType, NormType


def _base_model_config(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    is_moe: bool = False,
    use_mla: bool = False,
    model_type: str | None = "unit_model",
    **extra,
) -> BaseModelConfig:
    return BaseModelConfig(
        num_layers=2,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        embedding_dim=num_q_heads * 128,
        mlp_hidden_dim=4096,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=32000,
        is_moe=is_moe,
        num_experts=8 if is_moe else 0,
        num_experts_per_tok=2 if is_moe else 0,
        model_type=model_type,
        use_mla=use_mla,
        **extra,
    )


def _profiling_model_config(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    is_moe: bool = False,
    use_mla: bool = False,
    model_type: str | None = "unit_model",
    **extra,
) -> ProfilingModelConfig:
    return ProfilingModelConfig(
        name="unit-model",
        num_layers=2,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        embedding_dim=num_q_heads * 128,
        mlp_hidden_dim=4096,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=32000,
        is_moe=is_moe,
        num_experts=8 if is_moe else 0,
        num_experts_per_tok=2 if is_moe else 0,
        model_type=model_type,
        use_mla=use_mla,
        dtype="bfloat16",
        **extra,
    )


def _mla_kwargs() -> dict[str, int]:
    return {
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "qk_head_dim": 192,
        "v_head_dim": 128,
    }


@pytest.mark.parametrize(
    ("num_q_heads", "num_kv_heads", "expected_variant"),
    [
        (32, 8, "gqa"),
        (32, 32, "mha"),
        (32, 1, "mqa"),
    ],
)
def test_dense_attention_binding_variants_are_head_topology_only(
    num_q_heads: int,
    num_kv_heads: int,
    expected_variant: str,
) -> None:
    dense_ffn_binding = bind_attention_family(
        _base_model_config(num_q_heads=num_q_heads, num_kv_heads=num_kv_heads)
    )
    moe_ffn_binding = bind_attention_family(
        _base_model_config(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            is_moe=True,
        )
    )

    assert dense_ffn_binding == AttentionFamilyBinding(
        family_id="dense_attention",
        variant_id=expected_variant,
        frozen=False,
        reason=f"num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}",
    )
    assert moe_ffn_binding.family_id == "dense_attention"
    assert moe_ffn_binding.variant_id == expected_variant


def test_mla_binding_uses_latent_family_for_runtime_cache_semantics() -> None:
    binding = bind_attention_family(
        _base_model_config(
            num_q_heads=128,
            num_kv_heads=128,
            use_mla=True,
            model_type="deepseek_v2",
            **_mla_kwargs(),
        )
    )

    assert binding.family_id == "latent_mla_attention"
    assert binding.variant_id == "mla"
    assert binding.frozen is False
    assert "use_mla=True" in binding.reason


def test_step3_mfa_binding_uses_dense_mqa_family_with_explicit_reason() -> None:
    binding = bind_attention_family(
        SimpleNamespace(
            num_q_heads=64,
            num_kv_heads=1,
            model_type="step3_text",
            use_mfa=True,
            use_mla=False,
            share_q_dim=2048,
            head_dim=256,
        )
    )

    assert binding.family_id == "dense_attention"
    assert binding.variant_id == "mqa"
    assert binding.frozen is False
    assert "use_mfa=True" in binding.reason


def test_step3_mfa_binding_rejects_mla_and_mfa_together() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        bind_attention_family(
            SimpleNamespace(
                num_q_heads=64,
                num_kv_heads=1,
                model_type="step3_text",
                use_mfa=True,
                use_mla=True,
                share_q_dim=2048,
                head_dim=256,
                **_mla_kwargs(),
            )
        )


def test_step3_mfa_binding_requires_single_kv_head() -> None:
    with pytest.raises(ValueError, match="use_mfa=True requires num_kv_heads=1"):
        bind_attention_family(
            SimpleNamespace(
                num_q_heads=64,
                num_kv_heads=8,
                model_type="step3_text",
                use_mfa=True,
                use_mla=False,
                share_q_dim=2048,
                head_dim=256,
            )
        )


def test_profiling_model_config_uses_same_binding_rules_as_runtime_config() -> None:
    runtime_binding = bind_attention_family(
        _base_model_config(num_q_heads=32, num_kv_heads=1)
    )
    profiling_binding = bind_attention_family(
        _profiling_model_config(num_q_heads=32, num_kv_heads=1)
    )

    assert runtime_binding.family_id == profiling_binding.family_id
    assert runtime_binding.variant_id == profiling_binding.variant_id == "mqa"


def test_frozen_dsa_marker_fails_fast_instead_of_falling_back_to_dense_gqa() -> None:
    model_config = _base_model_config(
        num_q_heads=128,
        num_kv_heads=128,
        model_type="deepseek_v32",
    )
    model_config.dsa_topk = 8

    binding = bind_attention_family(model_config)

    assert binding.family_id == "dsa_attention"
    assert binding.variant_id == "dsa"
    assert binding.frozen is True
    with pytest.raises(NotImplementedError, match="DSA attention is frozen"):
        binding.require_enabled_for_execution()


def test_unknown_exotic_attention_fields_fail_fast_without_dense_fallback() -> None:
    model_config = _base_model_config(num_q_heads=32, num_kv_heads=8)
    model_config.sliding_window_pattern = "unit-test-exotic"

    with pytest.raises(ValueError, match="Unrecognized exotic attention fields"):
        bind_attention_family(model_config)


def test_invalid_head_topology_raises_clear_configuration_error() -> None:
    model_config = _base_model_config(num_q_heads=8, num_kv_heads=16)

    with pytest.raises(ValueError, match="Unsupported attention head topology"):
        bind_attention_family(model_config)
