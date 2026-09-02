"""Regression coverage for legacy positional model-config constructors."""

from __future__ import annotations

from frontier.config.model_config import BaseModelConfig
from frontier.profiling.common.model_config import ModelConfig
from frontier.types import ActivationType, NormType


def _required_values() -> tuple[object, ...]:
    return (
        2,
        4,
        4,
        16,
        32,
        128,
        True,
        False,
        False,
        ActivationType.SILU,
        NormType.RMS_NORM,
        True,
        256,
    )


def test_profiling_model_config_keeps_legacy_optional_positional_order() -> None:
    config = ModelConfig(
        "legacy",
        *_required_values(),
        False,  # is_neox_style
        10000,  # rope_theta
        None,  # rope_scaling
        1.0,  # partial_rotary_factor
        False,  # no_tensor_parallel
        False,  # is_moe
        0,  # num_experts
        0,  # num_experts_per_tok
        None,  # moe_layers_enum
        True,  # use_qk_norm
    )

    assert config.is_neox_style is False
    assert config.rope_theta == 10000
    assert config.use_qk_norm is True
    assert config.dense_mlp_hidden_dim == 32


def test_runtime_model_config_keeps_legacy_optional_positional_order() -> None:
    config = BaseModelConfig(
        *_required_values(),
        True,  # use_qk_norm
        False,  # attn_output_gate
        False,  # is_neox_style
        10000.0,  # rope_theta
        None,  # rope_scaling
        1.0,  # partial_rotary_factor
        False,  # no_tensor_parallel
        False,  # is_moe
        0,  # num_experts
        0,  # num_experts_per_tok
        None,  # moe_layers_enum
        "llama",  # model_type
    )

    assert config.use_qk_norm is True
    assert config.attn_output_gate is False
    assert config.is_neox_style is False
    assert config.model_type == "llama"
    assert config.dense_mlp_hidden_dim == 32


def test_typed_width_properties_preserve_pure_moe_legacy_intermediate_size() -> None:
    config = BaseModelConfig(
        *_required_values(),
        use_qk_norm=True,
        is_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        moe_layers_enum=None,
        model_type="generic_moe",
    )

    assert config.intermediate_size == config.mlp_hidden_dim == 32
    assert config.moe_intermediate_size == config.mlp_hidden_dim == 32


def test_profiling_typed_width_properties_preserve_pure_moe_legacy_intermediate_size() -> None:
    config = ModelConfig(
        "generic_moe",
        *_required_values(),
        is_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        moe_layers_enum=None,
    )

    assert config.intermediate_size == config.mlp_hidden_dim == 32
    assert config.moe_intermediate_size == config.mlp_hidden_dim == 32
