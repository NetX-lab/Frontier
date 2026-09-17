from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.attention import (
    GATED_DELTA_NET_ATTENTION_FAMILY,
    bind_attention_family,
    bind_layer_attention,
    get_attention_runtime_kv_layout,
)
from frontier.attention.gdn.config import (
    GatedDeltaNetConfig,
    LayerAttentionSpec,
)
from frontier.attention.model_binding import resolve_layer_attention_specs
from frontier.attention.ops import AttentionMemoryLayout
from frontier.attention.profiling_mapping import (
    get_required_profiling_columns,
    validate_attention_profiling_dataframe,
)
from frontier.config.model_config import BaseModelConfig
from frontier.profiling.common.model_config import ModelConfig as ProfilingModelConfig
from frontier.profiling.gdn.inputs import GDNProfileInput
from frontier.types import ActivationType, MeasurementType, NormType


@pytest.mark.parametrize(
    "model_type,profile_id,aliases,expected",
    [
        ("qwen3_5_moe_text", None, (), (True, True, True)),
        (None, None, ("Qwen3_5MoeForCausalLM",), (True, True, True)),
        (None, "qwen3_5_moe", (), (False, True, True)),
        ("other", None, ("Qwen3_5MoeForCausalLM",), (False, False, False)),
        ("other", "qwen3_5_moe", (), (False, False, ValueError)),
        ("qwen3_5_moe_text", "generic", (), (True, True, True)),
        (" qwen3_5_moe_text ", None, (), (False, False, True)),
        (None, " qwen3_5_moe ", (), (False, False, True)),
        (" ", None, (" Qwen3_5MoeForCausalLM ",), (False, False, True)),
    ],
)
def test_qwen_identity_preserves_registry_and_topology_precedence(
    model_type, profile_id, aliases, expected
) -> None:
    from frontier.attention.gdn.config import is_qwen3_5_profile_config
    from frontier.model_architectures import (
        _matches_qwen3_5_moe,
        _requires_qwen3_5_profile_identity,
    )

    config = SimpleNamespace(
        model_type=model_type,
        model_architecture_profile=profile_id,
        architectures=aliases,
    )
    assert _matches_qwen3_5_moe(config) is expected[0]
    assert _requires_qwen3_5_profile_identity().predicate(config) is expected[1]
    if expected[2] is ValueError:
        with pytest.raises(ValueError, match="requires model_type"):
            is_qwen3_5_profile_config(config)
    else:
        assert is_qwen3_5_profile_config(config) is expected[2]


def _runtime_config(**overrides) -> BaseModelConfig:
    values = dict(
        num_layers=2,
        num_q_heads=8,
        num_kv_heads=2,
        embedding_dim=512,
        mlp_hidden_dim=1024,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=1024,
        model_type="unit_model",
    )
    values.update(overrides)
    return BaseModelConfig(**values)


def _qwen35_runtime_config(num_layers: int = 8, **overrides) -> BaseModelConfig:
    values = dict(
        num_layers=num_layers,
        num_q_heads=4,
        num_kv_heads=2,
        embedding_dim=256,
        mlp_hidden_dim=64,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=1024,
        model_type="qwen3_5_moe_text",
        model_architecture_profile="qwen3_5_moe",
        architectures=("Qwen3_5MoeForCausalLM",),
        is_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        full_attention_interval=4,
    )
    values.update(overrides)
    return BaseModelConfig(**values)


def test_qwen35_real_profile_has_pinned_layer_counts_and_ids() -> None:
    config = BaseModelConfig.create_from_name("Qwen3.8-2.4T-A95B-Quark-MXFP4")

    assert config.num_layers == 92
    assert config.get_num_gdn_layers() == 69
    assert config.get_num_full_attention_layers() == 23
    assert [spec.global_layer_id for spec in config.get_layer_attention_specs() if spec.is_full_attention] == list(range(3, 92, 4))
    assert config.get_layer_attention_spec(0) == LayerAttentionSpec(0, "gated_delta_net", "qwen3_5")
    assert config.get_layer_attention_spec(3) == LayerAttentionSpec(3, "dense_attention", "gqa")


def test_qwen35_eight_layer_fixture_has_six_gdn_and_two_dense_layers() -> None:
    config = _qwen35_runtime_config()

    assert config.get_num_gdn_layers() == 6
    assert config.get_num_full_attention_layers() == 2
    assert [spec.global_layer_id for spec in config.get_layer_attention_specs() if spec.is_full_attention] == [3, 7]


def test_hybrid_whole_model_binder_fails_and_layer_binder_is_explicit() -> None:
    config = _qwen35_runtime_config()

    with pytest.raises(ValueError, match="explicit global layer id"):
        bind_attention_family(config)
    assert bind_layer_attention(config, 0).is_gdn
    assert bind_layer_attention(config, 3).is_full_attention
    with pytest.raises(ValueError, match="out of range"):
        bind_layer_attention(config, config.num_layers)


def test_dense_and_mla_legacy_binders_remain_homogeneous() -> None:
    dense = _runtime_config()
    assert bind_attention_family(dense).family_id == "dense_attention"

    mla = _runtime_config(
        num_q_heads=16,
        num_kv_heads=1,
        use_mla=True,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
    )
    assert bind_attention_family(mla).family_id == "latent_mla_attention"


def test_qwen3_next_linear_metadata_is_not_reinterpreted_as_gdn() -> None:
    for model_name in (
        "qwen3-next-80b-a3b-instruct-reduced-l2",
        "qwen3-next-80b-a3b-instruct-reduced-l20",
    ):
        config = BaseModelConfig.create_from_name(model_name)
        assert config.get_num_gdn_layers() == 0
        assert config.get_num_full_attention_layers() == config.num_layers
        assert bind_attention_family(config).family_id == "dense_attention"
        assert all(spec.family_id == "dense_attention" for spec in config.get_layer_attention_specs())


def test_non_qwen_linear_attention_literal_is_not_a_gdn_activation() -> None:
    config = _runtime_config(
        model_type="qwen3_next",
        layer_types=("linear_attention", "full_attention"),
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    )

    assert resolve_layer_attention_specs(config) == (
        LayerAttentionSpec(0, "dense_attention", "gqa"),
        LayerAttentionSpec(1, "dense_attention", "gqa"),
    )


def test_gdn_family_uses_fixed_state_without_runtime_kv_helpers() -> None:
    family = GATED_DELTA_NET_ATTENTION_FAMILY
    assert family.memory_layout is AttentionMemoryLayout.FIXED_STATE
    assert family.requires_runtime_kv_helpers is False
    assert family.kv_factor is None
    assert family.profiling_order == (
        "gdn_input_projections",
        "gdn_core_prefill",
        "gdn_core_decode",
        "gdn_output_projection",
    )
    with pytest.raises(ValueError, match="does not declare"):
        family.resolve_runtime_num_kv_heads(_qwen35_runtime_config())
    with pytest.raises(NotImplementedError, match="does not declare a runtime kv_factor"):
        get_attention_runtime_kv_layout(
            family,
            runtime_num_kv_heads_per_worker=1,
            runtime_head_size=128,
        )


def test_gdn_state_layout_is_fixed_and_tp_partitioned() -> None:
    config = GatedDeltaNetConfig(
        conv_kernel_size=4,
        key_head_dim=32,
        value_head_dim=32,
        num_key_heads=2,
        num_value_heads=4,
    )
    layout = config.get_state_layout(tensor_parallel_size=2)

    assert layout.conv_state_shape == (3, 128)
    assert layout.recurrent_state_shape == (2, 32, 32)
    assert layout.total_bytes == 3 * 128 * 2 + 2 * 32 * 32 * 4


def test_gdn_profile_input_rejects_same_batch_mixed_phase() -> None:
    mixed = GDNProfileInput.mixed(
        decode_batch_size=1,
        decode_context_len=128,
        prefill_seq_len=16,
    )
    assert mixed.phase == "mixed"
    with pytest.raises(ValueError, match="same-batch GDN prefill/decode"):
        mixed.require_supported_phase()
    assert GDNProfileInput.prefill(seq_len=16).phase == "prefill"
    assert GDNProfileInput.decode(batch_size=2, context_len=128).phase == "decode"


def test_gdn_profiling_fixture_matches_device_event_schema() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixture_path = repo_root / "tests" / "fixtures" / "pr31_hybrid" / "gdn.csv"
    frame = pd.read_csv(fixture_path)

    validate_attention_profiling_dataframe(
        frame,
        GATED_DELTA_NET_ATTENTION_FAMILY,
        measurement_type=MeasurementType.DEVICE_EVENT,
    )
    assert tuple(frame["measurement_type"].unique()) == ("DEVICE_EVENT",)
    assert set(get_required_profiling_columns(GATED_DELTA_NET_ATTENTION_FAMILY)).issubset(frame.columns)


def test_qwen35_profiling_config_preserves_hybrid_schedule() -> None:
    config = ProfilingModelConfig.from_model_name("Qwen3.8-2.4T-A95B-Quark-MXFP4")

    assert config.get_num_gdn_layers() == 69
    assert config.get_num_full_attention_layers() == 23
    assert config.get_layer_attention_spec(0).family_id == "gated_delta_net"
    assert config.get_layer_attention_spec(3).family_id == "dense_attention"
    assert config.get_gdn_config().conv_kernel_size == 4


def test_gdn_import_path_is_cpu_safe() -> None:
    # Importing semantic contracts must not initialize optional GPU stacks.
    import frontier.attention.gdn  # noqa: F401
    import frontier.profiling.gdn.inputs  # noqa: F401
