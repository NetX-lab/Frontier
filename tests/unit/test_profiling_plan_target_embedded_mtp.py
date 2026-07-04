from __future__ import annotations

from types import SimpleNamespace

from frontier.model_architectures import get_model_architecture_profile
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan


def _model_config(**overrides):
    cfg = SimpleNamespace(
        model_type="unit_qwen3_next_like",
        model_architecture_profile="step3_text",
        embedding_dim=128,
        mlp_hidden_dim=256,
        num_q_heads=8,
        num_kv_heads=4,
        no_tensor_parallel=False,
        is_moe=True,
        post_attn_norm=True,
        share_expert_dim=64,
    )
    for name, value in overrides.items():
        setattr(cfg, name, value)
    cfg.supports_share_expert = lambda: get_model_architecture_profile(
        cfg
    ).supports_share_expert(cfg)
    return cfg


def test_target_embedded_mtp_profiling_plan_uses_registry_ops() -> None:
    plan = build_profiling_plan(
        _model_config(),
        tp_size=2,
        attn_tp=[2],
        ffn_tp=[2],
        is_moe=True,
        include_target_embedded_mtp=True,
    )

    import frontier.spec_decode.mtp_registry as registry

    for op_name in registry.get_target_embedded_mtp_linear_ops():
        assert op_name in plan["enabled_ops"]
    for op_name in registry.get_target_embedded_mtp_same_tp_linear_ops():
        assert op_name in plan["enabled_ops"]


def test_target_embedded_mtp_post_attention_norm_respects_model_capability() -> None:
    plan = build_profiling_plan(
        _model_config(post_attn_norm=False),
        tp_size=2,
        attn_tp=[2],
        ffn_tp=[2],
        is_moe=True,
        include_target_embedded_mtp=True,
    )

    assert "post_attention_layernorm" not in plan["enabled_ops"]
