"""Producer-level regressions for profile-owned typed FFN contracts."""

from __future__ import annotations

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan


def _contracts_by_kind(plan: dict) -> dict[str, dict]:
    return {
        str(contract["layer_kind"]): contract
        for contract in plan["typed_layer_contracts"]
    }


def test_step3_plan_keeps_dense_routed_and_shared_domains_typed() -> None:
    """Step3 must expose independent widths and semantic TP domains."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
    plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[8],
        ffn_tp=[1],
        is_moe=False,
    )

    contracts = _contracts_by_kind(plan)

    assert contracts["dense"]["effective_ffn_width"] == 18432
    assert contracts["dense"]["tensor_parallel_mode"] == "attention_tp"
    assert contracts["dense"]["tensor_parallel_sizes"] == [8]
    assert contracts["dense"]["selected_tensor_parallel_size"] == 8

    assert contracts["routed"]["effective_ffn_width"] == 5120
    assert contracts["routed"]["tensor_parallel_mode"] == "moe_tp"
    assert contracts["routed"]["expert_parallel_mode"] == "on"
    assert contracts["routed"]["tensor_parallel_sizes"] == [1]
    assert contracts["routed"]["selected_tensor_parallel_size"] is None

    assert contracts["shared"]["effective_ffn_width"] == 5120
    assert contracts["shared"]["tensor_parallel_mode"] == "attention_tp"
    assert contracts["shared"]["tensor_parallel_sizes"] == [8]
    assert contracts["shared"]["selected_tensor_parallel_size"] == 8


def test_step3_plan_uses_independent_moe_tp_domain() -> None:
    """Mixed plans must not advertise routed TP sizes from the dense domain."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
    plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[8],
        ffn_tp=[8],
        moe_tp=[1],
        is_moe=True,
    )

    contracts = _contracts_by_kind(plan)

    assert contracts["dense"]["tensor_parallel_sizes"] == [8]
    assert contracts["dense"]["selected_tensor_parallel_size"] == 8
    assert contracts["shared"]["tensor_parallel_sizes"] == [8]
    assert contracts["shared"]["selected_tensor_parallel_size"] == 8
    assert contracts["routed"]["tensor_parallel_sizes"] == [1]
    assert contracts["routed"]["selected_tensor_parallel_size"] is None

    # The compatibility scalar follows the selected dense contract, while the
    # typed list remains the source of truth for each operator family.
    assert plan["padded_n_expanded_embd"] == 18432


def test_step3_mixed_moe_plan_keeps_dense_mlp_operators_enabled() -> None:
    """A mixed MoE plan must profile its dense boundary MLP operators too."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
    plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[8],
        ffn_tp=[8],
        moe_tp=[1],
        is_moe=True,
    )

    assert {"mlp_up_proj", "mlp_down_proj", "mlp_act"}.issubset(
        set(plan["enabled_ops"])
    )


def test_pure_dense_and_pure_moe_plans_keep_legacy_active_domains() -> None:
    """Pure models expose only their applicable typed domains."""

    dense_config = ModelConfig.from_model_name("llama3.1-8b")
    dense_plan = build_profiling_plan(
        model_config=dense_config,
        tp_size=2,
        attn_tp=[2],
        ffn_tp=[2],
        is_moe=False,
    )
    assert set(_contracts_by_kind(dense_plan)) == {"dense"}
    assert dense_plan["typed_layer_contracts"][0]["effective_ffn_width"] == 14336

    moe_config = ModelConfig.from_model_name("mixtral_8x7b_moe")
    moe_plan = build_profiling_plan(
        model_config=moe_config,
        tp_size=2,
        attn_tp=[2],
        ffn_tp=[2],
        is_moe=True,
    )
    assert set(_contracts_by_kind(moe_plan)) == {"routed"}
    assert moe_plan["typed_layer_contracts"][0]["effective_ffn_width"] == 14336


def test_attention_only_plan_does_not_select_ffn_domains() -> None:
    """An attention-only role must not materialize any FFN family metadata."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
    plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[8],
        ffn_tp=[8],
        moe_tp=[1],
        include_ffn=False,
    )

    assert plan["attn_enabled"] is True
    assert plan["ffn_enabled"] is False
    assert plan["ffn_sharded_enabled"] is False
    assert plan["typed_layer_contracts"] == []
    assert plan["typed_operator_contracts"]
    assert all(
        metadata["layer_kind"] is None
        for metadata in plan["typed_operator_contracts"].values()
    )
    assert "attn_pre_proj_wq" in plan["enabled_ops"]
    assert not {
        "mlp_up_proj",
        "mlp_act",
        "mlp_down_proj",
        "share_expert_up_proj",
        "moe_grouped_gemm",
    }.intersection(plan["enabled_ops"])
