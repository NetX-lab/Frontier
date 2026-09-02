"""RED coverage for the linear-op producer's typed operator contract."""

from __future__ import annotations

import json

import pytest

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.linear_op.linear_op_wrapper import LinearOpWrapper
from frontier.profiling.linear_op import main as linear_op_main
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan
from frontier.profiling.utils.replicated_ops import (
    deduplicate_tp1_rows,
    split_replicated_result,
)


def _operator_contracts(plan: dict) -> dict[str, dict]:
    contracts = plan["typed_operator_contracts"]
    assert isinstance(contracts, dict)
    return contracts


def test_step3_plan_binds_each_operator_to_its_profile_owned_contract() -> None:
    """Mixed Step3 rows must retain independent width and TP/EP identities."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
    plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[8],
        ffn_tp=[8],
        moe_tp=[1],
        is_moe=False,
    )

    contracts = _operator_contracts(plan)
    dense = contracts["mlp_up_proj"]
    shared = contracts["share_expert_up_proj"]
    routed = contracts["moe_grouped_gemm"]
    attention = contracts["attn_pre_proj"]

    assert dense["operator_family_id"] == "ffn"
    assert dense["layer_kind"] == "dense"
    assert dense["dimension_source"] == "dense_mlp_hidden_dim"
    assert dense["effective_ffn_width"] == 18432
    assert dense["tensor_parallel_mode"] == "attention_tp"
    assert dense["tensor_parallel_sizes"] == [8]
    assert dense["selected_tensor_parallel_size"] == 8
    assert dense["selected_expert_parallel_size"] is None
    assert dense["expert_parallel_mode"] == "off"

    assert shared["operator_family_id"] == "share_expert"
    assert shared["layer_kind"] == "shared"
    assert shared["dimension_source"] == "share_expert_dim"
    assert shared["effective_ffn_width"] == 5120
    assert shared["tensor_parallel_mode"] == "attention_tp"
    assert shared["tensor_parallel_sizes"] == [8]
    assert shared["selected_tensor_parallel_size"] == 8

    assert routed["operator_family_id"] == "moe"
    assert routed["layer_kind"] == "routed"
    assert routed["dimension_source"] == "routed_mlp_hidden_dim"
    assert routed["effective_ffn_width"] == 5120
    assert routed["tensor_parallel_mode"] == "moe_tp"
    assert routed["tensor_parallel_sizes"] == [1]
    assert routed["selected_tensor_parallel_size"] is None
    assert routed["selected_expert_parallel_size"] is None
    assert routed["expert_parallel_mode"] == "on"

    assert attention["operator_family_id"] == "dense_attention"
    assert attention["layer_kind"] is None
    assert attention["effective_ffn_width"] is None
    assert attention["tensor_parallel_mode"] == "attention_tp"
    assert attention["tensor_parallel_sizes"] == [8]
    assert attention["selected_tensor_parallel_size"] == 8

    # The producer contract is a stable CSV scalar and must round-trip without
    # relying on Python's repr format.
    json.dumps(contracts, sort_keys=True)


def test_typed_producer_preserves_an_empty_domain_without_cross_domain_fallback() -> None:
    """An omitted routed domain must remain empty instead of reusing FFN TP."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
    plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[8],
        ffn_tp=[8],
        moe_tp=[],
        is_moe=False,
    )

    routed = _operator_contracts(plan)["moe_grouped_gemm"]
    assert routed["tensor_parallel_sizes"] == []
    assert routed["selected_tensor_parallel_size"] is None


@pytest.mark.parametrize(
    "domain_name,domain_value",
    [
        ("attn_tp", [0]),
        ("ffn_tp", [-1]),
        ("moe_tp", [True]),
        ("attn_tp", ["8"]),
    ],
)
def test_typed_producer_rejects_invalid_tp_domain_values(
    domain_name: str, domain_value: list[object]
) -> None:
    """TP domains accept positive integers only, including for inactive families."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
    kwargs = {
        "model_config": config,
        "tp_size": 8,
        "attn_tp": [8],
        "ffn_tp": [8],
        "moe_tp": [1],
        "is_moe": False,
    }
    kwargs[domain_name] = domain_value

    with pytest.raises(ValueError, match="TP|tensor parallel|positive"):
        build_profiling_plan(**kwargs)


def test_typed_producer_rejects_width_that_is_not_divisible_by_selected_tp() -> None:
    """A selected typed width must be legal for its declared TP domain."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
    config.dense_mlp_hidden_dim = 18433

    with pytest.raises(ValueError, match="dense|width|divisible"):
        build_profiling_plan(
            model_config=config,
            tp_size=8,
            attn_tp=[8],
            ffn_tp=[8],
            moe_tp=[1],
            is_moe=False,
        )


def test_no_tensor_parallel_disables_non_replicated_typed_selection() -> None:
    """The metadata must reflect a no-TP config even when a TP value is passed."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
    config.no_tensor_parallel = True
    plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[8],
        ffn_tp=[8],
        moe_tp=[1],
        is_moe=False,
    )

    contracts = _operator_contracts(plan)
    assert contracts["mlp_up_proj"]["selected_tensor_parallel_size"] is None
    assert contracts["moe_grouped_gemm"]["selected_tensor_parallel_size"] is None
    assert contracts["input_layernorm"]["selected_tensor_parallel_size"] == 1


def test_pure_dense_and_pure_moe_producers_expose_only_active_layer_kinds() -> None:
    """Pure model plans must not inherit unrelated mixed-layer metadata."""

    dense_config = ModelConfig.from_model_name("llama3.1-8b")
    dense_plan = build_profiling_plan(
        model_config=dense_config,
        tp_size=2,
        attn_tp=[2],
        ffn_tp=[2],
        moe_tp=[],
        is_moe=False,
    )
    dense_kinds = {
        contract["layer_kind"]
        for contract in _operator_contracts(dense_plan).values()
        if contract["layer_kind"] is not None
    }
    assert dense_kinds == {"dense"}

    moe_config = ModelConfig.from_model_name("mixtral_8x7b_moe")
    moe_plan = build_profiling_plan(
        model_config=moe_config,
        tp_size=2,
        attn_tp=[2],
        ffn_tp=[2],
        moe_tp=[2],
        is_moe=True,
    )
    moe_kinds = {
        contract["layer_kind"]
        for contract in _operator_contracts(moe_plan).values()
        if contract["layer_kind"] is not None
    }
    assert moe_kinds == {"routed"}


def test_attention_only_plan_has_registry_metadata_without_ffn_fallback() -> None:
    """Attention-only collection keeps attention metadata independently typed."""

    config = ModelConfig.from_model_name("llama3.1-8b")
    plan = build_profiling_plan(
        model_config=config,
        tp_size=4,
        attn_tp=[4],
        ffn_tp=[],
        moe_tp=[],
        disable_replicated=True,
        is_moe=False,
    )

    contracts = _operator_contracts(plan)
    assert contracts["attn_pre_proj"]["tensor_parallel_mode"] == "attention_tp"
    assert contracts["attn_pre_proj"]["selected_tensor_parallel_size"] == 4
    assert not any(name.startswith("mlp_") for name in plan["enabled_ops"])


def test_replicated_split_keeps_matching_typed_operator_metadata() -> None:
    """Splitting a row must partition metadata with the timing operators."""

    result = {
        "num_tokens": 2,
        "model_arch": "llama",
        "num_tensor_parallel_workers": 8,
        "time_stats": {
            "input_layernorm": 1.0,
            "mlp_up_proj": 2.0,
        },
        "typed_operator_contracts": {
            "input_layernorm": {
                "operator_family_id": "memory",
                "tensor_parallel_mode": "replicated",
            },
            "mlp_up_proj": {
                "operator_family_id": "ffn",
                "layer_kind": "dense",
                "effective_ffn_width": 18432,
                "tensor_parallel_mode": "attention_tp",
            },
        },
    }

    sharded, replicated = split_replicated_result(
        result,
        {"input_layernorm"},
        unpadded_n_embd=7168,
        unpadded_n_expanded_embd=18432,
    )

    assert set(sharded["typed_operator_contracts"]) == {"mlp_up_proj"}
    assert set(replicated["typed_operator_contracts"]) == {"input_layernorm"}
    assert replicated["num_tensor_parallel_workers"] == 1


def test_tp1_dedup_keeps_first_typed_row_for_each_workload() -> None:
    """Replicated TP=1 rows remain deterministic after typed metadata is added."""

    first = {
        "num_tokens": 2,
        "model_arch": "llama",
        "num_tensor_parallel_workers": 1,
        "typed_operator_contracts": {"input_layernorm": {"tensor_parallel_mode": "replicated"}},
    }
    duplicate = {
        **first,
        "typed_operator_contracts": {"input_layernorm": {"tensor_parallel_mode": "replicated", "source": "duplicate"}},
    }
    sharded = {
        "num_tokens": 2,
        "model_arch": "llama",
        "num_tensor_parallel_workers": 8,
        "typed_operator_contracts": {"mlp_up_proj": {"tensor_parallel_mode": "attention_tp"}},
    }

    deduped = deduplicate_tp1_rows(
        [first, duplicate, sharded],
        tp1_key_fields=("num_tokens", "model_arch"),
    )
    assert deduped == [first, sharded]


def test_linear_wrapper_result_propagates_profile_owned_typed_contracts() -> None:
    """Wrapper output must carry typed operator metadata beside legacy scalars."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
    typed_contracts = {
        "mlp_up_proj": {
            "operator_family_id": "ffn",
            "layer_kind": "dense",
            "effective_ffn_width": 18432,
            "tensor_parallel_mode": "attention_tp",
        },
        "share_expert_up_proj": {
            "operator_family_id": "share_expert",
            "layer_kind": "shared",
            "effective_ffn_width": 5120,
            "tensor_parallel_mode": "attention_tp",
        },
    }
    wrapper = object.__new__(LinearOpWrapper)
    wrapper.model_config = config
    wrapper.num_tensor_parallel_workers = 8
    wrapper.profiling_plan = {
        "padded_n_embd": 7168,
        "padded_n_expanded_embd": 18432,
        "typed_operator_contracts": typed_contracts,
    }

    stats = wrapper._build_profile_result(  # pylint: disable=protected-access
        {"mlp_up_proj": {"mean": 1.0}},
        num_tokens=2,
    )

    assert stats["typed_operator_contracts"] == typed_contracts
    assert stats["typed_operator_contracts"] is not typed_contracts
    assert stats["n_expanded_embd"] == config.mlp_hidden_dim
    assert stats["padded_n_expanded_embd"] == 18432


def test_main_split_preserves_typed_compatibility_width() -> None:
    """Typed rows keep the plan-selected width when replicated timing is split."""

    config = type(
        "Config",
        (),
        {"embedding_dim": 7168, "mlp_hidden_dim": 5120},
    )()
    result = {
        "num_tensor_parallel_workers": 8,
        "time_stats": {"input_layernorm": 1.0, "mlp_up_proj": 2.0},
        "padded_n_expanded_embd": 18432,
        "typed_operator_contracts": {
            "mlp_up_proj": {
                "layer_kind": "dense",
                "effective_ffn_width": 18432,
            },
            "input_layernorm": {"layer_kind": None},
        },
    }

    _, replicated = linear_op_main._split_linear_op_result(  # pylint: disable=protected-access
        result,
        {"input_layernorm"},
        config,
    )

    assert replicated["padded_n_expanded_embd"] == 18432
    assert set(replicated["typed_operator_contracts"]) == {"input_layernorm"}


def test_main_split_keeps_legacy_width_fallback_without_typed_metadata() -> None:
    """Legacy result rows retain the historical unpadded scalar behavior."""

    config = type(
        "Config",
        (),
        {"embedding_dim": 7168, "mlp_hidden_dim": 5120},
    )()
    result = {
        "num_tensor_parallel_workers": 8,
        "time_stats": {"input_layernorm": 1.0},
        "padded_n_expanded_embd": 18432,
    }

    _, replicated = linear_op_main._split_linear_op_result(  # pylint: disable=protected-access
        result,
        {"input_layernorm"},
        config,
    )

    assert replicated["padded_n_expanded_embd"] == 5120
