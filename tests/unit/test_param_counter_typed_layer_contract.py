from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.config import ReplicaConfig
from frontier.config.model_config import BaseModelConfig, ModelArch
from frontier.model_architectures import LayerKind
from frontier.operators.spec import TensorParallelMode
from frontier.types import ActivationType, ClusterType, NormType
from frontier.utils.param_counter import ParamCounter


def test_step3_param_counter_uses_profile_owned_mixed_layer_widths() -> None:
    replica_config = ReplicaConfig(
        model_name="step3-moe-noquant",
        device="h200",
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
        num_pipeline_stages=1,
    )
    counter = ParamCounter(replica_config, ClusterType.MONOLITHIC)

    # Step3 declares dense=18432 and routed/shared=5120.  The legacy
    # mlp_hidden_dim field intentionally remains 5120 for routed compatibility.
    assert counter._get_dense_mlp_params_per_layer(tensor_parallel_size=8) == 49_545_216
    assert counter._get_routed_moe_params_per_layer(tensor_parallel_size=1) == 660_946_944
    assert counter._get_share_expert_params_per_layer(tensor_parallel_size=8) == 13_762_560
    # Five dense boundary layers own only dense weights; the 56 routed layers
    # own routed experts plus the shared expert.
    assert counter.get_num_mlp_parameters_per_device() == 38_031_458_304


def test_pure_moe_parameter_basis_uses_routed_width_and_moe_tp() -> None:
    """The public pure-MoE path must not use the dense helper as a routed alias."""

    replica_config = ReplicaConfig(
        model_name="qwen3-next-80b-a3b-instruct-reduced-l2",
        device="h800",
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
        num_pipeline_stages=1,
    )
    counter = ParamCounter(replica_config, ClusterType.MONOLITHIC)

    # Qwen3-Next is pure MoE: the routed basis belongs to the MoE TP domain,
    # while the private dense helper has no valid dense layer to resolve.
    with pytest.raises(ValueError, match="dense layer contract is inactive"):
        counter._get_dense_mlp_params_per_layer(tensor_parallel_size=8)
    assert counter.get_num_mlp_parameters_per_device() == 404_750_336


def test_param_counter_dispatches_tp_by_enum_identity(monkeypatch) -> None:
    """A TP domain remains semantic when an enum value label changes."""

    replica_config = ReplicaConfig(
        model_name="step3-moe-noquant",
        device="h200",
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
        num_pipeline_stages=1,
    )
    counter = ParamCounter(replica_config, ClusterType.MONOLITHIC)

    # The resolver contract is represented by the enum member, not by its
    # display value.  A renamed label must not route dense Step3 work to the
    # stale attention TP configured on the replica.
    monkeypatch.setattr(TensorParallelMode.ATTENTION_TP, "_value_", "renamed")
    try:
        contract = counter._resolve_profile_layer_contract(
            layer_kind=LayerKind.DENSE,
            tensor_parallel_size=8,
        )
    finally:
        monkeypatch.setattr(TensorParallelMode.ATTENTION_TP, "_value_", "attention_tp")

    assert contract is not None
    assert contract.tensor_parallel_size == 8


def test_param_counter_rejects_nonuniform_moe_layer_map_with_pp() -> None:
    """A single per-stage count cannot represent an uneven MoE layer map."""

    model_config = BaseModelConfig(
        num_layers=8,
        num_q_heads=8,
        num_kv_heads=4,
        embedding_dim=128,
        mlp_hidden_dim=256,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=32000,
        model_type="unit_custom_model",
        model_arch=ModelArch.GENERIC,
        is_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        share_expert_dim=64,
        # Stage 0 owns layers 0..3 (four MoE layers); stage 1 owns only layer 4.
        moe_layers_enum="0,1,2,3,4",
    )
    replica_config = SimpleNamespace(
        model_config=model_config,
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        num_pipeline_stages=2,
    )
    counter = ParamCounter(replica_config, ClusterType.MONOLITHIC)

    with pytest.raises(ValueError, match="non-uniform.*MoE.*pipeline"):
        counter._get_num_moe_layers_per_pipeline_stage()


def test_ffn_parameter_count_rejects_invalid_width() -> None:
    """An invalid FFN width must fail before producing a misleading zero count."""

    with pytest.raises(ValueError, match="width.*positive integer"):
        ParamCounter._get_ffn_weight_params(
            width=0,
            embedding_dim=128,
            use_gated_mlp=True,
            tensor_parallel_size=1,
        )


def test_ffn_parameter_count_rejects_invalid_embedding_dim() -> None:
    """An invalid model hidden size must not be converted into zero parameters."""

    with pytest.raises(ValueError, match="embedding_dim.*positive integer"):
        ParamCounter._get_ffn_weight_params(
            width=128,
            embedding_dim=0,
            use_gated_mlp=True,
            tensor_parallel_size=1,
        )


def test_ffn_parameter_count_rejects_invalid_tensor_parallel_size() -> None:
    """A non-positive or non-integral TP domain must fail explicitly."""

    with pytest.raises(ValueError, match="tensor_parallel_size.*positive integer"):
        ParamCounter._get_ffn_weight_params(
            width=128,
            embedding_dim=128,
            use_gated_mlp=True,
            tensor_parallel_size=0,
        )
