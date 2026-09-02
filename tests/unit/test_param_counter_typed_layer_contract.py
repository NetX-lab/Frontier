from __future__ import annotations

from frontier.config import ReplicaConfig
from frontier.types import ClusterType
from frontier.utils.param_counter import ParamCounter


def test_step3_param_counter_uses_profile_owned_mixed_layer_widths() -> None:
    """Parameter accounting follows the profile-owned dense/routed widths."""

    replica_config = ReplicaConfig(
        model_name="step-moe-noquant",
        device="h200",
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
        num_pipeline_stages=1,
    )
    counter = ParamCounter(replica_config, ClusterType.MONOLITHIC)

    profile = replica_config.model_config.get_model_architecture_profile()
    assert replica_config.model_config.dense_mlp_hidden_dim == 18_432
    assert replica_config.model_config.routed_mlp_hidden_dim == 5_120
    assert profile.resolve_layer_contract(
        replica_config.model_config,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    ).effective_ffn_width == 18_432

    assert counter._get_dense_mlp_params_per_layer(8) == 49_545_216
    assert counter._get_routed_moe_params_per_layer(1) == 660_946_944
    assert counter._get_share_expert_params_per_layer(8) == 13_762_560
    assert counter.get_num_mlp_parameters_per_device() == 38_031_458_304


def test_param_counter_assigns_each_ffn_contract_only_to_its_layer_kind() -> None:
    """Mixed models count dense boundaries separately from routed layers."""

    replica_config = ReplicaConfig(
        model_name="step-moe-noquant",
        device="h200",
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
        num_pipeline_stages=1,
    )
    counter = ParamCounter(replica_config, ClusterType.MONOLITHIC)

    assert counter.get_num_mlp_parameters_per_device() == 38_031_458_304


def test_param_counter_pure_moe_does_not_materialize_dense_contract() -> None:
    """Pure-MoE layers own routed and shared experts, without dense MLP weights."""

    replica_config = ReplicaConfig(
        model_name="Step2Mini-tiny",
        device="h800",
        attn_tensor_parallel_size=2,
        moe_tensor_parallel_size=2,
        moe_expert_parallel_size=2,
        num_pipeline_stages=1,
    )
    counter = ParamCounter(replica_config, ClusterType.MONOLITHIC)

    assert counter.get_num_mlp_parameters_per_device() == 151_126_016
    assert counter.get_num_parameters_per_device() == 188_874_752
