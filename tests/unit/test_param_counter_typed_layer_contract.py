from __future__ import annotations

from frontier.config import ReplicaConfig
from frontier.model_architectures import LayerKind
from frontier.operators.spec import TensorParallelMode
from frontier.types import ClusterType
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
    assert counter.get_num_mlp_parameters_per_device() == 40_874_803_200


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
