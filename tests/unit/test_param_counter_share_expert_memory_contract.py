from __future__ import annotations

import pytest

from frontier.config import ReplicaConfig
from frontier.operators.binding import build_operator_manifest
from frontier.types import ClusterType
from frontier.utils.param_counter import ParamCounter


def test_qwen3_next_param_counter_excludes_shared_expert_from_system_metrics_memory_contract() -> None:
    replica_config = ReplicaConfig(
        model_name="qwen3-next-80b-a3b-instruct-reduced-l2",
        device="h800",
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        num_pipeline_stages=1,
    )
    counter = ParamCounter(replica_config, ClusterType.MONOLITHIC)

    assert replica_config.model_config.model_type == "qwen3_next"
    assert replica_config.model_config.get_model_architecture_profile().profile_id == "generic"
    assert replica_config.model_config.supports_share_expert()

    assert counter._get_share_expert_params_per_layer(tensor_parallel_size=1) == 0
    assert counter.get_num_mlp_parameters_per_device() == 3_229_614_080
    assert counter.get_num_parameters_per_device() == 3_267_362_816
    assert 2 * counter.get_num_parameters_per_device() == 6_534_725_632


@pytest.mark.parametrize(
    (
        "model_name",
        "attn_tp",
        "moe_tp",
        "moe_ep",
        "expected_profile_id",
        "expected_share_expert_params_per_layer",
        "expected_total_parameters",
        "expected_total_memory_bytes",
    ),
    [
        (
            "Step2Mini-tiny",
            2,
            2,
            2,
            "step2_mini",
            6_291_456,
            214_040_576,
            428_081_152,
        ),
        (
            "step-moe-noquant-small",
            4,
            4,
            4,
            "step3_text",
            27_525_120,
            8_104_370_176,
            16_208_740_352,
        ),
    ],
)
def test_step_profiles_param_counter_still_count_shared_expert_memory(
    model_name: str,
    attn_tp: int,
    moe_tp: int,
    moe_ep: int,
    expected_profile_id: str,
    expected_share_expert_params_per_layer: int,
    expected_total_parameters: int,
    expected_total_memory_bytes: int,
) -> None:
    replica_config = ReplicaConfig(
        model_name=model_name,
        device="h800",
        attn_tensor_parallel_size=attn_tp,
        moe_tensor_parallel_size=moe_tp,
        moe_expert_parallel_size=moe_ep,
        num_pipeline_stages=1,
    )
    counter = ParamCounter(replica_config, ClusterType.MONOLITHIC)

    assert replica_config.model_config.get_model_architecture_profile().profile_id == expected_profile_id
    assert replica_config.model_config.supports_share_expert()
    assert (
        counter._get_share_expert_params_per_layer(counter._get_attn_tp_size())
        == expected_share_expert_params_per_layer
    )
    assert counter.get_num_parameters_per_device() == expected_total_parameters
    assert 2 * counter.get_num_parameters_per_device() == expected_total_memory_bytes


def test_qwen3_next_operator_manifest_still_exposes_shared_expert_ops() -> None:
    replica_config = ReplicaConfig(
        model_name="qwen3-next-80b-a3b-instruct-reduced-l2",
        device="h800",
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        num_pipeline_stages=1,
    )

    manifest = build_operator_manifest(replica_config.model_config)

    assert replica_config.model_config.model_type == "qwen3_next"
    assert replica_config.model_config.get_model_architecture_profile().profile_id == "generic"
    assert replica_config.model_config.supports_share_expert()
    assert [binding.family_id for binding in manifest.family_bindings] == [
        "dense_attention",
        "memory",
        "moe",
        "share_expert",
    ]
    assert {
        "share_expert_up_proj",
        "share_expert_act",
        "share_expert_down_proj",
    }.issubset({operator.name for operator in manifest.operators()})
