"""Constructor-owned configuration fields and registry metadata contracts."""

from unittest.mock import Mock, call

import pytest

from frontier.attention.gdn import guards
from frontier.config.config import (
    ClusterConfig,
    MetricsConfig,
    OrcaSchedulerConfig,
    ReplicaConfig,
    SimulationConfig,
    SpeculativeDecodingConfig,
    VllmV1SchedulerConfig,
)
from frontier.config.model_config import BaseModelConfig
from frontier.operators.typed_contracts import validate_typed_operator_metadata


@pytest.mark.parametrize(
    "role", [None, "prefill", "decode", "decode_attn", "decode_ffn"]
)
@pytest.mark.parametrize(
    "network_device,capacity", [("a100_pairwise_nvlink", 4), ("a100_dgx", 8)]
)
@pytest.mark.parametrize("tp_size", [4, 8])
def test_replica_gdn_guard_uses_initialized_topology(
    monkeypatch, role, network_device, capacity, tp_size
):
    guard = Mock(wraps=guards.validate_gdn_runtime_support)
    monkeypatch.setattr(guards, "validate_gdn_runtime_support", guard)
    kwargs = dict(
        model_name="Qwen3.8-2.4T-A95B-Quark-MXFP4",
        cluster_prefix=role,
        network_device=network_device,
        attn_tensor_parallel_size=tp_size,
        moe_tensor_parallel_size=tp_size,
    )
    if tp_size > capacity:
        with pytest.raises(
            ValueError, match="^GDN cross-node execution is unsupported$"
        ):
            ReplicaConfig(**kwargs)
    else:
        replica = ReplicaConfig(**kwargs)
        assert replica.world_size == tp_size
        assert replica.node_config.num_devices_per_node == capacity
        assert replica.speculative_decoding_config.enabled is False
    assert guard.call_count == 1
    assert guard.call_args.kwargs == dict(
        speculative_enabled=False,
        num_pipeline_stages=1,
        moe_expert_parallel_size=1,
        attn_dp=1,
        cross_node=tp_size > capacity,
    )


@pytest.mark.parametrize(
    "sys_arch", ["co-location", "pd-disaggregation", "pd-af-disaggregation"]
)
@pytest.mark.parametrize("prefix_cache", [None, False, True])
def test_simulation_guard_keeps_optional_roles_and_scheduler_capabilities(
    monkeypatch, tmp_path, sys_arch, prefix_cache
):
    primary = ReplicaConfig(model_name="llama2_7b_dense_example")
    cluster = ClusterConfig(
        replica_config=primary,
        replica_scheduler_config=(
            OrcaSchedulerConfig()
            if prefix_cache is None
            else VllmV1SchedulerConfig(enable_prefix_caching=prefix_cache)
        ),
    )
    simulation = SimulationConfig(
        cluster_config=cluster,
        metrics_config=MetricsConfig(output_dir=str(tmp_path), write_metrics=False),
    )
    speculative = ReplicaConfig(
        model_name="llama2_7b_dense_example",
        num_pipeline_stages=2,
        attn_dp=2,
        speculative_decoding_config=SpeculativeDecodingConfig(enabled=True),
    )
    ffn = ReplicaConfig(
        model_name="llama2_7b_dense_example", cluster_prefix="decode_ffn"
    )
    cluster.prefill_replica_config = primary
    cluster.decode_replica_config = speculative
    cluster.decode_attn_replica_config = None
    cluster.decode_ffn_replica_config = ffn
    simulation.sys_arch = sys_arch
    guard = Mock(wraps=guards.validate_gdn_runtime_support)
    monkeypatch.setattr(guards, "validate_gdn_runtime_support", guard)

    simulation._validate_gdn_runtime_guards()

    assert guard.call_args_list == [
        call(
            replica.model_config,
            prefix_cache_enabled=bool(prefix_cache),
            pd_enabled=sys_arch != "co-location",
            speculative_enabled=replica is speculative,
            num_pipeline_stages=replica.num_pipeline_stages,
            moe_expert_parallel_size=replica.moe_expert_parallel_size,
            attn_dp=replica.attn_dp,
        )
        for replica in (primary, speculative, ffn)
    ]


def test_typed_metadata_retains_external_profile_validation():
    model = BaseModelConfig.create_from_name("step-moe-noquant")
    metadata = model.get_model_architecture_profile().resolve_layer_contract(
        model, layer_id=0, operator_name="mlp_up_proj"
    ).typed_metadata_identity()
    assert validate_typed_operator_metadata(
        metadata, operator_name="mlp_up_proj", expected_metadata=metadata
    ) == metadata
    with pytest.raises(
        TypeError, match="^architecture_profile must expose attention_linear_ops$"
    ):
        validate_typed_operator_metadata(
            metadata,
            operator_name="mlp_up_proj",
            expected_metadata=metadata,
            architecture_profile=object(),
        )
