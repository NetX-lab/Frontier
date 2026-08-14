from __future__ import annotations

import pytest

from frontier.config.parallel_semantics import (
    build_collective_sim_layout,
    resolve_frontier_parallelism_mapping,
    validate_frontier_shared_parallel_domains,
)


def test_moe_data_parallel_size_is_cluster_replica_capacity() -> None:
    mapping = resolve_frontier_parallelism_mapping(
        model_profile="moe",
        tensor_parallel_size=4,
        data_parallel_size=3,
        enable_expert_parallel=True,
    )

    assert mapping.cluster_num_replicas == 3
    assert mapping.attn_tensor_parallel_size == 4
    assert mapping.attn_data_parallel_size == 1
    assert mapping.moe_tensor_parallel_size == 1
    assert mapping.moe_expert_parallel_size == 4
    assert mapping.attention_parallel_size == mapping.moe_parallel_size == 4


def test_dense_data_parallel_size_creates_replicas_without_ep_lanes() -> None:
    mapping = resolve_frontier_parallelism_mapping(
        model_profile="dense",
        tensor_parallel_size=4,
        data_parallel_size=3,
        enable_expert_parallel=False,
    )

    assert mapping.cluster_num_replicas == 3
    assert mapping.attn_data_parallel_size == 1
    assert mapping.moe_expert_parallel_size == 1


def test_shared_domain_rejects_attention_dp_lanes() -> None:
    mapping = resolve_frontier_parallelism_mapping(
        model_profile="moe",
        tensor_parallel_size=4,
        data_parallel_size=1,
        enable_expert_parallel=True,
    )
    invalid = mapping.__class__(
        cluster_num_replicas=mapping.cluster_num_replicas,
        attn_tensor_parallel_size=4,
        attn_data_parallel_size=2,
        moe_tensor_parallel_size=2,
        moe_expert_parallel_size=4,
    )

    with pytest.raises(ValueError, match="attn_data_parallel_size=1"):
        validate_frontier_shared_parallel_domains(invalid)


def test_collective_sim_attention_dp_uses_cluster_capacity_only() -> None:
    mapping = resolve_frontier_parallelism_mapping(
        model_profile="moe",
        tensor_parallel_size=4,
        data_parallel_size=3,
        enable_expert_parallel=True,
    )

    attention_layout = build_collective_sim_layout(
        mapping=mapping,
        num_pipeline_stages=2,
        domain="attention",
    )
    moe_layout = build_collective_sim_layout(
        mapping=mapping,
        num_pipeline_stages=2,
        domain="moe",
    )

    assert attention_layout.dp == 3
    assert attention_layout.tp == 4
    assert moe_layout.dp == 3
    assert moe_layout.ep == 4
