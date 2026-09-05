from __future__ import annotations

import pytest

from frontier.config.parallel_semantics import (
    build_collective_sim_layout,
    resolve_frontier_parallelism_mapping,
    validate_frontier_shared_parallel_domains,
)


def test_moe_num_replicas_is_cluster_replica_capacity() -> None:
    mapping = resolve_frontier_parallelism_mapping(
        model_profile="moe",
        tensor_parallel_size=4,
        num_replicas=3,
        enable_expert_parallel=True,
        attn_dp=2,
    )

    assert mapping.cluster_num_replicas == 3
    assert mapping.attn_tensor_parallel_size == 4
    assert mapping.attn_dp == 2
    assert mapping.moe_tensor_parallel_size == 1
    assert mapping.moe_expert_parallel_size == 8
    assert mapping.attention_parallel_size == mapping.moe_parallel_size == 8


def test_dense_num_replicas_creates_replicas_without_ep_lanes() -> None:
    mapping = resolve_frontier_parallelism_mapping(
        model_profile="dense",
        tensor_parallel_size=4,
        num_replicas=3,
        enable_expert_parallel=False,
    )

    assert mapping.cluster_num_replicas == 3
    assert mapping.attn_dp == 1
    assert mapping.moe_expert_parallel_size == 1


def test_shared_domain_accepts_attention_dp_lanes_when_domains_match() -> None:
    mapping = resolve_frontier_parallelism_mapping(
        model_profile="moe",
        tensor_parallel_size=4,
        num_replicas=1,
        enable_expert_parallel=True,
    )
    valid = mapping.__class__(
        cluster_num_replicas=mapping.cluster_num_replicas,
        attn_tensor_parallel_size=4,
        attn_dp=2,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
    )
    validate_frontier_shared_parallel_domains(valid)

    invalid = valid.__class__(
        cluster_num_replicas=valid.cluster_num_replicas,
        attn_tensor_parallel_size=4,
        attn_dp=2,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=4,
    )
    with pytest.raises(ValueError, match="attn_tp\\*attn_dp"):
        validate_frontier_shared_parallel_domains(invalid)


def test_collective_sim_attention_dp_uses_replica_local_domain() -> None:
    mapping = resolve_frontier_parallelism_mapping(
        model_profile="moe",
        tensor_parallel_size=4,
        num_replicas=3,
        enable_expert_parallel=True,
        attn_dp=2,
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

    assert attention_layout.dp == 2
    assert attention_layout.tp == 4
    assert moe_layout.dp == 3
    assert moe_layout.ep == 8


def test_resolver_maps_tp_dp_ep_domains_with_one_replica_pod() -> None:
    mapping = resolve_frontier_parallelism_mapping(
        model_profile="moe",
        tensor_parallel_size=4,
        num_replicas=1,
        enable_expert_parallel=True,
        attn_dp=2,
    )

    assert mapping.to_dict() == {
        "cluster_num_replicas": 1,
        "attn_tensor_parallel_size": 4,
        "attn_dp": 2,
        "moe_tensor_parallel_size": 1,
        "moe_expert_parallel_size": 8,
    }
    attention_layout = build_collective_sim_layout(
        mapping=mapping,
        num_pipeline_stages=1,
        domain="attention",
    )
    moe_layout = build_collective_sim_layout(
        mapping=mapping,
        num_pipeline_stages=1,
        domain="moe",
    )
    assert attention_layout.world_size == 8
    assert moe_layout.world_size == 8
