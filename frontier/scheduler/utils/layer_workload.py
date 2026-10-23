"""Materialize one model layer's expert-parallel workload."""

from typing import Any

from frontier.moe_ep_workload import (
    build_contiguous_expert_ownership,
    materialize_layer_ep_workload,
    resolve_routing_details,
)
from frontier.types import ClusterType
from frontier.moe_routing_policy import build_wave_routing_source, resolve_routing_count_policy, validate_position_routing_scope


def materialize_layer_workload(
    *, scheduler: Any, batch: Any, target_replica_id: int, global_layer_id: int,
    routing_token_count: int | None = None,
    source_batches=None,
) -> Any:
    """Validate scheduler inputs and build a canonical per-layer EP workload."""

    replica_config = getattr(scheduler._config, "replica_config", None)
    model_config = getattr(replica_config, "model_config", None)
    if replica_config is None or model_config is None:
        raise ValueError("Per-layer EP materialization requires replica_config.model_config")
    if not model_config.is_moe:
        raise ValueError("Per-layer EP materialization is invalid for a dense model")
    if type(target_replica_id) is not int or target_replica_id < 0:
        raise ValueError("target_replica_id must be an exact non-negative int")
    if type(global_layer_id) is not int or global_layer_id < 0:
        raise ValueError("global_layer_id must be an exact non-negative int")
    routing_attr = {
        ClusterType.PREFILL: "_prefill_routing_details",
        ClusterType.DECODE: "_decode_routing_details",
        ClusterType.DECODE_FFN: "_decode_ffn_routing_details",
        ClusterType.MONOLITHIC: "_monolithic_routing_details",
    }.get(scheduler._cluster_type)
    if routing_attr is None:
        raise ValueError(
            "Per-layer EP materialization is unsupported for cluster "
            f"{scheduler._cluster_type!r}"
        )
    routing_details = getattr(scheduler._predictor, routing_attr, None)
    if routing_details is None:
        raise ValueError(f"Missing {routing_attr} for {scheduler._cluster_type.name} EP materialization")
    total_expert_num = getattr(replica_config, "total_expert_num", None)
    moe_ep_size = getattr(replica_config, "moe_expert_parallel_size", None)
    router_topk = getattr(replica_config, "router_topk", None)
    if type(total_expert_num) is not int or total_expert_num <= 0:
        raise ValueError("total_expert_num must be an exact positive int")
    if type(moe_ep_size) is not int or moe_ep_size <= 0:
        raise ValueError("moe_expert_parallel_size must be an exact positive int")
    if type(router_topk) is not int or router_topk <= 0:
        raise ValueError("router_topk must be an exact positive int")
    # Physical routing width includes graph padding; a source policy masks it.
    if routing_token_count is None:
        routing_token_count = batch.get_effective_total_tokens_for_compute(scheduler._cluster_type)
    policy_id = getattr(replica_config, "moe_routing_distribution_type", "balanced")
    policy = resolve_routing_count_policy(policy_id)
    routing_source = None
    if policy.requires_source:
        validate_position_routing_scope(replica_config, scheduler._cluster_type)
        if source_batches is None:
            raise ValueError("source-based layer materialization requires explicit source_batches")
        if len(source_batches) != replica_config.attn_dp:
            raise ValueError("routing sources must match the admitted attention DP lanes")
        routing_source = build_wave_routing_source(
            source_batches, policy_id=policy_id, cluster_type=scheduler._cluster_type,
            gathered_physical_rows=routing_token_count, model_config=model_config,
        )
    return materialize_layer_ep_workload(
        routing_ratios=None if policy.requires_source else resolve_routing_details(
            routing_details, target_replica_id, global_layer_id
        ),
        target_replica_id=target_replica_id,
        global_layer_id=global_layer_id,
        routing_token_count=routing_token_count,
        router_topk=router_topk,
        total_expert_num=total_expert_num,
        moe_expert_parallel_size=moe_ep_size,
        expert_to_ep=build_contiguous_expert_ownership(total_expert_num, moe_ep_size),
        policy_id=policy_id,
        routing_source=routing_source,
    )
