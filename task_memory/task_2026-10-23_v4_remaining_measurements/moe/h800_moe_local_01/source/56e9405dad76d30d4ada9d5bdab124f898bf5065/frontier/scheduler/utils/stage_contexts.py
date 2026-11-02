"""Build stage admission contexts from cluster topology and model settings."""

from __future__ import annotations

from typing import Any


def pipeline_layer_bounds(stage_id: int, layers_per_stage: int) -> tuple[int, int]:
    """Return the global half-open layer range owned by one PP stage."""
    if type(stage_id) is not int or stage_id < 0:
        raise ValueError(
            "pipeline stage_id must be an exact non-negative int, "
            f"got {stage_id!r}"
        )
    if type(layers_per_stage) is not int or layers_per_stage <= 0:
        raise ValueError(
            "num_layers_per_pipeline_stage must be an exact positive int, "
            f"got {layers_per_stage!r}"
        )
    first_layer_id = stage_id * layers_per_stage
    return first_layer_id, first_layer_id + layers_per_stage

from frontier.scheduler.replica_stage_scheduler.stage_execution_context import StageExecutionContext
from frontier.types import ClusterType


def build_stage_execution_contexts(
    *,
    cluster: Any,
    cluster_type: ClusterType,
    replica_config: Any,
    replica_dp_size: int,
) -> dict[tuple[int, int], StageExecutionContext]:
    """Create one admission context for every physical Replica stage."""

    model_config = getattr(replica_config, "model_config", None)
    if replica_config is None or model_config is None:
        raise ValueError("Stage execution contexts require replica_config.model_config")
    model_is_moe = bool(getattr(model_config, "is_moe", False))
    shared_ep_clusters = (
        ClusterType.MONOLITHIC,
        ClusterType.PREFILL,
        ClusterType.DECODE,
        ClusterType.DECODE_FFN,
    )
    has_local_ep_domain = model_is_moe and cluster_type in shared_ep_clusters
    configured_ep_size = getattr(replica_config, "moe_expert_parallel_size", None)
    if has_local_ep_domain:
        if type(configured_ep_size) is not int or configured_ep_size <= 0:
            raise ValueError(
                "MoE stage execution contexts require an exact positive "
                "moe_expert_parallel_size"
            )
        ep_size = configured_ep_size
    else:
        ep_size = 1

    contexts: dict[tuple[int, int], StageExecutionContext] = {}
    shared_full_stage_capacity = int(replica_dp_size or 1)
    for replica_id, replica in cluster.replicas.items():
        if type(replica_id) is not int or replica_id < 0:
            raise ValueError("Cluster Replica IDs must be exact non-negative ints")
        num_stages = getattr(replica, "num_pipeline_stages", None)
        if num_stages is None:
            num_stages = getattr(replica_config, "num_pipeline_stages", None)
        if type(num_stages) is not int or num_stages <= 0:
            raise ValueError("Replica num_pipeline_stages must be an exact positive int")
        for stage_id in range(num_stages):
            contexts[(replica_id, stage_id)] = StageExecutionContext(
                replica_id=replica_id,
                stage_id=stage_id,
                ep_size=ep_size,
                full_stage_capacity=(
                    shared_full_stage_capacity
                    if cluster_type in (ClusterType.MONOLITHIC, ClusterType.PREFILL, ClusterType.DECODE)
                    else 1
                ),
            )
    return contexts
