"""Batch builders used by EP workload prediction paths."""

import copy
from dataclasses import replace
from typing import Any, Callable

from frontier.entities import Batch, EPBatchGroup, Request
from frontier.entities.batch import DecodeCudaGraphMetadata


def build_ep_lane_batch(
    *,
    source_batch: Batch,
    layer_id: int,
    ep_id: int,
    layer_workload: Any,
    create_batch_group: Callable[..., EPBatchGroup],
    cluster_type: Any,
) -> EPBatchGroup:
    """Build an EP lane batch for predictor evaluation without request mutation."""

    lane_workload = layer_workload.lane(ep_id)
    logic_num_tokens = list(lane_workload.local_token_counts)
    logic_requests = [Request(0.0, 0, num_tokens) for num_tokens in logic_num_tokens]
    lane_batch = create_batch_group(
        logic_requests,
        logic_num_tokens,
        source_batch.replica_id,
        ep_id,
        getattr(source_batch, "time", 0.0) or 0.0,
        [source_batch.id],
        lane_workload,
    )
    lane_batch.set_global_id(source_batch.global_id)
    lane_batch.source_batches = [source_batch]
    lane_batch.decode_ffn_layer_id = layer_id
    lane_batch.afd_stage_idx = getattr(source_batch, "afd_stage_idx", None)
    effective_tokens_getter = getattr(source_batch, "get_effective_total_tokens_for_compute", None)
    effective_tokens = (
        int(effective_tokens_getter(cluster_type))
        if callable(effective_tokens_getter)
        else int(source_batch.total_num_tokens)
    )
    if effective_tokens <= 0:
        raise ValueError("Prefill EP lane requires positive pre-routing effective tokens")
    lane_batch.moe_pre_routing_effective_total_tokens = effective_tokens
    return lane_batch


def build_virtual_global_batch(
    sample_batch: Batch,
    total_global_tokens: int,
    total_global_prefill_tokens: int,
) -> Batch:
    """Create a predictor-only batch for one cross-DP token domain."""

    if type(total_global_tokens) is not int or total_global_tokens < 0:
        raise ValueError("total_global_tokens must be a non-negative int")
    if (
        type(total_global_prefill_tokens) is not int
        or total_global_prefill_tokens < 0
        or total_global_prefill_tokens > total_global_tokens
    ):
        raise ValueError("total_global_prefill_tokens must be within the aggregate token range")
    virtual_batch = copy.copy(sample_batch)
    virtual_batch._num_tokens = [total_global_tokens]
    virtual_batch._total_num_tokens = total_global_tokens
    virtual_batch._num_prefill_tokens = total_global_prefill_tokens
    metadata = getattr(virtual_batch, "decode_cuda_graph_metadata", None)
    if metadata is not None and total_global_tokens != sample_batch.total_num_tokens:
        if not isinstance(metadata, DecodeCudaGraphMetadata):
            raise TypeError("decode_cuda_graph_metadata must be DecodeCudaGraphMetadata")
        total_decode_tokens = total_global_tokens - total_global_prefill_tokens
        virtual_batch.decode_cuda_graph_metadata = replace(
            metadata,
            original_total_tokens=total_global_tokens,
            padded_total_tokens=total_global_tokens,
            original_decode_batch_size=total_decode_tokens,
            padded_decode_batch_size=total_decode_tokens,
        )
    return virtual_batch
