"""DECODE_FFN cluster scheduler state initialization."""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from frontier.scheduler.utils.m2n_state import M2NTransferState


def map_source_replica_to_target(
    source_replica_ordinal: int,
    target_replica_ids: List[int] | Tuple[int, ...],
) -> int:
    """Map a source Replica ordinal to a stable target Replica id."""
    if type(source_replica_ordinal) is not int or source_replica_ordinal < 0:
        raise ValueError(
            "source_replica_ordinal must be an exact non-negative int, "
            f"got {source_replica_ordinal!r}"
        )
    if type(target_replica_ids) not in {list, tuple} or not target_replica_ids:
        raise ValueError("target_replica_ids must be a non-empty list or tuple")
    if any(type(replica_id) is not int or replica_id < 0 for replica_id in target_replica_ids):
        raise ValueError("target_replica_ids must contain exact non-negative ints")
    if len(set(target_replica_ids)) != len(target_replica_ids):
        raise ValueError("target_replica_ids must not contain duplicates")
    return target_replica_ids[source_replica_ordinal % len(target_replica_ids)]


def initialize_decode_ffn_state(scheduler: Any, logger) -> None:
    """Initialize M2N grouping and EP waiting-room state for DECODE_FFN."""
    scheduler._m2n_state = M2NTransferState()
    attn_num_replicas = getattr(
        scheduler._config, "decode_attn_cluster_num_replicas", None
    )
    if attn_num_replicas is None:
        raise ValueError(
            "decode_attn_cluster_num_replicas must be set for DECODE_FFN grouping"
        )
    if int(attn_num_replicas) <= 0:
        raise ValueError(
            "DECODE_ATTN cluster capacity must be a positive cluster-level "
            f"num_replicas, got {attn_num_replicas}"
        )
    source_replica_count = int(attn_num_replicas)
    scheduler._ffn_group_micro_batches = source_replica_count
    attn_replica_id_start = getattr(
        scheduler._config, "decode_attn_replica_id_start_for_ffn", None
    )
    if attn_replica_id_start is None:
        raise ValueError(
            "decode_attn_replica_id_start_for_ffn must be set for "
            "DECODE_FFN lane barrier"
        )
    attn_replica_id_start = int(attn_replica_id_start)
    scheduler._ffn_expected_lanes = [
        (attn_replica_id_start + ordinal, None)
        for ordinal in range(source_replica_count)
    ]
    if len(scheduler._ffn_expected_lanes) != source_replica_count:
        raise ValueError(
            "DECODE_ATTN Replica grouping mismatch with expected source topology: "
            f"expected={len(scheduler._ffn_expected_lanes)} configured={source_replica_count}"
        )
    scheduler._ffn_replica_ids = sorted(scheduler._cluster.replicas.keys())
    if not scheduler._ffn_replica_ids:
        raise ValueError("DECODE_FFN cluster must have at least one replica")
    if len(scheduler._ffn_replica_ids) != scheduler._num_replicas:
        raise ValueError(
            "DECODE_FFN replica ID inventory mismatch: "
            f"ids={scheduler._ffn_replica_ids}, num_replicas={scheduler._num_replicas}"
        )
    scheduler._ffn_expected_lanes_by_target: Dict[int, List[Tuple[int, int]]] = {
        replica_id: [] for replica_id in scheduler._ffn_replica_ids
    }
    scheduler._ffn_lane_to_target_replica: Dict[Tuple[int, int], int] = {}
    for lane_ordinal, lane in enumerate(scheduler._ffn_expected_lanes):
        target_replica_id = scheduler._map_source_attn_replica_to_ffn_replica(
            lane_ordinal, scheduler._ffn_replica_ids
        )
        scheduler._ffn_lane_to_target_replica[lane] = target_replica_id
        scheduler._ffn_expected_lanes_by_target[target_replica_id].append(lane)
    expected_group_sizes = {
        replica_id: len(lanes)
        for replica_id, lanes in scheduler._ffn_expected_lanes_by_target.items()
    }
    scheduler._ffn_group_micro_batches = max(expected_group_sizes.values(), default=1)
    scheduler._ffn_idle_lanes = set()
    total_requests = getattr(scheduler._request_generator_config, "num_requests", None)
    if total_requests is not None:
        total_requests = int(total_requests)
        if total_requests < len(scheduler._ffn_expected_lanes):
            scheduler._ffn_idle_lanes = set(
                scheduler._ffn_expected_lanes[total_requests:]
            )
            if scheduler._ffn_idle_lanes:
                logger.info(
                    "[FFN-GROUPING] Precomputed idle lanes for barrier: "
                    f"idle_lanes={sorted(scheduler._ffn_idle_lanes)} "
                    f"total_requests={total_requests}"
                )
    scheduler._ffn_outstanding_group_credit_per_lane = 0
    logger.info(
        "[FFN-GROUPING] Initialized with "
        f"{source_replica_count} full-stage source Replicas for strict "
        "(layer_id, afd_stage_idx) grouping"
    )
    scheduler._ep_allgather_waiting_room = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: {"batches": {}, "arrival_times": {}})
        )
    )
    scheduler._ep_alltoall_dispatch_waiting_room = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: {"batches": {}, "arrival_times": {}})
        )
    )
