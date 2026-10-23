"""Validation for prefix-cache scheduler configuration."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from frontier.types import ClusterSchedulerType, ClusterType, ReplicaSchedulerType, RequestGeneratorType


def validate_prefix_cache_config(
    *,
    replica_scheduler_config: Any,
    cluster_type: ClusterType,
    num_replicas: int,
    cluster_scheduler_config: Any,
    request_generator_config: Any,
) -> None:
    """Fail fast when prefix caching lacks sticky routing or trace metadata."""

    if not bool(getattr(replica_scheduler_config, "enable_prefix_caching", False)):
        return
    scheduler_type = replica_scheduler_config.get_type()
    if scheduler_type not in {
        ReplicaSchedulerType.VLLM_V1,
        ReplicaSchedulerType.SGLANG,
        ReplicaSchedulerType.SJ2Q_FASTSERVE_LITE,
        ReplicaSchedulerType.SJ2Q_PENALTY_ONLY,
        ReplicaSchedulerType.SJ2Q_BOUNDED_CARRYOVER,
    }:
        raise ValueError(
            "Prefix caching only supports vllm_v1, sj2q_fastserve_lite, "
            "sj2q_penalty_only, sj2q_bounded_carryover, or sglang replica "
            f"schedulers. Got {scheduler_type}."
        )
    if cluster_type not in (ClusterType.MONOLITHIC, ClusterType.PREFILL):
        return
    cluster_scheduler_type = cluster_scheduler_config.get_type()
    if num_replicas > 1 and cluster_scheduler_type not in {
        ClusterSchedulerType.STICKY_ROUND_ROBIN,
        ClusterSchedulerType.STICKY_LOR,
    }:
        raise ValueError(
            "Multi-replica prefix caching requires a sticky cluster scheduler. "
            f"Got {cluster_scheduler_type}."
        )
    if request_generator_config is None:
        return
    request_generator_type = request_generator_config.get_type()
    if request_generator_type != RequestGeneratorType.TRACE_REPLAY:
        raise ValueError(
            "Prefix caching requires a trace request source with session_id "
            "and block_hash_ids metadata before scheduling. "
            f"Got {request_generator_type}."
        )
    trace_file = Path(request_generator_config.trace_file)
    if not trace_file.exists():
        raise ValueError(
            "Prefix caching trace request source requires an existing trace file "
            f"with session_id and block_hash_ids columns. Got {trace_file}."
        )
    with trace_file.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"session_id", "block_hash_ids"}
        missing_columns = sorted(required_columns - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(
                "Prefix caching trace request source requires session_id and "
                f"block_hash_ids columns before scheduling. Missing columns: {missing_columns}."
            )
        for row_number, row in enumerate(reader, start=2):
            missing_values = sorted(
                column for column in required_columns
                if row.get(column) is None or not row[column].strip()
            )
            if missing_values:
                raise ValueError(
                    "Prefix caching trace request source requires non-empty "
                    f"session_id and block_hash_ids values before scheduling. Trace file: {trace_file}; "
                    f"row {row_number}; missing values: {missing_values}."
                )
