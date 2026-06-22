#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.config.config import (
    FixedRequestLengthGeneratorConfig,
    TraceRequestLengthGeneratorConfig,
    RoundRobinClusterSchedulerConfig,
    StickyRoundRobinClusterSchedulerConfig,
    SyntheticRequestGeneratorConfig,
    TraceRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.types import ClusterType


class _DummyClusterScheduler(BaseClusterScheduler):
    def schedule(self):
        raise NotImplementedError


def _build_scheduler(
    *,
    num_replicas: int,
    cluster_type: ClusterType,
    cluster_scheduler_config,
    replica_scheduler_config,
    request_generator_config=None,
):
    scheduler = object.__new__(_DummyClusterScheduler)
    scheduler._cluster_type = cluster_type
    scheduler._num_replicas = num_replicas
    scheduler._config = SimpleNamespace(
        cluster_scheduler_config=cluster_scheduler_config,
    )
    scheduler._request_generator_config = request_generator_config
    return scheduler


def test_prefix_cache_requires_sticky_scheduler_for_multi_replica_clusters() -> None:
    scheduler = _build_scheduler(
        num_replicas=2,
        cluster_type=ClusterType.MONOLITHIC,
        cluster_scheduler_config=RoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
    )

    with pytest.raises(ValueError, match="sticky"):
        scheduler._validate_prefix_cache_cluster_config(
            VllmV1SchedulerConfig(enable_prefix_caching=True)
        )


def test_prefix_cache_allows_sticky_scheduler_for_multi_replica_clusters() -> None:
    scheduler = _build_scheduler(
        num_replicas=2,
        cluster_type=ClusterType.PREFILL,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
    )

    scheduler._validate_prefix_cache_cluster_config(
        VllmV1SchedulerConfig(enable_prefix_caching=True)
    )


def test_prefix_cache_rejects_plain_synthetic_request_source_before_scheduling() -> None:
    scheduler = _build_scheduler(
        num_replicas=1,
        cluster_type=ClusterType.PREFILL,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
        request_generator_config=SyntheticRequestGeneratorConfig(
            length_generator_config=FixedRequestLengthGeneratorConfig(
                prefill_tokens=32,
                decode_tokens=8,
            )
        ),
    )

    with pytest.raises(ValueError, match="session_id.*block_hash_ids"):
        scheduler._validate_prefix_cache_cluster_config(
            VllmV1SchedulerConfig(enable_prefix_caching=True)
        )


def test_prefix_cache_rejects_synthetic_trace_length_source_before_scheduling(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "synthetic_length_trace.csv"
    trace_file.write_text(
        "num_prefill_tokens,num_decode_tokens,session_id,block_hash_ids\n"
        "32,8,7,11|22\n",
        encoding="utf-8",
    )
    scheduler = _build_scheduler(
        num_replicas=1,
        cluster_type=ClusterType.PREFILL,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
        request_generator_config=SyntheticRequestGeneratorConfig(
            length_generator_config=TraceRequestLengthGeneratorConfig(
                trace_file=str(trace_file),
            )
        ),
    )

    with pytest.raises(ValueError, match="trace request source"):
        scheduler._validate_prefix_cache_cluster_config(
            VllmV1SchedulerConfig(enable_prefix_caching=True)
        )


def test_prefix_cache_allows_trace_request_source_with_prefix_metadata(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "prefix_trace.csv"
    trace_file.write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens,session_id,block_hash_ids\n"
        "0.0,32,8,7,11|22\n",
        encoding="utf-8",
    )
    scheduler = _build_scheduler(
        num_replicas=1,
        cluster_type=ClusterType.PREFILL,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
        request_generator_config=TraceRequestGeneratorConfig(
            trace_file=str(trace_file)
        ),
    )

    scheduler._validate_prefix_cache_cluster_config(
        VllmV1SchedulerConfig(enable_prefix_caching=True)
    )


def test_prefix_cache_rejects_trace_request_source_without_block_hash_ids(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "missing_prefix_trace.csv"
    trace_file.write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens,session_id\n"
        "0.0,32,8,7\n",
        encoding="utf-8",
    )
    scheduler = _build_scheduler(
        num_replicas=1,
        cluster_type=ClusterType.PREFILL,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
        request_generator_config=TraceRequestGeneratorConfig(
            trace_file=str(trace_file)
        ),
    )

    with pytest.raises(ValueError, match="block_hash_ids"):
        scheduler._validate_prefix_cache_cluster_config(
            VllmV1SchedulerConfig(enable_prefix_caching=True)
        )


@pytest.mark.parametrize(
    ("row", "missing_column"),
    [
        ("0.0,32,8,,11|22\n", "session_id"),
        ("0.0,32,8,7,\n", "block_hash_ids"),
    ],
)
def test_prefix_cache_rejects_trace_request_source_with_empty_metadata_values(
    tmp_path: Path,
    row: str,
    missing_column: str,
) -> None:
    trace_file = tmp_path / "empty_prefix_metadata_trace.csv"
    trace_file.write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens,session_id,block_hash_ids\n"
        + row,
        encoding="utf-8",
    )
    scheduler = _build_scheduler(
        num_replicas=1,
        cluster_type=ClusterType.PREFILL,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
        request_generator_config=TraceRequestGeneratorConfig(
            trace_file=str(trace_file)
        ),
    )

    with pytest.raises(ValueError, match=fr"row 2.*{missing_column}"):
        scheduler._validate_prefix_cache_cluster_config(
            VllmV1SchedulerConfig(enable_prefix_caching=True)
        )
