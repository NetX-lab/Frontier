"""pd-disaggregation transfer entity contract tests."""

from types import SimpleNamespace

import pytest

from frontier.types import ClusterType


def test_kv_cache_transfer_info_computes_end_time_effective_bytes_and_dict() -> None:
    from frontier.entities import KVCacheTransferInfo

    batch = SimpleNamespace(id=11, global_id=101)
    info = KVCacheTransferInfo(
        batch=batch,
        source_cluster_type=ClusterType.PREFILL,
        target_cluster_type=ClusterType.DECODE,
        source_replica_id=3,
        source_dp_id=2,
        kv_cache_size_bytes=4096,
        transfer_time_ms=2.5,
        transfer_start_time=7.0,
        enable_compression=True,
        compression_ratio=2.0,
        enable_latency_hiding=True,
        transfer_protocol="rdma",
        transfer_requests=True,
    )

    assert info.is_completed is True
    assert info.transfer_end_time == pytest.approx(7.0025)
    assert info.effective_data_size_bytes == 2048

    data = info.to_dict()
    assert data["batch_id"] == 11
    assert data["batch_global_id"] == 101
    assert data["source_cluster_type"] == "PREFILL"
    assert data["target_cluster_type"] == "DECODE"
    assert data["source_replica_id"] == 3
    assert data["kv_cache_size_bytes"] == 4096
    assert data["effective_data_size_bytes"] == 2048
    assert data["transfer_time_ms"] == pytest.approx(2.5)
    assert data["transfer_start_time"] == pytest.approx(7.0)
    assert data["transfer_end_time"] == pytest.approx(7.0025)
    assert data["enable_compression"] is True
    assert data["compression_ratio"] == pytest.approx(2.0)
    assert data["enable_latency_hiding"] is True
    assert data["transfer_protocol"] == "rdma"
    assert data["transfer_requests"] is True


def test_m2n_transfer_info_validates_direction_and_sets_pipeline_stage() -> None:
    from frontier.entities import M2NTransferInfo

    batch = SimpleNamespace(id=12, global_id=102)
    info = M2NTransferInfo(
        batch=batch,
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        source_replica_id=4,
        source_dp_id=1,
        activation_size_bytes=8192,
        transfer_time_ms=1.2,
        transfer_start_time=9.0,
        enable_compression=True,
        compression_ratio=4.0,
        layer_id=5,
        afd_stage_idx=6,
        target_ffn_replica_id=7,
    )

    assert info.is_completed is True
    assert info.transfer_end_time == pytest.approx(9.0012)
    assert info.effective_data_size_bytes == 2048
    assert info.pipeline_stage == "attn_to_ffn"
    assert info.is_attn_to_ffn is True
    assert info.is_ffn_to_attn is False

    data = info.to_dict()
    assert data["batch_id"] == 12
    assert data["batch_global_id"] == 102
    assert data["source_cluster_type"] == "DECODE_ATTN"
    assert data["target_cluster_type"] == "DECODE_FFN"
    assert data["source_replica_id"] == 4
    assert data["source_dp_id"] == 1
    assert data["activation_size_bytes"] == 8192
    assert data["effective_data_size_bytes"] == 2048
    assert data["transfer_time_ms"] == pytest.approx(1.2)
    assert data["transfer_start_time"] == pytest.approx(9.0)
    assert data["transfer_end_time"] == pytest.approx(9.0012)
    assert data["enable_p2p_optimization"] is True
    assert data["p2p_protocol"] == "nvlink"
    assert data["enable_compression"] is True
    assert data["compression_ratio"] == pytest.approx(4.0)
    assert data["enable_latency_hiding"] is False
    assert data["layer_id"] == 5
    assert data["afd_stage_idx"] == 6
    assert data["pipeline_stage"] == "attn_to_ffn"
    assert data["target_ffn_replica_id"] == 7


def test_m2n_transfer_info_rejects_non_m2n_cluster_pairs() -> None:
    from frontier.entities import M2NTransferInfo

    batch = SimpleNamespace(id=13, global_id=103)

    with pytest.raises(ValueError, match="DECODE_ATTN <-> DECODE_FFN"):
        M2NTransferInfo(
            batch=batch,
            source_cluster_type=ClusterType.PREFILL,
            target_cluster_type=ClusterType.DECODE,
            source_replica_id=0,
            source_dp_id=0,
            activation_size_bytes=1,
            transfer_time_ms=0.1,
            transfer_start_time=0.0,
        )


def test_kv_cache_transfer_start_event_targets_decode_cluster_for_routing() -> None:
    from frontier.cluster_simulator import ClusterSimulator
    from frontier.events.kv_cache_transfer_start_event import KVCacheTransferStartEvent

    batch = SimpleNamespace(id=14, global_id=104, requests=[])
    event = KVCacheTransferStartEvent(
        time=1.0,
        source_replica_id=0,
        source_dp_id=0,
        source_cluster_type=ClusterType.PREFILL,
        target_cluster_type=ClusterType.DECODE,
        batch=batch,
        kv_cache_size_bytes=1024,
        transfer_time_ms=0.5,
    )
    simulator = object.__new__(ClusterSimulator)
    simulator._cluster_type = ClusterType.PREFILL

    assert event.get_target_cluster() is ClusterType.DECODE
    assert simulator._determine_target_cluster(event) is ClusterType.DECODE


def test_transfer_info_without_compression_preserves_original_size() -> None:
    from frontier.entities import KVCacheTransferInfo, M2NTransferInfo

    batch = SimpleNamespace(id=21, global_id=201)
    kv_info = KVCacheTransferInfo(
        batch=batch,
        source_cluster_type=ClusterType.PREFILL,
        target_cluster_type=ClusterType.DECODE,
        source_replica_id=0,
        source_dp_id=0,
        kv_cache_size_bytes=0,
        transfer_time_ms=0.0,
        transfer_start_time=3.0,
    )
    assert kv_info.effective_data_size_bytes == 0
    assert kv_info.transfer_end_time == pytest.approx(3.0)

    m2n_info = M2NTransferInfo(
        batch=batch,
        source_cluster_type=ClusterType.DECODE_FFN,
        target_cluster_type=ClusterType.DECODE_ATTN,
        source_replica_id=0,
        source_dp_id=0,
        activation_size_bytes=128,
        transfer_time_ms=0.0,
        transfer_start_time=3.5,
    )
    assert m2n_info.effective_data_size_bytes == 128
    assert m2n_info.transfer_end_time == pytest.approx(3.5)
    assert m2n_info.pipeline_stage == "ffn_to_attn"
    assert m2n_info.is_attn_to_ffn is False
    assert m2n_info.is_ffn_to_attn is True
