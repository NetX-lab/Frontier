from pathlib import Path


def test_non_ffn_cluster_scheduler_uses_full_stage_identity() -> None:
    source = Path(
        "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py"
    ).read_text(encoding="utf-8")
    round_robin_source = Path(
        "frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.py"
    ).read_text(encoding="utf-8")

    block_start = source.index("        else:\n            # Every non-FFN Replica")
    block_end = source.index("        self._request_queue = []", block_start)
    non_ffn_block = source[block_start:block_end]
    assert "replica_local_id=None" in non_ffn_block
    assert "for dp_id in range(self._replica_scheduler_count)" not in non_ffn_block
    assert "(replica_id, None, request)" in round_robin_source


def test_production_scheduler_surface_has_no_retired_replica_dp_size() -> None:
    production_paths = [
        Path("frontier/entities/replica.py"),
        Path("frontier/scheduler/global_scheduler/base_global_scheduler.py"),
        Path("frontier/scheduler/cluster_scheduler/random_cluster_scheduler.py"),
        Path("frontier/scheduler/cluster_scheduler/lor_cluster_scheduler.py"),
    ]
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in production_paths
    )

    assert "def dp_size(" not in combined
    assert "_replica_dp_size" not in combined


def test_afd_transport_identity_has_no_dp_named_fields() -> None:
    production_paths = [
        Path("frontier/entities/batch.py"),
        Path("frontier/entities/m2n_transfer_info.py"),
        Path("frontier/entities/kv_cache_transfer_info.py"),
        Path("frontier/events/m2n_transfer_start_event.py"),
        Path("frontier/events/kv_cache_transfer_start_event.py"),
        Path("frontier/events/cluster_batch_end_event.py"),
    ]
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in production_paths
    )

    assert "decode_attn_original_dp_id" not in combined
    assert "source_dp_id" not in combined


def test_decode_sync_event_uses_replica_local_identity() -> None:
    source = Path("frontier/events/decode_sync_event.py").read_text(
        encoding="utf-8"
    )

    assert "self._dp_id" not in source
    assert '"dp_id"' not in source
    assert "replica_local_id" in source


def test_replica_stage_scheduler_uses_replica_local_identity() -> None:
    source = Path(
        "frontier/scheduler/replica_stage_scheduler/replica_stage_schduler.py"
    ).read_text(encoding="utf-8")

    assert "replica_local_id" in source
    assert "dp_id" not in source
    assert "_dp_id" not in source


def test_replica_stage_schedule_event_uses_replica_local_identity() -> None:
    source = Path("frontier/events/replica_stage_schedule_event.py").read_text(
        encoding="utf-8"
    )

    assert "replica_local_id" in source
    assert "dp_id" not in source
    assert "_dp_id" not in source


def test_replica_schedule_event_uses_replica_local_identity() -> None:
    source = Path("frontier/events/replica_schedule_event.py").read_text(
        encoding="utf-8"
    )

    assert "replica_local_id" in source
    assert "dp_id" not in source
    assert "_dp_id" not in source


def test_batch_stage_events_use_replica_local_identity() -> None:
    sources = [
        Path("frontier/events/batch_stage_arrival_event.py").read_text(
            encoding="utf-8"
        ),
        Path("frontier/events/batch_stage_end_event.py").read_text(
            encoding="utf-8"
        ),
    ]

    for source in sources:
        assert "replica_local_id" in source
        assert "dp_id" not in source
        assert "_dp_id" not in source


def test_cluster_batch_end_event_uses_replica_local_identity() -> None:
    source = Path("frontier/events/cluster_batch_end_event.py").read_text(
        encoding="utf-8"
    )

    assert "replica_local_id" in source
    assert "dp_id" not in source
    assert "_dp_id" not in source


def test_global_batch_end_event_uses_replica_local_identity() -> None:
    source = Path("frontier/events/global_batch_end_event.py").read_text(
        encoding="utf-8"
    )

    assert "replica_local_id" in source
    assert "dp_id" not in source
    assert "_dp_id" not in source


def test_batch_end_and_prefill_sync_events_use_replica_local_identity() -> None:
    sources = [
        Path("frontier/events/batch_end_event.py").read_text(encoding="utf-8"),
        Path("frontier/events/prefill_sync_event.py").read_text(
            encoding="utf-8"
        ),
    ]

    for source in sources:
        assert "replica_local_id" in source
        assert "dp_id" not in source
        assert "_dp_id" not in source


def test_cluster_schedule_events_use_replica_local_identity() -> None:
    sources = [
        Path("frontier/events/cluster_schedule_event.py").read_text(
            encoding="utf-8"
        ),
        Path("frontier/events/periodic_schedule_event.py").read_text(
            encoding="utf-8"
        ),
    ]

    for source in sources:
        assert "replica_local_id" in source
        assert "dp_id" not in source
        assert "_dp_replica_set" not in source


def test_sticky_round_robin_uses_ep_identity_for_ffn_targets() -> None:
    source = Path(
        "frontier/scheduler/cluster_scheduler/sticky_round_robin_cluster_scheduler.py"
    ).read_text(encoding="utf-8")

    assert "ep_id" in source
    assert "dp_id" not in source


def test_thinking_mode_requeue_uses_replica_local_identity() -> None:
    request_source = Path("frontier/entities/request.py").read_text(
        encoding="utf-8"
    )
    event_source = Path(
        "frontier/events/thinking_round_requeue_event.py"
    ).read_text(encoding="utf-8")

    assert "thinking_home_replica_local_id" in request_source
    assert "thinking_home_dp_id" not in request_source
    assert "replica_local_id" in event_source
    assert "thinking_home_dp_id" not in event_source
    assert '"dp_id"' not in event_source


def test_simulator_trace_identity_uses_replica_local_identity() -> None:
    sources = [
        Path("frontier/simulator.py").read_text(encoding="utf-8"),
        Path("frontier/cluster_simulator.py").read_text(encoding="utf-8"),
    ]

    for source in sources:
        assert "replica_local_id" in source
        assert "_dp_id" not in source
        assert '"dp_id"' not in source
        assert "'dp_id'" not in source
