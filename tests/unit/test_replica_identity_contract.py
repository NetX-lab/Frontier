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
