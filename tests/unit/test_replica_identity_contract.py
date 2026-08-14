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
