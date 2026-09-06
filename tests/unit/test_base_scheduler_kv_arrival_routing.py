from types import SimpleNamespace

from frontier.config.global_vars import set_global_vars
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.types import ClusterType


class _Request:
    id = 7
    num_prefill_tokens = 4
    num_decode_tokens = 2
    num_processed_tokens = 4
    total_tokens = 6
    is_prefill_complete = True
    current_decode_token_index = 0
    completed_layer_count = 0

    def on_disaggregated_decode_handoff(self, time, cluster_type):
        self.handoff = (time, cluster_type)

    def on_arrival(self, time, cluster_type):
        self.arrival = (time, cluster_type)


class _Scheduler(BaseClusterScheduler):
    def schedule(self):
        return []


def test_kv_arrival_uses_decode_attention_kv_handler():
    """KV handoff must use the KV handler, not the M2N return handler."""
    set_global_vars("online", "pd-af-disaggregation")
    scheduler = object.__new__(_Scheduler)
    scheduler._cluster_type = ClusterType.DECODE_ATTN
    scheduler._request_queue = []
    request = _Request()
    batch = SimpleNamespace(id=11, requests=[request])
    transfer_info = SimpleNamespace(
        kv_cache_size_bytes=128,
        source_cluster_type=ClusterType.PREFILL,
    )

    events = _Scheduler.on_kv_cache_arrival(
        scheduler, 1.0, batch, transfer_info
    )

    assert len(events) == 1
    assert scheduler._request_queue == [request]
    assert request.handoff == (1.0, ClusterType.DECODE_ATTN)
