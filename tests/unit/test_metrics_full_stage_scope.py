from types import SimpleNamespace

from frontier.metrics.metrics_store import MetricsStore
from frontier.types import ClusterType


class _Meter:
    def __init__(self) -> None:
        self.points = []

    def put(self, time: float, value: float) -> None:
        self.points.append((time, value))


def _store() -> MetricsStore:
    store = object.__new__(MetricsStore)
    store._config = SimpleNamespace(
        write_metrics=True,
        store_utilization_metrics=True,
        store_operation_metrics=False,
    )
    store._cluster_configs = {
        ClusterType.DECODE_FFN: SimpleNamespace(num_replicas=1),
    }
    store._replica_busy_time = {
        ClusterType.DECODE_FFN: [[[_Meter()]]],
    }
    store._replica_mfu = {
        ClusterType.DECODE_FFN: [[[_Meter()]]],
    }
    store._mfu_calculator = {
        ClusterType.DECODE_FFN: SimpleNamespace(get_mfu=lambda _stage: 37.0),
    }
    store._pending_frontier_stage_batch_ledger_rows = {}
    store._pending_frontier_stage_batch_ledger_row_keys = {}
    store._pending_frontier_stage_batch_ledger_rows_by_key = {}
    store._emit_op_level_traces = lambda **_kwargs: None
    store._should_capture_frontier_stage_batch_ledger = lambda: False
    return store


def test_dense_full_stage_metrics_do_not_use_ep_lane_zero() -> None:
    store = _store()
    full_stage_busy = _Meter()
    full_stage_mfu = _Meter()
    store._replica_full_stage_busy_time = {
        ClusterType.DECODE_FFN: [[full_stage_busy]],
    }
    store._replica_full_stage_mfu = {
        ClusterType.DECODE_FFN: [[full_stage_mfu]],
    }

    store.on_replica_stage_schedule(
        1.0,
        0,
        0,
        SimpleNamespace(request_ids=[], execution_time=0.25),
        None,
        ClusterType.DECODE_FFN,
        None,
    )

    assert full_stage_busy.points == [(1.0, 100)]
    assert full_stage_mfu.points == [(1.0, 37.0)]
    assert store._replica_busy_time[ClusterType.DECODE_FFN][0][0][0].points == []


def test_dense_full_stage_metrics_close_on_stage_end() -> None:
    store = _store()
    full_stage_busy = _Meter()
    full_stage_mfu = _Meter()
    store._replica_full_stage_busy_time = {
        ClusterType.DECODE_FFN: [[full_stage_busy]],
    }
    store._replica_full_stage_mfu = {
        ClusterType.DECODE_FFN: [[full_stage_mfu]],
    }

    store.on_batch_stage_end(
        SimpleNamespace(),
        2.0,
        0,
        0,
        ClusterType.DECODE_FFN,
        None,
    )

    assert full_stage_busy.points == [(2.0, 0)]
    assert full_stage_mfu.points == [(2.0, 0)]


def test_dense_full_stage_memory_metrics_use_separate_series() -> None:
    store = _store()
    full_stage_memory = _Meter()
    store._replica_full_stage_memory_usage = {
        ClusterType.DECODE_FFN: [full_stage_memory],
    }

    store.on_replica_schedule(
        3.0,
        0,
        42,
        ClusterType.DECODE_FFN,
        None,
    )

    assert full_stage_memory.points == [(3.0, 42)]
