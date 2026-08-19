from types import SimpleNamespace

from frontier.entities.batch_stage import BatchStage
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


def test_full_stage_ledger_key_preserves_absent_local_identity() -> None:
    store = object.__new__(MetricsStore)

    key = store._frontier_stage_batch_ledger_key(
        batch_id=9,
        replica_id=2,
        stage_id=1,
        cluster_type=ClusterType.DECODE_FFN,
        replica_local_id=None,
    )

    assert key == (ClusterType.DECODE_FFN.name, 2, None, 1, 9)


def test_stage_ledger_identity_is_captured_from_live_batch_state() -> None:
    request = SimpleNamespace(
        id=7,
        runtime_epoch=2,
        current_decode_token_index=4,
        is_prefill_complete=False,
    )
    batch = SimpleNamespace(
        id=11,
        requests=[request],
        schedule_epoch=3,
        afd_stage_idx=5,
        is_moe=True,
    )
    batch_stage = BatchStage(
        batch_id=11,
        replica_id=0,
        pipeline_stage=0,
        execution_time=0.1,
        model_execution_time=0.1,
        requests=[request],
        num_tokens=[1],
        cluster_type=ClusterType.MONOLITHIC,
    )

    batch_stage.attach_runtime_identity(batch)

    assert batch_stage.runtime_identity == {
        "iteration_ids": [3],
        "schedule_epoch": 3,
        "afd_stage_idx": 5,
        "operation_id": 11,
        "operation_kind": "ep_ffn",
    }
