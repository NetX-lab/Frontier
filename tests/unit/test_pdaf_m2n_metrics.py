from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from frontier.entities import Request
from frontier.entities.batch import Batch, DenseFFNBatchGroup, EPBatchGroup
from frontier.entities.m2n_transfer_info import M2NTransferInfo
from frontier.events.m2n_transfer_end_event import M2NTransferEndEvent
from frontier.events.m2n_transfer_start_event import M2NTransferStartEvent
from frontier.metrics.metrics_store import MetricsStore
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
    EPBatchGroupPlan,
)
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType


def _metrics_store(*, write_metrics: bool = True) -> MetricsStore:
    store = object.__new__(MetricsStore)
    store._config = SimpleNamespace(
        write_metrics=write_metrics,
        enable_op_level_tracing=False,
        subsamples=None,
        save_table_to_wandb=False,
        store_plots=False,
    )
    store._trace_store = None
    store._cluster_configs = {}
    return store


def _transfer_info(
    source_cluster_type: ClusterType,
    target_cluster_type: ClusterType,
) -> M2NTransferInfo:
    request = SimpleNamespace(id=7)
    batch = SimpleNamespace(id=11, requests=[request])
    return M2NTransferInfo(
        batch=batch,
        source_cluster_type=source_cluster_type,
        target_cluster_type=target_cluster_type,
        source_replica_id=0,
        source_dp_id=0,
        activation_size_bytes=4096,
        transfer_time_ms=2.5,
        transfer_start_time=1.0,
        layer_id=4,
        afd_stage_idx=0,
    )


class _TraceBatch:
    id = 11
    requests = [SimpleNamespace(id=7)]

    def get_effective_total_tokens_for_transfer(
        self,
        cluster_type: ClusterType,
    ) -> int:
        return 8


class _TraceStore:
    def __init__(self) -> None:
        self.events = []

    def log_event(self, event) -> None:
        self.events.append(event)


def _trace_replica_config() -> SimpleNamespace:
    model_config = SimpleNamespace(
        embedding_dim=4096,
        is_moe=False,
    )
    return SimpleNamespace(
        model_config=model_config,
        model_name="unit-model",
        attn_tensor_parallel_size=1,
        attn_data_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        num_pipeline_stages=1,
        router_topk=1,
    )


def _decode_ffn_scheduler() -> VLLMv1EngineReplicaScheduler:
    scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._replica_id = 0
    scheduler._replica_local_id = 0
    scheduler._af_pipeline_num_micro_batch = 1
    scheduler._num_running_batches = 0
    scheduler._m2n_immediate_batch_queue = []
    return scheduler


def _dense_ffn_batch(request: Request, source_batch_id: int) -> DenseFFNBatchGroup:
    return DenseFFNBatchGroup(
        requests=[request],
        num_tokens=[1],
        replica_id=0,
        time=1.0,
        source_batch_ids=[source_batch_id],
        cluster_type=ClusterType.DECODE_FFN,
    )


def _materialized_ep_ffn_batch(
    source_batch: Batch,
    ep_id: int,
) -> EPBatchGroup:
    cluster_scheduler = object.__new__(RoundRobinClusterScheduler)
    cluster_scheduler._cluster_type = ClusterType.DECODE_FFN
    cluster_scheduler._config = SimpleNamespace(
        replica_config=SimpleNamespace(
            model_config=SimpleNamespace(is_moe=True),
        ),
    )
    plan = EPBatchGroupPlan(
        replica_id=0,
        ep_id=ep_id,
        layer_global_id=0,
        afd_stage_idx=0,
        group_time=1.0,
        pre_routing_effective_total_tokens=1,
        source_batches=(source_batch,),
        source_batch_ids=(source_batch.id,),
        per_expert_tokens=((ep_id, 1),),
    )
    return cluster_scheduler._materialize_ep_batch_group(plan)


def test_decode_ffn_schedule_records_m2n_waiting_time() -> None:
    """The actual replica drain time includes barrier and periodic delay."""

    scheduler = _decode_ffn_scheduler()

    request = Request(arrived_at=0.0, num_prefill_tokens=16, num_decode_tokens=4)
    request.on_arrival(1.0, ClusterType.DECODE_FFN)
    batch = _dense_ffn_batch(request, 7)
    scheduler._m2n_immediate_batch_queue = [batch]

    scheduled_batches = scheduler.on_schedule(5.0)

    assert scheduled_batches == [batch]
    assert request.get_cluster_waiting_time(ClusterType.DECODE_FFN) == 4.0
    assert request._is_waiting[ClusterType.DECODE_FFN] is False


def test_decode_ffn_waiting_time_accumulates_across_m2n_rounds() -> None:
    scheduler = _decode_ffn_scheduler()
    request = Request(arrived_at=0.0, num_prefill_tokens=16, num_decode_tokens=4)

    request.on_arrival(1.0, ClusterType.DECODE_FFN)
    scheduler._m2n_immediate_batch_queue = [_dense_ffn_batch(request, 7)]
    scheduler.on_schedule(5.0)

    request.on_arrival(8.0, ClusterType.DECODE_FFN)
    scheduler._m2n_immediate_batch_queue = [_dense_ffn_batch(request, 8)]
    scheduler.on_schedule(11.0)

    assert request.get_cluster_waiting_time(ClusterType.DECODE_FFN) == 7.0
    assert request._is_waiting[ClusterType.DECODE_FFN] is False


def test_decode_ffn_ep_lanes_accumulate_one_logical_waiting_interval() -> None:
    request = Request(arrived_at=0.0, num_prefill_tokens=16, num_decode_tokens=4)
    request.on_arrival(1.0, ClusterType.DECODE_FFN)
    source_batch = Batch(
        replica_id=0,
        requests=[request],
        num_tokens=[1],
        is_moe=True,
    )
    batches = [
        _materialized_ep_ffn_batch(source_batch, ep_id)
        for ep_id in (0, 1)
    ]
    schedulers = [_decode_ffn_scheduler(), _decode_ffn_scheduler()]
    for scheduler, batch in zip(schedulers, batches, strict=True):
        scheduler._m2n_immediate_batch_queue = [batch]

    assert all(batch.requests != [request] for batch in batches)
    assert all(batch.source_batches == [source_batch] for batch in batches)

    with patch.object(
        request,
        "on_leave_waiting_queue",
        wraps=request.on_leave_waiting_queue,
    ) as leave_waiting:
        scheduled_batches = [
            scheduler.on_schedule(5.0)
            for scheduler in schedulers
        ]

    assert scheduled_batches == [[batches[0]], [batches[1]]]
    assert leave_waiting.call_count == 2
    leave_waiting.assert_called_with(5.0, ClusterType.DECODE_FFN)
    assert request.get_cluster_waiting_time(ClusterType.DECODE_FFN) == 4.0


def test_decode_ffn_invalid_ep_batch_does_not_partially_close_waiting() -> None:
    scheduler = _decode_ffn_scheduler()
    real_request = Request(
        arrived_at=0.0,
        num_prefill_tokens=16,
        num_decode_tokens=4,
    )
    real_request.on_arrival(1.0, ClusterType.DECODE_FFN)
    valid_dense_batch = _dense_ffn_batch(real_request, 6)
    logic_request = Request(
        arrived_at=0.0,
        num_prefill_tokens=0,
        num_decode_tokens=1,
    )
    batch = EPBatchGroup(
        requests=[logic_request],
        num_tokens=[1],
        replica_id=0,
        ep_id=0,
        time=1.0,
        source_batch_ids=[7],
        per_expert_tokens={0: 1},
        cluster_type=ClusterType.DECODE_FFN,
        is_moe=True,
    )
    queued_batches = [valid_dense_batch, batch]
    scheduler._m2n_immediate_batch_queue = list(queued_batches)

    with pytest.raises(
        ValueError,
        match="DECODE_FFN EP batch requires non-empty source_batches",
    ):
        scheduler.on_schedule(5.0)

    assert real_request.get_cluster_waiting_time(ClusterType.DECODE_FFN) == 0.0
    assert real_request._is_waiting[ClusterType.DECODE_FFN] is True
    assert scheduler._m2n_immediate_batch_queue == queued_batches
    assert scheduler._num_running_batches == 0


def test_decode_ffn_ep_source_batch_without_requests_fails_fast() -> None:
    scheduler = _decode_ffn_scheduler()
    logic_request = Request(
        arrived_at=0.0,
        num_prefill_tokens=0,
        num_decode_tokens=1,
    )
    empty_source_batch = Batch(
        replica_id=0,
        requests=[],
        num_tokens=[],
        is_moe=True,
    )
    batch = EPBatchGroup(
        requests=[logic_request],
        num_tokens=[1],
        replica_id=0,
        ep_id=0,
        time=1.0,
        source_batch_ids=[empty_source_batch.id],
        per_expert_tokens={0: 1},
        cluster_type=ClusterType.DECODE_FFN,
        is_moe=True,
    )
    batch.source_batches = [empty_source_batch]
    scheduler._m2n_immediate_batch_queue = [batch]

    with pytest.raises(
        ValueError,
        match="DECODE_FFN EP source batch requires non-empty requests",
    ):
        scheduler.on_schedule(5.0)

    assert scheduler._m2n_immediate_batch_queue == [batch]
    assert scheduler._num_running_batches == 0


def test_decode_ffn_materialized_ep_waiting_accumulates_across_rounds() -> None:
    request = Request(arrived_at=0.0, num_prefill_tokens=16, num_decode_tokens=4)

    for arrival_time, schedule_time in ((1.0, 5.0), (8.0, 11.0)):
        request.on_arrival(arrival_time, ClusterType.DECODE_FFN)
        source_batch = Batch(
            replica_id=0,
            requests=[request],
            num_tokens=[1],
            is_moe=True,
        )
        for ep_id in (0, 1):
            scheduler = _decode_ffn_scheduler()
            scheduler._m2n_immediate_batch_queue = [
                _materialized_ep_ffn_batch(source_batch, ep_id)
            ]
            scheduler.on_schedule(schedule_time)

    assert request.get_cluster_waiting_time(ClusterType.DECODE_FFN) == 7.0
    assert request._is_waiting[ClusterType.DECODE_FFN] is False


def test_decode_ffn_idle_lane_does_not_close_real_request_waiting() -> None:
    scheduler = _decode_ffn_scheduler()
    request = Request(arrived_at=0.0, num_prefill_tokens=16, num_decode_tokens=4)
    request.on_arrival(1.0, ClusterType.DECODE_FFN)
    idle_batch = Batch(
        replica_id=0,
        requests=[],
        num_tokens=[],
        is_idle=True,
        is_moe=True,
    )
    scheduler._m2n_immediate_batch_queue = [idle_batch]

    scheduled_batches = scheduler.on_schedule(5.0)

    assert scheduled_batches == [idle_batch]
    assert request.get_cluster_waiting_time(ClusterType.DECODE_FFN) == 0.0
    assert request._is_waiting[ClusterType.DECODE_FFN] is True


@pytest.mark.parametrize(
    ("source", "target", "direction_key"),
    [
        (
            ClusterType.DECODE_ATTN,
            ClusterType.DECODE_FFN,
            "attn_to_ffn_transfers",
        ),
        (
            ClusterType.DECODE_FFN,
            ClusterType.DECODE_ATTN,
            "ffn_to_attn_transfers",
        ),
    ],
)
def test_m2n_callbacks_record_direction_duration_and_size(
    source: ClusterType,
    target: ClusterType,
    direction_key: str,
) -> None:
    store = _metrics_store()
    transfer_info = _transfer_info(source, target)

    store.on_m2n_transfer_start(
        time=1.0,
        source_replica_id=0,
        source_cluster_type=source,
        target_cluster_type=target,
        activation_size_bytes=4096,
        transfer_info=transfer_info,
    )
    store.on_m2n_transfer_end(
        time=1.0025,
        duration=2.5,
        size_bytes=4096,
        source_cluster_type=source,
        target_cluster_type=target,
        transfer_info=transfer_info,
    )

    metrics = store._m2n_transfer_metrics
    assert metrics["transfer_count"] == 1
    assert metrics[direction_key] == 1
    assert metrics["total_transfer_time"] == pytest.approx(2.5)
    assert metrics["total_data_transferred"] == 4096
    assert len(metrics["transfer_times"]) == 1
    assert len(metrics["transfer_sizes"]) == 1


def test_m2n_callbacks_do_not_create_aggregate_metrics_when_disabled() -> None:
    store = _metrics_store(write_metrics=False)
    transfer_info = _transfer_info(
        ClusterType.DECODE_ATTN,
        ClusterType.DECODE_FFN,
    )

    store.on_m2n_transfer_start(
        time=1.0,
        source_replica_id=0,
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        activation_size_bytes=4096,
        transfer_info=transfer_info,
    )
    store.on_m2n_transfer_end(
        time=1.0025,
        duration=2.5,
        size_bytes=4096,
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        transfer_info=transfer_info,
    )

    assert not hasattr(store, "_m2n_transfer_metrics")


def test_m2n_end_without_start_fails_fast_when_metrics_are_enabled() -> None:
    store = _metrics_store()
    transfer_info = _transfer_info(
        ClusterType.DECODE_ATTN,
        ClusterType.DECODE_FFN,
    )

    with pytest.raises(ValueError, match="without a recorded transfer start"):
        store.on_m2n_transfer_end(
            time=1.0025,
            duration=2.5,
            size_bytes=4096,
            source_cluster_type=ClusterType.DECODE_ATTN,
            target_cluster_type=ClusterType.DECODE_FFN,
            transfer_info=transfer_info,
        )


def test_f2a_op_traces_are_emitted_when_aggregate_metrics_are_disabled() -> None:
    store = _metrics_store(write_metrics=False)
    store._config.enable_op_level_tracing = True
    store._trace_store = _TraceStore()
    replica_config = _trace_replica_config()
    store._cluster_configs = {
        ClusterType.DECODE_FFN: SimpleNamespace(replica_config=replica_config),
        ClusterType.DECODE_ATTN: SimpleNamespace(replica_config=replica_config),
    }
    transfer_info = M2NTransferInfo(
        batch=_TraceBatch(),
        source_cluster_type=ClusterType.DECODE_FFN,
        target_cluster_type=ClusterType.DECODE_ATTN,
        source_replica_id=0,
        source_dp_id=1,
        activation_size_bytes=65536,
        transfer_time_ms=2.5,
        transfer_start_time=1.0,
        layer_id=4,
        afd_stage_idx=0,
    )

    store.on_m2n_transfer_start(
        time=1.0,
        source_replica_id=0,
        source_cluster_type=ClusterType.DECODE_FFN,
        target_cluster_type=ClusterType.DECODE_ATTN,
        activation_size_bytes=65536,
        transfer_info=transfer_info,
    )
    store.on_m2n_transfer_end(
        time=1.0025,
        duration=2.5,
        size_bytes=65536,
        source_cluster_type=ClusterType.DECODE_FFN,
        target_cluster_type=ClusterType.DECODE_ATTN,
        transfer_info=transfer_info,
    )

    assert [event.name for event in store._trace_store.events] == [
        "m2n_transfer_ffn_to_attn",
        "m2n_transfer_ffn_to_attn_recv",
    ]
    send_event, receive_event = store._trace_store.events
    assert send_event.ts_start == pytest.approx(1.0)
    assert receive_event.ts_start == pytest.approx(1.0)
    assert send_event.duration_ms == pytest.approx(2.5)
    assert receive_event.duration_ms == pytest.approx(2.5)
    assert send_event.meta["activation_size_bytes"] == 65536
    assert receive_event.meta["activation_size_bytes"] == 65536
    assert not hasattr(store, "_m2n_transfer_metrics")


class _FailingTransferRequest:
    id = 7

    def validate_inter_cluster_transfer_start(self, **kwargs) -> None:
        raise ValueError("start invariant violated")

    def validate_inter_cluster_transfer_end(self, **kwargs) -> None:
        raise ValueError("end invariant violated")

    def on_inter_cluster_transfer_start(self, **kwargs) -> None:
        raise ValueError("start invariant violated")

    def on_inter_cluster_transfer_end(self, **kwargs) -> None:
        raise ValueError("end invariant violated")

    def on_m2n_transfer_complete(
        self,
        transfer_time: float,
        is_attn_to_ffn: bool,
    ) -> None:
        return None


class _EventMetricsStore:
    def on_m2n_transfer_start(self, *args, **kwargs) -> None:
        return None

    def on_m2n_transfer_end(self, *args, **kwargs) -> None:
        return None


class _EventClusterScheduler(BaseClusterScheduler):
    def schedule(self):
        raise AssertionError("schedule() is not used by these unit tests")


class _EventScheduler:
    def get_cluster_scheduler(self, cluster_type: ClusterType):
        cluster_scheduler = object.__new__(_EventClusterScheduler)
        cluster_scheduler._cluster_type = cluster_type
        if cluster_type == ClusterType.DECODE_FFN:
            cluster_scheduler._ffn_expected_lanes = [(0, 0)]
            cluster_scheduler._ffn_group_micro_batches = 1
            cluster_scheduler._m2n_waiting_by_layer = {}
        cluster_scheduler.on_m2n_arrival = lambda *args: []
        return cluster_scheduler


def _failing_hook_batch() -> SimpleNamespace:
    return SimpleNamespace(
        id=11,
        global_id=13,
        requests=[_FailingTransferRequest()],
        afd_stage_idx=0,
    )


def _timeline_request() -> Request:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=0,
        num_decode_tokens=2,
    )
    request._is_prefill_complete = True
    return request


def _request_transfer_state(request: Request) -> tuple:
    return (
        request._af_roundtrip_inflight,
        request._decode_ffn_enter_time,
        request._decode_ffn_residence_time,
        request._latest_stage_completed_at,
        request._m2n_transfer_time_attn_to_ffn,
        request._m2n_transfer_time_ffn_to_attn,
    )


def _timeline_batch(requests: list[Request]) -> SimpleNamespace:
    return SimpleNamespace(
        id=11,
        global_id=13,
        requests=requests,
        afd_stage_idx=0,
    )


def _ffn_event_scheduler():
    cluster_scheduler = _EventScheduler().get_cluster_scheduler(
        ClusterType.DECODE_FFN
    )
    cluster_scheduler.on_m2n_arrival = Mock(return_value=[])
    scheduler = SimpleNamespace(
        get_cluster_scheduler=Mock(return_value=cluster_scheduler)
    )
    return scheduler, cluster_scheduler


def test_request_f2a_start_validator_rejects_missing_roundtrip_without_mutation() -> None:
    request = _timeline_request()
    request._decode_ffn_enter_time = 0.5
    before = _request_transfer_state(request)

    with pytest.raises(
        ValueError,
        match="F->A transfer start without active roundtrip",
    ):
        request.validate_inter_cluster_transfer_start(
            time=1.0,
            source_cluster=ClusterType.DECODE_FFN,
            target_cluster=ClusterType.DECODE_ATTN,
            activation_size_bytes=4096,
        )

    assert _request_transfer_state(request) == before


def test_request_f2a_start_validator_rejects_negative_residence_without_mutation() -> None:
    request = _timeline_request()
    request._af_roundtrip_inflight = True
    request._decode_ffn_enter_time = 1.5
    before = _request_transfer_state(request)

    with pytest.raises(ValueError, match="Negative DECODE_FFN residence"):
        request.validate_inter_cluster_transfer_start(
            time=1.0,
            source_cluster=ClusterType.DECODE_FFN,
            target_cluster=ClusterType.DECODE_ATTN,
            activation_size_bytes=4096,
        )

    assert _request_transfer_state(request) == before


def test_request_f2a_end_validator_rejects_missing_roundtrip_without_mutation() -> None:
    request = _timeline_request()
    before = _request_transfer_state(request)

    with pytest.raises(
        ValueError,
        match="F->A transfer end without active roundtrip",
    ):
        request.validate_inter_cluster_transfer_end(
            time=1.0,
            source_cluster=ClusterType.DECODE_FFN,
            target_cluster=ClusterType.DECODE_ATTN,
            activation_size_bytes=4096,
        )

    assert _request_transfer_state(request) == before


@pytest.mark.parametrize(
    (
        "validator_name",
        "source_cluster",
        "target_cluster",
        "roundtrip_inflight",
        "decode_ffn_enter_time",
    ),
    [
        (
            "validate_inter_cluster_transfer_start",
            ClusterType.DECODE_ATTN,
            ClusterType.DECODE_FFN,
            False,
            None,
        ),
        (
            "validate_inter_cluster_transfer_start",
            ClusterType.DECODE_FFN,
            ClusterType.DECODE_ATTN,
            True,
            0.5,
        ),
        (
            "validate_inter_cluster_transfer_end",
            ClusterType.DECODE_ATTN,
            ClusterType.DECODE_FFN,
            True,
            None,
        ),
        (
            "validate_inter_cluster_transfer_end",
            ClusterType.DECODE_FFN,
            ClusterType.DECODE_ATTN,
            True,
            None,
        ),
    ],
)
def test_request_transfer_validator_accepts_legal_state_without_mutation(
    validator_name: str,
    source_cluster: ClusterType,
    target_cluster: ClusterType,
    roundtrip_inflight: bool,
    decode_ffn_enter_time: float | None,
) -> None:
    request = _timeline_request()
    request._af_roundtrip_inflight = roundtrip_inflight
    request._decode_ffn_enter_time = decode_ffn_enter_time
    before = _request_transfer_state(request)

    getattr(request, validator_name)(
        time=1.0,
        source_cluster=source_cluster,
        target_cluster=target_cluster,
        activation_size_bytes=4096,
    )

    assert _request_transfer_state(request) == before


@pytest.mark.parametrize(
    (
        "source_cluster",
        "target_cluster",
        "roundtrip_inflight",
        "decode_ffn_enter_time",
    ),
    [
        (
            ClusterType.DECODE_ATTN,
            ClusterType.DECODE_FFN,
            False,
            None,
        ),
        (
            ClusterType.DECODE_FFN,
            ClusterType.DECODE_ATTN,
            True,
            0.5,
        ),
    ],
)
def test_request_transfer_start_hook_reuses_validator(
    source_cluster: ClusterType,
    target_cluster: ClusterType,
    roundtrip_inflight: bool,
    decode_ffn_enter_time: float | None,
) -> None:
    request = _timeline_request()
    request._af_roundtrip_inflight = roundtrip_inflight
    request._decode_ffn_enter_time = decode_ffn_enter_time
    request.validate_inter_cluster_transfer_start = Mock(
        wraps=request.validate_inter_cluster_transfer_start,
    )

    request.on_inter_cluster_transfer_start(
        time=1.0,
        source_cluster=source_cluster,
        target_cluster=target_cluster,
        activation_size_bytes=4096,
    )

    request.validate_inter_cluster_transfer_start.assert_called_once_with(
        time=1.0,
        source_cluster=source_cluster,
        target_cluster=target_cluster,
        activation_size_bytes=4096,
    )


@pytest.mark.parametrize(
    ("source_cluster", "target_cluster"),
    [
        (ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN),
        (ClusterType.DECODE_FFN, ClusterType.DECODE_ATTN),
    ],
)
def test_request_transfer_end_hook_reuses_validator(
    source_cluster: ClusterType,
    target_cluster: ClusterType,
) -> None:
    request = _timeline_request()
    request._af_roundtrip_inflight = True
    request.validate_inter_cluster_transfer_end = Mock(
        wraps=request.validate_inter_cluster_transfer_end,
    )

    request.on_inter_cluster_transfer_end(
        time=1.0,
        source_cluster=source_cluster,
        target_cluster=target_cluster,
        activation_size_bytes=4096,
    )

    request.validate_inter_cluster_transfer_end.assert_called_once_with(
        time=1.0,
        source_cluster=source_cluster,
        target_cluster=target_cluster,
        activation_size_bytes=4096,
    )


def test_m2n_start_propagates_request_timeline_invariant_failure() -> None:
    event = M2NTransferStartEvent(
        time=1.0,
        source_replica_id=0,
        source_dp_id=0,
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        batch=_failing_hook_batch(),
        activation_size_bytes=4096,
        transfer_time_ms=2.5,
        layer_id=4,
        afd_stage_idx=0,
    )

    with pytest.raises(ValueError, match="start invariant violated"):
        event.handle_event(_EventScheduler(), _EventMetricsStore())


def test_m2n_start_validates_all_requests_before_metrics_or_mutation() -> None:
    requests = [_timeline_request(), _timeline_request()]
    requests[1]._af_roundtrip_inflight = True
    before = [_request_transfer_state(request) for request in requests]
    metrics_store = Mock()
    event = M2NTransferStartEvent(
        time=1.0,
        source_replica_id=0,
        source_dp_id=0,
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        batch=_timeline_batch(requests),
        activation_size_bytes=4096,
        transfer_time_ms=2.5,
        layer_id=4,
        afd_stage_idx=0,
    )

    with pytest.raises(
        ValueError,
        match="A->F transfer start while roundtrip already in-flight",
    ):
        event.handle_event(SimpleNamespace(), metrics_store)

    assert [_request_transfer_state(request) for request in requests] == before
    metrics_store.on_m2n_transfer_start.assert_not_called()


@pytest.mark.parametrize(
    ("cohort_case", "error_match"),
    [
        ("empty", "request cohort must not be empty"),
        ("duplicate", "duplicate request IDs"),
    ],
)
def test_m2n_start_rejects_invalid_request_cohort_before_all_mutation(
    cohort_case: str,
    error_match: str,
) -> None:
    request = _timeline_request()
    batch_requests = [] if cohort_case == "empty" else [request, request]
    before = _request_transfer_state(request)
    metrics_store = Mock()
    event = M2NTransferStartEvent(
        time=1.0,
        source_replica_id=0,
        source_dp_id=0,
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        batch=_timeline_batch(batch_requests),
        activation_size_bytes=4096,
        transfer_time_ms=2.5,
        layer_id=4,
        afd_stage_idx=0,
    )

    with pytest.raises(ValueError, match=error_match):
        try:
            event.handle_event(SimpleNamespace(), metrics_store)
        finally:
            assert _request_transfer_state(request) == before
            metrics_store.on_m2n_transfer_start.assert_not_called()


def test_m2n_f2a_start_validates_all_requests_before_mutation() -> None:
    requests = [_timeline_request(), _timeline_request()]
    for request in requests:
        request._af_roundtrip_inflight = True
    requests[0]._decode_ffn_enter_time = 0.5
    requests[1]._decode_ffn_enter_time = None
    before = [_request_transfer_state(request) for request in requests]
    metrics_store = Mock()
    event = M2NTransferStartEvent(
        time=1.0,
        source_replica_id=0,
        source_dp_id=0,
        source_cluster_type=ClusterType.DECODE_FFN,
        target_cluster_type=ClusterType.DECODE_ATTN,
        batch=_timeline_batch(requests),
        activation_size_bytes=4096,
        transfer_time_ms=2.5,
        layer_id=4,
        afd_stage_idx=0,
    )

    with pytest.raises(ValueError, match="DECODE_FFN exit without entry"):
        event.handle_event(SimpleNamespace(), metrics_store)

    assert [_request_transfer_state(request) for request in requests] == before
    metrics_store.on_m2n_transfer_start.assert_not_called()


def test_m2n_end_propagates_request_timeline_invariant_failure() -> None:
    transfer_info = M2NTransferInfo(
        batch=_failing_hook_batch(),
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        source_replica_id=0,
        source_dp_id=0,
        activation_size_bytes=4096,
        transfer_time_ms=2.5,
        transfer_start_time=1.0,
        layer_id=4,
        afd_stage_idx=0,
    )
    event = M2NTransferEndEvent(time=1.0025, transfer_info=transfer_info)

    with pytest.raises(ValueError, match="end invariant violated"):
        event.handle_event(_EventScheduler(), _EventMetricsStore())


def test_m2n_end_validates_all_requests_before_metrics_or_mutation() -> None:
    requests = [_timeline_request(), _timeline_request()]
    requests[0]._af_roundtrip_inflight = True
    requests[1]._af_roundtrip_inflight = False
    before = [_request_transfer_state(request) for request in requests]
    batch = _timeline_batch(requests)
    transfer_info = M2NTransferInfo(
        batch=batch,
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        source_replica_id=0,
        source_dp_id=0,
        activation_size_bytes=4096,
        transfer_time_ms=2.5,
        transfer_start_time=1.0,
        layer_id=4,
        afd_stage_idx=0,
    )
    before_transfer_end_time = transfer_info.transfer_end_time
    event = M2NTransferEndEvent(time=1.005, transfer_info=transfer_info)
    scheduler, cluster_scheduler = _ffn_event_scheduler()
    metrics_store = Mock()

    with pytest.raises(
        ValueError,
        match="A->F transfer end without active roundtrip",
    ):
        event.handle_event(scheduler, metrics_store)

    assert transfer_info.transfer_end_time == before_transfer_end_time
    assert [_request_transfer_state(request) for request in requests] == before
    metrics_store.on_m2n_transfer_end.assert_not_called()
    cluster_scheduler.on_m2n_arrival.assert_not_called()


@pytest.mark.parametrize(
    ("cohort_case", "error_match"),
    [
        ("empty", "request cohort must not be empty"),
        ("duplicate", "duplicate request IDs"),
    ],
)
def test_m2n_end_rejects_invalid_request_cohort_before_all_mutation(
    cohort_case: str,
    error_match: str,
) -> None:
    request = _timeline_request()
    request._af_roundtrip_inflight = True
    batch_requests = [] if cohort_case == "empty" else [request, request]
    before = _request_transfer_state(request)
    transfer_info = M2NTransferInfo(
        batch=_timeline_batch(batch_requests),
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        source_replica_id=0,
        source_dp_id=0,
        activation_size_bytes=4096,
        transfer_time_ms=2.5,
        transfer_start_time=1.0,
        layer_id=4,
        afd_stage_idx=0,
    )
    before_transfer_end_time = transfer_info.transfer_end_time
    event = M2NTransferEndEvent(time=1.005, transfer_info=transfer_info)
    cluster_scheduler = SimpleNamespace(
        preflight_m2n_arrival=Mock(),
        on_m2n_arrival=Mock(return_value=[]),
    )
    scheduler = SimpleNamespace(
        get_cluster_scheduler=Mock(return_value=cluster_scheduler),
    )
    metrics_store = Mock()

    with pytest.raises(ValueError, match=error_match):
        try:
            event.handle_event(scheduler, metrics_store)
        finally:
            assert transfer_info.transfer_end_time == before_transfer_end_time
            assert _request_transfer_state(request) == before
            scheduler.get_cluster_scheduler.assert_not_called()
            cluster_scheduler.preflight_m2n_arrival.assert_not_called()
            metrics_store.on_m2n_transfer_end.assert_not_called()
            cluster_scheduler.on_m2n_arrival.assert_not_called()


def test_m2n_end_rejects_open_ffn_entry_before_all_mutation() -> None:
    requests = [_timeline_request(), _timeline_request()]
    for request in requests:
        request._af_roundtrip_inflight = True
    requests[1]._decode_ffn_enter_time = 0.75
    before = [_request_transfer_state(request) for request in requests]
    batch = _timeline_batch(requests)
    transfer_info = M2NTransferInfo(
        batch=batch,
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        source_replica_id=0,
        source_dp_id=0,
        activation_size_bytes=4096,
        transfer_time_ms=2.5,
        transfer_start_time=1.0,
        layer_id=4,
        afd_stage_idx=0,
    )
    before_transfer_end_time = transfer_info.transfer_end_time
    event = M2NTransferEndEvent(time=1.005, transfer_info=transfer_info)
    scheduler, cluster_scheduler = _ffn_event_scheduler()
    metrics_store = Mock()

    with pytest.raises(ValueError, match="DECODE_FFN entry already open"):
        event.handle_event(scheduler, metrics_store)

    assert transfer_info.transfer_end_time == before_transfer_end_time
    assert [_request_transfer_state(request) for request in requests] == before
    metrics_store.on_m2n_transfer_end.assert_not_called()
    cluster_scheduler.on_m2n_arrival.assert_not_called()


def test_m2n_f2a_end_validates_all_requests_before_metrics_or_mutation() -> None:
    requests = [_timeline_request(), _timeline_request()]
    requests[0]._af_roundtrip_inflight = True
    requests[1]._af_roundtrip_inflight = False
    before = [_request_transfer_state(request) for request in requests]
    batch = _timeline_batch(requests)
    transfer_info = M2NTransferInfo(
        batch=batch,
        source_cluster_type=ClusterType.DECODE_FFN,
        target_cluster_type=ClusterType.DECODE_ATTN,
        source_replica_id=0,
        source_dp_id=0,
        activation_size_bytes=4096,
        transfer_time_ms=2.5,
        transfer_start_time=1.0,
        layer_id=4,
        afd_stage_idx=0,
    )
    before_transfer_end_time = transfer_info.transfer_end_time
    event = M2NTransferEndEvent(time=1.005, transfer_info=transfer_info)
    cluster_scheduler = SimpleNamespace(
        preflight_m2n_arrival=Mock(),
        on_m2n_arrival=Mock(return_value=[]),
    )
    scheduler = SimpleNamespace(
        get_cluster_scheduler=Mock(return_value=cluster_scheduler),
    )
    metrics_store = Mock()

    with pytest.raises(
        ValueError,
        match="F->A transfer end without active roundtrip",
    ):
        event.handle_event(scheduler, metrics_store)

    assert transfer_info.transfer_end_time == before_transfer_end_time
    assert [_request_transfer_state(request) for request in requests] == before
    metrics_store.on_m2n_transfer_end.assert_not_called()
    cluster_scheduler.on_m2n_arrival.assert_not_called()
