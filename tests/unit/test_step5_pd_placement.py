"""Native-source DP policy and Step structural endpoint/handoff contracts."""

from types import SimpleNamespace
import heapq

import pytest

from frontier.config import (
    AnalyticalKVCacheTransferConfig, ClusterConfig, MetricsConfig,
    RandomForrestExecutionTimePredictorConfig, SyntheticRequestGeneratorConfig,
    VllmLoadBalancingClusterSchedulerConfig, VllmV1SchedulerConfig, global_vars,
)
from frontier.attention.model_binding import bind_layer_attention
from frontier.entities import Batch, Cluster, DummyForwardBatch, ExecutionTime, Request, StageExecutionTime
from frontier.entities.time_components import CommunicationOperatorTimes
from frontier.events.batch_stage_end_event import BatchStageEndEvent
from frontier.events.replica_schedule_event import ReplicaScheduleEvent
from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import (
    AnalyticalKVCacheTransferPredictor,
)
from frontier.scheduler.cluster_scheduler.vllm_load_balancing_cluster_scheduler import (
    VllmLoadBalancingClusterScheduler,
)
from frontier.scheduler.request_load import RequestLoad
from frontier.scheduler.utils.vllm_dp_load_balancer import VllmDPLoadBalancer
from frontier.types import ClusterType
from tests.unit.test_step5_structural_pd import (
    _GlobalScheduler, _Metrics, _harness, _replica_config,
    _request_owned_by_prefill, _start_event, simulation_globals,
)
from tests.unit.test_stage_execution_time import _execution_kwargs


class _StageCosts:
    """Fixed CPU layer timings for event lifecycle tests, not calibration inputs."""

    def __init__(self, replica, replica_ids):
        self._model_config = replica.model_config
        self._num_layers_per_pipeline_stage = self._model_config.num_layers
        routing = {expert: 1 / replica.total_expert_num for expert in range(replica.total_expert_num)}
        self._prefill_routing_details = self._decode_routing_details = {
            replica_id: {layer: routing for layer in range(self._num_layers_per_pipeline_stage)}
            for replica_id in replica_ids}
        self._stages = {}

    def predict_stage_execution_time(self, batch, stage_id, cluster_type=None, *, num_layers,
                                     layer_id, include_attention=True, include_ffn=True):
        key = (layer_id, num_layers, include_attention, include_ffn)
        if key not in self._stages:
            values = {name: 0.0 if type(value) in (int, float) else value
                      for name, value in _execution_kwargs().items()}
            moe = include_ffn and self._model_config.is_moe_layer(layer_id)
            attention = bind_layer_attention(self._model_config, layer_id)
            values.update(num_layers_per_pipeline_stage=1, is_moe=moe, global_layer_id=layer_id,
                          attention_family_id=attention.family_id,
                          attention_variant_id=attention.variant_id,
                          attention_layer_pre_proj_execution_time=float(include_attention),
                          mlp_layer_up_proj_execution_time=float(include_ffn and not moe),
                          moe_grouped_gemm_time=float(moe),
                          communication_operator_times=CommunicationOperatorTimes({
                              'expert_parallel_alltoall_dispatch': 1.0,
                              'expert_parallel_alltoall_combine': 1.0}) if moe else None)
            self._stages[key] = StageExecutionTime.from_execution_time(
                ExecutionTime(**values), num_layers=num_layers, first_layer_id=layer_id)
        return self._stages[key]


class _ExecutionMetrics(_Metrics):
    ep_wave_reporting_enabled = stage_execution_reporting_enabled = False

    def __init__(self):
        super().__init__()
        self.outputs = []

    def on_replica_stage_schedule(self, *args, **kwargs):
        pass

    def on_batch_stage_end(self, *args):
        pass

    def on_batch_end(self, time, batch, replica_id, memory, role, lane_id):
        self.outputs.append((time, batch, role, lane_id))

    def _on_request_end(self, *args):
        pass


class _PDExecution:
    """Dispatch every returned DES event; retain the queue at observation boundaries."""

    def __init__(self, scheduler, metrics=None):
        self.scheduler = scheduler
        self.metrics = metrics or _ExecutionMetrics()
        self.queue, self.batches, self.dummy_ends, self.releases = [], [], set(), set()
        self.time = 0.0

    def add(self, events):
        for event in events:
            heapq.heappush(self.queue, (event._priority_number, event))

    def until(self, condition):
        while not condition():
            assert self.queue, 'Event queue drained before the expected lifecycle boundary'
            _, event = heapq.heappop(self.queue)
            self.time = event.time
            children = event.handle_event(self.scheduler, self.metrics)
            self.add(children)
            if isinstance(event, ReplicaScheduleEvent):
                self.batches.extend((batch, list(batch.num_tokens), list(batch.num_context_tokens))
                                    for batch in event._batches if not isinstance(batch, DummyForwardBatch))
                peer_ids = {child._replica_local_id for child in children
                            if isinstance(child, ReplicaScheduleEvent)}
                dp_size = self.scheduler.get_cluster_scheduler(event._cluster_type)._replica_dp_size
                if peer_ids == set(range(dp_size)) - {event._replica_local_id}:
                    self.releases.add(event._cluster_type)
            if isinstance(event, BatchStageEndEvent) and isinstance(event._batch, DummyForwardBatch):
                self.dummy_ends.add((event._cluster_type, event._replica_local_id))

    def drain(self, role):
        self.until(lambda: not self.queue)
        cluster = self.scheduler.get_cluster_scheduler(role)
        lanes = list(cluster._replica_schedulers.values())
        assert all(not lane._dp_wave_running and lane._dp_wave_steps == 0
                   and not lane._waits_at_dp_sync and not lane._dummy_forwards_in_flight for lane in lanes)
        assert {(role, lane._replica_local_id) for lane in lanes} <= self.dummy_ends
        assert role in self.releases


def _scheduler(role=ClusterType.DECODE, **changes):
    replica = changes.pop("replica", None) or _replica_config()
    generator = SyntheticRequestGeneratorConfig()
    config = ClusterConfig(
        cluster_type=role, replica_config=replica,
        num_replicas=changes.pop("num_replicas", 1),
        cluster_scheduler_config=VllmLoadBalancingClusterSchedulerConfig(
            policy_version=changes.pop("policy_version", "v0.28.0")),
        replica_scheduler_config=VllmV1SchedulerConfig(
            num_blocks=0, num_blocks_mode="memory_planner", block_size=16,
            max_model_len=8192, max_tokens_in_batch=512, batch_size_cap=1,
            enable_chunked_prefill=True, watermark_blocks_fraction=0,
        ),
        execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
            enable_dummy_mode=changes.pop("dummy", True)),
    )
    predictor = AnalyticalKVCacheTransferPredictor(
        AnalyticalKVCacheTransferConfig(kv_connector=changes.pop("connector", "nixl")))
    available = changes.pop("available", {ClusterType.PREFILL, ClusterType.DECODE})
    assert not changes
    cluster = Cluster(config, MetricsConfig(write_metrics=False), generator)
    return VllmLoadBalancingClusterScheduler(
        config=config, cluster=cluster,
        request_generator_config=generator, predictor=_StageCosts(replica, cluster.replicas),
        kv_cache_transfer_predictor=predictor, available_clusters=available,
    )


def test_client_count_scales_inflight_and_local_waiting():
    balancer = VllmDPLoadBalancer(4, policy_version="v0.28.0", client_count=3, client_index=2)
    assert balancer.eng_start_index == 2  # floor(4 * 2 / 3), native :1638-1640.
    assert balancer.select(0, 10) == 2
    assert balancer.frontend_counts[2].waiting == 3
    balancer.frontend_counts = [RequestLoad(0, 2)] * 4
    # Native inflight floor is 3, not 1, even after stats reset.
    assert balancer._v028_score(2) == 3
    assert balancer.select(0, 11) == 3
    assert balancer.engine_inflight == [0, 0, 1, 1]


@pytest.mark.parametrize(("usage", "score"), [(0, 3), (0.5, 3), (0.625, 4.5), (0.75, 6), (1, 9)])
def test_native_kv_pressure_ramp(usage, score):
    balancer = VllmDPLoadBalancer(2, policy_version="v0.28.0", client_count=2)
    balancer.frontend_counts[0] = RequestLoad(2, 1, usage)
    assert balancer._v028_score(0) == score


def test_affinity_does_not_rotate_or_double_count_and_terminal_is_idempotent():
    balancer = VllmDPLoadBalancer(4, policy_version="v0.28.0", client_count=2)
    assert balancer.select(0, 10) == 0
    assert balancer.select(0, 10) == 0
    assert balancer.eng_start_index == 1
    assert balancer.frontend_counts[0].waiting == 2
    assert balancer.engine_inflight == [1, 0, 0, 0]
    balancer.finish_requests([10, 10, 999])
    assert balancer.request_owners == {}
    assert balancer.engine_inflight == [0] * 4


def test_rotating_tie_after_stats_reset_keeps_unfinished_floor():
    balancer = VllmDPLoadBalancer(4, policy_version="v0.28.0")
    assert [balancer.select(0, i) for i in range(4)] == [0, 1, 2, 3]
    # Idle publication replaces local counts but not request ownership.
    assert balancer.select(0.05, 4) == 0
    assert balancer.engine_inflight == [2, 1, 1, 1]
    balancer.finish_requests(range(5))
    assert balancer.select(5.1, 5) == 1


def test_source_coordinator_previous_step_then_current_after_100ms():
    balancer = VllmDPLoadBalancer(4, policy_version="v0.28.0")
    balancer.report(0, 0, 0, RequestLoad(0, 4))
    balancer.report(0.001, 1, 1, RequestLoad(0, 8))
    # The first publish delivers the prior step: lane 1 ties with 2 and 3.
    assert balancer.select(0.05, 10) == 1
    assert balancer.frontend_counts[1] == RequestLoad(1, 0)
    assert balancer.stats_changed
    # Current reports would have picked lane 2, so the stale view matters.
    assert balancer._v028_score(1) == 1
    # Latching removes the 50 ms collection minimum, so the first publish
    # happened at 1 ms; the next is 100 ms later (coordinator :274-285).
    assert balancer.next_publish_ms == 101
    assert balancer.select(0.100, 11) == 2
    assert balancer.last_publish_ms == 1
    assert balancer.select(0.101, 12) == 3
    assert balancer.last_publish_ms == 101
    assert not balancer.stats_changed
    assert balancer.frontend_counts[1] == RequestLoad(0, 8)
    assert balancer.next_publish_ms == 5101


@pytest.mark.parametrize("role", [ClusterType.PREFILL, ClusterType.DECODE])
def test_step_pd_constructs_one_endpoint_with_four_request_owners(role):
    scheduler = _scheduler(role)
    assert len(scheduler._cluster.replicas) == 1
    assert len(scheduler._replica_schedulers) == 4
    assert scheduler._config.replica_config.world_size == 32


@pytest.mark.parametrize("changes", [
    {"policy_version": "v0.10.2"}, {"dummy": False}, {"connector": "none"},
    {"available": {ClusterType.PREFILL}}, {"num_replicas": 2},
])
def test_pd_scope_is_not_a_removed_guard(changes):
    with pytest.raises(ValueError, match="vllm_load_balancing"):
        _scheduler(**changes)


def test_non_step_pd_is_rejected():
    from tests.unit.test_vllm_dp_load_balancer import _policy_scheduler
    with pytest.MonkeyPatch.context() as patch:
        with pytest.raises(ValueError, match="paired Step"):
            _policy_scheduler(patch, cluster_type=ClusterType.PREFILL)


def test_decode_cannot_be_admitted_again_through_cluster_queue():
    scheduler = _scheduler()
    scheduler.add_request(Request(0, 16, 3))
    with pytest.raises(ValueError, match="NIXL target placement"):
        scheduler.schedule_at(0)


def test_p_client_finishes_at_response_while_source_kv_is_retained():
    scheduler = _scheduler(ClusterType.PREFILL)
    request = Request(0, 529, 3)
    scheduler.add_request(request)
    [(replica_id, lane_id, _)] = scheduler.schedule_at(0)
    lane = scheduler.get_replica_scheduler(replica_id, lane_id)
    lane.add_request(request)
    execution = _PDExecution(_GlobalScheduler(scheduler, _scheduler()))
    execution.add([ReplicaScheduleEvent(0, replica_id, ClusterType.PREFILL, lane_id)])
    for step in range(2):
        execution.until(lambda: len(execution.metrics.outputs) == step + 1)
        assert sum(scheduler._load_balancer.engine_inflight) == (1 if step == 0 else 0)
    assert not request.completed
    assert request.id in lane._pending_kv_transfer_requests
    assert request.id in lane._allocation_map
    execution.drain(ClusterType.PREFILL)


def test_selected_d_lane_survives_capacity_retry_read_ready_and_decode(monkeypatch):
    source, _, old_global, _ = _harness()
    target = _scheduler()
    scheduler = _GlobalScheduler(old_global._clusters[ClusterType.PREFILL], target)
    request = _request_owned_by_prefill(source, 17)
    event = _start_event(request, source)
    execution = _PDExecution(scheduler)
    metrics = execution.metrics
    replica_id = target._serving_replica_id
    selected_lane = target.get_replica_scheduler(replica_id, 0)
    reserve = selected_lane.reserve_kv_transfer_target
    monkeypatch.setattr(selected_lane, "reserve_kv_transfer_target", lambda request: None)
    execution.add(event.handle_event(scheduler, metrics))
    assert selected_lane.has_pending_kv_transfer_starts()
    assert target._load_balancer.engine_inflight == [1, 0, 0, 0]
    assert selected_lane.get_request_load().waiting == 1
    assert not metrics.starts
    # Even a contrary fresh snapshot cannot migrate the deferred reservation.
    target._load_balancer.frontend_counts = [RequestLoad(0, 99)] + [RequestLoad(0, 0)] * 3
    monkeypatch.setattr(selected_lane, "reserve_kv_transfer_target", reserve)
    selected_lane._retry_pending_kv_transfer_starts(1.1)
    assert len(selected_lane._popped_output_events) == 1
    execution.add(selected_lane.consume_popped_output_events())
    execution.until(lambda: len(metrics.starts) == 1)
    [end] = [queued for _, queued in execution.queue if hasattr(queued, '_transfer_info')]
    assert selected_lane.get_request_load().waiting == 1  # Receiving remote KV.
    info = end._transfer_info
    assert (info.target_replica_id, info.target_replica_local_id) == (replica_id, 0)
    execution.until(lambda: len(metrics.ends) == 1)
    assert selected_lane.get_request_load().waiting == 1  # Ready, admitted once.
    assert selected_lane._waiting_requests == [request]
    assert all(request not in target.get_replica_scheduler(replica_id, i)._waiting_requests
               for i in range(1, 4))
    execution.drain(ClusterType.DECODE)
    assert request.completed
    assert target._load_balancer.request_owners == {}
    assert target._load_balancer.engine_inflight == [0] * 4
    assert len(metrics.starts) == len(metrics.ends) == 1
