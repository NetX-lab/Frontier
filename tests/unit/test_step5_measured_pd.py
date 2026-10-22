"""CPU PD contract checks; fixture timings are not native qualification."""

from dataclasses import replace
import pytest

from frontier.config import (
    AnalyticalKVCacheTransferConfig, ClusterConfig, MetricsConfig,
    RandomForrestExecutionTimePredictorConfig, SyntheticRequestGeneratorConfig,
    VllmLoadBalancingClusterSchedulerConfig, VllmV1SchedulerConfig,
)
from frontier.entities import Cluster, Request
from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import (
    AnalyticalKVCacheTransferPredictor,
)
from frontier.scheduler.cluster_scheduler.vllm_load_balancing_cluster_scheduler import (
    VllmLoadBalancingClusterScheduler,
)
from frontier.types import ClusterType
from tests.unit.test_step5_pd_placement import _PDExecution, _StageCosts, _scheduler
from tests.unit.test_step5_structural_pd import (
    _GlobalScheduler,
    _Metrics,
    _harness,
    _request_owned_by_prefill,
    _replica_config,
    _start_event,
    capacity,
    simulation_globals,
    step_config,
)


@pytest.mark.parametrize("mode", ["offline", "online"])
def test_step_nixl_cost_shape_uses_computed_frontier_without_changing_outputs(
    monkeypatch, mode,
):
    from frontier.config import global_vars

    monkeypatch.setattr(global_vars, "SIMULATION_MODE", mode)
    source, _, old_scheduler, _ = _harness()
    target_cluster = _scheduler()
    scheduler = _GlobalScheduler(
        old_scheduler.get_cluster_scheduler(ClusterType.PREFILL), target_cluster,
    )
    request = _request_owned_by_prefill(source, 4105)
    execution = _PDExecution(scheduler)
    execution.add(_start_event(request, source).handle_event(scheduler, execution.metrics))
    [end] = [event for _, event in execution.queue]
    info = end._transfer_info
    target = target_cluster.get_replica_scheduler(
        info.target_replica_id, info.target_replica_local_id,
    )
    execution.until(lambda: len(execution.metrics.ends) == 1)

    contexts = []
    for step in range(9):
        execution.until(lambda: request.num_emitted_decode_tokens == step + 1)
        _, tokens, context = execution.batches[step]
        contexts.extend(context)
        assert tokens == [1]
        assert request.num_emitted_decode_tokens == step + 1
    execution.drain(ClusterType.DECODE)

    assert contexts == list(range(4104, 4113))
    assert request.completed
    assert request.num_processed_tokens == 4114
    assert info.kv_cache_size_bytes == 103_687_956
    assert source._allocation_map == target._allocation_map == {}
    assert target._scheduled_num_computed_tokens_by_request == {}


class _FixtureCosts(_StageCosts):
    """Explicit CPU contract input, not fitted H200 or native evidence."""

    def __init__(self, *, duration=3.5, changes=None, covered=True):
        super().__init__(_replica_config(), ())
        self.reads = []
        self.duration = duration
        self.changes = changes or {}
        self.covered = covered

    def get_nixl_transfer_coverage(
        self, *, request, source_replica_config, target_replica_config,
        cache_capacity, target_cost_provider,
    ):
        from frontier.entities.kv_cache_transfer_info import NixlTransferCoverage

        if not self.covered:
            return None
        return replace(NixlTransferCoverage(
            num_prefill_tokens=4105, num_computed_tokens=4080,
            cache_capacity=cache_capacity,
            source_attention_tp=8, target_attention_tp=8,
            source_device="h800", target_device="h800",
            profile_identity="CPU_CONTRACT_FIXTURE_NOT_NATIVE",
            duration_scope="read_to_gpu_ready",
        ), **self.changes)

    def get_nixl_transfer_time_ms(self, *, coverage, payload):
        self.reads.append((coverage, payload))
        return self.duration


def _measured_cluster(role, costs, *, max_tokens=8192, attn_dp=4):
    generator = SyntheticRequestGeneratorConfig()
    replica_config = _replica_config()
    replica_config.attn_dp = attn_dp
    replica_config.moe_expert_parallel_size = replica_config.attn_tensor_parallel_size * attn_dp
    config = ClusterConfig(
        cluster_type=role, replica_config=replica_config,
        cluster_scheduler_config=VllmLoadBalancingClusterSchedulerConfig(
            policy_version="v0.28.0",
        ),
        replica_scheduler_config=VllmV1SchedulerConfig(
            num_blocks=0, num_blocks_mode="memory_planner", block_size=16,
            max_model_len=8192, max_tokens_in_batch=max_tokens, batch_size_cap=1,
            enable_chunked_prefill=False, watermark_blocks_fraction=0,
        ),
        execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
            enable_dummy_mode=False,
        ),
    )
    cluster = Cluster(config, MetricsConfig(write_metrics=False), generator)
    if isinstance(costs, _FixtureCosts):
        costs._prefill_routing_details.update(_StageCosts(replica_config, cluster.replicas)._prefill_routing_details)
    return VllmLoadBalancingClusterScheduler(
        config=config, cluster=cluster,
        request_generator_config=generator, predictor=costs,
        kv_cache_transfer_predictor=AnalyticalKVCacheTransferPredictor(
            AnalyticalKVCacheTransferConfig(kv_connector="nixl"),
        ),
        available_clusters={ClusterType.PREFILL, ClusterType.DECODE},
    )


def test_measured_nixl_uses_covered_read_cost_and_prompt_tail_in_existing_owners(monkeypatch):
    def no_analytical_fallback(*args, **kwargs):
        raise AssertionError("Measured NIXL must not price analytical bandwidth")

    monkeypatch.setattr(AnalyticalKVCacheTransferPredictor, "get_transfer_time", no_analytical_fallback)
    costs = _FixtureCosts()
    source_cluster = _measured_cluster(ClusterType.PREFILL, costs)
    target_cluster = _measured_cluster(ClusterType.DECODE, costs)
    scheduler = _GlobalScheduler(source_cluster, target_cluster)
    source = source_cluster.get_replica_scheduler(source_cluster._serving_replica_id, 0)
    request = Request(0, 4105, 9)
    source.add_request(request)
    from frontier.events.replica_schedule_event import ReplicaScheduleEvent
    execution = _PDExecution(scheduler)
    execution.add([ReplicaScheduleEvent(0.1, source._replica_id, ClusterType.PREFILL, 0)])
    execution.until(lambda: len(execution.batches) == 1)
    prefill = execution.batches[0][0]
    assert prefill.num_tokens == [4080]
    execution.until(lambda: len(execution.metrics.outputs) == 1)
    assert request.num_emitted_decode_tokens == 0
    metrics = execution.metrics
    execution.until(lambda: len(metrics.starts) == 1)
    [end] = [event for _, event in execution.queue if hasattr(event, '_transfer_info')]
    info = end._transfer_info
    target = target_cluster.get_replica_scheduler(
        info.target_replica_id, info.target_replica_local_id,
    )

    assert info.transfer_time_ms == 3.5
    # 4080 prefix tokens: 32 live SWA pages and 255 sparse/summary pages.
    assert info.kv_cache_size_bytes == 102_203_904
    assert info.target_reserved_common_blocks == 1020
    assert target.on_schedule(end.time - 0.001) == []
    assert request.id in source._allocation_map
    assert request.num_emitted_decode_tokens == 0
    execution.until(lambda: len(metrics.ends) == 1)
    assert info.target_ready_time == info.source_release_time == end.time
    assert source._allocation_map == {}
    _, deferred, widths = target._schedule_decode_waiting_requests(16)
    assert deferred == widths == []
    assert request.id in target._kv_transfer_ready_request_ids
    assert request.num_emitted_decode_tokens == 0

    execution.until(lambda: len(execution.batches) == 2)
    first = execution.batches[1][0]
    assert first.num_tokens == [25]
    assert first.num_context_tokens == [4080]
    assert first.request_is_decoding == [False]
    assert first.get_decode_cuda_graph_runtime_mode() == "NONE"
    assert first.cache_block_zeroing is None
    execution.until(lambda: request.num_emitted_decode_tokens == 1)
    assert request.num_emitted_decode_tokens == 1
    assert request.first_decode_token_completed_at == execution.metrics.outputs[-1][0]
    assert info.to_dict()["num_computed_tokens"] == 4080
    for step in range(8):
        execution.until(lambda: request.num_emitted_decode_tokens == step + 2)
        _, tokens, context = execution.batches[step + 2]
        assert tokens == [1]
        assert context == [4105 + step]
    execution.drain(ClusterType.DECODE)
    assert request.completed and request.num_emitted_decode_tokens == 9
    assert target._allocation_map == {}
    assert target_cluster._load_balancer.request_owners == {}
    assert len(metrics.starts) == len(metrics.ends) == 1
    assert costs.reads


def test_measured_constructor_rejects_a_provider_without_coverage():
    with pytest.raises(ValueError, match="measured NIXL coverage"):
        _measured_cluster(ClusterType.PREFILL, object())


@pytest.mark.parametrize("changes", [
    {"num_prefill_tokens": 4106}, {"num_computed_tokens": 4104},
    {"num_computed_tokens": True}, {"source_attention_tp": 4},
    {"target_attention_tp": 4}, {"source_device": "h200"},
    {"target_device": "h200"}, {"profile_identity": ""},
    {"duration_scope": "read_network_only"},
])
def test_measured_coverage_rejects_wrong_geometry_or_incomplete_readiness(
    changes, step_config, capacity,
):
    costs = _FixtureCosts(changes=changes)
    predictor = AnalyticalKVCacheTransferPredictor(AnalyticalKVCacheTransferConfig(kv_connector="nixl"))
    with pytest.raises(ValueError, match="Measured NIXL"):
        predictor.get_measured_nixl_coverage(
            request=Request(0, 4105, 9), source_replica_config=step_config,
            target_replica_config=step_config, cache_capacity=capacity,
            cost_provider=costs, target_cost_provider=costs,
        )


def test_measured_missing_cost_unwinds_reservation_and_frontend_owner():
    costs = _FixtureCosts(duration=None)
    source_cluster = _measured_cluster(ClusterType.PREFILL, costs)
    target_cluster = _measured_cluster(ClusterType.DECODE, costs)
    source = source_cluster.get_replica_scheduler(source_cluster._serving_replica_id, 0)
    request = _request_owned_by_prefill(source, 4105)
    metrics = _Metrics()
    with pytest.raises(ValueError, match="READ cost"):
        _start_event(request, source).handle_event(_GlobalScheduler(source_cluster, target_cluster), metrics)
    assert source._kv_transfer_start_claimed_request_ids == set()
    assert request.id in source._allocation_map
    assert target_cluster._load_balancer.request_owners == {}
    for lane in target_cluster._replica_schedulers.values():
        assert lane._allocation_map == {}
        assert lane._scheduled_num_computed_tokens_by_request == {}
        assert lane._kv_transfer_receiving_request_ids == set()
    assert metrics.starts == metrics.ends == []


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), True, "3.5"])
def test_measured_cost_rejects_invalid_or_missing_samples(duration, step_config, capacity):
    costs = _FixtureCosts(duration=duration)
    predictor = AnalyticalKVCacheTransferPredictor(AnalyticalKVCacheTransferConfig(kv_connector="nixl"))
    request = Request(0, 4105, 9)
    coverage = predictor.get_measured_nixl_coverage(
        request=request, source_replica_config=step_config,
        target_replica_config=step_config, cache_capacity=capacity,
        cost_provider=costs, target_cost_provider=costs,
    )
    with pytest.raises(ValueError, match="READ cost"):
        predictor.get_measured_nixl_transfer_info_for_request(
            request=request, replica_config=step_config, cache_capacity=capacity,
            source_prompt_common_blocks=257, target_reserved_common_blocks=1020,
            coverage=coverage, cost_provider=costs,
        )


def test_measured_missing_request_coverage_rejects_before_placement():
    costs = _FixtureCosts(covered=False)
    source_cluster = _measured_cluster(ClusterType.PREFILL, costs)
    target_cluster = _measured_cluster(ClusterType.DECODE, costs)
    source = source_cluster.get_replica_scheduler(source_cluster._serving_replica_id, 0)
    request = _request_owned_by_prefill(source, 4105)
    with pytest.raises(ValueError, match="Missing measured NIXL coverage"):
        _start_event(request, source).handle_event(_GlobalScheduler(source_cluster, target_cluster), _Metrics())
    assert target_cluster._load_balancer.request_owners == {}
    assert source._kv_transfer_start_claimed_request_ids == set()
    assert all(not lane._allocation_map for lane in target_cluster._replica_schedulers.values())


def test_uncovered_first_d_budget_rejects_before_source_claim_or_placement():
    costs = _FixtureCosts()
    source_cluster = _measured_cluster(ClusterType.PREFILL, costs)
    target_cluster = _measured_cluster(ClusterType.DECODE, costs, max_tokens=16)
    source = source_cluster.get_replica_scheduler(source_cluster._serving_replica_id, 0)
    request = _request_owned_by_prefill(source, 4105)
    with pytest.raises(ValueError, match="D token budget"):
        _start_event(request, source).handle_event(_GlobalScheduler(source_cluster, target_cluster), _Metrics())
    assert source._kv_transfer_start_claimed_request_ids == set()
    assert target_cluster._load_balancer.request_owners == {}
    assert all(not lane._allocation_map for lane in target_cluster._replica_schedulers.values())


def test_remote_prefix_uses_existing_allocation_plan_without_a_query(capacity):
    from frontier.kv_cache.base_kv_cache_manager import KVCacheManager

    manager = KVCacheManager(16, 64, False, "sha256", 0, cache_capacity=capacity)
    request = Request(0, 128, 9)
    before = manager.block_pool.get_num_free_blocks()
    with pytest.raises(ValueError, match="positive"):
        manager.allocate_slots(request, 0)
    with pytest.raises(ValueError, match="positive"):
        manager.allocate_slots(request, 0, scheduler_num_computed_tokens=0)
    with pytest.raises(ValueError, match="positive"):
        manager.allocate_slots(request, False, scheduler_num_computed_tokens=112)
    result = manager.allocate_slots(request, 0, scheduler_num_computed_tokens=112)
    assert result is not None
    assert manager.get_num_blocks_for_request(request) == 28
    assert manager.block_pool.get_num_free_blocks() == before - 28
    assert request.num_processed_tokens == request.num_emitted_decode_tokens == 0
    manager.free(request)
    assert manager.block_pool.get_num_free_blocks() == before


def test_shared_legacy_glm_nixl_keeps_non_dummy_full_prompt_contract():
    from tests.unit.test_glm53_structural_pd import (
        _harness as glm_harness, _request_owned_by_prefill as glm_request,
        _start_event as glm_start,
    )

    source, target, scheduler, predictor = glm_harness()
    scheduler.get_cluster_scheduler(ClusterType.PREFILL)._config.execution_time_predictor_config.enable_dummy_mode = False
    request = glm_request(source, 128)
    [end] = glm_start(request, source).handle_event(scheduler, _Metrics())
    info = end._transfer_info
    assert info.nixl_coverage is None
    assert info.source_prompt_common_blocks == info.target_reserved_common_blocks == 2
    assert info.kv_cache_size_bytes == 2 * source._kv_cache_capacity.resident_bytes_per_common_block
    assert info.transfer_time_ms == predictor.get_transfer_time(
        ClusterType.PREFILL, ClusterType.DECODE, None, info.kv_cache_size_bytes,
    )
    end.handle_event(scheduler, _Metrics())
    for step in range(3):
        now = end.time + (step + 1) * 0.01
        [batch] = target.on_schedule(now)
        assert batch.num_tokens == [1]
        assert batch.num_context_tokens == [127 + step]
        batch.on_batch_end(now + 0.005, ClusterType.DECODE)
        target.on_batch_end(batch)
    assert request.completed and request.num_emitted_decode_tokens == 3
    assert source._allocation_map == target._allocation_map == {}


@pytest.mark.parametrize("changed", [False, True])
def test_capacity_retry_rejects_changed_coverage_and_releases_reserved_owners(monkeypatch, changed):
    costs = _FixtureCosts()
    source_cluster = _measured_cluster(ClusterType.PREFILL, costs)
    target_cluster = _measured_cluster(ClusterType.DECODE, costs)
    scheduler = _GlobalScheduler(source_cluster, target_cluster)
    source = source_cluster.get_replica_scheduler(source_cluster._serving_replica_id, 0)
    target = target_cluster.get_replica_scheduler(target_cluster._serving_replica_id, 0)
    request = _request_owned_by_prefill(source, 4105)
    execution = _PDExecution(scheduler)
    metrics = execution.metrics
    start = _start_event(request, source)
    allocate = target._can_allocate_request
    monkeypatch.setattr(target, "_can_allocate_request", lambda *args, **kwargs: False)
    [wake] = start.handle_event(scheduler, metrics)
    from frontier.events.replica_schedule_event import ReplicaScheduleEvent

    assert isinstance(wake, ReplicaScheduleEvent)
    execution.add([wake])
    monkeypatch.setattr(target, "_can_allocate_request", allocate)
    target._retry_pending_kv_transfer_starts(2)
    [retry] = target.consume_popped_output_events()
    assert target._allocation_map[request.id] == 1020
    assert target._scheduled_num_computed_tokens_by_request[request.id] == 4080
    if not changed:
        execution.add([retry])
        for step in range(9):
            execution.until(lambda: request.num_emitted_decode_tokens == step + 1)
            assert execution.batches[step][1] == [25 if step == 0 else 1]
        execution.drain(ClusterType.DECODE)
        assert request.completed and request.num_emitted_decode_tokens == 9
        assert source._allocation_map == target._allocation_map == {}
        assert target_cluster._load_balancer.request_owners == {}
        assert len(metrics.starts) == len(metrics.ends) == 1
        return
    costs.changes = {"num_computed_tokens": 4096}
    with pytest.raises(RuntimeError, match="coverage changed"):
        retry.handle_event(scheduler, metrics)
    assert retry._nixl_coverage.num_computed_tokens == 4080
    assert source._kv_transfer_start_claimed_request_ids == set()
    assert request.id in source._pending_kv_transfer_requests
    assert target_cluster._load_balancer.request_owners == {}
    assert target._allocation_map == target._scheduled_num_computed_tokens_by_request == {}
    assert target._kv_transfer_receiving_request_ids == set()
    assert metrics.starts == metrics.ends == []


@pytest.mark.parametrize("mutation", ["missing_cost", "missing_coverage", "wrong_duration"])
def test_measured_completion_failure_releases_transfer_owners_without_ready(mutation):
    costs = _FixtureCosts()
    source_cluster = _measured_cluster(ClusterType.PREFILL, costs)
    target_cluster = _measured_cluster(ClusterType.DECODE, costs)
    scheduler = _GlobalScheduler(source_cluster, target_cluster)
    source = source_cluster.get_replica_scheduler(source_cluster._serving_replica_id, 0)
    request = _request_owned_by_prefill(source, 4105)
    metrics = _Metrics()
    [end] = _start_event(request, source).handle_event(scheduler, metrics)
    info = end._transfer_info
    target = target_cluster.get_replica_scheduler(info.target_replica_id, info.target_replica_local_id)
    if mutation == "missing_cost":
        costs.duration = None
    elif mutation == "missing_coverage":
        costs.covered = False
    else:
        info.transfer_time_ms += 1
    with pytest.raises((ValueError, RuntimeError)):
        end.handle_event(scheduler, metrics)
    assert source._kv_transfer_start_claimed_request_ids == set()
    assert request.id in source._pending_kv_transfer_requests
    assert request.id in source._allocation_map
    assert target_cluster._load_balancer.request_owners == {}
    assert target._allocation_map == target._scheduled_num_computed_tokens_by_request == {}
    assert target._kv_transfer_receiving_request_ids == target._kv_transfer_ready_request_ids == set()
    assert info.target_ready_time is info.source_release_time is None
    assert len(metrics.starts) == 1 and metrics.ends == []
    assert request.num_emitted_decode_tokens == 0
