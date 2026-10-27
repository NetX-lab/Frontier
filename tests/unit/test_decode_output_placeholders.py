"""Unified DECODE batch queue depth 2 with vLLM AsyncScheduler output placeholders."""

from types import SimpleNamespace

import pytest

from frontier.config import (
    ClusterConfig,
    FixedRequestLengthGeneratorConfig,
    PoissonRequestIntervalGeneratorConfig,
    ReplicaConfig,
    Sj2qFastserveLiteSchedulerConfig,
    SpeculativeDecodingConfig,
    SyntheticRequestGeneratorConfig,
    VllmV1SchedulerConfig,
    global_vars,
)
from frontier.entities import Replica, Request
from frontier.scheduler.replica_scheduler.sj2q_fastserve_lite_replica_scheduler import (
    SJ2QFastServeLiteReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    StageExecutionContext,
)
from frontier.scheduler.utils.replica_config import resolve_replica_scheduler_config
from frontier.types import ClusterType


class _ClusterScheduler:
    def __init__(self, replica_id: int, ep_size: int) -> None:
        self._config = SimpleNamespace(decode_attn_micro_batch_size=4)
        self._context = StageExecutionContext(
            replica_id=replica_id, stage_id=0, ep_size=ep_size
        )
        self.lanes = []

    def get_stage_execution_context(self, replica_id: int, stage_id: int):
        return self._context

    def get_replica_scheduler(self, replica_id: int, lane_id: int):
        return self.lanes[lane_id]

    def on_replica_batch_scheduled(self, time, replica_id, lane_id, batch) -> None:
        assert batch.scheduled_at == time


def _dense_replica_config(**fields) -> ReplicaConfig:
    options = dict(
        model_name="meta-llama/Llama-2-7b-hf",
        device="a100",
        network_device="a100_pairwise_nvlink",
    )
    options.update(fields)
    return ReplicaConfig(**options)


def _decode_lane(
    scheduler_class=VLLMv1EngineReplicaScheduler,
    config_class=VllmV1SchedulerConfig,
    *,
    replica_fields=None,
    cluster_type=ClusterType.DECODE,
    **config_fields,
):
    """A dense DP1 lane whose engine runs the non-DP batch-queue loop."""
    replica_config = _dense_replica_config(**(replica_fields or {}))
    request_config = SyntheticRequestGeneratorConfig(
        num_requests=1,
        length_generator_config=FixedRequestLengthGeneratorConfig(
            prefill_tokens=16, decode_tokens=1, max_tokens=64
        ),
        interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=1.0),
    )
    replica = Replica(replica_config, request_config, cluster_type)
    scheduler_fields = dict(
        num_blocks=100,
        block_size=16,
        batch_size_cap=4,
        max_tokens_in_batch=256,
        enable_chunked_prefill=True,
        max_concurrent_batches=2,
    )
    scheduler_fields.update(config_fields)
    return scheduler_class(
        replica_config=replica_config,
        replica_scheduler_config=config_class(**scheduler_fields),
        request_generator_config=request_config,
        replica=replica,
        predictor=object(),
        cluster_scheduler=_ClusterScheduler(replica.id, ep_size=1),
        cluster_type=cluster_type,
    )


def _step5_dp_lanes(monkeypatch, *, max_concurrent_batches=2):
    """Two attention-DP lanes of a Step-5 NIXL DECODE Replica with grouped KV allocation."""
    monkeypatch.setattr(global_vars, "KV_CONNECTOR", "nixl")
    replica_config = ReplicaConfig(
        model_name="stepfun-ai/Step-5-Preview",
        device="h200",
        network_device="h200_dgx",
        attn_tensor_parallel_size=8,
        attn_dp=2,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=16,
        attention_backend="STEP5_DSA",
        kv_cache_dtype="auto",
    )
    request_config = SyntheticRequestGeneratorConfig()
    replica = Replica(replica_config, request_config, ClusterType.DECODE)
    cluster = _ClusterScheduler(replica.id, ep_size=16)
    for lane_id in range(2):
        cluster.lanes.append(VLLMv1EngineReplicaScheduler(
            replica_config=replica_config,
            replica_scheduler_config=VllmV1SchedulerConfig(
                max_concurrent_batches=max_concurrent_batches,
                num_blocks=0,
                num_blocks_mode="memory_planner",
                enable_runtime_non_kv_cache_overhead_profiling=False,
                block_size=256,
                max_tokens_in_batch=192,
                enable_chunked_prefill=True,
                max_model_len=2048,
                batch_size_cap=8,
            ),
            request_generator_config=request_config,
            replica=replica,
            predictor=object(),
            cluster_scheduler=cluster,
            cluster_type=ClusterType.DECODE,
            replica_local_id=lane_id,
        ))
    return cluster.lanes


def _complete_prefill(request: Request) -> Request:
    request._num_processed_tokens = request.num_prefill_tokens
    request._is_prefill_complete = True
    request._prefill_completed_at = 0.0
    return request


def _prefilled(prompt_tokens: int, decode_tokens: int) -> Request:
    return _complete_prefill(Request(0.0, prompt_tokens, decode_tokens))


def _add_decoding_request(lane, prompt_tokens: int, decode_tokens: int) -> Request:
    """Hand a prefilled request to a DECODE lane that already holds its prompt blocks."""
    request = Request(0.0, prompt_tokens, decode_tokens)
    lane._allocate_request(request, prompt_tokens)
    lane.add_request(_complete_prefill(request))
    return request


def _end(lane, batch, time: float) -> None:
    """Apply a batch's output as GlobalBatchEndEvent does, then pop it from the engine."""
    batch.apply_preempted_step_samples(time, ClusterType.DECODE)
    batch.on_batch_end(time, ClusterType.DECODE)
    lane.on_batch_end(batch)


def _rows(batch):
    return [
        (request.id, num_tokens, num_context_tokens)
        for request, num_tokens, num_context_tokens in zip(
            batch.requests, batch.num_tokens, batch.num_context_tokens
        )
    ]


def test_decode_role_override_reaches_only_the_decode_scheduler():
    cluster = ClusterConfig(
        replica_config=_dense_replica_config(),
        replica_scheduler_config=VllmV1SchedulerConfig(),
        prefill_cluster_num_replicas=1,
        decode_cluster_num_replicas=1,
        decode_replica_scheduler_config_max_concurrent_batches=2,
    )
    roles = cluster.get_cluster_configs_for_disaggregation()

    decode = resolve_replica_scheduler_config(roles[ClusterType.DECODE], ClusterType.DECODE)
    prefill = resolve_replica_scheduler_config(roles[ClusterType.PREFILL], ClusterType.PREFILL)

    assert decode.max_concurrent_batches == 2
    assert prefill.max_concurrent_batches is None


def test_only_an_explicit_decode_capacity_uses_output_placeholders():
    assert _decode_lane()._uses_output_placeholders
    assert not _decode_lane(max_concurrent_batches=None)._uses_output_placeholders
    assert not _decode_lane(
        cluster_type=ClusterType.PREFILL, max_concurrent_batches=None
    )._uses_output_placeholders
    with pytest.raises(NotImplementedError, match="Explicit PREFILL queue capacity requires"):
        _decode_lane(cluster_type=ClusterType.PREFILL)


@pytest.mark.parametrize("scheduler_class,config_class,replica_fields,config_fields", [
    (VLLMv1EngineReplicaScheduler, VllmV1SchedulerConfig, {"num_pipeline_stages": 2}, {}),
    (VLLMv1EngineReplicaScheduler, VllmV1SchedulerConfig, {}, {"scheduling_policy": "priority"}),
    (
        VLLMv1EngineReplicaScheduler, VllmV1SchedulerConfig,
        {"speculative_decoding_config": SpeculativeDecodingConfig(enabled=True)}, {},
    ),
    (
        VLLMv1EngineReplicaScheduler, VllmV1SchedulerConfig,
        {"model_name": "Qwen3.8-2.4T-A95B-Quark-MXFP4"}, {},
    ),
    (SJ2QFastServeLiteReplicaScheduler, Sj2qFastserveLiteSchedulerConfig, {}, {}),
])
def test_explicit_decode_capacity_rejects_unsupported_scopes(
    scheduler_class, config_class, replica_fields, config_fields
):
    with pytest.raises(NotImplementedError, match="Explicit DECODE queue capacity requires"):
        _decode_lane(
            scheduler_class, config_class, replica_fields=replica_fields, **config_fields
        )


def test_in_flight_request_is_scheduled_again_until_its_last_token_is_in_flight():
    lane = _decode_lane()
    first, second = (_add_decoding_request(lane, 16, 3) for _ in range(2))

    # With the queue empty, two schedules fill it; the second schedules each
    # request again on its output placeholder, one context token further.
    older, newer = lane.on_schedule(1.0)
    assert _rows(older) == [(first.id, 1, 16), (second.id, 1, 16)]
    assert _rows(newer) == [(first.id, 1, 17), (second.id, 1, 17)]
    assert newer.request_mutation_signatures == [(0, 0, 0, 2)] * 2

    # The older output is applied, not dropped as stale, and frees one slot.
    _end(lane, older, 2.0)
    assert (first.num_processed_tokens, second.num_processed_tokens) == (17, 17)
    [third] = lane.on_schedule(2.0)
    assert _rows(third) == [(first.id, 1, 18), (second.id, 1, 18)]

    # Each request's last output token is now in flight.
    _end(lane, newer, 3.0)
    assert lane.on_schedule(3.0) == []
    _end(lane, third, 4.0)
    assert first.completed and second.completed
    assert first.num_processed_tokens == second.num_processed_tokens == 19
    assert not lane.has_unfinished_requests


def test_preemption_victim_with_a_step_in_flight_still_takes_that_step_sample():
    # Three blocks of 16 tokens: the first step fits both 15-token prompts,
    # the second needs one more block per request.
    lane = _decode_lane(num_blocks=3, enable_preemption=True)
    head, tail = (_add_decoding_request(lane, 15, 3) for _ in range(2))

    older, newer = lane.on_schedule(1.0)
    assert _rows(older) == [(head.id, 1, 15), (tail.id, 1, 15)]
    # FCFS preempts the running tail, which the second schedule had not reached.
    assert _rows(newer) == [(head.id, 1, 16)]
    assert [request.id for request in lane._waiting_requests] == [tail.id]
    assert tail.has_preempted_step

    _end(lane, older, 2.0)
    assert not tail.has_preempted_step
    assert (head.num_processed_tokens, tail.num_processed_tokens) == (16, 16)

    pending = [newer] + lane.on_schedule(2.0)
    time = 2.0
    while pending:
        time += 1.0
        _end(lane, pending.pop(0), time)
        pending.extend(lane.on_schedule(time))
    assert head.completed and tail.completed
    assert head.num_processed_tokens == tail.num_processed_tokens == 18
    assert lane._allocation_map == {}


def test_dp_lane_with_grouped_allocation_runs_two_steps_of_one_request(monkeypatch):
    lane, sibling = _step5_dp_lanes(monkeypatch)
    assert lane._runs_dp_dummy_passes and lane._is_grouped_kv_allocation()
    requests = []
    for _ in range(2):
        request = _prefilled(100, 3)
        assert lane.reserve_kv_transfer_target(request) is not None
        lane.mark_kv_transfer_target_ready(request)
        lane.add_request(request)
        requests.append(request)

    # One batch per busy-loop iteration. NIXL D recomputes the last prompt
    # token first. A Step engine waits at the DP sync after the first
    # iteration of a wave, until the idle sibling's dummy forward joins it.
    [older] = lane.on_schedule(1.0)
    assert [row[1:] for row in _rows(older)] == [(1, 99), (1, 99)]
    assert lane._waits_at_dp_sync
    [dummy_forward] = sibling.on_schedule(1.0)
    assert sibling.on_dummy_forward_end(1.5)
    sibling.on_schedule(1.5)
    assert sibling.consume_dp_sync_release() == [0] and not lane._waits_at_dp_sync

    # The second batch allocates its token while the first is in flight.
    [newer] = lane.on_schedule(1.5)
    assert [row[1:] for row in _rows(newer)] == [(1, 100), (1, 100)]
    assert lane._blocking_batch_id == older.id

    _end(lane, older, 2.0)
    [third] = lane.on_schedule(2.0)
    assert [row[1:] for row in _rows(third)] == [(1, 101), (1, 101)]
    _end(lane, newer, 3.0)
    assert lane.on_schedule(3.0) == []
    _end(lane, third, 4.0)
    assert all(request.completed for request in requests)
    assert lane._allocation_map == {}


def test_grouped_admission_bounds_the_tokens_of_both_in_flight_batches(monkeypatch):
    # The context exceeds the 512-token sliding window, so the in-flight tokens
    # bound the window groups' blocks.
    request = _prefilled(1500, 3)
    blocks = {}
    for capacity in (None, 2):
        lane, _ = _step5_dp_lanes(monkeypatch, max_concurrent_batches=capacity)
        blocks[capacity] = lane._get_grouped_admission_blocks(request)
        expected = lane._kv_cache_manager.get_num_admission_blocks(
            1502, max_model_len=2048, max_in_flight_tokens=(capacity or 1) * 192
        )
        assert blocks[capacity] == expected
    assert blocks[2] > blocks[None]
