"""Real vLLM-v1 scheduler ownership tests for fixed GDN state slots."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.config import (
    BaseModelConfig,
    FixedRequestLengthGeneratorConfig,
    PoissonRequestIntervalGeneratorConfig,
    ReplicaConfig,
    SyntheticRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.entities import Replica, Request
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    StageExecutionContext,
)
from frontier.scheduler.utils.memory_planner import MemoryPlanner
from frontier.types import ActivationType, ClusterType, NormType


class _ClusterScheduler:
    def __init__(self, replica_id: int, num_stages: int) -> None:
        self._contexts = {
            (replica_id, stage_id): StageExecutionContext(
                replica_id=replica_id,
                stage_id=stage_id,
                ep_size=1,
            )
            for stage_id in range(num_stages)
        }

    def get_stage_execution_context(self, replica_id: int, stage_id: int):
        return self._contexts[(replica_id, stage_id)]


def _build_scheduler(*, capacity: int) -> VLLMv1EngineReplicaScheduler:
    replica_config = ReplicaConfig(
        model_name="Qwen3.8-2.4T-A95B-Quark-MXFP4",
        device="a100",
        network_device="a100_pairwise_nvlink",
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        memory_margin_fraction=0.1,
    )
    request_generator_config = SyntheticRequestGeneratorConfig(
        num_requests=2,
        length_generator_config=FixedRequestLengthGeneratorConfig(
            prefill_tokens=8,
            decode_tokens=2,
        ),
        interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=1.0),
    )
    replica = Replica(
        replica_config,
        request_generator_config,
        ClusterType.MONOLITHIC,
    )
    scheduler_config = VllmV1SchedulerConfig(
        num_blocks=100,
        block_size=16,
        batch_size_cap=capacity,
        max_tokens_in_batch=4,
        enable_chunked_prefill=True,
    )
    return VLLMv1EngineReplicaScheduler(
        replica_config=replica_config,
        replica_scheduler_config=scheduler_config,
        request_generator_config=request_generator_config,
        replica=replica,
        predictor=object(),
        cluster_scheduler=_ClusterScheduler(replica.id, replica.num_pipeline_stages),
        cluster_type=ClusterType.MONOLITHIC,
    )


def _request() -> Request:
    return Request(arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=2)


def _tiny_hybrid_model() -> BaseModelConfig:
    return BaseModelConfig(
        num_layers=4,
        num_q_heads=4,
        num_kv_heads=2,
        embedding_dim=256,
        mlp_hidden_dim=64,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=1024,
        model_type="qwen3_5_moe_text",
        model_architecture_profile="qwen3_5_moe",
        architectures=("Qwen3_5MoeForCausalLM",),
        is_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        full_attention_interval=2,
    )


@pytest.mark.parametrize("num_blocks_mode", ["memory_planner", "memory_planner_profiled"])
def test_real_scheduler_automatic_gdn_reservation_uses_effective_capacity(
    monkeypatch: pytest.MonkeyPatch, num_blocks_mode: str
) -> None:
    model = _tiny_hybrid_model()
    original_create_from_name = BaseModelConfig.create_from_name

    def create_fixture_model(cls, name):
        if name == "tiny-gdn-scheduler":
            return model
        return original_create_from_name(name)

    monkeypatch.setattr(
        BaseModelConfig,
        "create_from_name",
        classmethod(create_fixture_model),
    )
    replica_config = ReplicaConfig(
        model_name="tiny-gdn-scheduler",
        device="a100",
        network_device="a100_pairwise_nvlink",
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        memory_margin_fraction=0.1,
    )
    request_generator_config = SyntheticRequestGeneratorConfig(
        num_requests=1,
        length_generator_config=FixedRequestLengthGeneratorConfig(
            prefill_tokens=8,
            decode_tokens=2,
        ),
        interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=1.0),
    )
    replica = Replica(replica_config, request_generator_config, ClusterType.MONOLITHIC)
    expected_planner = MemoryPlanner(
        replica_config,
        replica,
        ClusterType.MONOLITHIC,
        max_num_seqs=2,
    )
    expected_blocks = expected_planner.get_num_blocks(
        block_size=16,
        gpu_memory_utilization=0.5,
    )
    expected_state_reservation = (
        expected_planner.get_gdn_state_memory_per_device_per_request_bytes() * 2
    )
    assert expected_state_reservation > 0

    scheduler_config = VllmV1SchedulerConfig(
        num_blocks=0,
        num_blocks_mode=num_blocks_mode,
        gpu_memory_utilization=0.5,
        batch_size_cap=2,
        max_tokens_in_batch=8,
    )
    scheduler = VLLMv1EngineReplicaScheduler(
        replica_config=replica_config,
        replica_scheduler_config=scheduler_config,
        request_generator_config=request_generator_config,
        replica=replica,
        predictor=object(),
        cluster_scheduler=_ClusterScheduler(replica.id, replica.num_pipeline_stages),
        cluster_type=ClusterType.MONOLITHIC,
    )

    planner_state = scheduler._gdn_state_slot_manager
    assert planner_state.capacity == 2
    assert scheduler._config.num_blocks == expected_blocks


def test_gdn_capacity_changes_fixed_reservation_without_changing_kv_denominator() -> None:
    model = _tiny_hybrid_model()
    replica_config = ReplicaConfig(
        model_name="Qwen3.8-2.4T-A95B-Quark-MXFP4",
        device="a100",
        network_device="a100_pairwise_nvlink",
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
    )
    replica_config.model_config = model
    request_generator_config = SyntheticRequestGeneratorConfig(
        num_requests=1,
        length_generator_config=FixedRequestLengthGeneratorConfig(
            prefill_tokens=8,
            decode_tokens=2,
        ),
        interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=1.0),
    )
    replica = Replica(replica_config, request_generator_config, ClusterType.MONOLITHIC)
    planners = [
        MemoryPlanner(replica_config, replica, ClusterType.MONOLITHIC, max_num_seqs=cap)
        for cap in (1, 3)
    ]
    reservations = [
        planner.get_gdn_state_memory_per_device_per_request_bytes() * cap
        for planner, cap in zip(planners, (1, 3))
    ]
    assert reservations[1] == reservations[0] * 3
    assert [planner._get_num_full_attention_layers_per_device() for planner in planners] == [2, 2]


def test_effective_memory_planner_capacity_includes_phase_specific_caps() -> None:
    config = SimpleNamespace(
        batch_size_cap=2,
        hidden_phase_batch_size_cap=5,
        final_phase_batch_size_cap=3,
    )
    assert (
        VLLMv1EngineReplicaScheduler._get_memory_planner_max_num_seqs(config) == 5
    )


def test_non_gdn_scheduler_does_not_create_state_slot_manager() -> None:
    replica_config = ReplicaConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        device="a100",
        network_device="a100_pairwise_nvlink",
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        memory_margin_fraction=0.1,
    )
    request_generator_config = SyntheticRequestGeneratorConfig(
        num_requests=1,
        length_generator_config=FixedRequestLengthGeneratorConfig(
            prefill_tokens=8,
            decode_tokens=2,
        ),
        interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=1.0),
    )
    replica = Replica(replica_config, request_generator_config, ClusterType.MONOLITHIC)
    scheduler = VLLMv1EngineReplicaScheduler(
        replica_config=replica_config,
        replica_scheduler_config=VllmV1SchedulerConfig(
            num_blocks=100,
            block_size=16,
            batch_size_cap=1,
            max_tokens_in_batch=8,
        ),
        request_generator_config=request_generator_config,
        replica=replica,
        predictor=object(),
        cluster_scheduler=_ClusterScheduler(replica.id, replica.num_pipeline_stages),
        cluster_type=ClusterType.MONOLITHIC,
    )
    assert scheduler._gdn_state_slot_manager is None


def test_real_scheduler_admission_owns_and_releases_gdn_slots() -> None:
    scheduler = _build_scheduler(capacity=2)
    first, second = _request(), _request()

    scheduler.add_request(first)
    scheduler.add_request(second)
    _, scheduled, _ = scheduler._schedule_waiting_requests(16)

    assert [request.id for request in scheduled] == [first.id, second.id]
    manager = scheduler._gdn_state_slot_manager
    assert manager.capacity == 2
    assert manager.retain(first.id).slot_id == 0
    assert manager.retain(second.id).slot_id == 1
    assert set(manager.active_request_ids) == {first.id, second.id}

    scheduler._free_request_resources(first)
    assert not manager.has_slot(first.id)
    assert manager.retain(second.id).slot_id == 1
    scheduler._free_request_resources(first)
    assert not manager.has_slot(first.id)


def test_idempotent_cleanup_releases_slot_after_request_leaves_scheduler_queues() -> None:
    scheduler = _build_scheduler(capacity=1)
    request = _request()
    scheduler.add_request(request)
    _, scheduled, _ = scheduler._schedule_waiting_requests(8)
    assert scheduled == [request]
    scheduler._running_requests.remove(request)
    scheduler._free_request_resources_by_id(request.id)
    assert scheduler._gdn_state_slot_manager.active_request_ids == ()


def test_real_scheduler_full_slot_pool_keeps_new_request_waiting() -> None:
    scheduler = _build_scheduler(capacity=1)
    first, second = _request(), _request()
    scheduler.add_request(first)
    scheduler.add_request(second)

    _, scheduled, _ = scheduler._schedule_waiting_requests(8)

    assert [request.id for request in scheduled] == [first.id]
    assert scheduler.peek_waiting_requests() == [second]
    assert scheduler._gdn_state_slot_manager.active_request_ids == (first.id,)


def test_admitted_owner_keeps_slot_across_two_waiting_resumptions() -> None:
    scheduler = _build_scheduler(capacity=1)
    request = _request()
    scheduler.add_request(request)
    _, scheduled, _ = scheduler._schedule_waiting_requests(8)
    assert scheduled == [request]
    slot_id = scheduler._gdn_state_slot_manager.retain(request.id).slot_id

    for expected_frontier in (0, 4):
        scheduler._running_requests.remove(request)
        scheduler._request_queue.append(request)
        scheduler._scheduled_num_computed_tokens_by_request[request.id] = (
            expected_frontier
        )
        _, resumed, _ = scheduler._schedule_waiting_requests(4)
        assert resumed == [request]
        assert scheduler._gdn_state_slot_manager.resume(request.id).slot_id == slot_id


def test_gdn_slot_allocation_rolls_back_kv_blocks_on_slot_failure() -> None:
    scheduler = _build_scheduler(capacity=1)
    request = _request()

    def fail_allocate(_request_id):
        raise MemoryError("injected slot allocation failure")

    scheduler._gdn_state_slot_manager.allocate = fail_allocate
    with pytest.raises(MemoryError, match="injected"):
        scheduler._allocate_request(request, num_new_tokens=4)

    assert request.id not in scheduler._allocation_map
    assert scheduler.num_allocated_blocks == 0
    assert scheduler._gdn_state_slot_manager.active_request_ids == ()


def test_gdn_preemption_rejects_before_slot_or_kv_mutation() -> None:
    scheduler = _build_scheduler(capacity=1)
    request = _request()
    scheduler.add_request(request)
    _, scheduled, _ = scheduler._schedule_waiting_requests(8)
    assert scheduled == [request]
    before_allocations = dict(scheduler._allocation_map)
    before_slots = scheduler._gdn_state_slot_manager.active_request_ids

    with pytest.raises(ValueError, match="preemption"):
        scheduler._preempt_request(request, [])

    assert scheduler._allocation_map == before_allocations
    assert scheduler._gdn_state_slot_manager.active_request_ids == before_slots
