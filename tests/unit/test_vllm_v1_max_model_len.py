"""The vLLM V1 schedulers take max_model_len from the model, as vLLM does.

``max_model_len`` defaults to the model's ``max_position_embeddings``; the
workload's length bound does not limit a request. A request, or a thinking
round when it is requeued, that does not fit is rejected on arrival with
vLLM's OpenAI-server rule: prompt < max_model_len and prompt + output <=
max_model_len.
"""

import pytest

from frontier.config import (
    FixedRequestLengthGeneratorConfig,
    PoissonRequestIntervalGeneratorConfig,
    ReplicaConfig,
    SglangSchedulerConfig,
    Sj2qBoundedCarryoverSchedulerConfig,
    Sj2qFastserveLiteSchedulerConfig,
    Sj2qPenaltyOnlySchedulerConfig,
    SyntheticRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.entities import Replica, Request
from frontier.entities.request import RequestRoundPlan
from frontier.scheduler.replica_scheduler.sglang_style_replica_scheduler import (
    SGLangStyleReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.sj2q_bounded_carryover_replica_scheduler import (
    SJ2QBoundedCarryoverReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.sj2q_fastserve_lite_replica_scheduler import (
    SJ2QFastServeLiteReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.sj2q_penalty_only_replica_scheduler import (
    SJ2QPenaltyOnlyReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    StageExecutionContext,
)
from frontier.types import ClusterType

MAX_MODEL_LEN = 64
WORKLOAD_MAX_TOKENS = 64

SCHEDULERS = [
    (VLLMv1EngineReplicaScheduler, VllmV1SchedulerConfig),
    (SGLangStyleReplicaScheduler, SglangSchedulerConfig),
    (SJ2QFastServeLiteReplicaScheduler, Sj2qFastserveLiteSchedulerConfig),
    (SJ2QPenaltyOnlyReplicaScheduler, Sj2qPenaltyOnlySchedulerConfig),
    (SJ2QBoundedCarryoverReplicaScheduler, Sj2qBoundedCarryoverSchedulerConfig),
]


class _ClusterScheduler:
    def __init__(self, replica_id: int, num_stages: int) -> None:
        self._contexts = {
            (replica_id, stage_id): StageExecutionContext(
                replica_id=replica_id, stage_id=stage_id, ep_size=1
            )
            for stage_id in range(num_stages)
        }

    def get_stage_execution_context(self, replica_id: int, stage_id: int):
        return self._contexts[(replica_id, stage_id)]


def _build_scheduler(
    scheduler_class=VLLMv1EngineReplicaScheduler,
    config_class=VllmV1SchedulerConfig,
    cluster_type=ClusterType.PREFILL,
    **config_fields,
):
    replica_config = ReplicaConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        device="a100",
        network_device="a100_pairwise_nvlink",
    )
    request_generator_config = SyntheticRequestGeneratorConfig(
        num_requests=1,
        length_generator_config=FixedRequestLengthGeneratorConfig(
            prefill_tokens=16, decode_tokens=1, max_tokens=WORKLOAD_MAX_TOKENS
        ),
        interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=1.0),
    )
    replica = Replica(replica_config, request_generator_config, cluster_type)
    return scheduler_class(
        replica_config=replica_config,
        replica_scheduler_config=config_class(
            num_blocks=100,
            block_size=16,
            batch_size_cap=4,
            max_tokens_in_batch=256,
            enable_chunked_prefill=True,
            **config_fields,
        ),
        request_generator_config=request_generator_config,
        replica=replica,
        predictor=object(),
        cluster_scheduler=_ClusterScheduler(replica.id, replica.num_pipeline_stages),
        cluster_type=cluster_type,
    )


def test_a_prompt_beyond_the_workload_bound_is_scheduled_whole() -> None:
    # Llama-2-7B's max_position_embeddings (4096) is the default max_model_len.
    scheduler = _build_scheduler()
    request = Request(0.0, 2 * WORKLOAD_MAX_TOKENS, 1)
    scheduler.add_request(request)

    batch = scheduler._schedule_prefill_only()

    assert dict(zip(batch.requests, batch.num_tokens)) == {request: 2 * WORKLOAD_MAX_TOKENS}


def test_decode_admits_a_prompt_beyond_the_workload_bound() -> None:
    scheduler = _build_scheduler(cluster_type=ClusterType.DECODE)
    request = Request(0.0, 2 * WORKLOAD_MAX_TOKENS, 4)
    # Prefill on the PREFILL role, then the KV arrival hands the request over.
    request.on_batch_schedule(0.0, ClusterType.PREFILL)
    request.on_batch_end(1.0, 2 * WORKLOAD_MAX_TOKENS, ClusterType.PREFILL)
    request.on_arrival(1.0, ClusterType.DECODE)
    scheduler.add_request(request)

    batch = scheduler._schedule_decode_only()

    assert dict(zip(batch.requests, batch.num_tokens)) == {request: 1}


@pytest.mark.parametrize("scheduler_class, config_class", SCHEDULERS)
@pytest.mark.parametrize(
    "round_plans",
    [
        # Only the prompt rule rejects a round without output tokens.
        [RequestRoundPlan(MAX_MODEL_LEN, 0), RequestRoundPlan(16, 4)],
        [RequestRoundPlan(MAX_MODEL_LEN - 8, 9)],
    ],
    ids=["prompt_reaches_max_model_len", "prompt_plus_output_exceeds_max_model_len"],
)
def test_a_request_that_does_not_fit_is_rejected_on_arrival(
    scheduler_class, config_class, round_plans
) -> None:
    scheduler = _build_scheduler(
        scheduler_class, config_class, max_model_len=MAX_MODEL_LEN
    )
    final_round = round_plans[-1]
    request = Request(
        0.0,
        final_round.num_prefill_tokens,
        final_round.num_decode_tokens,
        thinking_depth=len(round_plans),
        thinking_round_plans=round_plans,
    )

    with pytest.raises(ValueError, match=f"do not fit max_model_len={MAX_MODEL_LEN}"):
        scheduler.add_request(request)


def test_a_request_that_fills_max_model_len_exactly_is_accepted() -> None:
    scheduler = _build_scheduler(max_model_len=MAX_MODEL_LEN)
    request = Request(0.0, MAX_MODEL_LEN - 8, 8)

    scheduler.add_request(request)

    assert scheduler._request_queue == [request]


def test_a_requeued_thinking_round_that_does_not_fit_is_rejected() -> None:
    scheduler = _build_scheduler(max_model_len=MAX_MODEL_LEN)
    request = Request(
        0.0,
        MAX_MODEL_LEN - 4,
        8,
        thinking_depth=2,
        thinking_round_plans=[
            RequestRoundPlan(16, 4),
            RequestRoundPlan(MAX_MODEL_LEN - 4, 8),
        ],
    )
    scheduler.add_request(request)
    scheduler._request_queue.clear()

    # The requeue installs the next round's lengths, then adds the request again.
    request.advance_thinking_round()
    with pytest.raises(ValueError, match=r"\(round index 1\)"):
        scheduler.add_request(request)


def test_max_model_len_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_model_len must be > 0"):
        VllmV1SchedulerConfig(max_model_len=0)
