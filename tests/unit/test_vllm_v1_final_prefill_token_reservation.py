"""PREFILL keeps ``final_prefill_reserved_tokens`` for a waiting final round.

While a final-round prefill waits, running hidden-round prefills leave the
reserved tokens of the iteration budget, and the final round is admitted with
them in the same iteration. The reserve must stay below every per-iteration
token budget, or running hidden-round prefills get no tokens while a final
round that cannot be admitted waits, and scheduling stops.
"""

import pytest

from frontier.config import (
    FixedRequestLengthGeneratorConfig,
    PoissonRequestIntervalGeneratorConfig,
    ReplicaConfig,
    SyntheticRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.entities import Replica, Request
from frontier.entities.request import RequestRoundPlan
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    StageExecutionContext,
)
from frontier.types import ClusterType

TOKEN_BUDGET = 32
RESERVED_TOKENS = 8


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


def _build_prefill_scheduler() -> VLLMv1EngineReplicaScheduler:
    replica_config = ReplicaConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        device="a100",
        network_device="a100_pairwise_nvlink",
    )
    request_generator_config = SyntheticRequestGeneratorConfig(
        num_requests=2,
        length_generator_config=FixedRequestLengthGeneratorConfig(
            prefill_tokens=16, decode_tokens=1
        ),
        interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=1.0),
    )
    replica = Replica(replica_config, request_generator_config, ClusterType.PREFILL)
    return VLLMv1EngineReplicaScheduler(
        replica_config=replica_config,
        replica_scheduler_config=VllmV1SchedulerConfig(
            num_blocks=100,
            block_size=16,
            batch_size_cap=4,
            max_tokens_in_batch=TOKEN_BUDGET,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=0,
            final_prefill_reserved_tokens=RESERVED_TOKENS,
        ),
        request_generator_config=request_generator_config,
        replica=replica,
        predictor=object(),
        cluster_scheduler=_ClusterScheduler(replica.id, replica.num_pipeline_stages),
        cluster_type=ClusterType.PREFILL,
    )


def _two_round_request(arrived_at: float, hidden_prompt: int, final_prompt: int) -> Request:
    return Request(
        arrived_at,
        final_prompt,
        1,
        thinking_depth=2,
        thinking_round_plans=[
            RequestRoundPlan(hidden_prompt, 1),
            RequestRoundPlan(final_prompt, 1),
        ],
    )


def test_running_hidden_prefill_leaves_the_reserved_tokens_to_a_waiting_final_round() -> None:
    scheduler = _build_prefill_scheduler()
    hidden = _two_round_request(0.0, hidden_prompt=64, final_prompt=16)
    scheduler.add_request(hidden)
    first = scheduler._schedule_prefill_only()
    assert dict(zip(first.requests, first.num_tokens)) == {hidden: TOKEN_BUDGET}

    final = _two_round_request(0.1, hidden_prompt=4, final_prompt=16)
    final.advance_thinking_round()
    scheduler.add_request(final)
    second = scheduler._schedule_prefill_only()

    assert dict(zip(second.requests, second.num_tokens)) == {
        hidden: TOKEN_BUDGET - RESERVED_TOKENS,
        final: RESERVED_TOKENS,
    }


@pytest.mark.parametrize(
    "budget_fields",
    [
        {"max_tokens_in_batch": TOKEN_BUDGET},
        {
            "max_tokens_in_batch": 4 * TOKEN_BUDGET,
            "enable_phase_aware_thinking_profile": True,
            "hidden_phase_max_tokens_in_batch": TOKEN_BUDGET,
        },
    ],
)
def test_a_reserve_that_fills_an_iteration_budget_is_rejected(budget_fields) -> None:
    with pytest.raises(ValueError, match="final_prefill_reserved_tokens must be below"):
        VllmV1SchedulerConfig(final_prefill_reserved_tokens=TOKEN_BUDGET, **budget_fields)
