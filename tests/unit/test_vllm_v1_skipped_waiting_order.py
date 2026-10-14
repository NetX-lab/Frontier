"""Skipped waiting requests keep their FCFS place at the head of the queue.

The vLLM target (fork ``ea95f571``) collects skipped requests in order and puts
them back with ``self.waiting.prepend_requests(skipped_waiting_requests)``
(``vllm/v1/core/sched/scheduler.py:882-883``), and ``FCFSRequestQueue``
prepends a queue as ``extendleft(reversed(requests))``
(``vllm/v1/core/sched/request_queue.py:102-105``).
"""

from frontier.config import (
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
from frontier.types import ClusterType


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


def _build_scheduler() -> VLLMv1EngineReplicaScheduler:
    replica_config = ReplicaConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        device="a100",
        network_device="a100_pairwise_nvlink",
    )
    request_generator_config = SyntheticRequestGeneratorConfig(
        num_requests=4,
        length_generator_config=FixedRequestLengthGeneratorConfig(
            prefill_tokens=8, decode_tokens=2
        ),
        interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=1.0),
    )
    replica = Replica(replica_config, request_generator_config, ClusterType.MONOLITHIC)
    return VLLMv1EngineReplicaScheduler(
        replica_config=replica_config,
        replica_scheduler_config=VllmV1SchedulerConfig(
            num_blocks=100,
            block_size=16,
            batch_size_cap=1,
            max_tokens_in_batch=12,
            enable_chunked_prefill=False,
        ),
        request_generator_config=request_generator_config,
        replica=replica,
        predictor=object(),
        cluster_scheduler=_ClusterScheduler(replica.id, replica.num_pipeline_stages),
        cluster_type=ClusterType.MONOLITHIC,
    )


def test_skipped_prompts_return_ahead_of_the_unexamined_requests() -> None:
    scheduler = _build_scheduler()
    long_1, long_2 = Request(0.0, 40, 2), Request(0.1, 40, 2)
    short_1, short_2 = Request(0.2, 8, 2), Request(0.3, 8, 2)
    for request in (long_1, long_2, short_1, short_2):
        scheduler.add_request(request)

    # Without chunked prefill both 40-token prompts exceed the 12-token budget
    # and are skipped; short_1 fills the one running slot, and short_2 is never
    # examined.
    _, scheduled, _ = scheduler._schedule_waiting_requests(12)

    assert scheduled == [short_1]
    waiting = scheduler._get_sorted_waiting_queue()
    assert [request.id for request in waiting] == [long_1.id, long_2.id, short_2.id]
