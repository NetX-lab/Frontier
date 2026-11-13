"""A MONOLITHIC recompute row is priced as prefill at the recompute cursor."""

from types import SimpleNamespace

import pytest

from frontier.config import (
    ReplicaConfig,
    SyntheticRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.config.speculative_decoding_config import SpeculativeDecodingConfig
from frontier.entities import Replica
from frontier.entities.batch import Batch
from frontier.entities.batch_stage import BatchStage
from frontier.entities.request import Request
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    StageExecutionContext,
)
from frontier.types import ClusterType


class _Predictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


def _request_at_cursor(cursor: int) -> Request:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=32,
        num_decode_tokens=16,
        num_processed_tokens=40,
    )
    request._is_prefill_complete = True
    request._prefill_completed_at = 1.0
    request._first_decode_token_completed_at = 1.0
    request.on_preempted(recompute=True, scheduler_num_computed_tokens=None)
    request.on_batch_end(2.0, cursor, ClusterType.MONOLITHIC)
    return request


def test_recompute_row_is_priced_as_prefill_at_the_cursor() -> None:
    cursor = 20
    width = 8
    request = _request_at_cursor(cursor)
    assert request.num_context_tokens == cursor
    assert request.num_processed_tokens == 40

    batch = Batch(
        replica_id=0,
        requests=[request],
        num_tokens=[width],
        is_moe=False,
    )
    assert batch.num_prefill_tokens == width

    stage = BatchStage(
        batch_id=batch.id,
        replica_id=0,
        pipeline_stage=0,
        execution_time=0.0,
        model_execution_time=0.0,
        requests=[request],
        num_tokens=[width],
        request_is_decoding=batch.request_is_decoding,
        cluster_type=ClusterType.MONOLITHIC,
    )
    assert stage.request_num_prefill_tokens == [width]

    predictor = _Predictor.__new__(_Predictor)
    predictor._config = SimpleNamespace(kv_cache_prediction_granularity=64)
    assert predictor._get_batch_prefill_attention_params(batch) == [(64, width)]


def test_final_recompute_chunk_commits_one_token_and_decodes_again() -> None:
    cursor = 20
    request = _request_at_cursor(cursor)
    processed_before = request.num_processed_tokens
    request.on_batch_end(
        3.0,
        processed_before - cursor,
        ClusterType.MONOLITHIC,
    )

    assert request.num_processed_tokens == processed_before + 1
    assert request.is_decoding
    batch = Batch(
        replica_id=0,
        requests=[request],
        num_tokens=[1],
        is_moe=False,
    )
    assert batch.num_prefill_tokens == 0


def test_a_decode_row_keeps_its_scheduled_phase_after_a_later_preemption() -> None:
    # vLLM runs a scheduled step as scheduled: a schedule pass that preempts
    # the request later turns its next execution into a recompute, not the
    # decode step already in the batch queue.
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=32,
        num_decode_tokens=16,
        num_processed_tokens=40,
    )
    request._is_prefill_complete = True
    request._prefill_completed_at = 1.0
    request._first_decode_token_completed_at = 1.0
    batch = Batch(replica_id=0, requests=[request], num_tokens=[1], is_moe=False)

    request.on_preempted(recompute=True, scheduler_num_computed_tokens=40)
    assert request.is_recomputing

    assert batch.request_is_decoding == [True]
    assert batch.num_prefill_tokens == 0
    assert batch.num_decode_tokens == 1
    predictor = _Predictor.__new__(_Predictor)
    predictor._config = SimpleNamespace(kv_cache_prediction_granularity=1)
    assert predictor._get_batch_prefill_attention_params(batch) == []
    assert predictor._get_batch_decode_attention_params(batch) == (1, 40)
    shape = SklearnExecutionTimePredictor._get_mla_batch_runtime_shape_components(batch)
    assert shape["prefill_active_token_counts"] == []
    assert shape["decode_active_token_counts"] == [1]
    stage = BatchStage(
        batch_id=batch.id,
        replica_id=0,
        pipeline_stage=0,
        execution_time=0.0,
        model_execution_time=0.0,
        requests=[request],
        num_tokens=[1],
        request_is_decoding=batch.request_is_decoding,
        cluster_type=ClusterType.MONOLITHIC,
    )
    assert stage.request_num_prefill_tokens == [0]


def _scheduler(
    cluster_type: ClusterType,
    *,
    num_stages: int = 1,
    speculative: bool = False,
) -> VLLMv1EngineReplicaScheduler:
    replica_config = ReplicaConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        device="a100",
        network_device="a100_pairwise_nvlink",
        num_pipeline_stages=num_stages,
        speculative_decoding_config=SpeculativeDecodingConfig(enabled=speculative),
    )
    request_config = SyntheticRequestGeneratorConfig()
    replica = Replica(replica_config, request_config, cluster_type)
    contexts = {
        stage_id: StageExecutionContext(
            replica_id=replica.id, stage_id=stage_id, ep_size=1
        )
        for stage_id in range(num_stages)
    }
    cluster_scheduler = SimpleNamespace(
        _config=SimpleNamespace(decode_attn_micro_batch_size=1),
        get_stage_execution_context=lambda replica_id, stage_id: contexts[stage_id],
    )
    return VLLMv1EngineReplicaScheduler(
        replica_config=replica_config,
        replica_scheduler_config=VllmV1SchedulerConfig(
            num_blocks=100,
            block_size=16,
            batch_size_cap=4,
            max_tokens_in_batch=128,
            max_model_len=256,
            enable_chunked_prefill=True,
        ),
        request_generator_config=request_config,
        replica=replica,
        predictor=object(),
        cluster_scheduler=cluster_scheduler,
        cluster_type=cluster_type,
    )


@pytest.mark.parametrize("num_stages", [1, 2])
def test_scheduler_decode_context_excludes_sampled_token_after_final_prefill(
    num_stages: int,
) -> None:
    scheduler = _scheduler(ClusterType.MONOLITHIC, num_stages=num_stages)
    request = Request(0.0, 128, 4)
    request.on_batch_end(1.0, 127, ClusterType.MONOLITHIC)
    scheduler._scheduled_num_computed_tokens_by_request[request.id] = 128
    final_prefill = scheduler._create_batch([request], [1])
    assert final_prefill.num_context_tokens == [127]
    assert final_prefill.request_is_decoding == [False]

    request.on_batch_end(2.0, 1, ClusterType.MONOLITHIC)
    assert request.num_processed_tokens == 129
    first_decode = scheduler._create_batch([request], [1])
    assert first_decode.num_context_tokens == [128]
    assert first_decode.request_is_decoding == [True]

    request.on_batch_end(3.0, 1, ClusterType.MONOLITHIC)
    assert request.num_processed_tokens == 130
    second_decode = scheduler._create_batch([request], [1])
    assert second_decode.num_context_tokens == [129]
    assert first_decode.num_context_tokens == [128]
    assert first_decode.request_is_decoding == [True]


@pytest.mark.parametrize("cluster_type", [ClusterType.DECODE, ClusterType.DECODE_ATTN])
def test_disaggregated_decode_context_keeps_all_completed_kv(cluster_type) -> None:
    scheduler = _scheduler(cluster_type)
    request = Request(0.0, 128, 4)
    request.on_batch_end(1.0, 128, ClusterType.PREFILL)
    assert request.num_processed_tokens == 128
    assert scheduler._create_batch([request], [1]).num_context_tokens == [128]
    request.on_batch_end(2.0, 1, cluster_type)
    assert request.num_processed_tokens == 129
    assert scheduler._create_batch([request], [1]).num_context_tokens == [129]


def test_scheduler_recompute_context_uses_cursor_then_settled_decode_frontier() -> None:
    scheduler = _scheduler(ClusterType.MONOLITHIC)
    request = _request_at_cursor(20)
    scheduler._scheduled_num_computed_tokens_by_request[request.id] = 28
    recompute = scheduler._create_batch([request], [8])
    assert recompute.num_context_tokens == [20]
    assert recompute.request_is_decoding == [False]
    request.on_batch_end(3.0, 20, ClusterType.MONOLITHIC)
    assert request.num_processed_tokens == 41
    assert scheduler._create_batch([request], [1]).num_context_tokens == [40]
    assert recompute.num_context_tokens == [20]


@pytest.mark.parametrize("num_stages", [1, 2])
def test_speculative_decode_keeps_verify_metadata_with_settled_context(num_stages) -> None:
    scheduler = _scheduler(
        ClusterType.MONOLITHIC, num_stages=num_stages, speculative=True
    )
    request = Request(0.0, 32, 16)
    request.initialize_spec_decode_state(enabled=True, method="eagle")
    request.on_batch_end(1.0, 32, ClusterType.MONOLITHIC)
    batch = scheduler._create_batch([request], [3])
    assert batch.num_context_tokens == [32]
    assert batch.num_tokens == [3]
    assert batch.request_is_decoding == [True]
    assert batch.spec_decode_metadata.planned_draft_tokens_per_request == [2]
    assert batch.spec_decode_metadata.verify_tokens_per_request == [3]
    assert batch.spec_decode_metadata.committed_tokens_per_request == [2]
