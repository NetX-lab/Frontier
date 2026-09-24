"""A MONOLITHIC recompute row is priced as prefill at the recompute cursor."""

from types import SimpleNamespace

from frontier.entities.batch import Batch
from frontier.entities.batch_stage import BatchStage
from frontier.entities.request import Request
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
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
    request.on_preempted(recompute=True, step_in_flight=False)
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
