"""The sample of a step that is still in flight when its request is preempted."""

from frontier.entities.batch import Batch, SpecDecodeBatchMetadata
from frontier.entities.request import Request
from frontier.types import ClusterType


def _decode_request(*, processed: int, prefill: int = 8, decode: int = 8) -> Request:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=prefill,
        num_decode_tokens=decode,
        num_processed_tokens=processed,
    )
    request._is_prefill_complete = True
    request._prefill_completed_at = 1.0
    request._first_decode_token_completed_at = 1.0
    return request


def _batch(request: Request, width: int, committed: int | None = None) -> Batch:
    batch = Batch(
        replica_id=0,
        requests=[request],
        num_tokens=[width],
        is_moe=False,
    )
    if committed is not None:
        batch.spec_decode_metadata = SpecDecodeBatchMetadata(
            method="ngram",
            planned_draft_tokens_per_request=[width - 1],
            verify_tokens_per_request=[width],
            accepted_draft_tokens_per_request=[max(committed - 1, 0)],
            rejected_draft_tokens_per_request=[width - committed],
            committed_tokens_per_request=[committed],
            uses_lookahead_slots=False,
        )
    batch._request_execution_signatures = [
        request._preempted_step.execution_signature
    ]
    return batch


def test_decode_step_in_flight_commits_its_tokens() -> None:
    plain = _decode_request(processed=10)
    plain.on_preempted(recompute=True, step_in_flight=True)
    stopped = _batch(plain, 1).apply_preempted_step_samples(5.0, ClusterType.MONOLITHIC)
    assert stopped == []
    assert plain.num_processed_tokens == 11
    assert plain.is_recomputing

    speculative = _decode_request(processed=10)
    speculative._spec_decode_enabled = True
    speculative.on_preempted(recompute=True, step_in_flight=True)
    stopped = _batch(speculative, 3, committed=2).apply_preempted_step_samples(
        5.0, ClusterType.MONOLITHIC
    )
    assert stopped == []
    assert speculative.num_processed_tokens == 12
    assert speculative.is_recomputing


def test_partial_recompute_chunk_samples_nothing() -> None:
    request = _decode_request(processed=40, prefill=16, decode=32)
    request.on_preempted(recompute=True, step_in_flight=False)
    request.on_batch_end(2.0, 20, ClusterType.MONOLITHIC)
    assert request.num_context_tokens == 20

    request.on_preempted(recompute=True, step_in_flight=True)
    stopped = _batch(request, 8).apply_preempted_step_samples(
        5.0, ClusterType.MONOLITHIC
    )

    assert stopped == []
    assert request.num_processed_tokens == 40
    assert request.num_context_tokens == 0


def test_final_recompute_chunk_commits_one_and_restarts() -> None:
    request = _decode_request(processed=40, prefill=16, decode=32)
    request.on_preempted(recompute=True, step_in_flight=False)
    request.on_batch_end(2.0, 20, ClusterType.MONOLITHIC)
    request.on_preempted(recompute=True, step_in_flight=True)

    stopped = _batch(request, 20).apply_preempted_step_samples(
        5.0, ClusterType.MONOLITHIC
    )

    assert stopped == []
    assert request.num_processed_tokens == 41
    assert request.num_context_tokens == 0
    assert request.is_recomputing


def test_partial_prefill_chunk_samples_nothing() -> None:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=32,
        num_decode_tokens=8,
        num_processed_tokens=10,
    )
    request._scheduled = True
    request.on_preempted(recompute=True, step_in_flight=True)

    stopped = _batch(request, 8).apply_preempted_step_samples(
        3.0, ClusterType.MONOLITHIC
    )

    assert stopped == []
    assert request.num_processed_tokens == 0
    assert not request.is_prefill_complete
    assert request.prefill_completed_at == 0


def test_final_prefill_chunk_grants_the_first_token_and_recomputes() -> None:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=32,
        num_decode_tokens=8,
        num_processed_tokens=10,
    )
    request._scheduled = True
    request.on_preempted(recompute=True, step_in_flight=True)

    stopped = _batch(request, 22).apply_preempted_step_samples(
        4.0, ClusterType.MONOLITHIC
    )

    assert stopped == []
    assert request.is_prefill_complete
    assert request.num_processed_tokens == 33
    assert request.prefill_completed_at == 4.0
    assert request.first_decode_token_completed_at == 4.0
    assert request.is_recomputing
    assert request.num_context_tokens == 0


def test_length_stop_completes_the_request_and_is_returned() -> None:
    request = _decode_request(processed=8, prefill=8, decode=1)
    request.on_preempted(recompute=True, step_in_flight=True)
    batch = _batch(request, 1)

    stopped = batch.apply_preempted_step_samples(6.0, ClusterType.MONOLITHIC)

    assert request.completed
    assert request.num_processed_tokens == request.total_tokens
    assert stopped == [(0, request)]


def test_new_batch_schedule_clears_the_preempted_step() -> None:
    request = _decode_request(processed=10)
    request.on_preempted(recompute=True, step_in_flight=True)
    assert request._preempted_step is not None

    request.on_batch_schedule(7.0, ClusterType.MONOLITHIC)

    assert request._preempted_step is None


def test_decode_role_commits_the_token_without_a_recompute_cursor() -> None:
    request = _decode_request(processed=10)
    request.on_preempted(recompute=False, step_in_flight=True)

    stopped = _batch(request, 1).apply_preempted_step_samples(5.0, ClusterType.DECODE)

    assert stopped == []
    assert request.num_processed_tokens == 11
    assert not request.is_recomputing
    assert request._num_recomputed_tokens is None
