"""The sample of a step that is still in flight when its request is preempted."""

from frontier.entities.batch import Batch
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


def _prefill_request(*, processed: int) -> Request:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=32,
        num_decode_tokens=8,
        num_processed_tokens=processed,
    )
    request._scheduled = True
    return request


def _in_flight_batch(request: Request, width: int) -> Batch:
    return Batch(replica_id=0, requests=[request], num_tokens=[width], is_moe=False)


def test_decode_step_in_flight_commits_its_tokens() -> None:
    plain = _decode_request(processed=10)
    batch = _in_flight_batch(plain, 1)
    plain.on_preempted(recompute=True, scheduler_num_computed_tokens=10)
    assert batch.apply_preempted_step_samples(5.0, ClusterType.MONOLITHIC) == []
    assert plain.num_processed_tokens == 11
    assert plain.is_recomputing

    speculative = _decode_request(processed=10)
    speculative._spec_decode_enabled = True
    speculative.record_spec_decode_iteration(
        verify_tokens=3, accepted_drafts=1, rejected_drafts=1, committed_tokens=2
    )
    batch = _in_flight_batch(speculative, 3)
    speculative.on_preempted(recompute=True, scheduler_num_computed_tokens=12)
    assert batch.apply_preempted_step_samples(5.0, ClusterType.MONOLITHIC) == []
    assert speculative.num_processed_tokens == 12
    assert speculative.is_recomputing


def test_a_victim_with_no_step_in_flight_has_no_pending_sample() -> None:
    request = _decode_request(processed=10)
    request.on_preempted(recompute=True, scheduler_num_computed_tokens=None)
    assert not request.has_preempted_step


def test_partial_recompute_chunk_leaves_no_pending_sample() -> None:
    request = _decode_request(processed=40, prefill=16, decode=32)
    request.on_preempted(recompute=True, scheduler_num_computed_tokens=None)
    request.on_batch_end(2.0, 20, ClusterType.MONOLITHIC)
    assert request.num_context_tokens == 20

    request.on_preempted(recompute=True, scheduler_num_computed_tokens=28)

    assert not request.has_preempted_step
    assert request.num_processed_tokens == 40
    assert request.num_context_tokens == 0


def test_final_recompute_chunk_commits_one_and_restarts() -> None:
    request = _decode_request(processed=40, prefill=16, decode=32)
    request.on_preempted(recompute=True, scheduler_num_computed_tokens=None)
    request.on_batch_end(2.0, 20, ClusterType.MONOLITHIC)
    batch = _in_flight_batch(request, 20)
    request.on_preempted(recompute=True, scheduler_num_computed_tokens=40)

    assert batch.apply_preempted_step_samples(5.0, ClusterType.MONOLITHIC) == []
    assert request.num_processed_tokens == 41
    assert request.num_context_tokens == 0
    assert request.is_recomputing


def test_partial_prefill_chunk_leaves_no_pending_sample() -> None:
    request = _prefill_request(processed=10)
    request.on_preempted(recompute=True, scheduler_num_computed_tokens=18)

    assert not request.has_preempted_step
    assert request.num_processed_tokens == 0
    assert not request.is_prefill_complete
    assert request.prefill_completed_at == 0


def test_final_prefill_chunk_grants_the_first_token_and_recomputes() -> None:
    request = _prefill_request(processed=10)
    batch = _in_flight_batch(request, 22)
    request.on_preempted(recompute=True, scheduler_num_computed_tokens=32)

    assert batch.apply_preempted_step_samples(4.0, ClusterType.MONOLITHIC) == []
    assert request.is_prefill_complete
    assert request.num_processed_tokens == 33
    assert request.prefill_completed_at == 4.0
    assert request.first_decode_token_completed_at == 4.0
    assert request.is_recomputing
    assert request.num_context_tokens == 0


def test_an_earlier_chunk_leaves_the_sample_to_the_final_chunk() -> None:
    request = _prefill_request(processed=0)
    # Both prompt chunks were scheduled before either ended.
    earlier_chunk = Batch(
        replica_id=0, requests=[request], num_tokens=[16], is_moe=False,
        num_context_tokens=[0],
    )
    final_chunk = Batch(
        replica_id=0, requests=[request], num_tokens=[16], is_moe=False,
        num_context_tokens=[16],
    )
    request.on_preempted(recompute=True, scheduler_num_computed_tokens=32)

    assert earlier_chunk.apply_preempted_step_samples(3.0, ClusterType.MONOLITHIC) == []
    assert request.num_processed_tokens == 0
    assert request.has_preempted_step
    assert final_chunk.apply_preempted_step_samples(4.0, ClusterType.MONOLITHIC) == []
    assert request.is_prefill_complete
    assert request.num_processed_tokens == 33
    assert request.prefill_completed_at == 4.0
    assert request.is_recomputing


def test_length_stop_completes_the_request_and_is_returned() -> None:
    request = _decode_request(processed=8, prefill=8, decode=1)
    batch = _in_flight_batch(request, 1)
    request.on_preempted(recompute=True, scheduler_num_computed_tokens=8)
    assert request.stops_on_preempted_step

    stopped = batch.apply_preempted_step_samples(6.0, ClusterType.MONOLITHIC)

    assert request.completed
    assert request.num_processed_tokens == request.total_tokens
    assert stopped == [(0, request)]


def test_a_sample_taken_at_readmission_is_not_applied_again() -> None:
    request = _decode_request(processed=10)
    batch = _in_flight_batch(request, 1)
    request.on_preempted(recompute=True, scheduler_num_computed_tokens=10)
    assert request.has_preempted_step and not request.stops_on_preempted_step

    request.on_preempted_step_end(7.0, ClusterType.MONOLITHIC)
    request.on_batch_schedule(7.0, ClusterType.MONOLITHIC)

    assert request.num_processed_tokens == 11
    assert request.is_recomputing and request.num_context_tokens == 0
    assert batch.apply_preempted_step_samples(8.0, ClusterType.MONOLITHIC) == []
    assert request.num_processed_tokens == 11


def test_decode_role_commits_the_token_without_a_recompute_cursor() -> None:
    request = _decode_request(processed=10)
    batch = _in_flight_batch(request, 1)
    request.on_preempted(recompute=False, scheduler_num_computed_tokens=10)

    assert batch.apply_preempted_step_samples(5.0, ClusterType.DECODE) == []
    assert request.num_processed_tokens == 11
    assert not request.is_recomputing
    assert request._num_recomputed_tokens is None
