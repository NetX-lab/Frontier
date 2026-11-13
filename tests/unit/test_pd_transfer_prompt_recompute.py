"""CPU Request/Batch fixtures for transferred prompt tails, not native qualification."""

import pytest

from frontier.entities.batch import Batch
from frontier.entities.request import Request
from frontier.types import ClusterType


def _batch(request: Request, num_tokens: int) -> Batch:
    return Batch(
        replica_id=0,
        requests=[request],
        num_tokens=[num_tokens],
        is_moe=False,
    )


def _completed_prefill() -> Request:
    request = Request(0.0, 4105, 9)
    request.on_arrival(0.0, ClusterType.PREFILL)
    batch = _batch(request, 4105)
    batch.on_schedule(1.0, ClusterType.PREFILL)
    batch.on_batch_end(2.0, ClusterType.PREFILL)
    request.on_kv_cache_transfer_start(2.5)
    return request


def _transfer_state(request: Request) -> dict:
    return {
        "processed_tokens": request.num_processed_tokens,
        "context_tokens": request.num_context_tokens,
        "emitted_decode_tokens": request.num_emitted_decode_tokens,
        "prefill_complete": request.is_prefill_complete,
        "decoding": request.is_decoding,
        "recomputing": request.is_recomputing,
        "preempted": request.preempted,
        "completed": request.completed,
        "prefill_completed_at": (
            request.prefill_completed_at if request.scheduled else None
        ),
        "first_decode_token_completed_at": request.first_decode_token_completed_at,
        "decode_first_token_completed_at": request.decode_first_token_completed_at,
        "transfer_start_time": request.kv_cache_transfer_start_time,
        "transfer_end_time": request.kv_cache_transfer_end_time,
        "transfer_time": request.kv_cache_transfer_time,
        "execution_signature": request.execution_signature,
    }


def test_transferred_prompt_tail_emits_one_then_eight_decode_outputs() -> None:
    request = _completed_prefill()
    assert request.is_prefill_complete
    assert request.num_processed_tokens == 4105
    assert request.num_emitted_decode_tokens == 0
    assert request.first_decode_token_completed_at == 0.0

    request.on_kv_cache_transfer_complete(3.0, 0.5, num_computed_tokens=4080)

    assert request.kv_cache_transfer_end_time == 3.0
    assert request.kv_cache_transfer_time == 0.5
    assert request.is_recomputing
    assert not request.is_decoding
    assert not request.preempted
    assert request.get_total_preemption_count() == 0
    assert request.num_prefill_tokens_cached == 0
    assert request.num_processed_tokens == 4105
    assert request.num_context_tokens == 4080
    assert request.num_emitted_decode_tokens == 0

    request.on_arrival(3.0, ClusterType.DECODE)
    tail = _batch(request, request.num_processed_tokens - request.num_context_tokens)
    assert tail.num_tokens == [25]
    assert tail.num_context_tokens == [4080]
    assert tail.request_is_decoding == [False]
    assert tail.num_prefill_tokens == 25
    assert tail.num_decode_tokens == 0
    tail.on_schedule(4.0, ClusterType.DECODE)
    tail.on_batch_end(5.0, ClusterType.DECODE)

    assert not request.is_recomputing
    assert request.is_decoding
    assert request.num_processed_tokens == 4106
    assert request.num_emitted_decode_tokens == 1
    assert request.remaining_decode_tokens == 8
    assert request.first_decode_token_completed_at == 5.0
    assert request.client_ttft == 5.0
    assert request.prefill_completed_at == request.ttft == 2.0
    assert not request.completed

    for output_count in range(2, 10):
        decode = _batch(request, 1)
        assert decode.request_is_decoding == [True]
        assert decode.num_prefill_tokens == 0
        assert decode.num_decode_tokens == 1
        decode.on_schedule(output_count + 3.5, ClusterType.DECODE)
        decode.on_batch_end(output_count + 4.0, ClusterType.DECODE)
        assert request.num_emitted_decode_tokens == output_count
        assert request.completed == (output_count == 9)

    assert request.num_processed_tokens == request.total_tokens == 4114
    assert request.remaining_decode_tokens == 0
    assert request.completed_at == 13.0


class _IntSubclass(int):
    pass


@pytest.mark.parametrize(
    "prefix",
    [-1, 4105, 4106, False, True, 4080.0, "4080", _IntSubclass(4080)],
    ids=[
        "negative", "full-prompt", "past-prompt", "false", "true", "float",
        "string", "int-subclass",
    ],
)
def test_invalid_transferred_prefix_is_rejected_without_mutation(prefix) -> None:
    request = _completed_prefill()
    before = _transfer_state(request)

    with pytest.raises(ValueError, match="num_computed_tokens"):
        request.on_kv_cache_transfer_complete(3.0, 0.5, num_computed_tokens=prefix)

    assert _transfer_state(request) == before


@pytest.mark.parametrize("prefill_tokens", [0, 1000], ids=["not-started", "partial"])
def test_incomplete_source_prefill_is_rejected_without_mutation(prefill_tokens) -> None:
    request = Request(0.0, 4105, 9)
    if prefill_tokens:
        prefill = _batch(request, prefill_tokens)
        prefill.on_schedule(1.0, ClusterType.PREFILL)
        prefill.on_batch_end(2.0, ClusterType.PREFILL)
    before = _transfer_state(request)

    with pytest.raises(ValueError, match="prefill"):
        request.on_kv_cache_transfer_complete(3.0, 0.5, num_computed_tokens=4080)

    assert _transfer_state(request) == before


@pytest.mark.parametrize("output_source", ["decode", "handoff"])
def test_already_emitted_output_is_rejected_without_mutation(output_source) -> None:
    request = _completed_prefill()
    if output_source == "decode":
        decode = _batch(request, 1)
        decode.on_schedule(3.0, ClusterType.DECODE)
        decode.on_batch_end(4.0, ClusterType.DECODE)
    else:
        request.on_disaggregated_decode_handoff(3.0, ClusterType.DECODE_ATTN)
    assert request.num_emitted_decode_tokens == 1
    before = _transfer_state(request)

    with pytest.raises(ValueError, match="emitted"):
        request.on_kv_cache_transfer_complete(5.0, 1.0, num_computed_tokens=4080)

    assert _transfer_state(request) == before


@pytest.mark.parametrize("recompute_source", ["transfer", "preemption"])
def test_active_recompute_is_rejected_without_mutation(recompute_source) -> None:
    request = _completed_prefill()
    if recompute_source == "transfer":
        request.on_kv_cache_transfer_complete(3.0, 0.5, num_computed_tokens=4080)
    else:
        request.on_preempted(recompute=True, scheduler_num_computed_tokens=None)
    assert request.is_recomputing
    before = _transfer_state(request)

    with pytest.raises(ValueError, match="recomput"):
        request.on_kv_cache_transfer_complete(5.0, 1.0, num_computed_tokens=4000)

    assert _transfer_state(request) == before


@pytest.mark.parametrize(
    "prefix,chunks",
    [
        (4080, [(8, 4088, 0), (8, 4096, 0), (9, 4106, 1)]),
        (0, [(2048, 2048, 0), (2048, 4096, 0), (9, 4106, 1)]),
        (4104, [(1, 4106, 1)]),
    ],
    ids=["partial-prefix", "zero-prefix", "one-token-tail"],
)
def test_prompt_recompute_chunks_emit_only_at_prompt_completion(prefix, chunks) -> None:
    request = _completed_prefill()
    request.on_kv_cache_transfer_complete(3.0, 0.5, num_computed_tokens=prefix)

    for index, (width, expected_context, expected_outputs) in enumerate(chunks):
        chunk = _batch(request, width)
        assert chunk.request_is_decoding == [False]
        assert chunk.num_prefill_tokens == width
        assert chunk.num_decode_tokens == 0
        chunk.on_schedule(3.5 + index, ClusterType.DECODE)
        chunk.on_batch_end(4.0 + index, ClusterType.DECODE)

        assert request.num_context_tokens == expected_context
        assert request.num_emitted_decode_tokens == expected_outputs
        assert request.num_processed_tokens == (4106 if expected_outputs else 4105)
        assert request.is_recomputing == (expected_outputs == 0)
        assert request.is_decoding == (expected_outputs == 1)
        assert request.first_decode_token_completed_at == (
            4.0 + index if expected_outputs else 0.0
        )
        assert request.prefill_completed_at == 2.0
        assert not request.completed


@pytest.mark.parametrize("explicit_none", [False, True], ids=["omitted", "explicit-none"])
@pytest.mark.parametrize("state", ["incomplete", "prefilled", "sampled", "recomputing"])
def test_legacy_transfer_completion_only_updates_transfer_timing(state, explicit_none) -> None:
    request = Request(0.0, 4105, 9) if state == "incomplete" else _completed_prefill()
    if state == "sampled":
        decode = _batch(request, 1)
        decode.on_schedule(3.0, ClusterType.DECODE)
        decode.on_batch_end(4.0, ClusterType.DECODE)
    elif state == "recomputing":
        request.on_preempted(recompute=True, scheduler_num_computed_tokens=None)
    before = _transfer_state(request)
    kwargs = {"num_computed_tokens": None} if explicit_none else {}

    request.on_kv_cache_transfer_complete(5.0, 0.5, **kwargs)
    request.on_kv_cache_transfer_complete(7.0, 0.5, **kwargs)
    request.on_kv_cache_transfer_complete(6.0, 0.25, **kwargs)

    assert _transfer_state(request) == before | {
        "transfer_end_time": 7.0,
        "transfer_time": 1.25,
    }
