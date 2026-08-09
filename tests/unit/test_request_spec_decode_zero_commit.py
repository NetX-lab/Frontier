import pytest

from frontier.entities import Request
from frontier.types import ClusterType


def _spec_decode_request() -> Request:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=4,
        num_decode_tokens=6,
        num_processed_tokens=4,
    )
    request._is_prefill_complete = True
    request._completed_layer_count = 8
    request._current_decode_token_index = 3
    request.initialize_spec_decode_state(
        enabled=True,
        method="mtp",
        num_speculative_tokens=2,
    )
    return request


@pytest.mark.parametrize(
    "cluster_type",
    [ClusterType.DECODE, ClusterType.DECODE_ATTN],
)
def test_disaggregated_zero_commit_resets_layers_without_advancing_tokens(
    cluster_type: ClusterType,
) -> None:
    request = _spec_decode_request()
    processed_tokens_before = request.num_processed_tokens
    decode_token_index_before = request.current_decode_token_index

    request.on_batch_end(
        time=1.0,
        num_tokens_processed=0,
        cluster_type=cluster_type,
    )

    assert request.completed_layer_count == 0
    assert request.num_processed_tokens == processed_tokens_before
    assert request.current_decode_token_index == decode_token_index_before
    assert request._spec_last_committed_tokens == 0


def test_monolithic_zero_commit_resets_layers_without_advancing_tokens() -> None:
    request = _spec_decode_request()
    processed_tokens_before = request.num_processed_tokens
    decode_token_index_before = request.current_decode_token_index

    request.on_batch_end(
        time=1.0,
        num_tokens_processed=0,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert request.completed_layer_count == 0
    assert request.num_processed_tokens == processed_tokens_before
    assert request.current_decode_token_index == decode_token_index_before
    assert request._spec_last_committed_tokens == 0


@pytest.mark.parametrize(
    "cluster_type",
    [ClusterType.DECODE, ClusterType.DECODE_ATTN, ClusterType.MONOLITHIC],
)
def test_positive_spec_commit_preserves_existing_rollout_semantics(
    cluster_type: ClusterType,
) -> None:
    request = _spec_decode_request()
    processed_tokens_before = request.num_processed_tokens
    decode_token_index_before = request.current_decode_token_index

    request.on_batch_end(
        time=1.0,
        num_tokens_processed=2,
        cluster_type=cluster_type,
    )

    assert request.completed_layer_count == 0
    assert request.num_processed_tokens == processed_tokens_before + 2
    assert request.current_decode_token_index == decode_token_index_before + 2
    assert request._spec_last_committed_tokens == 2
