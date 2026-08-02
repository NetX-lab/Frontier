from frontier.entities import Request
from frontier.types import ClusterType
import pytest


def test_pdaf_decode_handoff_seeds_first_output_without_extra_decode_stack():
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=4,
        num_decode_tokens=3,
    )

    request.on_batch_end(
        time=1.0,
        num_tokens_processed=4,
        cluster_type=ClusterType.PREFILL,
    )
    request.on_disaggregated_decode_handoff(
        time=1.1,
        cluster_type=ClusterType.DECODE_ATTN,
    )

    assert request.num_processed_tokens == 4
    assert request.num_emitted_decode_tokens == 1
    assert request.remaining_decode_tokens == 2

    request.on_batch_end(2.0, 1, ClusterType.DECODE_ATTN)
    request.on_batch_end(3.0, 1, ClusterType.DECODE_ATTN)

    assert request.num_processed_decode_tokens == 2
    assert request.num_emitted_decode_tokens == 3
    assert request.completed
    assert request.completed_at == 3.0


def test_pdaf_decode_handoff_is_idempotent_and_requires_prefill_completion():
    request = Request(arrived_at=0.0, num_prefill_tokens=2, num_decode_tokens=2)

    with pytest.raises(ValueError, match="before prefill completes"):
        request.on_disaggregated_decode_handoff(0.1, ClusterType.DECODE_ATTN)

    request.on_batch_end(1.0, 2, ClusterType.PREFILL)
    request.on_disaggregated_decode_handoff(1.1, ClusterType.DECODE_ATTN)
    request.on_disaggregated_decode_handoff(1.2, ClusterType.DECODE_ATTN)

    assert request.num_emitted_decode_tokens == 1
    assert request.remaining_decode_tokens == 1


def test_single_decode_token_does_not_create_handoff_progress():
    request = Request(arrived_at=0.0, num_prefill_tokens=2, num_decode_tokens=1)
    request.on_batch_end(1.0, 2, ClusterType.PREFILL)
    request.on_disaggregated_decode_handoff(1.1, ClusterType.DECODE_ATTN)

    assert request.num_emitted_decode_tokens == 0
    assert request.remaining_decode_tokens == 1
