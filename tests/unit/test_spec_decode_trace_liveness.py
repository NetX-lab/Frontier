from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pytest

from frontier.config.config import SpeculativeDecodingConfig
from frontier.entities import Request
from frontier.spec_decode.runtime import (
    compute_iteration_outcome,
    get_planned_draft_tokens,
)
from frontier.types import ClusterType


REQUEST_ID = "request-7"


def _trace_config(
    tmp_path: Path,
    *,
    committed_tokens: list[int],
    scheduled_draft_tokens: list[int],
    per_request: bool,
) -> tuple[SpeculativeDecodingConfig, Optional[str]]:
    if per_request:
        payload = {
            "per_request_committed_tokens_per_iteration": {
                REQUEST_ID: committed_tokens,
            },
            "per_request_scheduled_draft_tokens_per_iteration": {
                REQUEST_ID: scheduled_draft_tokens,
            },
        }
        request_id: Optional[str] = REQUEST_ID
    else:
        payload = {
            "committed_tokens_per_iteration": committed_tokens,
            "scheduled_draft_tokens_per_iteration": scheduled_draft_tokens,
        }
        request_id = None

    trace_path = tmp_path / (
        "per-request-trace.json" if per_request else "global-trace.json"
    )
    trace_path.write_text(json.dumps(payload), encoding="utf-8")

    return (
        SpeculativeDecodingConfig(
            enabled=True,
            method="qwen3_moe_mtp",
            num_speculative_tokens=1,
            acceptance_trace_file=str(trace_path),
            mtp_n_predict=1,
            mtp_num_layers=1,
        ),
        request_id,
    )


def _spec_decode_request() -> Request:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=4,
        num_decode_tokens=2,
        num_processed_tokens=4,
    )
    request._is_prefill_complete = True
    request.initialize_spec_decode_state(
        enabled=True,
        method="qwen3_moe_mtp",
        num_speculative_tokens=1,
    )
    return request


def _run_iteration(
    config: SpeculativeDecodingConfig,
    request: Request,
    *,
    request_id: Optional[str],
) -> int:
    iteration_index = request.spec_total_iterations
    planned_draft_tokens = get_planned_draft_tokens(
        config,
        request.remaining_decode_tokens,
        iteration_index=iteration_index,
        request_id=request_id,
    )
    outcome = compute_iteration_outcome(
        config,
        request.remaining_decode_tokens,
        planned_draft_tokens=planned_draft_tokens,
        iteration_index=iteration_index,
        request_id=request_id,
    )
    request.record_spec_decode_iteration(
        verify_tokens=outcome.verify_tokens,
        accepted_drafts=outcome.accepted_draft_tokens,
        rejected_drafts=outcome.rejected_draft_tokens,
        committed_tokens=outcome.committed_tokens,
    )
    request.on_batch_end(
        time=float(iteration_index + 1),
        num_tokens_processed=outcome.committed_tokens,
        cluster_type=ClusterType.DECODE,
    )
    return outcome.committed_tokens


@pytest.mark.parametrize("per_request", [False, True])
def test_final_trace_entry_that_completes_request_does_not_read_past_trace(
    tmp_path: Path,
    per_request: bool,
) -> None:
    config, request_id = _trace_config(
        tmp_path,
        committed_tokens=[0, 1, 1],
        scheduled_draft_tokens=[1, 0, 0],
        per_request=per_request,
    )
    request = _spec_decode_request()

    committed_tokens = [
        _run_iteration(config, request, request_id=request_id)
        for _ in range(3)
    ]

    assert committed_tokens == [0, 1, 1]
    assert request.spec_total_iterations == 3
    assert request.spec_total_committed_tokens == 2
    assert request.remaining_decode_tokens == 0
    assert request.completed is True

    next_iteration = request.spec_total_iterations
    assert (
        get_planned_draft_tokens(
            config,
            request.remaining_decode_tokens,
            iteration_index=next_iteration,
            request_id=request_id,
        )
        == 0
    )
    completed_outcome = compute_iteration_outcome(
        config,
        request.remaining_decode_tokens,
        iteration_index=next_iteration,
        request_id=request_id,
    )
    assert completed_outcome.committed_tokens == 0
    assert completed_outcome.verify_tokens == 0


@pytest.mark.parametrize(
    ("per_request", "expected_error"),
    [
        (
            False,
            "scheduled draft trace exhausted: iteration_index=2, trace_len=2",
        ),
        (
            True,
            "per-request scheduled draft trace exhausted: "
            "request_id='request-7', iteration_index=2, trace_len=2",
        ),
    ],
)
def test_unfinished_request_fails_fast_when_next_trace_entry_is_missing(
    tmp_path: Path,
    per_request: bool,
    expected_error: str,
) -> None:
    config, request_id = _trace_config(
        tmp_path,
        committed_tokens=[0, 1],
        scheduled_draft_tokens=[1, 0],
        per_request=per_request,
    )
    request = _spec_decode_request()

    committed_tokens = [
        _run_iteration(config, request, request_id=request_id)
        for _ in range(2)
    ]

    assert committed_tokens == [0, 1]
    assert request.spec_total_iterations == 2
    assert request.spec_total_committed_tokens == 1
    assert request.remaining_decode_tokens == 1
    assert request.completed is False

    with pytest.raises(ValueError, match=re.escape(expected_error)):
        get_planned_draft_tokens(
            config,
            request.remaining_decode_tokens,
            iteration_index=request.spec_total_iterations,
            request_id=request_id,
        )
