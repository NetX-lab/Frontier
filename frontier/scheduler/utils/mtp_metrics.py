"""Speculative-decoding terminal metrics helpers."""

from typing import Any


def record_terminal_completion_delay(batch: Any, terminal_delay_s: float) -> None:
    """Record terminal MTP tail work only for requests with terminal rows."""
    delay_value = float(terminal_delay_s)
    if delay_value < 0.0:
        raise ValueError(
            f"terminal MTP completion delay must be >= 0, got={delay_value}"
        )
    if delay_value == 0.0:
        return

    metadata = getattr(batch, "spec_decode_metadata", None)
    if metadata is None:
        raise ValueError(
            "terminal MTP completion delay requires spec_decode_metadata"
        )
    terminal_rows = getattr(
        metadata,
        "terminal_overshoot_verify_tokens_per_request",
        None,
    )
    if terminal_rows is None:
        raise ValueError(
            "terminal MTP completion delay requires terminal overshoot rows"
        )
    if len(terminal_rows) != len(batch.requests):
        raise ValueError(
            "terminal overshoot row count mismatch: "
            f"rows={len(terminal_rows)}, requests={len(batch.requests)}"
        )
    if not any(len(rows) > 0 for rows in terminal_rows):
        raise ValueError(
            "positive terminal MTP completion delay has no active request rows"
        )
    request_ids = [
        int(request.id)
        for request, rows in zip(batch.requests, terminal_rows)
        if len(rows) > 0
    ]
    if not request_ids:
        raise ValueError(
            "positive terminal MTP completion delay has no request-local "
            "terminal rows"
        )
    batch.add_spec_terminal_completion_delay(request_ids, delay_value)
