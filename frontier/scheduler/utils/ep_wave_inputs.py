"""Pure input preparation for Replica-local expert-parallel waves."""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple

from frontier.entities import Batch


class EPWaveInputs(NamedTuple):
    """Validated source batches and aggregate input for one EP wave."""

    source_batches: dict[int, Batch]
    step_id: int
    non_idle_batches: tuple[Batch, ...]
    sample_batch: Batch
    aggregate_batch: Batch
    total_step_tokens: int
    total_step_prefill_tokens: int


def prepare_ep_wave_inputs(
    *,
    source_batches: Mapping[int, Batch],
    batch: Batch,
    step_id_getter: Callable[[Batch], int],
    aggregate_batch_builder: Callable[[Batch, int, int], Batch],
) -> EPWaveInputs:
    """Validate lane ownership and build the aggregate predictor input."""

    if not isinstance(source_batches, Mapping) or not source_batches:
        raise ValueError("EP wave source_batches must be a non-empty lane mapping")
    normalized: dict[int, Batch] = {}
    for lane_id, source_batch in source_batches.items():
        if type(lane_id) is not int or lane_id < 0:
            raise ValueError(f"EP wave lane ID must be a non-negative int, got {lane_id!r}")
        if not isinstance(source_batch, Batch):
            raise TypeError(
                "EP wave source_batches values must be Batch instances, "
                f"got {type(source_batch).__name__}"
            )
        normalized[lane_id] = source_batch

    step_id = int(step_id_getter(batch))
    for lane_id, source_batch in normalized.items():
        if int(step_id_getter(source_batch)) != step_id:
            raise ValueError("all EP wave source batches must share one forward step ID")
    non_idle = tuple(source_batch for source_batch in normalized.values() if not source_batch.is_idle)
    if not non_idle:
        raise ValueError("EP wave requires a non-idle source batch")
    sample_batch = non_idle[0]
    total_tokens = sum(int(source_batch.total_num_tokens) for source_batch in non_idle)
    total_prefill_tokens = sum(int(source_batch.num_prefill_tokens) for source_batch in non_idle)
    aggregate_batch = (
        sample_batch
        if len(non_idle) == 1
        else aggregate_batch_builder(sample_batch, total_tokens, total_prefill_tokens)
    )
    return EPWaveInputs(
        source_batches=normalized,
        step_id=step_id,
        non_idle_batches=non_idle,
        sample_batch=sample_batch,
        aggregate_batch=aggregate_batch,
        total_step_tokens=total_tokens,
        total_step_prefill_tokens=total_prefill_tokens,
    )
