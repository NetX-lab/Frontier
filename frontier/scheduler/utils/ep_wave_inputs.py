"""Pure input preparation for Replica-local expert-parallel waves."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, NamedTuple

from frontier.entities import Batch, DummyForwardBatch


class EPWaveInputs(NamedTuple):
    """Validated source batches and aggregate input for one EP wave."""

    source_batches: dict[int, Batch]
    step_id: int
    sample_batch: Batch
    aggregate_batch: Batch
    total_step_tokens: int
    total_step_prefill_tokens: int
    lane_compute_tokens: dict[int, int]
    cuda_graph: bool


@dataclass(frozen=True)
class DPForwardShape:
    """Tokens each attention-DP rank runs in one forward (vLLM num_tokens_across_dp)."""

    num_tokens_across_dp: tuple[int, ...]
    cuda_graph: bool


def dp_forward_shape(wave_inputs: EPWaveInputs, attn_dp: int) -> DPForwardShape:
    """Return the per-DP-rank forward widths of one EP wave.

    Every DP rank joins each step's MoE collectives, an idle rank through a
    dummy forward, so the wave must carry one source batch per DP lane.
    """

    lanes = sorted(wave_inputs.lane_compute_tokens)
    if lanes != list(range(attn_dp)):
        raise ValueError(
            "DP-group MoE collectives need one source batch per attention-DP "
            f"lane 0..{attn_dp - 1}, got lanes {lanes}"
        )
    return DPForwardShape(
        tuple(wave_inputs.lane_compute_tokens[lane] for lane in lanes),
        wave_inputs.cuda_graph,
    )


def prepare_ep_wave_inputs(
    *,
    source_batches: Mapping[int, Batch],
    batch: Batch,
    step_id_getter: Callable[[Batch], int],
    aggregate_batch_builder: Callable[[Batch, int, int], Batch],
    cluster_type: Any,
) -> EPWaveInputs:
    """Validate the lane mapping and build the aggregate predictor input.

    The aggregate runs at the lanes' physical compute width. Each lane runs
    its forward at its CUDA-graph capture size or, eager, at its own tokens.
    Unless the engine is eager, vLLM then pads every DP rank, a dummy pass
    included, to the largest of those widths (gpu_model_runner
    get_dp_padding). A role whose batches carry no decode CUDA-graph metadata
    runs eager.
    """

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
    lane_batches = tuple(normalized.values())
    # A dummy forward has no runtime shape of its own; the aggregate takes a
    # request-carrying lane's shape whenever one runs.
    sample_batch = next(
        (
            source_batch
            for source_batch in lane_batches
            if not isinstance(source_batch, DummyForwardBatch)
        ),
        lane_batches[0],
    )
    total_tokens = sum(int(source_batch.total_num_tokens) for source_batch in lane_batches)
    total_prefill_tokens = sum(int(source_batch.num_prefill_tokens) for source_batch in lane_batches)
    lane_compute_tokens = [
        int(source_batch.get_effective_total_tokens_for_compute(cluster_type))
        for source_batch in lane_batches
    ]
    native_bound = [
        getattr(source_batch, "forward_start_descriptor", None) is not None
        for source_batch in lane_batches
    ]
    if any(native_bound) and not all(native_bound):
        raise ValueError("EP wave cannot mix agreed and unagreed forward inputs")
    shared_padding = (
        any(source_batch.get_decode_cuda_graph_runtime_mode() != "NONE" for source_batch in lane_batches)
        if all(native_bound) else
        any(source_batch.decode_cuda_graph_metadata is not None for source_batch in lane_batches)
    )
    if shared_padding:
        lane_compute_tokens = [max(lane_compute_tokens)] * len(lane_batches)
    aggregate_batch = (
        sample_batch
        if len(lane_batches) == 1
        else aggregate_batch_builder(sample_batch, sum(lane_compute_tokens), total_prefill_tokens)
    )
    if all(native_bound) and len(lane_batches) > 1:
        aggregate_batch.forward_start_descriptor = replace(
            sample_batch.forward_start_descriptor,
            num_tokens=sum(lane_compute_tokens), num_reqs=None,
        )
    return EPWaveInputs(
        source_batches=normalized,
        step_id=step_id,
        sample_batch=sample_batch,
        aggregate_batch=aggregate_batch,
        total_step_tokens=total_tokens,
        total_step_prefill_tokens=total_prefill_tokens,
        lane_compute_tokens=dict(zip(normalized, lane_compute_tokens)),
        cuda_graph=shared_padding,
    )
