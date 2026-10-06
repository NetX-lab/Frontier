"""Pure timing and request-selection helpers for collective completion handlers."""

import math
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class PrefillFinalTiming:
    pipeline_time: float
    cpu_overhead: float
    forward_launch_stall_time: float
    explicit_model_time: float
    total_time: float
    completion_time: float
    actual_execution_time: float


@dataclass(frozen=True)
class DecodeFinalTiming:
    pipeline_time: float
    cpu_overhead: float
    forward_launch_stall_time: float
    draft_proposer_time: float
    mtp_terminal_overshoot_time: float
    total_time: float


def attention_delay_seconds(execution_time: Any) -> float:
    """Convert a single-layer attention duration from milliseconds to seconds."""

    return execution_time.get_single_layer_attention_scope_time() * 1e-3


def validate_decode_layer_advance(requests: Iterable[Any], total_layers: int) -> None:
    """Reject requests whose layer counter cannot advance by one layer."""

    for request in requests:
        if request.completed_layer_count >= total_layers:
            raise ValueError(
                "Decode post_moe layer counter cannot advance: "
                f"request_id={request.id}, "
                f"completed_layer_count={request.completed_layer_count}, "
                f"total_layers={total_layers}, "
                "current_decode_token_index="
                f"{request.current_decode_token_index}, "
                "spec_last_committed_tokens="
                f"{getattr(request, '_spec_last_committed_tokens', None)}; "
                "possible missing prior decode-step reset"
            )


def advance_decode_layer(requests: Iterable[Any], total_layers: int) -> None:
    """Credit one completed layer to each request, validating the whole set first.

    Every layer-completion path owes each decoding request exactly one credit
    per executed layer; the caller selects which requests are decoding.
    Validation runs over the full selection before any counter moves, so a
    request that cannot advance leaves its peers untouched.
    """

    requests = list(requests)
    validate_decode_layer_advance(requests, total_layers)
    for request in requests:
        request.mb_on_step_layer_count_increment(num_layers_completed=1)


def prepare_prefill_final_timing(
    execution_time: Any,
    component_times_ms: Iterable[float],
    sync_time: float,
    original_start_time: float,
) -> PrefillFinalTiming:
    """Prepare PREFILL final-stage timing values without scheduler mutation.

    The layer events simulated the stage's device work up to ``sync_time``; with
    the pipeline send it is the device span that hides the forward's launch.
    """

    elapsed_stage_wall_time = sync_time - original_start_time
    if elapsed_stage_wall_time < 0:
        raise ValueError(
            "Prefill sync completion time is earlier than the recorded stage start "
            f"time: sync_time={sync_time}, original_start_time={original_start_time}, "
            f"elapsed_stage_wall_time={elapsed_stage_wall_time}"
        )

    explicit_model_time = math.fsum(component_times_ms) * 1e-3
    pipeline_time = execution_time.pipeline_time * 1e-3
    cpu_overhead = execution_time.total_time - execution_time.model_time
    if cpu_overhead < 0:
        raise ValueError(
            "Prefill stage CPU overhead cannot be negative: "
            f"total_time={execution_time.total_time}, "
            f"model_time={execution_time.model_time}, "
            f"stage_cpu_overhead={cpu_overhead}"
        )
    forward_launch_stall_time = execution_time.forward_launch_stall_time(
        elapsed_stage_wall_time + pipeline_time
    )
    total_time = pipeline_time + cpu_overhead + forward_launch_stall_time
    return PrefillFinalTiming(
        pipeline_time=pipeline_time,
        cpu_overhead=cpu_overhead,
        forward_launch_stall_time=forward_launch_stall_time,
        explicit_model_time=explicit_model_time,
        total_time=total_time,
        completion_time=sync_time + total_time,
        actual_execution_time=sync_time + total_time - original_start_time,
    )


def prepare_decode_final_timing(
    execution_time: Any, elapsed_stage_wall_time: float
) -> DecodeFinalTiming:
    """Prepare DECODE final-stage timing values without scheduler mutation.

    ``elapsed_stage_wall_time`` is the device work the layer events simulated
    since the stage start; with the pipeline send it is the device span that
    hides the forward's launch.
    """

    pipeline_time = execution_time.pipeline_time * 1e-3
    cpu_overhead = max(
        execution_time.total_time - execution_time.model_time,
        0.0,
    )
    forward_launch_stall_time = execution_time.forward_launch_stall_time(
        elapsed_stage_wall_time + pipeline_time
    )
    draft_proposer_time = execution_time.decode_draft_proposer_time * 1e-3
    mtp_terminal_overshoot_time = (
        float(getattr(execution_time, "mtp_terminal_overshoot_time", 0.0)) * 1e-3
    )
    return DecodeFinalTiming(
        pipeline_time=pipeline_time,
        cpu_overhead=cpu_overhead,
        forward_launch_stall_time=forward_launch_stall_time,
        draft_proposer_time=draft_proposer_time,
        mtp_terminal_overshoot_time=mtp_terminal_overshoot_time,
        total_time=(
            pipeline_time + cpu_overhead + forward_launch_stall_time + draft_proposer_time
        ),
    )
