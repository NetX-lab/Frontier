"""Pure preparation for EP combine completion callbacks."""

from typing import Any, Callable, NamedTuple

from frontier.moe_ep_workload import resolve_ep_lane_workload
from frontier.scheduler.utils.expert_parallel import (
    resolve_ep_execution_time,
    resolve_source_batch_ids,
    validate_token_conservation,
)


class EPCombineCompletionPlan(NamedTuple):
    """Validated inputs needed before EP combine scheduler mutation."""

    canonical_ep_id: int
    source_batch_ids: tuple[int, ...]
    ffn_execution_time: float
    raw_batches: tuple[tuple[int, Any], ...]
    active_requests_by_batch: tuple[tuple[int, tuple[Any, ...]], ...]
    activation_bytes_by_ep_id: tuple[tuple[int, int], ...]


def prepare_ep_combine_completion(
    *,
    ep_batches: dict[int, Any],
    raw_batch_lookup: Callable[[int], Any | None],
    cluster_name: str,
    replica_id: int,
    stage_id: int,
    batch_global_id: int,
    token_validator: Callable[[int, Any, str], None] = validate_token_conservation,
) -> EPCombineCompletionPlan:
    """Validate and collect completion inputs without mutating runtime state."""

    if not ep_batches:
        raise ValueError("EP combine completion requires at least one EP batch")
    for ep_id, ep_batch in ep_batches.items():
        lane_workload = resolve_ep_lane_workload(ep_batch, required=True)
        if lane_workload is None:
            raise ValueError(f"Missing EP lane workload for ep_id={ep_id}")
        token_validator(
            int(ep_batch.total_num_tokens),
            lane_workload,
            context=(
                f"EP AllToAll combine collective - EP batch "
                f"(cluster={cluster_name}, replica={replica_id}, stage={stage_id}, "
                f"ep_id={ep_id}, batch_global_id={batch_global_id})"
            ),
        )

    canonical_ep_id = min(ep_batches)
    source_batch_ids = tuple(resolve_source_batch_ids(ep_batches))
    ffn_execution_time = resolve_ep_execution_time(ep_batches)

    raw_batches = []
    for batch_id in source_batch_ids:
        raw_batch = raw_batch_lookup(batch_id)
        if raw_batch is None:
            raise ValueError(f"Missing raw batch for id={batch_id} in _raw_batch_waiting_for_m2n_back")
        raw_batches.append((batch_id, raw_batch))

    active_requests_by_batch = []
    for batch_id, raw_batch in raw_batches:
        active_requests = tuple(
            request
            for request, runtime_epoch in zip(
                raw_batch.requests,
                raw_batch.request_runtime_epochs,
            )
            if int(getattr(request, "runtime_epoch", 0)) == int(runtime_epoch)
        )
        active_requests_by_batch.append((batch_id, active_requests))

    activation_bytes_by_ep_id = tuple(
        (
            ep_id,
            int(getattr(ep_batch, "activation_bytes", 0))
            if getattr(ep_batch, "activation_bytes", 0)
            else 0,
        )
        for ep_id, ep_batch in ep_batches.items()
    )
    return EPCombineCompletionPlan(
        canonical_ep_id=canonical_ep_id,
        source_batch_ids=source_batch_ids,
        ffn_execution_time=ffn_execution_time,
        raw_batches=tuple(raw_batches),
        active_requests_by_batch=tuple(active_requests_by_batch),
        activation_bytes_by_ep_id=activation_bytes_by_ep_id,
    )
