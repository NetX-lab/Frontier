"""EP AllToAll combine readiness and collective admission."""

from typing import Any, Callable

import math
from numbers import Real


def handle_combine_ready(
    scheduler: Any,
    *,
    time: float,
    replica_id: int,
    stage_id: int,
    batch: Any,
    ep_id: int,
    resolve_collective_kind: Callable,
    prepare_timing: Callable,
):
    """Admit one EP lane into the combine barrier and schedule completion."""
    from frontier.events.ep_alltoall_combine_collective_event import (
        EPAllToAllCombineCollectiveEvent,
    )
    from frontier.logger import get_cluster_logger

    logger = get_cluster_logger(__name__, scheduler._cluster_type.name)
    if (
        not isinstance(time, Real)
        or isinstance(time, bool)
        or not math.isfinite(float(time))
    ):
        raise ValueError(
            f"EP combine arrival time must be a finite int or float, got {time!r}"
        )
    time = float(time)
    (
        batch_global_id,
        ep_wait_room,
        expected_ep_ids,
        is_complete,
    ) = scheduler._validate_ep_barrier_arrival(
        phase="combine",
        waiting_rooms=scheduler._ep_allgather_waiting_room,
        replica_id=replica_id,
        stage_id=stage_id,
        batch=batch,
        ep_id=ep_id,
    )
    existing_batches = {} if ep_wait_room is None else ep_wait_room["batches"]
    existing_arrival_times = (
        {} if ep_wait_room is None else ep_wait_room["arrival_times"]
    )
    prospective_batches = dict(existing_batches)
    prospective_arrival_times = dict(existing_arrival_times)
    prospective_batches[ep_id] = batch
    prospective_arrival_times[ep_id] = time
    expected_ep_size = len(expected_ep_ids)
    if not is_complete:
        if ep_wait_room is None:
            ep_wait_room = scheduler._ep_allgather_waiting_room[replica_id][stage_id][
                batch_global_id
            ]
        ep_wait_room["batches"][ep_id] = batch
        ep_wait_room["arrival_times"][ep_id] = time
    logger.info(
        f"[EP-WAIT-ROOM][ENTER] time={time:.3f}s, batch_id={batch.id}, "
        f"global_id={batch_global_id}, replica={replica_id}, stage={stage_id}, ep_id={ep_id}"
    )
    arrived_ep_ids = list(prospective_batches.keys())
    arrived_batch_ids = [prospective_batches[eid].id for eid in arrived_ep_ids]
    logger.info(
        f"[EP-WAIT-ROOM][STATUS] global_id={batch_global_id}, "
        f"arrived={len(prospective_batches)}/{expected_ep_size}, "
        f"ep_ids={arrived_ep_ids}, batch_ids={arrived_batch_ids}"
    )
    if not is_complete:
        logger.info(
            f"[DEBUG] Waiting for more EP replicas: "
            f"{len(ep_wait_room['batches'])}/{expected_ep_size}"
        )
        return []

    logger.info(
        "[DEBUG] All EP replicas arrived! Creating "
        "EPAllToAllCombineCollectiveEvent"
    )
    model_config = scheduler._config.replica_config.model_config
    scheduler._get_step3_ep_alltoall_payload_bytes(prospective_batches)
    ep_collective_kind = resolve_collective_kind(
        model_config,
        scheduler._cluster_type,
        expected_ep_size,
    )
    timing = prepare_timing(
        prospective_batches=prospective_batches,
        prospective_arrival_times=prospective_arrival_times,
        expected_ep_size=expected_ep_size,
        collective_kind=ep_collective_kind,
        cluster_type=scheduler._cluster_type,
        hidden_size=int(model_config.embedding_dim),
        predict_alltoall=scheduler._predictor.predict_alltoall_time,
        predict_allgather=scheduler._predictor.predict_allgather_time,
        collective_time_validator=scheduler._validate_ep_collective_exec_time,
    )
    if ep_wait_room is None:
        ep_wait_room = scheduler._ep_allgather_waiting_room[replica_id][stage_id][
            batch_global_id
        ]
    ep_wait_room["batches"][ep_id] = batch
    ep_wait_room["arrival_times"][ep_id] = time
    logger.info(
        f"[DEBUG] Creating EPAllToAllCombineCollectiveEvent at "
        f"time={timing.final_event_time:.3f}s, combine_end_time={timing.combine_end_time:.3f}s, "
        f"sync_time={timing.sync_time:.3f}s, exec_time={timing.exec_time_ms:.3f}ms, "
        f"post_combine_time={timing.post_combine_time_s:.6f}s, "
        f"data_size={timing.data_size_bytes} bytes ({timing.payload_description})"
    )
    return [
        EPAllToAllCombineCollectiveEvent(
            timing.final_event_time,
            replica_id,
            stage_id,
            batch_global_id,
            combine_end_time=timing.combine_end_time,
        )
    ]
