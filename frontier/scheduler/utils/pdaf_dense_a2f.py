"""Dense PD-AF attention-to-FFN transfer handling."""

from __future__ import annotations

from typing import Any

from frontier.types import ClusterType


def release_dense_a2f(
    scheduler: Any,
    time: float,
    batch: Any,
    *,
    replica_id: int,
    replica_local_id: int | None,
    layer_id: int,
    logger: Any,
) -> list:
    """Create the dense A2F transfer and release its stage lane."""

    from frontier.events.m2n_transfer_start_event import M2NTransferStartEvent
    from frontier.events.replica_schedule_event import ReplicaScheduleEvent

    lane = (replica_id, replica_local_id)
    barrier_round_id = scheduler._peek_decode_attn_barrier_round_id()
    activation_size, transfer_time = scheduler._validate_decode_attn_a2f_predictor_result(
        scheduler._m2n_transfer_predictor.get_transfer_info(
            source_cluster_type=ClusterType.DECODE_ATTN,
            target_cluster_type=ClusterType.DECODE_FFN,
            batch=batch,
            replica_config=scheduler._config.replica_config,
        )
    )
    transfer_event = M2NTransferStartEvent(
        time=time,
        source_replica_id=replica_id,
        source_replica_local_id=replica_local_id,
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        batch=batch,
        activation_size_bytes=activation_size,
        transfer_time_ms=transfer_time,
        layer_id=layer_id,
        afd_stage_idx=batch.afd_stage_idx,
        source_execution_replica_id=replica_id,
        source_execution_replica_local_id=replica_local_id,
    )
    schedule_event = ReplicaScheduleEvent(
        time,
        replica_id,
        scheduler._cluster_type,
        replica_local_id,
    )
    phase_update = scheduler._set_decode_attn_batch_cohort_phase(
        batch,
        phase="ffn_inflight",
        replica_id=replica_id,
        replica_local_id=replica_local_id,
        layer_id=layer_id,
        prepare_only=True,
    )
    scheduler._commit_decode_attn_batch_phases([phase_update])
    batch.decode_attn_barrier_round_id = barrier_round_id
    batch.decode_attn_barrier_expected_lanes = (lane,)
    scheduler._decode_attn_barrier_round_counter = barrier_round_id + 1
    logger.info(
        f"[A2F-DENSE-STREAM] layer={layer_id} afd_stage_idx={batch.afd_stage_idx} "
        f"lane={lane} batch_id={batch.id} round={batch.decode_attn_barrier_round_id}"
    )
    return [transfer_event, schedule_event]
