"""Builders for PD-AF transfer entries that do not mutate scheduler state."""

from typing import Iterable

from frontier.entities import Batch
from frontier.entities.m2n_transfer_info import M2NTransferInfo
from frontier.types import ClusterType


def build_decode_ffn_idle_entry(
    *,
    time: float,
    lane: tuple[int, int],
    layer_id: int,
    afd_stage_idx: int,
    barrier_round_id: int | None,
    expected_lanes: tuple[tuple[int, int], ...],
    is_moe: bool,
) -> tuple[Batch, M2NTransferInfo]:
    """Build one idle DECODE_ATTN-to-DECODE_FFN barrier entry."""

    idle_batch = Batch(
        replica_id=lane[0],
        requests=[],
        num_tokens=[],
        is_idle=True,
        is_moe=is_moe,
    )
    idle_batch.afd_stage_idx = afd_stage_idx
    idle_batch.decode_attn_original_replica_id = lane[0]
    idle_batch.decode_attn_original_replica_local_id = lane[1]
    idle_batch.decode_attn_barrier_round_id = barrier_round_id
    idle_batch.decode_attn_barrier_expected_lanes = expected_lanes
    idle_batch.decode_ffn_layer_id = layer_id
    idle_batch.time = time

    idle_transfer = M2NTransferInfo(
        batch=idle_batch,
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        source_replica_id=lane[0],
        source_replica_local_id=lane[1],
        activation_size_bytes=0,
        transfer_time_ms=0.0,
        transfer_start_time=time,
        layer_id=layer_id,
        afd_stage_idx=afd_stage_idx,
    )
    return idle_batch, idle_transfer


def build_decode_ffn_idle_entries(
    *,
    time: float,
    lanes: Iterable[tuple[int, int]],
    layer_id: int,
    afd_stage_idx: int,
    barrier_round_id: int | None,
    expected_lanes: tuple[tuple[int, int], ...],
    is_moe: bool,
) -> list[tuple[tuple[int, int], tuple[Batch, M2NTransferInfo]]]:
    """Build idle entries for the requested FFN lanes."""

    return [
        (
            lane,
            build_decode_ffn_idle_entry(
                time=time,
                lane=lane,
                layer_id=layer_id,
                afd_stage_idx=afd_stage_idx,
                barrier_round_id=barrier_round_id,
                expected_lanes=expected_lanes,
                is_moe=is_moe,
            ),
        )
        for lane in lanes
    ]
