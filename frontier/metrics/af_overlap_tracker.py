"""Wall-clock overlap tracker for PD-AF Attention/FFN/M2N pipeline stages."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from frontier.types import ClusterType

Interval = Tuple[float, float]
StageKey = Tuple[ClusterType, int, int | None, int, int]


def merge_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    """Return sorted disjoint intervals covering the union of ``intervals``."""
    valid = [
        (float(start), float(end))
        for start, end in intervals
        if float(end) > float(start)
    ]
    if not valid:
        return []
    valid.sort()
    merged: List[Interval] = [valid[0]]
    for start, end in valid[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            if end > prev_end:
                merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def intersect_intervals(
    first: Sequence[Interval],
    second: Sequence[Interval],
) -> List[Interval]:
    """Return sorted disjoint intervals covering the intersection of two merged lists."""
    result: List[Interval] = []
    i = 0
    j = 0
    while i < len(first) and j < len(second):
        start = max(first[i][0], second[j][0])
        end = min(first[i][1], second[j][1])
        if end > start:
            result.append((start, end))
        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1
    return result


def intervals_duration_ms(intervals: Sequence[Interval]) -> float:
    """Return total duration of disjoint intervals in milliseconds."""
    return sum(end - start for start, end in intervals) * 1e3


class AFOverlapTracker:
    """Track DECODE_ATTN, DECODE_FFN, and M2N active intervals for overlap metrics."""

    def __init__(self) -> None:
        self._pending_stages: Dict[StageKey, float] = {}
        self._decode_attn_intervals: List[Interval] = []
        self._decode_ffn_intervals: List[Interval] = []
        self._m2n_a2f_intervals: List[Interval] = []
        self._m2n_f2a_intervals: List[Interval] = []

    def record_stage_start(
        self,
        *,
        cluster_type: ClusterType,
        replica_id: int,
        replica_local_id: int | None,
        stage_id: int,
        batch_id: int,
        start_time: float,
    ) -> None:
        if cluster_type not in (ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN):
            return
        key: StageKey = (
            cluster_type,
            int(replica_id),
            None if replica_local_id is None else int(replica_local_id),
            int(stage_id),
            int(batch_id),
        )
        self._pending_stages[key] = float(start_time)

    def record_stage_end(
        self,
        *,
        cluster_type: ClusterType,
        replica_id: int,
        replica_local_id: int | None,
        stage_id: int,
        batch_id: int,
        end_time: float,
        fallback_start_time: Optional[float] = None,
    ) -> None:
        if cluster_type not in (ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN):
            return
        key: StageKey = (
            cluster_type,
            int(replica_id),
            None if replica_local_id is None else int(replica_local_id),
            int(stage_id),
            int(batch_id),
        )
        start_time = self._pending_stages.pop(key, None)
        if start_time is None and fallback_start_time is not None:
            start_time = float(fallback_start_time)
        if start_time is None:
            return
        end_val = float(end_time)
        if end_val <= start_time:
            return
        if cluster_type == ClusterType.DECODE_ATTN:
            self._decode_attn_intervals.append((start_time, end_val))
        else:
            self._decode_ffn_intervals.append((start_time, end_val))

    def record_m2n_transfer(
        self,
        *,
        source_cluster_type: ClusterType,
        target_cluster_type: ClusterType,
        start_time: float,
        end_time: float,
    ) -> None:
        start_val = float(start_time)
        end_val = float(end_time)
        if end_val <= start_val:
            return
        if (
            source_cluster_type == ClusterType.DECODE_ATTN
            and target_cluster_type == ClusterType.DECODE_FFN
        ):
            self._m2n_a2f_intervals.append((start_val, end_val))
        elif (
            source_cluster_type == ClusterType.DECODE_FFN
            and target_cluster_type == ClusterType.DECODE_ATTN
        ):
            self._m2n_f2a_intervals.append((start_val, end_val))

    def has_data(self) -> bool:
        return bool(self._decode_attn_intervals or self._decode_ffn_intervals)

    def summarize(
        self,
        *,
        af_pipeline_num_micro_batch: Optional[int] = None,
    ) -> Dict[str, Any]:
        attn = merge_intervals(self._decode_attn_intervals)
        ffn = merge_intervals(self._decode_ffn_intervals)
        m2n_a2f = merge_intervals(self._m2n_a2f_intervals)
        m2n_f2a = merge_intervals(self._m2n_f2a_intervals)
        m2n = merge_intervals([*m2n_a2f, *m2n_f2a])
        offload = merge_intervals([*ffn, *m2n])
        decode_all = merge_intervals([*attn, *offload])

        attn_ms = intervals_duration_ms(attn)
        ffn_ms = intervals_duration_ms(ffn)
        m2n_ms = intervals_duration_ms(m2n)
        m2n_a2f_ms = intervals_duration_ms(m2n_a2f)
        m2n_f2a_ms = intervals_duration_ms(m2n_f2a)
        offload_ms = intervals_duration_ms(offload)
        decode_wall_ms = intervals_duration_ms(decode_all)

        attn_ffn_overlap_ms = intervals_duration_ms(intersect_intervals(attn, ffn))
        attn_m2n_overlap_ms = intervals_duration_ms(intersect_intervals(attn, m2n))
        attn_offload_overlap_ms = intervals_duration_ms(
            intersect_intervals(attn, offload)
        )
        m2n_bidir_overlap_ms = intervals_duration_ms(
            intersect_intervals(m2n_a2f, m2n_f2a)
        )

        serial_equiv_ms = attn_ms + ffn_ms + m2n_ms
        overlap_savings_ms = max(0.0, serial_equiv_ms - decode_wall_ms)

        summary: Dict[str, Any] = {
            "decode_attn_active_time_ms": attn_ms,
            "decode_ffn_active_time_ms": ffn_ms,
            "m2n_active_time_ms": m2n_ms,
            "m2n_attn_to_ffn_active_time_ms": m2n_a2f_ms,
            "m2n_ffn_to_attn_active_time_ms": m2n_f2a_ms,
            "m2n_bidirectional_overlap_time_ms": m2n_bidir_overlap_ms,
            "attn_ffn_compute_overlap_time_ms": attn_ffn_overlap_ms,
            "attn_m2n_overlap_time_ms": attn_m2n_overlap_ms,
            "attn_with_ffn_or_m2n_overlap_time_ms": attn_offload_overlap_ms,
            "ffn_hidden_ratio": (
                attn_ffn_overlap_ms / ffn_ms if ffn_ms > 0.0 else 0.0
            ),
            "af_offload_hidden_ratio": (
                attn_offload_overlap_ms / offload_ms if offload_ms > 0.0 else 0.0
            ),
            "decode_phase_wall_time_ms": decode_wall_ms,
            "serial_equivalent_time_ms": serial_equiv_ms,
            "pipeline_overlap_savings_ms": overlap_savings_ms,
        }
        if af_pipeline_num_micro_batch is not None:
            summary["af_pipeline_num_micro_batch"] = int(af_pipeline_num_micro_batch)
        return summary
