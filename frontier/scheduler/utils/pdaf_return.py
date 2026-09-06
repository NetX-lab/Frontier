"""DECODE_ATTN F2A return barrier handling."""

from collections import defaultdict, deque
from typing import Any


def release_ready_return_round(
    scheduler: Any,
    round_key: tuple,
    expected_lanes: list[tuple[int, int]],
    logger: Any,
) -> list[Any]:
    """Release one complete per-lane F2A return round."""

    if scheduler._cluster_type.name != "DECODE_ATTN":
        raise ValueError(
            "_release_decode_attn_ready_return_round is only valid for DECODE_ATTN cluster"
        )
    room = scheduler._f2a_waiting_by_round.get(round_key)
    if room is None:
        return []
    replica_id, next_layer_id, afd_stage_idx = round_key[:3]
    per_lane_batches = room["per_lane_queues"]
    if not all(per_lane_batches.get(lane) for lane in expected_lanes):
        return []
    released_batches = [per_lane_batches[lane].popleft() for lane in expected_lanes]
    if all(not lane_queue for lane_queue in per_lane_batches.values()):
        scheduler._f2a_waiting_by_round.pop(round_key, None)
    logger.info(
        f"[F2A-GROUP-RELEASE] replica={replica_id} next_layer={next_layer_id} "
        f"afd_stage_idx={afd_stage_idx} lanes={len(expected_lanes)}"
    )
    return released_batches


def enqueue_return_round(
    scheduler: Any,
    micro_batch: Any,
    *,
    receipt: dict[str, Any],
    logger: Any,
) -> bool:
    """Admit one F2A receipt and enqueue released batches after the barrier."""

    if scheduler._cluster_type.name != "DECODE_ATTN":
        raise ValueError(
            "_enqueue_decode_attn_return_round is only valid for DECODE_ATTN cluster"
        )
    replica_id = receipt["replica_id"]
    lane = receipt["lane"]
    batch_global_id = receipt["batch_global_id"]
    decode_token_index = receipt["decode_token_index"]
    next_layer_id = receipt["next_layer_id"]
    afd_stage_idx = receipt["afd_stage_idx"]
    round_key = receipt["round_key"]
    stored_expected_lanes = receipt["stored_expected_lanes"]
    expected_lanes = receipt["expected_lanes"]
    room = receipt["room"]
    if room is None:
        room = {
            "per_lane_queues": defaultdict(deque),
            "expected_lanes": stored_expected_lanes,
        }
        scheduler._f2a_waiting_by_round[round_key] = room
    elif room["expected_lanes"] is None and stored_expected_lanes is not None:
        room["expected_lanes"] = stored_expected_lanes
    room["per_lane_queues"][lane].append(micro_batch)
    ready_lanes = sum(
        1 for expected_lane in expected_lanes if room["per_lane_queues"].get(expected_lane)
    )
    logger.info(
        f"[F2A-GROUP-READY] replica={replica_id} global_id={batch_global_id} "
        f"token_idx={decode_token_index} next_layer={next_layer_id} "
        f"afd_stage_idx={afd_stage_idx} lane={lane} "
        f"depth={len(room['per_lane_queues'][lane])} "
        f"ready_lanes={ready_lanes}/{len(expected_lanes)}"
    )
    released_batches = release_ready_return_round(
        scheduler, round_key, expected_lanes, logger
    )
    enqueued_batches = 0
    for ready_batch in released_batches:
        scheduler._set_decode_attn_batch_cohort_phase(
            ready_batch,
            phase="local_attn",
            replica_id=int(ready_batch.decode_attn_original_replica_id),
            replica_local_id=ready_batch.decode_attn_original_replica_local_id,
            layer_id=int(ready_batch.af_inflight_layer_count),
        )
        if getattr(ready_batch, "trace_replay_initial_hydration_moe_head_consumed", False):
            scheduler.get_replica_scheduler(
                int(ready_batch.decode_attn_original_replica_id),
                ready_batch.decode_attn_original_replica_local_id,
            ).on_batch_end(ready_batch)
            logger.info(
                "[AF-ARRIVAL][DROP] mb=%s global_id=%s dropped after synthetic "
                "trace-replay hydration head completed its first MoE consume",
                ready_batch.id,
                ready_batch.global_id,
            )
            continue
        scheduler._af_batch_queue.append(ready_batch)
        enqueued_batches += 1
        logger.info(
            f"[AF-ARRIVAL][ENQUEUE] mb={ready_batch.id} global_id={ready_batch.global_id} "
            f"re-enqueued to AF priority queue after F→A round barrier; "
            f"af_queue_size={len(scheduler._af_batch_queue)}"
        )
    return enqueued_batches > 0
