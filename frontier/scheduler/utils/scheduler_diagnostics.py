"""Read-only state snapshots for cluster scheduler diagnostics."""

from typing import Any, Dict, List


def _materialized_state(scheduler: Any, state_name: str) -> Any:
    """Read a state owner without invoking compatibility-property getters."""
    return vars(scheduler).get(state_name)


def _materialized_attention_queue(scheduler: Any) -> Any:
    """Return the A-to-F queue only when its owner or direct field exists."""
    direct_queue = vars(scheduler).get("_af_batch_queue")
    if direct_queue is not None:
        return direct_queue
    state = _materialized_state(scheduler, "_attention_transfer_state")
    return () if state is None else state.batch_queue


def _materialized_m2n_raw_batches(scheduler: Any) -> Any:
    """Return the raw M2N inventory without creating its state owner."""
    direct_batches = vars(scheduler).get("_raw_batch_waiting_for_m2n_back")
    if direct_batches is not None:
        return direct_batches
    state = _materialized_state(scheduler, "_m2n_state")
    return None if state is None else state.raw_batches


def scheduler_is_empty(scheduler: Any) -> bool:
    """Return whether queues and all child Replica schedulers are empty."""
    request_queue = len(scheduler._request_queue)
    af_queue = len(_materialized_attention_queue(scheduler))
    schedulers = list(scheduler._replica_schedulers.items())
    schedulers.extend(
        ((replica_id, None), child)
        for replica_id, child in getattr(scheduler, "_full_stage_replica_schedulers", {}).items()
    )
    replica_states = [(key, child.is_empty()) for key, child in schedulers]
    from frontier.logger import get_cluster_logger
    get_cluster_logger(__name__, scheduler._cluster_type.name).info(
        "[IDLE-CHECK][%s] request_queue=%s, af_batch_queue=%s, replica_empty=%s",
        scheduler._cluster_type.name,
        request_queue,
        af_queue,
        [(str(key), empty) for key, empty in replica_states],
    )
    return request_queue == 0 and af_queue == 0 and all(empty for _, empty in replica_states)


def format_ep_trace_identity(identity: Dict[str, Any]) -> str:
    """Serialize a validated EP trace identity in stable field order."""

    required = (
        "replica_id", "stage_id", "request_ids", "request_runtime_epochs",
        "iteration_ids", "schedule_epoch", "afd_stage_idx", "operation_id",
        "operation_kind",
    )
    if any(field not in identity for field in required):
        raise ValueError("EP trace identity is incomplete")
    return (
        f"replica_id={int(identity['replica_id'])}, "
        f"stage_id={int(identity['stage_id'])}, "
        f"request_ids={list(identity['request_ids'])}, "
        f"request_runtime_epochs={list(identity['request_runtime_epochs'])}, "
        f"iteration_ids={list(identity['iteration_ids'])}, "
        f"schedule_epoch={int(identity['schedule_epoch'])}, "
        f"afd_stage_idx={int(identity['afd_stage_idx'])}, "
        f"operation_id={int(identity['operation_id'])}, "
        f"operation_kind={identity['operation_kind']}"
    )


class SchedulerDiagnostics:
    """Build fail-fast, JSON-compatible snapshots from a cluster scheduler."""

    @staticmethod
    def request_id(request: Any) -> int:
        if not hasattr(request, "id"):
            raise TypeError(f"Expected Request-like object with id, got {type(request)}")
        return int(request.id)

    @classmethod
    def request_collection(cls, requests: Any) -> Dict[str, Any]:
        if requests is None:
            return {"status": "not_applicable"}
        values = list(requests.values()) if isinstance(requests, dict) else list(requests)
        return {
            "count": len(values),
            "request_ids": [cls.request_id(request) for request in values],
            "requests": [
                {
                    "id": cls.request_id(request),
                    "arrived_at": getattr(request, "arrived_at", None),
                    "num_prefill_tokens": getattr(request, "num_prefill_tokens", None),
                    "num_decode_tokens": getattr(request, "num_decode_tokens", None),
                    "num_processed_tokens": getattr(request, "num_processed_tokens", None),
                    "current_decode_token_index": getattr(
                        request, "current_decode_token_index", None
                    ),
                    "completed_layer_count": getattr(request, "completed_layer_count", None),
                    "af_roundtrip_inflight": getattr(request, "af_roundtrip_inflight", None),
                    "completed": getattr(request, "completed", None),
                }
                for request in values
            ],
        }

    @staticmethod
    def batch_id(batch: Any) -> int:
        if not hasattr(batch, "id"):
            raise TypeError(f"Expected Batch-like object with id, got {type(batch)}")
        return int(batch.id)

    @classmethod
    def batch_collection(cls, batches: Any) -> Dict[str, Any]:
        if batches is None:
            return {"status": "not_applicable"}
        values = list(batches)
        return {
            "count": len(values),
            "batch_ids": [cls.batch_id(batch) for batch in values],
            "batch_global_ids": [getattr(batch, "global_id", None) for batch in values],
            "request_ids": [list(getattr(batch, "request_ids", [])) for batch in values],
            "batches": [
                {
                    "id": cls.batch_id(batch),
                    "global_id": getattr(batch, "global_id", None),
                    "replica_id": getattr(batch, "replica_id", None),
                    "afd_stage_idx": getattr(batch, "afd_stage_idx", None),
                    "target_ffn_replica_id": getattr(batch, "target_ffn_replica_id", None),
                    "total_num_tokens": getattr(batch, "total_num_tokens", None),
                    "request_ids": list(getattr(batch, "request_ids", [])),
                    "is_idle": getattr(batch, "is_idle", None),
                }
                for batch in values
            ],
        }

    @staticmethod
    def lane_tuple(lane: Any) -> List[Any]:
        if not isinstance(lane, tuple) or len(lane) != 2:
            raise TypeError(
                "Expected lane tuple(replica_id, replica_local_id), "
                f"got {lane!r}"
            )
        return [lane[0], lane[1]]

    @classmethod
    def transfer_pairs(cls, pairs: Any) -> Dict[str, Any]:
        values = list(pairs)
        details = []
        batches = []
        for pair in values:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(f"Expected (batch, transfer_info) pair, got {pair!r}")
            batch, transfer_info = pair
            batches.append(batch)
            details.append(
                {
                    "batch_id": cls.batch_id(batch),
                    "batch_global_id": getattr(batch, "global_id", None),
                    "request_ids": list(getattr(batch, "request_ids", [])),
                    "source_lane": [
                        getattr(transfer_info, "source_replica_id", None),
                        getattr(transfer_info, "source_replica_local_id", None),
                    ],
                    "target_ffn_replica_id": getattr(
                        transfer_info, "target_ffn_replica_id", None
                    ),
                    "layer_id": getattr(transfer_info, "layer_id", None),
                    "afd_stage_idx": getattr(transfer_info, "afd_stage_idx", None),
                    "activation_size_bytes": getattr(
                        transfer_info, "activation_size_bytes", None
                    ),
                }
            )
        return {
            "count": len(values),
            "batch_ids": [cls.batch_id(batch) for batch in batches],
            "request_ids": [list(getattr(batch, "request_ids", [])) for batch in batches],
            "pairs": details,
        }

    @classmethod
    def waiting_groups(
        cls, waiting_by_layer: Dict[tuple[int, int] | tuple[int, int, int], Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        groups = []
        for group_key, room in sorted(waiting_by_layer.items(), key=lambda item: str(item[0])):
            if not isinstance(group_key, tuple) or len(group_key) not in (2, 3):
                raise TypeError(
                    "Expected DECODE_FFN waiting key(layer, stage[, round]), "
                    f"got {group_key!r}"
                )
            key_state = {"layer_id": group_key[0], "afd_stage_idx": group_key[1]}
            if len(group_key) == 3:
                key_state["barrier_round_id"] = group_key[2]
            if "per_lane_queues" not in room or "lanes_rr_order" not in room:
                raise RuntimeError(
                    f"M2N waiting room {group_key} missing per_lane_queues or lanes_rr_order"
                )
            lane_queues = [
                {
                    "lane": cls.lane_tuple(lane),
                    "queue": cls.transfer_pairs(lane_queue),
                }
                for lane, lane_queue in sorted(
                    room["per_lane_queues"].items(), key=lambda item: str(item[0])
                )
            ]
            groups.append(
                {
                    "key": key_state,
                    "lanes_rr_order": [
                        cls.lane_tuple(lane) for lane in list(room["lanes_rr_order"])
                    ],
                    "rr_cursor": room.get("rr_cursor"),
                    "lane_queues": lane_queues,
                }
            )
        return groups

    @classmethod
    def ready_groups(cls, ready_groups: Any) -> List[Dict[str, Any]]:
        return [cls.transfer_pairs(group) for group in list(ready_groups)]

    @classmethod
    def raw_waiting_map(cls, raw_batch_waiting_map: Dict[Any, Any]) -> Dict[str, Any]:
        if raw_batch_waiting_map is None:
            raise RuntimeError(
                "_raw_batch_waiting_for_m2n_back is required for cluster diagnostics"
            )
        keys = sorted(raw_batch_waiting_map.keys())
        batches = [raw_batch_waiting_map[key] for key in keys]
        return {
            "count": len(raw_batch_waiting_map),
            "keys": [int(key) for key in keys],
            "batch_ids": [cls.batch_id(batch) for batch in batches],
            "request_ids": [list(getattr(batch, "request_ids", [])) for batch in batches],
        }

    @classmethod
    def collect(cls, scheduler: Any) -> Dict[str, Any]:
        """Return the existing cluster diagnostic schema with fail-fast checks."""
        required_attrs = ("_cluster_type", "_request_queue", "_replica_schedulers")
        for attr_name in required_attrs:
            if attr_name not in vars(scheduler):
                raise RuntimeError(f"Cluster scheduler missing required debug field {attr_name}")

        cluster_type = scheduler._cluster_type
        if getattr(cluster_type, "name", None) == "DECODE_ATTN":
            attention_state = _materialized_state(scheduler, "_attention_transfer_state")
            if attention_state is None and "_af_batch_queue" not in vars(scheduler):
                raise RuntimeError("DECODE_ATTN scheduler missing _af_batch_queue")
            af_queue = cls.batch_collection(_materialized_attention_queue(scheduler))
        else:
            af_queue = {"status": "not_applicable"}

        if getattr(cluster_type, "name", None) == "DECODE_FFN":
            m2n_state = _materialized_state(scheduler, "_m2n_state")
            if m2n_state is None:
                raise RuntimeError("DECODE_FFN scheduler missing _m2n_waiting_by_layer")
            m2n_waiting_groups = cls.waiting_groups(m2n_state.waiting_by_layer)
            m2n_ready_groups = cls.ready_groups(m2n_state.ready_groups)
        else:
            m2n_waiting_groups = {"status": "not_applicable"}
            m2n_ready_groups = {"status": "not_applicable"}

        raw_batches = _materialized_m2n_raw_batches(scheduler)
        if raw_batches is None and getattr(cluster_type, "name", None) == "DECODE_FFN":
            raise RuntimeError(
                "DECODE_FFN scheduler missing _raw_batch_waiting_for_m2n_back"
            )

        replica_states = {}
        scheduler_items = list(scheduler._replica_schedulers.items())
        scheduler_items.extend(
            ((replica_id, None), replica_scheduler)
            for replica_id, replica_scheduler in getattr(
                scheduler, "_full_stage_replica_schedulers", {}
            ).items()
        )
        for scheduler_key, replica_scheduler in sorted(
            scheduler_items, key=lambda item: str(item[0])
        ):
            if not hasattr(replica_scheduler, "get_debug_state"):
                raise RuntimeError(f"Replica scheduler {scheduler_key} missing get_debug_state()")
            replica_states[str(scheduler_key)] = replica_scheduler.get_debug_state()

        return {
            "scheduler_class": scheduler.__class__.__name__,
            "cluster_type": cluster_type.name,
            "request_queue": cls.request_collection(scheduler._request_queue),
            "af_queue": af_queue,
            "m2n_waiting_groups": m2n_waiting_groups,
            "m2n_ready_groups": m2n_ready_groups,
            "raw_batch_waiting_map": (
                cls.raw_waiting_map(raw_batches)
                if raw_batches is not None
                else {"status": "not_applicable"}
            ),
            "replica_schedulers": replica_states,
        }
