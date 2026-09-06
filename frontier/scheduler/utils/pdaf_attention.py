"""Read-only topology queries for the PD-AF attention stage."""

from typing import Any, List

from frontier.scheduler.utils.pdaf_transfer import LaneIdentityScope
from frontier.types import ClusterType


def get_a2f_active_local_attn_lanes(
    scheduler: Any,
    *,
    cohort_id: int,
    request_ids: tuple[int, ...],
    afd_stage_idx: int,
    layer_id: int,
) -> List[tuple[int, int]]:
    """Return active local-attention lanes for one A-to-F wave."""
    if type(cohort_id) is not int or cohort_id < 0:
        raise ValueError("DECODE_ATTN A-to-F cohort_id must be an exact non-negative int")
    if type(request_ids) is not tuple:
        raise ValueError("DECODE_ATTN A-to-F cohort request IDs must be an exact tuple")
    for request_id in request_ids:
        if type(request_id) is not int or request_id < 0:
            raise ValueError(
                "DECODE_ATTN A-to-F cohort request IDs must contain exact "
                f"non-negative ints, got {request_id!r}"
            )
    afd_stage_idx = scheduler._validate_decode_attn_a2f_topology_value(
        afd_stage_idx, field_name="active local-attn afd_stage_idx"
    )
    layer_id = scheduler._validate_decode_attn_a2f_topology_value(
        layer_id, field_name="active local-attn layer_id"
    )

    replica_schedulers = getattr(scheduler, "_replica_schedulers", None)
    if type(replica_schedulers) is not dict:
        raise RuntimeError("DECODE_ATTN A-to-F replica scheduler topology must be an exact dict")
    scheduler_lanes = scheduler._normalize_m2n_lanes(
        list(replica_schedulers),
        identity_scope=LaneIdentityScope.FULL_STAGE,
        field_name="DECODE_ATTN A-to-F active lane topology",
        require_nonempty=False,
    )
    active_lanes: List[tuple[int, int]] = []
    requested_ids = set(request_ids)
    for lane in scheduler_lanes:
        replica_scheduler = replica_schedulers[lane]
        cohort_states = getattr(replica_scheduler, "_decode_attn_active_cohort_states", None)
        if type(cohort_states) is not dict:
            raise RuntimeError("DECODE_ATTN A-to-F active cohort states must be an exact dict")
        cohort_state = cohort_states.get(cohort_id)
        if cohort_state is None:
            continue
        if type(cohort_state) is not dict:
            raise RuntimeError("DECODE_ATTN A-to-F active cohort state must be an exact dict")
        all_request_ids = cohort_state.get("all_request_ids")
        pending_request_ids = cohort_state.get("pending_request_ids")
        if type(all_request_ids) is not set or type(pending_request_ids) is not set:
            raise RuntimeError("DECODE_ATTN A-to-F cohort request registries must be exact sets")
        for registry_name, registry in (("all_request_ids", all_request_ids), ("pending_request_ids", pending_request_ids)):
            for request_id in registry:
                if type(request_id) is not int or request_id < 0:
                    raise RuntimeError(
                        f"DECODE_ATTN A-to-F cohort {registry_name} must contain exact non-negative ints, got {request_id!r}"
                    )
        if not pending_request_ids or not requested_ids.issubset(all_request_ids):
            continue
        active_stage_indices, stage_phases, stage_layers = scheduler._validate_decode_attn_wave_stages(
            cohort_state, context="A-to-F active local-attn lane"
        )
        if afd_stage_idx not in active_stage_indices:
            continue
        if stage_phases[afd_stage_idx] == "local_attn" and stage_layers[afd_stage_idx] == layer_id:
            active_lanes.append(lane)
    return active_lanes


def get_stage_slot_active_lanes(
    scheduler: Any,
    afd_stage_idx: int,
    *,
    replica_id: int | None = None,
    phase: str | None = None,
    layer_id: int | None = None,
) -> List[tuple[int, int]]:
    """Return lanes with an active stage slot, including legacy state fallback."""
    if type(scheduler._cluster_type) is not ClusterType:
        raise RuntimeError("DECODE_ATTN cluster type must be a ClusterType")
    if scheduler._cluster_type != ClusterType.DECODE_ATTN:
        raise ValueError("_get_decode_attn_stage_slot_active_lanes is only valid for DECODE_ATTN cluster")
    if type(afd_stage_idx) is not int or afd_stage_idx < 0:
        raise ValueError("DECODE_ATTN active stage slot must be an exact non-negative int, got %r" % (afd_stage_idx,))
    if replica_id is not None and (type(replica_id) is not int or replica_id < 0):
        raise ValueError("DECODE_ATTN active stage replica_id must be an exact non-negative int, got %r" % (replica_id,))
    replica_schedulers = getattr(scheduler, "_replica_schedulers", None)
    if type(replica_schedulers) is not dict:
        raise RuntimeError("DECODE_ATTN replica scheduler topology must be an exact dict")
    scheduler_lanes = scheduler._normalize_m2n_lanes(
        list(replica_schedulers), identity_scope=LaneIdentityScope.FULL_STAGE,
        field_name="DECODE_ATTN replica scheduler lane topology", require_nonempty=False,
    )
    active_lanes: List[tuple[int, int]] = []
    for lane_replica_id, lane_replica_local_id in scheduler_lanes:
        if replica_id is not None and lane_replica_id != replica_id:
            continue
        replica_scheduler = replica_schedulers[(lane_replica_id, lane_replica_local_id)]
        get_slots = getattr(replica_scheduler, "get_decode_attn_active_stage_slots", None)
        if callable(get_slots):
            raw_slots = get_slots(phase=phase, layer_id=layer_id)
        else:
            states = getattr(replica_scheduler, "_decode_attn_active_cohort_states", {})
            if type(states) is not dict:
                raise RuntimeError("DECODE_ATTN active cohort states must be an exact dict")
            raw_slots = []
            for state in states.values():
                if type(state) is not dict or "afd_stage_idx" not in state:
                    if type(state) is not dict:
                        raise RuntimeError("DECODE_ATTN active cohort state must be an exact dict")
                    continue
                if phase is not None and state.get("af_phase") != phase:
                    continue
                state_layer_id = state.get("current_layer_id")
                if layer_id is not None and state_layer_id is not None:
                    if type(state_layer_id) is not int or state_layer_id < 0:
                        raise RuntimeError("DECODE_ATTN cohort current_layer_id must be an exact non-negative int, got %r" % (state_layer_id,))
                    if state_layer_id != layer_id:
                        continue
                raw_slots.append(state["afd_stage_idx"])
        if type(raw_slots) not in {list, tuple, set}:
            raise RuntimeError("DECODE_ATTN active stage slots must be an exact list, tuple, or set, got %r" % (raw_slots,))
        for slot in raw_slots:
            if type(slot) is not int or slot < 0:
                raise RuntimeError("DECODE_ATTN active stage slot must be an exact non-negative int, got %r" % (slot,))
            if slot == afd_stage_idx:
                active_lanes.append((lane_replica_id, lane_replica_local_id))
                break
    return active_lanes


def get_a2f_expected_lanes(scheduler: Any, afd_stage_idx: int | None = None, *, layer_id: int | None = None) -> List[tuple[int, int]]:
    """Resolve A-to-F expected lanes using stage, active-wave, then configured state."""
    if scheduler._cluster_type != ClusterType.DECODE_ATTN:
        raise ValueError("_get_decode_attn_a2f_expected_lanes is only valid for DECODE_ATTN cluster")
    if afd_stage_idx is not None:
        afd_stage_idx = scheduler._validate_decode_attn_a2f_topology_value(afd_stage_idx, field_name="expected-lane afd_stage_idx")
    if layer_id is not None:
        layer_id = scheduler._validate_decode_attn_a2f_topology_value(layer_id, field_name="expected-lane layer_id")
    if afd_stage_idx is not None:
        lanes = get_stage_slot_active_lanes(scheduler, afd_stage_idx, phase="local_attn", layer_id=layer_id)
        if lanes:
            return scheduler._normalize_m2n_lanes(lanes, identity_scope=LaneIdentityScope.FULL_STAGE, field_name="DECODE_ATTN A-to-F active stage lane topology", require_nonempty=True)
    request_map = getattr(scheduler, "_decode_attn_active_serving_wave_request_ids_by_lane", {})
    expected = scheduler._normalize_m2n_lanes(getattr(scheduler, "_decode_attn_active_serving_wave_expected_lanes", ()), identity_scope=LaneIdentityScope.FULL_STAGE, field_name="DECODE_ATTN A-to-F active wave lane topology", require_nonempty=False)
    if type(request_map) is not dict:
        raise RuntimeError("DECODE_ATTN A-to-F active wave request topology must be an exact dict")
    request_lanes = scheduler._normalize_m2n_lanes(list(request_map), identity_scope=LaneIdentityScope.FULL_STAGE, field_name="DECODE_ATTN A-to-F active wave request lane topology", require_nonempty=False)
    if expected:
        return expected
    if request_lanes:
        return sorted(request_lanes)
    configured = getattr(scheduler, "_a2f_expected_lanes", None)
    return [] if configured is None else scheduler._normalize_m2n_lanes(configured, identity_scope=LaneIdentityScope.FULL_STAGE, field_name="DECODE_ATTN A-to-F scheduler lane topology", require_nonempty=False)


def get_f2a_expected_lanes(scheduler: Any, replica_id: int, *, afd_stage_idx: int | None = None) -> List[tuple[int, int]]:
    """Resolve F-to-A lanes with inventory validation and idle-lane filtering."""
    if scheduler._cluster_type != ClusterType.DECODE_ATTN:
        raise ValueError("_get_decode_attn_f2a_expected_lanes is only valid for DECODE_ATTN cluster")
    if type(replica_id) is not int or replica_id < 0:
        raise ValueError("DECODE_ATTN F-to-A replica_id must be an exact non-negative int, got %r" % (replica_id,))
    if afd_stage_idx is not None and (type(afd_stage_idx) is not int or afd_stage_idx < 0):
        raise ValueError("DECODE_ATTN F-to-A afd_stage_idx must be an exact non-negative int, got %r" % (afd_stage_idx,))
    replica_schedulers = getattr(scheduler, "_replica_schedulers", None)
    if type(replica_schedulers) is not dict:
        raise RuntimeError("DECODE_ATTN replica scheduler topology must be an exact dict")
    replicas = getattr(getattr(scheduler, "_cluster", None), "replicas", None)
    if type(replicas) is not dict:
        raise RuntimeError("DECODE_ATTN replica inventory must be an exact dict")
    for rid in replicas:
        if type(rid) is not int or rid < 0:
            raise RuntimeError("DECODE_ATTN replica inventory IDs must be exact non-negative ints, got %r" % (rid,))
    if replica_id not in replicas:
        raise ValueError("DECODE_ATTN F-to-A replica is outside the cluster replica inventory: replica_id=%s, replica_ids=%s" % (replica_id, list(replicas)))
    idle = scheduler._normalize_m2n_lanes(list(getattr(scheduler, "_decode_attn_idle_expected_lanes", set())), identity_scope=LaneIdentityScope.FULL_STAGE, field_name="DECODE_ATTN idle lane topology", require_nonempty=False)
    idle = set(idle)
    if afd_stage_idx is not None:
        lanes = [lane for lane in get_stage_slot_active_lanes(scheduler, afd_stage_idx, replica_id=replica_id, phase="ffn_inflight") if lane not in idle]
        if lanes:
            return lanes
    request_map = getattr(scheduler, "_decode_attn_active_serving_wave_request_ids_by_lane", {})
    expected = scheduler._normalize_m2n_lanes(getattr(scheduler, "_decode_attn_active_serving_wave_expected_lanes", ()), identity_scope=LaneIdentityScope.FULL_STAGE, field_name="DECODE_ATTN active wave lane topology", require_nonempty=False)
    if type(request_map) is not dict:
        raise RuntimeError("DECODE_ATTN active wave request topology must be an exact dict")
    request_lanes = scheduler._normalize_m2n_lanes(list(request_map), identity_scope=LaneIdentityScope.FULL_STAGE, field_name="DECODE_ATTN active wave request lane topology", require_nonempty=False)
    source = expected or sorted(request_lanes)
    if source:
        return [lane for lane in source if lane[0] == replica_id and lane not in idle]
    configured = getattr(scheduler, "_f2a_expected_lanes", None)
    if configured is not None:
        lanes = scheduler._normalize_m2n_lanes(configured, identity_scope=LaneIdentityScope.FULL_STAGE, field_name="DECODE_ATTN F-to-A scheduler lane topology", require_nonempty=False)
        if lanes:
            return [lane for lane in lanes if lane[0] == replica_id and lane not in idle]
    count = scheduler._replica_scheduler_count
    if type(count) is not int or count <= 0:
        raise RuntimeError("DECODE_ATTN replica scheduler count must be an exact positive int, got %r" % (count,))
    lane = (replica_id, None)
    return [] if lane in idle else [lane]
