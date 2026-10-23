"""State transitions for PD-AF attention batches."""

from copy import deepcopy
from typing import Any, Callable, List, Optional

from frontier.entities import Batch
from frontier.types import ClusterType

def prepare_decode_attn_batch_phase(
    scheduler,
    batch: Batch,
    *,
    phase: str,
    replica_id: int,
    replica_local_id: int | None,
    layer_id: int | None = None,
) -> Optional[dict[str, Any]]:
    """Prepare a cohort phase update without mutating cohort state."""

    if scheduler._cluster_type != ClusterType.DECODE_ATTN:
        return None

    cohort_id = getattr(batch, "decode_attn_cohort_id", None)
    if cohort_id is None:
        return None
    if type(cohort_id) is not int or cohort_id < 0:
        raise ValueError(
            "DECODE_ATTN cohort ID must be an exact non-negative int, "
            f"got {cohort_id!r}"
        )
    if type(phase) is not str or phase not in {"local_attn", "ffn_inflight"}:
        raise ValueError(f"Unsupported DECODE_ATTN cohort phase: {phase!r}")

    batch_replica_id = getattr(batch, "decode_attn_original_replica_id", None)
    batch_replica_local_id = getattr(
        batch, "decode_attn_original_replica_local_id", None
    )
    if batch_replica_id is None:
        batch_replica_id = replica_id
    if replica_local_id is not None:
        raise ValueError(
            "DECODE_ATTN cohort phase requires full-stage identity with "
            f"replica_local_id=None, got {replica_local_id!r}"
        )
    if batch_replica_local_id is not None:
        raise ValueError(
            "DECODE_ATTN cohort batch requires full-stage identity with "
            f"replica_local_id=None, got {batch_replica_local_id!r}"
        )
    batch_replica_local_id = None
    if type(batch_replica_id) is not int or batch_replica_id < 0:
        raise ValueError(
            "DECODE_ATTN cohort lane replica_id must be an exact "
            f"non-negative int, got {batch_replica_id!r}"
        )
    if type(layer_id) is not int and layer_id is not None:
        raise ValueError(
            "DECODE_ATTN cohort layer_id must be None or an exact int, "
            f"got {layer_id!r}"
        )
    if layer_id is not None and layer_id < 0:
        raise ValueError(
            "DECODE_ATTN cohort layer_id must be non-negative, "
            f"got {layer_id!r}"
        )

    replica_schedulers = getattr(scheduler, "_replica_schedulers", None)
    if type(replica_schedulers) is not dict:
        raise RuntimeError(
            "DECODE_ATTN replica scheduler topology must be an exact dict"
        )
    lane = (batch_replica_id, batch_replica_local_id)
    if lane not in replica_schedulers:
        raise ValueError(
            "DECODE_ATTN cohort lane is absent from the replica scheduler "
            f"topology: lane={lane}"
        )
    replica_scheduler = replica_schedulers[lane]
    cohort_states = getattr(
        replica_scheduler,
        "_decode_attn_active_cohort_states",
        {},
    )
    if type(cohort_states) is not dict:
        raise RuntimeError(
            "DECODE_ATTN active cohort states must be an exact dict"
        )
    cohort_state = cohort_states.get(cohort_id)
    if cohort_state is not None and type(cohort_state) is not dict:
        raise RuntimeError(
            "DECODE_ATTN active cohort state must be an exact dict"
        )

    afd_stage_idx = getattr(batch, "afd_stage_idx", None)
    if afd_stage_idx is not None and (
        type(afd_stage_idx) is not int or afd_stage_idx < 0
    ):
        raise ValueError(
            "DECODE_ATTN cohort afd_stage_idx must be None or an exact "
            f"non-negative int, got {afd_stage_idx!r}"
        )
    if cohort_state is not None and afd_stage_idx is not None:
        active_stage_indices, _, _ = (
            scheduler._validate_decode_attn_wave_stages(
                cohort_state,
                context="cohort phase update",
            )
        )
        if afd_stage_idx not in active_stage_indices:
            raise ValueError(
                "DECODE_ATTN cohort stage is not active: "
                f"stage={afd_stage_idx}, "
                f"active={sorted(active_stage_indices)}"
            )

    return {
        "batch": batch,
        "cohort_state": cohort_state,
        "cohort_id": cohort_id,
        "phase": phase,
        "layer_id": layer_id,
        "afd_stage_idx": afd_stage_idx,
    }

def apply_decode_attn_batch_phase(
    prepared_update: Optional[dict[str, Any]],
) -> None:
    """Commit a previously validated cohort phase update."""

    if prepared_update is None:
        return
    cohort_state = prepared_update["cohort_state"]
    if cohort_state is None:
        return

    phase = prepared_update["phase"]
    layer_id = prepared_update["layer_id"]
    afd_stage_idx = prepared_update["afd_stage_idx"]
    if afd_stage_idx is None:
        cohort_state["af_phase"] = phase
        if layer_id is not None:
            cohort_state["current_layer_id"] = layer_id
        return

    stage_phases = cohort_state["stage_phases"]
    stage_phases[afd_stage_idx] = phase
    if layer_id is not None:
        stage_layers = cohort_state["stage_current_layer_ids"]
        stage_layers[afd_stage_idx] = layer_id
        cohort_state["current_layer_id"] = layer_id

    phases = set(stage_phases.values())
    cohort_state["af_phase"] = phases.pop() if len(phases) == 1 else "mixed"

def commit_decode_attn_batch_phases(
    scheduler,
    prepared_updates: List[Optional[dict[str, Any]]],
    *,
    apply_fn: Callable[[Optional[dict[str, Any]]], None] = apply_decode_attn_batch_phase,
) -> None:
    """Apply prepared cohort updates atomically after all preflight checks."""

    if type(prepared_updates) is not list:
        raise RuntimeError(
            "DECODE_ATTN prepared cohort updates must be an exact list"
        )

    prospective_states: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for prepared_update in prepared_updates:
        if prepared_update is None:
            continue
        if type(prepared_update) is not dict:
            raise RuntimeError(
                "DECODE_ATTN prepared cohort update must be an exact dict"
            )
        cohort_state = prepared_update.get("cohort_state")
        if cohort_state is None:
            continue
        if type(cohort_state) is not dict:
            raise RuntimeError(
                "DECODE_ATTN prepared cohort state must be an exact dict"
            )

        state_key = id(cohort_state)
        state_pair = prospective_states.get(state_key)
        if state_pair is None:
            prospective_state = deepcopy(cohort_state)
            state_pair = (cohort_state, prospective_state)
            prospective_states[state_key] = state_pair
        else:
            prospective_state = state_pair[1]

        prospective_update = dict(prepared_update)
        prospective_update["cohort_state"] = prospective_state
        apply_fn(prospective_update)

    for cohort_state, prospective_state in prospective_states.values():
        cohort_state.clear()
        cohort_state.update(prospective_state)

def set_decode_attn_batch_phase(
    scheduler,
    batch: Batch,
    *,
    phase: str,
    replica_id: int,
    replica_local_id: int | None,
    layer_id: int | None = None,
    prepare_only: bool = False,
    prepare_fn: Callable[..., Optional[dict[str, Any]]] = prepare_decode_attn_batch_phase,
    apply_fn: Callable[[Optional[dict[str, Any]]], None] = apply_decode_attn_batch_phase,
) -> Optional[dict[str, Any]]:
    prepared_update = prepare_fn(
        scheduler,
        batch,
        phase=phase,
        replica_id=replica_id,
        replica_local_id=replica_local_id,
        layer_id=layer_id,
    )
    if prepare_only:
        return prepared_update
    apply_fn(prepared_update)
