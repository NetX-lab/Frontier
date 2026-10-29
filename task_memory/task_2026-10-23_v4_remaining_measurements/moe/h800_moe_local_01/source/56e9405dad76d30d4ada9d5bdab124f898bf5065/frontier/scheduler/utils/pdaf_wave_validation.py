"""Validation for DECODE_ATTN stage-local forward waves."""

from typing import Any


def validate_wave_stages(
    wave_state: dict[str, Any], *, context: str
) -> tuple[set[int], dict[int, str], dict[int, int]]:
    """Validate active stages, phases, and layer IDs for one wave."""

    if type(wave_state) is not dict:
        raise RuntimeError(f"DECODE_ATTN {context} active cohort state must be an exact dict")
    active = wave_state.get("active_stage_indices")
    if type(active) is not set or not active:
        raise RuntimeError(f"DECODE_ATTN {context} cohort active_stage_indices must be a non-empty exact set")
    for stage_idx in active:
        if type(stage_idx) is not int or stage_idx < 0:
            raise RuntimeError(f"DECODE_ATTN {context} cohort active stage indices must contain exact non-negative ints, got {stage_idx!r}")
    phases = wave_state.get("stage_phases")
    if type(phases) is not dict:
        raise RuntimeError(f"DECODE_ATTN {context} cohort stage phases must be an exact dict")
    for stage_idx, phase in phases.items():
        if type(stage_idx) is not int or stage_idx < 0:
            raise RuntimeError(f"DECODE_ATTN {context} cohort stage phase indices must contain exact non-negative ints, got {stage_idx!r}")
        if type(phase) is not str or phase not in {"local_attn", "ffn_inflight"}:
            raise RuntimeError(f"DECODE_ATTN {context} cohort stage phase must be local_attn or ffn_inflight, got {phase!r}")
    if set(phases) != active:
        raise RuntimeError(f"DECODE_ATTN {context} cohort stage phase key set must exactly match active stages: phase_keys={sorted(phases)}, active={sorted(active)}")
    layers = wave_state.get("stage_current_layer_ids")
    if type(layers) is not dict:
        raise RuntimeError(f"DECODE_ATTN {context} cohort stage layers must be an exact dict")
    for stage_idx, layer_id in layers.items():
        if type(stage_idx) is not int or stage_idx < 0:
            raise RuntimeError(f"DECODE_ATTN {context} cohort stage layer indices must contain exact non-negative ints, got {stage_idx!r}")
        if type(layer_id) is not int or layer_id < 0:
            raise RuntimeError(f"DECODE_ATTN {context} cohort stage layer must be an exact non-negative int, got {layer_id!r}")
    if set(layers) != active:
        raise RuntimeError(f"DECODE_ATTN {context} cohort stage layer key set must exactly match active stages: layer_keys={sorted(layers)}, active={sorted(active)}")
    return active, phases, layers


def validate_a2f_wave_phase(
    scheduler: Any,
    batch: Any,
    *,
    layer_id: int,
    afd_stage_idx: int,
    context: str,
) -> None:
    """Validate the stage-local phase and layer of an A-to-F wave."""

    wave_id = getattr(batch, "decode_attn_cohort_id", None)
    lane = (
        getattr(batch, "decode_attn_original_replica_id", None),
        getattr(batch, "decode_attn_original_replica_local_id", None),
    )
    replica_schedulers = getattr(scheduler, "_replica_schedulers", None)
    if type(replica_schedulers) is not dict or lane not in replica_schedulers:
        raise ValueError(f"DECODE_ATTN A-to-F {context} cohort lane is absent from the replica scheduler topology: lane={lane}")
    wave_states = getattr(replica_schedulers[lane], "_decode_attn_active_cohort_states", None)
    if type(wave_states) is not dict:
        raise RuntimeError(f"DECODE_ATTN A-to-F {context} active cohort states must be an exact dict")
    wave_state = wave_states.get(wave_id)
    if type(wave_state) is not dict:
        raise ValueError(f"DECODE_ATTN A-to-F {context} references an inactive or unknown cohort: cohort_id={wave_id}, lane={lane}")
    active, phases, layers = validate_wave_stages(wave_state, context=f"A-to-F {context}")
    if afd_stage_idx not in active:
        raise ValueError(f"DECODE_ATTN A-to-F {context} stage is not active in the cohort: stage={afd_stage_idx}, active={sorted(active)}")
    aggregate_phase = wave_state.get("af_phase")
    if type(aggregate_phase) is not str:
        raise RuntimeError(f"DECODE_ATTN A-to-F {context} cohort af_phase must be an exact str, got {aggregate_phase!r}")
    aggregate_layer = wave_state.get("current_layer_id")
    if type(aggregate_layer) is not int or aggregate_layer < 0:
        raise RuntimeError(f"DECODE_ATTN A-to-F {context} cohort current_layer_id must be an exact non-negative int, got {aggregate_layer!r}")
    if phases[afd_stage_idx] != "local_attn":
        raise ValueError(f"DECODE_ATTN A-to-F {context} cohort stage is not in local_attn phase: stage={afd_stage_idx}, phase={phases[afd_stage_idx]!r}")
    if layers[afd_stage_idx] != layer_id:
        raise ValueError(f"DECODE_ATTN A-to-F {context} cohort layer mismatch: expected={layer_id}, got={layers[afd_stage_idx]}")
