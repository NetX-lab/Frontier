"""Decode-attention cohort state for the PD-AF decode-attention role.

A PD-AF ``DECODE_ATTN`` replica processes its requests in cohorts that move
through the attention and FFN stages together.  These methods own the cohort
identity, its stage slots and its phase.
"""

from typing import Dict, List, Optional

from frontier.config import global_vars
from frontier.entities.batch import Batch, Request
from frontier.logger import get_cluster_logger
from frontier.types import ClusterType


class DecodeAttentionCohort:
    """Cohort identity, stage slots and phase for the decode-attention role."""

    def _get_decode_attn_active_cohort_states(self) -> Dict[int, Dict[str, object]]:
        cohort_states = getattr(self, "_decode_attn_active_cohort_states", None)
        if cohort_states is None:
            cohort_states = {}
            self._decode_attn_active_cohort_states = cohort_states
        return cohort_states

    def _allocate_decode_attn_cohort_id(self) -> int:
        if self._cluster_type != ClusterType.DECODE_ATTN:
            raise ValueError(
                "DECODE_ATTN cohort IDs can only be allocated by a DECODE_ATTN scheduler"
            )
        cohort_id = int(getattr(self, "_decode_attn_next_cohort_id", 0))
        self._decode_attn_next_cohort_id = cohort_id + 1
        return cohort_id

    @staticmethod

    def _validate_decode_attn_wave_stages(
        cohort_state: Dict[str, object],
    ) -> tuple[set[int], Dict[int, str], Dict[int, int]]:
        if type(cohort_state) is not dict:
            raise RuntimeError("DECODE_ATTN active cohort state must be an exact dict")

        active_stage_indices = cohort_state.get("active_stage_indices")
        if type(active_stage_indices) is not set or not active_stage_indices:
            raise RuntimeError(
                "DECODE_ATTN active_stage_indices must be a non-empty exact set"
            )
        for stage_idx in active_stage_indices:
            if type(stage_idx) is not int or stage_idx < 0:
                raise RuntimeError(
                    "DECODE_ATTN active stage index must be an exact non-negative "
                    f"int, got {stage_idx!r}"
                )

        stage_phases = cohort_state.get("stage_phases")
        if type(stage_phases) is not dict:
            raise RuntimeError("DECODE_ATTN stage phases must be an exact dict")
        for stage_idx, stage_phase in stage_phases.items():
            if type(stage_idx) is not int or stage_idx < 0:
                raise RuntimeError(
                    "DECODE_ATTN stage phase index must be an exact non-negative "
                    f"int, got {stage_idx!r}"
                )
            if type(stage_phase) is not str or stage_phase not in {
                "local_attn",
                "ffn_inflight",
            }:
                raise RuntimeError(
                    "DECODE_ATTN stage phase must be local_attn or ffn_inflight, "
                    f"got {stage_phase!r}"
                )
        if set(stage_phases) != active_stage_indices:
            raise RuntimeError(
                "DECODE_ATTN stage phase key set must exactly match active stages: "
                f"phase_keys={sorted(stage_phases)}, "
                f"active={sorted(active_stage_indices)}"
            )

        stage_layers = cohort_state.get("stage_current_layer_ids")
        if type(stage_layers) is not dict:
            raise RuntimeError("DECODE_ATTN stage layers must be an exact dict")
        for stage_idx, stage_layer in stage_layers.items():
            if type(stage_idx) is not int or stage_idx < 0:
                raise RuntimeError(
                    "DECODE_ATTN stage layer index must be an exact non-negative "
                    f"int, got {stage_idx!r}"
                )
            if type(stage_layer) is not int or stage_layer < 0:
                raise RuntimeError(
                    "DECODE_ATTN stage layer must be an exact non-negative int, "
                    f"got {stage_layer!r}"
                )
        if set(stage_layers) != active_stage_indices:
            raise RuntimeError(
                "DECODE_ATTN stage layer key set must exactly match active stages: "
                f"layer_keys={sorted(stage_layers)}, "
                f"active={sorted(active_stage_indices)}"
            )

        return active_stage_indices, stage_phases, stage_layers

    def get_decode_attn_active_stage_slots(
        self,
        *,
        phase: str | None = None,
        layer_id: int | None = None,
    ) -> tuple[int, ...]:
        if self._cluster_type != ClusterType.DECODE_ATTN:
            return tuple()
        if phase is not None and type(phase) is not str:
            raise ValueError(
                "DECODE_ATTN active-stage phase filter must be an exact str, "
                f"got {phase!r}"
            )
        if layer_id is not None and (
            type(layer_id) is not int or layer_id < 0
        ):
            raise ValueError(
                "DECODE_ATTN active-stage layer filter must be an exact "
                f"non-negative int, got {layer_id!r}"
            )

        active_stage_slots: set[int] = set()
        cohort_states = getattr(self, "_decode_attn_active_cohort_states", {})
        if type(cohort_states) is not dict:
            raise RuntimeError(
                "DECODE_ATTN active cohort states must be an exact dict"
            )
        for cohort_id, state in cohort_states.items():
            if type(cohort_id) is not int or cohort_id < 0:
                raise RuntimeError(
                    "DECODE_ATTN active cohort ID must be an exact non-negative "
                    f"int, got {cohort_id!r}"
                )
            if type(state) is not dict:
                raise RuntimeError(
                    "DECODE_ATTN active cohort state must be an exact dict"
                )
            pending_request_ids = state.get("pending_request_ids")
            if type(pending_request_ids) is not set:
                raise RuntimeError(
                    "DECODE_ATTN pending request IDs must be an exact set, "
                    f"got {pending_request_ids!r}"
                )
            for request_id in pending_request_ids:
                if type(request_id) is not int or request_id < 0:
                    raise RuntimeError(
                        "DECODE_ATTN pending request ID must be an exact "
                        f"non-negative int, got {request_id!r}"
                    )
            if not pending_request_ids:
                continue

            active_indices, stage_phases, stage_layers = (
                self._validate_decode_attn_wave_stages(state)
            )
            for stage_idx in active_indices:
                stage_layer = stage_layers[stage_idx]
                if (
                    layer_id is not None
                    and stage_layer != layer_id
                ):
                    continue
                stage_phase = stage_phases[stage_idx]
                if phase is not None and stage_phase != phase:
                    continue
                active_stage_slots.add(stage_idx)

        return tuple(sorted(active_stage_slots))

    def set_decode_attn_cohort_phase_for_batch(
        self,
        batch: Batch,
        *,
        phase: str,
        layer_id: int | None = None,
    ) -> None:
        if self._cluster_type != ClusterType.DECODE_ATTN:
            return

        cohort_id = batch.decode_attn_cohort_id
        if cohort_id is None:
            return

        normalized_phase = str(phase)
        if normalized_phase not in {"local_attn", "ffn_inflight"}:
            raise ValueError(
                f"Unsupported DECODE_ATTN cohort phase: {normalized_phase}"
            )

        cohort_state = self._get_decode_attn_active_cohort_states().get(
            int(cohort_id)
        )
        if cohort_state is None:
            return

        stage_idx = batch.afd_stage_idx
        if stage_idx is None:
            cohort_state["af_phase"] = normalized_phase
            if layer_id is not None:
                cohort_state["current_layer_id"] = int(layer_id)
            return

        active_stage_indices, stage_phases, stage_layers = (
            self._validate_decode_attn_wave_stages(cohort_state)
        )
        normalized_stage_idx = int(stage_idx)
        if normalized_stage_idx not in active_stage_indices:
            raise ValueError(
                "DECODE_ATTN cohort stage is not active: "
                f"stage={normalized_stage_idx}, active={sorted(active_stage_indices)}"
            )
        stage_phases[normalized_stage_idx] = normalized_phase
        if layer_id is not None:
            stage_layers[normalized_stage_idx] = int(layer_id)
            cohort_state["current_layer_id"] = int(layer_id)

        phases = set(stage_phases.values())
        cohort_state["af_phase"] = phases.pop() if len(phases) == 1 else "mixed"
