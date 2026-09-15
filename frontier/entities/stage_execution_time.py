"""Ordered per-layer execution timing and stage-level aggregation.

``ExecutionTime`` describes one real transformer layer.  This module owns the
stage view used by schedulers and metrics: it keeps the layer records in their
execution order and charges stage-owned work exactly once.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import fields
import math
from types import MappingProxyType
from typing import Any

from frontier.entities.execution_time import ExecutionTime
from frontier.entities.time_components import (
    AttentionOperatorTimes,
    AttentionTime,
    CommunicationOperatorTimes,
    CommunicationTime,
    MLPOperatorTimes,
    MLPTime,
    MoEOperatorTimes,
    MoETime,
)


_PER_LAYER_PUBLIC_NAMES = frozenset(
    {
        "attention_time",
        "moe_comm_time",
        "moe_comp_time",
        "mlp_layer_up_proj_execution_time",
        "mlp_layer_down_proj_execution_time",
        "mlp_layer_act_execution_time",
        "mlp_all_reduce_time",
        "mlp_norm_time",
        "attention_pre_proj_time",
        "attention_post_proj_time",
        "attention_all_reduce_time",
        "moe_tensor_parallel_allgather_time",
        "share_expert_tensor_parallel_allreduce_time",
        "attention_rope_execution_time",
        "attention_kv_cache_save_execution_time",
        "attention_decode_execution_time",
        "attention_prefill_execution_time",
        "attn_mla_kv_cache_save_time",
        "attn_mla_prefill_kv_up_proj_time",
        "attn_mla_prefill_time",
        "attn_mla_decode_q_latent_proj_time",
        "attn_mla_decode_time",
        "attn_mla_v_up_proj_time",
        "attn_norm_time",
        "dp_input_allreduce_time",
        "dp_output_allreduce_time",
        "add_time",
        "add_attn_residual_time",
        "add_ffn_residual_time",
        "moe_grouped_gemm_time",
        "expert_parallel_communication_time",
        "moe_gating_time",
        "moe_gating_linear_time",
        "moe_gating_routing_topk_time",
        "moe_shuffling_time",
        "share_expert_up_proj_time",
        "share_expert_down_proj_time",
        "share_expert_act_time",
        "share_expert_time",
    }
)

_STAGE_ONLY_PUBLIC_NAMES = frozenset(
    {
        "pipeline_time",
        "pipeline_parallel_communication_time",
        "schedule_time",
        "sampler_e2e_time",
        "prepare_inputs_e2e_time",
        "process_model_outputs_time",
        "ray_comm_time",
        "pp_receiver_head_runtime_time",
        "pp_producer_send_path_runtime_time",
        "pp_prefill_consumer_active_runtime_time",
        "pp_stage_boundary_residual_runtime_time",
        "pp_stage_boundary_handoff_time",
        "decode_draft_proposer_time",
        "mtp_terminal_overshoot_time",
    }
)

_PER_LAYER_PRIVATE_NAMES = frozenset(
    {
        "_attention_rope_execution_time",
        "_attention_kv_cache_save_execution_time",
        "_attention_decode_execution_time",
        "_attention_prefill_execution_time",
        "_attention_layer_pre_proj_execution_time",
        "_attention_layer_post_proj_execution_time",
        "_attn_mla_kv_cache_save_time",
        "_attn_mla_prefill_kv_up_proj_time",
        "_attn_mla_prefill_time",
        "_attn_mla_decode_q_latent_proj_time",
        "_attn_mla_decode_time",
        "_attn_mla_v_up_proj_time",
        "_attn_norm_time",
        "_mlp_layer_up_proj_execution_time",
        "_mlp_layer_down_proj_execution_time",
        "_mlp_layer_act_execution_time",
        "_mlp_norm_time",
        "_add_time",
        "_add_attn_residual_time",
        "_add_ffn_residual_time",
        "_tensor_parallel_communication_time",
        "_attn_tensor_parallel_allreduce_time",
        "_moe_tensor_parallel_allreduce_time",
        "_tensor_parallel_allgather_time",
        "_share_expert_tensor_parallel_allreduce_time",
        "_dp_input_allreduce_time",
        "_dp_output_allreduce_time",
        "_expert_parallel_communication_time",
        "_moe_gating_time",
        "_moe_gating_linear_time",
        "_moe_gating_routing_topk_time",
        "_moe_shuffling_time",
        "_moe_grouped_gemm_time",
        "_share_expert_up_proj_time",
        "_share_expert_down_proj_time",
        "_share_expert_act_time",
    }
)


class StageExecutionTime:
    """Ordered real-layer timings plus stage-owned work.

    The class deliberately does not inherit ``BaseEntity``.  A timing record
    is diagnostic/scheduling data and must not consume a simulator entity ID.
    ``stage_execution_time`` is retained as an owner record for transfer,
    CPU-overhead, and terminal fields.  Its per-layer component values are
    ignored for model aggregation; the ordered ``layer_execution_times`` are
    authoritative for all layer work.
    """

    def __init__(
        self,
        layer_execution_times: Sequence[ExecutionTime],
        *,
        stage_execution_time: ExecutionTime | None = None,
    ) -> None:
        if not isinstance(layer_execution_times, Sequence):
            raise TypeError("layer_execution_times must be a sequence")
        if not layer_execution_times:
            raise ValueError("StageExecutionTime requires at least one layer result")
        if any(not isinstance(layer, ExecutionTime) for layer in layer_execution_times):
            raise TypeError("StageExecutionTime layer results must be ExecutionTime objects")
        self._layer_execution_times = tuple(layer_execution_times)
        self._stage_execution_time = (
            stage_execution_time
            if stage_execution_time is not None
            else self._layer_execution_times[0]
        )
        if not isinstance(self._stage_execution_time, ExecutionTime):
            raise TypeError("stage_execution_time must be an ExecutionTime object")

        layer_ids = self.global_layer_ids
        if any(layer_id is not None for layer_id in layer_ids) and any(
            layer_id is None for layer_id in layer_ids
        ):
            raise ValueError(
                "StageExecutionTime layer identities must be complete when provided"
            )
        if len([layer_id for layer_id in layer_ids if layer_id is not None]) != len(
            set(layer_id for layer_id in layer_ids if layer_id is not None)
        ):
            raise ValueError("StageExecutionTime global_layer_ids must be unique")

    @classmethod
    def from_execution_time(
        cls,
        source: ExecutionTime,
        *,
        num_layers: int,
        first_layer_id: int | None = 0,
        attention_family_id: str = "dense_attention",
        attention_variant_id: str = "unknown",
        layer_id_step: int = 1,
    ) -> "StageExecutionTime":
        """Expand one per-layer source payload into an ordered stage result.

        Predictors use one source payload for equal attention queries.  Each
        expansion receives a separate copied layer object and identity, while
        the owner remains a single stage record for once-only work.
        """

        if not isinstance(source, ExecutionTime):
            raise TypeError("source must be an ExecutionTime object")
        if type(num_layers) is not int or num_layers <= 0:
            raise ValueError("num_layers must be a positive int")
        if type(layer_id_step) is not int or layer_id_step <= 0:
            raise ValueError("layer_id_step must be a positive int")
        if first_layer_id is not None and (
            type(first_layer_id) is not int or first_layer_id < 0
        ):
            raise ValueError("first_layer_id must be a non-negative int or None")

        if num_layers == 1 and source.global_layer_id is not None:
            return cls((source,), stage_execution_time=source)

        # A resolved source identity is authoritative for callers that already
        # selected a concrete layer family.  Homogeneous legacy callers still
        # receive the historical dense/unknown defaults.
        resolved_attention_family_id = (
            source.attention_family_id
            if source.attention_family_id is not None
            else attention_family_id
        )
        resolved_attention_variant_id = (
            source.attention_variant_id
            if source.attention_variant_id is not None
            else attention_variant_id
        )

        layers: list[ExecutionTime] = []
        for offset in range(num_layers):
            global_layer_id = (
                None
                if first_layer_id is None
                else first_layer_id + offset * layer_id_step
            )
            layers.append(
                source.as_single_layer(
                    global_layer_id=(
                        0 if global_layer_id is None else global_layer_id
                    ),
                    attention_family_id=resolved_attention_family_id,
                    attention_variant_id=resolved_attention_variant_id,
                )
            )
            if global_layer_id is None:
                layers[-1]._global_layer_id = None
        return cls(tuple(layers), stage_execution_time=source)

    @property
    def layer_execution_times(self) -> tuple[ExecutionTime, ...]:
        """Return ordered layer results without exposing mutable containers."""

        return self._layer_execution_times

    @property
    def stage_execution_time(self) -> ExecutionTime:
        """Return the stage owner record for once-only work."""

        return self._stage_execution_time

    @property
    def num_layers(self) -> int:
        return len(self._layer_execution_times)

    @property
    def global_layer_ids(self) -> tuple[int | None, ...]:
        return tuple(layer.global_layer_id for layer in self._layer_execution_times)

    @property
    def attention_family_ids(self) -> tuple[str | None, ...]:
        return tuple(layer.attention_family_id for layer in self._layer_execution_times)

    @property
    def attention_variant_ids(self) -> tuple[str | None, ...]:
        return tuple(layer.attention_variant_id for layer in self._layer_execution_times)

    @staticmethod
    def _sum_values(values: Iterable[float]) -> float:
        return math.fsum(float(value) for value in values)

    @staticmethod
    def _sum_operator_maps(
        operator_sources: Iterable[Any],
        *,
        exclude: frozenset[str] = frozenset(),
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        for source in operator_sources:
            if source is None:
                continue
            for name, value in source.op_times.items():
                if name in exclude:
                    continue
                values[name] = values.get(name, 0.0) + float(value)
        return values

    @property
    def attention_operator_times(self) -> AttentionOperatorTimes | None:
        """Return the first layer's operator view for legacy layer probes.

        Stage-wide maps are available through ``op_times`` and the private
        aggregate helpers used by that property.  Keeping this accessor at
        layer scope preserves existing trace consumers that inspect a single
        representative layer while the layer records remain authoritative.
        """

        return self._first_layer().attention_operator_times

    def _aggregate_attention_operator_times(self) -> AttentionOperatorTimes | None:
        values = self._sum_operator_maps(
            layer.attention_operator_times for layer in self._layer_execution_times
        )
        return AttentionOperatorTimes(values) if values else None

    @property
    def mlp_operator_times(self) -> MLPOperatorTimes | None:
        return self._first_layer().mlp_operator_times

    def _aggregate_mlp_operator_times(self) -> MLPOperatorTimes | None:
        values = self._sum_operator_maps(
            layer.mlp_operator_times for layer in self._layer_execution_times
        )
        return MLPOperatorTimes(values) if values else None

    @property
    def moe_operator_times(self) -> MoEOperatorTimes | None:
        return self._first_layer().moe_operator_times

    def _aggregate_moe_operator_times(self) -> MoEOperatorTimes | None:
        values = self._sum_operator_maps(
            layer.moe_operator_times for layer in self._layer_execution_times
        )
        return MoEOperatorTimes(values) if values else None

    @property
    def communication_operator_times(self) -> CommunicationOperatorTimes | None:
        first_layer_values = self._sum_operator_maps(
            (self._first_layer().communication_operator_times,),
            exclude=frozenset({"pipeline_parallel_send_recv"}),
        )
        first_owner = self._stage_execution_time.communication_operator_times
        if first_owner is not None and "pipeline_parallel_send_recv" in first_owner.op_times:
            first_layer_values["pipeline_parallel_send_recv"] = float(
                first_owner.op_times["pipeline_parallel_send_recv"]
            )
        return (
            CommunicationOperatorTimes(first_layer_values)
            if first_layer_values
            else None
        )

    def _aggregate_communication_operator_times(
        self,
    ) -> CommunicationOperatorTimes | None:
        values = self._sum_operator_maps(
            (
                layer.communication_operator_times
                for layer in self._layer_execution_times
            ),
            exclude=frozenset({"pipeline_parallel_send_recv"}),
        )
        owner = self._stage_execution_time.communication_operator_times
        if owner is not None:
            pipeline_name = "pipeline_parallel_send_recv"
            if pipeline_name in owner.op_times:
                values[pipeline_name] = float(owner.op_times[pipeline_name])
        return CommunicationOperatorTimes(values) if values else None

    @property
    def op_times(self) -> Mapping[str, float]:
        values: dict[str, float] = {}
        for operator_times in (
            self._aggregate_attention_operator_times(),
            self._aggregate_mlp_operator_times(),
            self._aggregate_moe_operator_times(),
            self._aggregate_communication_operator_times(),
        ):
            if operator_times is None:
                continue
            for name, value in operator_times.op_times.items():
                values[name] = float(value)
        return MappingProxyType(values)

    def _layer_value(self, name: str) -> float:
        return self._sum_values(getattr(layer, name) for layer in self._layer_execution_times)

    def _owner_value(self, name: str) -> float:
        return float(getattr(self._stage_execution_time, name))

    @property
    def attention_time_component(self) -> AttentionTime:
        return self._first_layer().attention_time_component

    @property
    def communication_time_component(self) -> CommunicationTime:
        component = self._first_layer().communication_time_component
        component.pipeline_parallel_send_recv_time = self._owner_value(
            "pipeline_parallel_communication_time"
        )
        component.operator_times = self.communication_operator_times
        return component

    @property
    def moe_or_mlp_time_component(self) -> MLPTime | MoETime:
        return self._first_layer().moe_or_mlp_time_component

    @property
    def model_time_ms(self) -> float:
        block_time_ms = self._sum_values(
            layer.get_single_layer_block_time()
            for layer in self._layer_execution_times
        )
        return (
            block_time_ms
            + self._owner_value("pipeline_parallel_communication_time")
            + self._owner_value("decode_draft_proposer_time")
            + self._owner_value("mtp_terminal_overshoot_time")
        )

    @property
    def model_time(self) -> float:
        return self.model_time_ms * 1e-3

    @property
    def total_time(self) -> float:
        return self.model_time + self._owner_cpu_overhead_ms() * 1e-3

    @property
    def diagnostic_total_time(self) -> float:
        return self.model_time + self._owner_diagnostic_cpu_overhead_ms() * 1e-3

    @property
    def diagnostic_total_time_ms(self) -> float:
        return self.diagnostic_total_time * 1e3

    def _owner_cpu_overhead_ms(self) -> float:
        return float(self._stage_execution_time._get_cpu_overhead())

    def _owner_diagnostic_cpu_overhead_ms(self) -> float:
        return float(self._stage_execution_time._get_diagnostic_cpu_overhead())

    # Historical accessors are one-layer probes.  Stage totals are exposed by
    # model_time and the aggregate component properties above; the probes keep
    # event paths that process one layer at a time source-compatible.
    def _first_layer(self) -> ExecutionTime:
        return self._layer_execution_times[0]

    def get_single_layer_attention_time(self) -> float:
        return self._first_layer().get_single_layer_attention_time()

    def get_single_layer_attention_scope_time(self) -> float:
        return self._first_layer().get_single_layer_attention_scope_time()

    def get_single_layer_moe_comp_time(self) -> float:
        return self._first_layer().get_single_layer_moe_comp_time()

    def get_single_layer_moe_pre_dispatch_time(self) -> float:
        return self._first_layer().get_single_layer_moe_pre_dispatch_time()

    def get_single_layer_moe_dispatch_time(self) -> float:
        return self._first_layer().get_single_layer_moe_dispatch_time()

    def get_single_layer_moe_post_dispatch_compute_time(self) -> float:
        return self._first_layer().get_single_layer_moe_post_dispatch_compute_time()

    def get_single_layer_moe_combine_time(self) -> float:
        return self._first_layer().get_single_layer_moe_combine_time()

    def get_single_layer_moe_post_combine_time(self) -> float:
        return self._first_layer().get_single_layer_moe_post_combine_time()

    def get_single_layer_moe_comm_time(self) -> float:
        return self._first_layer().get_single_layer_moe_comm_time()

    def get_single_layer_add_time(self) -> float:
        return self._first_layer().get_single_layer_add_time()

    def get_single_layer_dp_input_allreduce_time(self) -> float:
        return self._first_layer().get_single_layer_dp_input_allreduce_time()

    def get_single_layer_dp_output_allreduce_time(self) -> float:
        return self._first_layer().get_single_layer_dp_output_allreduce_time()

    def get_single_layer_block_time(self) -> float:
        return self._first_layer().get_single_layer_block_time()

    def get_single_layer_post_attention_time(self) -> float:
        return self._first_layer().get_single_layer_post_attention_time()

    def __getattr__(self, name: str) -> Any:
        if name in _PER_LAYER_PUBLIC_NAMES:
            return self._layer_value(name)
        if name in _STAGE_ONLY_PUBLIC_NAMES:
            return self._owner_value(name)
        if name in _PER_LAYER_PRIVATE_NAMES:
            # Metrics adapters and legacy trace builders read private fields
            # as single-layer source values before constructing their own
            # one-layer payload.  Stage totals are exposed through
            # ``model_time`` and the aggregate public component properties;
            # returning a summed private field here would make that adapter
            # multiply a stage twice.
            return getattr(self._first_layer(), name)
        if name == "_is_moe":
            values = {bool(getattr(layer, name)) for layer in self._layer_execution_times}
            return values.pop() if len(values) == 1 else None
        if name == "_num_layers_per_pipeline_stage":
            return self.num_layers
        if name.startswith("_"):
            owner = self._stage_execution_time
            if hasattr(owner, name):
                return getattr(owner, name)
        raise AttributeError(name)


__all__ = ["StageExecutionTime"]
