from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Union

from frontier.entities.base_entity import BaseEntity
from frontier.entities.time_components import (
    AttentionTime,
    AttentionOperatorTimes,
    build_attention_operator_times_from_op_times,
    build_communication_operator_times_from_op_times,
    build_mlp_operator_times_from_op_times,
    build_moe_operator_times_from_op_times,
    CommunicationOperatorTimes,
    execution_op_time_values_by_attr,
    MLPOperatorTimes,
    MLPTime,
    MoEOperatorTimes,
    MoETime,
    CommunicationTime,
    normalize_execution_op_times,
    OverheadTime,
    ResidualTime,
)


class ExecutionTime(BaseEntity):
    """
    Aggregated execution time for a batch processing through pipeline stage(s).

    Uses composition of time components (AttentionTime, MLPTime/MoETime,
    CommunicationTime, OverheadTime, ResidualTime) to support cluster-specific
    execution time modeling.

    For single-layer granularity (PD+AF disaggregation): num_layers_per_pipeline_stage=1
    For multi-layer aggregation (monolithic): num_layers_per_pipeline_stage=N
    """

    def __init__(
        self,
        num_layers_per_pipeline_stage: int,
        attention_rope_execution_time: float,
        attention_kv_cache_save_execution_time: float,
        attention_decode_execution_time: float,
        attention_prefill_execution_time: float,
        attention_layer_pre_proj_execution_time: float,
        attention_layer_post_proj_execution_time: float,
        attn_norm_time: float,
        mlp_norm_time: float,
        add_time: float,  # Deprecated: use add_attn_residual_time + add_ffn_residual_time
        tensor_parallel_communication_time: float,
        pipeline_parallel_communication_time: float,
        expert_parallel_communication_time: float,
        moe_gating_time: float,  # Deprecated: use moe_gating_linear_time + moe_gating_routing_topk_time
        moe_shuffling_time: float,
        schedule_time: float,
        sampler_e2e_time: float,
        prepare_inputs_e2e_time: float,
        process_model_outputs_time: float,
        ray_comm_time: float,
        is_moe: bool,
        pp_producer_send_path_runtime_time: float = 0.0,
        pp_receiver_head_runtime_time: float = 0.0,
        pp_prefill_consumer_active_runtime_time: float = 0.0,
        pp_stage_boundary_residual_runtime_time: float = 0.0,
        mlp_layer_up_proj_execution_time: float = 0.0,
        mlp_layer_down_proj_execution_time: float = 0.0,
        mlp_layer_act_execution_time: float = 0.0,
        moe_grouped_gemm_time: float = 0.0,
        moe_gating_linear_time: float = 0.0,
        moe_gating_routing_topk_time: float = 0.0,
        add_attn_residual_time: float = 0.0,
        add_ffn_residual_time: float = 0.0,
        share_expert_up_proj_time: float = 0.0,
        share_expert_down_proj_time: float = 0.0,
        share_expert_act_time: float = 0.0,
        tensor_parallel_allgather_time: float = 0.0,
        share_expert_tensor_parallel_allreduce_time: float = 0.0,
        dp_input_allreduce_time: float = 0.0,
        dp_output_allreduce_time: float = 0.0,
        attn_tensor_parallel_allreduce_time: float | None = None,
        moe_tensor_parallel_allreduce_time: float | None = None,
        pp_stage_boundary_handoff_time: float = 0.0,
        decode_draft_proposer_time: float = 0.0,
        mtp_terminal_overshoot_time: float = 0.0,
        attn_mla_kv_cache_save_time: float = 0.0,
        attn_mla_prefill_kv_up_proj_time: float = 0.0,
        attn_mla_prefill_time: float = 0.0,
        attn_mla_decode_q_latent_proj_time: float = 0.0,
        attn_mla_decode_time: float = 0.0,
        attn_mla_v_up_proj_time: float = 0.0,
        attention_operator_times: AttentionOperatorTimes | None = None,
        communication_operator_times: CommunicationOperatorTimes | None = None,
        mlp_operator_times: MLPOperatorTimes | None = None,
        moe_operator_times: MoEOperatorTimes | None = None,
        op_times: Mapping[str, float] | None = None,
    ) -> None:
        self._id = ExecutionTime.generate_id()

        self._num_layers_per_pipeline_stage = num_layers_per_pipeline_stage
        self._is_moe = is_moe
        if is_moe and mlp_operator_times is not None:
            raise ValueError("mlp_operator_times are only valid for dense MLP components")
        if not is_moe and moe_operator_times is not None:
            raise ValueError("moe_operator_times are only valid for MoE components")

        merged_op_times = self._merge_operator_time_sources(
            normalize_execution_op_times(op_times or {}),
            attention_operator_times,
            mlp_operator_times,
            moe_operator_times,
            communication_operator_times,
        )
        if attention_operator_times is None:
            attention_operator_times = build_attention_operator_times_from_op_times(
                merged_op_times
            )
        if is_moe:
            if moe_operator_times is None:
                moe_operator_times = build_moe_operator_times_from_op_times(
                    merged_op_times
                )
        elif mlp_operator_times is None:
            mlp_operator_times = build_mlp_operator_times_from_op_times(merged_op_times)
        if communication_operator_times is None:
            communication_operator_times = (
                build_communication_operator_times_from_op_times(merged_op_times)
            )
        attention_operator_times = deepcopy(attention_operator_times)
        mlp_operator_times = deepcopy(mlp_operator_times)
        moe_operator_times = deepcopy(moe_operator_times)
        communication_operator_times = deepcopy(communication_operator_times)

        op_time_attr_values = execution_op_time_values_by_attr(merged_op_times)

        attn_tensor_parallel_allreduce_is_provided = (
            "attn_tensor_parallel_allreduce_time" in op_time_attr_values
            or attn_tensor_parallel_allreduce_time is not None
        )
        moe_tensor_parallel_allreduce_is_provided = (
            "moe_tensor_parallel_allreduce_time" in op_time_attr_values
            or moe_tensor_parallel_allreduce_time is not None
        )

        def read_op_time_attr(attr_name: str, legacy_value: float | None) -> float:
            return op_time_attr_values.get(
                attr_name,
                0.0 if legacy_value is None else legacy_value,
            )

        attention_kv_cache_save_execution_time = read_op_time_attr(
            "attention_kv_cache_save_execution_time",
            attention_kv_cache_save_execution_time,
        )
        attention_decode_execution_time = read_op_time_attr(
            "attention_decode_execution_time",
            attention_decode_execution_time,
        )
        attention_prefill_execution_time = read_op_time_attr(
            "attention_prefill_execution_time",
            attention_prefill_execution_time,
        )
        attn_norm_time = read_op_time_attr("attn_norm_time", attn_norm_time)
        mlp_norm_time = read_op_time_attr("mlp_norm_time", mlp_norm_time)
        add_attn_residual_time = read_op_time_attr(
            "add_attn_residual_time",
            add_attn_residual_time,
        )
        add_ffn_residual_time = read_op_time_attr(
            "add_ffn_residual_time",
            add_ffn_residual_time,
        )
        mlp_layer_up_proj_execution_time = read_op_time_attr(
            "mlp_layer_up_proj_execution_time",
            mlp_layer_up_proj_execution_time,
        )
        mlp_layer_down_proj_execution_time = read_op_time_attr(
            "mlp_layer_down_proj_execution_time",
            mlp_layer_down_proj_execution_time,
        )
        mlp_layer_act_execution_time = read_op_time_attr(
            "mlp_layer_act_execution_time",
            mlp_layer_act_execution_time,
        )
        moe_grouped_gemm_time = read_op_time_attr(
            "moe_grouped_gemm_time",
            moe_grouped_gemm_time,
        )
        moe_gating_linear_time = read_op_time_attr(
            "moe_gating_linear_time",
            moe_gating_linear_time,
        )
        moe_gating_routing_topk_time = read_op_time_attr(
            "moe_gating_routing_topk_time",
            moe_gating_routing_topk_time,
        )
        moe_shuffling_time = read_op_time_attr(
            "moe_shuffling_time",
            moe_shuffling_time,
        )
        share_expert_up_proj_time = read_op_time_attr(
            "share_expert_up_proj_time",
            share_expert_up_proj_time,
        )
        share_expert_down_proj_time = read_op_time_attr(
            "share_expert_down_proj_time",
            share_expert_down_proj_time,
        )
        share_expert_act_time = read_op_time_attr(
            "share_expert_act_time",
            share_expert_act_time,
        )
        attn_tensor_parallel_allreduce_time = read_op_time_attr(
            "attn_tensor_parallel_allreduce_time",
            attn_tensor_parallel_allreduce_time,
        )
        moe_tensor_parallel_allreduce_time = read_op_time_attr(
            "moe_tensor_parallel_allreduce_time",
            moe_tensor_parallel_allreduce_time,
        )
        tensor_parallel_allgather_time = read_op_time_attr(
            "tensor_parallel_allgather_time",
            tensor_parallel_allgather_time,
        )
        share_expert_tensor_parallel_allreduce_time = read_op_time_attr(
            "share_expert_tensor_parallel_allreduce_time",
            share_expert_tensor_parallel_allreduce_time,
        )
        pipeline_parallel_communication_time = read_op_time_attr(
            "pipeline_parallel_send_recv_time",
            pipeline_parallel_communication_time,
        )
        attn_mla_kv_cache_save_time = read_op_time_attr(
            "attn_mla_kv_cache_save_time",
            attn_mla_kv_cache_save_time,
        )
        attn_mla_prefill_kv_up_proj_time = read_op_time_attr(
            "attn_mla_prefill_kv_up_proj_time",
            attn_mla_prefill_kv_up_proj_time,
        )
        attn_mla_prefill_time = read_op_time_attr(
            "attn_mla_prefill_time",
            attn_mla_prefill_time,
        )
        attn_mla_decode_q_latent_proj_time = read_op_time_attr(
            "attn_mla_decode_q_latent_proj_time",
            attn_mla_decode_q_latent_proj_time,
        )
        attn_mla_decode_time = read_op_time_attr(
            "attn_mla_decode_time",
            attn_mla_decode_time,
        )
        attn_mla_v_up_proj_time = read_op_time_attr(
            "attn_mla_v_up_proj_time",
            attn_mla_v_up_proj_time,
        )
        expert_parallel_alltoall_time = read_op_time_attr(
            "expert_parallel_alltoall_time",
            0.0,
        )
        expert_parallel_communication_time = read_op_time_attr(
            "expert_parallel_alltoall_time",
            expert_parallel_communication_time,
        )

        # Handle backward compatibility: if new fields are not provided, split moe_gating_time equally
        if moe_gating_linear_time == 0.0 and moe_gating_routing_topk_time == 0.0 and moe_gating_time > 0.0:
            # Legacy mode: split the old moe_gating_time equally between the two new fields
            moe_gating_linear_time = moe_gating_time * 0.5
            moe_gating_routing_topk_time = moe_gating_time * 0.5

        # Handle backward compatibility: if new add fields are not provided, split add_time equally
        if add_attn_residual_time == 0.0 and add_ffn_residual_time == 0.0 and add_time > 0.0:
            # Legacy mode: split the old add_time equally between the two new fields
            add_attn_residual_time = add_time * 0.5
            add_ffn_residual_time = add_time * 0.5

        # Build time components from flat parameters
        self._attention_time = AttentionTime(
            attention_prefill_execution_time=attention_prefill_execution_time,
            attention_decode_execution_time=attention_decode_execution_time,
            attention_layer_pre_proj_execution_time=attention_layer_pre_proj_execution_time,
            attention_layer_post_proj_execution_time=attention_layer_post_proj_execution_time,
            attention_rope_execution_time=attention_rope_execution_time,
            attention_kv_cache_save_execution_time=attention_kv_cache_save_execution_time,
            attn_mla_kv_cache_save_time=attn_mla_kv_cache_save_time,
            attn_mla_prefill_kv_up_proj_time=attn_mla_prefill_kv_up_proj_time,
            attn_mla_prefill_time=attn_mla_prefill_time,
            attn_mla_decode_q_latent_proj_time=attn_mla_decode_q_latent_proj_time,
            attn_mla_decode_time=attn_mla_decode_time,
            attn_mla_v_up_proj_time=attn_mla_v_up_proj_time,
            attn_norm_time=attn_norm_time,
            operator_times=attention_operator_times,
        )

        if is_moe:
            self._moe_or_mlp_time = MoETime(
                moe_grouped_gemm_time=moe_grouped_gemm_time,
                moe_gating_linear_time=moe_gating_linear_time,
                moe_gating_routing_topk_time=moe_gating_routing_topk_time,
                moe_shuffling_time=moe_shuffling_time,
                mlp_norm_time=mlp_norm_time,
                share_expert_up_proj_time=share_expert_up_proj_time,
                share_expert_down_proj_time=share_expert_down_proj_time,
                share_expert_act_time=share_expert_act_time,
                operator_times=moe_operator_times,
            )
        else:
            self._moe_or_mlp_time = MLPTime(
                mlp_layer_up_proj_execution_time=mlp_layer_up_proj_execution_time,
                mlp_layer_down_proj_execution_time=mlp_layer_down_proj_execution_time,
                mlp_layer_act_execution_time=mlp_layer_act_execution_time,
                mlp_norm_time=mlp_norm_time,
                operator_times=mlp_operator_times,
            )

        use_split_tp_allreduce = (
            attn_tensor_parallel_allreduce_is_provided
            or moe_tensor_parallel_allreduce_is_provided
        )
        legacy_tensor_parallel_allreduce_time = (
            0.0 if use_split_tp_allreduce else tensor_parallel_communication_time
        )

        self._communication_time = CommunicationTime(
            attn_tensor_parallel_allreduce_time=attn_tensor_parallel_allreduce_time,
            moe_tensor_parallel_allreduce_time=moe_tensor_parallel_allreduce_time,
            tensor_parallel_allreduce_time=legacy_tensor_parallel_allreduce_time,
            tensor_parallel_allgather_time=tensor_parallel_allgather_time,
            share_expert_tensor_parallel_allreduce_time=share_expert_tensor_parallel_allreduce_time,
            dp_input_allreduce_time=dp_input_allreduce_time,
            dp_output_allreduce_time=dp_output_allreduce_time,
            pipeline_parallel_send_recv_time=pipeline_parallel_communication_time,
            # EP communication is tracked in ExecutionTime, but we keep this for future extensions
            expert_parallel_allgather_time=0.0,
            expert_parallel_alltoall_time=expert_parallel_alltoall_time,
            operator_times=communication_operator_times,
        )

        self._overhead_time = OverheadTime(
            schedule_time=schedule_time,
            sampler_e2e_time=sampler_e2e_time,
            prepare_inputs_e2e_time=prepare_inputs_e2e_time,
            process_model_outputs_time=process_model_outputs_time,
            ray_comm_time=ray_comm_time,
            pp_producer_send_path_runtime_time=pp_producer_send_path_runtime_time,
            pp_receiver_head_runtime_time=pp_receiver_head_runtime_time,
            pp_prefill_consumer_active_runtime_time=pp_prefill_consumer_active_runtime_time,
            pp_stage_boundary_residual_runtime_time=(
                pp_stage_boundary_residual_runtime_time
            ),
            pp_stage_boundary_handoff_time=pp_stage_boundary_handoff_time,
        )

        self._residual_time = ResidualTime(
            add_attn_residual_time=add_attn_residual_time,
            add_ffn_residual_time=add_ffn_residual_time,
        )

        # Compatibility note: flat fields are retained for deprecated read paths.
        self._attention_rope_execution_time = attention_rope_execution_time
        self._attention_kv_cache_save_execution_time = attention_kv_cache_save_execution_time
        self._attention_decode_execution_time = attention_decode_execution_time
        self._attention_prefill_execution_time = attention_prefill_execution_time
        self._attn_mla_kv_cache_save_time = attn_mla_kv_cache_save_time
        self._attn_mla_prefill_kv_up_proj_time = attn_mla_prefill_kv_up_proj_time
        self._attn_mla_prefill_time = attn_mla_prefill_time
        self._attn_mla_decode_q_latent_proj_time = attn_mla_decode_q_latent_proj_time
        self._attn_mla_decode_time = attn_mla_decode_time
        self._attn_mla_v_up_proj_time = attn_mla_v_up_proj_time
        self._attention_layer_pre_proj_execution_time = attention_layer_pre_proj_execution_time
        self._attention_layer_post_proj_execution_time = attention_layer_post_proj_execution_time
        self._mlp_layer_up_proj_execution_time = mlp_layer_up_proj_execution_time
        self._mlp_layer_down_proj_execution_time = mlp_layer_down_proj_execution_time
        self._mlp_layer_act_execution_time = mlp_layer_act_execution_time
        self._attn_norm_time = attn_norm_time
        self._mlp_norm_time = mlp_norm_time
        self._add_time = add_time  # Deprecated: kept for backward compatibility
        self._add_attn_residual_time = add_attn_residual_time
        self._add_ffn_residual_time = add_ffn_residual_time
        self._tensor_parallel_communication_time = legacy_tensor_parallel_allreduce_time
        self._attn_tensor_parallel_allreduce_time = attn_tensor_parallel_allreduce_time
        self._moe_tensor_parallel_allreduce_time = moe_tensor_parallel_allreduce_time
        self._has_attn_tensor_parallel_allreduce_time = (
            attn_tensor_parallel_allreduce_is_provided
        )
        self._has_moe_tensor_parallel_allreduce_time = (
            moe_tensor_parallel_allreduce_is_provided
        )
        self._tensor_parallel_allgather_time = tensor_parallel_allgather_time
        self._share_expert_tensor_parallel_allreduce_time = (
            share_expert_tensor_parallel_allreduce_time
        )
        self._dp_input_allreduce_time = dp_input_allreduce_time
        self._dp_output_allreduce_time = dp_output_allreduce_time
        self._pipeline_parallel_communication_time = pipeline_parallel_communication_time
        self._schedule_time = schedule_time
        self._sampler_e2e_time = sampler_e2e_time
        self._prepare_inputs_e2e_time = prepare_inputs_e2e_time
        self._process_model_outputs_time = process_model_outputs_time
        self._ray_comm_time = ray_comm_time
        self._pp_producer_send_path_runtime_time = pp_producer_send_path_runtime_time
        self._pp_receiver_head_runtime_time = pp_receiver_head_runtime_time
        self._pp_prefill_consumer_active_runtime_time = (
            pp_prefill_consumer_active_runtime_time
        )
        self._pp_stage_boundary_residual_runtime_time = (
            pp_stage_boundary_residual_runtime_time
        )
        self._pp_stage_boundary_handoff_time = pp_stage_boundary_handoff_time
        self._expert_parallel_communication_time = expert_parallel_communication_time
        self._moe_gating_time = moe_gating_time  # Deprecated: kept for backward compatibility
        self._moe_gating_linear_time = moe_gating_linear_time
        self._moe_gating_routing_topk_time = moe_gating_routing_topk_time
        self._moe_shuffling_time = moe_shuffling_time
        self._moe_grouped_gemm_time = moe_grouped_gemm_time
        self._share_expert_up_proj_time = share_expert_up_proj_time
        self._share_expert_down_proj_time = share_expert_down_proj_time
        self._share_expert_act_time = share_expert_act_time
        self._decode_draft_proposer_time = decode_draft_proposer_time
        self._mtp_terminal_overshoot_time = mtp_terminal_overshoot_time
        self._op_times = normalize_execution_op_times(merged_op_times)
        self._refresh_op_time_attr_values()

    @staticmethod
    def _merge_operator_time_sources(
        base_op_times: Mapping[str, float],
        *operator_time_sources,
    ) -> dict[str, float]:
        merged = normalize_execution_op_times(base_op_times)
        for operator_times in operator_time_sources:
            if operator_times is None:
                continue
            for op_name, time_ms in normalize_execution_op_times(
                operator_times.op_times
            ).items():
                if op_name in merged and merged[op_name] != time_ms:
                    raise ValueError(
                        f"Conflicting operator timing for {op_name}: "
                        f"{merged[op_name]} != {time_ms}"
                    )
                merged[op_name] = time_ms
        return normalize_execution_op_times(merged)

    def _refresh_op_time_attr_values(self) -> None:
        self._op_time_attr_values = execution_op_time_values_by_attr(self._op_times)

    def _merged_replacement_operator_time_source(
        self,
        old_operator_times,
        new_operator_times,
    ) -> dict[str, float]:
        updated_op_times = dict(self._op_times)
        if old_operator_times is not None:
            for op_name in old_operator_times.op_times:
                updated_op_times.pop(op_name, None)
        if new_operator_times is not None:
            for op_name, time_ms in normalize_execution_op_times(
                new_operator_times.op_times
            ).items():
                if op_name in updated_op_times:
                    raise ValueError(
                        f"Conflicting operator timing for {op_name}: "
                        f"{updated_op_times[op_name]} != {time_ms}"
                    )
                updated_op_times[op_name] = time_ms
        return normalize_execution_op_times(updated_op_times)

    def _replace_operator_time_source(self, old_operator_times, new_operator_times) -> None:
        updated_op_times = self._merged_replacement_operator_time_source(
            old_operator_times,
            new_operator_times,
        )
        self._op_times = normalize_execution_op_times(updated_op_times)
        self._refresh_op_time_attr_values()

    def _replace_operator_time_values(self, op_times: Mapping[str, float]) -> None:
        updated_op_times = dict(self._op_times)
        updated_op_times.update(normalize_execution_op_times(op_times))
        self._op_times = normalize_execution_op_times(updated_op_times)
        self._refresh_op_time_attr_values()
        self._attention_time.operator_times = build_attention_operator_times_from_op_times(
            self._op_times
        )
        if isinstance(self._moe_or_mlp_time, MoETime):
            self._moe_or_mlp_time.operator_times = build_moe_operator_times_from_op_times(
                self._op_times
            )
        else:
            self._moe_or_mlp_time.operator_times = build_mlp_operator_times_from_op_times(
                self._op_times
            )
        self._communication_time.operator_times = (
            build_communication_operator_times_from_op_times(self._op_times)
        )

    def _time_attr_value(self, attr_name: str, legacy_value: float) -> float:
        return self._op_time_attr_values.get(attr_name, legacy_value)

    def _scaled_time_attr_value(self, attr_name: str, legacy_value: float) -> float:
        return (
            self._time_attr_value(attr_name, legacy_value)
            * self._num_layers_per_pipeline_stage
        )

    def _get_expert_parallel_communication_time(self) -> float:
        return self._time_attr_value(
            "expert_parallel_alltoall_time",
            self._expert_parallel_communication_time,
        )

    def _get_moe_grouped_gemm_time(self) -> float:
        if not isinstance(self._moe_or_mlp_time, MoETime):
            return 0.0
        return self._time_attr_value(
            "moe_grouped_gemm_time",
            self._moe_or_mlp_time.moe_grouped_gemm_time,
        )

    def _get_moe_gating_time(self) -> float:
        if not isinstance(self._moe_or_mlp_time, MoETime):
            return 0.0
        return self._time_attr_value(
            "moe_gating_linear_time",
            self._moe_or_mlp_time.moe_gating_linear_time,
        ) + self._time_attr_value(
            "moe_gating_routing_topk_time",
            self._moe_or_mlp_time.moe_gating_routing_topk_time,
        )

    def _get_moe_shuffling_time(self) -> float:
        if not isinstance(self._moe_or_mlp_time, MoETime):
            return 0.0
        return self._time_attr_value(
            "moe_shuffling_time",
            self._moe_or_mlp_time.moe_shuffling_time,
        )

    def _get_tensor_parallel_allgather_time(self) -> float:
        return self._time_attr_value(
            "tensor_parallel_allgather_time",
            self._communication_time.tensor_parallel_allgather_time,
        )

    def _get_share_expert_tensor_parallel_allreduce_time(self) -> float:
        return self._time_attr_value(
            "share_expert_tensor_parallel_allreduce_time",
            self._communication_time.share_expert_tensor_parallel_allreduce_time,
        )

    def _get_dp_input_allreduce_time(self) -> float:
        return self._time_attr_value(
            "dp_input_allreduce_time",
            self._communication_time.dp_input_allreduce_time,
        )

    def _get_dp_output_allreduce_time(self) -> float:
        return self._time_attr_value(
            "dp_output_allreduce_time",
            self._communication_time.dp_output_allreduce_time,
        )

    # Component accessors (new API)
    @property
    def op_times(self) -> Mapping[str, float]:
        """Read-only canonical per-operator timing map in milliseconds."""
        return MappingProxyType(self._op_times)

    @property
    def attention_time_component(self) -> AttentionTime:
        """Get the attention time component."""
        return deepcopy(self._attention_time)

    @property
    def attention_operator_times(self) -> AttentionOperatorTimes | None:
        """Get structured single-layer attention operator timings."""
        return deepcopy(self._attention_time.operator_times)

    @attention_operator_times.setter
    def attention_operator_times(
        self,
        operator_times: AttentionOperatorTimes | None,
    ) -> None:
        old_operator_times = self._attention_time.operator_times
        updated_op_times = self._merged_replacement_operator_time_source(
            old_operator_times,
            operator_times,
        )
        self._attention_time.operator_times = deepcopy(operator_times)
        self._op_times = updated_op_times
        self._refresh_op_time_attr_values()

    @property
    def moe_or_mlp_time_component(self) -> Union[MLPTime, MoETime]:
        """Get the MoE or MLP time component."""
        return deepcopy(self._moe_or_mlp_time)

    @property
    def mlp_operator_times(self) -> MLPOperatorTimes | None:
        """Get structured single-layer dense MLP operator timings."""
        if isinstance(self._moe_or_mlp_time, MLPTime):
            return deepcopy(self._moe_or_mlp_time.operator_times)
        return None

    @mlp_operator_times.setter
    def mlp_operator_times(self, operator_times: MLPOperatorTimes | None) -> None:
        if not isinstance(self._moe_or_mlp_time, MLPTime):
            raise ValueError("mlp_operator_times are only valid for dense MLP components")
        old_operator_times = self._moe_or_mlp_time.operator_times
        updated_op_times = self._merged_replacement_operator_time_source(
            old_operator_times,
            operator_times,
        )
        self._moe_or_mlp_time.operator_times = deepcopy(operator_times)
        self._op_times = updated_op_times
        self._refresh_op_time_attr_values()

    @property
    def moe_operator_times(self) -> MoEOperatorTimes | None:
        """Get structured single-layer MoE operator timings."""
        if isinstance(self._moe_or_mlp_time, MoETime):
            return deepcopy(self._moe_or_mlp_time.operator_times)
        return None

    @moe_operator_times.setter
    def moe_operator_times(self, operator_times: MoEOperatorTimes | None) -> None:
        if not isinstance(self._moe_or_mlp_time, MoETime):
            raise ValueError("moe_operator_times are only valid for MoE components")
        old_operator_times = self._moe_or_mlp_time.operator_times
        updated_op_times = self._merged_replacement_operator_time_source(
            old_operator_times,
            operator_times,
        )
        self._moe_or_mlp_time.operator_times = deepcopy(operator_times)
        self._op_times = updated_op_times
        self._refresh_op_time_attr_values()

    @property
    def communication_time_component(self) -> CommunicationTime:
        """Get the communication time component."""
        return deepcopy(self._communication_time)

    @property
    def communication_operator_times(self) -> CommunicationOperatorTimes | None:
        """Get structured single-layer communication operator timings."""
        return deepcopy(self._communication_time.operator_times)

    @communication_operator_times.setter
    def communication_operator_times(
        self,
        operator_times: CommunicationOperatorTimes | None,
    ) -> None:
        old_operator_times = self._communication_time.operator_times
        updated_op_times = self._merged_replacement_operator_time_source(
            old_operator_times,
            operator_times,
        )
        self._communication_time.operator_times = deepcopy(operator_times)
        self._op_times = updated_op_times
        self._refresh_op_time_attr_values()

    @property
    def overhead_time_component(self) -> OverheadTime:
        """Get the overhead time component."""
        return deepcopy(self._overhead_time)

    @property
    def residual_time_component(self) -> ResidualTime:
        """Get the residual time component."""
        return deepcopy(self._residual_time)

    def override_moe_grouped_gemm_time(self, time: float) -> None:
        """Override MoE grouped GEMM time (updates both component and flat field)."""
        if isinstance(self._moe_or_mlp_time, MoETime):
            self._replace_operator_time_values({"moe_grouped_gemm": time})
            self._moe_or_mlp_time.moe_grouped_gemm_time = time
        self._moe_grouped_gemm_time = time

    def override_moe_times(self, grouped_gemm_time: float, expert_parallel_comm_time: float,
                          gating_time: float, shuffling_time: float,
                          gating_linear_time: float = 0.0, gating_routing_topk_time: float = 0.0) -> None:
        """Override all MoE-specific execution times with accumulated values from all layers.

        Args:
            grouped_gemm_time: Total grouped GEMM time across all layers
            expert_parallel_comm_time: Total EP communication time across all layers
            gating_time: Total gating time (deprecated, use gating_linear_time + gating_routing_topk_time)
            shuffling_time: Total shuffling time across all layers
            gating_linear_time: Total gating linear time across all layers (new)
            gating_routing_topk_time: Total gating routing topk time across all layers (new)
        """
        # Handle backward compatibility: if new fields are not provided, split gating_time equally
        if gating_linear_time == 0.0 and gating_routing_topk_time == 0.0 and gating_time > 0.0:
            effective_gating_linear_time = gating_time * 0.5
            effective_gating_routing_topk_time = gating_time * 0.5
        else:
            effective_gating_linear_time = gating_linear_time
            effective_gating_routing_topk_time = gating_routing_topk_time

        if isinstance(self._moe_or_mlp_time, MoETime):
            self._replace_operator_time_values(
                {
                    "moe_grouped_gemm": grouped_gemm_time,
                    "moe_gating_linear": effective_gating_linear_time,
                    "moe_gating_routing_topk": effective_gating_routing_topk_time,
                    "moe_shuffling": shuffling_time,
                    "expert_parallel_allreduce": expert_parallel_comm_time,
                }
            )
            self._moe_or_mlp_time.moe_grouped_gemm_time = grouped_gemm_time
            self._moe_or_mlp_time.moe_gating_linear_time = effective_gating_linear_time
            self._moe_or_mlp_time.moe_gating_routing_topk_time = (
                effective_gating_routing_topk_time
            )
            self._moe_or_mlp_time.moe_shuffling_time = shuffling_time

        self._moe_grouped_gemm_time = grouped_gemm_time
        self._expert_parallel_communication_time = expert_parallel_comm_time
        self._moe_gating_time = gating_time  # Deprecated
        self._moe_gating_linear_time = gating_linear_time
        self._moe_gating_routing_topk_time = gating_routing_topk_time
        self._moe_shuffling_time = shuffling_time

    # ========================================================================
    # Refactored Properties: Delegate to Time Components (Single-Layer Granularity)
    # ========================================================================

    @property
    def attention_time(self) -> float:
        """
        Get total attention execution time for all layers in this stage.

        For single-layer granularity (num_layers=1): returns single-layer time
        For multi-layer aggregation (num_layers>1): returns aggregated time
        """
        return self._attention_time.total_time() * self._num_layers_per_pipeline_stage

    @property
    def moe_comm_time(self) -> float:
        """
        Get MoE communication time (dispatch + return) for all layers in this stage.

        Includes expert_parallel_communication_time and moe_shuffling_time.
        """
        if not isinstance(self._moe_or_mlp_time, MoETime):
            return 0.0
        return (
            self._get_expert_parallel_communication_time()
            + self._get_moe_shuffling_time()
        ) * self._num_layers_per_pipeline_stage

    @property
    def moe_comp_time(self) -> float:
        """
        Get MoE computation time (grouped GEMM + gating) for all layers in this stage.
        """
        if not isinstance(self._moe_or_mlp_time, MoETime):
            return 0.0
        return (
            self._get_moe_grouped_gemm_time()
            + self._get_moe_gating_time()
        ) * self._num_layers_per_pipeline_stage

    @property
    def pipeline_time(self) -> float:
        """Get pipeline parallel communication time (not scaled by layers)."""
        return self.pipeline_parallel_communication_time

    def get_single_layer_attention_time(self) -> float:
        """
        Get attention execution time for a single layer.

        IMPORTANT: Components are already at single-layer granularity.
        No division by num_layers needed.
        """
        return self._attention_time.total_time()

    def get_single_layer_moe_comp_time(self) -> float:
        """
        Get MoE computation time for a single layer.

        IMPORTANT: Components are already at single-layer granularity.
        No division by num_layers needed.
        """
        if not isinstance(self._moe_or_mlp_time, MoETime):
            raise ValueError("MoE computation time is only available for MoE models")
        return (
            self._get_moe_grouped_gemm_time()
            + self._get_moe_gating_time()
        )

    def get_single_layer_moe_comm_time(self) -> float:
        """
        Get MoE communication time for a single layer.

        Returns the full bidirectional communication time (dispatch + return) for one layer.

        IMPORTANT: Components are already at single-layer granularity.
        No division by num_layers needed.
        """
        if not isinstance(self._moe_or_mlp_time, MoETime):
            raise ValueError("MoE communication time is only available for MoE models")
        return (
            self._get_expert_parallel_communication_time()
            + self._get_moe_shuffling_time()
        )

    def get_single_layer_add_time(self) -> float:
        """
        Get residual connection time for a single layer.

        IMPORTANT: Components are already at single-layer granularity.
        No division by num_layers needed.
        """
        return self._residual_time.add_time

    def get_single_layer_dp_input_allreduce_time(self) -> float:
        """Get DP input allreduce time for a single layer in milliseconds."""
        return self._get_dp_input_allreduce_time()

    def get_single_layer_dp_output_allreduce_time(self) -> float:
        """Get DP output allreduce time for a single layer in milliseconds."""
        return self._get_dp_output_allreduce_time()

    def get_single_layer_block_time(self) -> float:
        """
        Get complete transformer block time for a single layer in milliseconds.

        This follows the canonical ExecutionTime composition:
        attention + (MLP/MoE + comm) + residual add.
        """
        return self._get_block_execution_time()

    def get_single_layer_post_attention_time(self) -> float:
        """
        Get post-attention portion of one layer in milliseconds.

        Equivalent to: single-layer block - single-layer attention.
        """
        post_attention_time = (
            self.get_single_layer_block_time() - self.get_single_layer_attention_time()
        )
        if post_attention_time < 0:
            raise ValueError(
                f"Invalid post-attention time: {post_attention_time} ms "
                f"(block={self.get_single_layer_block_time()} ms, "
                f"attention={self.get_single_layer_attention_time()} ms)"
            )
        return post_attention_time

    # ========================================================================
    # Refactored Internal Helper Methods: Use Time Components
    # ========================================================================

    def _get_mlp_layer_execution_time(self) -> float:
        """
        Get MLP layer execution time (single layer).

        Includes MLP computation + TP allreduce + MLP norm.
        """
        if not isinstance(self._moe_or_mlp_time, MLPTime):
            return 0.0
        return (
            self._moe_or_mlp_time.total_time()
            + self._get_moe_tp_allreduce_time()
        )

    def _get_moe_execution_time(self) -> float:
        """
        Get MoE execution time (single layer).

        Includes grouped GEMM, gating, shuffling, and EP communication.
        """
        if not isinstance(self._moe_or_mlp_time, MoETime):
            return 0.0
        return (
            self._moe_or_mlp_time.total_time()
            + self._get_expert_parallel_communication_time()
            + self._get_moe_tp_allreduce_time()
            + self._get_tensor_parallel_allgather_time()
            + self._get_share_expert_tensor_parallel_allreduce_time()
            + self._get_dp_input_allreduce_time()
            + self._get_dp_output_allreduce_time()
        )

    def _get_attn_tp_allreduce_time(self) -> float:
        attr_name = "attn_tensor_parallel_allreduce_time"
        if attr_name in self._op_time_attr_values:
            return self._op_time_attr_values[attr_name]
        if self._has_attn_tensor_parallel_allreduce_time:
            return self._communication_time.attn_tensor_parallel_allreduce_time
        return self._communication_time.tensor_parallel_allreduce_time

    def _get_moe_tp_allreduce_time(self) -> float:
        attr_name = "moe_tensor_parallel_allreduce_time"
        if attr_name in self._op_time_attr_values:
            return self._op_time_attr_values[attr_name]
        if self._has_moe_tensor_parallel_allreduce_time:
            return self._communication_time.moe_tensor_parallel_allreduce_time
        return self._communication_time.tensor_parallel_allreduce_time

    def _get_attention_layer_execution_time(self) -> float:
        """
        Get attention layer execution time (single layer).

        Includes all attention operations + TP allreduce.
        """
        return (
            self._attention_time.total_time()
            + self._get_attn_tp_allreduce_time()
        )

    def _get_block_execution_time(self) -> float:
        """
        Get complete transformer block execution time (single layer).

        Includes attention + MLP/MoE + residual connection.
        """
        return (
            self._get_attention_layer_execution_time()
            + (self._get_moe_execution_time() if self._is_moe else self._get_mlp_layer_execution_time())
            + self._residual_time.add_time
        )

    def _get_cpu_overhead(self) -> float:
        """Get total CPU overhead time (not scaled by layers)."""
        return self._overhead_time.simulated_total_time()

    def _get_diagnostic_cpu_overhead(self) -> float:
        """Get diagnostic CPU overhead time including inactive components."""
        return self._overhead_time.diagnostic_total_time()

    # ========================================================================
    # Refactored Property Accessors: Delegate to Time Components
    # ========================================================================

    @property
    def num_layers(self) -> int:
        """Number of layers in this pipeline stage."""
        return self._num_layers_per_pipeline_stage

    # MLP Component Properties
    @property
    def mlp_layer_up_proj_execution_time(self) -> float:
        """MLP up projection time (aggregated across all layers)."""
        if isinstance(self._moe_or_mlp_time, MLPTime):
            return self._scaled_time_attr_value(
                "mlp_layer_up_proj_execution_time",
                self._moe_or_mlp_time.mlp_layer_up_proj_execution_time,
            )
        return 0.0

    @property
    def mlp_layer_down_proj_execution_time(self) -> float:
        """MLP down projection time (aggregated across all layers)."""
        if isinstance(self._moe_or_mlp_time, MLPTime):
            return self._scaled_time_attr_value(
                "mlp_layer_down_proj_execution_time",
                self._moe_or_mlp_time.mlp_layer_down_proj_execution_time,
            )
        return 0.0

    @property
    def mlp_layer_act_execution_time(self) -> float:
        """MLP activation time (aggregated across all layers)."""
        if isinstance(self._moe_or_mlp_time, MLPTime):
            return self._scaled_time_attr_value(
                "mlp_layer_act_execution_time",
                self._moe_or_mlp_time.mlp_layer_act_execution_time,
            )
        return 0.0

    @property
    def mlp_all_reduce_time(self) -> float:
        """TP allreduce time for MLP (aggregated across all layers)."""
        return self._get_moe_tp_allreduce_time() * self._num_layers_per_pipeline_stage

    @property
    def mlp_norm_time(self) -> float:
        """MLP layer norm time (aggregated across all layers). Supports both MLP and MoE models."""
        if isinstance(self._moe_or_mlp_time, MLPTime):
            return self._scaled_time_attr_value(
                "mlp_norm_time",
                self._moe_or_mlp_time.mlp_norm_time,
            )
        elif isinstance(self._moe_or_mlp_time, MoETime):
            return self._scaled_time_attr_value(
                "mlp_norm_time",
                self._moe_or_mlp_time.mlp_norm_time,
            )
        return 0.0

    # Attention Component Properties
    @property
    def attention_pre_proj_time(self) -> float:
        """Attention pre-projection (QKV) time (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "attention_layer_pre_proj_execution_time",
            self._attention_time.attention_layer_pre_proj_execution_time,
        )

    @property
    def attention_post_proj_time(self) -> float:
        """Attention post-projection time (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "attention_layer_post_proj_execution_time",
            self._attention_time.attention_layer_post_proj_execution_time,
        )

    @property
    def attention_all_reduce_time(self) -> float:
        """TP allreduce time for attention (aggregated across all layers)."""
        return self._get_attn_tp_allreduce_time() * self._num_layers_per_pipeline_stage

    @property
    def moe_tensor_parallel_allgather_time(self) -> float:
        """TP allgather time for FFN input (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "tensor_parallel_allgather_time",
            self._communication_time.tensor_parallel_allgather_time,
        )

    @property
    def share_expert_tensor_parallel_allreduce_time(self) -> float:
        """TP allreduce time for shared expert output (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "share_expert_tensor_parallel_allreduce_time",
            self._communication_time.share_expert_tensor_parallel_allreduce_time,
        )

    @property
    def attention_rope_execution_time(self) -> float:
        """RoPE execution time (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "attention_rope_execution_time",
            self._attention_time.attention_rope_execution_time,
        )

    @property
    def attention_kv_cache_save_execution_time(self) -> float:
        """KV cache save time (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "attention_kv_cache_save_execution_time",
            self._attention_time.attention_kv_cache_save_execution_time,
        )

    @property
    def attention_decode_execution_time(self) -> float:
        """Attention decode time (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "attention_decode_execution_time",
            self._attention_time.attention_decode_execution_time,
        )

    @property
    def attention_prefill_execution_time(self) -> float:
        """Attention prefill time (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "attention_prefill_execution_time",
            self._attention_time.attention_prefill_execution_time,
        )

    @property
    def attn_mla_kv_cache_save_time(self) -> float:
        """MLA latent KV cache save time (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "attn_mla_kv_cache_save_time",
            self._attention_time.attn_mla_kv_cache_save_time,
        )

    @property
    def attn_mla_prefill_kv_up_proj_time(self) -> float:
        """MLA prefill KV up-projection time (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "attn_mla_prefill_kv_up_proj_time",
            self._attention_time.attn_mla_prefill_kv_up_proj_time,
        )

    @property
    def attn_mla_prefill_time(self) -> float:
        """MLA prefill attention time (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "attn_mla_prefill_time",
            self._attention_time.attn_mla_prefill_time,
        )

    @property
    def attn_mla_decode_q_latent_proj_time(self) -> float:
        """MLA decode Q latent projection time (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "attn_mla_decode_q_latent_proj_time",
            self._attention_time.attn_mla_decode_q_latent_proj_time,
        )

    @property
    def attn_mla_decode_time(self) -> float:
        """MLA decode attention time (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "attn_mla_decode_time",
            self._attention_time.attn_mla_decode_time,
        )

    @property
    def attn_mla_v_up_proj_time(self) -> float:
        """MLA V up-projection time (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "attn_mla_v_up_proj_time",
            self._attention_time.attn_mla_v_up_proj_time,
        )

    @property
    def attn_norm_time(self) -> float:
        """Attention layer norm time (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "attn_norm_time",
            self._attention_time.attn_norm_time,
        )

    # Communication Component Properties
    @property
    def pipeline_parallel_communication_time(self) -> float:
        """Pipeline parallel communication time (not scaled by layers)."""
        return self._time_attr_value(
            "pipeline_parallel_send_recv_time",
            self._communication_time.pipeline_parallel_send_recv_time,
        )

    @property
    def dp_input_allreduce_time(self) -> float:
        """DP input allreduce time (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "dp_input_allreduce_time",
            self._communication_time.dp_input_allreduce_time,
        )

    @property
    def dp_output_allreduce_time(self) -> float:
        """DP output allreduce time (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "dp_output_allreduce_time",
            self._communication_time.dp_output_allreduce_time,
        )

    # Overhead Component Properties
    @property
    def schedule_time(self) -> float:
        """Scheduling overhead time (not scaled by layers)."""
        return self._overhead_time.schedule_time

    @property
    def sampler_e2e_time(self) -> float:
        """Sampler end-to-end time (not scaled by layers)."""
        return self._overhead_time.sampler_e2e_time

    @property
    def prepare_inputs_e2e_time(self) -> float:
        """Input preparation time (not scaled by layers)."""
        return self._overhead_time.prepare_inputs_e2e_time

    @property
    def process_model_outputs_time(self) -> float:
        """Output processing time (not scaled by layers)."""
        return self._overhead_time.process_model_outputs_time

    @property
    def ray_comm_time(self) -> float:
        """Ray communication overhead time (not scaled by layers)."""
        return self._overhead_time.ray_comm_time

    @property
    def pp_receiver_head_runtime_time(self) -> float:
        """PP receiver-head runtime overhead time (not scaled by layers)."""
        return self._overhead_time.pp_receiver_head_runtime_time

    @property
    def pp_producer_send_path_runtime_time(self) -> float:
        """PP producer send-path runtime overhead time (not scaled by layers)."""
        return self._overhead_time.pp_producer_send_path_runtime_time

    @property
    def pp_prefill_consumer_active_runtime_time(self) -> float:
        """PP prefill consumer-active runtime overhead time (not scaled by layers)."""
        return self._overhead_time.pp_prefill_consumer_active_runtime_time

    @property
    def pp_stage_boundary_residual_runtime_time(self) -> float:
        """Shared-domain PP boundary residual runtime overhead time (not scaled)."""
        return self._overhead_time.pp_stage_boundary_residual_runtime_time

    @property
    def pp_stage_boundary_handoff_time(self) -> float:
        """Stage-aware PP handoff overhead time (not scaled by layers)."""
        return self._overhead_time.pp_stage_boundary_handoff_time

    @property
    def decode_draft_proposer_time(self) -> float:
        """Batch-level draft proposer execution time (not scaled by layers)."""
        return self._decode_draft_proposer_time

    @property
    def mtp_terminal_overshoot_time(self) -> float:
        """Terminal MTP trace-row work paid after logical output completion."""
        return self._mtp_terminal_overshoot_time

    # Residual Component Properties
    @property
    def add_time(self) -> float:
        """Residual connection time (aggregated across all layers). Sum of attn + ffn residuals."""
        single_layer_add_time = self._time_attr_value(
            "add_attn_residual_time",
            self._residual_time.add_attn_residual_time,
        ) + self._time_attr_value(
            "add_ffn_residual_time",
            self._residual_time.add_ffn_residual_time,
        )
        return single_layer_add_time * self._num_layers_per_pipeline_stage

    @property
    def add_attn_residual_time(self) -> float:
        """Attention residual connection time (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "add_attn_residual_time",
            self._residual_time.add_attn_residual_time,
        )

    @property
    def add_ffn_residual_time(self) -> float:
        """FFN/MoE residual connection time (aggregated across all layers)."""
        return self._scaled_time_attr_value(
            "add_ffn_residual_time",
            self._residual_time.add_ffn_residual_time,
        )

    # MoE Component Properties
    @property
    def moe_grouped_gemm_time(self) -> float:
        """MoE grouped GEMM time (aggregated across all layers)."""
        if isinstance(self._moe_or_mlp_time, MoETime):
            return self._scaled_time_attr_value(
                "moe_grouped_gemm_time",
                self._moe_or_mlp_time.moe_grouped_gemm_time,
            )
        return 0.0

    @property
    def expert_parallel_communication_time(self) -> float:
        """EP communication time (aggregated across all layers)."""
        if not isinstance(self._moe_or_mlp_time, MoETime):
            return 0.0
        return self._scaled_time_attr_value(
            "expert_parallel_alltoall_time",
            self._expert_parallel_communication_time,
        )

    @property
    def moe_gating_time(self) -> float:
        """MoE gating time (aggregated across all layers). Sum of linear + routing_topk."""
        if isinstance(self._moe_or_mlp_time, MoETime):
            return self.moe_gating_linear_time + self.moe_gating_routing_topk_time
        return 0.0

    @property
    def moe_gating_linear_time(self) -> float:
        """MoE gating linear time (aggregated across all layers)."""
        if isinstance(self._moe_or_mlp_time, MoETime):
            return self._scaled_time_attr_value(
                "moe_gating_linear_time",
                self._moe_or_mlp_time.moe_gating_linear_time,
            )
        return 0.0

    @property
    def moe_gating_routing_topk_time(self) -> float:
        """MoE gating routing topk time (aggregated across all layers)."""
        if isinstance(self._moe_or_mlp_time, MoETime):
            return self._scaled_time_attr_value(
                "moe_gating_routing_topk_time",
                self._moe_or_mlp_time.moe_gating_routing_topk_time,
            )
        return 0.0

    @property
    def moe_shuffling_time(self) -> float:
        """MoE shuffling time (aggregated across all layers)."""
        if isinstance(self._moe_or_mlp_time, MoETime):
            return self._scaled_time_attr_value(
                "moe_shuffling_time",
                self._moe_or_mlp_time.moe_shuffling_time,
            )
        return 0.0

    @property
    def share_expert_up_proj_time(self) -> float:
        """Shared expert up projection time (aggregated across all layers)."""
        if isinstance(self._moe_or_mlp_time, MoETime):
            return self._scaled_time_attr_value(
                "share_expert_up_proj_time",
                self._moe_or_mlp_time.share_expert_up_proj_time,
            )
        return 0.0

    @property
    def share_expert_down_proj_time(self) -> float:
        """Shared expert down projection time (aggregated across all layers)."""
        if isinstance(self._moe_or_mlp_time, MoETime):
            return self._scaled_time_attr_value(
                "share_expert_down_proj_time",
                self._moe_or_mlp_time.share_expert_down_proj_time,
            )
        return 0.0

    @property
    def share_expert_act_time(self) -> float:
        """Shared expert activation time (aggregated across all layers)."""
        if isinstance(self._moe_or_mlp_time, MoETime):
            return self._scaled_time_attr_value(
                "share_expert_act_time",
                self._moe_or_mlp_time.share_expert_act_time,
            )
        return 0.0

    @property
    def share_expert_time(self) -> float:
        """Total shared expert time (aggregated across all layers)."""
        if isinstance(self._moe_or_mlp_time, MoETime):
            return (
                self.share_expert_up_proj_time
                + self.share_expert_down_proj_time
                + self.share_expert_act_time
            )
        return 0.0

    # ========================================================================
    # Aggregated Time Properties
    # ========================================================================

    @property
    def model_time(self) -> float:
        """
        Get total model execution time in seconds.

        Includes all computation and communication for all layers in this pipeline stage.
        Includes batch-level draft proposer and terminal MTP overshoot execution when present.
        Does not include CPU overhead (scheduling, sampling, etc.).
        """
        # Calculate per-layer block time
        single_layer_block_time = self._get_block_execution_time()

        # Aggregate across all layers
        total_computation_time = single_layer_block_time * self._num_layers_per_pipeline_stage

        # Add pipeline parallel communication (not scaled by layers)
        pipeline_stage_execution_time = (
            total_computation_time
            + self.pipeline_parallel_communication_time
            + self._decode_draft_proposer_time
            + self._mtp_terminal_overshoot_time
        )

        # Return in seconds
        return pipeline_stage_execution_time * 1e-3

    @property
    def model_time_ms(self) -> float:
        """Get total model execution time in milliseconds."""
        return self.model_time * 1e3

    @property
    def total_time(self) -> float:
        """
        Get total end-to-end execution time in seconds.

        Includes model time + active CPU overhead.
        """
        return self.model_time + self._get_cpu_overhead() * 1e-3

    @property
    def diagnostic_total_time(self) -> float:
        """
        Get diagnostic end-to-end execution time in seconds.

        Includes model time + active CPU overhead + diagnostic-only overhead.
        """
        return self.model_time + self._get_diagnostic_cpu_overhead() * 1e-3

    @property
    def diagnostic_total_time_ms(self) -> float:
        """Get diagnostic end-to-end execution time in milliseconds."""
        return self.diagnostic_total_time * 1e3
