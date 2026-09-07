"""Builders for metrics-only execution-time payloads."""

from frontier.entities.execution_time import ExecutionTime


def build_single_layer_metrics_execution_time(original: ExecutionTime) -> ExecutionTime:
    """Copy the scheduler's metrics fields at single-layer granularity."""

    return ExecutionTime(
        num_layers_per_pipeline_stage=1,
        attention_rope_execution_time=original._attention_rope_execution_time,
        attention_kv_cache_save_execution_time=original._attention_kv_cache_save_execution_time,
        attention_decode_execution_time=original._attention_decode_execution_time,
        attention_prefill_execution_time=original._attention_prefill_execution_time,
        attention_layer_pre_proj_execution_time=original._attention_layer_pre_proj_execution_time,
        attention_layer_post_proj_execution_time=original._attention_layer_post_proj_execution_time,
        attn_norm_time=original._attn_norm_time,
        mlp_norm_time=original._mlp_norm_time,
        add_time=original._add_time,
        add_attn_residual_time=original._add_attn_residual_time,
        add_ffn_residual_time=original._add_ffn_residual_time,
        tensor_parallel_communication_time=original._tensor_parallel_communication_time,
        attn_tensor_parallel_allreduce_time=(
            original._attn_tensor_parallel_allreduce_time
            if original._has_attn_tensor_parallel_allreduce_time
            else None
        ),
        moe_tensor_parallel_allreduce_time=(
            original._moe_tensor_parallel_allreduce_time
            if original._has_moe_tensor_parallel_allreduce_time
            else None
        ),
        tensor_parallel_allgather_time=original._tensor_parallel_allgather_time,
        share_expert_tensor_parallel_allreduce_time=original._share_expert_tensor_parallel_allreduce_time,
        dp_input_allreduce_time=original._dp_input_allreduce_time,
        dp_output_allreduce_time=original._dp_output_allreduce_time,
        pipeline_parallel_communication_time=original._pipeline_parallel_communication_time,
        expert_parallel_communication_time=original._expert_parallel_communication_time,
        moe_gating_time=original._moe_gating_time,
        moe_gating_linear_time=original._moe_gating_linear_time,
        moe_gating_routing_topk_time=original._moe_gating_routing_topk_time,
        moe_shuffling_time=original._moe_shuffling_time,
        schedule_time=original._schedule_time,
        sampler_e2e_time=original._sampler_e2e_time,
        prepare_inputs_e2e_time=original._prepare_inputs_e2e_time,
        pp_producer_send_path_runtime_time=original._pp_producer_send_path_runtime_time,
        pp_receiver_head_runtime_time=original._pp_receiver_head_runtime_time,
        pp_prefill_consumer_active_runtime_time=original._pp_prefill_consumer_active_runtime_time,
        process_model_outputs_time=original._process_model_outputs_time,
        ray_comm_time=original._ray_comm_time,
        is_moe=original._is_moe,
        mlp_layer_up_proj_execution_time=original._mlp_layer_up_proj_execution_time,
        mlp_layer_down_proj_execution_time=original._mlp_layer_down_proj_execution_time,
        mlp_layer_act_execution_time=original._mlp_layer_act_execution_time,
        moe_grouped_gemm_time=original._moe_grouped_gemm_time,
        share_expert_up_proj_time=original._share_expert_up_proj_time,
        share_expert_down_proj_time=original._share_expert_down_proj_time,
        share_expert_act_time=original._share_expert_act_time,
        decode_draft_proposer_time=original._decode_draft_proposer_time,
        mtp_terminal_overshoot_time=original._mtp_terminal_overshoot_time,
    )
