"""Physical SGLang Qwen hybrid boundaries, selected only by the runtime adapter.

Legacy attributes are aggregation carriers, not claims that fused work is the
corresponding unfused operator. The structured names retain physical identity.
Registering this family does not select it for existing model profiles.
"""

from frontier.operators.spec import (
    OperatorFamilySpec, OperatorPhase, OperatorRole, OperatorSpec,
    ResourceClass, TensorParallelMode, TraceKind,
)


SGLANG_RUNTIME_CONTRACT = "sglang_qwen35_rocm_separate_shared_v1"

# component, legacy carrier, component group
_BOUNDARIES = (
    ("input_layernorm", "attn_norm_time", "attention"),
    ("gdn_input_projections", "attention_layer_pre_proj_execution_time", "attention"),
    ("gdn_core_decode", "attention_decode_execution_time", "attention"),
    ("gdn_output_projection", "attention_layer_post_proj_execution_time", "attention"),
    ("attn_pre_proj_qknorm", "attention_layer_pre_proj_execution_time", "attention"),
    ("attn_rope", "attention_rope_execution_time", "attention"),
    ("attn_kv_cache_write", "attention_kv_cache_save_execution_time", "attention"),
    ("attn_decode", "attention_decode_execution_time", "attention"),
    ("attn_post_proj_gate", "attention_layer_post_proj_execution_time", "attention"),
    ("attention_tp_allreduce", "attn_tensor_parallel_allreduce_time", "communication"),
    ("post_attention_layernorm", "mlp_norm_time", "moe"),
    ("shared_expert_gate_up", "share_expert_up_proj_time", "moe"),
    ("shared_expert_activation", "share_expert_act_time", "moe"),
    ("shared_expert_down", "share_expert_down_proj_time", "moe"),
    ("moe_router_linear", "moe_gating_linear_time", "moe"),
    ("moe_routing_topk", "moe_gating_routing_topk_time", "moe"),
    ("moe_sorting", "moe_shuffling_time", "moe"),
    ("moe_experts_quant_gemm_combine", "moe_grouped_gemm_time", "moe"),
    ("shared_gate_sigmoid_mul_add", "share_expert_act_time", "moe"),
    # ONE reduction after combining routed and shared expert output. The
    # independent shared-expert reduction carrier must remain zero.
    ("mlp_tp_allreduce", "moe_tensor_parallel_allreduce_time", "communication"),
)
_BOUNDARY_NAMES = frozenset(row[0] for row in _BOUNDARIES)


def runtime_operator_name(component: str) -> str:
    if component not in _BOUNDARY_NAMES:
        raise ValueError(f"Unsupported SGLang runtime boundary: {component}")
    return f"sglang_{component}"


def runtime_operator_attrs(group: str | None = None) -> dict[str, str]:
    return {runtime_operator_name(name): attr for name, attr, owner in _BOUNDARIES
            if group is None or owner == group}


SGLANG_RUNTIME_FAMILY = OperatorFamilySpec(
    family_id="sglang_runtime", display_name="SGLang Qwen Hybrid Runtime",
    supported_variants=(SGLANG_RUNTIME_CONTRACT,),
    operators=tuple(OperatorSpec(
        name=runtime_operator_name(name),
        role=OperatorRole.COMMUNICATION if group == "communication" else OperatorRole.DECODE_KERNEL,
        phases=(OperatorPhase.DECODE,),
        execution_time_attr=attr,
        resource_class=ResourceClass.COMM if group == "communication" else ResourceClass.COMP,
        trace_kind=TraceKind.COMM if group == "communication" else TraceKind.COMPUTE,
        # This family has its own strict profile contract, not linear_op.csv
        # aliases or automatic generic profiling/training targets.
        profiling_target=False, predictor_target=False,
        tp_mode=(TensorParallelMode.REPLICATED if name in {
            "input_layernorm", "post_attention_layernorm", "moe_router_linear",
            "moe_routing_topk", "shared_gate_sigmoid_mul_add",
        } else TensorParallelMode.MOE_TP if name in {
            "moe_sorting", "moe_experts_quant_gemm_combine", "mlp_tp_allreduce",
        } else TensorParallelMode.ATTENTION_TP),
    ) for name, attr, group in _BOUNDARIES),
)
