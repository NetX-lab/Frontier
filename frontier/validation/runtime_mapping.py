"""Explicit SGLang-to-Frontier boundary audit; never split fused measurements.

This is a semantic map against the native registries, not a replacement cost
model. Unsupported fusions and measurement families stay visible and fail CSV
admission instead of silently changing simulator numerics.
"""

from dataclasses import dataclass
import argparse
import json
from pathlib import Path

from frontier.operators.binding import bind_operator_query
from frontier.operators.sglang_family import runtime_operator_name
from frontier.validation.operator_trace import CONTRACT, model_layer_kinds


@dataclass(frozen=True)
class Boundary:
    targets: tuple[str, ...]
    relationship: str = "matching_boundary"
    note: str = "Matching work boundary still requires compatible backend, dtype, shape and timing profiles."


def _rules(phase, layer_id):
    core = f"gdn_core_{phase}"
    common = {
        "input_layernorm": Boundary(("input_layernorm",) if layer_id == 0
            else ("input_layernorm", "add_ffn_residual"),
            "matching_boundary" if layer_id == 0 else "fused_cross_layer",
            "Layer 0 has no residual input; later input norms fuse the preceding layer's FFN residual add."),
        "post_attention_layernorm": Boundary(("post_attention_layernorm", "add_attn_residual"),
            "fused", "Current-layer attention residual addition is fused with normalization."),
        "gdn_input_projections": Boundary(("gdn_input_projections",)),
        core: Boundary((core,)),
        "gdn_output_projection": Boundary(("gdn_output_projection",)),
        "attn_pre_proj_qknorm": Boundary(("attn_pre_proj",)),
        "attn_rope": Boundary(("attn_rope",)),
        "attn_kv_cache_write": Boundary(("attn_kv_cache_save",)),
        f"attn_{phase}": Boundary((f"attn_{phase}",)),
        "attn_post_proj_gate": Boundary(("attn_post_proj",), "native_gap",
            "Runtime fuses output gating into this measured group; the generic native post projection omits gating."),
        "attention_tp_allreduce": Boundary(("attn_tensor_parallel_allreduce",), "collective_in_situ",
            "One TP collective; observed duration includes rank-arrival skew, not just transfer cost."),
        "shared_expert_gate_up": Boundary(("share_expert_up_proj",)),
        "shared_expert_activation": Boundary(("share_expert_act",)),
        "shared_expert_down": Boundary(("share_expert_down_proj",)),
        "moe_router_linear": Boundary(("moe_gating_linear",)),
        "moe_routing_topk": Boundary(("moe_gating_routing_topk",)),
        "moe_sorting": Boundary(("moe_shuffling",), "backend_boundary_check",
            "AITER sorting must be checked against the native producer's shuffling/dispatch boundary."),
        "moe_experts_quant_gemm_combine": Boundary(("moe_grouped_gemm",), "backend_boundary_check",
            "Two MXFP4 quantization/GEMM stages plus combine are inseparable in this runtime group."),
        "shared_gate_sigmoid_mul_add": Boundary((), "native_gap",
            "Fused shared gate dot product, sigmoid, multiply and add have no canonical native operator."),
        "mlp_tp_allreduce": Boundary(("moe_tensor_parallel_allreduce", "share_expert_tensor_parallel_allreduce"),
            "fused_collective", "SGLang reduces the combined routed+shared output ONCE on TP=8. Do not "
            "assign that duration twice to native routed and shared reductions."),
    }
    common.update({
        "gdn": Boundary(("gdn_input_projections", core, "gdn_output_projection"), "composite",
            "Whole GDN event span; cannot divide into three native operator timings."),
        "full_attention": Boundary(("attn_pre_proj", "attn_rope", "attn_kv_cache_save",
            f"attn_{phase}", "attn_post_proj"), "native_gap",
            "Whole mixer span includes the output gate, which is missing from the generic native projection path."),
        "attention_reduce_and_mlp_norm": Boundary(("attn_tensor_parallel_allreduce",
            "post_attention_layernorm", "add_attn_residual"), "composite",
            "Collective plus fused residual/norm; no decomposition into invented communication and compute costs."),
        "mlp_including_tp_reduce": Boundary(("share_expert_up_proj", "share_expert_act",
            "share_expert_down_proj", "moe_gating_linear", "moe_gating_routing_topk", "moe_shuffling",
            "moe_grouped_gemm", "moe_tensor_parallel_allreduce", "share_expert_tensor_parallel_allreduce"),
            "native_gap", "Includes shared gating/addition and one combined reduction; native compute "
            "and communication boundaries are not equivalent to this total."),
        "moe_shared_expert": Boundary(("share_expert_up_proj", "share_expert_act", "share_expert_down_proj"),
            "composite", "Whole shared MLP, excluding its separate gate and combined TP reduction."),
        "moe_gate": common["moe_router_linear"], "moe_topk": common["moe_routing_topk"],
        "moe_experts": Boundary(("moe_shuffling", "moe_grouped_gemm"), "composite",
            "Dispatcher, sorting and MXFP4 quantization/GEMMs/combine; do not label the whole scope grouped GEMM."),
        "gdn_attn": Boundary((core,), "partial",
            "The inner attention module omits GDN transformations performed by the surrounding mixer."),
        "gdn_norm": Boundary(("gdn_output_projection",), "partial", "Native output scope also includes projection."),
        "gdn_out_proj": Boundary(("gdn_output_projection",), "partial", "Native output scope also includes gated norm."),
        "attention_qkv_proj": Boundary(("attn_pre_proj",), "partial", "Module projection excludes subsequent QK norm."),
        "attention_rotary_emb": common["attn_rope"],
        "attention_attn": Boundary(("attn_kv_cache_save", f"attn_{phase}"), "composite",
            "Radix attention boundary includes cache write and attention kernels."),
        "attention_o_proj": Boundary(("attn_post_proj",)),
    })
    return common


def map_runtime_components(components, *, phase, layer_id, measurement_type, model_config):
    schedule = model_layer_kinds(model_config)
    if phase not in {"prefill", "decode"} or type(layer_id) is not int or not 0 <= layer_id < len(schedule):
        raise ValueError("Runtime map requires a valid model layer and prefill/decode phase")
    if measurement_type not in {"CUDA_EVENT", "KERNEL_ONLY", "HIP_GRAPH_EVENT"}:
        raise ValueError("Unknown measurement family")
    components = list(components)
    if not components or len(set(components)) != len(components):
        raise ValueError("Runtime components must be nonempty and unique")
    rules = _rules(phase, layer_id)
    profile = model_config.get_model_architecture_profile()
    architecture_ops = set(profile.linear_attention.sharded_ops + profile.linear_attention.replicated_ops)
    mappings = []
    for component in components:
        if component not in rules:
            raise ValueError(f"No admitted runtime boundary for {component}")
        if ((component.startswith("gdn") and schedule[layer_id] != "gdn")
                or ((component.startswith("attn_") or component in {"full_attention", "attention_qkv_proj",
                    "attention_rotary_emb", "attention_attn", "attention_o_proj"}) and schedule[layer_id] != "attention")):
            raise ValueError("Runtime mixer boundary disagrees with model layer schedule")
        rule = rules[component]
        targets = []
        for name in rule.targets:
            if name in architecture_ops:
                owner = "dense_attention"
            else:
                binding = bind_operator_query(name)
                if phase not in {p.value for p in binding.operator.phases}:
                    raise ValueError("Native operator is not enabled for this phase")
                owner = binding.family_id
            targets.append({"operator": name, "family": owner,
                "layer_offset": -1 if component == "input_layernorm" and name == "add_ffn_residual" else 0})
        try:
            runtime_target = runtime_operator_name(component) if phase == "decode" else None
        except ValueError:
            runtime_target = None
        mappings.append({"runtime_component": component, "native_targets": targets,
            "relationship": rule.relationship, "note": rule.note,
            "opt_in_runtime_operator": runtime_target,
            "native_csv_export_admitted": False})
    return {"contract": CONTRACT, "model_architecture_profile": profile.profile_id,
        "phase": phase, "layer_id": layer_id, "layer_kind": schedule[layer_id],
        "measurement_type": measurement_type,
        "native_measurement_family_supported": measurement_type != "HIP_GRAPH_EVENT",
        "mappings": mappings,
        "note": "Generic-native semantic mapping only; no calibration or numeric replacement. "
                "opt_in_runtime_operator identifies the separate explicit SGLang cost adapter, not CSV admission. Fused/partial groups "
                "must not be split, duplicated or relabeled. Embedding/final norm/logits remain "
                "outside native decoder model_time; the final FFN residual is fused in the final norm."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--measurement-type", required=True)
    parser.add_argument("--components", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from frontier.profiling.common.model_config import ModelConfig
    result = map_runtime_components(args.components, phase=args.phase, layer_id=args.layer,
        measurement_type=args.measurement_type, model_config=ModelConfig.from_model_name(args.model))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
