"""Shape contracts and actual SGLang BF16 projection/activation callables.

No checkpoint load or routed-expert work. Output reductions, GDN core, top-k
and shared-output gating remain separate runtime scopes. The GDN output scope
includes the native gated norm immediately before its GEMM.
"""

from types import SimpleNamespace


DENSE_PRIMITIVES = ("shared_expert_gate_up", "shared_expert_activation", "shared_expert_down",
                    "moe_router_linear", "gdn_input_projections", "gdn_output_projection")


def validate_dense_spec(spec, tp, name=None):
    required = {"input_width", "linear_kinds", "output_partitions", "output_widths", "local_weight_shapes",
                "tp", "weight_dtype", "includes_reduction", "input_transform"}
    if (not isinstance(spec, dict) or set(spec) != required or spec["tp"] != tp
            or type(spec["input_width"]) is not int or spec["input_width"] < 1
            or spec["weight_dtype"] != "bfloat16" or spec["includes_reduction"] is not False
            or spec["input_transform"] not in {"none", "gdn_rmsnorm_gated"}):
        raise ValueError("Invalid dense primitive shape contract")
    kinds, parts, outputs, weights = (spec[k] for k in
        ("linear_kinds", "output_partitions", "output_widths", "local_weight_shapes"))
    if (not all(isinstance(v, (list, tuple)) for v in (kinds, parts, outputs, weights))
            or not outputs or any(type(v) is not int or v < 1 for v in outputs)
            or len(kinds) != len(parts) or len(kinds) != len(weights)
            or (kinds and len(kinds) != len(outputs))):
        raise ValueError("Malformed dense projection layout")
    if not kinds and (len(outputs) != 1 or spec["input_width"] != 2 * outputs[0]):
        raise ValueError("Silu/multiply requires paired input halves")
    expected_transform = "gdn_rmsnorm_gated" if name == "gdn_output_projection" else "none"
    if name is not None and spec["input_transform"] != expected_transform:
        raise ValueError("Dense input transform disagrees with primitive ownership")
    for kind, partition, output, weight in zip(kinds, parts, outputs, weights):
        if (kind not in {"merged", "row", "replicated"} or not partition
                or any(type(v) is not int or v < 1 for v in partition)
                or tuple(weight) != (output, spec["input_width"])
                or (kind == "merged" and (any(v % tp for v in partition) or sum(partition) // tp != output))
                or (kind != "merged" and tuple(partition) != (output,))):
            raise ValueError("Dense per-rank weights disagree with projection partitions")


def dense_primitive_spec(name, model, tp):
    """CPU-safe per-rank shapes; the router is replicated, not TP-sharded."""
    if name not in DENSE_PRIMITIVES or type(tp) is not int or tp < 1:
        raise ValueError("Unknown dense primitive or invalid TP size")
    quant = model.quantization_config
    if quant is not None and (quant.quantized_operations is None
            or set(quant.quantized_operations) - {"moe_grouped_gemm"}):
        raise ValueError("Dense primitive calibration requires unquantized dense/shared projections")
    hidden, shared, gdn = model.embedding_dim, model.share_expert_dim, model.get_gdn_config()
    if not gdn or not shared or shared % tp:
        raise ValueError("Shared width and GDN heads must support this TP size")
    gdn.get_state_layout(tensor_parallel_size=tp)
    specs = {
        "shared_expert_gate_up": (hidden, ("merged",), ((shared, shared),), (2 * shared // tp,)),
        "shared_expert_activation": (2 * shared // tp, (), (), (shared // tp,)),
        "shared_expert_down": (shared // tp, ("row",), ((hidden,),), (hidden,)),
        "moe_router_linear": (hidden, ("replicated",), ((model.num_experts,),), (model.num_experts,)),
        "gdn_input_projections": (hidden, ("merged", "merged"),
            ((gdn.key_dim, gdn.key_dim, gdn.value_dim, gdn.value_dim),
             (gdn.num_value_heads, gdn.num_value_heads)),
            ((2 * gdn.key_dim + 2 * gdn.value_dim) // tp, 2 * gdn.num_value_heads // tp)),
        "gdn_output_projection": (gdn.value_dim // tp, ("row",), ((hidden,),), (hidden,)),
    }
    width, kinds, partitions, outputs = specs[name]
    return {"input_width": width, "linear_kinds": kinds, "output_partitions": partitions,
            "output_widths": outputs, "local_weight_shapes": tuple((out, width) for out in outputs) if kinds else (),
            "tp": tp, "weight_dtype": "bfloat16", "includes_reduction": False,
            "input_transform": "gdn_rmsnorm_gated" if name == "gdn_output_projection" else "none"}


def make_dense_primitive(name, size, model, group, rank, random):
    """Return a callable, FP32-math reference, and the selected runtime method."""
    import torch
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.linear import MergedColumnParallelLinear, ReplicatedLinear, RowParallelLinear

    spec = dense_primitive_spec(name, model, group.world_size)
    x = random((size, spec["input_width"]))
    if name == "shared_expert_activation":
        activation = SiluAndMul()
        # BaseFusedOp resolves platform dispatch on its first call. Discovery
        # and any compilation happen outside all measured graph replays.
        activation(x)
        method = activation._forward_method.__name__
        if method == "forward_native":
            raise ValueError("Native activation fallback is not the fused runtime primitive")
        width = spec["output_widths"][0]
        reference = (torch.nn.functional.silu(x[:, :width].float()) * x[:, width:].float()).to(x.dtype)
        return lambda: activation(x), reference, f"SiluAndMul.{method}"
    norm = None
    projection_input = x
    if name == "gdn_output_projection":
        from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm
        gdn = model.get_gdn_config()
        local_heads = gdn.num_value_heads // group.world_size
        core = random((size * local_heads, gdn.value_head_dim), .1)
        gate = random(core.shape, .1)
        norm = RMSNorm(gdn.value_head_dim, eps=model.rms_norm_eps, group_size=None,
                       norm_before_gate=True, device="cuda", dtype=x.dtype,
                       activation=gdn.output_gate_type)
        norm.weight.data.copy_(random((gdn.value_head_dim,), .05) + 1.)
        gate_reference = (torch.nn.functional.silu(gate.float()) if gdn.output_gate_type == "silu"
                          else torch.sigmoid(gate.float()))
        projection_input = ((core.float() * torch.rsqrt(
            core.float().square().mean(-1, keepdim=True) + model.rms_norm_eps))
            * norm.weight.float() * gate_reference).to(x.dtype).reshape(size, -1)
        norm(core, gate)  # compile/discover outside graph capture and timing
    linears, references = [], []
    for kind, partitions, shape in zip(spec["linear_kinds"], spec["output_partitions"], spec["local_weight_shapes"]):
        common = dict(bias=False, params_dtype=x.dtype, quant_config=None)
        sharding = dict(tp_rank=rank, tp_size=group.world_size)
        if kind == "merged":
            linear = MergedColumnParallelLinear(spec["input_width"], list(partitions), **common, **sharding)
        elif kind == "row":
            linear = RowParallelLinear(spec["input_width"] * group.world_size, partitions[0],
                input_is_parallel=True, reduce_results=False, **common, **sharding)
        else:
            linear = ReplicatedLinear(spec["input_width"], partitions[0], **common)
        linear = linear.to(device="cuda", dtype=x.dtype)
        if (tuple(linear.weight.shape) != shape or linear.weight.dtype != x.dtype
                or type(linear.quant_method).__name__ != "UnquantizedLinearMethod"):
            raise ValueError("Runtime linear shape/precision/method does not match the dense contract")
        linear.weight.data.copy_(random(shape, spec["input_width"] ** -.5))
        references.append(torch.nn.functional.linear(projection_input.float(), linear.weight.float()).to(x.dtype))
        linears.append(linear)
    backend = "+".join(f"{type(layer).__name__}.{type(layer.quant_method).__name__}" for layer in linears)
    if name == "gdn_input_projections":
        from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet, _gdn_use_alt_stream
        if _gdn_use_alt_stream:
            raise ValueError("This isolated GDN input scope requires the sequential ROCm projection path")
        # Run the native input-projection helper without constructing stateful
        # attention/cache modules. Both projections share the same hidden input.
        owner = SimpleNamespace(in_proj_qkvz=linears[0], in_proj_ba=linears[1], alt_stream=None,
                                _fused_input_proj_cpu_enabled=SimpleNamespace(value=False))
        return lambda: Qwen3_5GatedDeltaNet._forward_input_proj(owner, x), tuple(references), backend
    if name == "gdn_output_projection":
        def output_projection():
            normalized = norm(core, gate).reshape(size, -1)
            return linears[0](normalized)[0]
        return output_projection, references[0], f"RMSNormGated+{backend}"
    return lambda: linears[0](x)[0], references[0], backend
