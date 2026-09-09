"""Shape and state contracts for SGLang's packed GDN decode core."""

from __future__ import annotations


GDN_PRIMITIVES = ("gdn_core_decode",)


def gdn_core_spec(model, tp):
    """Return the model-derived, per-rank packed-decode state layout."""
    if type(tp) is not int or tp < 1:
        raise ValueError("Invalid GDN tensor-parallel size")
    gdn = model.get_gdn_config()
    if gdn is None:
        raise ValueError("GDN core profiling requires a GDN model")
    layout = gdn.get_state_layout(tensor_parallel_size=tp)
    return {
        "q_heads": gdn.num_key_heads // tp,
        "value_heads": gdn.num_value_heads // tp,
        "key_head_dim": gdn.key_head_dim,
        "value_head_dim": gdn.value_head_dim,
        "mixed_qkv_width": gdn.conv_dim // tp,
        "projected_qkvz_width": (2 * gdn.key_dim + 2 * gdn.value_dim) // tp,
        "projected_ba_width": 2 * gdn.num_value_heads // tp,
        "gate_width": gdn.num_value_heads // tp,
        "conv_kernel_size": gdn.conv_kernel_size,
        "conv_state_shape": (gdn.conv_dim // tp, layout.conv_state_shape[0]),
        "recurrent_state_shape": layout.recurrent_state_shape,
        "activation_dtype": "bfloat16",
        "activation": "silu",
        "conv_state_dtype": "bfloat16",
        "recurrent_state_dtype": "float32",
        "conv_bias": False,
        "padding_cache_index": -1,
        "includes_input_reordering": True,
        "includes_gate_materialization": True,
        "includes_gated_norm": False,
    }


def validate_gdn_core_spec(spec, tp):
    required = {
        "q_heads", "value_heads", "key_head_dim", "value_head_dim",
        "mixed_qkv_width", "projected_qkvz_width", "projected_ba_width",
        "gate_width", "conv_kernel_size",
        "conv_state_shape", "recurrent_state_shape", "activation_dtype", "activation",
        "conv_state_dtype", "recurrent_state_dtype", "conv_bias", "padding_cache_index",
        "includes_input_reordering", "includes_gate_materialization", "includes_gated_norm",
    }
    if not isinstance(spec, dict) or set(spec) != required or type(tp) is not int or tp < 1:
        raise ValueError("Invalid GDN core shape contract")
    integers = ("q_heads", "value_heads", "key_head_dim", "value_head_dim",
                "mixed_qkv_width", "projected_qkvz_width", "projected_ba_width",
                "gate_width", "conv_kernel_size")
    if (any(type(spec[key]) is not int or spec[key] < 1 for key in integers)
            or spec["activation_dtype"] != "bfloat16"
            or spec["activation"] != "silu"
            or spec["conv_state_dtype"] != "bfloat16"
            or spec["recurrent_state_dtype"] != "float32"
            or spec["conv_bias"] is not False
            or spec["padding_cache_index"] != -1
            or spec["includes_input_reordering"] is not True
            or spec["includes_gate_materialization"] is not True
            or spec["includes_gated_norm"] is not False):
        raise ValueError("Malformed GDN core state/precision contract")
    q, hv, k, v = (spec[key] for key in
                    ("q_heads", "value_heads", "key_head_dim", "value_head_dim"))
    if (hv % q or spec["gate_width"] != hv
            or spec["mixed_qkv_width"] != 2 * q * k + hv * v
            or spec["projected_qkvz_width"] != spec["mixed_qkv_width"] + hv * v
            or spec["projected_ba_width"] != 2 * hv
            or tuple(spec["conv_state_shape"]) !=
                (spec["mixed_qkv_width"], spec["conv_kernel_size"] - 1)
            or tuple(spec["recurrent_state_shape"]) != (hv, v, k)):
        raise ValueError("Inconsistent GDN core head/state shapes")


def make_gdn_core_primitive(size, logical_size, model, group, random):
    """Build one independent SGLang packed-decode invocation and FP32 reference."""
    import torch
    import torch.nn.functional as F
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    if (type(size) is not int or type(logical_size) is not int
            or not 1 <= logical_size <= size):
        raise ValueError("GDN decode needs positive logical <= physical size")
    spec = gdn_core_spec(model, group.world_size)
    validate_gdn_core_spec(spec, group.world_size)
    if model.activation != "silu":
        raise ValueError("Packed GDN producer currently implements SiLU convolution only")
    qh, vh = spec["q_heads"], spec["value_heads"]
    kd, vd = spec["key_head_dim"], spec["value_head_dim"]
    projected_qkvz = random((size, spec["projected_qkvz_width"]), .1)
    projected_ba = random((size, spec["projected_ba_width"]), .1)
    conv_weight = random((spec["mixed_qkv_width"], spec["conv_kernel_size"]), .1)
    A_log = torch.zeros(vh, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros(vh, device="cuda", dtype=torch.float32)
    conv_state = random((logical_size, *spec["conv_state_shape"]), .1)
    recurrent_state = (torch.randn((logical_size, *spec["recurrent_state_shape"]),
                                   device="cuda", dtype=torch.float32) * .01).contiguous()
    cache_indices = torch.cat((
        torch.arange(logical_size, device="cuda", dtype=torch.int32),
        torch.full((size - logical_size,), spec["padding_cache_index"],
                   device="cuda", dtype=torch.int32),
    ))

    # Independent reference for active rows. It follows the packed Triton
    # kernel's BF16 beta rounding and FP32 recurrent-state update exactly.
    conv_initial = conv_state.clone()
    recurrent_initial = recurrent_state.clone()
    q_width, v_width = qh * kd, vh * vd
    query, key, value, z_ref = projected_qkvz.split(
        (q_width, q_width, v_width, v_width), dim=-1)
    b_ref, a_ref = projected_ba.split((vh, vh), dim=-1)
    mixed_ref = torch.cat((query, key, value.reshape(size, -1)), dim=-1)
    active_x = mixed_ref[:logical_size].float()
    window = torch.cat((conv_initial.float(), active_x.unsqueeze(-1)), dim=-1)
    conv_ref = F.silu((window * conv_weight.float().unsqueeze(0)).sum(-1)).to(
        projected_qkvz.dtype)
    q, k, v = torch.split(conv_ref, (qh * kd, qh * kd, vh * vd), dim=-1)
    q = q.view(logical_size, qh, kd).float()
    k = k.view(logical_size, qh, kd).float()
    v = v.view(logical_size, vh, vd)
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q * (kd ** -0.5)
    head_map = torch.arange(vh, device="cuda") // (vh // qh)
    q, k = q[:, head_map], k[:, head_map]
    decay = torch.exp(-torch.exp(A_log)[None, :] * F.softplus(
        a_ref[:logical_size].float() + dt_bias[None, :]))
    beta = torch.sigmoid(b_ref[:logical_size].float()).to(projected_ba.dtype).float()
    state_ref = recurrent_initial * decay[:, :, None, None]
    residual = v.float() - torch.einsum("bhvk,bhk->bhv", state_ref, k)
    residual = residual * beta[:, :, None]
    state_ref = state_ref + residual[:, :, :, None] * k[:, :, None, :]
    output_ref = torch.zeros((1, size, vh, vd), device="cuda", dtype=projected_qkvz.dtype)
    output_ref[:, :logical_size] = torch.einsum(
        "bhvk,bhk->bhv", state_ref, q).to(projected_qkvz.dtype).unsqueeze(0)
    conv_final = conv_initial.clone()
    conv_final[:, :, :-1] = conv_initial[:, :, 1:]
    conv_final[:, :, -1] = mixed_ref[:logical_size]

    kernel = TritonGDNKernel()
    if not kernel.supports_packed_decode:
        raise ValueError("Pinned SGLang runtime does not select packed GDN decode")

    def fn():
        query, key, value, z = projected_qkvz.split(
            (q_width, q_width, v_width, v_width), dim=-1)
        b, a = projected_ba.split((vh, vh), dim=-1)
        b, a = b.contiguous(), a.contiguous()
        mixed = torch.cat((query.reshape(size, -1), key.reshape(size, -1),
                           value.reshape(size, -1)), dim=-1)
        convolved = causal_conv1d_update(
            mixed, conv_state, conv_weight, None, "silu",
            conv_state_indices=cache_indices,
        )
        output = kernel.packed_decode(
            convolved, a, b, A_log=A_log, dt_bias=dt_bias,
            scale=kd ** -0.5, ssm_states=recurrent_state,
            cache_indices=cache_indices, num_v_heads=vh, head_v_dim=vd,
        )
        # Z is a strided tail view of the wider QKVZ projection. Qwen's
        # pre-norm reshape materializes it; the native trace charges that copy
        # before the gated-norm anchor, so it belongs to the core boundary.
        z = z.reshape(-1, vd)
        # Basic slices are views, so state evidence adds no timed kernels.
        return output, z, conv_state, recurrent_state[:, :2]

    reference = (output_ref, z_ref.float().to(projected_qkvz.dtype).reshape(-1, vd),
                 conv_final, state_ref[:, :2])
    mutable = (conv_state, recurrent_state)
    backend = "Qwen3.5.reorder+SGLang.causal_conv1d_update+TritonGDNKernel.packed_decode+Z.materialize"
    return fn, reference, mutable, backend, spec
