"""Shape contracts and native routing primitives for Qwen3.8 SGLang MoE."""

from __future__ import annotations


MOE_ROUTING_PRIMITIVES = ("moe_routing_topk",)
MOE_ROUTED_PRIMITIVES = ("moe_sorting", "moe_experts_quant_gemm_combine")


def moe_routing_spec(model):
    """Return the checkpoint-pinned replicated router/top-k boundary."""
    if (not model.is_moe or model.num_experts < 1
            or model.num_experts_per_tok < 1
            or model.num_experts_per_tok > model.num_experts):
        raise ValueError("MoE routing requires a valid expert configuration")
    return {
        "hidden_size": model.embedding_dim,
        "num_experts": model.num_experts,
        "top_k": model.num_experts_per_tok,
        "logits_dtype": "bfloat16",
        "weights_dtype": "float32",
        "ids_dtype": "int32",
        "scoring": "softmax",
        # The Qwen3.8 checkpoint omits norm_topk_prob; the active SGLang
        # Qwen2MoeSparseMoeBlock therefore passes the false value through.
        "renormalize": False,
        "backend": "aiter.fused_moe.fused_topk/topk_softmax",
        "includes_router_linear": False,
        "includes_padding_mask": False,
    }


def validate_moe_routing_spec(spec):
    required = {
        "hidden_size", "num_experts", "top_k", "logits_dtype",
        "weights_dtype", "ids_dtype", "scoring", "renormalize", "backend",
        "includes_router_linear", "includes_padding_mask",
    }
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError("Invalid MoE routing shape contract")
    if (any(type(spec[k]) is not int or spec[k] < 1
            for k in ("hidden_size", "num_experts", "top_k"))
            or spec["top_k"] > spec["num_experts"]
            or spec["logits_dtype"] != "bfloat16"
            or spec["weights_dtype"] != "float32"
            or spec["ids_dtype"] != "int32"
            or spec["scoring"] != "softmax"
            or spec["renormalize"] is not False
            or spec["backend"] != "aiter.fused_moe.fused_topk/topk_softmax"
            or spec["includes_router_linear"] is not False
            or spec["includes_padding_mask"] is not False):
        raise ValueError("Malformed MoE routing backend/precision contract")


def moe_routed_spec(model, tp):
    """Return the TP-local routed MXFP4 shapes and AITER boundary ownership."""
    if (type(tp) is not int or tp < 1 or not model.is_moe
            or model.routed_mlp_hidden_dim is None
            or model.routed_mlp_hidden_dim % tp):
        raise ValueError("Invalid tensor-parallel routed-expert configuration")
    local_intermediate = model.routed_mlp_hidden_dim // tp
    if local_intermediate % 256:
        raise ValueError("Qwen3.8 AITER MXFP4 profiling requires 256-aligned local experts")
    return {
        "hidden_size": model.embedding_dim,
        "global_intermediate_size": model.routed_mlp_hidden_dim,
        "local_intermediate_size": local_intermediate,
        "num_experts": model.num_experts,
        "top_k": model.num_experts_per_tok,
        "tensor_parallel_size": tp,
        "block_size_m": 32,
        "activation": "silu",
        "output_dtype": "bfloat16",
        "activation_quant": "dynamic_mxfp4_per_1x32_e8m0",
        "weight_quant": "mxfp4_per_1x32_e8m0",
        "w13_packed_shape": (
            model.num_experts, 2 * local_intermediate,
            model.embedding_dim // 2),
        "w2_packed_shape": (
            model.num_experts, model.embedding_dim,
            local_intermediate // 2),
        "w13_scale_shape": (
            model.num_experts, 2 * local_intermediate,
            model.embedding_dim // 32),
        "w2_scale_shape": (
            model.num_experts, model.embedding_dim,
            local_intermediate // 32),
        "sorting_backend": "aiter.moe_sorting/opus",
        "experts_backend": "aiter.fused_moe_2stages",
        "assignment_reconstruction": "largest_remaining_count_then_expert_id",
        "includes_sorting": False,
        "includes_tp_allreduce": False,
    }


def validate_moe_routed_spec(spec, tp):
    required = {
        "hidden_size", "global_intermediate_size", "local_intermediate_size",
        "num_experts", "top_k", "tensor_parallel_size", "block_size_m",
        "activation", "output_dtype", "activation_quant", "weight_quant",
        "w13_packed_shape", "w2_packed_shape", "w13_scale_shape",
        "w2_scale_shape", "sorting_backend", "experts_backend",
        "assignment_reconstruction", "includes_sorting", "includes_tp_allreduce",
    }
    if (not isinstance(spec, dict) or set(spec) != required
            or type(tp) is not int or tp < 1):
        raise ValueError("Invalid routed MoE shape contract")
    integers = (
        "hidden_size", "global_intermediate_size", "local_intermediate_size",
        "num_experts", "top_k", "tensor_parallel_size", "block_size_m",
    )
    if (any(type(spec[k]) is not int or spec[k] < 1 for k in integers)
            or spec["tensor_parallel_size"] != tp
            or spec["global_intermediate_size"] != spec["local_intermediate_size"] * tp
            or spec["top_k"] > spec["num_experts"]
            or spec["block_size_m"] != 32
            or spec["activation"] != "silu"
            or spec["output_dtype"] != "bfloat16"
            or spec["activation_quant"] != "dynamic_mxfp4_per_1x32_e8m0"
            or spec["weight_quant"] != "mxfp4_per_1x32_e8m0"
            or spec["sorting_backend"] != "aiter.moe_sorting/opus"
            or spec["experts_backend"] != "aiter.fused_moe_2stages"
            or spec["assignment_reconstruction"]
                != "largest_remaining_count_then_expert_id"
            or spec["includes_sorting"] is not False
            or spec["includes_tp_allreduce"] is not False):
        raise ValueError("Malformed routed MoE backend/precision contract")
    e, h, local = spec["num_experts"], spec["hidden_size"], spec["local_intermediate_size"]
    expected = {
        "w13_packed_shape": (e, 2 * local, h // 2),
        "w2_packed_shape": (e, h, local // 2),
        "w13_scale_shape": (e, 2 * local, h // 32),
        "w2_scale_shape": (e, h, local // 32),
    }
    if any(tuple(spec[name]) != shape for name, shape in expected.items()):
        raise ValueError("Inconsistent routed MoE packed tensor shapes")


def validate_moe_route_workload(size, physical_expert_counts, spec):
    counts = tuple(physical_expert_counts)
    if (type(size) is not int or size < 1 or len(counts) != spec["num_experts"]
            or any(type(value) is not int or not 0 <= value <= size for value in counts)
            or sum(counts) != size * spec["top_k"]):
        raise ValueError("Invalid routed MoE physical expert counts")
    return {
        "physical_size": size,
        "active_experts": sum(value > 0 for value in counts),
        "maximum_expert_load": max(counts),
        "sorted_token_blocks": sum(
            (value + spec["block_size_m"] - 1) // spec["block_size_m"]
            for value in counts if value),
    }


def reconstruct_topk_ids(size, physical_expert_counts, spec):
    """Construct a deterministic valid assignment for an observed route histogram."""
    validate_moe_route_workload(size, physical_expert_counts, spec)
    remaining = list(physical_expert_counts)
    rows = []
    for _ in range(size):
        selected = sorted(
            (expert for expert, count in enumerate(remaining) if count),
            key=lambda expert: (-remaining[expert], expert),
        )[:spec["top_k"]]
        if len(selected) != spec["top_k"]:
            raise ValueError("Expert histogram cannot form unique per-token top-k assignments")
        rows.append(tuple(selected))
        for expert in selected:
            remaining[expert] -= 1
    if any(remaining):
        raise ValueError("Expert histogram reconstruction left unassigned routes")
    return tuple(rows)


def make_moe_sorting_primitive(
        size, physical_expert_counts, model, group, *, validate=True):
    """Build an exact-histogram AITER Opus sorting invocation."""
    import torch
    from aiter.fused_moe import moe_sorting

    spec = moe_routed_spec(model, group.world_size)
    validate_moe_routed_spec(spec, group.world_size)
    workload = validate_moe_route_workload(size, physical_expert_counts, spec)
    assignments = reconstruct_topk_ids(size, physical_expert_counts, spec)
    topk_ids = torch.tensor(assignments, device="cuda", dtype=torch.int32)
    topk_weights = torch.full(
        topk_ids.shape, 1. / spec["top_k"], device="cuda", dtype=torch.float32)

    def raw():
        return moe_sorting(
            topk_ids, topk_weights, spec["num_experts"], spec["hidden_size"],
            torch.bfloat16, spec["block_size_m"], accumulate=True)

    expected_valid = torch.tensor(
        [workload["sorted_token_blocks"] * spec["block_size_m"], size],
        device="cuda", dtype=torch.int32)
    if validate:
        # Validate Opus' packed token-slot encoding and expert-block ownership
        # for representative independent invocations. Only raw() is timed.
        sorted_ids, sorted_weights, sorted_experts, valid, _ = raw()
        torch.testing.assert_close(valid, expected_valid, atol=0, rtol=0)
        blocks = workload["sorted_token_blocks"]
        active = [expert for expert, count in enumerate(physical_expert_counts) if count]
        torch.testing.assert_close(
            sorted_experts[:blocks],
            torch.tensor(active, device="cuda", dtype=torch.int32), atol=0, rtol=0)
        encoded = sorted_ids[:blocks * spec["block_size_m"]].view(
            blocks, spec["block_size_m"])
        weights = sorted_weights[:blocks * spec["block_size_m"]].view_as(encoded)
        token_mask = (1 << 24) - 1
        tokens, slots = encoded & token_mask, encoded >> 24
        seen = [0] * spec["num_experts"]
        for block, expert in enumerate(active):
            valid_lanes = tokens[block] < size
            if (not bool((slots[block][valid_lanes] < spec["top_k"]).all())
                    or not bool((weights[block][~valid_lanes] == 0).all())):
                raise ValueError("Malformed AITER sorted route padding")
            lane_tokens = tokens[block][valid_lanes].long()
            lane_slots = slots[block][valid_lanes].long()
            if not bool((topk_ids[lane_tokens, lane_slots] == expert).all()):
                raise ValueError("AITER sorted route disagrees with input assignment")
            torch.testing.assert_close(
                weights[block][valid_lanes], topk_weights[lane_tokens, lane_slots],
                atol=0, rtol=0)
            seen[expert] = int(valid_lanes.sum().item())
        if tuple(seen) != tuple(physical_expert_counts):
            raise ValueError("AITER sorted route disagrees with expert histogram")

    def fn():
        return raw()[3]

    return fn, expected_valid, (), spec["sorting_backend"], spec, workload


def make_moe_experts_primitive(
        size, physical_expert_counts, model, group, *, validate=True):
    """Build the two-stage MXFP4 expert boundary after an untimed route sort."""
    import torch
    import aiter
    from aiter.fused_moe import fused_moe_2stages, moe_sorting

    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise ValueError("Native MXFP4 dtype is required for routed expert profiling")
    spec = moe_routed_spec(model, group.world_size)
    validate_moe_routed_spec(spec, group.world_size)
    workload = validate_moe_route_workload(size, physical_expert_counts, spec)
    assignments = reconstruct_topk_ids(size, physical_expert_counts, spec)
    topk_ids = torch.tensor(assignments, device="cuda", dtype=torch.int32)
    topk_weights = torch.full(
        topk_ids.shape, 1. / spec["top_k"], device="cuda", dtype=torch.float32)
    sorted_ids, sorted_weights, sorted_experts, valid, moe_out = moe_sorting(
        topk_ids, topk_weights, spec["num_experts"], spec["hidden_size"],
        torch.bfloat16, spec["block_size_m"], accumulate=True)
    expected_valid = torch.tensor(
        [workload["sorted_token_blocks"] * spec["block_size_m"], size],
        device="cuda", dtype=torch.int32)
    if validate:
        torch.testing.assert_close(valid, expected_valid, atol=0, rtol=0)

    # Zero-valued packed weights keep the exact tensor extents, expert address
    # strides and memory traffic while providing an independent exact-zero
    # oracle. Zero is invariant under the checkpoint's 16x16 weight shuffle;
    # the marker selects the same AITER kernels as processed Quark weights.
    w13 = torch.zeros(
        spec["w13_packed_shape"], device="cuda", dtype=torch.uint8,
    ).view(torch.float4_e2m1fn_x2)
    w2 = torch.zeros(
        spec["w2_packed_shape"], device="cuda", dtype=torch.uint8,
    ).view(torch.float4_e2m1fn_x2)
    w13.is_shuffled = True
    w2.is_shuffled = True
    w13_scale = torch.zeros(
        spec["w13_scale_shape"], device="cuda", dtype=torch.uint8)
    w2_scale = torch.zeros(
        spec["w2_scale_shape"], device="cuda", dtype=torch.uint8)
    hidden_states = (
        torch.randn((size, spec["hidden_size"]), device="cuda", dtype=torch.bfloat16)
        * .1).contiguous()
    reference = torch.zeros_like(hidden_states)

    def fn():
        return fused_moe_2stages(
            hidden_states, w13, w2, spec["top_k"], sorted_ids, sorted_weights,
            sorted_experts, valid, moe_out, True, spec["block_size_m"],
            activation=aiter.ActivationType.Silu,
            quant_type=aiter.QuantType.per_1x32,
            q_dtype_a=torch.float4_e2m1fn_x2,
            q_dtype_w=torch.float4_e2m1fn_x2,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    return fn, reference, (moe_out,), spec["experts_backend"], spec, workload


def make_moe_routing_primitive(name, size, model, random):
    """Build the native AITER top-k call and independently verify its routing."""
    import torch
    from sglang.srt.layers.moe.topk import _use_aiter, fused_topk

    if name not in MOE_ROUTING_PRIMITIVES:
        raise ValueError(f"Unknown MoE routing primitive {name}")
    if not _use_aiter:
        raise ValueError("Expected the captured SGLang AITER top-k path")
    spec = moe_routing_spec(model)
    validate_moe_routing_spec(spec)
    hidden_states = random((size, spec["hidden_size"]), .1)
    logits = random((size, spec["num_experts"]), 1.)
    reference_scores = torch.softmax(logits.float(), dim=-1)
    reference_weights, reference_ids = torch.topk(
        reference_scores, spec["top_k"], dim=-1)

    # AITER owns output ordering. Validate the expert set and each gathered
    # probability once per independent invocation without adding canonicalizing
    # sort/gather kernels to the measured graph boundary.
    eager_weights, eager_ids = fused_topk(
        hidden_states, logits, spec["top_k"], spec["renormalize"])
    torch.testing.assert_close(
        torch.sort(eager_ids, dim=-1).values,
        torch.sort(reference_ids.to(torch.int32), dim=-1).values,
        atol=0, rtol=0)
    torch.testing.assert_close(
        eager_weights, reference_scores.gather(1, eager_ids.long()),
        atol=1e-6, rtol=1e-5)

    def fn():
        # Returning weights avoids imposing a non-native expert-id ordering on
        # the graph. The eager checks above cover ids and id/weight pairing.
        return fused_topk(
            hidden_states, logits, spec["top_k"], spec["renormalize"])[0]

    return fn, reference_weights, (), spec["backend"], spec
