"""Shape/workload contracts for Qwen3.8 SGLang decode-attention primitives."""

from __future__ import annotations

import math


ATTENTION_PRIMITIVES = (
    "attn_pre_proj_qknorm",
    "attn_rope",
    "attn_kv_cache_write",
    "attn_decode",
)


def attention_decode_spec(model, tp):
    """Return the pinned TP-local Qwen attention layout and boundary ownership."""
    if type(tp) is not int or tp < 1 or model.num_q_heads % tp:
        raise ValueError("Invalid attention tensor-parallel size")
    head_dim = model.get_head_dim()
    if model.num_kv_heads >= tp:
        if model.num_kv_heads % tp:
            raise ValueError("KV heads do not divide the attention TP size")
        kv_heads = model.num_kv_heads // tp
    else:
        if tp % model.num_kv_heads:
            raise ValueError("Attention TP size does not replicate KV heads evenly")
        kv_heads = 1
    q_heads = model.num_q_heads // tp
    rotary_dim = int(head_dim * model.partial_rotary_factor)
    if (not model.use_qk_norm or not model.attn_output_gate or rotary_dim < 1
            or rotary_dim > head_dim or rotary_dim % 2):
        raise ValueError("Unsupported Qwen attention norm/gate/rotary configuration")
    projected_width = 2 * q_heads * head_dim + 2 * kv_heads * head_dim
    return {
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
        "projected_qkv_gate_width": projected_width,
        "local_qkv_weight_shape": (projected_width, model.embedding_dim),
        "activation_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "kv_cache_layout": "NHD",
        "page_size": 1,
        "attention_backend": "aiter.paged_attention_ragged",
        "qk_norm_backend": "fused_qk_gemma_rmsnorm_with_gate",
        "rope_backend": "SGLang.RotaryEmbedding",
        "kv_write_backend": "SGLang.store_cache",
        "qk_norm_eps": model.rms_norm_eps,
        "rope_theta": model.rope_theta,
        "max_position_embeddings": model.max_position_embeddings,
        "is_neox_style": True,
        "output_gate": True,
        "includes_output_projection": False,
        "context_pattern": "uniform_logical_prefix_padding_suffix",
        "padding_context_length": 1,
    }


def validate_attention_decode_spec(spec, tp):
    required = {
        "q_heads", "kv_heads", "head_dim", "rotary_dim", "projected_qkv_gate_width",
        "local_qkv_weight_shape", "activation_dtype", "kv_cache_dtype", "kv_cache_layout",
        "page_size", "attention_backend", "qk_norm_backend", "rope_backend",
        "kv_write_backend", "qk_norm_eps", "rope_theta", "max_position_embeddings",
        "is_neox_style", "output_gate", "includes_output_projection", "context_pattern",
        "padding_context_length",
    }
    if not isinstance(spec, dict) or set(spec) != required or type(tp) is not int or tp < 1:
        raise ValueError("Invalid attention decode shape contract")
    integers = ("q_heads", "kv_heads", "head_dim", "rotary_dim",
                "projected_qkv_gate_width", "page_size", "max_position_embeddings",
                "padding_context_length")
    if (any(type(spec[k]) is not int or spec[k] < 1 for k in integers)
            or type(spec["qk_norm_eps"]) not in (int, float)
            or not math.isfinite(spec["qk_norm_eps"]) or spec["qk_norm_eps"] <= 0
            or type(spec["rope_theta"]) not in (int, float)
            or not math.isfinite(spec["rope_theta"]) or spec["rope_theta"] <= 0
            or spec["activation_dtype"] != "bfloat16"
            or spec["kv_cache_dtype"] != "bfloat16"
            or spec["kv_cache_layout"] != "NHD" or spec["page_size"] != 1
            or spec["attention_backend"] != "aiter.paged_attention_ragged"
            or spec["qk_norm_backend"] != "fused_qk_gemma_rmsnorm_with_gate"
            or spec["rope_backend"] != "SGLang.RotaryEmbedding"
            or spec["kv_write_backend"] != "SGLang.store_cache"
            or spec["is_neox_style"] is not True or spec["output_gate"] is not True
            or spec["includes_output_projection"] is not False
            or spec["context_pattern"] != "uniform_logical_prefix_padding_suffix"
            or spec["padding_context_length"] != 1):
        raise ValueError("Malformed attention decode backend/precision contract")
    qh, kh, dim = spec["q_heads"], spec["kv_heads"], spec["head_dim"]
    weight = tuple(spec["local_qkv_weight_shape"])
    if (len(weight) != 2 or any(type(v) is not int or v < 1 for v in weight)
            or spec["rotary_dim"] > dim or spec["rotary_dim"] % 2
            or spec["projected_qkv_gate_width"] != 2 * qh * dim + 2 * kh * dim
            or weight[0] != spec["projected_qkv_gate_width"]):
        raise ValueError("Inconsistent attention head/projection layout")


def validate_attention_workload(size, logical_size, physical_context_lens, spec):
    """Validate the bounded uniform-context decode workload used for interpolation."""
    if (type(size) is not int or type(logical_size) is not int
            or not 1 <= logical_size <= size
            or not isinstance(physical_context_lens, (tuple, list))
            or len(physical_context_lens) != size
            or any(type(v) is not int or v < 1 for v in physical_context_lens)):
        raise ValueError("Invalid attention logical/physical decode workload")
    active = tuple(physical_context_lens[:logical_size])
    padding = tuple(physical_context_lens[logical_size:])
    if len(set(active)) != 1 or any(v != spec["padding_context_length"] for v in padding):
        raise ValueError("Attention calibration requires uniform logical contexts and suffix padding")
    if active[0] >= spec["max_position_embeddings"]:
        raise ValueError("Attention context exceeds the model rotary cache")
    return {
        "logical_size": logical_size,
        "padding_count": size - logical_size,
        "active_context_length": active[0],
        "padding_context_length": spec["padding_context_length"],
    }


def _reference_rope(x, positions, cache, rotary_dim, head_dim):
    import torch

    original_shape = x.shape
    x = x.float().view(len(positions), -1, head_dim)
    rotated, passed = x[..., :rotary_dim], x[..., rotary_dim:]
    cos, sin = cache.index_select(0, positions).float().chunk(2, dim=-1)
    cos, sin = cos[:, None, :], sin[:, None, :]
    first, second = rotated.chunk(2, dim=-1)
    rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
    return torch.cat((rotated, passed), dim=-1).reshape(original_shape)


def make_attention_primitive(name, size, logical_size, physical_context_lens,
                             model, group, rank, random):
    """Build one native attention-boundary invocation with an independent reference."""
    import torch

    if name not in ATTENTION_PRIMITIVES:
        raise ValueError(f"Unknown attention primitive {name}")
    spec = attention_decode_spec(model, group.world_size)
    validate_attention_decode_spec(spec, group.world_size)
    workload = validate_attention_workload(size, logical_size, physical_context_lens, spec)
    qh, kh, dim = spec["q_heads"], spec["kv_heads"], spec["head_dim"]
    dtype = torch.bfloat16

    if name == "attn_pre_proj_qknorm":
        from sglang.srt.layers.linear import QKVParallelLinear
        from sglang.srt.models.utils import fused_qk_gemma_rmsnorm_with_gate

        x = random((size, model.embedding_dim), .1)
        linear = QKVParallelLinear(
            model.embedding_dim, dim, model.num_q_heads * 2, model.num_kv_heads,
            bias=False, quant_config=None, tp_rank=rank, tp_size=group.world_size,
            kv_tp_rank=rank, kv_tp_size=group.world_size,
        ).to(device="cuda", dtype=dtype)
        linear.weight.data.copy_(random(linear.weight.shape, model.embedding_dim ** -.5))
        q_weight = random((dim,), .05)
        k_weight = random((dim,), .05)
        projected = torch.nn.functional.linear(x.float(), linear.weight.float())
        q_gate, key, value = projected.split((2 * qh * dim, kh * dim, kh * dim), dim=-1)
        q_gate = q_gate.view(size, qh, 2 * dim)
        query, gate = q_gate.split((dim, dim), dim=-1)
        key = key.view(size, kh, dim)
        query = (query * torch.rsqrt(query.square().mean(-1, keepdim=True)
                                     + spec["qk_norm_eps"]) * (1 + q_weight.float())).to(dtype)
        key = (key * torch.rsqrt(key.square().mean(-1, keepdim=True)
                                 + spec["qk_norm_eps"]) * (1 + k_weight.float())).to(dtype)
        reference = (query.reshape(size, -1), key.reshape(size, -1), value.to(dtype),
                     gate.to(dtype).reshape(size, -1))

        def fn():
            qkv, _ = linear(x)
            q_gate, key, value = qkv.split((2 * qh * dim, kh * dim, kh * dim), dim=-1)
            query, key, gate = fused_qk_gemma_rmsnorm_with_gate(
                q_gate, key, q_weight, k_weight, spec["qk_norm_eps"], dim, qh)
            return (query.view(size, -1), key.view(size, -1), value,
                    gate.view(size, -1))

        backend = f"QKVParallelLinear.{type(linear.quant_method).__name__}+{spec['qk_norm_backend']}"
        return fn, reference, (), backend, spec, workload

    positions = torch.tensor([v - 1 for v in physical_context_lens], device="cuda", dtype=torch.int64)
    if name == "attn_rope":
        from sglang.srt.layers.rotary_embedding import get_rope

        rope = get_rope(
            head_size=dim, rotary_dim=dim,
            max_position=spec["max_position_embeddings"], rope_scaling=model.rope_scaling,
            base=spec["rope_theta"], partial_rotary_factor=model.partial_rotary_factor,
            is_neox_style=True, dtype=dtype,
        )
        query = random((size, qh * dim), .1)
        key = random((size, kh * dim), .1)
        query_initial, key_initial = query.clone(), key.clone()
        cache = rope.cos_sin_cache.to(device="cuda")
        reference = (
            _reference_rope(query_initial, positions, cache, spec["rotary_dim"], dim).to(dtype),
            _reference_rope(key_initial, positions, cache, spec["rotary_dim"], dim).to(dtype),
        )
        return (lambda: rope(positions, query, key)), reference, (query, key), \
            spec["rope_backend"], spec, workload

    lengths = torch.tensor(physical_context_lens, device="cuda", dtype=torch.int32)
    starts = torch.cumsum(lengths, 0) - lengths + 1
    total_pages = sum(physical_context_lens) + 1
    key = random((size, kh, dim), .1)
    value = random((size, kh, dim), .1)

    if name == "attn_kv_cache_write":
        from sglang.kernels.ops.kvcache.kvcache import store_cache

        locations = starts.clone()
        locations[logical_size:] = 0
        key_cache = random((total_pages, kh * dim), .1)
        value_cache = random((total_pages, kh * dim), .1)
        key_reference, value_reference = key_cache.clone(), value_cache.clone()
        key_reference[locations[:logical_size].long()] = key[:logical_size].reshape(logical_size, -1)
        value_reference[locations[:logical_size].long()] = value[:logical_size].reshape(logical_size, -1)

        def fn():
            store_cache(key.reshape(size, -1), value.reshape(size, -1), key_cache, value_cache,
                        locations, row_bytes=kh * dim * 2, v_row_bytes=kh * dim * 2,
                        size_limit=total_pages, reserved_skip_index=0)
            return key_cache, value_cache

        return fn, (key_reference, value_reference), (key_cache, value_cache), \
            spec["kv_write_backend"], spec, workload

    from aiter import paged_attention_ragged

    query = random((size, qh, dim))
    key_cache = random((total_pages, 1, kh, dim))
    value_cache = random((total_pages, 1, kh, dim))
    # torch.cumsum promotes integer inputs unless dtype is explicit. AITER's
    # ABI reads this buffer as int32; an int64 indptr makes it interpret the
    # high half of each element as the next sequence boundary.
    kv_indptr = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32),
                           torch.cumsum(lengths, 0, dtype=torch.int32)))
    kv_indices = torch.arange(1, total_pages, device="cuda", dtype=torch.int32)
    last_page_lens = torch.ones(size, device="cuda", dtype=torch.int32)
    # SGLang fixes this to the model-wide maximum at backend construction;
    # AITER uses the value for workspace addressing, not merely launch pruning.
    partitions = (spec["max_position_embeddings"] + 255) // 256
    workspace_bytes = size * qh * partitions * dim * 4 + 2 * size * qh * partitions * 4
    workspace = torch.empty(workspace_bytes, device="cuda", dtype=torch.uint8)
    output = torch.full_like(query, float("nan"))
    one = torch.ones(1, device="cuda", dtype=torch.float32)
    scale = dim ** -.5
    refs = []
    offset = 1
    for lane, length in enumerate(physical_context_lens):
        keys = key_cache[offset:offset + length, 0].float()
        values = value_cache[offset:offset + length, 0].float()
        if kh == 1:
            keys = keys.expand(-1, qh, -1)
            values = values.expand(-1, qh, -1)
        elif kh != qh:
            repeat = qh // kh
            keys = keys.repeat_interleave(repeat, dim=1)
            values = values.repeat_interleave(repeat, dim=1)
        scores = torch.einsum("hd,lhd->hl", query[lane].float(), keys) * scale
        probabilities = torch.softmax(scores, dim=-1).to(dtype)
        refs.append(torch.einsum("hl,lhd->hd", probabilities, values.to(dtype)))
        offset += length
    reference = torch.stack(refs).to(dtype)

    def fn():
        return paged_attention_ragged(
            output, workspace, query, key_cache, value_cache, scale, kv_indptr,
            kv_indices, last_page_lens, 1, partitions, None, "auto", "NHD",
            0.0, one, one, None, 256,
        )

    return fn, reference, (), spec["attention_backend"], spec, workload
