"""Source workloads for eager DeepSeek-V4 FlashMLA attention components.

These workloads do not admit serving or predict native auxiliary-stream latency.
SWA window spans describe logical interval unions, not physical gather execution.
Source: vLLM 2cf0a6915ce544dc493a0990f2ea38d81601128a,
models/deepseek_v4/{attention,compressor,nvidia/flashmla}.py.
"""

from __future__ import annotations

from typing import Any

from frontier.attention.model_binding import bind_layer_attention
from frontier.entities.batch import Batch, EPBatchGroup
from frontier.types import ClusterType


def _integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _validate_source_batch(batch):
    if (
        isinstance(batch, EPBatchGroup) or not isinstance(batch, Batch)
        or getattr(batch, "lane_workload", None) is not None
    ):
        raise ValueError("V4 attention requires a source DP attention Batch")
    if batch.is_idle or not batch.requests:
        raise ValueError("V4 attention requires a nonempty active Batch")
    if batch.spec_decode_metadata is not None:
        raise NotImplementedError("V4 speculative attention is unsupported")
    if batch.afd_stage_metadata is not None:
        raise NotImplementedError("V4 PD attention is unsupported")
    graph = batch.decode_cuda_graph_metadata
    if batch.get_decode_cuda_graph_runtime_mode() != "NONE" or (
        graph is not None and graph.capture_hit is not False
    ):
        raise NotImplementedError("V4 component requires actual eager graph state")
    fields = (batch.num_tokens, batch.num_context_tokens, batch.request_is_decoding)
    if any(not isinstance(v, (list, tuple)) or len(v) != len(batch.requests)
           for v in fields):
        raise ValueError("V4 scheduling snapshots require one entry per request")
    for query, context, decoding in zip(*fields):
        _integer(query, "query rows")
        _integer(context, "context rows", minimum=0)
        if type(decoding) is not bool:
            raise ValueError("request_is_decoding entries must be bool")
        if decoding and query != 1:
            raise ValueError("V4 ordinary decode requires one query row")
    expected_prefill = sum(q for q, d in zip(fields[0], fields[2]) if not d)
    for name, expected in (
        ("total_num_tokens", sum(fields[0])),
        ("num_prefill_tokens", expected_prefill),
        ("num_decode_tokens", sum(fields[0]) - expected_prefill),
    ):
        value = _integer(getattr(batch, name), name, minimum=0)
        if value != expected:
            raise ValueError(f"V4 {name} does not match scheduling snapshots")


def _call(input_shape, output_shape, **features):
    """One source component call, not an internal GPU kernel launch count.

    Shapes are logical tensor views, not packed-storage allocation claims.
    """
    return {"input_shape": input_shape, "output_shape": output_shape, **features}


def build_deepseek_v4_attention_workload(
    model_config, replica_config, batch, global_layer_id: int, *,
    cluster_type, execution_mode: str, has_attention_metadata: bool,
    prefix_caching: bool, decode_context_parallel_size: int,
    prefill_context_parallel_size: int,
) -> dict[str, Any]:
    """Build per-call physical scope workloads from scheduling snapshots."""
    if model_config.get_model_architecture_profile().profile_id != "deepseek_v4":
        raise ValueError("V4 attention workload requires the DeepSeek-V4 profile")
    if cluster_type is not ClusterType.MONOLITHIC:
        raise NotImplementedError("V4 attention component requires co-location")
    if execution_mode != "eager" or has_attention_metadata is not True:
        raise NotImplementedError("V4 component requires real metadata eager execution")
    if prefix_caching is not False:
        raise NotImplementedError("V4 component does not support prefix caching")
    if _integer(decode_context_parallel_size, "DCP") != 1:
        raise NotImplementedError("V4 attention component requires DCP1")
    if _integer(prefill_context_parallel_size, "PCP") != 1:
        raise NotImplementedError("V4 attention component requires PCP1")
    if _integer(replica_config.num_pipeline_stages, "PP") != 1:
        raise NotImplementedError("V4 attention component requires PP1")
    if replica_config.attention_backend != "FLASHMLA_SPARSE_DSV4":
        raise NotImplementedError("V4 component requires explicit FLASHMLA_SPARSE_DSV4")
    if replica_config.speculative_decoding_config.enabled:
        raise NotImplementedError("V4 speculative attention is unsupported")
    _validate_source_batch(batch)
    layer = bind_layer_attention(model_config, global_layer_id)
    tokens = sum(batch.num_tokens)
    hidden = _integer(model_config.embedding_dim, "hidden width")
    head = _integer(model_config.get_head_dim(), "head dimension")
    if head != 512:
        raise NotImplementedError("V4 FlashMLA requires head dimension 512")
    rank = _integer(model_config.q_lora_rank, "Q rank")
    output_rank = _integer(model_config.o_lora_rank, "output rank")
    heads = _integer(model_config.num_q_heads, "Q heads")
    groups = _integer(model_config.o_groups, "output groups")
    _integer(model_config.sliding_window, "sliding window")
    tp = _integer(replica_config.attn_tensor_parallel_size, "attention TP")
    _integer(replica_config.attn_dp, "attention DP")
    if heads % tp or groups % tp or heads % groups:
        raise ValueError("V4 Q heads and output groups must divide attention TP")
    local_heads = heads // tp
    local_groups = groups // tp
    if local_heads > 128:
        raise NotImplementedError("V4 FlashMLA supports at most 128 local Q heads")
    padded_heads = 64 if local_heads <= 64 else 128
    query = (tokens, local_heads, head)
    padded = (tokens, padded_heads, head)
    quantized_output = (tokens, local_groups, heads // groups * head)
    grouped = (tokens, local_groups, output_rank)
    ratio = model_config.compress_ratios[global_layer_id]
    contexts = tuple(batch.num_context_tokens)
    queries = tuple(batch.num_tokens)
    sequences = tuple(c + q for c, q in zip(contexts, queries))
    history = tuple(s // ratio for s in sequences) if ratio > 1 else (0,) * len(queries)
    writes = (tuple(s // ratio - c // ratio for s, c in zip(sequences, contexts))
              if ratio > 1 else (0,) * len(queries))
    decode_indices = tuple(i for i, q in enumerate(queries) if q == 1)
    prefill_indices = tuple(i for i, q in enumerate(queries) if q > 1)
    operators = {
        "dsv4_fused_qkv_projection": (_call((tokens, hidden), (tokens, rank + head)),),
        "dsv4_fused_qkv_rmsnorm": (_call((tokens, rank + head), (tokens, rank + head)),),
        "dsv4_q_projection": (_call((tokens, rank), query),),
        "dsv4_fused_q_kv_prepare": (_call(
            query, padded, kv_input_shape=(tokens, head), kv_cache_write_rows=tokens,
        ),),
        "dsv4_sparse_attention": (_call(
            padded, padded,
            swa_logical_window_span_rows=tuple(
                q + min(c, model_config.sliding_window - 1)
                for q, c in zip(queries, contexts)
            ),
            compressed_history_rows=history,
            query_intervals=tuple(zip(contexts, sequences)),
            compression_ratio=ratio, window_size=model_config.sliding_window,
            decode_request_indices=decode_indices,
            prefill_request_indices=prefill_indices,
        ),),
        "dsv4_output_inv_rope_quant": (_call(query, quantized_output),),
        "dsv4_grouped_output_projection": (_call(quantized_output, grouped),),
        "dsv4_output_projection": (_call(
            (tokens, local_groups * output_rank), (tokens, hidden),
        ),),
    }
    if ratio > 1:
        operators.update(_compressor_workloads(
            "dsv4_compressor", tokens, hidden, head, ratio, writes,
        ))
    if layer.variant_id == "csa":
        index_heads = _integer(model_config.index_n_heads, "index heads")
        index_dim = _integer(model_config.index_head_dim, "index dimension")
        _integer(model_config.index_topk, "index topk")
        index_query = (tokens, index_heads, index_dim)
        operators["dsv4_index_weights_projection"] = (
            _call((tokens, hidden), (tokens, index_heads)),
        )
        operators.update(_compressor_workloads(
            "dsv4_index_compressor", tokens, hidden, index_dim, ratio, writes,
        ))
        if max(sequences) // ratio <= model_config.index_topk:
            operators["dsv4_index_short_context_fill"] = (
                _call((tokens,), (tokens, model_config.index_topk)),
            )
        else:
            operators.update({
                "dsv4_index_query_projection": (_call((tokens, rank), index_query),),
                "dsv4_index_fused_query_prepare": (_call(index_query, index_query),),
                # Candidate counts are logical ranges, not a materialized logits
                # allocation or a claim about the runtime's internal chunking.
                "dsv4_index_scores": (_call(
                    index_query, None, compressed_history_rows=history,
                ),),
                "dsv4_index_topk": (_call(
                    (tokens,), (tokens, model_config.index_topk),
                    compressed_history_rows=history,
                ),),
            })
    return {
        "global_layer_id": global_layer_id, "variant": layer.variant_id,
        "num_tokens": tokens, "num_local_q_heads": local_heads,
        "num_padded_q_heads": padded_heads, "operators": operators,
        "request_intervals": tuple(zip(contexts, sequences)),
        "semantic_is_decoding": tuple(batch.request_is_decoding),
        "source_request_order": decode_indices + prefill_indices,
        "decode_request_indices": decode_indices,
        "prefill_request_indices": prefill_indices,
        "num_decode_query_tokens": sum(queries[i] for i in decode_indices),
        "num_prefill_query_tokens": sum(queries[i] for i in prefill_indices),
        "new_compressed_rows": writes,
    }


def _compressor_workloads(prefix, tokens, hidden, head, ratio, writes):
    overlap = 2 if ratio == 4 else 1
    width = 2 * overlap * head
    operators = {
        f"{prefix}_projection": (_call((tokens, hidden), (tokens, width)),),
        f"{prefix}_save_partial_states": (_call(
            (tokens, width), None, state_width=width,
            kv_shape=(tokens, overlap * head), score_shape=(tokens, overlap * head),
        ),),
    }
    if ratio != 128 or any(writes):
        operators[f"{prefix}_fused_compress_store"] = (_call(
            (tokens, width), None, new_compressed_rows=writes,
        ),)
    return operators
