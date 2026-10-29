"""DeepSeek-V4 replay batch features shared by training and runtime lookup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from frontier.attention.deepseek_v4 import (
    CSA_VARIANT, HCA_VARIANT, UNCOMPRESSED_VARIANT, DeepSeekV4CoverageError, _integer,
)

PHASE_DECODE = "decode"
PHASE_FIRST_TAIL = "first_tail"
PHASE_PREFILL = "prefill"
PHASE_DUMMY = "dummy"
DSV4_GEOMETRY_COLUMNS = (
    "min_query_len", "max_query_len", "min_context_len", "max_context_len", "max_key_len",
)
_ROWS = ("padded_rows",)
_CACHE_WRITE = ("padded_rows", "num_query_tokens")
_COMPRESS = ("padded_rows", "num_sequences", "num_compressed_writes")
_INDEX = ("padded_rows", "num_sequences", "sum_compressed_key_pairs")
_ATTENTION = ("padded_rows", "num_sequences", "sum_swa_key_pairs", "sum_selected_key_pairs")
# Explicit physical scope classes: AttentionOperatorRole has different semantics.
DSV4_FEATURE_COLUMNS = {
    "dsv4_fused_qkv_projection": _ROWS,
    "dsv4_fused_qkv_rmsnorm": _ROWS,
    "dsv4_q_projection": _ROWS,
    "dsv4_query_padding": _ROWS,
    "dsv4_attention_output_zero": _ROWS,
    "dsv4_output_inv_rope_quant": _ROWS,
    "dsv4_grouped_output_projection": _ROWS,
    "dsv4_output_projection": _ROWS,
    "dsv4_compressor_projection": _ROWS,
    "dsv4_index_weights_projection": _ROWS,
    "dsv4_index_compressor_projection": _ROWS,
    "dsv4_index_query_projection": _ROWS,
    "dsv4_index_fused_query_prepare": _ROWS,
    "dsv4_index_short_context_fill": _ROWS,
    "dsv4_fused_q_kv_prepare": _CACHE_WRITE,
    "dsv4_compressor_save_partial_states": _CACHE_WRITE,
    "dsv4_index_compressor_save_partial_states": _CACHE_WRITE,
    "dsv4_compressor_fused_compress_store": _COMPRESS,
    "dsv4_index_compressor_fused_compress_store": _COMPRESS,
    "dsv4_index_scores": _INDEX,
    "dsv4_index_topk": _INDEX,
    "dsv4_sparse_attention": _ATTENTION,
}


@dataclass(frozen=True)
class DeepSeekV4BatchFeatures:
    phase: str
    values: dict[str, int]


def dsv4_feature_columns(scope: str, phase: str) -> tuple[str, ...]:
    """Return the measured scope's feature schema for the batch phase."""
    columns = DSV4_FEATURE_COLUMNS[scope]
    return _ROWS if phase == PHASE_DUMMY else columns


def _floor_sum(n: int, ratio: int) -> int:
    m = n // ratio
    return ratio * m * (m - 1) // 2 + m * (n - ratio * m + 1)


def _clipped_floor_sum(n: int, ratio: int, topk: int) -> int:
    boundary = ratio * topk
    return _floor_sum(min(n, boundary), ratio) + max(0, n - boundary) * topk


def _swa_sum(n: int, window: int) -> int:
    unclipped = min(n, window)
    return unclipped * (unclipped + 1) // 2 + max(0, n - window) * window


def build_dsv4_features(
    query_lengths: Sequence[int], context_lengths: Sequence[int],
    is_decoding: Sequence[bool], *, variant: str, compress_ratio: int,
    sliding_window: int, index_topk: int, padded_rows: int,
) -> DeepSeekV4BatchFeatures:
    """Sum per-query causal work in closed form, without iterating over tokens."""
    if not query_lengths or not (len(query_lengths) == len(context_lengths) == len(is_decoding)):
        raise ValueError("V4 requires matching non-empty query, context and decoding lists")
    _integer(padded_rows, "padded_rows")
    _integer(sliding_window, "sliding_window")
    _integer(index_topk, "index_topk")
    if variant not in (CSA_VARIANT, HCA_VARIANT, UNCOMPRESSED_VARIANT):
        raise ValueError(f"Unknown V4 attention variant: {variant}")
    compressed = variant in (CSA_VARIANT, HCA_VARIANT)
    if compressed:
        _integer(compress_ratio, "compress_ratio", minimum=2)
    for q, c, decoding in zip(query_lengths, context_lengths, is_decoding):
        _integer(q, "query length")
        _integer(c, "context length", minimum=0)
        if type(decoding) is not bool:
            raise ValueError("V4 is_decoding entries must be bool")
        if decoding and q != 1:
            raise ValueError("V4 ordinary decode requires one query row")
    if all(is_decoding):
        phase = PHASE_DECODE
    elif all(q == 1 for q in query_lengths):
        phase = PHASE_FIRST_TAIL
    elif not any(is_decoding) and all(q > 1 for q in query_lengths):
        phase = PHASE_PREFILL
    else:
        raise DeepSeekV4CoverageError(
            f"V4 coverage missing for query/decoding mix: {list(zip(query_lengths, is_decoding))}"
        )
    pairs = tuple(zip(query_lengths, context_lengths))
    compressed_pairs = sum(_floor_sum(c + q, compress_ratio) - _floor_sum(c, compress_ratio)
                           for q, c in pairs) if compressed else 0
    selected_pairs = (sum(_clipped_floor_sum(c + q, compress_ratio, index_topk)
                          - _clipped_floor_sum(c, compress_ratio, index_topk) for q, c in pairs)
                      if variant == CSA_VARIANT else compressed_pairs)
    return DeepSeekV4BatchFeatures(phase, {
        "num_sequences": len(query_lengths), "num_query_tokens": sum(query_lengths),
        "padded_rows": padded_rows,
        "sum_swa_key_pairs": sum(_swa_sum(c + q, sliding_window) - _swa_sum(c, sliding_window)
                                 for q, c in pairs),
        "sum_compressed_key_pairs": compressed_pairs,
        "sum_selected_key_pairs": selected_pairs,
        "num_compressed_writes": sum((c + q) // compress_ratio - c // compress_ratio
                                     for q, c in pairs) if compressed else 0,
        "min_query_len": min(query_lengths), "max_query_len": max(query_lengths),
        "min_context_len": min(context_lengths), "max_context_len": max(context_lengths),
        "max_key_len": max(c + q for q, c in pairs),
    })


def build_dsv4_dummy_features(padded_rows: int) -> DeepSeekV4BatchFeatures:
    """Native dummy calls have dispatch rows and no request geometry."""
    return DeepSeekV4BatchFeatures(PHASE_DUMMY, {"padded_rows": _integer(padded_rows, "padded_rows")})
