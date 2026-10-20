"""Source-defined sparse-MLA workload features shared by fitting and lookup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


# FlashAttnMLASparseMetadataBuilder in vLLM 0.28 (2cf0a6915ce5).
DSA_DECODE_QUERY_LIMIT = 256
DSA_ROW_SCHEMA_VERSION = "dsa_attention.v1"
_DECODE_QUERY_LIMIT_BY_HEADS = {16: 128, 32: 128, 64: 256, 128: 256}
DSA_FEATURE_COLUMNS = {
    "attn_dsa_indexer": {
        "decode": ("num_query_tokens", "sum_key_len"),
        "prefill": ("num_query_tokens", "sum_causal_key_pairs"),
    },
    "attn_dsa_mla": {
        "decode": ("num_query_tokens", "sum_selected_key_len"),
        "prefill": ("num_query_tokens", "sum_selected_causal_key_pairs"),
    },
}
DSA_WORKLOAD_PATHS = {"decode": "decode", "extend": "decode", "prefill": "prefill"}
DSA_PROFILING_COLUMNS = (
    "prefill_seq_lens", "prefill_kv_cache_sizes", "decode_kv_cache_sizes",
    "index_topk", "max_key_len", "attention_variant", "is_prefill",
)


class DSACoverageError(ValueError):
    """The requested workload is outside the measured DSA domain."""


def dsa_decode_query_limit(num_query_heads_per_rank: int) -> int:
    return _DECODE_QUERY_LIMIT_BY_HEADS.get(num_query_heads_per_rank, DSA_DECODE_QUERY_LIMIT)


@dataclass(frozen=True)
class DSABatchFeatures:
    path: str
    workload: str
    values: dict[str, int]


def dsa_predictor_feature_columns() -> dict[str, tuple[str, ...]]:
    """Return the union of each operator's phase-qualified feature columns."""
    return {
        operator: tuple(dict.fromkeys(name for phase in paths.values() for name in phase))
        for operator, paths in DSA_FEATURE_COLUMNS.items()
    }


def build_dsa_features(
    query_lengths: Sequence[int], context_lengths: Sequence[int], index_topk: int,
    *, decode_query_limit: int = DSA_DECODE_QUERY_LIMIT,
) -> DSABatchFeatures:
    """Use cached context plus scheduled queries, exactly as dsa_rows does.

    Causal work sums the visible keys of every query, clipping each query's
    selected keys separately rather than clipping a sequence's final length.
    """
    if not query_lengths or len(query_lengths) != len(context_lengths):
        raise ValueError("DSA needs matching non-empty query and context lengths")
    if index_topk <= 0 or any(q <= 0 for q in query_lengths) or any(c < 0 for c in context_lengths):
        raise ValueError("DSA requires positive queries/top-k and non-negative contexts")
    paths = {"decode" if q <= decode_query_limit else "prefill" for q in query_lengths}
    if len(paths) != 1:
        raise DSACoverageError("DSA coverage missing for mixed decode/prefill kernel paths")
    path = paths.pop()
    workload = "prefill" if path == "prefill" else ("decode" if max(query_lengths) == 1 else "extend")
    key_lengths = [c + q for q, c in zip(query_lengths, context_lengths)]
    selected_pairs = 0
    for q, context in zip(query_lengths, context_lengths):
        unclipped = min(q, max(0, index_topk - context))
        selected_pairs += (
            unclipped * context + unclipped * (unclipped + 1) // 2
            + (q - unclipped) * index_topk
        )
    return DSABatchFeatures(path, workload, {
        "num_query_tokens": sum(query_lengths),
        "sum_key_len": sum(key_lengths),
        "sum_selected_key_len": sum(min(k, index_topk) for k in key_lengths),
        "sum_causal_key_pairs": sum(q * c + q * (q + 1) // 2 for q, c in zip(query_lengths, context_lengths)),
        "sum_selected_causal_key_pairs": selected_pairs,
        "num_sequences": len(query_lengths),
        "min_query_len": min(query_lengths), "max_query_len": max(query_lengths),
        "min_context_len": min(context_lengths), "max_context_len": max(context_lengths),
        "max_key_len": max(key_lengths),
    })
