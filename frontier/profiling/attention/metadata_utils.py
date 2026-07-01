"""Utilities for attention profiling metadata post-processing."""

from __future__ import annotations

import pandas as pd

from frontier.attention.string_coercion import coerce_truthy_bool


def add_chunked_prefill_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Attach chunked-prefill metadata columns without changing core features."""
    required_cols = {"prefill_chunk_size", "kv_cache_size", "is_prefill"}
    if not required_cols.issubset(df.columns):
        return df

    normalized_is_prefill = coerce_truthy_bool(df["is_prefill"])
    kv_cache_size = pd.to_numeric(df["kv_cache_size"], errors="coerce").fillna(0)
    prefill_chunk_size = pd.to_numeric(
        df["prefill_chunk_size"], errors="coerce"
    ).fillna(0)

    chunk_start_token = kv_cache_size.where(normalized_is_prefill, 0)
    chunk_end_token = (kv_cache_size + prefill_chunk_size).where(
        normalized_is_prefill, 0
    )

    result = df.copy()
    result["is_chunked_prefill_sample"] = normalized_is_prefill & (kv_cache_size > 0)
    result["chunk_start_token"] = chunk_start_token.astype(int)
    result["chunk_end_token"] = chunk_end_token.astype(int)

    computed_total_prefill_tokens = chunk_end_token.astype(int)
    if "total_prefill_tokens" not in result.columns:
        result["total_prefill_tokens"] = computed_total_prefill_tokens
    elif "is_true_mixed_batch" in result.columns:
        is_true_mixed_batch = coerce_truthy_bool(result["is_true_mixed_batch"])
        existing_total_prefill_tokens = pd.to_numeric(
            result["total_prefill_tokens"], errors="coerce"
        )
        # Preserve explicit true-mixed prefill token counts while using chunk metadata
        # for all non-true-mixed rows.
        result["total_prefill_tokens"] = (
            existing_total_prefill_tokens
            .where(is_true_mixed_batch, computed_total_prefill_tokens)
            .fillna(computed_total_prefill_tokens)
            .astype(int)
        )
    else:
        result["total_prefill_tokens"] = pd.to_numeric(
            result["total_prefill_tokens"], errors="coerce"
        ).fillna(computed_total_prefill_tokens).astype(int)
    return result
