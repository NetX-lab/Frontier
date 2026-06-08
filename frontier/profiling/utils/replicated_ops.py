"""Shared utility for splitting profiling results into sharded and replicated rows."""

from typing import Dict, List, Optional, Set, Tuple


def split_replicated_result(
    result: Dict,
    replicated_op_names: Set[str],
    unpadded_n_embd: Optional[int] = None,
    unpadded_n_expanded_embd: Optional[int] = None,
) -> Tuple[Dict, Dict]:
    """Split a profiling result into sharded-only and replicated-only rows.

    The sharded row keeps the original TP size and padded dimensions.
    The replicated row forces num_tensor_parallel_workers=1 and uses
    unpadded dimensions (if provided).

    Args:
        result: Original profiling result dict with "time_stats" key.
        replicated_op_names: Set of op names considered replicated.
        unpadded_n_embd: Original embedding dim (for linear_op dimension correction).
        unpadded_n_expanded_embd: Original MLP hidden dim (for linear_op dimension correction).

    Returns:
        (sharded_row, replicated_row) — both are new dicts (shallow copy).
    """
    time_stats = result["time_stats"]

    replicated_stats = {k: v for k, v in time_stats.items() if k in replicated_op_names}
    sharded_stats = {k: v for k, v in time_stats.items() if k not in replicated_op_names}

    sharded_row = {**result, "time_stats": sharded_stats}

    replicated_row = {**result, "time_stats": replicated_stats}
    replicated_row["num_tensor_parallel_workers"] = 1
    if unpadded_n_embd is not None:
        replicated_row["padded_n_embd"] = unpadded_n_embd
    if unpadded_n_expanded_embd is not None:
        replicated_row["padded_n_expanded_embd"] = unpadded_n_expanded_embd

    return sharded_row, replicated_row


def deduplicate_tp1_rows(
    results: List[Dict],
    tp1_key_fields: Tuple[str, ...] = ("num_tokens",),
) -> List[Dict]:
    """Remove duplicate TP=1 replicated rows, keeping first occurrence per key.

    This function operates on in-memory results from a single profile_model() call.
    quant_signature is NOT available at this stage (added post-hoc to DataFrame).

    Args:
        results: List of result dicts (mix of sharded and replicated rows).
        tp1_key_fields: Fields that form the dedup key for TP=1 rows.
            Linear op: ("num_tokens", "model_arch") — model_arch in result dict.
            MoE: ("num_tokens",) — model_arch not in result dict; EP excluded
            because replicated ops are EP-agnostic.

    Returns:
        Deduplicated list.
    """
    seen_tp1_keys: set = set()
    deduped: List[Dict] = []
    for r in results:
        if r["num_tensor_parallel_workers"] == 1:
            key = tuple(r.get(f) for f in tp1_key_fields)
            if key in seen_tp1_keys:
                continue
            seen_tp1_keys.add(key)
        deduped.append(r)
    return deduped
