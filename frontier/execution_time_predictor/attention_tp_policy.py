from __future__ import annotations

from typing import Optional, Set

from frontier.types import ClusterType

ATTENTION_NON_LINEAR_OPS = frozenset(
    {
        "attn_prefill",
        "attn_decode",
        "attn_kv_cache_save",
    }
)

ATTENTION_LINEAR_OPS = frozenset(
    {
        "attn_pre_proj",
        "attn_post_proj",
        "attn_rope",
    }
)


def is_attention_tp_supported(*, requested_tp_size: int, num_kv_heads: int) -> bool:
    """Return whether attention TP sharding follows vLLM KV-head semantics.

    Rules (same as runtime semantics):
    - When ``num_kv_heads >= tp``: KV heads are partitioned, requiring divisibility.
    - When ``num_kv_heads < tp``: KV heads are replicated, requiring ``tp % num_kv_heads == 0``.
    """
    if requested_tp_size <= 0:
        raise ValueError(
            f"requested_tp_size must be positive, got {requested_tp_size}"
        )
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")

    if num_kv_heads >= requested_tp_size:
        return num_kv_heads % requested_tp_size == 0
    return requested_tp_size % num_kv_heads == 0


def resolve_effective_attention_tp_size(
    *,
    op_name: str,
    requested_tp_size: int,
    num_kv_heads: int,
    cluster_type: Optional[ClusterType],
    warning_cache: Optional[Set[str]] = None,
    include_linear_ops: bool = False,
) -> int:
    """Resolve attention TP size with fail-fast validation.

    Frontier uses a single representative TP lane, but TP degree must still obey
    KV-head sharding/replication constraints. Unsupported TP settings are surfaced
    as explicit errors instead of silently falling back to TP=1.
    """
    if requested_tp_size == 1:
        return 1

    if is_attention_tp_supported(
        requested_tp_size=requested_tp_size,
        num_kv_heads=num_kv_heads,
    ):
        return requested_tp_size

    if op_name in ATTENTION_NON_LINEAR_OPS:
        should_validate = True
    elif include_linear_ops and op_name in ATTENTION_LINEAR_OPS:
        should_validate = True
    else:
        should_validate = False

    if not should_validate:
        return requested_tp_size

    cluster_name = cluster_type.name if isinstance(cluster_type, ClusterType) else "UNKNOWN"
    warning_key = (
        f"{cluster_name}:{op_name}:"
        f"tp{requested_tp_size}:kv{num_kv_heads}:linear{int(include_linear_ops)}"
    )
    if warning_cache is not None:
        warning_cache.add(warning_key)

    raise ValueError(
        "Unsupported attention TP configuration: "
        f"operation={op_name}, cluster={cluster_name}, requested_tp={requested_tp_size}, "
        f"num_kv_heads={num_kv_heads}. "
        "Supported rules: (num_kv_heads >= tp and num_kv_heads % tp == 0) or "
        "(num_kv_heads < tp and tp % num_kv_heads == 0)."
    )
