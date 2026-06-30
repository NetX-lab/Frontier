"""Shared attention operator family registry."""

from frontier.attention.families import (
    DENSE_ATTENTION_FAMILY,
    DSA_ATTENTION_FAMILY,
    LATENT_MLA_ATTENTION_FAMILY,
    get_attention_family,
    iter_attention_families,
    iter_execution_enabled_families,
)
from frontier.attention.model_binding import (
    AttentionFamilyBinding,
    bind_attention_family,
)
from frontier.attention.memory import (
    AttentionRuntimeKVLayout,
    get_attention_runtime_kv_layout,
)
from frontier.attention.ops import AttentionRuntimeMetaContract


def __getattr__(name: str):
    if name == "get_attention_trace_op_times":
        from frontier.attention.trace_mapping import get_attention_trace_op_times

        return get_attention_trace_op_times
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AttentionFamilyBinding",
    "AttentionRuntimeKVLayout",
    "AttentionRuntimeMetaContract",
    "DENSE_ATTENTION_FAMILY",
    "DSA_ATTENTION_FAMILY",
    "LATENT_MLA_ATTENTION_FAMILY",
    "bind_attention_family",
    "get_attention_family",
    "get_attention_runtime_kv_layout",
    "get_attention_trace_op_times",
    "iter_attention_families",
    "iter_execution_enabled_families",
]
