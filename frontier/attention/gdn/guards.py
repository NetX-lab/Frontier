"""Fail-fast runtime boundaries for the initial GDN simulator integration."""

from __future__ import annotations

from typing import Any


def model_has_gdn(model_config: Any) -> bool:
    getter = getattr(model_config, "get_num_gdn_layers", None)
    return callable(getter) and int(getter()) > 0


def validate_gdn_runtime_support(
    model_config: Any,
    *,
    prefix_cache_enabled: bool = False,
    pd_enabled: bool = False,
    speculative_enabled: bool = False,
    preemption_requires_state_drop: bool = False,
    num_pipeline_stages: int = 1,
    moe_expert_parallel_size: int = 1,
    attn_dp: int = 1,
    cross_node: bool = False,
    waiting: bool = False,
) -> None:
    """Validate supported GDN execution boundaries before state mutation."""

    if not model_has_gdn(model_config):
        return
    if prefix_cache_enabled:
        raise ValueError("GDN prefix caching is unsupported")
    if pd_enabled:
        raise ValueError("GDN P-to-D state transfer is unsupported")
    if speculative_enabled:
        raise ValueError("GDN speculative decoding is unsupported")
    if preemption_requires_state_drop:
        raise ValueError("GDN preemption requiring state recovery is unsupported")
    if int(num_pipeline_stages) > 1:
        raise ValueError("GDN pipeline parallelism above PP1 is unsupported")
    if int(moe_expert_parallel_size) > 1:
        raise ValueError("GDN expert parallelism above EP1 is unsupported")
    if int(attn_dp) > 1:
        raise ValueError("GDN attention data parallelism above DP1 is unsupported")
    if cross_node:
        raise ValueError("GDN cross-node execution is unsupported")
    # Waiting retains the existing slot and is not a state-dropping preemption.
    if waiting:
        return
