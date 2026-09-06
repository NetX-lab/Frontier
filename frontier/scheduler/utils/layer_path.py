"""Shared layer-path predicates for MoE cluster schedulers."""

from typing import Any, Iterable


def uses_shared_layer_path(
    *,
    cluster_type: Any,
    allowed_clusters: Iterable[Any],
    model_config: Any,
    predictor: Any,
    layer_id: int,
    routing_attribute: str,
    require_moe_layer: bool,
) -> bool:
    """Return whether a cluster uses the canonical per-layer MoE path."""

    if cluster_type not in tuple(allowed_clusters):
        return False
    if model_config is None or not getattr(model_config, "is_moe", False):
        return False
    if not isinstance(layer_id, int) or layer_id < 0:
        raise ValueError("layer_id must be an exact non-negative int")
    is_moe_layer = bool(model_config.is_moe_layer(layer_id))
    if require_moe_layer and not is_moe_layer:
        return False
    if is_moe_layer and getattr(predictor, routing_attribute, None) is None:
        raise ValueError(f"Missing {routing_attribute} for MoE layer path")
    return True
