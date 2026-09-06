"""Dense-layer reference timing helpers for mixed-MoE metrics."""

from __future__ import annotations

from typing import Any, Optional


def first_dense_layer_id(model_config: Any) -> Optional[int]:
    """Return the first dense FFN layer in a mixed-MoE model."""

    if model_config is None or not getattr(model_config, "is_moe", False):
        return None
    get_moe_layer_ids = getattr(model_config, "get_moe_layer_ids", None)
    if not callable(get_moe_layer_ids) or not hasattr(model_config, "num_layers"):
        return None
    moe_layer_ids = set(get_moe_layer_ids())
    num_layers = int(model_config.num_layers)
    if not moe_layer_ids or len(moe_layer_ids) >= num_layers:
        return None
    return next(
        (layer_id for layer_id in range(num_layers) if layer_id not in moe_layer_ids),
        None,
    )


def predict_dense_reference(
    *,
    predictor: Any,
    batch: Any,
    stage_id: int,
    cluster_type: Any,
    model_config: Any,
) -> Any:
    """Predict and validate one dense reference layer for trace augmentation."""

    layer_id = first_dense_layer_id(model_config)
    if layer_id is None:
        return None
    execution_time = predictor.predict_stage_execution_time(
        batch,
        stage_id,
        cluster_type=cluster_type,
        num_layers=1,
        layer_id=layer_id,
    )
    if execution_time._is_moe:
        raise ValueError(
            f"Expected dense execution for layer_id={layer_id}, "
            "but predictor returned is_moe=True"
        )
    components = (
        execution_time._mlp_layer_up_proj_execution_time,
        execution_time._mlp_layer_act_execution_time,
        execution_time._mlp_layer_down_proj_execution_time,
    )
    if any(component <= 0.0 for component in components):
        raise ValueError(
            "Dense reference execution_time must provide positive "
            "mlp_up_proj/mlp_act/mlp_down_proj components"
        )
    return execution_time
