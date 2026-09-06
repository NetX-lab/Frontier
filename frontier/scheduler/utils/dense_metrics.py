"""Dense-layer reference timing helpers for mixed-MoE metrics."""

from __future__ import annotations

from typing import Any, Optional

from frontier.scheduler.utils.execution_time_metrics import (
    build_single_layer_metrics_execution_time,
)


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


def complete_dense_layer(
    scheduler: Any,
    *,
    time: float,
    replica_id: int,
    stage_id: int,
    batch: Any,
    layer_id: int,
    phase: str,
    metrics_store: Any,
) -> list:
    """Advance a dense layer through the scheduler's existing phase handler."""
    if phase == "prefill":
        return scheduler.on_prefill_sync_collective(
            time,
            replica_id,
            stage_id,
            int(batch.global_id),
            "post_moe",
            layer_id,
            metrics_store,
            direct_batch=batch,
        )
    if phase == "decode":
        return scheduler.on_decode_sync_collective(
            time,
            replica_id,
            stage_id,
            scheduler._get_decode_sync_wait_key(batch),
            "post_moe",
            layer_id,
            metrics_store,
            direct_batch=batch,
        )
    raise ValueError(f"Unsupported dense layer completion phase: {phase!r}")


def build_prefill_metrics_execution_time(
    *,
    original_execution_time: Any,
    sample_batch: Any,
    predictor: Any,
    stage_id: int,
    cluster_type: Any,
    model_config: Any,
) -> Any:
    """Build prefill metrics payload and annotate mixed-MoE dense timing."""
    corrected = build_single_layer_metrics_execution_time(original_execution_time)
    dense_layer_id = first_dense_layer_id(model_config)
    if dense_layer_id is None:
        return corrected
    dense_reference = predict_dense_reference(
        predictor=predictor,
        batch=sample_batch,
        stage_id=stage_id,
        cluster_type=cluster_type,
        model_config=model_config,
    )
    if dense_reference is None:
        return corrected
    corrected._trace_dense_mlp_layer_up_proj_execution_time = (
        dense_reference._mlp_layer_up_proj_execution_time
    )
    corrected._trace_dense_mlp_layer_act_execution_time = (
        dense_reference._mlp_layer_act_execution_time
    )
    corrected._trace_dense_mlp_layer_down_proj_execution_time = (
        dense_reference._mlp_layer_down_proj_execution_time
    )
    corrected._trace_dense_layer_id = dense_layer_id
    return corrected
