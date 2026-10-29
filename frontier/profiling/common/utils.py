"""Utility functions for profiling."""

from typing import Optional
from dataclasses import replace

import pandas as pd
import torch

from frontier.config import BaseModelConfig, PrecisionType, get_quantization_manager


def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
) -> None:
    """Initialize model weights with random values.

    The model weights must be randomly initialized for accurate performance
    measurements. Additionally, the model weights should not cause NaNs in the
    forward pass. We empirically found that initializing the weights with
    values between -1e-3 and 1e-3 works well for most models.
    """
    for param in model.state_dict().values():
        param.data.uniform_(low, high)


def get_operation_precision(op_name: Optional[str]) -> Optional[PrecisionType]:
    if not op_name:
        return None
    quant_manager = get_quantization_manager()
    if not quant_manager.is_operation_supported(op_name):
        return None
    return quant_manager.get_precision(op_name)


def raise_if_fp8_requested(op_name: str, error_message: str) -> None:
    precision = get_operation_precision(op_name)
    if precision == PrecisionType.FP8:
        raise ImportError(error_message)


def configure_quantization_manager_for_model_name(model_name: str) -> None:
    quant_manager = get_quantization_manager()
    model_config = BaseModelConfig.create_from_name(model_name)
    if model_config.get_model_architecture_profile().profile_id == "step5":
        from frontier.attention.step5 import describe_step5_layer

        # Native module exclusions distinguish dense/shared BF16 from routed FP8.
        # The generic operation defaults do not contain those module identities.
        quantized_operations = {
            projection.operator_name
            for layer_id in range(model_config.num_layers)
            for projection in describe_step5_layer(
                model_config, layer_id, attention_tp_size=1,
                moe_tp_size=1, expert_parallel_size=1,
            ).projections
            if projection.weight_precision is PrecisionType.FP8
            and projection.operator_name in quant_manager.MODEL_CONFIG_QUANT_OPS
        }
        if model_config.quantization_config is not None:
            model_config.quantization_config = replace(
                model_config.quantization_config,
                quantized_operations=sorted(quantized_operations),
            )
    quant_manager.configure_from_model_config(model_config)


def merge_profiling_rows_on_feature_columns(*frames: pd.DataFrame) -> pd.DataFrame:
    """Merge profiling rows by feature columns without letting NaNs erase measurements.

    Profiling CSVs often append supplemental rows for the same feature tuple.
    When later rows contain partial timing columns, those NaNs must not wipe out
    previously measured values for the same tuple.
    """
    non_empty_frames = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not non_empty_frames:
        return pd.DataFrame()

    merged = pd.concat(non_empty_frames, ignore_index=True, sort=False)
    feature_cols = [col for col in merged.columns if not col.startswith("time_stats.")]
    if not feature_cols:
        return merged.drop_duplicates().reset_index(drop=True)

    merged = merged.reset_index(drop=True)
    merged["_merge_row_order"] = range(len(merged))
    collapsed_rows = []
    for _, group in merged.groupby(feature_cols, dropna=False, sort=False):
        group = group.sort_values("_merge_row_order", kind="stable")
        collapsed_rows.append(group.ffill().iloc[-1])

    result = pd.DataFrame(collapsed_rows).drop(columns="_merge_row_order", errors="ignore")
    return result.sort_values(feature_cols, kind="stable").reset_index(drop=True)
