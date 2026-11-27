"""Role-local model scoping for the PD-AF post-attention norm."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.types import ClusterType, MeasurementType


def _manager() -> ExecutionTimePredictionModelManager:
    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    manager._active_measurement_type = MeasurementType.KERNEL_ONLY
    manager._measurement_family_name = lambda _measurement_type: "kernel_only"
    manager._cluster_scoped_models = {}
    manager._block_size = 16
    return manager


def test_decode_attn_trains_role_local_h200_norm_and_overrides_ffn_view(
    tmp_path,
) -> None:
    manager = _manager()
    model_config = BaseModelConfig.create_from_name("Qwen3-4B")
    replica_config = SimpleNamespace(
        device="h200",
        model_name="Qwen3-4B",
        model_config=model_config,
        num_pipeline_stages=1,
        network_device="h200_dgx",
    )
    linear_ops_file = tmp_path / "linear_op_kernel_only.csv"
    linear_ops_file.write_text("placeholder\n", encoding="utf-8")

    load_calls: list[tuple[str, int]] = []
    manager._get_linear_op_tp_key = lambda *_args, **_kwargs: 2
    layernorm_df = pd.DataFrame(
        {
            "num_tokens": [8],
            "time_stats.post_attention_layernorm.median": [0.001632],
        }
    )
    manager._load_linear_op_df = lambda path, tp: (
        load_calls.append((path, tp)) or layernorm_df
    )
    sentinel = SimpleNamespace(name="h200-role-local-norm")
    train_calls: list[dict] = []
    manager._train_single_model = lambda **kwargs: (
        train_calls.append(kwargs) or sentinel
    )

    result = manager._train_post_attention_layernorm_for_cluster(
        cluster_type=ClusterType.DECODE_ATTN,
        replica_config=replica_config,
        execution_time_predictor_config=SimpleNamespace(),
        linear_ops_file=str(linear_ops_file),
        trained_model_signatures=set(),
    )

    assert result is sentinel
    assert load_calls == [(str(linear_ops_file), 2)]
    assert len(train_calls) == 1
    train_call = train_calls[0]
    assert train_call["model_name"] == "post_attention_layernorm"
    assert train_call["target_col"] == (
        "time_stats.post_attention_layernorm.median"
    )
    assert train_call["feature_cols"] == ["num_tokens"]
    assert train_call["register_model"] is False
    assert train_call["training_context"]["device"] == "h200"
    assert train_call["training_context"]["tensor_parallel_size"] == 2
    assert train_call["df"]["time_stats.post_attention_layernorm.median"].iloc[
        0
    ] == 0.001632
    assert manager._cluster_scoped_models[
        ("kernel_only", ClusterType.DECODE_ATTN, "post_attention_layernorm")
    ] is sentinel

    # The bare registry is the HPP2 FFN model; the DECODE_ATTN view must not
    # select it, while the DECODE_FFN view may keep it.
    hpp2_norm = SimpleNamespace(name="hpp2-ffn-norm")
    manager._trained_models_kernel_only = {
        "post_attention_layernorm": hpp2_norm
    }
    manager._trained_models_kernel_only_by_contract = {}
    manager._models_by_precision_kernel_only = {}
    manager._models_by_precision_kernel_only_by_contract = {}
    manager._cpu_overhead_models = {}
    manager._is_ffn_typed_model_for_cluster = lambda *_args: False
    manager._resolve_cluster_model_contract = lambda *_args: None
    manager._get_family_model = (
        lambda _family, model_name, requested_identity=None: (
            hpp2_norm if model_name == "post_attention_layernorm" else None
        )
    )

    attn_view = manager._models_view_for_family(
        "kernel_only", ClusterType.DECODE_ATTN
    )
    ffn_view = manager._models_view_for_family(
        "kernel_only", ClusterType.DECODE_FFN
    )

    assert attn_view["post_attention_layernorm"] is sentinel
    assert ffn_view["post_attention_layernorm"] is hpp2_norm