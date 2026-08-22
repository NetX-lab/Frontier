from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.types import ClusterType


def test_mixed_layer_moe_materializes_moe_and_dense_mlp_predictors(
    monkeypatch, tmp_path
) -> None:
    """Mixed-layer MoE training must expose both runtime FFN branches."""
    import frontier.execution_time_predictor.shared_prediction_model_manager as manager_module

    linear_file = tmp_path / "linear_op.csv"
    moe_file = tmp_path / "moe.csv"
    linear_file.write_text("placeholder\n", encoding="utf-8")
    moe_file.write_text("placeholder\n", encoding="utf-8")

    model_config = SimpleNamespace(
        is_moe=True,
        num_layers=3,
        get_num_moe_layers=lambda: 2,
        get_model_arch=lambda: "unit_mixed_moe",
        supports_share_expert=lambda: False,
        use_qk_norm=False,
        num_kv_heads=8,
    )
    replica_config = SimpleNamespace(
        device="h800",
        model_name="unit_mixed_moe",
        model_config=model_config,
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
        speculative_decoding_config=None,
    )

    linear_df = pd.DataFrame(
        {
            "num_tokens": [1, 2],
            "time_stats.mlp_up_proj.median": [1.0, 2.0],
            "time_stats.mlp_down_proj.median": [1.0, 2.0],
            "time_stats.mlp_act.median": [1.0, 2.0],
            "time_stats.post_attention_layernorm.median": [1.0, 2.0],
        }
    )
    moe_df = pd.DataFrame(
        {
            "num_tokens": [1, 2],
            "time_stats.moe_grouped_gemm.median": [3.0, 4.0],
        }
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)
    manager._active_measurement_type = SimpleNamespace(value="cuda_event")
    manager._measurement_family_name = lambda _measurement_type: "cuda_event"
    manager._validate_moe_dataset_contract = lambda *args, **kwargs: None
    manager._load_linear_op_df = lambda _path, tp: linear_df.assign(_loaded_tp=tp)
    manager._load_moe_df = lambda *args, **kwargs: moe_df

    trained = []

    def _train_single_model(**kwargs):
        trained.append(kwargs)
        return kwargs["model_name"]

    manager._train_single_model = _train_single_model
    monkeypatch.setattr(
        manager_module,
        "_get_moe_family_model_names",
        lambda: ["moe_grouped_gemm"],
    )

    models = manager._train_ffn_models_for_cluster(
        ClusterType.PREFILL,
        replica_config,
        execution_time_predictor_config=SimpleNamespace(),
        linear_ops_file=str(linear_file),
        moe_file=str(moe_file),
        is_moe_model=True,
        trained_model_signatures=set(),
    )

    assert set(models) >= {
        "moe_grouped_gemm",
        "mlp_up_proj",
        "mlp_down_proj",
        "mlp_act",
        "post_attention_layernorm",
    }
    assert {
        row["model_name"]: row["training_context"]["tensor_parallel_size"]
        for row in trained
    }["moe_grouped_gemm"] == 1
    assert {
        row["model_name"]: row["training_context"]["tensor_parallel_size"]
        for row in trained
    }["mlp_up_proj"] == 8
    assert {
        row["model_name"]: row["training_context"]["is_moe_model"]
        for row in trained
    }["mlp_up_proj"] is False
    assert {
        row["model_name"]: row.get("persist_exact_lookup", True)
        for row in trained
    }["moe_grouped_gemm"] is True


def test_mixed_layer_predicate_preserves_pure_moe_and_dense_contracts() -> None:
    manager = object.__new__(ExecutionTimePredictionModelManager)
    mixed = SimpleNamespace(is_moe=True, num_layers=3, get_num_moe_layers=lambda: 2)
    pure_moe = SimpleNamespace(is_moe=True, num_layers=3, get_num_moe_layers=lambda: 3)
    dense = SimpleNamespace(is_moe=False, num_layers=3, get_num_moe_layers=lambda: 0)

    assert manager._is_mixed_layer_moe_model(mixed, True)
    assert not manager._is_mixed_layer_moe_model(pure_moe, True)
    assert not manager._is_mixed_layer_moe_model(dense, False)


@pytest.mark.parametrize(
    ("cluster_type", "is_moe_model", "expected_tp"),
    [
        (ClusterType.DECODE_FFN, False, 8),
        (ClusterType.DECODE_FFN, True, 8),
        (ClusterType.PREFILL, False, 1),
        (ClusterType.PREFILL, True, 8),
        (ClusterType.MONOLITHIC, False, 1),
        (ClusterType.MONOLITHIC, True, 8),
    ],
)
def test_ffn_tp_key_uses_the_active_ffn_domain(
    cluster_type: ClusterType,
    is_moe_model: bool,
    expected_tp: int,
) -> None:
    manager = object.__new__(ExecutionTimePredictionModelManager)
    replica_config = SimpleNamespace(
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=8,
    )

    assert (
        manager._get_ffn_tp_key(cluster_type, replica_config, is_moe_model)
        == expected_tp
    )


def test_mixed_layer_dense_training_does_not_reorder_legacy_share_expert_models(
    monkeypatch, tmp_path
) -> None:
    """Mixed-layer additions must not perturb legacy RF training order."""
    import frontier.execution_time_predictor.shared_prediction_model_manager as manager_module

    linear_file = tmp_path / "linear_op.csv"
    moe_file = tmp_path / "moe.csv"
    linear_file.write_text("placeholder\n", encoding="utf-8")
    moe_file.write_text("placeholder\n", encoding="utf-8")

    model_config = SimpleNamespace(
        is_moe=True,
        num_layers=3,
        get_num_moe_layers=lambda: 2,
        get_model_arch=lambda: "unit_mixed_moe",
        supports_share_expert=lambda: True,
        use_qk_norm=False,
        num_kv_heads=8,
    )
    replica_config = SimpleNamespace(
        device="h800",
        model_name="unit_mixed_moe",
        model_config=model_config,
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
        speculative_decoding_config=None,
    )
    linear_df = pd.DataFrame(
        {
            "num_tokens": [1, 2],
            "time_stats.mlp_up_proj.median": [1.0, 2.0],
            "time_stats.mlp_down_proj.median": [1.0, 2.0],
            "time_stats.mlp_act.median": [1.0, 2.0],
            "time_stats.post_attention_layernorm.median": [1.0, 2.0],
            "time_stats.share_expert_up_proj.median": [1.0, 2.0],
        }
    )
    moe_df = pd.DataFrame(
        {"num_tokens": [1, 2], "time_stats.moe_grouped_gemm.median": [3.0, 4.0]}
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)
    manager._active_measurement_type = SimpleNamespace(value="cuda_event")
    manager._measurement_family_name = lambda _measurement_type: "cuda_event"
    manager._validate_moe_dataset_contract = lambda *args, **kwargs: None
    manager._load_linear_op_df = lambda _path, _tp: linear_df
    manager._load_moe_df = lambda *args, **kwargs: moe_df
    manager._get_linear_op_tp_key = lambda *args, **kwargs: 8
    trained = []
    manager._train_single_model = lambda **kwargs: trained.append(kwargs) or kwargs["model_name"]
    monkeypatch.setattr(manager_module, "_get_moe_family_model_names", lambda: ["moe_grouped_gemm"])
    monkeypatch.setattr(
        manager_module,
        "get_family_profiling_names",
        lambda family: ["share_expert_up_proj"]
        if family is manager_module.SHARE_EXPERT_FAMILY
        else ["moe_grouped_gemm"],
    )

    manager._train_ffn_models_for_cluster(
        ClusterType.PREFILL,
        replica_config,
        execution_time_predictor_config=SimpleNamespace(),
        linear_ops_file=str(linear_file),
        moe_file=str(moe_file),
        is_moe_model=True,
        trained_model_signatures=set(),
    )

    names = [row["model_name"] for row in trained]
    assert names.index("share_expert_up_proj") < names.index("mlp_up_proj")


def test_random_forest_estimator_uses_fixed_predictor_seed() -> None:
    """The shared manager must use the deterministic predictor RNG contract."""
    from frontier.types import ExecutionTimePredictorType

    manager = object.__new__(ExecutionTimePredictionModelManager)
    config = SimpleNamespace(
        get_type=lambda: ExecutionTimePredictorType.RANDOM_FORREST,
        num_estimators=[500],
        max_depth=[32],
        min_samples_split=[2],
    )
    estimator, _ = manager._create_estimator_and_params(config)
    assert estimator.random_state == 0
