from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from frontier.attention.families import DENSE_ATTENTION_FAMILY
from frontier.attention.ops import AttentionOperatorRole
from frontier.config import ReplicaConfig
from frontier.execution_time_predictor import shared_prediction_model_manager
from frontier.execution_time_predictor import sklearn_execution_time_predictor
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.types import ClusterType, MeasurementType


def _dense_model_config() -> SimpleNamespace:
    return SimpleNamespace(
        use_mla=False,
        num_q_heads=32,
        num_kv_heads=8,
        is_step2_mini=lambda: False,
        is_moe=False,
        supports_share_expert=lambda: False,
        uses_fused_add_norm=False,
    )


def _make_manager_without_init() -> ExecutionTimePredictionModelManager:
    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    manager._active_measurement_type = MeasurementType.CUDA_EVENT
    manager._attention_tp_warning_cache = set()
    return manager


def _make_dense_replica_config() -> ReplicaConfig:
    return ReplicaConfig(
        cluster_prefix="decode",
        model_name="llama3.3-70b",
        device="h800",
        network_device="h800_dgx",
        attn_tensor_parallel_size=1,
        attn_data_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
    )


def test_shared_manager_trains_eager_attention_decode_for_pd_decode_cluster(
    tmp_path,
) -> None:
    manager = _make_manager_without_init()
    replica_config = _make_dense_replica_config()

    linear_ops_file = tmp_path / "linear_op.csv"
    attn_file = tmp_path / "attention.csv"
    linear_ops_file.write_text("", encoding="utf-8")
    attn_file.write_text("", encoding="utf-8")

    manager._load_linear_op_df = lambda *_args, **_kwargs: pd.DataFrame(  # type: ignore[attr-defined]
        {
            "num_tokens": [1, 2],
            "time_stats.input_layernorm.median": [0.1, 0.2],
            "time_stats.attn_pre_proj.median": [0.1, 0.2],
            "time_stats.attn_post_proj.median": [0.1, 0.2],
            "time_stats.attn_rope.median": [0.1, 0.2],
        }
    )
    manager._load_attention_df = lambda *_args, **_kwargs: pd.DataFrame(  # type: ignore[attr-defined]
        {
            "is_decode": [False, True],
            "is_mixed_batch": [False, False],
            "is_true_mixed_batch": [False, False],
            "num_tokens": [16, 2],
            "total_tokens": [16, 2],
            "batch_size": [1, 2],
            "kv_cache_size": [0, 256],
            "prefill_chunk_size": [16, 0],
            "time_stats.attn_kv_cache_save.median": [0.1, 0.2],
            "time_stats.attn_prefill.median": [0.3, 0.0],
            "time_stats.attn_decode.median": [0.0, 0.4],
        }
    )
    manager._get_attention_df_with_derived_features = (  # type: ignore[attr-defined]
        lambda df: df.assign(
            prefill_chunk_size_squared=df["prefill_chunk_size"] ** 2
        )
    )

    trained_ops: list[str] = []

    def _fake_train_single_model(*, model_name: str, **_kwargs):
        trained_ops.append(model_name)
        return SimpleNamespace()

    manager._train_single_model = _fake_train_single_model  # type: ignore[attr-defined]

    manager._train_attn_models_for_cluster(  # type: ignore[attr-defined]
        cluster_type=ClusterType.DECODE,
        replica_config=replica_config,
        execution_time_predictor_config=SimpleNamespace(),
        replica_scheduler_config=SimpleNamespace(block_size=16),
        linear_ops_file=str(linear_ops_file),
        attn_file=str(attn_file),
        trained_model_signatures=set(),
    )

    assert "attn_decode" in trained_ops


def test_shared_manager_dense_physical_attention_models_follow_family_mapping(
    tmp_path,
    monkeypatch,
) -> None:
    manager = _make_manager_without_init()
    replica_config = _make_dense_replica_config()

    linear_ops_file = tmp_path / "linear_op.csv"
    attn_file = tmp_path / "attention.csv"
    linear_ops_file.write_text("", encoding="utf-8")
    attn_file.write_text("", encoding="utf-8")

    manager._load_linear_op_df = lambda *_args, **_kwargs: pd.DataFrame(  # type: ignore[attr-defined]
        {
            "num_tokens": [1],
            "time_stats.input_layernorm.median": [0.1],
            "time_stats.attn_pre_proj.median": [0.1],
            "time_stats.attn_post_proj.median": [0.1],
            "time_stats.attn_rope.median": [0.1],
        }
    )
    manager._load_attention_df = lambda *_args, **_kwargs: pd.DataFrame(  # type: ignore[attr-defined]
        {
            "is_decode": [False, True],
            "is_mixed_batch": [False, False],
            "is_true_mixed_batch": [False, False],
            "total_tokens": [16, 2],
            "batch_size": [1, 2],
            "kv_cache_size": [0, 256],
            "prefill_chunk_size": [16, 0],
            "prefill_chunk_size_squared": [256, 0],
            "catalog_kv_feature": [10, 20],
            "catalog_prefill_feature": [30, 0],
            "catalog_decode_feature": [0, 40],
            "time_stats.catalog_kv.median": [0.1, 0.2],
            "time_stats.catalog_prefill.median": [0.3, 0.0],
            "time_stats.catalog_decode.median": [0.0, 0.4],
        }
    )
    manager._get_attention_df_with_derived_features = lambda df: df  # type: ignore[attr-defined]

    monkeypatch.setattr(
        shared_prediction_model_manager,
        "get_enabled_predictor_metric_names",
        lambda family: (
            ("catalog_kv", "catalog_prefill", "catalog_decode")
            if family is DENSE_ATTENTION_FAMILY
            else ()
        ),
    )
    monkeypatch.setattr(
        shared_prediction_model_manager,
        "get_enabled_predictor_metric_name_by_role",
        lambda family, role: (
            {
                AttentionOperatorRole.CACHE_WRITE: "catalog_kv",
                AttentionOperatorRole.PREFILL_KERNEL: "catalog_prefill",
                AttentionOperatorRole.DECODE_KERNEL: "catalog_decode",
            }[role]
            if family is DENSE_ATTENTION_FAMILY
            else ""
        ),
    )
    monkeypatch.setattr(
        shared_prediction_model_manager,
        "get_enabled_predictor_median_columns",
        lambda family: (
            (
                "time_stats.catalog_kv.median",
                "time_stats.catalog_prefill.median",
                "time_stats.catalog_decode.median",
            )
            if family is DENSE_ATTENTION_FAMILY
            else ()
        ),
    )
    monkeypatch.setattr(
        shared_prediction_model_manager,
        "get_enabled_shared_predictor_feature_columns",
        lambda family: (
            {
                "catalog_kv": ("catalog_kv_feature",),
                "catalog_prefill": ("catalog_prefill_feature",),
                "catalog_decode": ("catalog_decode_feature",),
            }
            if family is DENSE_ATTENTION_FAMILY
            else {}
        ),
    )

    trained: dict[str, dict[str, object]] = {}

    def _fake_train_single_model(
        *,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        **_kwargs,
    ):
        trained[model_name] = {
            "feature_cols": tuple(feature_cols),
            "target_col": target_col,
            "num_rows": len(df),
        }
        return SimpleNamespace()

    manager._train_single_model = _fake_train_single_model  # type: ignore[attr-defined]

    manager._train_attn_models_for_cluster(  # type: ignore[attr-defined]
        cluster_type=ClusterType.DECODE,
        replica_config=replica_config,
        execution_time_predictor_config=SimpleNamespace(),
        replica_scheduler_config=SimpleNamespace(block_size=16),
        linear_ops_file=str(linear_ops_file),
        attn_file=str(attn_file),
        trained_model_signatures=set(),
    )

    assert trained["catalog_kv"] == {
        "feature_cols": ("catalog_kv_feature",),
        "target_col": "time_stats.catalog_kv.median",
        "num_rows": 2,
    }
    assert trained["catalog_prefill"] == {
        "feature_cols": ("catalog_prefill_feature",),
        "target_col": "time_stats.catalog_prefill.median",
        "num_rows": 1,
    }
    assert trained["catalog_decode"] == {
        "feature_cols": ("catalog_decode_feature",),
        "target_col": "time_stats.catalog_decode.median",
        "num_rows": 1,
    }


def test_shared_manager_dense_physical_attention_roles_do_not_depend_on_family_order(
    tmp_path,
    monkeypatch,
) -> None:
    manager = _make_manager_without_init()
    replica_config = _make_dense_replica_config()

    linear_ops_file = tmp_path / "linear_op.csv"
    attn_file = tmp_path / "attention.csv"
    linear_ops_file.write_text("", encoding="utf-8")
    attn_file.write_text("", encoding="utf-8")

    manager._load_linear_op_df = lambda *_args, **_kwargs: pd.DataFrame(  # type: ignore[attr-defined]
        {
            "num_tokens": [1],
            "time_stats.input_layernorm.median": [0.1],
            "time_stats.attn_pre_proj.median": [0.1],
            "time_stats.attn_post_proj.median": [0.1],
            "time_stats.attn_rope.median": [0.1],
        }
    )
    manager._load_attention_df = lambda *_args, **_kwargs: pd.DataFrame(  # type: ignore[attr-defined]
        {
            "is_decode": [False, True],
            "is_mixed_batch": [False, False],
            "is_true_mixed_batch": [False, False],
            "prefill_chunk_size": [16, 0],
            "cache_feature": [10, 20],
            "prefill_feature": [30, 0],
            "decode_feature": [0, 40],
            "time_stats.role_cache.median": [0.1, 0.2],
            "time_stats.role_prefill.median": [0.3, 0.0],
            "time_stats.role_decode.median": [0.0, 0.4],
        }
    )
    manager._get_attention_df_with_derived_features = lambda df: df  # type: ignore[attr-defined]

    monkeypatch.setattr(
        shared_prediction_model_manager,
        "get_enabled_predictor_metric_names",
        lambda family: (
            ("role_prefill", "role_decode", "role_cache")
            if family is DENSE_ATTENTION_FAMILY
            else ()
        ),
    )
    monkeypatch.setattr(
        shared_prediction_model_manager,
        "get_enabled_predictor_metric_name_by_role",
        lambda family, role: (
            {
                AttentionOperatorRole.CACHE_WRITE: "role_cache",
                AttentionOperatorRole.PREFILL_KERNEL: "role_prefill",
                AttentionOperatorRole.DECODE_KERNEL: "role_decode",
            }[role]
            if family is DENSE_ATTENTION_FAMILY
            else ""
        ),
    )
    monkeypatch.setattr(
        shared_prediction_model_manager,
        "get_enabled_predictor_median_columns",
        lambda family: (
            (
                "time_stats.role_prefill.median",
                "time_stats.role_decode.median",
                "time_stats.role_cache.median",
            )
            if family is DENSE_ATTENTION_FAMILY
            else ()
        ),
    )
    monkeypatch.setattr(
        shared_prediction_model_manager,
        "get_enabled_shared_predictor_feature_columns",
        lambda family: (
            {
                "role_cache": ("cache_feature",),
                "role_prefill": ("prefill_feature",),
                "role_decode": ("decode_feature",),
            }
            if family is DENSE_ATTENTION_FAMILY
            else {}
        ),
    )

    trained: dict[str, dict[str, object]] = {}

    def _fake_train_single_model(
        *,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        **_kwargs,
    ):
        trained[model_name] = {
            "feature_cols": tuple(feature_cols),
            "target_col": target_col,
            "num_rows": len(df),
        }
        return SimpleNamespace()

    manager._train_single_model = _fake_train_single_model  # type: ignore[attr-defined]

    manager._train_attn_models_for_cluster(  # type: ignore[attr-defined]
        cluster_type=ClusterType.DECODE,
        replica_config=replica_config,
        execution_time_predictor_config=SimpleNamespace(),
        replica_scheduler_config=SimpleNamespace(block_size=16),
        linear_ops_file=str(linear_ops_file),
        attn_file=str(attn_file),
        trained_model_signatures=set(),
    )

    assert trained["role_cache"] == {
        "feature_cols": ("cache_feature",),
        "target_col": "time_stats.role_cache.median",
        "num_rows": 2,
    }
    assert trained["role_prefill"] == {
        "feature_cols": ("prefill_feature",),
        "target_col": "time_stats.role_prefill.median",
        "num_rows": 1,
    }
    assert trained["role_decode"] == {
        "feature_cols": ("decode_feature",),
        "target_col": "time_stats.role_decode.median",
        "num_rows": 1,
    }


class _ConcreteSklearnPredictor(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise AssertionError("_get_estimator should not be called by this unit test")


def test_sklearn_attention_model_names_follow_dense_family_catalog(
    monkeypatch,
) -> None:
    predictor = _ConcreteSklearnPredictor.__new__(_ConcreteSklearnPredictor)
    predictor._model_config = _dense_model_config()

    monkeypatch.setattr(
        "frontier.execution_time_predictor.sklearn_execution_time_predictor.get_enabled_predictor_metric_names",
        lambda family: (
            ("catalog_kv", "catalog_prefill", "catalog_decode")
            if family is DENSE_ATTENTION_FAMILY
            else ()
        ),
    )

    assert predictor._get_attention_model_names() == [
        "catalog_kv",
        "catalog_prefill",
        "catalog_decode",
    ]


def test_sklearn_kv_cache_save_training_uses_shared_family_mapping(
    monkeypatch,
) -> None:
    predictor = _ConcreteSklearnPredictor.__new__(_ConcreteSklearnPredictor)
    predictor._attention_input_file = "unused.csv"
    predictor._compute_input_file = "unused_linear.csv"
    predictor._model_config = _dense_model_config()

    predictor._get_compute_model_names = lambda: []  # type: ignore[attr-defined]
    predictor._load_attention_df = lambda _path: pd.DataFrame(  # type: ignore[attr-defined]
        {
            "catalog_kv_feature": [10, 20],
            "time_stats.catalog_kv.median": [0.1, 0.2],
        }
    )
    predictor._get_attention_df_with_derived_features = lambda df: df  # type: ignore[attr-defined]

    monkeypatch.setattr(
        "frontier.execution_time_predictor.sklearn_execution_time_predictor.get_enabled_predictor_metric_names",
        lambda family: (
            ("catalog_kv", "catalog_prefill", "catalog_decode")
            if family is DENSE_ATTENTION_FAMILY
            else ()
        ),
    )
    monkeypatch.setattr(
        sklearn_execution_time_predictor,
        "get_enabled_predictor_metric_name_by_role",
        lambda family, role: (
            {
                AttentionOperatorRole.CACHE_WRITE: "catalog_kv",
                AttentionOperatorRole.PREFILL_KERNEL: "catalog_prefill",
                AttentionOperatorRole.DECODE_KERNEL: "catalog_decode",
            }[role]
            if family is DENSE_ATTENTION_FAMILY
            else ""
        ),
    )
    monkeypatch.setattr(
        "frontier.execution_time_predictor.sklearn_execution_time_predictor.get_enabled_predictor_median_columns",
        lambda family: (
            (
                "time_stats.catalog_kv.median",
                "time_stats.catalog_prefill.median",
                "time_stats.catalog_decode.median",
            )
            if family is DENSE_ATTENTION_FAMILY
            else ()
        ),
    )
    monkeypatch.setattr(
        "frontier.execution_time_predictor.sklearn_execution_time_predictor.get_enabled_shared_predictor_feature_columns",
        lambda family: (
            {
                "catalog_kv": ("catalog_kv_feature",),
                "catalog_prefill": ("catalog_prefill_feature",),
                "catalog_decode": ("catalog_decode_feature",),
            }
            if family is DENSE_ATTENTION_FAMILY
            else {}
        ),
    )

    trained: dict[str, dict[str, object]] = {}

    def _fake_train_model(
        *,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ):
        trained[model_name] = {
            "feature_cols": tuple(feature_cols),
            "target_col": target_col,
            "num_rows": len(df),
        }
        return SimpleNamespace()

    predictor._train_model = _fake_train_model  # type: ignore[attr-defined]

    predictor._train_compute_models()

    assert trained == {
        "catalog_kv": {
            "feature_cols": ("catalog_kv_feature",),
            "target_col": "time_stats.catalog_kv.median",
            "num_rows": 2,
        }
    }


def test_sklearn_kv_cache_save_training_uses_role_not_family_order(
    monkeypatch,
) -> None:
    predictor = _ConcreteSklearnPredictor.__new__(_ConcreteSklearnPredictor)
    predictor._attention_input_file = "unused.csv"
    predictor._compute_input_file = "unused_linear.csv"
    predictor._model_config = _dense_model_config()

    predictor._get_compute_model_names = lambda: []  # type: ignore[attr-defined]
    predictor._load_attention_df = lambda _path: pd.DataFrame(  # type: ignore[attr-defined]
        {
            "cache_feature": [10, 20],
            "time_stats.role_cache.median": [0.1, 0.2],
        }
    )
    predictor._get_attention_df_with_derived_features = lambda df: df  # type: ignore[attr-defined]

    monkeypatch.setattr(
        sklearn_execution_time_predictor,
        "get_enabled_predictor_metric_names",
        lambda family: (
            ("role_prefill", "role_decode", "role_cache")
            if family is DENSE_ATTENTION_FAMILY
            else ()
        ),
    )
    monkeypatch.setattr(
        sklearn_execution_time_predictor,
        "get_enabled_predictor_metric_name_by_role",
        lambda family, role: (
            {
                AttentionOperatorRole.CACHE_WRITE: "role_cache",
                AttentionOperatorRole.PREFILL_KERNEL: "role_prefill",
                AttentionOperatorRole.DECODE_KERNEL: "role_decode",
            }[role]
            if family is DENSE_ATTENTION_FAMILY
            else ""
        ),
    )
    monkeypatch.setattr(
        sklearn_execution_time_predictor,
        "get_enabled_predictor_median_columns",
        lambda family: (
            (
                "time_stats.role_prefill.median",
                "time_stats.role_decode.median",
                "time_stats.role_cache.median",
            )
            if family is DENSE_ATTENTION_FAMILY
            else ()
        ),
    )
    monkeypatch.setattr(
        sklearn_execution_time_predictor,
        "get_enabled_shared_predictor_feature_columns",
        lambda family: (
            {
                "role_cache": ("cache_feature",),
                "role_prefill": ("prefill_feature",),
                "role_decode": ("decode_feature",),
            }
            if family is DENSE_ATTENTION_FAMILY
            else {}
        ),
    )

    trained: dict[str, dict[str, object]] = {}

    def _fake_train_model(
        *,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ):
        trained[model_name] = {
            "feature_cols": tuple(feature_cols),
            "target_col": target_col,
            "num_rows": len(df),
        }
        return SimpleNamespace()

    predictor._train_model = _fake_train_model  # type: ignore[attr-defined]

    predictor._train_compute_models()

    assert trained == {
        "role_cache": {
            "feature_cols": ("cache_feature",),
            "target_col": "time_stats.role_cache.median",
            "num_rows": 2,
        }
    }


def test_sklearn_prefill_and_decode_training_use_shared_family_mapping(
    monkeypatch,
) -> None:
    predictor = _ConcreteSklearnPredictor.__new__(_ConcreteSklearnPredictor)
    predictor._attention_input_file = "unused.csv"
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._model_config = _dense_model_config()

    predictor._load_attention_df = lambda _path: pd.DataFrame(  # type: ignore[attr-defined]
        {
            "is_decode": [False, True],
            "is_mixed_batch": [False, False],
            "is_true_mixed_batch": [False, False],
            "prefill_chunk_size": [16, 0],
            "catalog_prefill_feature": [30, 0],
            "catalog_decode_feature": [0, 40],
            "time_stats.catalog_prefill.median": [0.3, 0.0],
            "time_stats.catalog_decode.median": [0.0, 0.4],
        }
    )
    predictor._get_attention_df_with_derived_features = lambda df: df  # type: ignore[attr-defined]

    monkeypatch.setattr(
        "frontier.execution_time_predictor.sklearn_execution_time_predictor.get_enabled_predictor_metric_names",
        lambda family: (
            ("catalog_kv", "catalog_prefill", "catalog_decode")
            if family is DENSE_ATTENTION_FAMILY
            else ()
        ),
    )
    monkeypatch.setattr(
        sklearn_execution_time_predictor,
        "get_enabled_predictor_metric_name_by_role",
        lambda family, role: (
            {
                AttentionOperatorRole.CACHE_WRITE: "catalog_kv",
                AttentionOperatorRole.PREFILL_KERNEL: "catalog_prefill",
                AttentionOperatorRole.DECODE_KERNEL: "catalog_decode",
            }[role]
            if family is DENSE_ATTENTION_FAMILY
            else ""
        ),
    )
    monkeypatch.setattr(
        "frontier.execution_time_predictor.sklearn_execution_time_predictor.get_enabled_predictor_median_columns",
        lambda family: (
            (
                "time_stats.catalog_kv.median",
                "time_stats.catalog_prefill.median",
                "time_stats.catalog_decode.median",
            )
            if family is DENSE_ATTENTION_FAMILY
            else ()
        ),
    )
    monkeypatch.setattr(
        "frontier.execution_time_predictor.sklearn_execution_time_predictor.get_enabled_shared_predictor_feature_columns",
        lambda family: (
            {
                "catalog_kv": ("catalog_kv_feature",),
                "catalog_prefill": ("catalog_prefill_feature",),
                "catalog_decode": ("catalog_decode_feature",),
            }
            if family is DENSE_ATTENTION_FAMILY
            else {}
        ),
    )

    trained: dict[str, dict[str, object]] = {}

    def _fake_train_model(
        *,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ):
        trained[model_name] = {
            "feature_cols": tuple(feature_cols),
            "target_col": target_col,
            "num_rows": len(df),
        }
        return SimpleNamespace()

    predictor._train_model = _fake_train_model  # type: ignore[attr-defined]

    predictor._train_attention_layer_models()

    assert trained["catalog_prefill"] == {
        "feature_cols": ("catalog_prefill_feature",),
        "target_col": "time_stats.catalog_prefill.median",
        "num_rows": 1,
    }
    assert trained["catalog_decode"] == {
        "feature_cols": ("catalog_decode_feature",),
        "target_col": "time_stats.catalog_decode.median",
        "num_rows": 1,
    }


def test_sklearn_prefill_and_decode_training_use_roles_not_family_order(
    monkeypatch,
) -> None:
    predictor = _ConcreteSklearnPredictor.__new__(_ConcreteSklearnPredictor)
    predictor._attention_input_file = "unused.csv"
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._model_config = _dense_model_config()

    predictor._load_attention_df = lambda _path: pd.DataFrame(  # type: ignore[attr-defined]
        {
            "is_decode": [False, True],
            "is_mixed_batch": [False, False],
            "is_true_mixed_batch": [False, False],
            "prefill_chunk_size": [16, 0],
            "prefill_feature": [30, 0],
            "decode_feature": [0, 40],
            "time_stats.role_prefill.median": [0.3, 0.0],
            "time_stats.role_decode.median": [0.0, 0.4],
        }
    )
    predictor._get_attention_df_with_derived_features = lambda df: df  # type: ignore[attr-defined]

    monkeypatch.setattr(
        sklearn_execution_time_predictor,
        "get_enabled_predictor_metric_names",
        lambda family: (
            ("role_prefill", "role_decode", "role_cache")
            if family is DENSE_ATTENTION_FAMILY
            else ()
        ),
    )
    monkeypatch.setattr(
        sklearn_execution_time_predictor,
        "get_enabled_predictor_metric_name_by_role",
        lambda family, role: (
            {
                AttentionOperatorRole.CACHE_WRITE: "role_cache",
                AttentionOperatorRole.PREFILL_KERNEL: "role_prefill",
                AttentionOperatorRole.DECODE_KERNEL: "role_decode",
            }[role]
            if family is DENSE_ATTENTION_FAMILY
            else ""
        ),
    )
    monkeypatch.setattr(
        sklearn_execution_time_predictor,
        "get_enabled_predictor_median_columns",
        lambda family: (
            (
                "time_stats.role_prefill.median",
                "time_stats.role_decode.median",
                "time_stats.role_cache.median",
            )
            if family is DENSE_ATTENTION_FAMILY
            else ()
        ),
    )
    monkeypatch.setattr(
        sklearn_execution_time_predictor,
        "get_enabled_shared_predictor_feature_columns",
        lambda family: (
            {
                "role_cache": ("cache_feature",),
                "role_prefill": ("prefill_feature",),
                "role_decode": ("decode_feature",),
            }
            if family is DENSE_ATTENTION_FAMILY
            else {}
        ),
    )

    trained: dict[str, dict[str, object]] = {}

    def _fake_train_model(
        *,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ):
        trained[model_name] = {
            "feature_cols": tuple(feature_cols),
            "target_col": target_col,
            "num_rows": len(df),
        }
        return SimpleNamespace()

    predictor._train_model = _fake_train_model  # type: ignore[attr-defined]

    predictor._train_attention_layer_models()

    assert trained["role_prefill"] == {
        "feature_cols": ("prefill_feature",),
        "target_col": "time_stats.role_prefill.median",
        "num_rows": 1,
    }
    assert trained["role_decode"] == {
        "feature_cols": ("decode_feature",),
        "target_col": "time_stats.role_decode.median",
        "num_rows": 1,
    }


def _patch_sklearn_dense_runtime_role_names(monkeypatch) -> None:
    monkeypatch.setattr(
        sklearn_execution_time_predictor,
        "get_enabled_predictor_metric_name_by_role",
        lambda family, role: (
            {
                AttentionOperatorRole.CACHE_WRITE: "runtime_cache",
                AttentionOperatorRole.PREFILL_KERNEL: "runtime_prefill",
                AttentionOperatorRole.DECODE_KERNEL: "runtime_decode",
            }[role]
            if family is DENSE_ATTENTION_FAMILY
            else ""
        ),
    )


def test_sklearn_runtime_dense_attention_cache_reads_follow_role_names(
    monkeypatch,
) -> None:
    _patch_sklearn_dense_runtime_role_names(monkeypatch)

    predictor = _ConcreteSklearnPredictor.__new__(_ConcreteSklearnPredictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._model_config = _dense_model_config()
    predictor._config = SimpleNamespace(kv_cache_prediction_granularity=16)
    predictor._attention_decode_batching_overhead_fraction = 0.0
    predictor._attention_prefill_batching_overhead_fraction = 0.0
    predictor._supports_operation = lambda operation: operation in {
        "runtime_cache",
        "runtime_prefill",
        "runtime_decode",
    }
    predictor._predictions = {
        "runtime_cache": {(5,): 1.25},
        "runtime_prefill": {(16, 9): 2.5},
        "runtime_decode": {(2, 32): 3.75},
    }
    predictor._get_batch_prefill_attention_params = lambda _batch: [(16, 3)]
    predictor._get_batch_decode_attention_params = lambda _batch: (2, 32)

    kv_batch = SimpleNamespace(
        total_num_tokens=5,
        num_prefill_tokens=0,
        num_decode_tokens=0,
        requests=[object()],
    )
    prefill_batch = SimpleNamespace(
        id=10,
        num_prefill_tokens=3,
        num_decode_tokens=0,
        num_tokens=[3],
        requests=[object()],
    )
    decode_batch = SimpleNamespace(
        num_prefill_tokens=0,
        num_decode_tokens=2,
        requests=[object(), object()],
    )

    assert predictor._get_attention_kv_cache_save_execution_time(kv_batch) == 1.25
    assert predictor._get_attention_prefill_execution_time(prefill_batch) == 2.5
    assert predictor._get_attention_decode_execution_time(decode_batch) == 3.75


def test_sklearn_runtime_prediction_caches_follow_role_names(monkeypatch) -> None:
    _patch_sklearn_dense_runtime_role_names(monkeypatch)

    predictor = _ConcreteSklearnPredictor.__new__(_ConcreteSklearnPredictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._max_tokens = 2
    predictor._config = SimpleNamespace(
        kv_cache_prediction_granularity=16,
        prediction_max_batch_size=2,
        prediction_max_tokens_per_request=16,
        prediction_max_prefill_chunk_size=2,
    )
    predictor._model_config = _dense_model_config()
    predictor._replica_config = SimpleNamespace(model_config=predictor._model_config)
    predictor._requires_dense_mlp_compute_models = lambda: False
    predictor._requires_target_embedded_mtp_compute_models = lambda: False
    predictor._get_model_prediction = lambda model_name, _model, _features: {
        "source_model": model_name
    }
    predictor._models = {
        "runtime_cache": SimpleNamespace(n_features_in_=1),
        "runtime_prefill": SimpleNamespace(),
        "runtime_decode": SimpleNamespace(),
        "attn_pre_proj": SimpleNamespace(n_features_in_=1),
        "attn_post_proj": SimpleNamespace(n_features_in_=1),
        "attn_rope": SimpleNamespace(n_features_in_=1),
        "input_layernorm": SimpleNamespace(n_features_in_=1),
        "post_attention_layernorm": SimpleNamespace(n_features_in_=1),
        "add": SimpleNamespace(n_features_in_=1),
    }

    compute_predictions = predictor._predict_for_compute_models()
    attention_predictions = predictor._predict_for_attention_layer_models()

    assert compute_predictions["runtime_cache"] == {"source_model": "runtime_cache"}
    assert attention_predictions["runtime_prefill"] == {
        "source_model": "runtime_prefill"
    }
    assert attention_predictions["runtime_decode"] == {"source_model": "runtime_decode"}
    assert "attn_kv_cache_save" not in compute_predictions
    assert "attn_prefill" not in attention_predictions
    assert "attn_decode" not in attention_predictions


def test_sklearn_runtime_dense_attention_support_gates_follow_role_names(
    monkeypatch,
) -> None:
    _patch_sklearn_dense_runtime_role_names(monkeypatch)

    predictor = _ConcreteSklearnPredictor.__new__(_ConcreteSklearnPredictor)
    predictor._enable_dummy_mode = False
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._model_config = _dense_model_config()
    predictor._replica_config = SimpleNamespace(num_pipeline_stages=1)
    predictor._supports_operation = lambda operation: operation in {
        "attention",
        "runtime_cache",
        "runtime_prefill",
        "runtime_decode",
    }
    predictor._should_use_hybrid_attention_measurement_for_spec_piecewise = (
        lambda _batch: False
    )
    predictor._temporary_measurement_type = lambda _measurement_type: SimpleNamespace(
        __enter__=lambda _self: None,
        __exit__=lambda _self, _exc_type, _exc, _tb: None,
    )
    predictor._log_step3_attention_shape = lambda _batch: None
    predictor._get_attention_prefill_execution_time = lambda _batch: 7.0
    predictor._get_attention_decode_execution_time = lambda _batch: 11.0
    predictor._get_attention_kv_cache_save_execution_time = lambda _batch: 13.0
    predictor._get_attention_layer_pre_proj_execution_time = lambda _batch: 0.0
    predictor._get_attention_layer_post_proj_execution_time = lambda _batch: 0.0
    predictor._get_attention_rope_execution_time = lambda _batch: 0.0
    predictor._get_attn_norm_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_attn_inter_norm_execution_time = lambda _batch: 0.0
    predictor._get_attn_wq_proj_execution_time = lambda _batch: 0.0

    batch = SimpleNamespace(
        id=21,
        is_idle=False,
        requests=[SimpleNamespace(num_prefill_tokens=7)],
        total_num_tokens=9,
        num_prefill_tokens=7,
        num_decode_tokens=2,
        spec_decode_metadata=None,
    )

    attention_time = predictor.predict_attention_layer_time(
        batch=batch,
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert attention_time.attention_prefill_execution_time == 7.0
    assert attention_time.attention_decode_execution_time == 11.0
    assert attention_time.attention_kv_cache_save_execution_time == 13.0
