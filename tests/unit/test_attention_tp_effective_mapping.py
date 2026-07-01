from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.attention.ops import AttentionOperatorRole
from frontier.execution_time_predictor import shared_prediction_model_manager
from frontier.execution_time_predictor import sklearn_execution_time_predictor
from frontier.execution_time_predictor.attention_tp_policy import (
    get_attention_non_linear_tp_policy_ops,
    resolve_effective_attention_tp_size,
)
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.types import ClusterType


class _DummySklearnPredictor(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        return None



def test_attention_policy_keeps_requested_tp_for_kv_head_replication() -> None:
    effective_tp = resolve_effective_attention_tp_size(
        op_name="attn_prefill",
        requested_tp_size=8,
        num_kv_heads=1,
        cluster_type=ClusterType.MONOLITHIC,
        warning_cache=set(),
        include_linear_ops=False,
    )
    assert effective_tp == 8


def test_attention_policy_non_linear_ops_are_derived_from_dense_family_spec() -> None:
    assert get_attention_non_linear_tp_policy_ops() == frozenset(
        {
            "attn_kv_cache_save",
            "attn_prefill",
            "attn_decode",
        }
    )


def test_attention_policy_raises_for_invalid_tp_kv_head_relation() -> None:
    with pytest.raises(ValueError, match="Unsupported attention TP configuration"):
        resolve_effective_attention_tp_size(
            op_name="attn_prefill",
            requested_tp_size=8,
            num_kv_heads=3,
            cluster_type=ClusterType.MONOLITHIC,
            warning_cache=set(),
            include_linear_ops=False,
        )


def test_sklearn_predictor_load_attention_df_uses_requested_tp(tmp_path: Path) -> None:
    predictor = object.__new__(_DummySklearnPredictor)
    predictor._replica_config = SimpleNamespace(attn_tensor_parallel_size=8)
    predictor._model_config = SimpleNamespace(
        embedding_dim=7168,
        num_q_heads=64,
        num_kv_heads=1,
    )
    predictor._block_size = 16
    predictor._cluster_type = ClusterType.MONOLITHIC

    predictor._get_profiling_metadata = lambda *_args, **_kwargs: None
    predictor._register_profiling_metadata_for_ops = lambda *_args, **_kwargs: None
    predictor._get_attention_model_names = lambda: ["attn_kv_cache_save"]

    attention_df = pd.DataFrame(
        {
            "n_embd": [7168, 7168],
            "n_q_head": [64, 64],
            "n_kv_head": [1, 1],
            "block_size": [16, 16],
            "num_tensor_parallel_workers": [1, 8],
            "time_stats.attn_kv_cache_save.median": [1.0, 2.0],
        }
    )
    input_file = tmp_path / "attention.csv"
    attention_df.to_csv(input_file, index=False)

    filtered = predictor._load_attention_df(str(input_file))
    assert len(filtered) == 1
    assert int(filtered.iloc[0]["num_tensor_parallel_workers"]) == 8


def test_sklearn_predictor_load_attention_df_uses_role_derived_dense_loader_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = object.__new__(_DummySklearnPredictor)
    predictor._replica_config = SimpleNamespace(attn_tensor_parallel_size=8)
    predictor._model_config = SimpleNamespace(
        embedding_dim=7168,
        num_q_heads=64,
        num_kv_heads=1,
    )
    predictor._block_size = 16
    predictor._cluster_type = ClusterType.MONOLITHIC

    predictor._get_profiling_metadata = lambda *_args, **_kwargs: None
    predictor._validate_active_measurement_type = lambda *_args, **_kwargs: None
    predictor._register_profiling_metadata_for_ops = lambda *_args, **_kwargs: None
    predictor._get_attention_model_names = lambda: ["role_cache"]

    monkeypatch.setattr(
        sklearn_execution_time_predictor,
        "get_enabled_predictor_median_column_by_role",
        lambda _family, role: {
            AttentionOperatorRole.CACHE_WRITE: "time_stats.role_cache.median",
        }[role],
        raising=False,
    )
    monkeypatch.setattr(
        sklearn_execution_time_predictor,
        "get_enabled_predictor_metric_name_by_role",
        lambda _family, role: {
            AttentionOperatorRole.PREFILL_KERNEL: "role_prefill",
        }[role],
    )
    observed_op_names: list[str] = []

    def _resolve_effective_tp(**kwargs):
        observed_op_names.append(kwargs["op_name"])
        return 8 if kwargs["op_name"] == "role_prefill" else 1

    monkeypatch.setattr(
        sklearn_execution_time_predictor,
        "resolve_effective_attention_tp_size",
        _resolve_effective_tp,
    )

    attention_df = pd.DataFrame(
        {
            "n_embd": [7168, 7168],
            "n_q_head": [64, 64],
            "n_kv_head": [1, 1],
            "block_size": [16, 16],
            "num_tensor_parallel_workers": [1, 8],
        }
    )
    input_file = tmp_path / "attention.csv"
    attention_df.to_csv(input_file, index=False)

    filtered = predictor._load_attention_df(str(input_file))

    assert observed_op_names == ["role_prefill"]
    assert len(filtered) == 1
    assert int(filtered.iloc[0]["num_tensor_parallel_workers"]) == 8
    assert "time_stats.role_cache.median" in filtered.columns
    assert filtered.iloc[0]["time_stats.role_cache.median"] == 0
    assert "time_stats.attn_kv_cache_save.median" not in filtered.columns


def test_shared_model_manager_load_attention_df_uses_requested_tp(tmp_path: Path) -> None:
    manager = ExecutionTimePredictionModelManager.__new__(ExecutionTimePredictionModelManager)
    manager._attention_tp_warning_cache = set()

    attention_df = pd.DataFrame(
        {
            "n_embd": [7168, 7168],
            "n_q_head": [64, 64],
            "n_kv_head": [1, 1],
            "block_size": [16, 16],
            "num_tensor_parallel_workers": [1, 8],
            "time_stats.attn_kv_cache_save.median": [1.0, 2.0],
        }
    )
    input_file = tmp_path / "attention.csv"
    attention_df.to_csv(input_file, index=False)

    replica_config = SimpleNamespace(
        attn_tensor_parallel_size=8,
        model_config=SimpleNamespace(embedding_dim=7168, num_q_heads=64, num_kv_heads=1),
    )
    scheduler_config = SimpleNamespace(block_size=16)

    filtered = manager._load_attention_df(
        str(input_file),
        replica_config,
        scheduler_config,
        cluster_type=ClusterType.MONOLITHIC,
    )
    assert len(filtered) == 1
    assert int(filtered.iloc[0]["num_tensor_parallel_workers"]) == 8


def test_shared_model_manager_load_attention_df_uses_role_derived_dense_loader_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ExecutionTimePredictionModelManager.__new__(ExecutionTimePredictionModelManager)
    manager._attention_tp_warning_cache = set()

    monkeypatch.setattr(
        shared_prediction_model_manager,
        "get_enabled_predictor_median_column_by_role",
        lambda _family, role: {
            AttentionOperatorRole.CACHE_WRITE: "time_stats.role_cache.median",
        }[role],
        raising=False,
    )
    monkeypatch.setattr(
        shared_prediction_model_manager,
        "get_enabled_predictor_metric_name_by_role",
        lambda _family, role: {
            AttentionOperatorRole.PREFILL_KERNEL: "role_prefill",
        }[role],
    )
    observed_op_names: list[str] = []

    def _resolve_effective_tp(**kwargs):
        observed_op_names.append(kwargs["op_name"])
        return 8 if kwargs["op_name"] == "role_prefill" else 1

    monkeypatch.setattr(
        shared_prediction_model_manager,
        "resolve_effective_attention_tp_size",
        _resolve_effective_tp,
    )

    attention_df = pd.DataFrame(
        {
            "n_embd": [7168, 7168],
            "n_q_head": [64, 64],
            "n_kv_head": [1, 1],
            "block_size": [16, 16],
            "num_tensor_parallel_workers": [1, 8],
        }
    )
    input_file = tmp_path / "attention.csv"
    attention_df.to_csv(input_file, index=False)

    replica_config = SimpleNamespace(
        attn_tensor_parallel_size=8,
        model_config=SimpleNamespace(embedding_dim=7168, num_q_heads=64, num_kv_heads=1),
    )
    scheduler_config = SimpleNamespace(block_size=16)

    filtered = manager._load_attention_df(
        str(input_file),
        replica_config,
        scheduler_config,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert observed_op_names == ["role_prefill"]
    assert len(filtered) == 1
    assert int(filtered.iloc[0]["num_tensor_parallel_workers"]) == 8
    assert "time_stats.role_cache.median" in filtered.columns
    assert filtered.iloc[0]["time_stats.role_cache.median"] == 0
    assert "time_stats.attn_kv_cache_save.median" not in filtered.columns
