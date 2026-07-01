#!/usr/bin/env python3
"""
Regression test: predictor max-token cache range must cover batch-level token budgets.

Bug (2026-02-01)
---------------
In vLLM v1 scheduling, the effective token count used for compute ops is the
*total tokens in batch* (e.g., batch_size * prefill_tokens). This can exceed
prediction_max_tokens_per_request (default=4096). If the predictor only
pre-computes caches up to prediction_max_tokens_per_request, runtime lookups
can fail with KeyError for common workloads, e.g.:
    prefill_tokens=2048, batch_size=4 => total_tokens=8192

Fix requirement:
When replica_scheduler_config exposes a token budget (e.g.,
VllmV1SchedulerConfig.max_tokens_in_batch), the predictor's internal max-token
range must be >= that budget so cache lookups are always in-range.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _dense_model_config() -> SimpleNamespace:
    return SimpleNamespace(
        use_mla=False,
        num_q_heads=32,
        num_kv_heads=8,
        is_step2_mini=lambda: False,
    )


def _write_csv(path: Path, header: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerow(row)


def test_max_tokens_covers_vllm_v1_max_tokens_in_batch(tmp_path: Path) -> None:
    from frontier.config import (
        MetricsConfig,
        RandomForrestExecutionTimePredictorConfig,
        ReplicaConfig,
        VllmV1SchedulerConfig,
    )
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )
    from frontier.types import ClusterType

    class _DummyPredictorImpl(SklearnExecutionTimePredictor):
        def _get_grid_search_params(self):
            return {}

        def _get_estimator(self):
            raise RuntimeError("Not used in dummy mode")

    replica_config = ReplicaConfig(
        model_name="step-moe-noquant",
        device="a800",
        network_device="a100_pairwise_nvlink",
        attn_tensor_parallel_size=1,
        attn_data_parallel_size=8,
    )
    model_config = replica_config.model_config
    assert model_config is not None

    compute_path = tmp_path / "linear_op.csv"
    _write_csv(
        compute_path,
        [
            "profiling_precision",
            "model_arch",
            "quant_signature",
            "measurement_type",
            "n_head",
            "n_kv_head",
            "n_embd",
            "n_expanded_embd",
            "use_gated_mlp",
            "vocab_size",
            "num_tensor_parallel_workers",
            "time_stats.attn_pre_proj.median",
        ],
        {
            "profiling_precision": model_config.get_default_precision().name,
            "model_arch": getattr(model_config, "model_arch", "generic"),
            "quant_signature": getattr(model_config, "get_quant_signature", lambda: "none")(),
            "measurement_type": "cuda_event",
            "n_head": model_config.num_q_heads,
            "n_kv_head": model_config.num_kv_heads,
            "n_embd": model_config.embedding_dim,
            "n_expanded_embd": model_config.mlp_hidden_dim,
            "use_gated_mlp": model_config.use_gated_mlp,
            "vocab_size": model_config.vocab_size,
            "num_tensor_parallel_workers": replica_config.attn_tensor_parallel_size,
            "time_stats.attn_pre_proj.median": 1.0,
        },
    )

    attention_path = tmp_path / "attention.csv"
    _write_csv(
        attention_path,
        [
            "profiling_precision",
            "model_arch",
            "quant_signature",
            "measurement_type",
            "n_embd",
            "n_q_head",
            "n_kv_head",
            "block_size",
            "num_tensor_parallel_workers",
            "prefill_chunk_size",
            "batch_size",
            "kv_cache_size",
            "time_stats.attn_kv_cache_save.median",
        ],
        {
            "profiling_precision": model_config.get_default_precision().name,
            "model_arch": getattr(model_config, "model_arch", "generic"),
            "quant_signature": getattr(model_config, "get_quant_signature", lambda: "none")(),
            "measurement_type": "cuda_event",
            "n_embd": model_config.embedding_dim,
            "n_q_head": model_config.num_q_heads,
            "n_kv_head": model_config.num_kv_heads,
            "block_size": 16,
            "num_tensor_parallel_workers": replica_config.attn_tensor_parallel_size,
            "prefill_chunk_size": 0,
            "batch_size": 1,
            "kv_cache_size": 0,
            "time_stats.attn_kv_cache_save.median": 0.1,
        },
    )

    predictor_config = RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=True)
    metrics_config = MetricsConfig(output_dir=str(tmp_path / "sim_out"))
    scheduler_config = VllmV1SchedulerConfig(max_tokens_in_batch=16384)

    predictor = _DummyPredictorImpl(
        predictor_config=predictor_config,
        replica_config=replica_config,
        replica_scheduler_config=scheduler_config,
        metrics_config=metrics_config,
        cluster_type=ClusterType.MONOLITHIC,
        training_file_paths={
            "compute_input_file": str(compute_path),
            "attention_input_file": str(attention_path),
        },
    )

    assert predictor._max_tokens >= scheduler_config.max_tokens_in_batch


def test_attention_layer_training_includes_mixed_prefill_model_when_features_exist() -> None:
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _DummyPredictorImpl(SklearnExecutionTimePredictor):
        def _get_grid_search_params(self):
            return {}

        def _get_estimator(self):
            raise RuntimeError("Not used in this unit test")

    predictor = _DummyPredictorImpl.__new__(_DummyPredictorImpl)
    predictor._attention_input_file = "unused.csv"
    predictor._model_config = _dense_model_config()

    predictor._load_attention_df = lambda _path: pd.DataFrame(  # type: ignore[attr-defined]
        {
            "prefill_chunk_size": [2048, 2048, 0],
            "batch_size": [1, 4, 4],
            "kv_cache_size": [0, 0, 2048],
            "total_tokens": [2048, 8192, 4],
            "avg_seq_len": [2048.0, 2048.0, 1.0],
            "min_seq_len": [2048, 2048, 1],
            "max_seq_len": [2048, 2048, 1],
            "seq_len_variance": [0.0, 0.0, 0.0],
            "seq_len_cv": [0.0, 0.0, 0.0],
            "time_stats.attn_prefill.median": [1.0, 1.2, 0.0],
            "time_stats.attn_decode.median": [0.0, 0.0, 0.2],
        }
    )

    captured: list[tuple[str, list[str], int]] = []

    def _fake_train_model(model_name: str, df: pd.DataFrame, feature_cols, target_col: str):
        del target_col
        captured.append((model_name, list(feature_cols), len(df)))
        return object()

    predictor._train_model = _fake_train_model  # type: ignore[attr-defined]

    models = predictor._train_attention_layer_models()

    assert "attn_prefill" in models
    assert "attn_decode" in models
    assert "attn_prefill_mixed" in models

    mixed_record = next(x for x in captured if x[0] == "attn_prefill_mixed")
    assert mixed_record[1] == [
        "avg_seq_len",
        "batch_cv_interaction",
        "batch_size",
        "batch_variance_interaction",
        "kv_cache_size",
        "max_seq_len",
        "min_seq_len",
        "seq_len_cv",
        "seq_len_range",
        "seq_len_variance",
        "total_tokens",
        "total_tokens_squared",
    ]
    assert mixed_record[2] == 1


def test_attention_layer_training_honors_is_prefill_for_mixed_rows() -> None:
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _DummyPredictorImpl(SklearnExecutionTimePredictor):
        def _get_grid_search_params(self):
            return {}

        def _get_estimator(self):
            raise RuntimeError("Not used in this unit test")

    predictor = _DummyPredictorImpl.__new__(_DummyPredictorImpl)
    predictor._attention_input_file = "unused.csv"
    predictor._model_config = _dense_model_config()

    predictor._load_attention_df = lambda _path: pd.DataFrame(  # type: ignore[attr-defined]
        {
            "is_prefill": [True, True, False],
            "prefill_chunk_size": [2048, 0, 0],
            "batch_size": [1, 2, 2],
            "kv_cache_size": [0, 1024, 1024],
            "total_tokens": [2048, 3584, 2],
            "avg_seq_len": [2048.0, 1792.0, 1.0],
            "min_seq_len": [2048, 1536, 1],
            "max_seq_len": [2048, 2048, 1],
            "seq_len_variance": [0.0, 65536.0, 0.0],
            "seq_len_cv": [0.0, 0.1433, 0.0],
            "time_stats.attn_prefill.median": [1.0, 1.1, 0.0],
            "time_stats.attn_decode.median": [0.0, 0.0, 0.2],
        }
    )

    captured: list[tuple[str, int]] = []

    def _fake_train_model(model_name: str, df: pd.DataFrame, feature_cols, target_col: str):
        del feature_cols, target_col
        captured.append((model_name, len(df)))
        return object()

    predictor._train_model = _fake_train_model  # type: ignore[attr-defined]

    models = predictor._train_attention_layer_models()

    assert "attn_prefill" in models
    assert "attn_decode" in models
    assert "attn_prefill_mixed" in models

    trained_sizes = {name: size for name, size in captured}
    assert trained_sizes["attn_prefill"] == 1
    assert trained_sizes["attn_decode"] == 1
    assert trained_sizes["attn_prefill_mixed"] == 1


def test_attention_layer_training_includes_decode_in_mixed_model_when_true_mixed_rows_exist() -> None:
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _DummyPredictorImpl(SklearnExecutionTimePredictor):
        def _get_grid_search_params(self):
            return {}

        def _get_estimator(self):
            raise RuntimeError("Not used in this unit test")

    predictor = _DummyPredictorImpl.__new__(_DummyPredictorImpl)
    predictor._attention_input_file = "unused.csv"
    predictor._model_config = _dense_model_config()

    predictor._load_attention_df = lambda _path: pd.DataFrame(  # type: ignore[attr-defined]
        {
            "is_prefill": [True, False, True],
            "is_true_mixed_batch": [False, False, True],
            "prefill_chunk_size": [2048, 0, 0],
            "batch_size": [1, 2, 3],
            "kv_cache_size": [0, 1024, 0],
            "num_prefill_seqs": [0, 0, 1],
            "num_decode_seqs": [0, 0, 2],
            "total_batch_size": [1, 2, 3],
            "total_prefill_tokens": [0, 0, 1024],
            "total_tokens": [2048, 2, 1026],
            "decode_avg_kv_cache_size": [0, 1024, 1024],
            "batch_composition_ratio": [0.0, 0.0, 1.0 / 3.0],
            "time_stats.attn_prefill.median": [1.0, 0.0, 1.1],
            "time_stats.attn_decode.median": [0.0, 0.2, 0.15],
        }
    )

    captured: list[tuple[str, list[str], int]] = []

    def _fake_train_model(
        model_name: str,
        df: pd.DataFrame,
        feature_cols,
        target_col: str,
    ):
        del target_col
        captured.append((model_name, list(feature_cols), len(df)))
        return object()

    predictor._train_model = _fake_train_model  # type: ignore[attr-defined]

    models = predictor._train_attention_layer_models()

    assert "attn_decode_in_mixed" in models
    decode_mixed_record = next(x for x in captured if x[0] == "attn_decode_in_mixed")
    assert decode_mixed_record[1] == [
        "decode_batch_size",
        "decode_avg_kv_cache_size",
        "num_prefill_seqs",
        "total_prefill_tokens",
        "total_batch_size",
        "batch_composition_ratio",
        "total_tokens",
    ]
    assert decode_mixed_record[2] == 1


def test_attention_decode_execution_time_mixed_batch_requires_decode_in_mixed_model() -> None:
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )
    from frontier.types import ClusterType

    class _DummyPredictorImpl(SklearnExecutionTimePredictor):
        def _get_grid_search_params(self):
            return {}

        def _get_estimator(self):
            raise RuntimeError("Not used in this unit test")

    predictor = _DummyPredictorImpl.__new__(_DummyPredictorImpl)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._attention_decode_batching_overhead_fraction = 0.1
    predictor._predictions = {
        "attn_decode": {(2, 1024): 0.123},
    }
    predictor._supports_operation = lambda _op: True  # type: ignore[attr-defined]
    predictor._get_batch_decode_attention_params = lambda _batch: (2, 1024)  # type: ignore[attr-defined]

    batch = SimpleNamespace(
        id=99,
        num_prefill_tokens=1024,
        num_decode_tokens=2,
    )

    with pytest.raises(ValueError, match="attn_decode_in_mixed"):
        predictor._get_attention_decode_execution_time(batch)


def test_attention_decode_execution_time_mixed_batch_uses_decode_in_mixed_prediction() -> None:
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )
    from frontier.types import ClusterType

    class _DummyPredictorImpl(SklearnExecutionTimePredictor):
        def _get_grid_search_params(self):
            return {}

        def _get_estimator(self):
            raise RuntimeError("Not used in this unit test")

    predictor = _DummyPredictorImpl.__new__(_DummyPredictorImpl)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._attention_decode_batching_overhead_fraction = 0.1
    predictor._predictions = {
        "attn_decode": {(2, 1024): 0.5},
        "attn_decode_in_mixed": {
            "_on_demand_prediction": True,
            "_feature_names": [
                "decode_batch_size",
                "decode_avg_kv_cache_size",
                "num_prefill_seqs",
                "total_prefill_tokens",
                "total_batch_size",
                "batch_composition_ratio",
                "total_tokens",
            ],
            "_model": object(),
        },
    }
    predictor._supports_operation = lambda _op: True  # type: ignore[attr-defined]
    predictor._get_batch_decode_attention_params = lambda _batch: (2, 1024)  # type: ignore[attr-defined]
    predictor._get_batch_decode_mixed_features = lambda _batch: {  # type: ignore[attr-defined]
        "decode_batch_size": 2,
        "decode_avg_kv_cache_size": 1024,
        "num_prefill_seqs": 1,
        "total_prefill_tokens": 1024,
        "total_batch_size": 3,
        "batch_composition_ratio": 1.0 / 3.0,
        "total_tokens": 1026,
    }
    predictor._get_on_demand_prediction = lambda model_name, _features: (  # type: ignore[attr-defined]
        0.042 if model_name == "attn_decode_in_mixed" else 0.0
    )

    batch = SimpleNamespace(
        id=100,
        num_prefill_tokens=1024,
        num_decode_tokens=2,
    )

    predicted = predictor._get_attention_decode_execution_time(batch)
    assert predicted == 0.042


def test_attention_decode_execution_time_mixed_batch_fail_fast_for_non_monolithic_cluster() -> None:
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )
    from frontier.types import ClusterType

    class _DummyPredictorImpl(SklearnExecutionTimePredictor):
        def _get_grid_search_params(self):
            return {}

        def _get_estimator(self):
            raise RuntimeError("Not used in this unit test")

    predictor = _DummyPredictorImpl.__new__(_DummyPredictorImpl)
    predictor._cluster_type = ClusterType.DECODE
    predictor._attention_decode_batching_overhead_fraction = 0.1
    predictor._predictions = {
        "attn_decode": {(2, 1024): 0.123},
        "attn_decode_in_mixed": {
            "_on_demand_prediction": True,
            "_feature_names": [
                "decode_batch_size",
                "decode_avg_kv_cache_size",
                "num_prefill_seqs",
                "total_prefill_tokens",
                "total_batch_size",
                "batch_composition_ratio",
                "total_tokens",
            ],
            "_model": object(),
        },
    }
    predictor._supports_operation = lambda _op: True  # type: ignore[attr-defined]
    predictor._get_batch_decode_attention_params = lambda _batch: (2, 1024)  # type: ignore[attr-defined]

    batch = SimpleNamespace(
        id=101,
        num_prefill_tokens=1024,
        num_decode_tokens=2,
    )

    with pytest.raises(ValueError, match="co-location"):
        predictor._get_attention_decode_execution_time(batch)


def test_prefill_mixed_on_demand_prediction_uses_model_feature_names() -> None:
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )
    from frontier.types import ClusterType

    class _DummyPredictorImpl(SklearnExecutionTimePredictor):
        def _get_grid_search_params(self):
            return {}

        def _get_estimator(self):
            raise RuntimeError("Not used in this unit test")

    model = SimpleNamespace(
        n_features_in_=3,
        _frontier_feature_names=[
            "z_feature",
            "a_feature",
            "m_feature",
        ],
    )

    predictor = _DummyPredictorImpl.__new__(_DummyPredictorImpl)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._model_config = _dense_model_config()
    predictor._models = {"attn_prefill_mixed": model}
    predictor._config = SimpleNamespace(
        prediction_max_batch_size=2,
        prediction_max_tokens_per_request=128,
        prediction_max_prefill_chunk_size=128,
        kv_cache_prediction_granularity=64,
    )
    predictor._get_model_prediction = lambda *_args, **_kwargs: {}  # type: ignore[attr-defined]

    predictions = predictor._predict_for_attention_layer_models()

    assert predictions["attn_prefill_mixed"]["_on_demand_prediction"] is True
    assert predictions["attn_prefill_mixed"]["_feature_names"] == [
        "z_feature",
        "a_feature",
        "m_feature",
    ]


def test_on_demand_prediction_uses_dataframe_feature_names() -> None:
    import warnings

    from sklearn.ensemble import RandomForestRegressor

    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _DummyPredictorImpl(SklearnExecutionTimePredictor):
        def _get_grid_search_params(self):
            return {}

        def _get_estimator(self):
            raise RuntimeError("Not used in this unit test")

    feature_names = ["decode_batch_size", "decode_avg_kv_cache_size"]
    training_df = pd.DataFrame(
        [[1.0, 128.0], [2.0, 256.0]],
        columns=feature_names,
    )
    model = RandomForestRegressor(n_estimators=1, random_state=0)
    model.fit(training_df, [0.1, 0.2])

    predictor = _DummyPredictorImpl.__new__(_DummyPredictorImpl)
    predictor._active_measurement_type = None
    predictor._measurement_family_name = lambda _measurement_type: "eager"  # type: ignore[method-assign]
    predictor._runtime_cache = {"eager": {"attn_decode_in_mixed": {}}}
    predictor._predictions = {
        "attn_decode_in_mixed": {
            "_on_demand_prediction": True,
            "_feature_names": feature_names,
            "_model": model,
        }
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prediction = predictor._get_on_demand_prediction(
            "attn_decode_in_mixed",
            {
                "decode_batch_size": 1.0,
                "decode_avg_kv_cache_size": 128.0,
            },
        )

    assert prediction >= 0.0
    assert not any("valid feature names" in str(warning.message) for warning in caught)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
