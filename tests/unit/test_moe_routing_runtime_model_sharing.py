"""Keep routing implementations distinct across shared prediction models."""

from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.config.config import RandomForrestExecutionTimePredictorConfig, ReplicaConfig
import frontier.execution_time_predictor.shared_prediction_model_manager as module
from frontier.moe_gating_runtime import get_moe_gating_runtime_context_metadata
from frontier.types import ClusterType, MeasurementType


RUNTIMES = ("standard_fused_topk", "uniform_topk")
MODEL_NAMES = ("moe_gating_routing_topk", "moe_gating_routing_topk__prefill_hot")


def manager_at(path):
    manager = object.__new__(module.ExecutionTimePredictionModelManager)
    manager._all_dummy_mode = False
    manager._active_measurement_type = MeasurementType.CUDA_EVENT
    manager._cache_dir = str(path)
    manager._cluster_configs = {}
    return manager


def replica(runtime):
    return ReplicaConfig(
        model_name="qwen3-a3b-30b-moe", device="h200",
        attn_tensor_parallel_size=4, attn_dp=2,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=8,
        moe_gating_routing_runtime_path=runtime,
    )


def training_frame():
    return pd.DataFrame([
        {"num_tokens": tokens, "routing_runtime_path": runtime,
         **get_moe_gating_runtime_context_metadata(context), "profiling_precision": "BF16",
         "measurement_type": "CUDA_EVENT",
         "time_stats.moe_gating_routing_topk.median": float(10 + index * 100),
         "time_stats.post_attention_layernorm.median": 1.0}
        for index, runtime in enumerate(RUNTIMES)
        for context in ("standalone_legacy", "prefill_hot")
        for tokens in (1, 2, 4, 8)
    ])


def small_predictor_config():
    return RandomForrestExecutionTimePredictorConfig(
        num_estimators=[2], max_depth=[2], min_samples_split=[2],
        k_fold_cv_splits=2, num_training_job_threads=1,
    )


def test_shared_training_separates_runtime_and_reuses_same_runtime(tmp_path, monkeypatch):
    """Exercise real FFN dedup, runtime filtering, fitting, storage, and projection."""
    manager = manager_at(tmp_path)
    frame = training_frame()
    source = tmp_path / "moe.csv"
    frame.to_csv(source, index=False)
    monkeypatch.setattr(module, "_get_moe_family_model_names", lambda: [MODEL_NAMES[0]])
    monkeypatch.setattr(module, "_get_prefill_hot_moe_gating_model_names", lambda: [MODEL_NAMES[1]])
    # Shape admission and unrelated operators are outside this sharing test.
    manager._validate_moe_dataset_contract = lambda *args, **kwargs: None
    manager._load_moe_df = lambda *args, **kwargs: frame.copy()
    manager._load_linear_op_df = lambda *args, **kwargs: frame.copy()
    configs = [replica(runtime) for runtime in RUNTIMES]
    signatures = set()
    for cluster, config in zip((ClusterType.PREFILL, ClusterType.DECODE), configs):
        manager._cluster_configs[cluster] = SimpleNamespace(replica_config=config)
        trained = manager._train_ffn_models_for_cluster(
            cluster, config, small_predictor_config(), str(source), str(source),
            True, signatures,
        )
        assert all(name in trained for name in MODEL_NAMES)
    for cluster, runtime in zip((ClusterType.PREFILL, ClusterType.DECODE), RUNTIMES):
        models = manager._models_view_for_family("eager", cluster)
        for name in MODEL_NAMES:
            assert models[name]._frontier_routing_runtime_path == runtime
            assert models[name].predict(pd.DataFrame({"num_tokens": [4]}))[0] == (
                10.0 if runtime == RUNTIMES[0] else 110.0
            )
    with pytest.raises(ValueError, match="multiple layer contracts or routing runtimes"):
        manager.get_models()
    manager._load_moe_df = lambda *args, **kwargs: pytest.fail("Equal runtime must reuse models")
    assert manager._train_ffn_models_for_cluster(
        ClusterType.MONOLITHIC, configs[1], small_predictor_config(),
        str(source), str(source), True, signatures,
    ) == {}
    assert manager._train_ffn_models_for_cluster(
        ClusterType.MONOLITHIC, replica(""), small_predictor_config(),
        str(source), str(source), True, signatures,
    ) == {}


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("runtime", RUNTIMES)
def test_runtime_survives_fresh_training_and_cache_hit(tmp_path, model_name, runtime):
    manager = manager_at(tmp_path)
    frame = training_frame()
    context = "prefill_hot" if model_name.endswith("__prefill_hot") else "standalone_legacy"
    frame = frame[(frame.routing_runtime_path == runtime) & (frame.gating_runtime_context == context)]
    contract = manager._resolve_typed_layer_contract(
        MODEL_NAMES[0], ClusterType.MONOLITHIC, replica(runtime), is_moe_model=True,
    )
    assert contract is not None
    kwargs = dict(
        model_name=model_name, df=frame, feature_cols=["num_tokens"],
        target_col="time_stats.moe_gating_routing_topk.median",
        execution_time_predictor_config=small_predictor_config(), layer_contract=contract,
    )
    trained = manager._train_single_model(**kwargs)
    restored = manager_at(tmp_path)
    restored._create_estimator_and_params = lambda *args: pytest.fail("Expected disk cache hit")
    cached = restored._train_single_model(**kwargs)
    assert cached._frontier_routing_runtime_path == runtime
    assert cached._frontier_model_hash == trained._frontier_model_hash
    assert restored.get_model(
        model_name, "BF16", layer_contract=contract, routing_runtime_path=runtime,
    ) is cached
    with pytest.raises(ValueError, match="conflicts"):
        restored._model_routing_runtime_identity(cached, RUNTIMES[1 - RUNTIMES.index(runtime)])


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("typed", [True, False])
def test_runtime_variants_preserve_precision_and_measurement_isolation(tmp_path, model_name, typed):
    manager = manager_at(tmp_path)
    contract = manager._resolve_typed_layer_contract(
        MODEL_NAMES[0], ClusterType.MONOLITHIC, replica(RUNTIMES[0]), is_moe_model=True,
    )
    assert contract is not None
    if not typed:
        contract = None
    stored = {}
    for measurement in (MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY):
        manager._active_measurement_type = measurement
        for precision in ("BF16", "FP8"):
            for runtime in RUNTIMES:
                model = SimpleNamespace(_frontier_routing_runtime_path=runtime)
                manager._store_model_precision(model_name, precision, model, layer_contract=contract)
                stored[measurement, precision, runtime] = model
    identity = manager._model_contract_identity(next(iter(stored.values())), contract)
    for (measurement, precision, runtime), model in stored.items():
        assert manager._get_family_model(
            manager._measurement_family_name(measurement), model_name,
            precision_key=precision, requested_identity=identity,
            requested_runtime_path=runtime,
        ) is model
    with pytest.raises(ValueError, match="multiple layer contracts or routing runtimes"):
        manager.get_model(model_name, "BF16", layer_contract=contract)


def test_routing_training_rejects_mixed_runtime_rows(tmp_path):
    manager = manager_at(tmp_path)
    with pytest.raises(ValueError, match="one routing_runtime_path"):
        manager._train_single_model(
            MODEL_NAMES[0], training_frame(), ["num_tokens"],
            "time_stats.moe_gating_routing_topk.median", small_predictor_config(),
        )


def test_dataset_validation_requires_explicit_runtime_rows(tmp_path):
    source = tmp_path / "uniform.csv"
    pd.DataFrame([{
        "num_experts": 8, "router_topk": 2, "hidden_dim": 4096,
        "expert_hidden_dim": 11008, "num_tensor_parallel_workers": 1,
        "expert_parallel_size": 1, "routing_runtime_path": "uniform_topk",
    }]).to_csv(source, index=False)
    config = SimpleNamespace(
        model_config=SimpleNamespace(num_experts=8, num_experts_per_tok=2,
                                     embedding_dim=4096, mlp_hidden_dim=11008),
        moe_tensor_parallel_size=1, moe_expert_parallel_size=1,
        moe_routing_distribution_type="balanced", moe_gating_routing_runtime_path="uniform_topk",
    )
    manager = manager_at(tmp_path)
    manager._validate_moe_dataset_contract(str(source), config, [MODEL_NAMES[0]], ClusterType.MONOLITHIC)
    config.moe_gating_routing_runtime_path = ""
    with pytest.raises(ValueError, match="No moe_gating_routing_topk profiling rows match"):
        manager._validate_moe_dataset_contract(str(source), config, [MODEL_NAMES[0]], ClusterType.MONOLITHIC)
