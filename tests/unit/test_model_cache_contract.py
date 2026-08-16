"""Tests for the shared execution-time model cache contract."""

from __future__ import annotations

import pandas as pd
import pickle
import pytest
from types import SimpleNamespace
from sklearn.ensemble import RandomForestRegressor

from frontier.config.config import RandomForrestExecutionTimePredictorConfig
from frontier.config.precision_type import PrecisionType
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.training.base_trainer import BaseTrainer
from frontier.types import ClusterType, MeasurementType
from frontier.execution_time_predictor.model_cache_contract import (
    MODEL_CACHE_CONTRACT_VERSION,
    attach_model_cache_metadata,
    build_canonical_operator_binding,
    build_model_cache_hash,
    build_training_options,
    validate_cached_model,
)
from frontier.execution_time_predictor.prediction_cache_contract import (
    PREDICTION_CACHE_CONTRACT_VERSION,
    prediction_grid_digest,
)


def test_training_cv_policy_is_shared_across_cache_producers() -> None:
    from frontier.execution_time_predictor.model_cache_contract import (
        resolve_training_cv_splits,
    )

    assert resolve_training_cv_splits(10, 3) == 3
    assert resolve_training_cv_splits(10, 1) == 2


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True, "64"])
def test_attention_training_identity_rejects_invalid_kv_granularity(invalid) -> None:
    with pytest.raises(ValueError, match="kv_cache_prediction_granularity.*positive integer"):
        build_training_options(
            "attn_prefill",
            k_fold_cv_splits=2,
            kv_cache_prediction_granularity=invalid,
        )


def test_non_attention_training_identity_does_not_require_kv_granularity() -> None:
    assert build_training_options(
        "attn_pre_proj",
        k_fold_cv_splits=2,
    ) == {"k_fold_cv_splits": 2}


class _Trainer(BaseTrainer):
    def _load_dataset(self) -> pd.DataFrame:
        raise AssertionError("not used")

    def _get_model_names(self):
        raise AssertionError("not used")

    def _get_feature_cols(self, model_name: str):
        raise AssertionError("not used")

    def _get_target_col(self, model_name: str):
        raise AssertionError("not used")


class _Predictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        return RandomForestRegressor(random_state=0)

    def _get_grid_search_params(self):
        return {
            "n_estimators": self._config.num_estimators,
            "max_depth": self._config.max_depth,
            "min_samples_split": self._config.min_samples_split,
        }

    def to_dict(self) -> dict:
        return {"probe": True}


def test_all_training_paths_use_the_same_model_hash(
    tmp_path,
) -> None:
    config = RandomForrestExecutionTimePredictorConfig(
        num_estimators=[7],
        max_depth=[3],
        min_samples_split=[2],
    )
    trainer = _Trainer(
        dataset_path="unused.csv",
        output_dir=str(tmp_path),
        predictor_type="random_forest",
        num_estimators=[7],
        max_depth=[3],
        min_samples_split=[2],
    )
    trainer._profiling_precision = PrecisionType.FP16
    trainer._measurement_type = MeasurementType.CUDA_EVENT
    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    predictor = _Predictor.__new__(_Predictor)
    predictor._config = config
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    dataframe = pd.DataFrame(
        {
            "num_tokens": [1, 2, 4],
            "time_stats.attn_pre_proj.median": [0.1, 0.2, 0.4],
            "profiling_precision": ["FP16", "FP16", "FP16"],
        }
    )

    feature_names = ["num_tokens"]
    target_col = "time_stats.attn_pre_proj.median"
    base_hash = trainer._get_model_hash(
        "attn_pre_proj", dataframe, feature_names, target_col
    )
    shared_hash = manager._get_model_hash(
        "attn_pre_proj",
        dataframe,
        config,
        "FP16",
        MeasurementType.CUDA_EVENT,
        feature_cols=feature_names,
        target_col=target_col,
    )
    standalone_hash = predictor._get_model_hash(
        "attn_pre_proj", dataframe, feature_names, target_col
    )

    assert base_hash == shared_hash == standalone_hash


def test_attention_cache_write_producers_share_three_feature_identity(
    tmp_path,
) -> None:
    config = RandomForrestExecutionTimePredictorConfig(
        num_estimators=[7],
        max_depth=[3],
        min_samples_split=[2],
    )
    trainer = _Trainer(
        dataset_path="unused.csv",
        output_dir=str(tmp_path),
        predictor_type="random_forest",
        num_estimators=[7],
        max_depth=[3],
        min_samples_split=[2],
    )
    trainer._profiling_precision = PrecisionType.FP16
    trainer._measurement_type = MeasurementType.CUDA_EVENT
    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    predictor = _Predictor.__new__(_Predictor)
    predictor._config = config
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    dataframe = pd.DataFrame(
        {
            "total_tokens": [1, 8, 16, 32],
            "kv_cache_size": [0, 0, 64, 128],
            "batch_size": [1, 1, 2, 4],
            "time_stats.attn_kv_cache_save.median": [0.1, 0.2, 0.4, 0.8],
            "profiling_precision": ["FP16"] * 4,
        }
    )
    feature_names = ["total_tokens", "kv_cache_size", "batch_size"]
    target_col = "time_stats.attn_kv_cache_save.median"

    base_hash = trainer._get_model_hash(
        "attn_kv_cache_save", dataframe, feature_names, target_col
    )
    shared_hash = manager._get_model_hash(
        "attn_kv_cache_save",
        dataframe,
        config,
        "FP16",
        MeasurementType.CUDA_EVENT,
        feature_cols=feature_names,
        target_col=target_col,
    )
    standalone_hash = predictor._get_model_hash(
        "attn_kv_cache_save", dataframe, feature_names, target_col
    )

    assert base_hash == shared_hash == standalone_hash


def test_model_hash_changes_for_each_identity_component() -> None:
    dataframe = pd.DataFrame(
        {
            "num_tokens": [1, 2, 4],
            "time_stats.attn_pre_proj.median": [0.1, 0.2, 0.4],
            "profiling_precision": ["FP16", "FP16", "FP16"],
        }
    )
    estimator = RandomForestRegressor(n_estimators=2, random_state=0)
    common = dict(
        model_name="attn_pre_proj",
        dataframe=dataframe,
        profiling_precision="FP16",
        measurement_type=MeasurementType.CUDA_EVENT,
        feature_names=["num_tokens"],
        target_col="time_stats.attn_pre_proj.median",
        estimator=estimator,
        hyperparameter_grid={"n_estimators": [2]},
        training_options={"k_fold_cv_splits": 2},
        operator_binding=build_canonical_operator_binding(
            "attn_pre_proj", dataframe=dataframe
        ),
    )
    baseline = build_model_cache_hash(**common)
    for field, value in (
        ("model_name", "attn_post_proj"),
        ("profiling_precision", "BF16"),
        ("measurement_type", MeasurementType.KERNEL_ONLY),
        ("feature_names", ["other_feature"]),
        ("target_col", "other_target"),
        ("hyperparameter_grid", {"n_estimators": [4]}),
        ("training_options", {"k_fold_cv_splits": 3}),
    ):
        changed = dict(common)
        changed[field] = value
        if field == "model_name":
            changed["operator_binding"] = build_canonical_operator_binding(
                value, dataframe=changed["dataframe"]
            )
        if field == "feature_names":
            changed["dataframe"] = dataframe.assign(other_feature=dataframe.num_tokens)
        if field == "target_col":
            changed["dataframe"] = dataframe.assign(other_target=dataframe.iloc[:, 1])
        assert build_model_cache_hash(**changed) != baseline, field

    changed_df = dataframe.copy()
    changed_df.loc[0, "time_stats.attn_pre_proj.median"] = 9.9
    changed = dict(common, dataframe=changed_df)
    assert build_model_cache_hash(**changed) != baseline


def test_model_hash_binds_the_declared_prediction_domain() -> None:
    dataframe = pd.DataFrame(
        {
            "num_tokens": [1, 2, 4],
            "time_stats.attn_pre_proj.median": [0.1, 0.2, 0.4],
            "profiling_precision": ["FP16"] * 3,
        }
    )
    estimator = RandomForestRegressor(n_estimators=2, random_state=0)
    common = dict(
        model_name="attn_pre_proj",
        dataframe=dataframe,
        profiling_precision="FP16",
        measurement_type=MeasurementType.CUDA_EVENT,
        feature_names=["num_tokens"],
        target_col="time_stats.attn_pre_proj.median",
        estimator=estimator,
        hyperparameter_grid={"n_estimators": [2]},
        operator_binding=build_canonical_operator_binding(
            "attn_pre_proj", dataframe=dataframe
        ),
    )
    interval_domain = {
        "contract_version": PREDICTION_CACHE_CONTRACT_VERSION,
        "operator_name": "attn_pre_proj",
        "feature_names": ["num_tokens"],
        "domain_kind": "integer_interval_interpolation",
        "bounds": {"num_tokens": {"min": 1, "max": 4}},
    }
    exact_domain = {
        **interval_domain,
        "domain_kind": "exact_rows",
        "training_keys": [[1], [2], [4]],
        "training_key_digest": prediction_grid_digest(
            ["num_tokens"], [[1], [2], [4]]
        ),
    }

    interval_hash = build_model_cache_hash(
        **common,
        feature_domain=interval_domain,
    )
    exact_hash = build_model_cache_hash(
        **common,
        feature_domain=exact_domain,
    )

    assert interval_hash != exact_hash


def test_model_hash_binds_operator_structural_context() -> None:
    dataframe = pd.DataFrame(
        {
            "num_tokens": [1, 2, 4],
            "time_stats.attn_pre_proj.median": [0.1, 0.2, 0.4],
        }
    )
    estimator = RandomForestRegressor(n_estimators=2, random_state=0)
    common = dict(
        model_name="attn_pre_proj",
        dataframe=dataframe,
        profiling_precision="FP16",
        measurement_type=MeasurementType.CUDA_EVENT,
        feature_names=["num_tokens"],
        target_col="time_stats.attn_pre_proj.median",
        estimator=estimator,
        hyperparameter_grid={"n_estimators": [2]},
    )
    baseline = build_model_cache_hash(
        **common,
        operator_binding={
            "operator_family": "dense_attention",
            "n_q_head": 32,
            "n_kv_head": 8,
            "head_size": 128,
            "tensor_parallel_size": 1,
        },
    )
    changed_binding = build_model_cache_hash(
        **common,
        operator_binding={
            "operator_family": "dense_attention",
            "n_q_head": 32,
            "n_kv_head": 8,
            "head_size": 128,
            "tensor_parallel_size": 2,
        },
    )

    assert changed_binding != baseline


def test_cached_model_persists_and_validates_operator_binding() -> None:
    dataframe = pd.DataFrame(
        {
            "num_tokens": [1, 2, 4],
            "time_stats.attn_pre_proj.median": [0.1, 0.2, 0.4],
        }
    )
    estimator = RandomForestRegressor(n_estimators=2, random_state=0)
    operator_binding = {
        "operator_family": "dense_attention",
        "n_q_head": 32,
        "n_kv_head": 8,
        "head_size": 128,
        "tensor_parallel_size": 1,
    }
    model_hash = build_model_cache_hash(
        model_name="attn_pre_proj",
        dataframe=dataframe,
        profiling_precision="FP16",
        measurement_type=MeasurementType.CUDA_EVENT,
        feature_names=["num_tokens"],
        target_col="time_stats.attn_pre_proj.median",
        estimator=estimator,
        hyperparameter_grid={"n_estimators": [2]},
        operator_binding=operator_binding,
    )
    domain = {
        "contract_version": PREDICTION_CACHE_CONTRACT_VERSION,
        "operator_name": "attn_pre_proj",
        "feature_names": ["num_tokens"],
        "domain_kind": "integer_interval_interpolation",
        "bounds": {"num_tokens": {"min": 1.0, "max": 4.0}},
    }
    attach_model_cache_metadata(
        estimator,
        model_name="attn_pre_proj",
        model_hash=model_hash,
        feature_names=["num_tokens"],
        target_col="time_stats.attn_pre_proj.median",
        feature_domain=domain,
        operator_binding=operator_binding,
    )

    assert estimator._frontier_operator_binding["operator_name"] == "attn_pre_proj"
    assert estimator._frontier_operator_binding["operator_family"] == "dense_attention"
    assert estimator._frontier_operator_binding["contract_version"] == 1
    for field, value in operator_binding.items():
        assert estimator._frontier_operator_binding[field] == value
    assert validate_cached_model(
        "attn_pre_proj",
        estimator,
        expected_model_hash=model_hash,
        feature_names=["num_tokens"],
        target_col="time_stats.attn_pre_proj.median",
        operator_binding=operator_binding,
    ) is estimator

    estimator._frontier_operator_binding["tensor_parallel_size"] = 2
    with pytest.raises(ValueError, match="operator binding mismatch"):
        validate_cached_model(
            "attn_pre_proj",
            estimator,
            expected_model_hash=model_hash,
            feature_names=["num_tokens"],
            target_col="time_stats.attn_pre_proj.median",
            operator_binding=operator_binding,
        )


def _valid_model_metadata(dataframe: pd.DataFrame):
    feature_names = ["num_tokens"]
    target_col = "time_stats.attn_pre_proj.median"
    estimator = RandomForestRegressor(n_estimators=2, random_state=0)
    operator_binding = build_canonical_operator_binding(
        "attn_pre_proj", dataframe=dataframe
    )
    model_hash = build_model_cache_hash(
        model_name="attn_pre_proj",
        dataframe=dataframe,
        profiling_precision="FP16",
        measurement_type=MeasurementType.CUDA_EVENT,
        feature_names=feature_names,
        target_col=target_col,
        estimator=estimator,
        hyperparameter_grid={"n_estimators": [2]},
        operator_binding=operator_binding,
    )
    domain = {
        "contract_version": PREDICTION_CACHE_CONTRACT_VERSION,
        "operator_name": "attn_pre_proj",
        "feature_names": feature_names,
        "domain_kind": "integer_interval_interpolation",
        "bounds": {"num_tokens": {"min": 1.0, "max": 4.0}},
    }
    attach_model_cache_metadata(
        estimator,
        model_name="attn_pre_proj",
        model_hash=model_hash,
        feature_names=feature_names,
        target_col=target_col,
        feature_domain=domain,
        operator_binding=operator_binding,
    )
    return estimator, model_hash, feature_names, target_col


def test_valid_model_cache_metadata_is_accepted() -> None:
    dataframe = pd.DataFrame(
        {
            "num_tokens": [1, 2, 4],
            "time_stats.attn_pre_proj.median": [0.1, 0.2, 0.4],
        }
    )
    model, model_hash, feature_names, target_col = _valid_model_metadata(dataframe)

    assert validate_cached_model(
        "attn_pre_proj",
        model,
        expected_model_hash=model_hash,
        feature_names=feature_names,
        target_col=target_col,
    ) is model
    assert model._frontier_model_cache_contract_version == MODEL_CACHE_CONTRACT_VERSION


def test_model_cache_requires_operator_bound_feature_domain() -> None:
    dataframe = pd.DataFrame(
        {
            "num_tokens": [1, 2, 4],
            "time_stats.attn_pre_proj.median": [0.1, 0.2, 0.4],
        }
    )
    model = RandomForestRegressor(n_estimators=2, random_state=0)
    model_hash = build_model_cache_hash(
        model_name="attn_pre_proj",
        dataframe=dataframe,
        profiling_precision="FP16",
        measurement_type=MeasurementType.CUDA_EVENT,
        feature_names=["num_tokens"],
        target_col="time_stats.attn_pre_proj.median",
        estimator=model,
        hyperparameter_grid={"n_estimators": [2]},
        operator_binding=build_canonical_operator_binding(
            "attn_pre_proj", dataframe=dataframe
        ),
    )
    domain = {
        "contract_version": PREDICTION_CACHE_CONTRACT_VERSION,
        "feature_names": ["num_tokens"],
        "domain_kind": "integer_interval_interpolation",
        "bounds": {"num_tokens": {"min": 1.0, "max": 4.0}},
    }

    with pytest.raises(ValueError, match="operator_binding|operator_name"):
        attach_model_cache_metadata(
            model,
            model_name="attn_pre_proj",
            model_hash=model_hash,
            feature_names=["num_tokens"],
            target_col="time_stats.attn_pre_proj.median",
            feature_domain=domain,
        )
        validate_cached_model(
            "attn_pre_proj",
            model,
            expected_model_hash=model_hash,
            feature_names=["num_tokens"],
            target_col="time_stats.attn_pre_proj.median",
        )


def test_model_cache_rejects_missing_or_wrong_contract_version() -> None:
    dataframe = pd.DataFrame(
        {
            "num_tokens": [1, 2, 4],
            "time_stats.attn_pre_proj.median": [0.1, 0.2, 0.4],
        }
    )
    model, model_hash, feature_names, target_col = _valid_model_metadata(dataframe)

    delattr(model, "_frontier_model_cache_contract_version")
    with pytest.raises(ValueError, match="unsupported cache contract version"):
        validate_cached_model(
            "attn_pre_proj",
            model,
            expected_model_hash=model_hash,
            feature_names=feature_names,
            target_col=target_col,
        )

    model, model_hash, feature_names, target_col = _valid_model_metadata(dataframe)
    model._frontier_feature_domain["contract_version"] = (
        PREDICTION_CACHE_CONTRACT_VERSION + 1
    )
    with pytest.raises(ValueError, match="feature domain has unsupported contract version"):
        validate_cached_model(
            "attn_pre_proj",
            model,
            expected_model_hash=model_hash,
            feature_names=feature_names,
            target_col=target_col,
        )

    model, model_hash, feature_names, target_col = _valid_model_metadata(dataframe)
    model._frontier_model_cache_contract_version = MODEL_CACHE_CONTRACT_VERSION + 1
    with pytest.raises(ValueError, match="unsupported cache contract version"):
        validate_cached_model(
            "attn_pre_proj",
            model,
            expected_model_hash=model_hash,
            feature_names=feature_names,
            target_col=target_col,
        )


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("_frontier_model_hash", "wrong-hash", "hash mismatch"),
        (
            "_frontier_estimator_params_digest",
            "wrong-params",
            "hyperparameter metadata mismatch",
        ),
        ("_frontier_feature_names", ["other_feature"], "feature schema mismatch"),
        ("_frontier_target_col", "other_target", "target column mismatch"),
        ("_frontier_feature_domain", None, "malformed feature-domain"),
    ],
)
def test_model_cache_rejects_malformed_identity_metadata(
    attribute: str,
    value,
    message: str,
) -> None:
    dataframe = pd.DataFrame(
        {
            "num_tokens": [1, 2, 4],
            "time_stats.attn_pre_proj.median": [0.1, 0.2, 0.4],
        }
    )
    model, model_hash, feature_names, target_col = _valid_model_metadata(dataframe)
    setattr(model, attribute, value)

    with pytest.raises(ValueError, match=message):
        validate_cached_model(
            "attn_pre_proj",
            model,
            expected_model_hash=model_hash,
            feature_names=feature_names,
            target_col=target_col,
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda domain: domain.pop("domain_kind"), "domain_kind"),
        (lambda domain: domain.pop("bounds"), "bounds"),
        (
            lambda domain: domain["bounds"]["num_tokens"].update(min=float("nan")),
            "invalid bounds",
        ),
    ],
)
def test_model_cache_rejects_malformed_domain_descriptor(mutator, message: str) -> None:
    dataframe = pd.DataFrame(
        {
            "num_tokens": [1, 2, 4],
            "time_stats.attn_pre_proj.median": [0.1, 0.2, 0.4],
        }
    )
    model, model_hash, feature_names, target_col = _valid_model_metadata(dataframe)
    mutator(model._frontier_feature_domain)

    with pytest.raises(ValueError, match=message):
        validate_cached_model(
            "attn_pre_proj",
            model,
            expected_model_hash=model_hash,
            feature_names=feature_names,
            target_col=target_col,
        )


def _model_with_domain(
    *,
    feature_names: list[str],
    domain: dict,
):
    """Build an estimator carrying a caller-supplied domain descriptor."""
    dataframe = pd.DataFrame(
        {
            **{name: [1, 2, 3, 4] for name in feature_names},
            "target": [0.1, 0.2, 0.3, 0.4],
        }
    )
    estimator = RandomForestRegressor(n_estimators=2, random_state=0)
    if len(feature_names) == 1:
        identity_domain = {
            "contract_version": PREDICTION_CACHE_CONTRACT_VERSION,
            "operator_name": "probe",
            "feature_names": feature_names,
            "domain_kind": "integer_interval_interpolation",
            "bounds": {
                name: {"min": 1.0, "max": 4.0} for name in feature_names
            },
        }
    else:
        identity_keys = [(index, index) for index in range(1, 5)]
        identity_domain = {
            "contract_version": PREDICTION_CACHE_CONTRACT_VERSION,
            "operator_name": "probe",
            "feature_names": feature_names,
            "domain_kind": "exact_rows",
            "bounds": {
                name: {"min": 1.0, "max": 4.0} for name in feature_names
            },
            "training_keys": [list(key) for key in identity_keys],
            "training_key_digest": prediction_grid_digest(
                feature_names, identity_keys
            ),
        }
    model_hash = build_model_cache_hash(
        model_name="probe",
        dataframe=dataframe,
        profiling_precision="FP16",
        measurement_type=MeasurementType.CUDA_EVENT,
        feature_names=feature_names,
        target_col="target",
        estimator=estimator,
        hyperparameter_grid={"n_estimators": [2]},
        # Use a valid interval domain to derive an identity hash; the domain
        # is replaced below so validation is exercised independently.
        feature_domain=identity_domain,
        operator_binding=build_canonical_operator_binding(
            "probe", dataframe=dataframe
        ),
    )
    attach_model_cache_metadata(
        estimator,
        model_name="probe",
        model_hash=model_hash,
        feature_names=feature_names,
        target_col="target",
        feature_domain=domain,
        operator_binding=build_canonical_operator_binding(
            "probe", dataframe=dataframe
        ),
    )
    return estimator, model_hash


def test_exact_row_domain_requires_matching_training_key_digest() -> None:
    names = ["batch_size", "kv_cache_size"]
    keys = [[1, 16], [2, 32]]
    domain = {
        "contract_version": PREDICTION_CACHE_CONTRACT_VERSION,
        "operator_name": "probe",
        "feature_names": names,
        "domain_kind": "exact_rows",
        "bounds": {
            "batch_size": {"min": 1.0, "max": 2.0},
            "kv_cache_size": {"min": 16.0, "max": 32.0},
        },
        "training_keys": keys,
    }
    with pytest.raises(ValueError, match="training_key_digest"):
        _model_with_domain(feature_names=names, domain=domain)

    domain["training_key_digest"] = "wrong-digest"
    with pytest.raises(ValueError, match="training_key_digest"):
        _model_with_domain(feature_names=names, domain=domain)

    domain["training_key_digest"] = prediction_grid_digest(names, keys)
    model, model_hash = _model_with_domain(feature_names=names, domain=domain)
    assert validate_cached_model(
        "probe",
        model,
        expected_model_hash=model_hash,
        feature_names=names,
        target_col="target",
    ) is model


def test_conditional_domain_requires_well_formed_constraints() -> None:
    names = ["kv_cache_size", "prefill_chunk_size_squared"]
    base = {
        "contract_version": PREDICTION_CACHE_CONTRACT_VERSION,
        "operator_name": "probe",
        "feature_names": names,
        "domain_kind": "conditional_interpolation",
        "bounds": {
            "kv_cache_size": {"min": 0.0, "max": 128.0},
            "prefill_chunk_size_squared": {"min": 1.0, "max": 4096.0},
        },
    }
    for constraints in (
        None,
        [],
        [{"type": "unknown", "features": ["kv_cache_size"], "max": 128}],
        [{"type": "sum_lte", "features": ["missing"], "max": 128}],
        [{"type": "sum_lte", "features": ["kv_cache_size"], "max": float("inf")}],
    ):
        domain = dict(base)
        if constraints is not None:
            domain["constraints"] = constraints
        with pytest.raises(ValueError, match="constraint"):
            _model_with_domain(feature_names=names, domain=domain)


def test_verified_cartesian_domain_validates_axis_product_descriptor() -> None:
    names = ["batch_size", "kv_cache_size"]
    base = {
        "contract_version": PREDICTION_CACHE_CONTRACT_VERSION,
        "operator_name": "probe",
        "feature_names": names,
        "domain_kind": "verified_cartesian_interpolation",
        "bounds": {
            "batch_size": {"min": 1.0, "max": 2.0},
            "kv_cache_size": {"min": 16.0, "max": 32.0},
        },
        "axis_values": {"batch_size": [1, 2], "kv_cache_size": [16, 32]},
        "axis_semantics": {
            "batch_size": "integer_interval",
            "kv_cache_size": "integer_interval",
        },
        "axis_product_size": 4,
        "axis_product_digest": prediction_grid_digest(
            names, [(1, 16), (1, 32), (2, 16), (2, 32)]
        ),
    }
    model, model_hash = _model_with_domain(feature_names=names, domain=base)
    assert validate_cached_model(
        "probe",
        model,
        expected_model_hash=model_hash,
        feature_names=names,
        target_col="target",
    ) is model

    malformed = dict(base)
    malformed["axis_values"] = {
        "batch_size": [1, 3],
        "kv_cache_size": [16, 32],
    }
    with pytest.raises(ValueError, match="axis_values|bounds|product"):
        _model_with_domain(feature_names=names, domain=malformed)


@pytest.mark.parametrize("loader", ["base", "shared", "standalone"])
def test_all_model_cache_loaders_validate_and_accept_valid_metadata(
    tmp_path,
    loader: str,
) -> None:
    dataframe = pd.DataFrame(
        {
            "num_tokens": [1, 2, 4],
            "time_stats.attn_pre_proj.median": [0.1, 0.2, 0.4],
        }
    )
    model, model_hash, feature_names, target_col = _valid_model_metadata(dataframe)
    cache_file = tmp_path / f"attn_pre_proj_{model_hash}.pkl"
    with cache_file.open("wb") as cache_stream:
        pickle.dump(model, cache_stream, protocol=pickle.HIGHEST_PROTOCOL)

    if loader == "base":
        cache_owner = _Trainer(
            dataset_path="unused.csv",
            output_dir=str(tmp_path),
            predictor_type="random_forest",
        )
        loaded = cache_owner._load_model_from_cache(
            "attn_pre_proj",
            model_hash,
            feature_cols=feature_names,
            target_col=target_col,
        )
    elif loader == "shared":
        cache_owner = ExecutionTimePredictionModelManager.__new__(
            ExecutionTimePredictionModelManager
        )
        cache_owner._cache_dir = str(tmp_path)
        loaded = cache_owner._load_model_from_cache(
            "attn_pre_proj",
            model_hash,
            feature_cols=feature_names,
            target_col=target_col,
        )
    else:
        cache_owner = _Predictor.__new__(_Predictor)
        cache_owner._cache_dir = str(tmp_path)
        cache_owner._config = type("Config", (), {"no_cache": False})()
        loaded = cache_owner._load_model_from_cache(
            "attn_pre_proj",
            model_hash,
            feature_cols=feature_names,
            target_col=target_col,
        )

    assert loaded._frontier_model_hash == model_hash


@pytest.mark.parametrize("owner", ["base", "shared", "standalone"])
def test_all_model_cache_stores_reject_uncontracted_estimators(tmp_path, owner: str) -> None:
    model = RandomForestRegressor(n_estimators=2, random_state=0)
    if owner == "base":
        cache_owner = _Trainer(
            dataset_path="unused.csv",
            output_dir=str(tmp_path),
            predictor_type="random_forest",
        )
    elif owner == "shared":
        cache_owner = ExecutionTimePredictionModelManager.__new__(
            ExecutionTimePredictionModelManager
        )
        cache_owner._cache_dir = str(tmp_path)
    else:
        cache_owner = _Predictor.__new__(_Predictor)
        cache_owner._cache_dir = str(tmp_path)
        cache_owner._config = type("Config", (), {"no_cache": False})()

    with pytest.raises(ValueError, match="cache contract version|feature schema"):
        cache_owner._store_model_in_cache("attn_pre_proj", "expected-hash", model)


@pytest.mark.parametrize("owner", ["base", "shared", "standalone"])
def test_all_model_cache_stores_validate_expected_operator_binding(
    tmp_path, owner: str
) -> None:
    """A producer must re-check its expected binding before writing an artifact."""
    dataframe = pd.DataFrame(
        {
            "num_tokens": [1, 2, 4],
            "time_stats.attn_pre_proj.median": [0.1, 0.2, 0.4],
        }
    )
    model, model_hash, feature_names, target_col = _valid_model_metadata(dataframe)
    expected_binding = dict(model._frontier_operator_binding)
    model._frontier_operator_binding["tensor_parallel_size"] = 2

    if owner == "base":
        cache_owner = _Trainer(
            dataset_path="unused.csv",
            output_dir=str(tmp_path),
            predictor_type="random_forest",
        )
    elif owner == "shared":
        cache_owner = ExecutionTimePredictionModelManager.__new__(
            ExecutionTimePredictionModelManager
        )
        cache_owner._cache_dir = str(tmp_path)
    else:
        cache_owner = _Predictor.__new__(_Predictor)
        cache_owner._cache_dir = str(tmp_path)
        cache_owner._config = type("Config", (), {"no_cache": False})()

    with pytest.raises(ValueError, match="operator binding mismatch"):
        cache_owner._store_model_in_cache(
            "attn_pre_proj",
            model_hash,
            model,
            operator_binding=expected_binding,
        )


@pytest.mark.parametrize("loader", ["base", "shared", "standalone"])
def test_all_model_cache_loaders_reject_wrong_hash(tmp_path, loader: str) -> None:
    dataframe = pd.DataFrame(
        {
            "num_tokens": [1, 2, 4],
            "time_stats.attn_pre_proj.median": [0.1, 0.2, 0.4],
        }
    )
    model, model_hash, feature_names, target_col = _valid_model_metadata(dataframe)
    cache_file = tmp_path / "attn_pre_proj_wrong-hash.pkl"
    with cache_file.open("wb") as cache_stream:
        pickle.dump(model, cache_stream, protocol=pickle.HIGHEST_PROTOCOL)

    if loader == "base":
        cache_owner = _Trainer(
            dataset_path="unused.csv",
            output_dir=str(tmp_path),
            predictor_type="random_forest",
        )
    elif loader == "shared":
        cache_owner = ExecutionTimePredictionModelManager.__new__(
            ExecutionTimePredictionModelManager
        )
        cache_owner._cache_dir = str(tmp_path)
    else:
        cache_owner = _Predictor.__new__(_Predictor)
        cache_owner._cache_dir = str(tmp_path)
        cache_owner._config = type("Config", (), {"no_cache": False})()

    with pytest.raises(ValueError, match="hash mismatch"):
        cache_owner._load_model_from_cache(
            "attn_pre_proj",
            "wrong-hash",
            feature_cols=feature_names,
            target_col=target_col,
        )


def test_base_trainer_training_persists_contract_and_reloads(tmp_path) -> None:
    trainer = _Trainer(
        dataset_path="unused.csv",
        output_dir=str(tmp_path),
        predictor_type="random_forest",
        num_estimators=[2],
        max_depth=[3],
        min_samples_split=[2],
        k_fold_cv_splits=2,
        num_training_job_threads=1,
    )
    trainer._profiling_precision = PrecisionType.FP16
    trainer._measurement_type = MeasurementType.CUDA_EVENT
    dataframe = pd.DataFrame(
        {
            "num_tokens": [1, 2, 4, 8],
            "time_stats.attn_pre_proj.median": [0.1, 0.2, 0.4, 0.8],
            "profiling_precision": ["FP16"] * 4,
        }
    )
    model = trainer._train_single_model(
        "attn_pre_proj",
        dataframe,
        ["num_tokens"],
        "time_stats.attn_pre_proj.median",
    )

    assert model._frontier_model_cache_contract_version == MODEL_CACHE_CONTRACT_VERSION
    reloaded = trainer._load_model_from_cache(
        "attn_pre_proj",
        model._frontier_model_hash,
        feature_cols=["num_tokens"],
        target_col="time_stats.attn_pre_proj.median",
    )
    assert reloaded._frontier_model_hash == model._frontier_model_hash


def test_standalone_predictor_training_persists_contract_and_reloads(tmp_path) -> None:
    predictor = _Predictor.__new__(_Predictor)
    predictor._cache_dir = str(tmp_path)
    predictor._config = type(
        "Config",
        (),
        {
            "no_cache": False,
            "num_estimators": [2],
            "max_depth": [3],
            "min_samples_split": [2],
            "k_fold_cv_splits": 2,
            "num_training_job_threads": 1,
        },
    )()
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    dataframe = pd.DataFrame(
        {
            "num_tokens": [1, 2, 4, 8],
            "time_stats.attn_pre_proj.median": [0.1, 0.2, 0.4, 0.8],
            "profiling_precision": ["FP16"] * 4,
        }
    )
    model = predictor._train_model(
        "attn_pre_proj",
        dataframe,
        ["num_tokens"],
        "time_stats.attn_pre_proj.median",
    )

    assert model._frontier_model_cache_contract_version == MODEL_CACHE_CONTRACT_VERSION
    reloaded = predictor._load_model_from_cache(
        "attn_pre_proj",
        model._frontier_model_hash,
        feature_cols=["num_tokens"],
        target_col="time_stats.attn_pre_proj.median",
    )
    assert reloaded._frontier_model_hash == model._frontier_model_hash


def test_base_and_standalone_training_produce_same_model_identity_and_predictions(
    tmp_path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "num_tokens": [1, 2, 4, 8],
            "time_stats.attn_pre_proj.median": [0.1, 0.2, 0.4, 0.8],
            "profiling_precision": ["FP16"] * 4,
        }
    )
    trainer = _Trainer(
        dataset_path="unused.csv",
        output_dir=str(tmp_path / "base"),
        predictor_type="random_forest",
        num_estimators=[2],
        max_depth=[3],
        min_samples_split=[2],
        k_fold_cv_splits=10,
        num_training_job_threads=1,
    )
    trainer._measurement_type = MeasurementType.CUDA_EVENT
    predictor = _Predictor.__new__(_Predictor)
    predictor._cache_dir = str(tmp_path / "standalone")
    predictor._config = type(
        "Config",
        (),
        {
            "no_cache": False,
            "num_estimators": [2],
            "max_depth": [3],
            "min_samples_split": [2],
            "k_fold_cv_splits": 10,
            "num_training_job_threads": 1,
        },
    )()
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT

    base_model = trainer._train_single_model(
        "attn_pre_proj",
        dataframe,
        ["num_tokens"],
        "time_stats.attn_pre_proj.median",
    )
    standalone_model = predictor._train_model(
        "attn_pre_proj",
        dataframe,
        ["num_tokens"],
        "time_stats.attn_pre_proj.median",
    )

    assert base_model._frontier_model_hash == standalone_model._frontier_model_hash
    features = dataframe[["num_tokens"]]
    assert base_model.predict(features).tolist() == standalone_model.predict(features).tolist()


@pytest.mark.parametrize("producer", ["shared", "standalone"])
def test_training_producers_fail_fast_on_missing_feature_and_target_schema(
    tmp_path,
    producer: str,
) -> None:
    dataframe = pd.DataFrame({"present": [1.0]})
    feature_names = ["missing_feature"]
    target_col = "missing_target"

    if producer == "shared":
        owner = ExecutionTimePredictionModelManager.__new__(
            ExecutionTimePredictionModelManager
        )
        train = lambda: owner._train_single_model(
            "attn_decode",
            dataframe,
            feature_names,
            target_col,
            object(),
            {"input_file": "attention.csv"},
        )
    else:
        owner = _Predictor.__new__(_Predictor)
        owner._cache_dir = str(tmp_path)
        owner._config = object()
        train = lambda: owner._train_model(
            "attn_decode",
            dataframe,
            feature_names,
            target_col,
        )

    with pytest.raises(
        ValueError,
        match=r"attn_decode.*missing_feature.*missing_target.*Re-run profiling",
    ):
        train()


def _structural_linear_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num_tokens": [1, 2, 4, 8],
            "time_stats.attn_pre_proj.median": [0.1, 0.2, 0.4, 0.8],
            "profiling_precision": ["FP16"] * 4,
            "measurement_type": ["CUDA_EVENT"] * 4,
            "n_head": [32] * 4,
            "n_kv_head": [8] * 4,
            "n_embd": [4096] * 4,
            "n_expanded_embd": [14336] * 4,
            "head_size": [128] * 4,
            "block_size": [16] * 4,
            "num_tensor_parallel_workers": [2] * 4,
            "model_arch": ["qwen3"] * 4,
            "model_architecture_profile": ["generic"] * 4,
            "quant_signature": ["fp16"] * 4,
        }
    )


def _structural_model_config() -> SimpleNamespace:
    profile = SimpleNamespace(profile_id="generic")
    return SimpleNamespace(
        num_q_heads=32,
        num_kv_heads=8,
        embedding_dim=4096,
        mlp_hidden_dim=14336,
        num_experts=None,
        num_experts_per_tok=None,
        get_name=lambda: "Qwen3-30B-A3B-tiny",
        get_model_arch=lambda: "qwen3",
        get_model_architecture_profile=lambda: profile,
        get_quant_signature=lambda: "fp16",
        get_head_dim=lambda: 128,
    )


def test_realistic_structural_slice_has_identical_hash_across_all_producers(
    tmp_path,
) -> None:
    dataframe = _structural_linear_training_frame()
    config = RandomForrestExecutionTimePredictorConfig(
        num_estimators=[7],
        max_depth=[3],
        min_samples_split=[2],
    )
    model_config = _structural_model_config()
    trainer = _Trainer(
        dataset_path="unused.csv",
        output_dir=str(tmp_path / "base"),
        predictor_type="random_forest",
        num_estimators=[7],
        max_depth=[3],
        min_samples_split=[2],
    )
    trainer._measurement_type = MeasurementType.CUDA_EVENT
    trainer.device = "h800"
    trainer.model_name = "Qwen3-30B-A3B-tiny"
    trainer.tensor_parallel_size = 2
    trainer.block_size = 16
    trainer.model_config = model_config

    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    shared_context = {
        "cluster_type": str(ClusterType.MONOLITHIC),
        "device": "h800",
        "model_name": "Qwen3-30B-A3B-tiny",
        "tensor_parallel_size": 2,
        "block_size": 16,
        "model_arch": "qwen3",
        "model_architecture_profile": "generic",
        "quant_signature": "fp16",
    }

    predictor = _Predictor.__new__(_Predictor)
    predictor._config = config
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._model_config = model_config
    predictor._replica_config = SimpleNamespace(
        device="h800",
        model_name="Qwen3-30B-A3B-tiny",
        attn_tensor_parallel_size=2,
        attn_data_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        expert_parallel_size=1,
        num_pipeline_stages=1,
        replica_scheduler_config=SimpleNamespace(block_size=16),
    )

    feature_names = ["num_tokens"]
    target_col = "time_stats.attn_pre_proj.median"
    hashes = {
        trainer._get_model_hash(
            "attn_pre_proj", dataframe, feature_names, target_col
        ),
        manager._get_model_hash(
            "attn_pre_proj",
            dataframe,
            config,
            "FP16",
            MeasurementType.CUDA_EVENT,
            feature_cols=feature_names,
            target_col=target_col,
            training_context=shared_context,
        ),
        predictor._get_model_hash(
            "attn_pre_proj", dataframe, feature_names, target_col
        ),
    }

    assert len(hashes) == 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("num_tensor_parallel_workers", 4),
        ("n_head", 64),
        ("model_arch", "qwen3_next"),
        ("quant_signature", "fp8"),
        ("measurement_type", "KERNEL_ONLY"),
    ],
)
def test_model_hash_changes_when_profile_structure_changes(
    field: str,
    replacement,
) -> None:
    dataframe = _structural_linear_training_frame()
    estimator = RandomForestRegressor(n_estimators=2, random_state=0)
    common = {
        "model_name": "attn_pre_proj",
        "profiling_precision": "FP16",
        "measurement_type": MeasurementType.CUDA_EVENT,
        "feature_names": ["num_tokens"],
        "target_col": "time_stats.attn_pre_proj.median",
        "estimator": estimator,
        "hyperparameter_grid": {"n_estimators": [2]},
    }
    baseline_binding = BaseTrainer._build_operator_binding(
        SimpleNamespace(), "attn_pre_proj", dataframe
    )
    baseline = build_model_cache_hash(
        dataframe=dataframe,
        operator_binding=baseline_binding,
        **common,
    )
    changed_dataframe = dataframe.copy()
    changed_dataframe[field] = replacement
    changed_binding = BaseTrainer._build_operator_binding(
        SimpleNamespace(), "attn_pre_proj", changed_dataframe
    )

    assert build_model_cache_hash(
        dataframe=changed_dataframe,
        operator_binding=changed_binding,
        **common,
    ) != baseline


def test_operator_binding_ignores_serving_max_model_len() -> None:
    baseline = _structural_linear_training_frame()
    baseline["max_model_len"] = 4096
    wider_serving = baseline.copy()
    wider_serving["max_model_len"] = 8192

    baseline_binding = BaseTrainer._build_operator_binding(
        SimpleNamespace(), "attn_pre_proj", baseline
    )
    wider_binding = BaseTrainer._build_operator_binding(
        SimpleNamespace(), "attn_pre_proj", wider_serving
    )

    assert baseline_binding == wider_binding


def test_model_hash_ignores_serving_max_model_len_metadata() -> None:
    baseline = _structural_linear_training_frame()
    baseline["max_model_len"] = 4096
    wider_serving = baseline.copy()
    wider_serving["max_model_len"] = 8192
    estimator = RandomForestRegressor(n_estimators=2, random_state=0)
    common = {
        "model_name": "attn_pre_proj",
        "profiling_precision": "FP16",
        "measurement_type": MeasurementType.CUDA_EVENT,
        "feature_names": ["num_tokens"],
        "target_col": "time_stats.attn_pre_proj.median",
        "estimator": estimator,
        "hyperparameter_grid": {"n_estimators": [2]},
    }
    baseline_binding = BaseTrainer._build_operator_binding(
        SimpleNamespace(), "attn_pre_proj", baseline
    )
    wider_binding = BaseTrainer._build_operator_binding(
        SimpleNamespace(), "attn_pre_proj", wider_serving
    )
    baseline_hash = build_model_cache_hash(
        dataframe=baseline,
        operator_binding=baseline_binding,
        **common,
    )
    wider_hash = build_model_cache_hash(
        dataframe=wider_serving,
        operator_binding=wider_binding,
        **common,
    )
    assert wider_hash == baseline_hash


def test_canonical_binding_accepts_and_validates_existing_binding() -> None:
    existing = {
        "operator_family": "attention",
        "profile_structure": {
            "n_head": 32,
            "num_tensor_parallel_workers": 2,
        },
    }

    binding = build_canonical_operator_binding(
        "attn_pre_proj",
        operator_binding=existing,
    )

    assert binding["operator_name"] == "attn_pre_proj"
    assert binding["profile_structure"] == existing["profile_structure"]


def test_canonical_binding_ignores_pandas_missing_structural_cells() -> None:
    dataframe = pd.DataFrame(
        {
            "n_head": pd.Series([pd.NA], dtype="Int64"),
            "num_tensor_parallel_workers": pd.Series([pd.NA], dtype="Int64"),
        }
    )

    binding = build_canonical_operator_binding(
        "attn_pre_proj",
        dataframe=dataframe,
    )

    assert "profile_structure" not in binding


def test_model_hash_requires_explicit_operator_binding() -> None:
    dataframe = pd.DataFrame(
        {
            "num_tokens": [1, 2],
            "time_stats.attn_pre_proj.median": [0.1, 0.2],
        }
    )

    with pytest.raises(ValueError, match="operator_binding.*required"):
        build_model_cache_hash(
            model_name="attn_pre_proj",
            dataframe=dataframe,
            profiling_precision="FP16",
            measurement_type=MeasurementType.CUDA_EVENT,
            feature_names=["num_tokens"],
            target_col="time_stats.attn_pre_proj.median",
            estimator=RandomForestRegressor(n_estimators=2, random_state=0),
            hyperparameter_grid={"n_estimators": [2]},
        )


def test_model_metadata_requires_explicit_operator_binding() -> None:
    model = RandomForestRegressor(n_estimators=2, random_state=0)
    domain = {
        "contract_version": PREDICTION_CACHE_CONTRACT_VERSION,
        "operator_name": "attn_pre_proj",
        "feature_names": ["num_tokens"],
        "domain_kind": "integer_interval_interpolation",
        "bounds": {"num_tokens": {"min": 1.0, "max": 2.0}},
    }

    with pytest.raises(ValueError, match="operator_binding.*required"):
        attach_model_cache_metadata(
            model,
            model_name="attn_pre_proj",
            model_hash="hash",
            feature_names=["num_tokens"],
            target_col="target",
            feature_domain=domain,
        )
