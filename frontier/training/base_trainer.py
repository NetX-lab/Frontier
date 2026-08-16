"""
Base trainer class for execution time prediction models.

This module provides the abstract base class for all trainers, defining the common
interface and shared utilities for training different model structures.
"""

import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd
from fasteners import InterProcessReaderWriterLock
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import numpy as np

from frontier.logger import init_logger
from frontier.config.precision_type import PrecisionType
from frontier.config.config import RandomForrestExecutionTimePredictorConfig, LinearRegressionExecutionTimePredictorConfig
from frontier.types import MeasurementType
from frontier.execution_time_predictor.model_cache_contract import (
    attach_model_cache_metadata,
    build_exact_feature_lookup,
    build_model_cache_hash,
    build_canonical_operator_binding,
    build_training_options,
    resolve_training_cv_splits,
    validate_cached_model,
)
from frontier.execution_time_predictor.prediction_cache_contract import (
    attach_feature_domain,
    build_feature_domain_descriptor,
)
from frontier.attention.training_partition import (
    prepare_dense_cache_write_training_rows,
    validate_cache_write_target_consistency,
)
from frontier.profiling.common.parallel_config import (
    validate_prediction_min_kv_cache_size,
)

logger = init_logger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for training execution time prediction models.
    
    This class provides common functionality for:
    - Loading profiling datasets
    - Training sklearn models with grid search
    - Caching trained models to disk
    - Model hash computation for cache management
    
    Subclasses should implement:
    - _load_dataset(): Load and filter profiling data
    - _get_model_names(): Return list of model names to train
    - _get_feature_cols(): Return feature column names
    - _get_target_col(): Return target column name for each model
    """
    
    def __init__(
        self,
        dataset_path: str,
        output_dir: str,
        predictor_type: str = "random_forest",
        **kwargs
    ):
        """
        Initialize the trainer.

        Args:
            dataset_path: Path to the profiling dataset CSV file
            output_dir: Directory to save trained models
            predictor_type: Type of predictor ("random_forest" or "linear_regression")
            **kwargs: Additional configuration parameters
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.predictor_type = predictor_type
        self.config = kwargs
        self.prediction_min_kv_cache_size = validate_prediction_min_kv_cache_size(
            kwargs.get("prediction_min_kv_cache_size", 0)
        )
        # Keep the normalized value visible in the trainer's serialized
        # configuration.  This is runtime materialization metadata and is
        # intentionally separate from model-training rows/identity.
        self.config["prediction_min_kv_cache_size"] = self.prediction_min_kv_cache_size
        self._profiling_precision = PrecisionType.FP16
        self._profiling_precision_source = None
        self._measurement_type = None
        self._measurement_type_source = None
        expected_measurement_type = kwargs.get("measurement_type")
        self._expected_measurement_type = (
            MeasurementType.from_string(expected_measurement_type)
            if expected_measurement_type is not None
            else None
        )

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Training configuration
        self.k_fold_cv_splits = kwargs.get("k_fold_cv_splits", 10)
        self.num_training_job_threads = kwargs.get("num_training_job_threads", -1)

        # Random Forest parameters (must match RandomForrestExecutionTimePredictorConfig defaults)
        self.num_estimators = kwargs.get("num_estimators", [250, 500, 750])
        self.max_depth = kwargs.get("max_depth", [8, 16, 32])
        self.min_samples_split = kwargs.get("min_samples_split", [2, 5, 10])

        # Linear Regression parameters (must match LinearRegressionExecutionTimePredictorConfig defaults)
        self.polynomial_degree = kwargs.get("polynomial_degree", list(range(1, 6)))
        self.polynomial_include_bias = kwargs.get("polynomial_include_bias", [True, False])
        self.polynomial_interaction_only = kwargs.get("polynomial_interaction_only", [True, False])
        self.fit_intercept = kwargs.get("fit_intercept", [True, False])

        # Create execution time predictor config for hash calculation (must match simulator)
        # Use default constructor to get all default field values, then override training-specific fields
        if predictor_type == "random_forest":
            self.execution_time_predictor_config = RandomForrestExecutionTimePredictorConfig()
            # Override training-specific fields
            self.execution_time_predictor_config.k_fold_cv_splits = self.k_fold_cv_splits
            self.execution_time_predictor_config.num_training_job_threads = self.num_training_job_threads
            self.execution_time_predictor_config.num_estimators = self.num_estimators
            self.execution_time_predictor_config.max_depth = self.max_depth
            self.execution_time_predictor_config.min_samples_split = self.min_samples_split
        elif predictor_type == "linear_regression":
            self.execution_time_predictor_config = LinearRegressionExecutionTimePredictorConfig()
            # Override training-specific fields
            self.execution_time_predictor_config.k_fold_cv_splits = self.k_fold_cv_splits
            self.execution_time_predictor_config.num_training_job_threads = self.num_training_job_threads
            self.execution_time_predictor_config.polynomial_degree = self.polynomial_degree
            self.execution_time_predictor_config.polynomial_include_bias = self.polynomial_include_bias
            self.execution_time_predictor_config.polynomial_interaction_only = self.polynomial_interaction_only
            self.execution_time_predictor_config.fit_intercept = self.fit_intercept
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")

        # The dataclass constructors are intentionally created with their
        # default values above so training-specific options can be overridden in
        # one place. Propagate the explicit lower bound as well; otherwise a
        # standalone producer silently serializes ``0`` while its caller asked
        # for a measured-domain start.
        self.execution_time_predictor_config.prediction_min_kv_cache_size = (
            self.prediction_min_kv_cache_size
        )

        logger.info(f"Initialized {self.__class__.__name__}")
        logger.info(f"Dataset path: {dataset_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Predictor type: {predictor_type}")

    def _set_profiling_precision(self, df: pd.DataFrame, source: str = "dataset") -> None:
        if "profiling_precision" not in df.columns:
            raise ValueError(
                f"profiling_precision column is missing in {source}. "
                "Profiling datasets must include explicit precision metadata."
            )

        precision_values = df["profiling_precision"].dropna().unique().tolist()
        if not precision_values:
            raise ValueError("profiling_precision column is empty")
        if len(precision_values) > 1:
            raise ValueError(
                f"Multiple profiling_precision values found: {precision_values}"
            )

        detected_precision = PrecisionType.from_string(str(precision_values[0]))
        if (
            self._profiling_precision_source is not None
            and detected_precision != self._profiling_precision
        ):
            raise ValueError(
                f"Profiling precision mismatch: existing={self._profiling_precision.name}, "
                f"new={detected_precision.name} (source={source})"
            )
        self._profiling_precision = detected_precision
        self._profiling_precision_source = source
        logger.info("Profiling precision detected: %s", self._profiling_precision.name)

    def _set_measurement_type(self, df: pd.DataFrame, source: str = "dataset") -> None:
        if "measurement_type" not in df.columns:
            raise ValueError(
                f"measurement_type column is missing in {source}. "
                "Profiling datasets must include explicit measurement metadata."
            )

        measurement_values = df["measurement_type"].dropna().unique().tolist()
        if not measurement_values:
            raise ValueError("measurement_type column is empty")
        if len(measurement_values) > 1:
            raise ValueError(
                f"Multiple measurement_type values found: {measurement_values}"
            )

        detected_measurement_type = MeasurementType.from_string(str(measurement_values[0]))
        if (
            self._measurement_type_source is not None
            and detected_measurement_type != self._measurement_type
        ):
            raise ValueError(
                f"measurement_type mismatch: existing={self._measurement_type.value}, "
                f"new={detected_measurement_type.value} (source={source})"
            )
        if (
            self._expected_measurement_type is not None
            and detected_measurement_type != self._expected_measurement_type
        ):
            raise ValueError(
                f"measurement_type mismatch for {source}: expected={self._expected_measurement_type.value}, "
                f"actual={detected_measurement_type.value}"
            )
        self._measurement_type = detected_measurement_type
        self._measurement_type_source = source
        logger.info("Measurement type detected: %s", self._measurement_type.value)

    def _set_dataset_metadata(self, df: pd.DataFrame, source: str = "dataset") -> None:
        self._set_profiling_precision(df, source=source)
        self._set_measurement_type(df, source=source)
    
    @abstractmethod
    def _load_dataset(self) -> pd.DataFrame:
        """
        Load and filter the profiling dataset.
        
        Returns:
            Filtered DataFrame ready for training
        """
        pass
    
    @abstractmethod
    def _get_model_names(self) -> List[str]:
        """
        Get the list of model names to train.
        
        Returns:
            List of model names (e.g., ["moe_gating", "moe_shuffling", "moe_grouped_gemm"])
        """
        pass
    
    @abstractmethod
    def _get_feature_cols(self, model_name: str) -> List[str]:
        """
        Get feature column names for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of feature column names
        """
        pass
    
    @abstractmethod
    def _get_target_col(self, model_name: str) -> str:
        """
        Get target column name for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Target column name
        """
        pass
    
    def _create_estimator_and_params(self):
        """Create estimator and grid search params based on predictor type."""
        if self.predictor_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(random_state=0)
            grid_search_params = {
                "n_estimators": self.num_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
            }
        elif self.predictor_type == "linear_regression":
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import PolynomialFeatures
            estimator = make_pipeline(PolynomialFeatures(), LinearRegression())
            grid_search_params = {
                "polynomialfeatures__degree": self.polynomial_degree,
                "polynomialfeatures__include_bias": self.polynomial_include_bias,
                "polynomialfeatures__interaction_only": self.polynomial_interaction_only,
                "linearregression__fit_intercept": self.fit_intercept,
            }
        else:
            raise ValueError(f"Unsupported predictor type: {self.predictor_type}")
        
        return estimator, grid_search_params
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
        """Calculate Mean Absolute Percentage Error (MAPE)."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        zero_true_mask = y_true == 0
        non_zero_true_mask = ~zero_true_mask

        error = np.zeros_like(y_true, dtype=float)
        error[non_zero_true_mask] = (
            np.abs((y_true[non_zero_true_mask] - y_pred[non_zero_true_mask]) / y_true[non_zero_true_mask]) * 100
        )
        error[zero_true_mask] = np.where(y_pred[zero_true_mask] == 0, 0, 100)

        return np.mean(error)
    
    def _get_scorer(self) -> Any:
        """Get the scorer for grid search."""
        return make_scorer(self.mean_absolute_percentage_error, greater_is_better=False)
    
    def _get_hash_relevant_config(self, config) -> Dict[str, Any]:
        """
        Extract only the configuration parameters that affect model performance.

        This legacy projection is retained for diagnostics and compatibility.
        Model-cache identity is generated by ``model_cache_contract`` so the
        standalone trainer and E2E manager cannot drift independently.

        Parameters that should be included:
        - Profiling data paths (determine input data source)
        - Prediction range parameters (determine prediction cache scope)
        - Performance adjustment parameters (affect predicted values)
        - ML hyperparameters (affect model structure)

        Parameters that should be excluded:
        - Training process parameters (k_fold_cv_splits, num_training_job_threads)
        - Runtime configuration (no_cache, skip_cpu_overhead_modeling, enable_dummy_mode, dummy_execution_time_ms)
        """
        hash_relevant_params = {
            # Category 1: Profiling data paths
            'linear_op_input_file': config.linear_op_input_file,
            'atten_input_file': config.atten_input_file,
            'all_reduce_input_file': config.all_reduce_input_file,
            'send_recv_input_file': config.send_recv_input_file,
            'moe_input_file': config.moe_input_file,
            'linear_op_kernel_only_input_file': config.linear_op_kernel_only_input_file,
            'atten_kernel_only_input_file': config.atten_kernel_only_input_file,
            'moe_kernel_only_input_file': config.moe_kernel_only_input_file,
            'cpu_overhead_input_file': config.cpu_overhead_input_file,

            # Category 2: Prediction range parameters
            'kv_cache_prediction_granularity': config.kv_cache_prediction_granularity,
            'prediction_min_kv_cache_size': getattr(config, 'prediction_min_kv_cache_size', 0),
            'prediction_max_prefill_chunk_size': config.prediction_max_prefill_chunk_size,
            'prediction_max_batch_size': config.prediction_max_batch_size,
            'prediction_max_tokens_per_request': config.prediction_max_tokens_per_request,

            # Category 3: Performance adjustment parameters
            'attention_decode_batching_overhead_fraction': config.attention_decode_batching_overhead_fraction,
            'attention_prefill_batching_overhead_fraction': config.attention_prefill_batching_overhead_fraction,
            'nccl_cpu_launch_overhead_ms': config.nccl_cpu_launch_overhead_ms,
            'nccl_cpu_skew_overhead_per_device_ms': config.nccl_cpu_skew_overhead_per_device_ms,
        }

        # Category 4: ML Hyperparameters (type-specific)
        if hasattr(config, 'num_estimators'):  # Random Forest
            hash_relevant_params['num_estimators'] = config.num_estimators
            hash_relevant_params['max_depth'] = config.max_depth
            hash_relevant_params['min_samples_split'] = config.min_samples_split
        elif hasattr(config, 'polynomial_degree'):  # Linear Regression
            hash_relevant_params['polynomial_degree'] = config.polynomial_degree
            hash_relevant_params['polynomial_include_bias'] = config.polynomial_include_bias
            hash_relevant_params['polynomial_interaction_only'] = config.polynomial_interaction_only
            hash_relevant_params['fit_intercept'] = config.fit_intercept

        return hash_relevant_params

    def _get_model_hash(
        self,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: List[str] | None = None,
        target_col: str | None = None,
        *,
        operator_binding: Dict[str, Any] | None = None,
    ) -> str:
        """
        Compute hash for model caching. Must match simulator's hash calculation.

        Hash is calculated from:
        1. Hash-relevant configuration parameters (excluding runtime/training process params)
        2. Model name
        3. DataFrame content hash

        This ensures that only changes to parameters that affect model performance
        will invalidate the cache.
        """
        if feature_cols is None or target_col is None:
            raise ValueError(
                "Model cache identity requires feature_cols and target_col; "
                "call _get_model_hash from a model training path."
            )
        estimator, grid_search_params = self._create_estimator_and_params()
        if self._measurement_type is None:
            raise ValueError("Model cache identity requires an active measurement type.")
        if "profiling_precision" not in df.columns:
            raise ValueError(
                f"Model cache identity requires profiling_precision metadata for {model_name}."
            )
        precision_values = df["profiling_precision"].dropna().unique().tolist()
        if len(precision_values) != 1:
            raise ValueError(
                f"Model cache identity requires one profiling_precision for {model_name}; "
                f"found {precision_values!r}."
            )
        if operator_binding is None:
            operator_binding = self._build_operator_binding(model_name, df)
        granularity = getattr(
            self,
            "kv_cache_prediction_granularity",
            getattr(
                self.execution_time_predictor_config,
                "kv_cache_prediction_granularity",
                None,
            ),
        )
        return build_model_cache_hash(
            model_name=model_name,
            dataframe=df,
            profiling_precision=precision_values[0],
            measurement_type=self._measurement_type,
            feature_names=feature_cols,
            target_col=target_col,
            estimator=estimator,
            hyperparameter_grid=grid_search_params,
            training_options=build_training_options(
                model_name,
                k_fold_cv_splits=self.k_fold_cv_splits,
                kv_cache_prediction_granularity=granularity,
            ),
            feature_domain=build_feature_domain_descriptor(
                df,
                feature_cols,
                operator_name=model_name,
            ),
            operator_binding=operator_binding,
        )

    def _build_operator_binding(
        self,
        model_name: str,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Build the physical/operator binding for one standalone model."""
        context: Dict[str, Any] = {}
        for field in (
            "cluster_type",
            "device",
            "model_name",
            "tensor_parallel_size",
            "attn_tensor_parallel_size",
            "moe_tensor_parallel_size",
            "expert_parallel_size",
            "moe_expert_parallel_size",
            "num_pipeline_stages",
            "block_size",
            "routing_runtime_path",
            "gating_runtime_context",
        ):
            value = getattr(self, field, None)
            if value is not None:
                context[field] = value

        model_config = getattr(self, "model_config", None)
        if model_config is not None:
            model_arch_getter = getattr(model_config, "get_model_arch", None)
            if callable(model_arch_getter):
                context["model_arch"] = model_arch_getter()
            else:
                context["model_arch"] = getattr(model_config, "model_arch", None)
            profile_getter = getattr(
                model_config, "get_model_architecture_profile", None
            )
            profile = profile_getter() if callable(profile_getter) else None
            context["model_architecture_profile"] = (
                getattr(profile, "profile_id", None)
                or getattr(model_config, "model_architecture_profile", None)
            )
            quant_getter = getattr(model_config, "get_quant_signature", None)
            if callable(quant_getter):
                context["quant_signature"] = quant_getter()
            for binding_name, model_field in (
                ("n_head", "num_q_heads"),
                ("n_q_head", "num_q_heads"),
                ("n_kv_head", "num_kv_heads"),
                ("n_embd", "embedding_dim"),
                ("n_expanded_embd", "mlp_hidden_dim"),
                ("num_experts", "num_experts"),
                ("router_topk", "num_experts_per_tok"),
            ):
                value = getattr(model_config, model_field, None)
                if value is not None:
                    context[binding_name] = value
            head_dim_getter = getattr(model_config, "get_head_dim", None)
            if callable(head_dim_getter):
                context["head_size"] = head_dim_getter()

        return build_canonical_operator_binding(
            model_name,
            context=context,
            dataframe=df,
        )
    
    def _load_model_from_cache(
        self,
        model_name: str,
        model_hash: str,
        *,
        feature_cols: List[str],
        target_col: str,
        operator_binding: Dict[str, Any] | None = None,
    ) -> BaseEstimator:
        """Load a cached model if it exists."""
        with InterProcessReaderWriterLock(f"{self.output_dir}/{model_hash}_model_lock.file").read_lock():
            cache_file = f"{self.output_dir}/{model_name}_{model_hash}.pkl"
            if not os.path.exists(cache_file):
                return None
            logger.info(f"Found cached model {model_name} (hash: {model_hash})")
            with open(cache_file, "rb") as cache_stream:
                model = pickle.load(cache_stream)
            return validate_cached_model(
                model_name,
                model,
                expected_model_hash=model_hash,
                feature_names=feature_cols,
                target_col=target_col,
                operator_binding=operator_binding,
            )
    
    def _store_model_in_cache(
        self,
        model_name: str,
        model_hash: str,
        model: BaseEstimator,
        *,
        operator_binding: Dict[str, Any] | None = None,
    ) -> None:
        """Store a trained model to cache."""
        validate_cached_model(
            model_name,
            model,
            expected_model_hash=model_hash,
            feature_names=getattr(model, "_frontier_feature_names", ()),
            target_col=getattr(model, "_frontier_target_col", ""),
            operator_binding=operator_binding,
        )
        with InterProcessReaderWriterLock(f"{self.output_dir}/{model_hash}_model_lock.file").write_lock():
            cache_file = f"{self.output_dir}/{model_name}_{model_hash}.pkl"
            pickle.dump(model, open(cache_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved model to {cache_file}")
    
    def _train_single_model(
        self,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str
    ) -> BaseEstimator:
        """
        Train a single model with grid search.
        
        Args:
            model_name: Name of the model
            df: Training data
            feature_cols: Feature column names
            target_col: Target column name
            
        Returns:
            Trained sklearn estimator
        """
        if len(df) == 0:
            raise ValueError(f"Training data for model {model_name} is empty")

        if model_name == "attn_kv_cache_save":
            df = prepare_dense_cache_write_training_rows(
                df,
                dataset_path=self.dataset_path,
            )

        required_cols = [*feature_cols, target_col]
        missing_cols = [column for column in required_cols if column not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Training data for model {model_name} is missing required columns "
                f"{missing_cols}; dataset={self.dataset_path!r}. Re-run profiling "
                "with the current feature schema."
            )
        nan_row_mask = df[required_cols].isna().any(axis=1)
        nan_row_count = int(nan_row_mask.sum())
        if nan_row_count:
            logger.warning(
                "Dropping %d/%d rows with NaN feature/target values before training %s "
                "(target=%s).",
                nan_row_count,
                len(df),
                model_name,
                target_col,
            )
            df = df.loc[~nan_row_mask].copy()
        if len(df) == 0:
            raise ValueError(
                f"Training data for model {model_name} is empty after dropping NaN "
                f"rows (target={target_col})."
            )

        if model_name == "attn_kv_cache_save":
            validate_cache_write_target_consistency(
                df,
                target_col=target_col,
                allow_repeated_measurements=True,
            )
        
        # Check cache
        operator_binding = self._build_operator_binding(model_name, df)
        model_hash = self._get_model_hash(
            model_name,
            df,
            feature_cols,
            target_col,
            operator_binding=operator_binding,
        )
        cached_model = self._load_model_from_cache(
            model_name,
            model_hash,
            feature_cols=feature_cols,
            target_col=target_col,
            operator_binding=operator_binding,
        )
        if cached_model:
            if len(feature_cols) > 1:
                cached_model._frontier_exact_lookup = build_exact_feature_lookup(
                    df, feature_cols, target_col
                )
            logger.info(f"Using cached model for {model_name}")
            return cached_model
        
        # Train new model
        logger.info(f"Training model {model_name} with {len(df)} samples")
        
        estimator, grid_search_params = self._create_estimator_and_params()
        
        cv = resolve_training_cv_splits(self.k_fold_cv_splits, len(df))
        
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=grid_search_params,
            scoring=self._get_scorer(),
            cv=cv,
            n_jobs=self.num_training_job_threads,
        )
        
        X, y = df[feature_cols], df[target_col]
        grid_search.fit(X, y)
        score = grid_search.score(X, y)
        
        logger.info(f"Trained model {model_name} with MAPE {-score:.2f}%")
        
        best_estimator = grid_search.best_estimator_
        attach_feature_domain(
            best_estimator,
            df,
            feature_cols,
            operator_name=model_name,
        )
        if len(feature_cols) > 1:
            best_estimator._frontier_exact_lookup = build_exact_feature_lookup(
                df, feature_cols, target_col
            )
        attach_model_cache_metadata(
            best_estimator,
            model_name=model_name,
            model_hash=model_hash,
            feature_names=feature_cols,
            target_col=target_col,
            feature_domain=best_estimator._frontier_feature_domain,
            operator_binding=operator_binding,
        )
        self._store_model_in_cache(
            model_name,
            model_hash,
            best_estimator,
            operator_binding=operator_binding,
        )

        return best_estimator
    
    def train(self) -> Dict[str, BaseEstimator]:
        """
        Train all models for this structure.
        
        Returns:
            Dictionary mapping model names to trained estimators
        """
        logger.info(f"Starting training for {self.__class__.__name__}")
        
        # Load dataset
        logger.info(f"Loading dataset from {self.dataset_path}")
        df = self._load_dataset()
        logger.info(f"Loaded {len(df)} rows after filtering")
        
        if len(df) == 0:
            raise ValueError(f"No data available after filtering. Check dataset path and filtering criteria.")
        
        # Train all models
        models = {}
        model_names = self._get_model_names()
        logger.info(f"Training {len(model_names)} models: {model_names}")
        
        for model_name in model_names:
            feature_cols = self._get_feature_cols(model_name)
            target_col = self._get_target_col(model_name)
            
            logger.info(f"\n--- Training {model_name} ---")
            logger.info(f"Features: {feature_cols}")
            logger.info(f"Target: {target_col}")
            
            models[model_name] = self._train_single_model(
                model_name=model_name,
                df=df,
                feature_cols=feature_cols,
                target_col=target_col
            )
        
        logger.info(f"\nTraining complete! Trained {len(models)} models")
        logger.info(f"Models saved to {self.output_dir}")
        
        return models
