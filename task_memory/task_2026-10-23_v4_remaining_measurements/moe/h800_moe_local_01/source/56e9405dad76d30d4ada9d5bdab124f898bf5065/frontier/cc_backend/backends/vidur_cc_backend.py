"""
Vidur ML-Based Collective Communication Backend.

This module provides an ML-based model for predicting collective communication
latencies using sklearn models trained on profiling data.

The Vidur backend uses trained machine learning models to predict communication
latencies based on profiling data from real hardware measurements.
"""

import hashlib
import os
import pickle
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fasteners import InterProcessReaderWriterLock
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from frontier.cc_backend.base_cc_backend import BaseCCBackend
from frontier.cc_backend.cc_backend_config import VidurCCBackendConfig
from frontier.cc_backend.cc_backend_factory import CCBackendFactory
from frontier.logger import init_logger
from frontier.types import CCBackendType, ClusterType

logger = init_logger(__name__)


class VidurCCBackend(BaseCCBackend):
    """
    ML-based collective communication backend using sklearn models.

    This backend uses trained machine learning models to predict
    communication latencies based on profiling data.

    Attributes:
        _models: Dictionary of trained sklearn models for each operation
        _predictions: Dictionary of prediction caches for each operation
        _all_reduce_input_file: Path to all-reduce profiling data
        _send_recv_input_file: Path to send-recv profiling data
    """

    def __init__(
        self,
        config: VidurCCBackendConfig,
        cluster_type: ClusterType,
        device_type: str,
        network_device: str,
        num_devices: int,
    ) -> None:
        """
        Initialize the Vidur ML-based CC backend.

        Args:
            config: Vidur backend configuration
            cluster_type: Type of cluster (MONOLITHIC, PREFILL, DECODE, etc.)
            device_type: Device type (e.g., "a100", "h100")
            network_device: Network device identifier (e.g., "a100_pairwise_nvlink")
            num_devices: Number of devices in the cluster
        """
        super().__init__(config, cluster_type, device_type, network_device, num_devices)

        self._config: VidurCCBackendConfig = config
        self._models: Dict[str, BaseEstimator] = {}
        self._predictions: Dict[str, Dict[Tuple, float]] = {}

        # Thread lock for protecting in-memory cache access
        self._cache_lock = threading.RLock()

        # Initialize file paths
        self._all_reduce_input_file = self._resolve_path(config.all_reduce_input_file)
        self._send_recv_input_file = self._resolve_path(config.send_recv_input_file)

        # Ensure cache directory exists
        self._cache_dir = config.cache_dir
        os.makedirs(self._cache_dir, exist_ok=True)

        # Analytical fallback parameters (config-driven)
        self._network_bandwidth_gbps = float(config.network_bandwidth_gbps)
        self._network_latency_us = float(config.network_latency_us)
        self._intra_node_bandwidth_gbps = float(config.intra_node_bandwidth_gbps)

        if self._network_bandwidth_gbps <= 0:
            raise ValueError(
                f"network_bandwidth_gbps must be positive, got {self._network_bandwidth_gbps}"
            )
        if self._intra_node_bandwidth_gbps <= 0:
            raise ValueError(
                f"intra_node_bandwidth_gbps must be positive, got {self._intra_node_bandwidth_gbps}"
            )
        if self._network_latency_us < 0:
            raise ValueError(
                f"network_latency_us must be non-negative, got {self._network_latency_us}"
            )

        # Initialize models
        self._initialize_models()

        logger.info(
            f"VidurCCBackend initialized with {len(self._models)} models, "
            f"all_reduce_file={self._all_reduce_input_file}, "
            f"send_recv_file={self._send_recv_input_file}, "
            f"fallback_network_bandwidth={self._network_bandwidth_gbps}Gbps, "
            f"fallback_intra_node_bandwidth={self._intra_node_bandwidth_gbps}Gbps, "
            f"fallback_latency={self._network_latency_us}us"
        )

    def _resolve_path(self, path_template: str) -> str:
        """
        Resolve path template with actual values.

        Args:
            path_template: Path template with placeholders like {profiling_data_dir}
                          and {NETWORK_DEVICE}

        Returns:
            Resolved path string
        """
        return path_template.format(
            profiling_data_dir=self._config.profiling_data_dir,
            NETWORK_DEVICE=self._network_device,
        )

    def _initialize_models(self) -> None:
        """
        Initialize ML models from cache or train new ones.

        This method loads profiling data and trains/loads models for
        all_reduce and send_recv operations.

        Thread-safe: Uses _cache_lock to protect in-memory cache updates.
        """
        # Load all_reduce model if profiling data exists
        if os.path.exists(self._all_reduce_input_file):
            try:
                all_reduce_df = self._load_all_reduce_df(self._all_reduce_input_file)
                if len(all_reduce_df) > 0:
                    all_reduce_df = self._get_all_reduce_df_with_derived_features(
                        all_reduce_df
                    )
                    model = self._train_model(
                        model_name="all_reduce",
                        df=all_reduce_df,
                        feature_cols=["num_tokens"],
                        target_col="time_stats.all_reduce.median",
                    )
                    predictions = self._generate_predictions("all_reduce", model)
                    # Thread-safe update of in-memory caches
                    with self._cache_lock:
                        self._models["all_reduce"] = model
                        self._predictions["all_reduce"] = predictions
                    logger.info(
                        f"Loaded all_reduce model with {len(all_reduce_df)} training samples"
                    )
                else:
                    logger.warning(
                        f"All-reduce profiling data is empty after filtering: {self._all_reduce_input_file}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load all_reduce model: {e}")
        else:
            logger.warning(
                f"All-reduce profiling data not found: {self._all_reduce_input_file}"
            )

        # Load send_recv model if profiling data exists
        if os.path.exists(self._send_recv_input_file):
            try:
                send_recv_df = self._load_send_recv_df(self._send_recv_input_file)
                if len(send_recv_df) > 0:
                    send_recv_df = self._get_send_recv_df_with_derived_features(
                        send_recv_df
                    )
                    model = self._train_model(
                        model_name="send_recv",
                        df=send_recv_df,
                        feature_cols=["num_tokens"],
                        target_col="time_stats.send_recv.median",
                    )
                    predictions = self._generate_predictions("send_recv", model)
                    # Thread-safe update of in-memory caches
                    with self._cache_lock:
                        self._models["send_recv"] = model
                        self._predictions["send_recv"] = predictions
                    logger.info(
                        f"Loaded send_recv model with {len(send_recv_df)} training samples"
                    )
                else:
                    logger.warning(
                        f"Send-recv profiling data is empty after filtering: {self._send_recv_input_file}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load send_recv model: {e}")
        else:
            logger.warning(
                f"Send-recv profiling data not found: {self._send_recv_input_file}"
            )

    # ========================================================================
    # Data Loading Methods (migrated from sklearn_execution_time_predictor)
    # ========================================================================

    def _read_input_file(self, file_path: str) -> pd.DataFrame:
        """
        Read profiling data from CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame with profiling data

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is corrupted or has invalid format
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Profiling data file not found: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            raise ValueError(f"Profiling data file is empty: {file_path}")
        except pd.errors.ParserError as e:
            raise ValueError(
                f"Profiling data file is corrupted or has invalid format: {file_path}. Error: {e}"
            )
        except Exception as e:
            raise ValueError(
                f"Failed to read profiling data file: {file_path}. Error: {e}"
            )

        if df.empty:
            logger.warning(f"Profiling data file contains no data: {file_path}")
            return df

        df = df.drop_duplicates()
        return df

    def _load_all_reduce_df(self, file_path: str) -> pd.DataFrame:
        """
        Load and filter all-reduce profiling data.

        Filters data based on:
        - num_workers matching tensor parallel size
        - devices_per_node matching tensor parallel size
        - collective type is 'all_reduce'

        Args:
            file_path: Path to all-reduce profiling CSV

        Returns:
            Filtered DataFrame
        """
        df = self._read_input_file(file_path)

        # Filter based on num_devices (tensor parallel size)
        filtered_df = df[
            (df["num_workers"] == self._num_devices)
            & (df["devices_per_node"] == self._num_devices)
            & (df["collective"] == "all_reduce")
        ]

        logger.debug(
            f"Loaded all_reduce data: {len(df)} total rows, "
            f"{len(filtered_df)} after filtering for num_devices={self._num_devices}"
        )

        return filtered_df

    def _load_send_recv_df(self, file_path: str) -> pd.DataFrame:
        """
        Load and filter send-recv profiling data.

        For multi-node setups, uses devices_per_node=1.
        For single-node setups, uses devices_per_node=2.

        Args:
            file_path: Path to send-recv profiling CSV

        Returns:
            Filtered DataFrame
        """
        df = self._read_input_file(file_path)

        # Determine devices_per_node based on configuration
        # For multi-node: devices_per_node=1 (inter-node communication)
        # For single-node: devices_per_node=2 (intra-node communication)
        devices_per_node = 2  # Default to intra-node

        filtered_df = df[
            (df["collective"] == "send_recv")
            & (df["devices_per_node"] == devices_per_node)
        ]

        logger.debug(
            f"Loaded send_recv data: {len(df)} total rows, "
            f"{len(filtered_df)} after filtering for devices_per_node={devices_per_node}"
        )

        return filtered_df

    def _get_all_reduce_df_with_derived_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add derived features to all-reduce DataFrame.

        Converts byte size to num_tokens for model training.
        Assumes FP16 format: 2 bytes per element.

        Args:
            df: Raw all-reduce DataFrame

        Returns:
            DataFrame with derived features
        """
        df_with_derived_features = df.copy()
        # Convert bytes to num_tokens
        # Assuming FP16: 2 bytes per element
        # For communication, size is typically embedding_dim * 2 bytes per token
        # We use a simplified conversion: num_tokens = size / 2
        df_with_derived_features["num_tokens"] = df_with_derived_features["size"] / 2
        return df_with_derived_features

    def _get_send_recv_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features to send-recv DataFrame.

        Converts byte size to num_tokens for model training.

        Args:
            df: Raw send-recv DataFrame

        Returns:
            DataFrame with derived features
        """
        df_with_derived_features = df.copy()
        # Convert bytes to num_tokens (same as all_reduce)
        df_with_derived_features["num_tokens"] = df_with_derived_features["size"] / 2
        return df_with_derived_features

    # ========================================================================
    # Model Training Methods (migrated from sklearn_execution_time_predictor)
    # ========================================================================

    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            MAPE value
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Handle zero true values separately to avoid division by zero
        zero_true_mask = y_true == 0
        non_zero_true_mask = ~zero_true_mask

        error = np.zeros_like(y_true, dtype=float)
        error[non_zero_true_mask] = (
            np.abs(
                (y_true[non_zero_true_mask] - y_pred[non_zero_true_mask])
                / y_true[non_zero_true_mask]
            )
            * 100
        )
        # For zero true values, if prediction is also 0, error is 0, else it is 100
        error[zero_true_mask] = np.where(y_pred[zero_true_mask] == 0, 0, 100)

        return np.mean(error)

    def _get_scorer(self) -> Any:
        """Get sklearn scorer for model training."""
        return make_scorer(
            VidurCCBackend.mean_absolute_percentage_error,
            greater_is_better=False,
        )

    def _get_model_hash(self, model_name: str, df: pd.DataFrame = None) -> str:
        """
        Generate a hash for model caching.

        Args:
            model_name: Name of the model
            df: Training DataFrame (optional)

        Returns:
            Hash string for cache identification
        """
        config_str = str(self._to_dict())

        if df is None:
            combined_str = f"{config_str}_{model_name}"
        else:
            df_hash_str = hashlib.md5(df.to_json().encode("utf-8")).hexdigest()
            combined_str = f"{config_str}_{model_name}_{df_hash_str}"

        return hashlib.md5(combined_str.encode("utf-8")).hexdigest()[0:8]

    def _to_dict(self) -> dict:
        """Convert configuration to dictionary for hashing."""
        return {
            "backend_type": "vidur",
            "cluster_type": str(self._cluster_type),
            "device_type": self._device_type,
            "network_device": self._network_device,
            "num_devices": self._num_devices,
            "k_fold_cv_splits": self._config.k_fold_cv_splits,
            "all_reduce_input_file": self._all_reduce_input_file,
            "send_recv_input_file": self._send_recv_input_file,
        }

    def _load_model_from_cache(
        self, model_name: str, model_hash: str
    ) -> Optional[BaseEstimator]:
        """
        Load a trained model from cache.

        Uses file locking for thread-safe access.

        Args:
            model_name: Name of the model
            model_hash: Hash for cache identification

        Returns:
            Cached model or None if not found or corrupted

        Note:
            If the cache file is corrupted, this method logs a warning
            and returns None, allowing the model to be retrained.
        """
        lock_file = f"{self._cache_dir}/{model_hash}_model_lock.file"
        try:
            with InterProcessReaderWriterLock(lock_file).read_lock():
                if self._config.no_cache:
                    return None

                cache_file = f"{self._cache_dir}/{model_name}_{model_hash}.pkl"
                if not os.path.exists(cache_file):
                    return None

                logger.debug(f"Found model {model_name} in cache: {cache_file}")
                try:
                    with open(cache_file, "rb") as f:
                        model = pickle.load(f)
                    return model
                except (pickle.UnpicklingError, EOFError, AttributeError) as e:
                    logger.warning(
                        f"Corrupted model cache file {cache_file}, will retrain: {e}"
                    )
                    return None
                except Exception as e:
                    logger.warning(f"Failed to load model from cache {cache_file}: {e}")
                    return None
        except Exception as e:
            logger.warning(f"Failed to acquire read lock for model cache: {e}")
            return None

    def _store_model_in_cache(
        self, model_name: str, model_hash: str, model: BaseEstimator
    ) -> None:
        """
        Store a trained model in cache.

        Uses file locking for thread-safe access.

        Args:
            model_name: Name of the model
            model_hash: Hash for cache identification
            model: Trained model to cache

        Note:
            If storing fails, this method logs a warning but does not raise
            an exception, as the model can still be used without caching.
        """
        lock_file = f"{self._cache_dir}/{model_hash}_model_lock.file"
        try:
            with InterProcessReaderWriterLock(lock_file).write_lock():
                cache_file = f"{self._cache_dir}/{model_name}_{model_hash}.pkl"
                try:
                    with open(cache_file, "wb") as f:
                        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.debug(f"Stored model {model_name} in cache: {cache_file}")
                except (IOError, OSError) as e:
                    logger.warning(
                        f"Failed to write model cache file {cache_file}: {e}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to serialize model to cache {cache_file}: {e}"
                    )
        except Exception as e:
            logger.warning(f"Failed to acquire write lock for model cache: {e}")

    def _get_estimator(self) -> BaseEstimator:
        """
        Get the sklearn estimator for model training.

        Returns:
            RandomForestRegressor instance
        """
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(random_state=42)

    def _get_grid_search_params(self) -> Dict[str, Any]:
        """
        Get grid search parameters for model training.

        Returns:
            Dictionary of parameter grids
        """
        return {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
        }

    def _train_model(
        self,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> BaseEstimator:
        """
        Train a sklearn model for communication prediction.

        Uses GridSearchCV for hyperparameter tuning and caches the trained model.

        Args:
            model_name: Name of the model
            df: Training DataFrame
            feature_cols: List of feature column names
            target_col: Target column name

        Returns:
            Trained sklearn model

        Raises:
            ValueError: If training data is empty
        """
        if len(df) == 0:
            raise ValueError(f"Training data for model {model_name} is empty")

        model_hash = self._get_model_hash(model_name, df)

        # Try to load from cache first
        cached_model = self._load_model_from_cache(model_name, model_hash)
        if cached_model is not None:
            logger.info(f"Loaded model {model_name} from cache")
            return cached_model

        logger.info(f"Training model {model_name} with {len(df)} samples")

        model = self._get_estimator()
        grid_search_params = self._get_grid_search_params()

        # Adjust CV splits based on data size
        cv = min(self._config.k_fold_cv_splits, len(df))
        if cv < 2:
            cv = 2

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid_search_params,
            scoring=self._get_scorer(),
            cv=cv,
            n_jobs=self._config.num_training_job_threads,
        )

        X, y = df[feature_cols], df[target_col]
        grid_search.fit(X, y)
        score = grid_search.score(X, y)

        logger.info(
            f"Trained model {model_name} with best params: {grid_search.best_params_}, "
            f"MAPE: {-score:.2f}%"
        )

        best_estimator = grid_search.best_estimator_
        # Attach model identity for prediction cache consistency
        setattr(best_estimator, "_frontier_model_hash", model_hash)

        self._store_model_in_cache(model_name, model_hash, best_estimator)

        return best_estimator

    # ========================================================================
    # Prediction Cache Methods
    # ========================================================================

    def _get_prediction_cache_hash(self, model_name: str, model: BaseEstimator) -> str:
        """
        Generate a hash for prediction cache.

        Args:
            model_name: Name of the model
            model: Trained model

        Returns:
            Hash string for prediction cache identification
        """
        config_hash = self._get_model_hash(model_name, df=None)

        # Use model hash for cache consistency
        model_identity = getattr(model, "_frontier_model_hash", None)
        if model_identity is None:
            try:
                model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
                model_identity = hashlib.md5(model_bytes).hexdigest()[0:8]
            except Exception:
                model_identity = "no_model_hash"

        combined = f"{config_hash}_{model_identity}"
        return hashlib.md5(combined.encode("utf-8")).hexdigest()[0:8]

    def _load_prediction_cache(
        self, model_name: str, prediction_hash: str
    ) -> Optional[Dict[Tuple, float]]:
        """
        Load prediction cache from disk.

        Args:
            model_name: Name of the model
            prediction_hash: Hash for cache identification

        Returns:
            Cached predictions or None if not found or corrupted

        Note:
            If the cache file is corrupted, this method logs a warning
            and returns None, allowing predictions to be regenerated.
        """
        lock_file = f"{self._cache_dir}/{prediction_hash}_prediction_lock.file"
        try:
            with InterProcessReaderWriterLock(lock_file).read_lock():
                if self._config.no_cache:
                    return None

                cache_file = (
                    f"{self._cache_dir}/{model_name}_{prediction_hash}_predictions.pkl"
                )
                if not os.path.exists(cache_file):
                    return None

                logger.debug(f"Found predictions for {model_name} in cache")
                try:
                    with open(cache_file, "rb") as f:
                        predictions = pickle.load(f)
                    return predictions
                except (pickle.UnpicklingError, EOFError, AttributeError) as e:
                    logger.warning(
                        f"Corrupted prediction cache file {cache_file}, will regenerate: {e}"
                    )
                    return None
                except Exception as e:
                    logger.warning(
                        f"Failed to load predictions from cache {cache_file}: {e}"
                    )
                    return None
        except Exception as e:
            logger.warning(f"Failed to acquire read lock for prediction cache: {e}")
            return None

    def _store_prediction_cache(
        self, model_name: str, prediction_hash: str, predictions: Dict[Tuple, float]
    ) -> None:
        """
        Store prediction cache to disk.

        Args:
            model_name: Name of the model
            prediction_hash: Hash for cache identification
            predictions: Dictionary of predictions to cache

        Note:
            If storing fails, this method logs a warning but does not raise
            an exception, as predictions can still be used without caching.
        """
        lock_file = f"{self._cache_dir}/{prediction_hash}_prediction_lock.file"
        try:
            with InterProcessReaderWriterLock(lock_file).write_lock():
                cache_file = (
                    f"{self._cache_dir}/{model_name}_{prediction_hash}_predictions.pkl"
                )
                try:
                    with open(cache_file, "wb") as f:
                        pickle.dump(predictions, f, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.debug(f"Stored predictions for {model_name} in cache")
                except (IOError, OSError) as e:
                    logger.warning(
                        f"Failed to write prediction cache file {cache_file}: {e}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to serialize predictions to cache {cache_file}: {e}"
                    )
        except Exception as e:
            logger.warning(f"Failed to acquire write lock for prediction cache: {e}")

    def _generate_predictions(
        self, model_name: str, model: BaseEstimator, max_tokens: int = 100000
    ) -> Dict[Tuple, float]:
        """
        Generate prediction cache for a model.

        Creates a lookup table for fast prediction during simulation.

        Args:
            model_name: Name of the model
            model: Trained sklearn model
            max_tokens: Maximum number of tokens to generate predictions for

        Returns:
            Dictionary mapping (num_tokens,) to predicted time
        """
        prediction_hash = self._get_prediction_cache_hash(model_name, model)

        # Try to load from cache first
        cached_predictions = self._load_prediction_cache(model_name, prediction_hash)
        if cached_predictions is not None:
            logger.info(f"Loaded predictions for {model_name} from cache")
            return cached_predictions

        logger.info(f"Generating predictions for {model_name}")

        # Generate predictions for a range of token counts
        num_token_range = np.arange(1, max_tokens + 1)
        X = pd.DataFrame({"num_tokens": num_token_range})

        predictions_array = model.predict(X)

        # Create lookup dictionary
        predictions = dict(zip([(int(x),) for x in num_token_range], predictions_array))

        # Store in cache
        self._store_prediction_cache(model_name, prediction_hash, predictions)

        # Also save as CSV for debugging
        X["prediction"] = predictions_array
        csv_file = f"{self._cache_dir}/{model_name}_{prediction_hash}_predictions.csv"
        X.to_csv(csv_file, index=False)

        return predictions

    # ========================================================================
    # Analytical Fallback Methods
    # ========================================================================

    def _get_fallback_bandwidth_gbps(self, num_devices: int) -> float:
        """Get fallback bandwidth for a collective based on participation scope."""
        # Heuristic: if the collective participants fit within this backend's device scope,
        # treat it as intra-node communication; otherwise use inter-node bandwidth.
        if num_devices <= self._num_devices:
            return self._intra_node_bandwidth_gbps
        return self._network_bandwidth_gbps

    def _get_fallback_latency_ms(self) -> float:
        """Get fallback latency in milliseconds."""
        return self._network_latency_us / 1000.0

    def _get_fallback_transfer_time_ms(
        self,
        effective_data_size_bytes: float,
        num_devices: int,
    ) -> float:
        """Convert effective data size to transfer time using fallback parameters."""
        bandwidth_gbps = self._get_fallback_bandwidth_gbps(num_devices)
        bandwidth_bytes_per_ms = (bandwidth_gbps * 1e9) / (8 * 1000)
        latency_ms = self._get_fallback_latency_ms()
        return max(0.0, latency_ms + (effective_data_size_bytes / bandwidth_bytes_per_ms))

    def _analytical_fallback_allreduce(
        self, data_size_bytes: int, num_devices: int
    ) -> float:
        """
        Fallback to analytical model for all-reduce prediction.

        Uses ring algorithm formula: time = latency + (2 * (n-1) / n * data_size / bandwidth)

        Args:
            data_size_bytes: Size of data in bytes
            num_devices: Number of participating devices

        Returns:
            Predicted time in milliseconds
        """
        if num_devices <= 1:
            return 0.0

        # Ring all-reduce data volume
        effective_data_size = 2 * (num_devices - 1) / num_devices * data_size_bytes
        return self._get_fallback_transfer_time_ms(
            effective_data_size_bytes=effective_data_size,
            num_devices=num_devices,
        )

    def _analytical_fallback_send_recv(self, data_size_bytes: int) -> float:
        """
        Fallback to analytical model for send-recv prediction.

        Uses simple formula: time = latency + (data_size / bandwidth)

        Args:
            data_size_bytes: Size of data in bytes

        Returns:
            Predicted time in milliseconds
        """
        return self._get_fallback_transfer_time_ms(
            effective_data_size_bytes=float(data_size_bytes),
            num_devices=2,
        )

    # ========================================================================
    # Prediction Methods (BaseCCBackend interface implementation)
    # ========================================================================

    def predict_allreduce(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict all-reduce communication time using ML model.

        Falls back to analytical model if ML model is not available.

        Thread-safe: Uses _cache_lock to protect in-memory cache reads.

        Args:
            data_size_bytes: Size of data in bytes
            num_devices: Number of participating devices
            cluster_type: Optional cluster type for context-aware prediction

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "allreduce")

        if num_devices <= 1:
            return 0.0

        # Thread-safe check if ML model is available
        with self._cache_lock:
            has_predictions = "all_reduce" in self._predictions
            predictions_cache = self._predictions.get("all_reduce", {})

        if not has_predictions:
            logger.debug("Using analytical fallback for allreduce")
            return self._analytical_fallback_allreduce(data_size_bytes, num_devices)

        # Convert bytes to num_tokens for lookup
        # Assuming FP16: 2 bytes per element
        num_tokens = max(1, data_size_bytes // 2)

        # Thread-safe lookup in prediction cache
        if (num_tokens,) in predictions_cache:
            result = predictions_cache[(num_tokens,)]
            logger.debug(
                f"predict_allreduce: data_size={data_size_bytes}, num_tokens={num_tokens}, "
                f"result={result:.6f} ms (ML model)"
            )
            return max(0.0, result)

        # Fallback to analytical if not in cache
        logger.debug(f"num_tokens={num_tokens} not in cache, using analytical fallback")
        return self._analytical_fallback_allreduce(data_size_bytes, num_devices)

    def predict_send_recv(
        self,
        data_size_bytes: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict point-to-point send/recv communication time using ML model.

        Falls back to analytical model if ML model is not available.

        Thread-safe: Uses _cache_lock to protect in-memory cache reads.

        Args:
            data_size_bytes: Size of data in bytes
            cluster_type: Optional cluster type for context-aware prediction

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_data_size(data_size_bytes)

        # Thread-safe check if ML model is available
        with self._cache_lock:
            has_predictions = "send_recv" in self._predictions
            predictions_cache = self._predictions.get("send_recv", {})

        if not has_predictions:
            logger.debug("Using analytical fallback for send_recv")
            return self._analytical_fallback_send_recv(data_size_bytes)

        # Convert bytes to num_tokens for lookup
        num_tokens = max(1, data_size_bytes // 2)

        # Thread-safe lookup in prediction cache
        if (num_tokens,) in predictions_cache:
            result = predictions_cache[(num_tokens,)]
            logger.debug(
                f"predict_send_recv: data_size={data_size_bytes}, num_tokens={num_tokens}, "
                f"result={result:.6f} ms (ML model)"
            )
            return max(0.0, result)

        # Fallback to analytical if not in cache
        logger.debug(f"num_tokens={num_tokens} not in cache, using analytical fallback")
        return self._analytical_fallback_send_recv(data_size_bytes)

    def predict_allgather(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict all-gather communication time.

        Currently uses analytical model as ML model is not trained for this operation.

        Args:
            data_size_bytes: Size of data per device in bytes
            num_devices: Number of participating devices
            cluster_type: Optional cluster type for context-aware prediction

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "allgather")

        if num_devices <= 1:
            return 0.0

        # Use analytical model (no ML model for allgather)
        # All-gather: total_data = n * data_size, effective = (n-1)/n * total
        total_data_size = data_size_bytes * num_devices
        effective_data_size = (num_devices - 1) / num_devices * total_data_size

        result = self._get_fallback_transfer_time_ms(
            effective_data_size_bytes=effective_data_size,
            num_devices=num_devices,
        )
        logger.debug(
            f"predict_allgather: data_size={data_size_bytes}, num_devices={num_devices}, "
            f"result={result:.6f} ms (analytical)"
        )
        return max(0.0, result)

    def predict_broadcast(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict broadcast communication time.

        Currently uses analytical model as ML model is not trained for this operation.

        Args:
            data_size_bytes: Size of data in bytes
            num_devices: Number of participating devices
            cluster_type: Optional cluster type for context-aware prediction

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "broadcast")

        if num_devices <= 1:
            return 0.0

        import math

        # Tree broadcast: log2(n) steps
        num_steps = math.ceil(math.log2(num_devices))

        single_step_time = self._get_fallback_transfer_time_ms(
            effective_data_size_bytes=float(data_size_bytes),
            num_devices=num_devices,
        )
        result = num_steps * single_step_time

        logger.debug(
            f"predict_broadcast: data_size={data_size_bytes}, num_devices={num_devices}, "
            f"num_steps={num_steps}, result={result:.6f} ms (analytical)"
        )
        return result

    def predict_reduce_scatter(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict reduce-scatter communication time.

        Currently uses analytical model as ML model is not trained for this operation.

        Args:
            data_size_bytes: Size of data in bytes
            num_devices: Number of participating devices
            cluster_type: Optional cluster type for context-aware prediction

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "reduce_scatter")

        if num_devices <= 1:
            return 0.0

        # Ring reduce-scatter: (n-1)/n * data_size
        effective_data_size = (num_devices - 1) / num_devices * data_size_bytes

        result = self._get_fallback_transfer_time_ms(
            effective_data_size_bytes=effective_data_size,
            num_devices=num_devices,
        )
        logger.debug(
            f"predict_reduce_scatter: data_size={data_size_bytes}, num_devices={num_devices}, "
            f"result={result:.6f} ms (analytical)"
        )
        return max(0.0, result)

    def predict_all_to_all(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict all-to-all communication time.

        Currently uses analytical model as ML model is not trained for this operation.

        Args:
            data_size_bytes: Total size of data in bytes
            num_devices: Number of participating devices
            cluster_type: Optional cluster type for context-aware prediction

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "all_to_all")

        if num_devices <= 1:
            return 0.0

        # All-to-all: (n-1) * data_per_device
        data_per_device = data_size_bytes // num_devices
        effective_data_size = (num_devices - 1) * data_per_device

        result = self._get_fallback_transfer_time_ms(
            effective_data_size_bytes=effective_data_size,
            num_devices=num_devices,
        )
        logger.debug(
            f"predict_all_to_all: data_size={data_size_bytes}, num_devices={num_devices}, "
            f"result={result:.6f} ms (analytical)"
        )
        return max(0.0, result)


# Register the backend with the factory
CCBackendFactory.register(CCBackendType.VIDUR, VidurCCBackend)
