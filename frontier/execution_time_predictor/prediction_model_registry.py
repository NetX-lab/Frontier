"""The trained-model registry, its cache keys and the persistent cache.

A trained estimator is identified by the configuration that produced it, the
profiling rows it saw, its precision and measurement family, and the layer
contract it covers.  These methods compute that identity, store and look up
estimators under it, and read and write the on-disk cache.
"""

import hashlib
import os
import pandas as pd
import pickle

from fasteners import InterProcessReaderWriterLock
from frontier.execution_time_predictor.cache_io import atomic_pickle_dump
from frontier.execution_time_predictor.prediction_model_identity import (
    MIGRATION_HELP_COMMAND,
    _build_exact_feature_lookup,
    _resolve_model_architecture_profile,
    _resolve_profile_typed_family_for_query,
    _serialize_selected_layer_cache_identity,
)
from frontier.logger import init_logger
from frontier.model_architectures import LayerKind, ResolvedLayerContract
from frontier.moe_gating_runtime import get_moe_gating_base_model_name
from frontier.types import ClusterType, MeasurementType
from sklearn.base import BaseEstimator
from typing import Any, Dict, List, Mapping, Optional, Tuple


logger = init_logger(__name__)


class PredictionModelRegistry:
    """Trained-model identity, in-memory registry and persistent cache."""

    def _get_hash_relevant_config(self, config) -> Dict[str, Any]:
        """
        Extract only the configuration parameters that affect model performance.

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
            'cpu_overhead_kernel_only_input_file': getattr(
                config,
                'cpu_overhead_kernel_only_input_file',
                config.cpu_overhead_input_file,
            ),

            # Category 2: Prediction range parameters
            'kv_cache_prediction_granularity': config.kv_cache_prediction_granularity,
            'prediction_max_prefill_chunk_size': config.prediction_max_prefill_chunk_size,
            'prediction_max_batch_size': config.prediction_max_batch_size,
            'prediction_max_tokens_per_request': config.prediction_max_tokens_per_request,

            # Category 3: Performance adjustment parameters
            'attention_decode_batching_overhead_fraction': config.attention_decode_batching_overhead_fraction,
            'attention_prefill_batching_overhead_fraction': config.attention_prefill_batching_overhead_fraction,
            'attn_pre_proj_calibration_scale': config.attn_pre_proj_calibration_scale,
            'prefill_phase_attn_pre_proj_calibration_scale': config.prefill_phase_attn_pre_proj_calibration_scale,
            'attn_post_proj_calibration_scale': config.attn_post_proj_calibration_scale,
            'prefill_phase_attn_post_proj_calibration_scale': config.prefill_phase_attn_post_proj_calibration_scale,
            'attn_decode_calibration_scale': config.attn_decode_calibration_scale,
            'attn_decode_in_mixed_calibration_scale': config.attn_decode_in_mixed_calibration_scale,
            'late_decode_attn_decode_calibration_scale': config.late_decode_attn_decode_calibration_scale,
            'attn_kv_cache_save_calibration_scale': config.attn_kv_cache_save_calibration_scale,
            'prefill_phase_attn_kv_cache_save_calibration_scale': config.prefill_phase_attn_kv_cache_save_calibration_scale,
            'mlp_up_proj_calibration_scale': config.mlp_up_proj_calibration_scale,
            'prefill_phase_mlp_up_proj_calibration_scale': config.prefill_phase_mlp_up_proj_calibration_scale,
            'mlp_down_proj_calibration_scale': config.mlp_down_proj_calibration_scale,
            'decode_phase_mlp_down_proj_calibration_scale': config.decode_phase_mlp_down_proj_calibration_scale,
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
        execution_time_predictor_config,
        profiling_precision: str,
        measurement_type: MeasurementType,
        layer_contract: Optional[ResolvedLayerContract] = None,
    ) -> str:
        """
        Calculate hash for model caching based on configuration and data.

        Hash is calculated from:
        1. Hash-relevant configuration parameters (excluding runtime/training process params)
        2. Model name
        3. DataFrame content hash

        This ensures that only changes to parameters that affect model performance
        will invalidate the cache.
        """
        # Extract only hash-relevant parameters
        hash_relevant_config = self._get_hash_relevant_config(execution_time_predictor_config)
        config_str = str(sorted(hash_relevant_config.items()))  # Sort for deterministic ordering

        # Calculate DataFrame hash
        df_hash_str = hashlib.md5(df.to_json().encode("utf-8")).hexdigest()

        selected_identity = _serialize_selected_layer_cache_identity(layer_contract)
        contract_component = (
            f"_{selected_identity}" if selected_identity is not None else ""
        )

        # Combine all components.  The selected semantic domain is part of a
        # typed key; physical layer occurrence is intentionally absent.
        combined_str = (
            f"{config_str}_{model_name}_{df_hash_str}_{profiling_precision}_"
            f"{measurement_type.value}{contract_component}"
        )
        hash_value = hashlib.md5(combined_str.encode("utf-8")).hexdigest()[0:8]

        # Debug output for hash calculation
        if model_name == "attn_pre_proj":
            logger.info(f"[DEBUG] Hash calculation for {model_name}:")
            logger.info(f"  - DataFrame shape: {df.shape}")
            logger.info(f"  - DataFrame hash: {df_hash_str[:16]}...")
            logger.info(f"  - Hash-relevant config keys: {sorted(hash_relevant_config.keys())}")
            logger.info(f"  - Final hash: {hash_value}")

        return hash_value

    def _get_profiling_precision_from_df(self, df: pd.DataFrame) -> str:
        """Extract profiling precision from DataFrame.

        FAIL-FAST: Raises ValueError if profiling_precision column is missing or invalid.
        This enforces strict metadata requirements and prevents silent fallbacks.
        """
        if "profiling_precision" not in df.columns:
            raise ValueError(
                "profiling_precision column is missing from profiling data. "
                f"Run '{MIGRATION_HELP_COMMAND}' to add required metadata columns to legacy CSV files."
            )

        precision_values = df["profiling_precision"].dropna().unique().tolist()
        if not precision_values:
            raise ValueError("profiling_precision column is empty")
        if len(precision_values) > 1:
            raise ValueError(
                f"Multiple profiling_precision values found: {precision_values}"
            )
        return str(precision_values[0]).upper()

    def _get_measurement_type_from_df(self, df: pd.DataFrame) -> MeasurementType:
        if "measurement_type" not in df.columns:
            raise ValueError(
                "measurement_type column is missing from profiling data. "
                f"Run '{MIGRATION_HELP_COMMAND}' to add required metadata columns to legacy CSV files."
            )

        measurement_values = df["measurement_type"].dropna().unique().tolist()
        if not measurement_values:
            raise ValueError("measurement_type column is empty")
        if len(measurement_values) > 1:
            raise ValueError(
                f"Multiple measurement_type values found: {measurement_values}"
            )
        return MeasurementType.from_string(str(measurement_values[0]))

    def _validate_active_measurement_type(self, df: pd.DataFrame) -> MeasurementType:
        measurement_type = self._get_measurement_type_from_df(df)
        if measurement_type != self._active_measurement_type:
            raise ValueError(
                f"measurement_type mismatch: expected {self._active_measurement_type.value} "
                f"but found {measurement_type.value}."
            )
        return measurement_type

    @staticmethod

    def _validate_cached_layer_cache_identity(
        *,
        model_name: str,
        model: BaseEstimator,
        requested_identity: str,
    ) -> None:
        """Reject a typed cache entry whose selected domain does not match."""

        cached_identity = getattr(model, "_frontier_layer_cache_identity", None)
        if cached_identity is None:
            raise ValueError(
                f"Cached model {model_name!r} is missing selected layer cache identity"
            )
        if not isinstance(cached_identity, str):
            raise ValueError(
                f"Cached model {model_name!r} has an invalid selected layer cache "
                f"identity of type {type(cached_identity).__name__}"
            )
        if cached_identity != requested_identity:
            raise ValueError(
                f"Cached model {model_name!r} selected layer cache identity mismatch: "
                f"cached={cached_identity!r}, requested={requested_identity!r}"
            )

    @staticmethod

    def _model_contract_identity(
        model: BaseEstimator,
        layer_contract: Optional[ResolvedLayerContract],
    ) -> Optional[str]:
        """Attach and return the selected semantic identity for a model."""

        requested_identity = _serialize_selected_layer_cache_identity(layer_contract)
        attached_identity = getattr(model, "_frontier_layer_cache_identity", None)
        if attached_identity is not None and not isinstance(attached_identity, str):
            raise TypeError(
                "_frontier_layer_cache_identity must be a string when present"
            )
        if (
            requested_identity is not None
            and attached_identity is not None
            and requested_identity != attached_identity
        ):
            raise ValueError(
                "model selected layer cache identity conflicts with the requested "
                "contract"
            )
        identity = requested_identity or attached_identity
        if identity is not None:
            setattr(model, "_frontier_layer_cache_identity", identity)
        return identity

    def _contract_model_registry(
        self, family_name: str
    ) -> Dict[Tuple[str, Optional[str]], BaseEstimator]:
        registry_attr = {
            "eager": "_trained_models_eager_by_contract",
            "device_event": "_trained_models_device_event_by_contract",
            "kernel_only": "_trained_models_kernel_only_by_contract",
        }.get(family_name)
        if registry_attr is None:
            raise ValueError(f"Unsupported family_name={family_name!r}")
        return getattr(self, registry_attr)

    def _contract_precision_registry(
        self, family_name: str
    ) -> Dict[str, Dict[Tuple[str, Optional[str]], BaseEstimator]]:
        registry_attr = {
            "eager": "_models_by_precision_eager_by_contract",
            "device_event": "_models_by_precision_device_event_by_contract",
            "kernel_only": "_models_by_precision_kernel_only_by_contract",
        }.get(family_name)
        if registry_attr is None:
            raise ValueError(f"Unsupported family_name={family_name!r}")
        return getattr(self, registry_attr)

    def _legacy_model_registry(self, family_name: str) -> Dict[str, BaseEstimator]:
        registry_attr = {
            "eager": "_trained_models_eager",
            "device_event": "_trained_models_device_event",
            "kernel_only": "_trained_models_kernel_only",
        }.get(family_name)
        if registry_attr is None:
            raise ValueError(f"Unsupported family_name={family_name!r}")
        return getattr(self, registry_attr)

    def _legacy_precision_registry(
        self, family_name: str
    ) -> Dict[str, Dict[str, BaseEstimator]]:
        registry_attr = {
            "eager": "_models_by_precision_eager",
            "device_event": "_models_by_precision_device_event",
            "kernel_only": "_models_by_precision_kernel_only",
        }.get(family_name)
        if registry_attr is None:
            raise ValueError(f"Unsupported family_name={family_name!r}")
        return getattr(self, registry_attr)

    def _legacy_precision_bucket(
        self, family_name: str, precision_key: str
    ) -> Dict[str, BaseEstimator]:
        if not isinstance(precision_key, str) or not precision_key:
            raise ValueError(
                f"precision_key must be a non-empty string, got {precision_key!r}"
            )
        registry = self._legacy_precision_registry(family_name)
        canonical_key = precision_key.upper()
        bucket = registry.get(canonical_key)
        if bucket is not None:
            return bucket
        for stored_key, stored_bucket in registry.items():
            if str(stored_key).upper() == canonical_key:
                return stored_bucket
        return {}

    def _store_model_precision(
        self,
        model_name: str,
        precision: str,
        model: BaseEstimator,
        layer_contract: Optional[ResolvedLayerContract] = None,
    ) -> None:
        if not isinstance(precision, str) or not precision.strip():
            raise ValueError(f"precision must be a non-empty string, got {precision!r}")
        precision_key = precision.upper()
        family_name = self._measurement_family_name(self._active_measurement_type)
        identity = self._model_contract_identity(model, layer_contract)
        if identity is None:
            self._legacy_model_registry(family_name)[model_name] = model
            self._legacy_precision_registry(family_name).setdefault(
                precision_key, {}
            )[model_name] = model
            return

        model_key = (model_name, identity)
        self._contract_model_registry(family_name)[model_key] = model
        self._contract_precision_registry(family_name).setdefault(
            precision_key, {}
        )[model_key] = model

    def _get_family_model(
        self,
        family_name: str,
        model_name: str,
        *,
        precision_key: Optional[str] = None,
        requested_identity: Optional[str] = None,
    ) -> Optional[BaseEstimator]:
        if precision_key is not None:
            precision_key = precision_key.upper()
            source = self._contract_precision_registry(family_name).get(
                precision_key, {}
            )
        else:
            source = self._contract_model_registry(family_name)

        typed_candidates = {
            identity: candidate
            for (candidate_name, identity), candidate in source.items()
            if candidate_name == model_name
        }
        legacy = (
            self._legacy_precision_bucket(family_name, precision_key).get(model_name)
            if precision_key is not None
            else self._legacy_model_registry(family_name).get(model_name)
        )
        legacy_identity = (
            getattr(legacy, "_frontier_layer_cache_identity", None)
            if legacy is not None
            else None
        )

        if requested_identity is not None:
            model = typed_candidates.get(requested_identity)
            if model is not None:
                return model
            if legacy is not None and legacy_identity == requested_identity:
                return legacy
            return None

        if len(typed_candidates) > 1:
            identities = sorted(
                "<legacy>" if identity is None else identity
                for identity in typed_candidates
            )
            raise ValueError(
                f"Model '{model_name}' has multiple layer contracts; provide "
                f"layer_contract. Available identities: {identities}"
            )
        if len(typed_candidates) == 1:
            typed_identity = next(iter(typed_candidates))
            if legacy is not None and legacy_identity != typed_identity:
                raise ValueError(
                    f"Model '{model_name}' has multiple layer contracts; provide "
                    f"layer_contract. Available identities: "
                    f"[{typed_identity!r}, {legacy_identity or '<legacy>'!r}]"
                )
            return next(iter(typed_candidates.values()))
        return legacy

    def _resolve_cluster_model_contract(
        self,
        cluster_type: Optional[ClusterType],
        model_name: str,
    ) -> Optional[ResolvedLayerContract]:
        """Resolve the typed domain requested by one cluster view."""

        if cluster_type is None:
            return None
        cluster_configs = getattr(self, "_cluster_configs", None) or {}
        cluster_config = cluster_configs.get(cluster_type)
        if cluster_config is None:
            return None
        replica_config = getattr(cluster_config, "replica_config", None)
        model_config = getattr(replica_config, "model_config", None)
        if replica_config is None or model_config is None:
            return None
        architecture_profile = _resolve_model_architecture_profile(model_config)
        if architecture_profile is None:
            return None
        base_name = get_moe_gating_base_model_name(model_name)
        typed_family = _resolve_profile_typed_family_for_query(
            architecture_profile, base_name
        )
        if typed_family is None:
            return None
        _, layer_kind = typed_family
        return self._resolve_typed_layer_contract(
            base_name,
            cluster_type,
            replica_config,
            is_moe_model=layer_kind is not LayerKind.DENSE,
        )

    def _is_ffn_typed_model_for_cluster(
        self,
        cluster_type: Optional[ClusterType],
        model_name: str,
    ) -> bool:
        """Return whether a model belongs to an FFN domain excluded by a view."""

        if cluster_type != ClusterType.DECODE_ATTN:
            return False
        cluster_config = self._cluster_configs.get(cluster_type)
        replica_config = getattr(cluster_config, "replica_config", None)
        model_config = getattr(replica_config, "model_config", None)
        architecture_profile = _resolve_model_architecture_profile(model_config)
        if architecture_profile is None:
            return False
        base_name = get_moe_gating_base_model_name(model_name)
        return _resolve_profile_typed_family_for_query(
            architecture_profile, base_name
        ) is not None

    def _models_view_for_family(
        self,
        family_name: str,
        cluster_type: Optional[ClusterType] = None,
    ) -> Dict[str, BaseEstimator]:
        """Project one measurement family's canonical registries to model names."""

        names = set(self._legacy_model_registry(family_name))
        names.update(
            model_name
            for model_name, _identity in self._contract_model_registry(family_name)
        )
        models: Dict[str, BaseEstimator] = {}
        for model_name in sorted(names):
            if self._is_ffn_typed_model_for_cluster(cluster_type, model_name):
                continue
            contract = self._resolve_cluster_model_contract(cluster_type, model_name)
            identity = _serialize_selected_layer_cache_identity(contract)
            model = self._get_family_model(
                family_name,
                model_name,
                requested_identity=identity,
            )
            if identity is not None and model is None:
                raise ValueError(
                    f"No trained model for {model_name!r} matches the typed contract "
                    f"requested by cluster {cluster_type!r}: {identity}"
                )
            if model is not None:
                models[model_name] = model
        return models

    def get_model(
        self,
        model_name: str,
        precision: Optional[str] = None,
        layer_contract: Optional[ResolvedLayerContract] = None,
    ) -> Optional[BaseEstimator]:
        """Get a model by name, precision, and optional typed contract."""

        if self._all_dummy_mode:
            return None
        requested_identity = _serialize_selected_layer_cache_identity(layer_contract)
        precision_key = precision.upper() if precision else None
        for family_name in ("eager", "device_event", "kernel_only"):
            model = self._get_family_model(
                family_name,
                model_name,
                precision_key=precision_key,
                requested_identity=requested_identity,
            )
            if model is not None:
                return model

        if precision_key is not None:
            available_precisions = sorted(
                {
                    str(value).upper()
                    for value in self._contract_precision_registry("eager")
                }
                | {
                    str(value).upper()
                    for value in self._contract_precision_registry("device_event")
                }
                | {
                    str(value).upper()
                    for value in self._contract_precision_registry("kernel_only")
                }
                | {
                    str(value).upper()
                    for value in self._legacy_precision_registry("eager")
                }
                | {
                    str(value).upper()
                    for value in self._legacy_precision_registry("device_event")
                }
                | {
                    str(value).upper()
                    for value in self._legacy_precision_registry("kernel_only")
                }
            )
            raise ValueError(
                f"Model '{model_name}' not available for precision '{precision_key}'. "
                f"Available precisions: {available_precisions}. "
                "Ensure profiling data matches the requested precision."
            )
        return None

    def _load_model_from_cache(self, model_name: str, model_hash: str) -> BaseEstimator:
        with InterProcessReaderWriterLock(f"{self._cache_dir}/{model_hash}_model_lock.file").read_lock():
            cache_file = f"{self._cache_dir}/{model_name}_{model_hash}.pkl"
            if not os.path.exists(cache_file):
                return None
            logger.info(f"✓ Loaded pre-trained model '{model_name}' from cache (hash: {model_hash})")
            logger.info(f"  Cache file: {cache_file}")
            return pickle.load(open(cache_file, "rb"))

    def _store_model_in_cache(self, model_name: str, model_hash: str, model: BaseEstimator) -> None:
        with InterProcessReaderWriterLock(f"{self._cache_dir}/{model_hash}_model_lock.file").write_lock():
            cache_file = f"{self._cache_dir}/{model_name}_{model_hash}.pkl"
            atomic_pickle_dump(model, cache_file)
            logger.info(f"✓ Saved trained model '{model_name}' to cache (hash: {model_hash})")
            logger.info(f"  Cache file: {cache_file}")

    def _ensure_exact_lookup_metadata(
        self,
        *,
        model_name: str,
        model_hash: str,
        model: BaseEstimator,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> None:
        """Persist exact measured rows before an on-demand model cache is stored."""
        if hasattr(model, "_frontier_exact_lookup"):
            exact_lookup = getattr(model, "_frontier_exact_lookup")
            if not isinstance(exact_lookup, Mapping):
                raise ValueError(
                    f"Exact lookup metadata for {model_name} must be a mapping"
                )
            return

        setattr(
            model,
            "_frontier_exact_lookup",
            _build_exact_feature_lookup(df, feature_cols, target_col),
        )
        self._store_model_in_cache(model_name, model_hash, model)
