"""The trained-model registry, its cache keys and the persistent cache.

A trained estimator is identified by the configuration that produced it, the
profiling rows it saw, its precision and measurement family, and the layer
contract it covers.  These methods compute that identity, store and look up
estimators under it, and read and write the on-disk cache.
"""

import hashlib
import json
import math
import os
import pandas as pd
import pickle
from pathlib import Path

from fasteners import InterProcessReaderWriterLock
from frontier.execution_time_predictor.cache_io import (
    atomic_json_dump,
    atomic_pickle_dump,
    dataset_fingerprint,
)
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

MODEL_BUNDLE_SCHEMA_VERSION = 1
MODEL_BUNDLE_IDENTITY_FIELDS = (
    "model_name", "model_architecture_profile", "device", "role", "topology",
    "graph_policy", "runtime_stack_signature", "physical_scope",
    "quant_signature", "evidence_kind", "coverage",
)


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

    def _load_model_from_cache(
        self,
        model_name: str,
        model_hash: str,
        *,
        expected_sha256: Optional[str] = None,
    ) -> BaseEstimator:
        with InterProcessReaderWriterLock(f"{self._cache_dir}/{model_hash}_model_lock.file").read_lock():
            cache_file = f"{self._cache_dir}/{model_name}_{model_hash}.pkl"
            if not os.path.exists(cache_file):
                return None
            if expected_sha256 is not None and dataset_fingerprint(cache_file) != expected_sha256:
                raise ValueError(f"Model bundle artifact SHA256 mismatch for {model_name!r}")
            logger.info(f"✓ Loaded pre-trained model '{model_name}' from cache (hash: {model_hash})")
            logger.info(f"  Cache file: {cache_file}")
            with open(cache_file, "rb") as stream:
                return pickle.load(stream)

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
        """Migrate missing lookup metadata once; a warm hit never republishes it."""
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

    @staticmethod
    def _canonical_measurement_identity(identity: Mapping[str, Any]) -> str:
        """Serialize explicit provenance without defaults or JSON coercions."""

        if not isinstance(identity, Mapping) or not identity:
            raise ValueError("measurement identity must be a nonempty mapping")
        if any(value is None or isinstance(value, (str, Mapping, list, tuple)) and not value
               for value in identity.values()):
            raise ValueError("measurement identity fields must be explicit and nonempty")

        def normalize(value, field):
            if isinstance(value, Mapping):
                result = {}
                for key, item in value.items():
                    if not isinstance(key, str) or not key.strip():
                        raise ValueError(f"measurement identity {field} requires nonempty string keys")
                    result[key] = normalize(item, f"{field}.{key}")
                return result
            if isinstance(value, (list, tuple)):
                return [normalize(item, f"{field}[{index}]") for index, item in enumerate(value)]
            if isinstance(value, str):
                if not value.strip():
                    raise ValueError(f"measurement identity {field} must be nonempty")
            elif type(value) is float:
                if not math.isfinite(value):
                    raise ValueError(f"measurement identity {field} must be finite JSON")
            elif type(value) not in (int, bool):
                raise ValueError(f"measurement identity {field} must be nonempty finite JSON")
            return value

        return json.dumps(
            normalize(identity, "identity"), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        )

    @staticmethod
    def _set_measurement_identity(model: BaseEstimator, identity: Mapping[str, Any]) -> str:
        """Attach the canonical measurement identity before atomic cache storage."""

        canonical = PredictionModelRegistry._canonical_measurement_identity(identity)
        attached = getattr(model, "_frontier_measurement_identity", None)
        if attached is not None and attached != canonical:
            raise ValueError("model measurement identity conflicts with the requested identity")
        model._frontier_measurement_identity = canonical
        return canonical

    @staticmethod
    def _validate_cached_measurement_identity(
        *, model_name: str, model: BaseEstimator, requested_identity: Mapping[str, Any],
    ) -> None:
        """Reject missing or mismatched provenance only on an explicit request."""

        requested = PredictionModelRegistry._canonical_measurement_identity(requested_identity)
        attached = getattr(model, "_frontier_measurement_identity", None)
        if attached is None:
            raise ValueError(f"Cached model {model_name!r} is missing measurement identity marker")
        if not isinstance(attached, str) or attached != requested:
            raise ValueError(f"Cached model {model_name!r} measurement identity mismatch")

    @staticmethod
    def _model_bundle_identity(identity: Mapping[str, Any]) -> Dict[str, Any]:
        canonical = PredictionModelRegistry._canonical_measurement_identity(identity)
        normalized = json.loads(canonical)
        for field in MODEL_BUNDLE_IDENTITY_FIELDS:
            if field not in normalized or normalized[field] in ("", [], {}):
                raise ValueError(f"Model bundle identity requires explicit nonempty {field}")
        return normalized

    def _bundle_model_path(self, model_name: str, model_hash: str) -> Path:
        """Resolve only existing-cache artifact names, never external paths."""

        for label, value in (("model_name", model_name), ("model_hash", model_hash)):
            if (
                not isinstance(value, str) or not value.strip() or value in (".", "..")
                or "/" in value or "\\" in value or "\x00" in value
            ):
                raise ValueError(f"Model bundle {label} must be a cache filename component")
        cache = Path(self._cache_dir).resolve()
        artifact = cache / f"{model_name}_{model_hash}.pkl"
        if artifact.resolve().parent != cache:
            raise ValueError(f"Model bundle artifact is outside existing cache: {artifact}")
        return artifact

    @staticmethod
    def _bundle_model_metadata(model_name: str, model: BaseEstimator) -> Dict[str, Any]:
        """Read persisted estimator metadata; never infer missing model contracts."""

        model_hash = getattr(model, "_frontier_model_hash", None)
        features = getattr(model, "_frontier_feature_names", None)
        target = getattr(model, "_frontier_target_col", None)
        marker = getattr(model, "_frontier_measurement_identity", None)
        if not isinstance(model_hash, str) or not model_hash.strip():
            raise ValueError(f"Model bundle {model_name!r} requires _frontier_model_hash")
        if (
            not isinstance(features, (list, tuple)) or not features
            or any(not isinstance(name, str) or not name.strip() for name in features)
            or len(set(features)) != len(features)
        ):
            raise ValueError(f"Model bundle {model_name!r} requires _frontier_feature_names")
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"Model bundle {model_name!r} requires _frontier_target_col")
        if not isinstance(marker, str):
            raise ValueError(f"Model bundle {model_name!r} is missing measurement identity marker")
        try:
            measurement_identity = json.loads(marker)
        except ValueError as exc:
            raise ValueError(f"Model bundle {model_name!r} has invalid measurement identity marker") from exc
        PredictionModelRegistry._validate_cached_measurement_identity(
            model_name=model_name, model=model, requested_identity=measurement_identity,
        )
        typed_identity = getattr(model, "_frontier_layer_cache_identity", None)
        if typed_identity is not None and (not isinstance(typed_identity, str) or not typed_identity):
            raise ValueError(f"Model bundle {model_name!r} has invalid typed identity marker")
        return {
            "model_hash": model_hash, "feature_names": list(features), "target_column": target,
            "typed_identity": typed_identity, "measurement_identity": measurement_identity,
        }

    @staticmethod
    def _model_bundle_id(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_bundle_measurement_identity(model_name, identity, measurement_identity) -> None:
        """Do not let a bundle relabel producer facts shared with its estimators."""

        for field in identity.keys() & measurement_identity.keys():
            declared = PredictionModelRegistry._canonical_measurement_identity({field: identity[field]})
            measured = PredictionModelRegistry._canonical_measurement_identity({field: measurement_identity[field]})
            if declared != measured:
                raise ValueError(f"Model bundle measurement identity mismatch for {model_name!r}: {field}")

    def publish_model_bundle(
        self, identity: Mapping[str, Any], models: Mapping[str, BaseEstimator],
    ) -> Dict[str, Any]:
        """Snapshot existing cache artifacts; publication is not native admission.

        The content-addressed manifest is written last. Estimator names, hashes,
        files and legacy lookup behavior are unchanged; no estimator is copied.
        The returned ``path`` locates the manifest and is not part of its content.
        """

        normalized_identity = self._model_bundle_identity(identity)
        if not isinstance(models, Mapping) or not models:
            raise ValueError("Model bundle requires a nonempty model mapping")
        artifacts = {}
        for model_name, model in models.items():
            metadata = self._bundle_model_metadata(model_name, model)
            self._validate_bundle_measurement_identity(
                model_name, normalized_identity, metadata["measurement_identity"],
            )
            path = self._bundle_model_path(model_name, metadata["model_hash"])
            sha256 = dataset_fingerprint(path)
            cached = self._load_model_from_cache(
                model_name, metadata["model_hash"], expected_sha256=sha256,
            )
            if cached is not None:
                self._validate_cached_measurement_identity(
                    model_name=model_name, model=cached,
                    requested_identity=metadata["measurement_identity"],
                )
            if cached is None or self._bundle_model_metadata(model_name, cached) != metadata:
                raise ValueError(f"Model bundle persisted metadata mismatch for {model_name!r}")
            artifacts[model_name] = {**metadata, "artifact": path.name, "sha256": sha256}
        payload = {
            "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
            "identity": normalized_identity, "models": artifacts,
        }
        bundle_id = self._model_bundle_id(payload)
        manifest = {**payload, "bundle_id": bundle_id}
        path = Path(self._cache_dir).resolve() / f"bundle_{bundle_id}.json"
        expected_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        with InterProcessReaderWriterLock(f"{path}.lock").write_lock():
            if path.exists():
                if path.read_bytes() != expected_bytes:
                    raise ValueError(f"Model bundle has conflicting existing content: {path}")
            else:
                atomic_json_dump(manifest, path)
        return {**manifest, "path": str(path)}

    def load_model_bundle(
        self, path: str | os.PathLike[str], expected_identity: Mapping[str, Any],
    ) -> Dict[str, BaseEstimator]:
        """Validate the complete snapshot before returning any cached estimator."""

        requested = self._model_bundle_identity(expected_identity)
        cache = Path(self._cache_dir).resolve()
        path = Path(path)
        if not path.is_absolute() and path.parent == Path("."):
            path = cache / path
        if path.resolve().parent != cache:
            raise ValueError("Model bundle path must be inside the existing cache")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        fields = {"schema_version", "identity", "models", "bundle_id"}
        if not isinstance(manifest, dict) or set(manifest) != fields:
            raise ValueError("Model bundle has invalid manifest fields")
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != MODEL_BUNDLE_SCHEMA_VERSION:
            raise ValueError("Model bundle has unsupported schema_version")
        if self._canonical_measurement_identity(manifest["identity"]) != self._canonical_measurement_identity(requested):
            raise ValueError("Model bundle requested identity mismatch")
        payload = {key: value for key, value in manifest.items() if key != "bundle_id"}
        bundle_id = self._model_bundle_id(payload)
        if manifest["bundle_id"] != bundle_id or path.name != f"bundle_{bundle_id}.json":
            raise ValueError("Model bundle self identity mismatch")
        entries = manifest["models"]
        if not isinstance(entries, dict) or not entries:
            raise ValueError("Model bundle requires nonempty models")
        metadata_fields = {
            "model_hash", "feature_names", "target_column", "typed_identity", "measurement_identity",
        }
        # Check every digest before unpickling even the first estimator. Recheck
        # under its existing cache read lock to exclude concurrent replacements.
        for name, entry in entries.items():
            if not isinstance(entry, dict) or set(entry) != metadata_fields | {"artifact", "sha256"}:
                raise ValueError(f"Model bundle has invalid metadata for {name!r}")
            artifact = self._bundle_model_path(name, entry["model_hash"])
            if entry["artifact"] != artifact.name:
                raise ValueError(f"Model bundle artifact path mismatch for {name!r}")
            if dataset_fingerprint(artifact) != entry["sha256"]:
                raise ValueError(f"Model bundle artifact SHA256 mismatch for {name!r}")
        models = {}
        for name, entry in entries.items():
            model = self._load_model_from_cache(
                name, entry["model_hash"], expected_sha256=entry["sha256"],
            )
            if model is None:
                raise ValueError(f"Model bundle artifact is missing for {name!r}")
            self._validate_cached_measurement_identity(
                model_name=name, model=model, requested_identity=entry["measurement_identity"],
            )
            self._validate_bundle_measurement_identity(name, requested, entry["measurement_identity"])
            if self._bundle_model_metadata(name, model) != {key: entry[key] for key in metadata_fields}:
                raise ValueError(f"Model bundle artifact metadata mismatch for {name!r}")
            models[name] = model
        return models
