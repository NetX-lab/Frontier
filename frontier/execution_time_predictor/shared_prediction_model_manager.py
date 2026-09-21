import hashlib
import json
import os
import pickle
from itertools import product
from typing import Dict, Set, List, Any, Tuple, Optional, Mapping, Iterable, cast

import numpy as np
import pandas as pd
from fasteners import InterProcessReaderWriterLock
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from frontier.attention.families import (
    DENSE_ATTENTION_FAMILY,
    LATENT_MLA_ATTENTION_FAMILY,
)
from frontier.attention.model_binding import resolve_runtime_attention_family
from frontier.attention.ops import AttentionOperatorRole
from frontier.attention.string_coercion import (
    coerce_truthy_bool,
    coerce_truthy_int,
)
from frontier.attention.profiling_mapping import (
    get_enabled_predictor_median_column_by_role,
    get_enabled_predictor_median_columns,
    get_enabled_predictor_metric_name_by_role,
    get_enabled_predictor_metric_names,
    get_enabled_shared_predictor_feature_columns,
    validate_attention_profiling_dataframe,
)
from frontier.config import MetricsConfig, ClusterConfig, global_vars
from frontier.types import ClusterType, CCBackendType, MeasurementType
from frontier.execution_time_predictor.attention_tp_policy import (
    resolve_effective_attention_tp_size,
)
from frontier.execution_time_predictor.cache_io import atomic_pickle_dump
from frontier.execution_time_predictor.measurement_input_paths import (
    substitute_input_path,
    resolve_measurement_input_paths,
    resolve_training_file_paths,
    resolve_event_measurement_type,
)
from frontier.execution_time_predictor.attention_dataset_contract import (
    enforce_mixed_attention_input_contract,
)
from frontier.operators.binding import resolve_operator_query_tp_mode
from frontier.operators.typed_contracts import (
    TYPED_OPERATOR_CONTRACTS_COLUMN,
    matches_resolved_layer_contract,
    validate_typed_operator_contracts,
)
from frontier.logger import init_logger
from frontier.execution_time_predictor.profiling_metadata import (
    infer_single_runtime_model_config,
    infer_single_runtime_profile,
    validate_model_architecture_profile,
)
from frontier.model_architectures import (
    LayerKind,
    ModelArchitectureProfile,
    ResolvedLayerContract,
    get_model_architecture_profile,
)
from frontier.moe_gating_runtime import (
    DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
    PrefillHotRowsUnavailableError,
    PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT,
    filter_moe_gating_rows_by_runtime_context,
    get_moe_gating_base_model_name,
    has_prefill_hot_moe_gating_rows,
    should_enable_prefill_hot_moe_gating_contract,
)
from frontier.moe_routing_runtime import (
    filter_moe_gating_routing_topk_rows,
    resolve_moe_gating_routing_runtime_path,
)
from frontier.operators.families import (
    FFN_FAMILY,
    MEMORY_FAMILY,
    MOE_FAMILY,
    SHARE_EXPERT_FAMILY,
    get_operator_family,
    get_family_profiling_names,
    get_family_profiling_name_set,
    is_moe_operator_ep_agnostic,
    resolve_moe_operator_tp_key,
)
from frontier.operators.spec import TensorParallelMode
from frontier.profiling.cpu_overhead.validation import (
    apply_cpu_overhead_schema_v2_defaults,
    validate_cpu_overhead_dataframe,
)
from frontier.spec_decode.runtime import is_target_embedded_mtp_enabled
from frontier.spec_decode.mtp_registry import (
    get_target_embedded_mtp_linear_ops,
    is_target_embedded_mtp_same_tp_linear_op,
)


from frontier.execution_time_predictor.layer_contract_resolution import (
    LayerContractResolution,
)
from frontier.execution_time_predictor.prediction_family_trainers import (
    PredictionFamilyTrainers,
)
from frontier.execution_time_predictor.prediction_model_identity import (
    MIGRATION_HELP_COMMAND,
    _add_layer_contract_to_training_context,
    _build_exact_feature_lookup,
    _get_contract_hash,
    _get_moe_family_model_names,
    _get_moe_family_operator_by_model_name,
    _get_moe_gating_family_model_names,
    _get_prefill_hot_moe_gating_model_names,
    _is_moe_gating_family_model_name,
    _layer_contract_kwargs,
    _normalize_layer_contract_context,
    _resolve_model_architecture_profile,
    _resolve_model_architecture_profile_id,
    _resolve_profile_typed_family_for_query,
    _serialize_selected_layer_cache_identity,
    _typed_row_matches_contract,
    _validate_typed_parallel_selection,
)
from frontier.execution_time_predictor.prediction_model_registry import (
    PredictionModelRegistry,
)
from frontier.execution_time_predictor.profiling_dataframe_loaders import (
    ProfilingDataFrameLoaders,
)
logger = init_logger(__name__)


class ExecutionTimePredictionModelManager(
    PredictionFamilyTrainers,
    ProfilingDataFrameLoaders,
    PredictionModelRegistry,
    LayerContractResolution,
):
    """
    Centralized manager for training and caching ML models used for execution time prediction.
    Analyzes all cluster configurations to determine the union of required prediction models.
    Shares trained sklearn estimators across multiple execution time predictors to avoid redundant training.

    Input (only transformer's comp part, a damo case for moe model) 
    → [input_layernorm]                    # Attention mou dk le
    → [attn_pre_proj → attn_rope → attn_prefill/decode → attn_kv_cache_save → attn_post_proj]  # Attention moudle
    → [add]                                 # Residual moudle
    → [post_attention_layernorm]           # FFN moudle
    → [mlp_up_proj → mlp_act → mlp_down_proj] or [moe_gating_linear → moe_gating_routing_topk → moe_shuffling → moe_grouped_gemm]  # FFN moudle
    → [add]                                 # Residual moudle
    → Output
    """
    
    def __init__(self, cluster_configs: Dict[ClusterType, ClusterConfig], metrics_config: MetricsConfig):
        self._cluster_configs = cluster_configs
        self._metrics_config = metrics_config
        self._cache_dir = metrics_config.cache_dir
        os.makedirs(self._cache_dir, exist_ok=True)
        self._attention_tp_warning_cache: Set[str] = set()
        self._gdn_predictors: Dict[ClusterType, Any] = {}

        # Check if all clusters are in dummy mode
        self._all_dummy_mode = self._check_all_dummy_mode()

        self._active_measurement_type = MeasurementType.CUDA_EVENT
        self._trained_models_eager = {}
        self._trained_models_device_event = {}
        self._trained_models_kernel_only = {}
        self._models_by_precision_eager = {}
        self._models_by_precision_device_event = {}
        self._models_by_precision_kernel_only = {}
        self._model_profiling_precision_eager = {}
        self._model_profiling_precision_device_event = {}
        self._model_profiling_precision_kernel_only = {}
        # Typed FFN models are keyed by their selected semantic domain.  Keep
        # the historical bare registries for untyped/legacy models only.
        self._trained_models_eager_by_contract = {}
        self._trained_models_device_event_by_contract = {}
        self._trained_models_kernel_only_by_contract = {}
        self._models_by_precision_eager_by_contract = {}
        self._models_by_precision_device_event_by_contract = {}
        self._models_by_precision_kernel_only_by_contract = {}

        if self._all_dummy_mode:
            logger.info("ExecutionTimePredictionModelManager running in DUMMY mode")
            logger.info("Skipping all ML model training and caching")
            self._required_capabilities = {}
            self._trained_models = {}
            self._models_by_precision = {}
            self._model_profiling_precision = {}
        else:
            # Analyze all cluster configurations to determine required prediction model capabilities
            # Required capabilities are retained for diagnostics; trained_model_signatures is the active cache key.
            self._required_capabilities = self._analyze_cluster_requirements()

            # Train all required prediction models once based on capabilities per cluster
            self._models_by_precision = {}
            self._model_profiling_precision = {}
            self._trained_models = self._train_all_required_models()

            logger.info(f"ExecutionTimePredictionModelManager initialized with capabilities: {self._required_capabilities}")

    def _check_all_dummy_mode(self) -> bool:
        """Check if all clusters are configured for dummy mode."""
        return all(
            cluster_config.execution_time_predictor_config.enable_dummy_mode
            for cluster_config in self._cluster_configs.values()
        )

    def _should_train_communication_models(self, cluster_config: ClusterConfig) -> bool:
        """Return whether shared-manager communication models should be trained."""
        cc_backend_config = getattr(cluster_config, "cc_backend_config", None)
        if cc_backend_config is None:
            return True
        return cc_backend_config.get_type() == CCBackendType.VIDUR

    def _analyze_cluster_requirements(self) -> Dict[str, Any]:
        """
        Analyze all cluster configurations to determine the union of required prediction models and capabilities.
        """
        capabilities = {
            'requires_attention': False,
            'requires_moe': False,
            'requires_pipeline_parallel': False,
            'requires_tensor_parallel': False,
            'requires_expert_parallel': False,
            'attn_tensor_parallel_sizes': set(),
            'moe_tensor_parallel_sizes': set(),
            'moe_expert_parallel_sizes': set(),
            'pipeline_stages': set(),
            'devices': set(),
            'network_devices': set(),
            'models': set(),
            'block_sizes': set(),
            'replica_scheduler_providers': set(),
        }
        
        for cluster_type, cluster_config in self._cluster_configs.items():
            replica_config = cluster_config.replica_config
            replica_scheduler_config = cluster_config.replica_scheduler_config
            
            # Determine what capabilities each cluster needs
            if cluster_type in [ClusterType.PREFILL, ClusterType.DECODE_ATTN, ClusterType.MONOLITHIC]:
                capabilities['requires_attention'] = True
                capabilities['attn_tensor_parallel_sizes'].add(replica_config.attn_tensor_parallel_size)
                
            if cluster_type in [ClusterType.PREFILL, ClusterType.DECODE_FFN, ClusterType.MONOLITHIC]:
                # Check if model is MoE based on model_config, NOT parallelism settings
                model_is_moe = (
                    replica_config.model_config is not None
                    and replica_config.model_config.is_moe
                )
                if model_is_moe:
                    capabilities['requires_moe'] = True
                    # Expert parallelism is enabled if moe_expert_parallel_size > 1
                    if replica_config.moe_expert_parallel_size > 1:
                        capabilities['requires_expert_parallel'] = True
                    capabilities['moe_tensor_parallel_sizes'].add(replica_config.moe_tensor_parallel_size)
                    capabilities['moe_expert_parallel_sizes'].add(replica_config.moe_expert_parallel_size)
                    
            if replica_config.num_pipeline_stages > 1:
                capabilities['requires_pipeline_parallel'] = True
                capabilities['pipeline_stages'].add(replica_config.num_pipeline_stages)
                
            if replica_config.attn_tensor_parallel_size > 1 or replica_config.moe_tensor_parallel_size > 1:
                capabilities['requires_tensor_parallel'] = True
                 
            capabilities['devices'].add(replica_config.device)
            capabilities['network_devices'].add(replica_config.network_device)
            capabilities['models'].add(replica_config.model_name)
            capabilities['block_sizes'].add(replica_scheduler_config.block_size)
            capabilities['replica_scheduler_providers'].add(str(replica_scheduler_config.get_type()))
        
        return capabilities

    @staticmethod

    def _measurement_family_name(measurement_type: MeasurementType) -> str:
        if measurement_type == MeasurementType.CUDA_EVENT:
            return "eager"
        if measurement_type == MeasurementType.DEVICE_EVENT:
            return "device_event"
        if measurement_type == MeasurementType.KERNEL_ONLY:
            return "kernel_only"
        raise ValueError(f"Unsupported measurement_type={measurement_type!r}")

    @staticmethod

    def _event_measurement_type_for_replica(replica_config) -> MeasurementType:
        """Select an event family from the configured device metadata only."""

        return resolve_event_measurement_type(replica_config)

    def _set_active_measurement_type(self, measurement_type: MeasurementType) -> None:
        self._active_measurement_type = measurement_type

    @staticmethod

    def _is_kernel_only_measurement_enabled_for_cluster(
        cluster_type: ClusterType,
    ) -> bool:
        if global_vars.get_sys_arch() == "pd-af-disaggregation":
            return cluster_type in (
                ClusterType.DECODE,
                ClusterType.DECODE_ATTN,
                ClusterType.DECODE_FFN,
                ClusterType.MONOLITHIC,
            )

        decode_cuda_graph_mode = str(global_vars.get_decode_cuda_graph_mode()).lower()
        use_cuda_graph = bool(global_vars.get_use_cuda_graph())

        if cluster_type == ClusterType.PREFILL:
            return False
        if cluster_type in (ClusterType.MONOLITHIC, ClusterType.DECODE):
            return decode_cuda_graph_mode != "none"
        if cluster_type in (ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN):
            return use_cuda_graph
        raise ValueError(f"Unsupported cluster_type={cluster_type!r}")

    def _get_measurement_types_for_cluster(
        self, cluster_type: ClusterType, replica_config
    ) -> List[MeasurementType]:
        event_measurement = self._event_measurement_type_for_replica(replica_config)
        if global_vars.get_sys_arch() == "pd-af-disaggregation":
            if cluster_type == ClusterType.PREFILL:
                return [event_measurement]
            if cluster_type == ClusterType.DECODE_ATTN:
                return [event_measurement, MeasurementType.KERNEL_ONLY]
            if cluster_type in (ClusterType.DECODE, ClusterType.DECODE_FFN):
                return [MeasurementType.KERNEL_ONLY]
            if cluster_type == ClusterType.MONOLITHIC:
                return [event_measurement, MeasurementType.KERNEL_ONLY]
            raise ValueError(f"Unsupported cluster_type={cluster_type!r}")

        if cluster_type == ClusterType.PREFILL:
            return [event_measurement]
        if cluster_type in (
            ClusterType.DECODE,
            ClusterType.DECODE_ATTN,
            ClusterType.DECODE_FFN,
        ):
            if self._is_kernel_only_measurement_enabled_for_cluster(cluster_type):
                return [MeasurementType.KERNEL_ONLY]
            return [event_measurement]
        if cluster_type == ClusterType.MONOLITHIC:
            if self._is_kernel_only_measurement_enabled_for_cluster(cluster_type):
                return [event_measurement, MeasurementType.KERNEL_ONLY]
            return [event_measurement]
        raise ValueError(f"Unsupported cluster_type={cluster_type!r}")

    def _resolve_measurement_input_files_for_config(
        self, replica_config, execution_time_predictor_config, measurement_type: MeasurementType
    ) -> Tuple[str, str, str, str, str, str]:
        paths = resolve_measurement_input_paths(
            execution_time_predictor_config,
            measurement_type,
            device=replica_config.device,
            model=replica_config.model_config.get_name(),
            network_device=replica_config.network_device,
        )
        return (
            paths.compute,
            paths.attention,
            paths.all_reduce,
            paths.send_recv,
            paths.cpu_overhead,
            paths.moe,
        )

    def _get_input_files_for_config(self, replica_config, execution_time_predictor_config) -> Tuple[str, str, str, str, str, str]:
        """
        Get input file paths for a given configuration.

        Returns tuple of: (linear_op_file, atten_file, all_reduce_file, send_recv_file, cpu_overhead_file, moe_file)
        """
        measurement_type = getattr(
            self, "_active_measurement_type", MeasurementType.CUDA_EVENT
        )
        return self._resolve_measurement_input_files_for_config(
            replica_config,
            execution_time_predictor_config,
            measurement_type,
        )

    def _create_estimator_and_params(self, execution_time_predictor_config):
        """
        Create estimator and grid search params based on predictor config type.
        """
        from frontier.types import ExecutionTimePredictorType
        
        if execution_time_predictor_config.get_type() == ExecutionTimePredictorType.RANDOM_FORREST:
            from sklearn.ensemble import RandomForestRegressor
            # Fix the predictor seed so model selection is independent of the
            # process-global RNG stream and training order.
            estimator = RandomForestRegressor(random_state=0)
            grid_search_params = {
                "n_estimators": execution_time_predictor_config.num_estimators,
                "max_depth": execution_time_predictor_config.max_depth,
                "min_samples_split": execution_time_predictor_config.min_samples_split,
            }
        elif execution_time_predictor_config.get_type() == ExecutionTimePredictorType.LINEAR_REGRESSION:
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import PolynomialFeatures
            estimator = make_pipeline(PolynomialFeatures(), LinearRegression())
            grid_search_params = {
                "polynomialfeatures__degree": execution_time_predictor_config.polynomial_degree,
                "polynomialfeatures__include_bias": execution_time_predictor_config.polynomial_include_bias,
                "polynomialfeatures__interaction_only": execution_time_predictor_config.polynomial_interaction_only,
                "linearregression__fit_intercept": execution_time_predictor_config.fit_intercept,
            }
        else:
            raise ValueError(f"Unsupported predictor type: {execution_time_predictor_config.get_type()}")
            
        return estimator, grid_search_params

    @staticmethod

    def mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
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
        return make_scorer(self.mean_absolute_percentage_error, greater_is_better=False)

    def _train_all_required_models(self) -> Dict[str, BaseEstimator]:
        """Train all prediction models required by cluster configurations."""
        combined_models: Dict[str, BaseEstimator] = {}
        trained_model_signatures = set()

        logger.info("=== ExecutionTimePredictionModelManager Training Summary ===")
        logger.info(f"Total clusters to process: {len(self._cluster_configs)}")

        for cluster_type, cluster_config in self._cluster_configs.items():
            replica_config = cluster_config.replica_config
            execution_time_predictor_config = cluster_config.execution_time_predictor_config
            replica_scheduler_config = cluster_config.replica_scheduler_config
            model_config = replica_config.model_config
            is_moe_model = model_config.is_moe

            logger.info(f"\n--- Processing Cluster: {cluster_type} ---")
            logger.info(f"Device: {replica_config.device}")
            logger.info(f"Model: {replica_config.model_name}")
            logger.info(f"Attention TP Size: {replica_config.attn_tensor_parallel_size}")
            logger.info(f"MoE TP Size: {replica_config.moe_tensor_parallel_size}")
            logger.info(f"Pipeline Stages: {replica_config.num_pipeline_stages}")
            logger.info(f"Network Device: {replica_config.network_device}")
            logger.info(f"Block Size: {replica_scheduler_config.block_size}")
            logger.info(f"Is MoE Model: {is_moe_model}")

            for measurement_type in self._get_measurement_types_for_cluster(
                cluster_type, replica_config
            ):
                self._set_active_measurement_type(measurement_type)
                self._load_gdn_predictor_for_cluster(
                    cluster_type,
                    replica_config,
                    execution_time_predictor_config,
                    measurement_type,
                )
                family_name = self._measurement_family_name(measurement_type)
                input_files = self._resolve_measurement_input_files_for_config(
                    replica_config, execution_time_predictor_config, measurement_type
                )
                linear_ops_file, attn_file, all_reduce_file, send_recv_file, cpu_overhead_file, moe_file = input_files

                logger.info("  Family: %s", family_name)
                logger.info(f"  - Linear Ops: {linear_ops_file} {'OK' if os.path.exists(linear_ops_file) else 'MISSING'}")
                logger.info(f"  - MoE: {moe_file} {'OK' if os.path.exists(moe_file) else 'MISSING'}")
                logger.info(f"  - Attention: {attn_file} {'OK' if os.path.exists(attn_file) else 'MISSING'}")
                logger.info(f"  - All-Reduce: {all_reduce_file} {'OK' if os.path.exists(all_reduce_file) else 'MISSING'}")
                logger.info(f"  - Send-Recv: {send_recv_file} {'OK' if os.path.exists(send_recv_file) else 'MISSING'}")
                logger.info(f"  - CPU Overhead: {cpu_overhead_file} {'OK' if os.path.exists(cpu_overhead_file) else 'MISSING'}")

                family_models: Dict[str, BaseEstimator] = {}

                if (
                    cluster_type
                    in [
                        ClusterType.PREFILL,
                        ClusterType.DECODE_ATTN,
                        ClusterType.DECODE,
                        ClusterType.MONOLITHIC,
                    ]
                ):
                    attention_models = self._train_attn_models_for_cluster(
                        cluster_type,
                        replica_config,
                        execution_time_predictor_config,
                        replica_scheduler_config,
                        linear_ops_file,
                        attn_file,
                        trained_model_signatures=trained_model_signatures,
                    )
                    family_models.update(attention_models)

                if cluster_type in [ClusterType.PREFILL, ClusterType.DECODE_FFN, ClusterType.DECODE, ClusterType.MONOLITHIC]:
                    ffn_models = self._train_ffn_models_for_cluster(
                        cluster_type,
                        replica_config,
                        execution_time_predictor_config,
                        linear_ops_file,
                        moe_file,
                        is_moe_model=is_moe_model,
                        trained_model_signatures=trained_model_signatures,
                    )
                    family_models.update(ffn_models)

                residual_models = self._train_residual_models_for_cluster(
                    cluster_type,
                    replica_config,
                    execution_time_predictor_config,
                    linear_ops_file,
                    trained_model_signatures=trained_model_signatures,
                )
                family_models.update(residual_models)

                should_train_comm_models = self._should_train_communication_models(cluster_config)
                if should_train_comm_models:
                    if replica_config.num_pipeline_stages > 1:
                        pipeline_models = self._train_pipeline_parallel_models_for_cluster(
                            cluster_type,
                            replica_config,
                            execution_time_predictor_config,
                            trained_model_signatures=trained_model_signatures,
                        )
                        family_models.update(pipeline_models)

                    if cluster_type in [ClusterType.PREFILL, ClusterType.DECODE_ATTN, ClusterType.DECODE, ClusterType.MONOLITHIC] and replica_config.attn_tensor_parallel_size > 1:
                        tensor_parallel_models = self._train_tensor_parallel_models_for_cluster(
                            cluster_type,
                            replica_config,
                            execution_time_predictor_config,
                            use_attn_tp=True,
                            trained_model_signatures=trained_model_signatures,
                        )
                        family_models.update(tensor_parallel_models)
                    elif cluster_type == ClusterType.DECODE_FFN and replica_config.moe_tensor_parallel_size > 1:
                        tensor_parallel_models = self._train_tensor_parallel_models_for_cluster(
                            cluster_type,
                            replica_config,
                            execution_time_predictor_config,
                            use_attn_tp=False,
                            trained_model_signatures=trained_model_signatures,
                        )
                        family_models.update(tensor_parallel_models)
                else:
                    logger.info(
                        "Skipping shared-manager communication model training for %s because cc_backend=%s provides runtime communication prediction.",
                        cluster_type,
                        cluster_config.cc_backend_config.get_name(),
                    )

                cpu_overhead_models = self._train_cpu_overhead_models_for_cluster(
                    cluster_type,
                    replica_config,
                    execution_time_predictor_config,
                    trained_model_signatures=trained_model_signatures,
                )
                family_models.update(cpu_overhead_models)

                for model_name, model in family_models.items():
                    combined_models[f"{family_name}:{model_name}"] = model

        logger.info(
            "Trained %d family-scoped models in total across all clusters", len(combined_models)
        )
        return combined_models

    def _load_gdn_predictor_for_cluster(
        self,
        cluster_type: ClusterType,
        replica_config: Any,
        execution_time_predictor_config: Any,
        measurement_type: MeasurementType,
    ) -> None:
        """Load pre-trained GDN artifacts for a hybrid model cluster.

        GDN fitting remains an explicit ``frontier.training.cli gdn`` action;
        simulator initialization only loads the resulting manifest/artifacts.
        """

        model_config = replica_config.model_config
        if model_config.get_num_gdn_layers() <= 0:
            return
        from frontier.execution_time_predictor.gdn_predictor import GDNPredictor

        gdn_file = substitute_input_path(
            execution_time_predictor_config.gdn_input_file,
            device=replica_config.device,
            model=model_config.get_name(),
        )
        if not os.path.isfile(gdn_file):
            raise FileNotFoundError(
                "Hybrid GDN configuration requires a standard gdn.csv input; "
                f"file does not exist: {gdn_file}"
            )
        predictor = GDNPredictor.from_directory(
            self._cache_dir,
            model_config=model_config,
            device=replica_config.device,
            tensor_parallel_size=replica_config.attn_tensor_parallel_size,
            measurement_type=measurement_type,
            dataset_path=gdn_file,
        )
        existing = self._gdn_predictors.get(cluster_type)
        if existing is not None and existing.identity != predictor.identity:
            raise ValueError(
                "Hybrid GDN cluster requests incompatible model identities: "
                f"cluster={cluster_type}, existing={existing.identity}, "
                f"new={predictor.identity}"
            )
        self._gdn_predictors[cluster_type] = predictor

    def get_gdn_predictor(self, cluster_type: ClusterType) -> Any | None:
        """Return the loaded GDN predictor for one cluster, if configured."""

        return self._gdn_predictors.get(cluster_type)

    def get_models(self) -> Dict[str, Dict[str, BaseEstimator]]:
        """Return the trained models grouped by measurement family."""
        if self._all_dummy_mode:
            logger.debug("Returning empty models dict for dummy mode")
            return {"eager": {}, "kernel_only": {}}
        models = {
            "eager": self._models_view_for_family("eager"),
            "kernel_only": self._models_view_for_family("kernel_only"),
        }
        device_event_models = self._models_view_for_family("device_event")
        if device_event_models:
            models["device_event"] = device_event_models
        return models

    def _event_family_for_cluster(self, cluster_type: ClusterType) -> str:
        cluster_config = (getattr(self, "_cluster_configs", None) or {}).get(
            cluster_type
        )
        replica_config = getattr(cluster_config, "replica_config", None)
        measurement_type = self._event_measurement_type_for_replica(replica_config)
        return self._measurement_family_name(measurement_type)

    def get_models_for_cluster(self, cluster_type: ClusterType) -> Dict[str, Dict[str, BaseEstimator]]:
        """Return a cluster-specific view of trained models grouped by measurement family."""
        if self._all_dummy_mode:
            return {"eager": {}, "kernel_only": {}}

        event_family = self._event_family_for_cluster(cluster_type)

        def _event_models() -> Dict[str, BaseEstimator]:
            return self._models_view_for_family(event_family, cluster_type)

        event_key = "eager" if event_family == "eager" else event_family

        if cluster_type == ClusterType.PREFILL:
            models = {
                event_key: _event_models(),
                "kernel_only": {},
            }
            return models
        if cluster_type in [ClusterType.DECODE, ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN]:
            if (
                global_vars.get_sys_arch() == "pd-af-disaggregation"
                and cluster_type == ClusterType.DECODE_ATTN
            ):
                models = {
                    event_key: _event_models(),
                    "kernel_only": self._models_view_for_family(
                        "kernel_only", cluster_type
                    ),
                }
                return models
            if not self._is_kernel_only_measurement_enabled_for_cluster(cluster_type):
                return {
                    event_key: _event_models(),
                    "kernel_only": {},
                }
            return {
                "eager": {},
                "kernel_only": self._models_view_for_family(
                    "kernel_only", cluster_type
                ),
            }
        if cluster_type == ClusterType.MONOLITHIC:
            kernel_only_models = {}
            if self._is_kernel_only_measurement_enabled_for_cluster(cluster_type):
                kernel_only_models = self._models_view_for_family(
                    "kernel_only", cluster_type
                )
            return {
                event_key: _event_models(),
                "kernel_only": kernel_only_models,
            }
        raise ValueError(f"Unsupported cluster_type={cluster_type!r}")

    def get_training_file_paths(self, cluster_type: ClusterType) -> Dict[str, str]:
        """Get the resolved profiling file paths for a specific cluster type."""
        if cluster_type not in self._cluster_configs:
            return {}

        cluster_config = self._cluster_configs[cluster_type]
        replica_config = cluster_config.replica_config
        execution_time_predictor_config = cluster_config.execution_time_predictor_config

        return resolve_training_file_paths(
            execution_time_predictor_config,
            device=replica_config.device,
            model=replica_config.model_config.get_name(),
            network_device=replica_config.network_device,
        )
