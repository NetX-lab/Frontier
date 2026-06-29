import hashlib
import os
import pickle
from itertools import product
from typing import Dict, Set, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
from fasteners import InterProcessReaderWriterLock
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from frontier.attention.families import DENSE_ATTENTION_FAMILY
from frontier.attention.ops import AttentionOperatorRole
from frontier.attention.profiling_mapping import (
    get_enabled_predictor_median_column_by_role,
    get_enabled_predictor_median_columns,
    get_enabled_predictor_metric_name_by_role,
    get_enabled_predictor_metric_names,
    get_enabled_shared_predictor_feature_columns,
)
from frontier.config import MetricsConfig, ClusterConfig, global_vars
from frontier.types import ClusterType, CCBackendType, MeasurementType
from frontier.execution_time_predictor.attention_tp_policy import (
    resolve_effective_attention_tp_size,
)
from frontier.execution_time_predictor.attention_dataset_contract import (
    enforce_mixed_attention_input_contract,
)
from frontier.logger import init_logger
from frontier.moe_gating_runtime import (
    DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
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
from frontier.profiling.cpu_overhead.validation import (
    apply_cpu_overhead_schema_v2_defaults,
    validate_cpu_overhead_dataframe,
)
from frontier.spec_decode.runtime import (
    TARGET_EMBEDDED_MTP_SAME_TP_LINEAR_OPS,
    is_target_embedded_mtp_enabled,
)

logger = init_logger(__name__)
MIGRATION_HELP_COMMAND = (
    "python -m frontier.profiling.migrate_csv_metadata --help"
)


def _build_exact_feature_lookup(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Dict[Tuple[float, ...], float]:
    """Build exact profiling-row lookups before falling back to regression."""
    if df.empty:
        return {}
    grouped = df.groupby(feature_cols, dropna=False)[target_col].mean()
    lookup: Dict[Tuple[float, ...], float] = {}
    for key, value in grouped.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        lookup[tuple(float(item) for item in key_tuple)] = float(value)
    return lookup


class ExecutionTimePredictionModelManager:
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

        # Check if all clusters are in dummy mode
        self._all_dummy_mode = self._check_all_dummy_mode()

        self._active_measurement_type = MeasurementType.CUDA_EVENT
        self._trained_models_eager = {}
        self._trained_models_kernel_only = {}
        self._models_by_precision_eager = {}
        self._models_by_precision_kernel_only = {}
        self._model_profiling_precision_eager = {}
        self._model_profiling_precision_kernel_only = {}

        if self._all_dummy_mode:
            logger.info("ExecutionTimePredictionModelManager running in DUMMY mode")
            logger.info("Skipping all ML model training and caching")
            self._required_capabilities = {}
            self._trained_models = {}
            self._models_by_precision = {}
            self._model_profiling_precision = {}
        else:
            # Analyze all cluster configurations to determine required prediction model capabilities
            # TODO: designed for reduce redundancy. Not use yet (consider to remove it). Currently, we use trained_model_signatures.
            # ignore it now
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
        if measurement_type == MeasurementType.KERNEL_ONLY:
            return "kernel_only"
        raise ValueError(f"Unsupported measurement_type={measurement_type!r}")

    def _set_active_measurement_type(self, measurement_type: MeasurementType) -> None:
        self._active_measurement_type = measurement_type

    @staticmethod
    def _is_kernel_only_measurement_enabled_for_cluster(
        cluster_type: ClusterType,
    ) -> bool:
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
        self, cluster_type: ClusterType
    ) -> List[MeasurementType]:
        if cluster_type == ClusterType.PREFILL:
            return [MeasurementType.CUDA_EVENT]
        if cluster_type in (
            ClusterType.DECODE,
            ClusterType.DECODE_ATTN,
            ClusterType.DECODE_FFN,
        ):
            if self._is_kernel_only_measurement_enabled_for_cluster(cluster_type):
                return [MeasurementType.KERNEL_ONLY]
            return [MeasurementType.CUDA_EVENT]
        if cluster_type == ClusterType.MONOLITHIC:
            if self._is_kernel_only_measurement_enabled_for_cluster(cluster_type):
                return [MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY]
            return [MeasurementType.CUDA_EVENT]
        raise ValueError(f"Unsupported cluster_type={cluster_type!r}")

    def _resolve_measurement_input_files_for_config(
        self, replica_config, execution_time_predictor_config, measurement_type: MeasurementType
    ) -> Tuple[str, str, str, str, str, str]:
        linear_op_file = execution_time_predictor_config.linear_op_input_file
        if not linear_op_file and execution_time_predictor_config.mlp_input_file:
            linear_op_file = execution_time_predictor_config.mlp_input_file

        cpu_overhead_file = execution_time_predictor_config.cpu_overhead_input_file

        if measurement_type == MeasurementType.CUDA_EVENT:
            compute_file = linear_op_file
            attention_file = execution_time_predictor_config.atten_input_file
            moe_file = execution_time_predictor_config.moe_input_file
        elif measurement_type == MeasurementType.KERNEL_ONLY:
            compute_file = execution_time_predictor_config.linear_op_kernel_only_input_file
            attention_file = execution_time_predictor_config.atten_kernel_only_input_file
            moe_file = execution_time_predictor_config.moe_kernel_only_input_file
            cpu_overhead_file = (
                getattr(
                    execution_time_predictor_config,
                    "cpu_overhead_kernel_only_input_file",
                    "",
                )
                or execution_time_predictor_config.cpu_overhead_input_file
            )
        else:
            raise ValueError(f"Unsupported measurement_type={measurement_type!r}")

        input_files = [
            compute_file,
            attention_file,
            execution_time_predictor_config.all_reduce_input_file,
            execution_time_predictor_config.send_recv_input_file,
            cpu_overhead_file,
            moe_file,
        ]

        for i in range(len(input_files)):
            input_files[i] = (
                input_files[i]
                .replace("{DEVICE}", replica_config.device)
                .replace("{MODEL}", replica_config.model_config.get_name())
                .replace("{NETWORK_DEVICE}", replica_config.network_device)
            )

        return tuple(input_files)

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

            for measurement_type in self._get_measurement_types_for_cluster(cluster_type):
                self._set_active_measurement_type(measurement_type)
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

                if cluster_type in [ClusterType.PREFILL, ClusterType.DECODE_ATTN, ClusterType.DECODE, ClusterType.MONOLITHIC]:
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

    def _get_ffn_tp_key(self, cluster_type: ClusterType, replica_config, is_moe_model: bool) -> int:
        if (
            is_moe_model
            and cluster_type in {
                ClusterType.PREFILL,
                ClusterType.DECODE_FFN,
                ClusterType.DECODE,
                ClusterType.MONOLITHIC,
            }
        ):
            return replica_config.moe_tensor_parallel_size
        return replica_config.attn_tensor_parallel_size

    def _get_linear_op_tp_key(self, op_name: str, cluster_type: ClusterType, replica_config, is_moe_model: bool) -> int:
        if op_name.startswith("mlp_"):
            raise ValueError(
                f"Linear op TP mapping helper does not handle mlp ops: {op_name}"
            )

        replicated_ops = {
            "input_layernorm",
            "post_attention_layernorm",
            "add",
            "emb",
            # Step3Text-specific replicated pre-proj sub-ops.
            "attn_pre_proj_qkv",
            "attn_pre_proj_q_norm",
        }
        if op_name in replicated_ops:
            if (
                is_target_embedded_mtp_enabled(
                    getattr(replica_config, "speculative_decoding_config", None)
                )
                and op_name in TARGET_EMBEDDED_MTP_SAME_TP_LINEAR_OPS
            ):
                return resolve_effective_attention_tp_size(
                    op_name="attn_pre_proj",
                    requested_tp_size=replica_config.attn_tensor_parallel_size,
                    num_kv_heads=replica_config.model_config.num_kv_heads,
                    cluster_type=cluster_type,
                    warning_cache=getattr(self, "_attention_tp_warning_cache", None),
                    include_linear_ops=True,
                )
            return 1

        if op_name in {
            "mtp_fusion_proj",
            "lm_head_linear",
        }:
            return resolve_effective_attention_tp_size(
                op_name="attn_pre_proj",
                requested_tp_size=replica_config.attn_tensor_parallel_size,
                num_kv_heads=replica_config.model_config.num_kv_heads,
                cluster_type=cluster_type,
                warning_cache=getattr(self, "_attention_tp_warning_cache", None),
                include_linear_ops=True,
            )

        if op_name in {
            "share_expert_up_proj",
            "share_expert_down_proj",
            "share_expert_act",
        }:
            return self._get_ffn_tp_key(cluster_type, replica_config, is_moe_model)

        if op_name.startswith("attn_"):
            return resolve_effective_attention_tp_size(
                op_name=op_name,
                requested_tp_size=replica_config.attn_tensor_parallel_size,
                num_kv_heads=replica_config.model_config.num_kv_heads,
                cluster_type=cluster_type,
                warning_cache=getattr(self, "_attention_tp_warning_cache", None),
                include_linear_ops=True,
            )

        raise ValueError(f"Unsupported linear op for TP mapping: {op_name}")

    @staticmethod
    def _get_moe_op_tp_key(op_name: str, replica_config) -> int:
        replicated_ops = {
            "moe_gating_linear",
            "moe_gating_routing_topk",
            "moe_shuffling",
        }
        if op_name in replicated_ops:
            return 1
        if op_name == "moe_grouped_gemm":
            return replica_config.moe_tensor_parallel_size
        raise ValueError(f"Unsupported MoE op for TP mapping: {op_name}")

    @staticmethod
    def _is_moe_op_ep_agnostic(op_name: str) -> bool:
        return op_name in {
            "moe_gating_linear",
            "moe_gating_routing_topk",
            "moe_shuffling",
        }

    def _validate_moe_dataset_contract(
        self,
        file_path: str,
        replica_config,
        model_names: List[str],
    ) -> None:
        """Validate op-level MoE profiling key coverage before model training."""
        df = pd.read_csv(file_path)
        required_columns = [
            "num_experts",
            "router_topk",
            "hidden_dim",
            "expert_hidden_dim",
            "num_tensor_parallel_workers",
            "expert_parallel_size",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"MoE dataset contract validation failed for {file_path}: "
                f"missing required columns {missing_columns}."
            )

        model_config = replica_config.model_config
        base_df = df[
            (df["num_experts"] == model_config.num_experts)
            & (df["router_topk"] == model_config.num_experts_per_tok)
            & (df["hidden_dim"] == model_config.embedding_dim)
            & (df["expert_hidden_dim"] == model_config.mlp_hidden_dim)
        ]

        if len(base_df) == 0:
            raise ValueError(
                "MoE dataset contract validation failed: no rows match model configuration in "
                f"{file_path}. Required: num_experts={model_config.num_experts}, "
                f"router_topk={model_config.num_experts_per_tok}, hidden_dim={model_config.embedding_dim}, "
                f"expert_hidden_dim={model_config.mlp_hidden_dim}."
            )

        available_pairs = sorted(
            {
                (int(tp), int(ep))
                for tp, ep in base_df[
                    ["num_tensor_parallel_workers", "expert_parallel_size"]
                ].drop_duplicates().itertuples(index=False, name=None)
            }
        )
        requested_routing_runtime_path = resolve_moe_gating_routing_runtime_path(
            getattr(replica_config, "moe_routing_mode", "simulation")
        )

        missing_requirements: List[str] = []
        for model_name in model_names:
            tp_key = self._get_moe_op_tp_key(model_name, replica_config)
            if self._is_moe_op_ep_agnostic(model_name):
                op_df = base_df[
                    base_df["num_tensor_parallel_workers"] == tp_key
                ]
                requirement = f"TP={tp_key}, EP=ANY"
            else:
                ep_key = replica_config.moe_expert_parallel_size
                op_df = base_df[
                    (base_df["num_tensor_parallel_workers"] == tp_key)
                    & (base_df["expert_parallel_size"] == ep_key)
                ]
                requirement = f"TP={tp_key}, EP={ep_key}"
            if model_name == "moe_gating_routing_topk":
                op_df = filter_moe_gating_routing_topk_rows(
                    op_df,
                    requested_runtime_path=requested_routing_runtime_path,
                    source_name=file_path,
                )
                requirement = (
                    f"{requirement}, routing_runtime_path="
                    f"{requested_routing_runtime_path}"
                )
            if len(op_df) == 0:
                missing_requirements.append(f"{model_name} requires {requirement}")

        if missing_requirements:
            requirement_text = "\n  - ".join(missing_requirements)
            raise ValueError(
                "MoE dataset contract validation failed before training.\n"
                f"File: {file_path}\n"
                "Missing op-level key coverage:\n"
                f"  - {requirement_text}\n"
                f"Available (TP, EP) pairs for matched model rows: {available_pairs}"
            )

    def _train_ffn_models_for_cluster(self, cluster_type: ClusterType, replica_config, execution_time_predictor_config,
                                        linear_ops_file: str, moe_file: str,
                                        is_moe_model: bool, trained_model_signatures: set) -> Dict[str, BaseEstimator]:
        """
        Train FFN/MoE models for a specific cluster.

        This function handles FFN-related operations in the Transformer layer:
        - FFN core operations (from linear_op.csv): mlp_up_proj, mlp_down_proj, mlp_act
        - MoE core operations (from moe.csv): moe_gating_linear, moe_gating_routing_topk, moe_shuffling, moe_grouped_gemm
        - Pre-FFN normalization (from linear_op.csv): post_attention_layernorm

        Transformer layer context:
            ... → Attention → add → [post_attention_layernorm] → [FFN/MoE] → add → ...
        """
        models = {}

        ffn_tp_key = self._get_ffn_tp_key(cluster_type, replica_config, is_moe_model)
        tp_size = ffn_tp_key

        # Create a signature for this FFN model configuration
        # Include model_arch in signature to distinguish Step2Mini-specific operations from generic models
        model_config = replica_config.model_config
        model_arch = model_config.get_model_arch() if model_config is not None else "generic"
        active_measurement_type = getattr(
            self, "_active_measurement_type", MeasurementType.CUDA_EVENT
        )
        ffn_signature = f"ffn_{replica_config.device}_{replica_config.model_name}_{tp_size}_moe{is_moe_model}_arch{model_arch}_family{self._measurement_family_name(active_measurement_type)}"

        if ffn_signature in trained_model_signatures:
            logger.info(f"Skipping FFN models training for {cluster_type} - already trained with signature {ffn_signature}")
            return models

        # Build training context for error messages
        training_context = {
            'cluster_type': str(cluster_type),
            'device': replica_config.device,
            'model_name': replica_config.model_name,
            'tensor_parallel_size': tp_size,
            'is_moe_model': is_moe_model,
            'model_arch': model_arch,
            'use_qk_norm': bool(getattr(model_config, 'use_qk_norm', False)),
        }

        # Choose input file based on model type
        if is_moe_model:
            moe_input_file = moe_file
            if not os.path.exists(moe_input_file):
                raise FileNotFoundError(f"MoE input file {moe_input_file} not found")
            logger.info(f"Loading MoE data for {cluster_type} from: {moe_input_file}")
            training_context['input_file'] = moe_input_file

            # MoE core operations with per-operation feature selection
            # Split gating into moe_gating_linear and moe_gating_routing_topk (Step 1.6)
            # Aligned with frontier/training/moe_trainer.py _get_feature_cols() method
            moe_model_names = [
                "moe_gating_linear",
                "moe_gating_routing_topk",
                "moe_shuffling",
                "moe_grouped_gemm",
            ]
            if should_enable_prefill_hot_moe_gating_contract(
                model_config=model_config,
                model_arch=model_arch,
                model_name=replica_config.model_name,
            ):
                include_prefill_hot_models = False
                try:
                    prefill_hot_probe_df = pd.read_csv(moe_input_file)
                    include_prefill_hot_models = has_prefill_hot_moe_gating_rows(
                        prefill_hot_probe_df
                    )
                except Exception as e:
                    logger.warning(
                        "Unable to probe prefill-hot gating rows in %s: %s",
                        moe_input_file,
                        e,
                    )
                    include_prefill_hot_models = False

                if include_prefill_hot_models:
                    moe_model_names.extend(
                        [
                            "moe_gating_linear__prefill_hot",
                            "moe_gating_routing_topk__prefill_hot",
                        ]
                    )
                else:
                    logger.warning(
                        "Prefill-hot gating contract enabled for model=%s, but "
                        "dataset %s has no usable prefill_hot rows; skipping "
                        "__prefill_hot pseudo-models in shared-manager training.",
                        replica_config.model_name,
                        moe_input_file,
                    )
            self._validate_moe_dataset_contract(
                moe_input_file,
                replica_config,
                [
                    "moe_gating_linear",
                    "moe_gating_routing_topk",
                    "moe_shuffling",
                    "moe_grouped_gemm",
                ],
            )
            requested_routing_runtime_path = resolve_moe_gating_routing_runtime_path(
                getattr(replica_config, "moe_routing_mode", "simulation")
            )

            moe_df_cache: Dict[
                Tuple[int, Optional[int], Optional[str], Optional[str]], pd.DataFrame
            ] = {}

            def _get_moe_df_for_op(
                model_name: str,
            ) -> Tuple[pd.DataFrame, int, Optional[int]]:
                base_model_name = get_moe_gating_base_model_name(model_name)
                tp_key = self._get_moe_op_tp_key(base_model_name, replica_config)
                if tp_key <= 0:
                    raise ValueError(
                        f"Invalid TP key for MoE training: {tp_key} (op={model_name})"
                    )

                ep_key: Optional[int]
                if self._is_moe_op_ep_agnostic(base_model_name):
                    ep_key = None
                else:
                    ep_key = replica_config.moe_expert_parallel_size

                runtime_path_key: Optional[str] = None
                if base_model_name == "moe_gating_routing_topk":
                    runtime_path_key = requested_routing_runtime_path

                gating_context_key: Optional[str] = None
                if base_model_name in {
                    "moe_gating_linear",
                    "moe_gating_routing_topk",
                }:
                    gating_context_key = DEFAULT_MOE_GATING_RUNTIME_CONTEXT
                    if model_name.endswith("__prefill_hot"):
                        gating_context_key = PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT

                cache_key = (tp_key, ep_key, runtime_path_key, gating_context_key)
                if cache_key not in moe_df_cache:
                    op_df = self._load_moe_df(
                        moe_input_file,
                        replica_config,
                        load_imbalance=False,
                        tensor_parallel_size=tp_key,
                        expert_parallel_size=ep_key,
                    )
                    if runtime_path_key is not None:
                        op_df = filter_moe_gating_routing_topk_rows(
                            op_df,
                            requested_runtime_path=runtime_path_key,
                            source_name=moe_input_file,
                        )
                    if gating_context_key is not None:
                        op_df = filter_moe_gating_rows_by_runtime_context(
                            op_df,
                            requested_context=gating_context_key,
                            source_name=moe_input_file,
                        )
                    moe_df_cache[cache_key] = op_df
                    ep_desc = "ANY" if ep_key is None else str(ep_key)
                    logger.info(
                        f"Loaded {len(moe_df_cache[cache_key])} rows for MoE training "
                        f"(op={model_name}, tp_key={tp_key}, ep_key={ep_desc}, "
                        f"routing_runtime_path={runtime_path_key or 'ANY'}, "
                        f"gating_runtime_context={gating_context_key or 'ANY'}, "
                        "auto feature mode)"
                    )
                return moe_df_cache[cache_key], tp_key, ep_key

            for model_name in moe_model_names:
                model_signature = f"{model_name}_{ffn_signature}"
                if model_signature not in trained_model_signatures:
                    try:
                        op_moe_df, moe_tp_key, moe_ep_key = _get_moe_df_for_op(model_name)
                    except ValueError as e:
                        if model_name.endswith("__prefill_hot"):
                            logger.warning(
                                "Skipping %s because prefill-hot gating rows are unavailable "
                                "for the requested TP/EP slice (%s).",
                                model_name,
                                e,
                            )
                            continue
                        raise
                    training_context['tensor_parallel_size'] = moe_tp_key
                    training_context['expert_parallel_size'] = (
                        "ANY" if moe_ep_key is None else moe_ep_key
                    )

                    # Per-operation feature selection.
                    if model_name == "moe_grouped_gemm":
                        available_load_features = [
                            f for f in self.MOE_LOAD_IMBALANCE_FEATURES
                            if f in op_moe_df.columns
                        ]
                        has_load_imbalance_features = (
                            len(available_load_features)
                            == len(self.MOE_LOAD_IMBALANCE_FEATURES)
                        )
                        if 0 < len(available_load_features) < len(self.MOE_LOAD_IMBALANCE_FEATURES):
                            missing_features = [
                                f for f in self.MOE_LOAD_IMBALANCE_FEATURES
                                if f not in op_moe_df.columns
                            ]
                            raise ValueError(
                                f"Partial load imbalance features found ({len(available_load_features)}/"
                                f"{len(self.MOE_LOAD_IMBALANCE_FEATURES)}) for {model_name} at TP={moe_tp_key}. "
                                f"Missing: {missing_features}."
                            )

                        if has_load_imbalance_features:
                            op_feature_cols = available_load_features
                            logger.info(
                                f"  {model_name}: Using load imbalance features "
                                f"({len(op_feature_cols)} features, TP={moe_tp_key})"
                            )
                        else:
                            op_feature_cols = ["num_tokens"]
                            logger.info(
                                f"  {model_name}: Load imbalance features not found; "
                                f"using num_tokens only (TP={moe_tp_key})."
                            )
                    elif model_name == "moe_shuffling":
                        available_load_features = [
                            f for f in self.MOE_LOAD_IMBALANCE_FEATURES
                            if f in op_moe_df.columns
                        ]
                        if len(available_load_features) == len(self.MOE_LOAD_IMBALANCE_FEATURES):
                            op_feature_cols = available_load_features
                            logger.info(
                                f"  {model_name}: Using load imbalance features "
                                f"({len(op_feature_cols)} features, TP={moe_tp_key})"
                            )
                        else:
                            # For shuffling we allow partial/legacy datasets and fall back to
                            # num_tokens-only training when the full load feature set is absent.
                            op_feature_cols = ["num_tokens"]
                            logger.info(
                                f"  {model_name}: Full load imbalance features unavailable; "
                                f"using num_tokens only (TP={moe_tp_key})."
                            )
                    else:
                        op_feature_cols = ["num_tokens"]
                        logger.info(
                            f"  {model_name}: Using num_tokens only (1 feature, TP={moe_tp_key})"
                        )

                    # Store feature_cols in training_context for this specific operation
                    training_context['feature_cols'] = op_feature_cols

                    target_op_name = get_moe_gating_base_model_name(model_name)
                    models[model_name] = self._train_single_model(
                        model_name=model_name,
                        df=op_moe_df,
                        feature_cols=op_feature_cols,
                        target_col=f"time_stats.{target_op_name}.median",
                        execution_time_predictor_config=execution_time_predictor_config,
                        training_context=training_context,
                    )
                    trained_model_signatures.add(model_signature)
                    logger.info(f"Trained {model_name} for {cluster_type} with features: {op_feature_cols}")

            # Step2Mini/Step3 share_expert operations (forward_3: shared expert alongside routed experts)
            model_config = replica_config.model_config
            if model_config is not None and model_config.supports_share_expert():
                # share_expert operations are trained from linear_op.csv (not moe.csv)
                if not os.path.exists(linear_ops_file):
                    raise FileNotFoundError(
                        f"Linear ops input file {linear_ops_file} not found for share_expert"
                    )

                share_expert_tp_key = self._get_linear_op_tp_key(
                    "share_expert_up_proj",
                    cluster_type,
                    replica_config,
                    is_moe_model,
                )
                share_expert_linear_ops_df = self._load_linear_op_df(
                    linear_ops_file, share_expert_tp_key
                )
                logger.info(f"Loaded {len(share_expert_linear_ops_df)} rows for share_expert training")

                step2mini_share_expert_model_names = ["share_expert_up_proj", "share_expert_down_proj", "share_expert_act"]
                for model_name in step2mini_share_expert_model_names:
                    model_signature = f"{model_name}_{ffn_signature}"
                    if model_signature not in trained_model_signatures:
                        # Update training context to reflect linear_op.csv source.
                        training_context['input_file'] = linear_ops_file
                        training_context['tensor_parallel_size'] = share_expert_tp_key
                        target_col = f"time_stats.{model_name}.median"
                        if target_col not in share_expert_linear_ops_df.columns:
                            raise ValueError(
                                f"share_expert operation '{model_name}' column '{target_col}' not found in profiling data. "
                                f"Ensure profiling was run with a model architecture that includes share_expert. "
                                f"Available columns: {list(share_expert_linear_ops_df.columns)}"
                            )
                        models[model_name] = self._train_single_model(
                            model_name=model_name,
                            df=share_expert_linear_ops_df,
                            feature_cols=["num_tokens"],
                            target_col=target_col,
                            execution_time_predictor_config=execution_time_predictor_config,
                            training_context=training_context,
                        )
                        trained_model_signatures.add(model_signature)
                        logger.info(f"Trained {model_name} for {cluster_type}")
        else:
            # Dense MLP operations from linear_op.csv
            if not os.path.exists(linear_ops_file):
                raise FileNotFoundError(f"Linear ops input file {linear_ops_file} not found")
            logger.info(f"Loading MLP data for {cluster_type} from: {linear_ops_file}")
            linear_ops_df = self._load_linear_op_df(linear_ops_file, ffn_tp_key)
            logger.info(f"Loaded {len(linear_ops_df)} rows for MLP training")
            training_context['input_file'] = linear_ops_file

            # MLP core operations
            mlp_model_names = ["mlp_up_proj", "mlp_down_proj", "mlp_act"]
            for model_name in mlp_model_names:
                model_signature = f"{model_name}_{ffn_signature}"
                if model_signature not in trained_model_signatures:
                    models[model_name] = self._train_single_model(
                        model_name=model_name,
                        df=linear_ops_df,
                        feature_cols=["num_tokens"],
                        target_col=f"time_stats.{model_name}.median",
                        execution_time_predictor_config=execution_time_predictor_config,
                        training_context=training_context,
                    )
                    trained_model_signatures.add(model_signature)
                    logger.info(f"Trained {model_name} for {cluster_type}")

        # Pre-FFN normalization (post_attention_layernorm) - always from linear_op.csv
        if not os.path.exists(linear_ops_file):
            raise FileNotFoundError(f"Linear ops input file {linear_ops_file} not found for post_attention_layernorm")
        layernorm_tp_key = self._get_linear_op_tp_key(
            "post_attention_layernorm",
            cluster_type,
            replica_config,
            is_moe_model,
        )
        linear_ops_df = self._load_linear_op_df(linear_ops_file, layernorm_tp_key)
        layernorm_context = dict(training_context)
        layernorm_context["input_file"] = linear_ops_file
        layernorm_context["tensor_parallel_size"] = layernorm_tp_key

        layernorm_model_name = "post_attention_layernorm"
        layernorm_signature = f"{layernorm_model_name}_{ffn_signature}"
        if layernorm_signature not in trained_model_signatures:
            models[layernorm_model_name] = self._train_single_model(
                model_name=layernorm_model_name,
                df=linear_ops_df,
                feature_cols=["num_tokens"],
                target_col=f"time_stats.{layernorm_model_name}.median",
                execution_time_predictor_config=execution_time_predictor_config,
                training_context=layernorm_context,
            )
            trained_model_signatures.add(layernorm_signature)
            logger.info(f"Trained {layernorm_model_name} for {cluster_type}")

        # Mark this FFN configuration as trained
        trained_model_signatures.add(ffn_signature)
        return models

    def _train_attn_models_for_cluster(self, cluster_type: ClusterType, replica_config, execution_time_predictor_config, replica_scheduler_config, linear_ops_file: str, attn_file: str, trained_model_signatures: set) -> Dict[str, BaseEstimator]:
        """
        Train attention-related models for a cluster.

        This function handles Attention-related operations in the Transformer layer:
        - Pre-attention normalization (from linear_op.csv): input_layernorm
        - Attention projections (from linear_op.csv): attn_pre_proj, attn_post_proj, attn_rope
        - Attention core operations (from attention.csv): attn_kv_cache_save, attn_prefill, attn_decode

        Transformer layer context:
            Input → [input_layernorm] → [attn_pre_proj → attn_rope → attn_prefill/decode → attn_kv_cache_save → attn_post_proj] → add → ...
        """
        models = {}
        tp_size = replica_config.attn_tensor_parallel_size

        # Include model_arch in signature to distinguish Step2Mini-specific operations from generic models
        model_config = replica_config.model_config
        model_arch = model_config.get_model_arch() if model_config is not None else "generic"
        attention_signature = f"attention_{replica_config.device}_{replica_config.model_name}_{tp_size}_{replica_scheduler_config.block_size}_arch{model_arch}_family{self._measurement_family_name(self._active_measurement_type)}"

        if attention_signature in trained_model_signatures:
            logger.info(f"Skipping attention models training for {cluster_type} - already trained")
            return models

        # Build training context for error messages
        training_context = {
            'cluster_type': str(cluster_type),
            'device': replica_config.device,
            'model_name': replica_config.model_name,
            'tensor_parallel_size': tp_size,
            'block_size': replica_scheduler_config.block_size,
            'model_arch': model_arch,
            'use_qk_norm': bool(getattr(model_config, 'use_qk_norm', False)),
        }

        # ========== Part 1: Linear operations from linear_op.csv ==========
        # These include: input_layernorm, attn_pre_proj, attn_post_proj, attn_rope
        if not os.path.exists(linear_ops_file):
            raise FileNotFoundError(f"Linear ops input file {linear_ops_file} not found")

        logger.info(f"Loading sharded attention linear-op data from: {linear_ops_file}")
        attn_tp_key = self._get_linear_op_tp_key(
            "attn_pre_proj",
            cluster_type,
            replica_config,
            is_moe_model=False,
        )
        required_columns = self._get_required_attn_linear_op_columns(model_config)
        attn_linear_ops_df = self._load_linear_op_df(
            linear_ops_file,
            attn_tp_key,
            required_columns=required_columns,
            training_context=training_context,
        )
        logger.info(
            f"Loaded {len(attn_linear_ops_df)} rows for sharded attention ops training"
        )

        # Pre-attention normalization: input_layernorm
        input_layernorm_tp_key = self._get_linear_op_tp_key(
            "input_layernorm",
            cluster_type,
            replica_config,
            is_moe_model=False,
        )
        input_layernorm_df = self._load_linear_op_df(
            linear_ops_file,
            input_layernorm_tp_key,
            required_columns=["time_stats.input_layernorm.median"],
            training_context=training_context,
        )
        input_layernorm_context = dict(training_context)
        input_layernorm_context["input_file"] = linear_ops_file
        input_layernorm_context["tensor_parallel_size"] = input_layernorm_tp_key

        layernorm_model_name = "input_layernorm"
        layernorm_signature = f"{layernorm_model_name}_{attention_signature}"
        if layernorm_signature not in trained_model_signatures:
            models[layernorm_model_name] = self._train_single_model(
                model_name=layernorm_model_name,
                df=input_layernorm_df,
                feature_cols=["num_tokens"],
                target_col=f"time_stats.{layernorm_model_name}.median",
                execution_time_predictor_config=execution_time_predictor_config,
                training_context=input_layernorm_context,
            )
            trained_model_signatures.add(layernorm_signature)
            logger.info(f"Trained {layernorm_model_name} for {cluster_type}")

        # Attention projections: attn_pre_proj, attn_post_proj, attn_rope
        attn_proj_context = dict(training_context)
        attn_proj_context["input_file"] = linear_ops_file
        attn_proj_context["tensor_parallel_size"] = attn_tp_key
        attn_proj_model_names = ["attn_pre_proj", "attn_post_proj", "attn_rope"]
        for model_name in attn_proj_model_names:
            model_signature = f"{model_name}_{attention_signature}"
            if model_signature not in trained_model_signatures:
                models[model_name] = self._train_single_model(
                    model_name=model_name,
                    df=attn_linear_ops_df,
                    feature_cols=["num_tokens"],
                    target_col=f"time_stats.{model_name}.median",
                    execution_time_predictor_config=execution_time_predictor_config,
                    training_context=attn_proj_context,
                )
                trained_model_signatures.add(model_signature)
                logger.info(f"Trained {model_name} for {cluster_type}")

        if is_target_embedded_mtp_enabled(
            getattr(replica_config, "speculative_decoding_config", None)
        ):
            required_mtp_columns = (
                self._get_required_target_embedded_mtp_linear_op_columns()
            )
            missing_mtp_columns = [
                col for col in required_mtp_columns if col not in attn_linear_ops_df.columns
            ]
            all_nan_mtp_columns = [
                col
                for col in required_mtp_columns
                if col in attn_linear_ops_df.columns
                and attn_linear_ops_df[col].isna().all()
            ]
            if missing_mtp_columns or all_nan_mtp_columns:
                raise ValueError(
                    "target-embedded MTP compute profiling columns are missing or all-NaN in "
                    f"{linear_ops_file}. "
                    f"Missing columns: {missing_mtp_columns}. "
                    f"All-NaN columns: {all_nan_mtp_columns}. "
                    "Re-run linear-op profiling with --include_target_embedded_mtp."
                )
            for model_name in ["mtp_fusion_proj", "lm_head_linear"]:
                model_signature = f"{model_name}_{attention_signature}"
                if model_signature not in trained_model_signatures:
                    models[model_name] = self._train_single_model(
                        model_name=model_name,
                        df=attn_linear_ops_df,
                        feature_cols=["num_tokens"],
                        target_col=f"time_stats.{model_name}.median",
                        execution_time_predictor_config=execution_time_predictor_config,
                        training_context=attn_proj_context,
                    )
                    trained_model_signatures.add(model_signature)
                    logger.info(
                        "Trained %s for %s (target-embedded MTP)",
                        model_name,
                        cluster_type,
                    )

        # Step2Mini-specific attention operations (forward_1: inter_norm + wq after Q split)
        # These operations are unique to Step2Mini architecture and should only be trained
        # when model_arch == "step2_mini"
        model_config = replica_config.model_config
        if model_config is not None and model_config.is_step2_mini():
            step2mini_attn_model_names = ["attn_inter_norm", "attn_wq_proj"]
            for model_name in step2mini_attn_model_names:
                model_signature = f"{model_name}_{attention_signature}"
                if model_signature not in trained_model_signatures:
                    target_col = f"time_stats.{model_name}.median"
                    if target_col not in attn_linear_ops_df.columns:
                        raise ValueError(
                            f"Step2Mini operation '{model_name}' column '{target_col}' not found in profiling data. "
                            f"Ensure profiling was run with Step2Mini model architecture. "
                            f"Available columns: {list(attn_linear_ops_df.columns)}"
                        )
                    models[model_name] = self._train_single_model(
                        model_name=model_name,
                        df=attn_linear_ops_df,
                        feature_cols=["num_tokens"],
                        target_col=target_col,
                        execution_time_predictor_config=execution_time_predictor_config,
                        training_context=attn_proj_context,
                    )
                    trained_model_signatures.add(model_signature)
                    logger.info(f"Trained Step2Mini {model_name} for {cluster_type}")

        # ========== Part 2: Attention core operations from attention.csv ==========
        if not os.path.exists(attn_file):
            raise FileNotFoundError(f"Attention input file {attn_file} not found")

        logger.info(f"Loading attention data from: {attn_file}")
        attention_df = self._load_attention_df(
            attn_file,
            replica_config,
            replica_scheduler_config,
            cluster_type=cluster_type,
        )
        attention_df = self._get_attention_df_with_derived_features(attention_df)
        logger.info(f"Loaded {len(attention_df)} rows for attention core training")
        training_context['input_file'] = attn_file
        measurement_type = self._active_measurement_type
        dense_attention_model_names = get_enabled_predictor_metric_names(
            DENSE_ATTENTION_FAMILY
        )
        dense_attention_target_columns = dict(
            zip(
                dense_attention_model_names,
                get_enabled_predictor_median_columns(DENSE_ATTENTION_FAMILY),
            )
        )
        dense_attention_feature_columns = get_enabled_shared_predictor_feature_columns(
            DENSE_ATTENTION_FAMILY
        )

        # Train kv_cache_save model
        kv_cache_model_name = get_enabled_predictor_metric_name_by_role(
            DENSE_ATTENTION_FAMILY,
            AttentionOperatorRole.CACHE_WRITE,
        )
        kv_cache_model_signature = f"{kv_cache_model_name}_{attention_signature}"
        if kv_cache_model_signature not in trained_model_signatures:
            kv_cache_feature_cols = list(
                dense_attention_feature_columns[kv_cache_model_name]
            )
            missing_cols = [
                col for col in kv_cache_feature_cols if col not in attention_df.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Missing columns for {kv_cache_model_name} training: {missing_cols}. "
                    "Re-run attention profiling with mixed-batch metadata."
                )
            models[kv_cache_model_name] = self._train_single_model(
                model_name=kv_cache_model_name,
                df=attention_df,
                feature_cols=kv_cache_feature_cols,
                target_col=dense_attention_target_columns[kv_cache_model_name],
                execution_time_predictor_config=execution_time_predictor_config,
                training_context=training_context,
            )
            trained_model_signatures.add(kv_cache_model_signature)
            logger.info(f"Trained {kv_cache_model_name} for {cluster_type}")

        # Split data for prefill and decode.
        # Mixed-batch prefill rows in attention_combined.csv use prefill_chunk_size=0,
        # so standard prefill training must keep only rows with positive chunk size.
        true_mixed_df = attention_df[attention_df["is_true_mixed_batch"]].copy()
        standard_df = attention_df[~attention_df["is_true_mixed_batch"]].copy()
        prefill_df = standard_df[~standard_df["is_decode"]].copy()
        decode_df = standard_df[standard_df["is_decode"]].copy()
        standard_prefill_df = pd.DataFrame()
        if measurement_type == MeasurementType.CUDA_EVENT:
            if "prefill_chunk_size" not in prefill_df.columns:
                raise ValueError(
                    "Missing required column 'prefill_chunk_size' in attention profiling data."
                )
            standard_prefill_df = prefill_df[prefill_df["prefill_chunk_size"] > 0].copy()

            prefill_model_name = get_enabled_predictor_metric_name_by_role(
                DENSE_ATTENTION_FAMILY,
                AttentionOperatorRole.PREFILL_KERNEL,
            )
            prefill_model_signature = f"{prefill_model_name}_{attention_signature}"
            if prefill_model_signature not in trained_model_signatures:
                if len(standard_prefill_df) == 0:
                    raise ValueError(
                        "No standard prefill rows (prefill_chunk_size > 0) found in eager attention profiling data."
                    )
                models[prefill_model_name] = self._train_single_model(
                    model_name=prefill_model_name,
                    df=standard_prefill_df,
                    feature_cols=list(dense_attention_feature_columns[prefill_model_name]),
                    target_col=dense_attention_target_columns[prefill_model_name],
                    execution_time_predictor_config=execution_time_predictor_config,
                    training_context=training_context,
                )
                trained_model_signatures.add(prefill_model_signature)
                logger.info(f"Trained {prefill_model_name} for {cluster_type}")

            decode_model_name = get_enabled_predictor_metric_name_by_role(
                DENSE_ATTENTION_FAMILY,
                AttentionOperatorRole.DECODE_KERNEL,
            )
            decode_model_signature = f"{decode_model_name}_{attention_signature}"
            if decode_model_signature not in trained_model_signatures:
                if len(decode_df) == 0:
                    logger.info(
                        "Skipping eager %s training for %s - no standard decode rows",
                        decode_model_name,
                        cluster_type,
                    )
                else:
                    decode_feature_cols = list(
                        dense_attention_feature_columns[decode_model_name]
                    )
                    missing_decode_cols = [
                        col
                        for col in [
                            *decode_feature_cols,
                            dense_attention_target_columns[decode_model_name],
                        ]
                        if col not in decode_df.columns
                    ]
                    if missing_decode_cols:
                        logger.info(
                            "Skipping eager %s training for %s - missing decode feature columns %s",
                            decode_model_name,
                            cluster_type,
                            missing_decode_cols,
                        )
                    else:
                        models[decode_model_name] = self._train_single_model(
                            model_name=decode_model_name,
                            df=decode_df,
                            feature_cols=decode_feature_cols,
                            target_col=dense_attention_target_columns[decode_model_name],
                            execution_time_predictor_config=execution_time_predictor_config,
                            training_context=training_context,
                        )
                        trained_model_signatures.add(decode_model_signature)
                        logger.info(f"Trained eager {decode_model_name} for {cluster_type}")
        elif measurement_type == MeasurementType.KERNEL_ONLY:
            decode_model_name = get_enabled_predictor_metric_name_by_role(
                DENSE_ATTENTION_FAMILY,
                AttentionOperatorRole.DECODE_KERNEL,
            )
            decode_model_signature = f"{decode_model_name}_{attention_signature}"
            if decode_model_signature not in trained_model_signatures:
                if len(decode_df) == 0:
                    raise ValueError(
                        "No standard decode rows found in kernel-only attention profiling data."
                    )
                models[decode_model_name] = self._train_single_model(
                    model_name=decode_model_name,
                    df=decode_df,
                    feature_cols=list(dense_attention_feature_columns[decode_model_name]),
                    target_col=dense_attention_target_columns[decode_model_name],
                    execution_time_predictor_config=execution_time_predictor_config,
                    training_context=training_context,
                )
                trained_model_signatures.add(decode_model_signature)
                logger.info(f"Trained {decode_model_name} for {cluster_type}")
        else:
            raise ValueError(f"Unsupported measurement_type={measurement_type!r}")

        # ========== Part 3: Mixed-batch prefill model (optional, high-dimensional) ==========
        # attn_prefill_mixed uses 12 features and requires on-demand prediction at runtime
        # Check if profiling data contains mixed-batch features
        mixed_batch_model_signature = f"attn_prefill_mixed_{attention_signature}"
        if measurement_type == MeasurementType.CUDA_EVENT and mixed_batch_model_signature not in trained_model_signatures:
            # Check for mixed-batch specific columns in the dataframe
            required_mixed_features = self.ATTN_PREFILL_MIXED_FEATURES
            has_mixed_batch_data = all(feat in prefill_df.columns for feat in required_mixed_features)
            
            if has_mixed_batch_data:
                logger.info(f"Training attn_prefill_mixed with {len(required_mixed_features)} features for {cluster_type}")
                
                # Filter for mixed-prefill rows (exclude true mixed prefill+decode rows)
                mixed_batch_df = prefill_df[
                    prefill_df["is_mixed_batch"] | (prefill_df["batch_size"] > 1)
                ].copy()
                
                if len(mixed_batch_df) > 0:
                    models["attn_prefill_mixed"] = self._train_single_model(
                        model_name="attn_prefill_mixed",
                        df=mixed_batch_df,
                        feature_cols=required_mixed_features,
                        target_col="time_stats.attn_prefill.median",  # Same target column as attn_prefill
                        execution_time_predictor_config=execution_time_predictor_config,
                        training_context=training_context,
                    )
                    trained_model_signatures.add(mixed_batch_model_signature)
                    logger.info(f"Trained attn_prefill_mixed with {len(mixed_batch_df)} samples for {cluster_type}")
                else:
                    logger.warning(f"No mixed-batch data (batch_size > 1) available for attn_prefill_mixed in {cluster_type}")
            else:
                missing_features = [f for f in required_mixed_features if f not in prefill_df.columns]
                logger.info(f"Skipping attn_prefill_mixed for {cluster_type} - missing features: {missing_features}")

        decode_in_mixed_signature = f"attn_decode_in_mixed_{attention_signature}"
        if measurement_type == MeasurementType.CUDA_EVENT and decode_in_mixed_signature not in trained_model_signatures:
            required_decode_mixed_features = self.ATTN_DECODE_IN_MIXED_FEATURES
            has_decode_mixed_data = all(
                feat in true_mixed_df.columns for feat in required_decode_mixed_features
            )
            if has_decode_mixed_data:
                if len(true_mixed_df) > 0:
                    models["attn_decode_in_mixed"] = self._train_single_model(
                        model_name="attn_decode_in_mixed",
                        df=true_mixed_df,
                        feature_cols=required_decode_mixed_features,
                        target_col="time_stats.attn_decode.median",
                        execution_time_predictor_config=execution_time_predictor_config,
                        training_context=training_context,
                    )
                    trained_model_signatures.add(decode_in_mixed_signature)
                    logger.info(
                        f"Trained attn_decode_in_mixed with {len(true_mixed_df)} samples for {cluster_type}"
                    )
                else:
                    logger.info(
                        f"Skipping attn_decode_in_mixed for {cluster_type} - no true mixed rows"
                    )
            else:
                missing_features = [
                    f for f in required_decode_mixed_features if f not in true_mixed_df.columns
                ]
                logger.info(
                    f"Skipping attn_decode_in_mixed for {cluster_type} - missing features: {missing_features}"
                )

        trained_model_signatures.add(attention_signature)
        return models

    def _train_residual_models_for_cluster(self, cluster_type: ClusterType, replica_config, execution_time_predictor_config,
                                           linear_ops_file: str, trained_model_signatures: set) -> Dict[str, BaseEstimator]:
        """
        Train residual connection models for a cluster.

        This function handles residual connection operations in the Transformer layer:
        - Residual add operation (from linear_op.csv): add

        Transformer layer context:
            ... → Attention → [add] → LayerNorm → FFN/MoE → [add] → ...

        The residual add operation is used after both Attention and FFN blocks,
        making it a common operation that serves both sub-layers.
        """
        models = {}

        model_config = replica_config.model_config

        # RMSNorm: add is fused into layernorm, no separate add model needed
        if model_config is not None and model_config.uses_fused_add_norm:
            logger.info(f"Skipping residual add model training for {cluster_type} "
                        f"— model uses fused add+norm (RMSNorm)")
            return models

        is_moe_model = model_config is not None and model_config.is_moe
        tp_size = self._get_linear_op_tp_key(
            "add",
            cluster_type,
            replica_config,
            is_moe_model,
        )

        # Create a signature for this residual model configuration
        residual_signature = f"residual_{replica_config.device}_{replica_config.model_name}_{tp_size}_family{self._measurement_family_name(self._active_measurement_type)}"

        if residual_signature in trained_model_signatures:
            logger.info(f"Skipping residual models training for {cluster_type} - already trained with signature {residual_signature}")
            return models

        if not os.path.exists(linear_ops_file):
            raise FileNotFoundError(f"Linear ops input file {linear_ops_file} not found for residual models")

        logger.info(f"Loading linear ops data for residual models from: {linear_ops_file}")
        linear_ops_df = self._load_linear_op_df(linear_ops_file, tp_size)
        logger.info(f"Loaded {len(linear_ops_df)} rows for residual training")

        # Build training context for error messages
        training_context = {
            'cluster_type': str(cluster_type),
            'device': replica_config.device,
            'model_name': replica_config.model_name,
            'tensor_parallel_size': tp_size,
            'input_file': linear_ops_file,
        }

        # Train the residual add model
        add_model_name = "add"
        add_signature = f"{add_model_name}_{residual_signature}"
        if add_signature not in trained_model_signatures:
            models[add_model_name] = self._train_single_model(
                model_name=add_model_name,
                df=linear_ops_df,
                feature_cols=["num_tokens"],
                target_col=f"time_stats.{add_model_name}.median",
                execution_time_predictor_config=execution_time_predictor_config,
                training_context=training_context,
            )
            trained_model_signatures.add(add_signature)
            logger.info(f"Trained {add_model_name} for {cluster_type}")

        # Mark this residual configuration as trained
        trained_model_signatures.add(residual_signature)
        return models

    def _train_pipeline_parallel_models_for_cluster(self, cluster_type: ClusterType, replica_config, execution_time_predictor_config, trained_model_signatures: set) -> Dict[str, BaseEstimator]:
        """Train pipeline parallel communication models for a cluster."""
        models = {}

        _, _, _, send_recv_input_file, _, _ = self._get_input_files_for_config(replica_config, execution_time_predictor_config)
        
        pp_signature = f"send_recv_{replica_config.network_device}_{replica_config.num_pipeline_stages}_{replica_config.attn_tensor_parallel_size}_family{self._measurement_family_name(self._active_measurement_type)}"
        
        if pp_signature in trained_model_signatures:
            logger.info(f"Skipping send_recv model training for {cluster_type} - already trained")
            return models
        
        send_recv_df = self._load_send_recv_df(send_recv_input_file, replica_config)
        send_recv_df = self._get_send_recv_df_with_derived_features(send_recv_df, replica_config)

        # Build training context for error messages
        training_context = {
            'cluster_type': str(cluster_type),
            'device': replica_config.device,
            'model_name': replica_config.model_name,
            'pipeline_stages': replica_config.num_pipeline_stages,
            'tensor_parallel_size': replica_config.attn_tensor_parallel_size,
            'network_device': replica_config.network_device,
            'input_file': send_recv_input_file,
        }

        models["send_recv"] = self._train_single_model(
            model_name="send_recv",
            df=send_recv_df,
            feature_cols=["num_tokens"],
            target_col="time_stats.send_recv.median",
            execution_time_predictor_config=execution_time_predictor_config,
            training_context=training_context,
        )

        trained_model_signatures.add(pp_signature)
        return models

    def _train_tensor_parallel_models_for_cluster(self, cluster_type: ClusterType, replica_config, execution_time_predictor_config, use_attn_tp: bool, trained_model_signatures: set) -> Dict[str, BaseEstimator]:
        """Train tensor parallel communication models for a cluster."""
        models = {}

        _, _, all_reduce_input_file, _, _, _ = self._get_input_files_for_config(replica_config, execution_time_predictor_config)
        
        # Use different tensor parallel size based on cluster type
        tp_size = replica_config.attn_tensor_parallel_size if use_attn_tp else replica_config.moe_tensor_parallel_size
        
        tp_signature = f"all_reduce_{replica_config.network_device}_{tp_size}_family{self._measurement_family_name(self._active_measurement_type)}"
        
        if tp_signature in trained_model_signatures:
            logger.info(f"Skipping all_reduce model training for {cluster_type} - already trained")
            return models
        
        # 添加详细的上下文信息
        training_context = {
            'cluster_type': cluster_type,
            'device': replica_config.device,
            'model_name': replica_config.model_name,
            'tensor_parallel_size': tp_size,
            'network_device': replica_config.network_device,
            'input_file': all_reduce_input_file,
            'use_attn_tp': use_attn_tp
        }
        
        logger.info(f"Loading all_reduce data for {cluster_type}: file={all_reduce_input_file}, tp_size={tp_size}")
        
        all_reduce_df = self._load_all_reduce_df(all_reduce_input_file, replica_config, tp_size)
        logger.info(f"Loaded {len(all_reduce_df)} rows for all_reduce training")
        
        all_reduce_df = self._get_all_reduce_df_with_derived_features(all_reduce_df, replica_config)
        logger.info(f"After feature engineering: {len(all_reduce_df)} rows")
        
        models["all_reduce"] = self._train_single_model(
            model_name="all_reduce",
            df=all_reduce_df,
            feature_cols=["num_tokens"],
            target_col="time_stats.all_reduce.median",
            execution_time_predictor_config=execution_time_predictor_config,
            training_context=training_context
        )
        
        trained_model_signatures.add(tp_signature)
        return models

    def _train_cpu_overhead_models_for_cluster(self, cluster_type: ClusterType, replica_config, execution_time_predictor_config, trained_model_signatures: set) -> Dict[str, BaseEstimator]:
        """Train CPU overhead models for a cluster."""
        models = {}

        if execution_time_predictor_config.skip_cpu_overhead_modeling:
            return models

        _, _, _, _, cpu_overhead_input_file, _ = self._get_input_files_for_config(replica_config, execution_time_predictor_config)
        
        cpu_signature = f"cpu_overhead_{replica_config.network_device}_{replica_config.model_name}_{replica_config.attn_tensor_parallel_size}_family{self._measurement_family_name(self._active_measurement_type)}"
        
        if cpu_signature in trained_model_signatures:
            logger.info(f"Skipping CPU overhead models training for {cluster_type} - already trained")
            return models
        
        cpu_overhead_df = self._load_cpu_overhead_df(cpu_overhead_input_file, replica_config)
        if cpu_overhead_df.empty:
            logger.warning(
                "Skipping CPU overhead model training for cluster %s due to missing/empty CPU overhead profiling data. file=%s",
                cluster_type,
                cpu_overhead_input_file,
            )
            trained_model_signatures.add(cpu_signature)
            return models

        # Build training context for error messages
        training_context = {
            'cluster_type': str(cluster_type),
            'device': replica_config.device,
            'model_name': replica_config.model_name,
            'tensor_parallel_size': replica_config.attn_tensor_parallel_size,
            'network_device': replica_config.network_device,
            'input_file': cpu_overhead_input_file,
        }

        model_names = [
            "schedule",
            "sampler_e2e",
            "prepare_inputs_e2e",
            "process_model_outputs",
            "ray_comm_time",
        ]

        for model_name in model_names:
            target_col = "ray_comm_time_mean" if model_name == "ray_comm_time" else f"{model_name}_median"

            model_signature = f"{model_name}_{cpu_signature}"
            if model_signature not in trained_model_signatures:
                feature_cols = [
                    "batch_size",
                    "num_prefill_tokens",
                    "num_decode_tokens",
                ]
                model = self._train_single_model(
                    model_name=model_name,
                    df=cpu_overhead_df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    execution_time_predictor_config=execution_time_predictor_config,
                    training_context=training_context,
                )
                model._frontier_exact_lookup = _build_exact_feature_lookup(
                    cpu_overhead_df,
                    feature_cols,
                    target_col,
                )
                models[model_name] = model
                trained_model_signatures.add(model_signature)

        trained_model_signatures.add(cpu_signature)
        return models

    def _train_single_model(self, model_name: str, df: pd.DataFrame, feature_cols: List[str], target_col: str, execution_time_predictor_config, training_context: Dict[str, Any] = None) -> BaseEstimator:
        """Train a single model with given data and configuration."""
        if len(df) == 0:
            # 提供详细的错误信息，以便调试
            context_info = ""
            if training_context:
                context_info = f"""
                Training Context:
                - Cluster Type: {training_context.get('cluster_type', 'Unknown')}
                - Device: {training_context.get('device', 'Unknown')}
                - Model Name: {training_context.get('model_name', 'Unknown')}
                - Pipeline Stages: {training_context.get('pipeline_stages', 'Unknown')}
                - Network Device: {training_context.get('network_device', 'Unknown')}
                - Tensor Parallel Size: {training_context.get('tensor_parallel_size', 'Unknown')}
                - Input File: {training_context.get('input_file', 'Unknown')}
                - Block Size: {training_context.get('block_size', 'Unknown')}
                - Feature Columns: {feature_cols}
                - Target Column: {target_col}
                """

            raise Exception(f"Training data for model {model_name} is empty.{context_info}")

        required_cols = feature_cols + [target_col]
        nan_row_mask = df[required_cols].isna().any(axis=1)
        nan_row_count = int(nan_row_mask.sum())
        if nan_row_count > 0:
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
                f"Training data for model {model_name} is empty after dropping NaN rows "
                f"(target={target_col})."
            )

        profiling_precision = self._get_profiling_precision_from_df(df)
        measurement_type = self._validate_active_measurement_type(df)
        model_hash = self._get_model_hash(
            model_name,
            df,
            execution_time_predictor_config,
            profiling_precision,
            measurement_type,
        )
        cached_model = self._load_model_from_cache(model_name, model_hash)
        if cached_model:
            self._store_model_precision(model_name, profiling_precision, cached_model)
            return cached_model

        # ============================================================
        # CACHE MISS: Model not found in cache
        # ============================================================
        # When running in production mode (non-dummy mode), we expect all models
        # to be pre-trained using the standalone training module and cached.
        # If a model is not found in cache, it indicates a configuration mismatch
        # or missing profiling/training step.
        #
        # To train models, use the standalone training workflow:
        # 1. Run profiling: tests/test_pd_af_profiling.sh
        # 2. Run training: tests/test_pd_af_training.sh
        # 3. Run simulation: tests/test_small_scale_pd_af_disaggregation_cluster_parallel.sh
        # ============================================================

        error_msg = f"""
        ❌ MODEL CACHE MISS ERROR ❌

        Model '{model_name}' with hash '{model_hash}' not found in cache directory: {self._cache_dir}

        Configuration Details:
        - Model Name: {model_name}
        - Cache Hash: {model_hash}
        - Cache Directory: {self._cache_dir}
        - Expected Cache File: {self._cache_dir}/{model_name}_{model_hash}.pkl
        """

        if training_context:
            error_msg += f"""
        Training Context:
        - Cluster Type: {training_context.get('cluster_type', 'Unknown')}
        - Device: {training_context.get('device', 'Unknown')}
        - Model Name: {training_context.get('model_name', 'Unknown')}
        - Tensor Parallel Size: {training_context.get('tensor_parallel_size', 'Unknown')}
        - Expert Parallel Size: {training_context.get('moe_expert_parallel_size', 'N/A')}
        - Input File: {training_context.get('input_file', 'Unknown')}
        - Feature Columns: {feature_cols}
        - Target Column: {target_col}
        """

        error_msg += f"""

        ⚠️  REQUIRED ACTION ⚠️

        This error indicates that the required model has not been pre-trained.
        Please follow the complete workflow:

        ============================================================

        NOTE: Real-time training is TEMPORARILY ENABLED for cache generation.
        """

        logger.warning(error_msg)
        logger.info(f"CACHE MISS: Training model '{model_name}' with hash '{model_hash}' in real-time...")

        # ============================================================
        # TEMPORARILY ENABLED: Real-time training code
        # ============================================================
        # This code performs real-time model training during simulation
        # initialization to generate missing cache files.
        # ============================================================

        estimator, grid_search_params = self._create_estimator_and_params(execution_time_predictor_config)

        cv = min(execution_time_predictor_config.k_fold_cv_splits, len(df)) if len(df) >= 2 else 2

        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=grid_search_params,
            scoring=self._get_scorer(),
            cv=cv,
            n_jobs=execution_time_predictor_config.num_training_job_threads,
        )

        X, y = df[feature_cols], df[target_col]
        grid_search.fit(X, y)
        score = grid_search.score(X, y)

        logger.info(f"✓ Trained model {model_name} with MAPE {-score}%")

        best_estimator = grid_search.best_estimator_
        # Persist feature metadata for runtime on-demand prediction (e.g., moe_grouped_gemm load imbalance mode).
        setattr(best_estimator, "_frontier_feature_names", list(feature_cols))
        setattr(best_estimator, "_frontier_target_col", target_col)
        # Tie the trained estimator to its cache hash so prediction caches can include model identity.
        setattr(best_estimator, "_frontier_model_hash", model_hash)

        self._store_model_in_cache(model_name, model_hash, best_estimator)
        self._store_model_precision(model_name, profiling_precision, best_estimator)
        return best_estimator

    # ========================================================================
    # Data Loading Methods
    # ========================================================================
    # These methods load profiling data from CSV files and apply filtering.
    #
    # Data source mapping:
    # - linear_op.csv (or mlp.csv for backward compatibility):
    #   - Attention projections: attn_pre_proj, attn_post_proj, attn_rope
    #   - MLP operations: mlp_up_proj, mlp_down_proj, mlp_act
    #   - LayerNorm operations: input_layernorm, post_attention_layernorm
    #   - Residual operations: add
    #
    # - attention.csv:
    #   - Attention core: attn_kv_cache_save, attn_prefill, attn_decode
    #
    # - moe.csv:
    #   - MoE operations: moe_gating_linear, moe_gating_routing_topk, moe_shuffling, moe_grouped_gemm
    # ========================================================================

    def _load_linear_op_df(
        self,
        file_path: str,
        tensor_parallel_size: int,
        required_columns: Optional[List[str]] = None,
        training_context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Load linear operation dataframe (linear_op.csv or mlp.csv) with tensor parallel filtering.

        This function loads profiling data for linear operations including:
        - Attention projections: attn_pre_proj, attn_post_proj, attn_rope
        - MLP operations: mlp_up_proj, mlp_down_proj, mlp_act
        - LayerNorm operations: input_layernorm, post_attention_layernorm
        - Residual operations: add

        Note: This function is for linear_op.csv data only. For MoE data, use _load_moe_df().

        Args:
            file_path: Path to the profiling CSV file (linear_op.csv or mlp.csv)
            tensor_parallel_size: Required tensor parallel size for filtering

        Returns:
            Filtered DataFrame

        Raises:
            FileNotFoundError: If the input file does not exist
            ValueError: If required columns are missing or no data matches filtering criteria
        """
        # Check file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Linear ops input file does not exist: {file_path}\n"
                f"Please run profiling first to generate this file.\n"
                f"Suggested command: bash frontier/profiling/example/test_profiling_linear_op.sh"
            )

        df = pd.read_csv(file_path)
        logger.info(f"Original linear ops data: {len(df)} rows, {len(df.columns)} columns")

        # Check required column
        if 'num_tensor_parallel_workers' not in df.columns:
            raise ValueError(
                f"Column 'num_tensor_parallel_workers' not found in {file_path}\n"
                f"Available columns: {list(df.columns)}\n"
                f"This may indicate a corrupted or incompatible profiling file."
            )

        # Show filtering conditions
        available_tp = sorted(df['num_tensor_parallel_workers'].unique())
        logger.info(f"Filtering conditions:")
        logger.info(f"  - num_tensor_parallel_workers == {tensor_parallel_size}")
        logger.info(f"  - Available num_tensor_parallel_workers: {available_tp}")

        # Apply filtering
        filtered_df = df[df["num_tensor_parallel_workers"] == tensor_parallel_size]
        logger.info(f"After filtering: {len(filtered_df)} rows")

        expected_use_qk_norm = None
        if training_context is not None and "use_qk_norm" in training_context:
            expected_use_qk_norm = bool(training_context["use_qk_norm"])

        if expected_use_qk_norm is True and "use_qk_norm" not in filtered_df.columns:
            raise ValueError(
                "linear_op profiling data is missing 'use_qk_norm' column for a model "
                "that requires QK-norm-aware filtering. "
                f"file={file_path}, model={training_context.get('model_name') if training_context else 'unknown'}"
            )

        if expected_use_qk_norm is not None and "use_qk_norm" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["use_qk_norm"].astype(bool) == expected_use_qk_norm
            ]
            logger.info(
                "After use_qk_norm filtering: %s rows (expected_use_qk_norm=%s)",
                len(filtered_df),
                expected_use_qk_norm,
            )

        if len(filtered_df) == 0:
            raise ValueError(
                f"No data matches the filtering criteria in {file_path}\n"
                f"Required tensor_parallel_size: {tensor_parallel_size}\n"
                f"Available tensor_parallel_sizes: {available_tp}\n"
                f"Please run profiling with the correct configuration."
            )

        if required_columns:
            self._validate_required_linear_op_columns(
                filtered_df,
                required_columns,
                file_path,
                training_context=training_context,
            )

        return filtered_df

    def _get_required_attn_linear_op_columns(self, model_config) -> List[str]:
        required_columns = [
            "time_stats.attn_pre_proj.median",
            "time_stats.attn_post_proj.median",
            "time_stats.attn_rope.median",
        ]
        if model_config is not None and bool(getattr(model_config, "use_qk_norm", False)):
            required_columns.append("use_qk_norm")
        if model_config is not None and model_config.is_step2_mini():
            required_columns.extend(
                ["time_stats.attn_inter_norm.median", "time_stats.attn_wq_proj.median"]
            )
        return required_columns

    @staticmethod
    def _get_required_target_embedded_mtp_linear_op_columns() -> List[str]:
        return [
            "time_stats.mtp_fusion_proj.median",
            "time_stats.lm_head_linear.median",
        ]

    @staticmethod
    def _validate_required_linear_op_columns(
        df: pd.DataFrame,
        required_columns: List[str],
        file_path: str,
        training_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        all_nan_columns = [
            col
            for col in required_columns
            if col in df.columns and df[col].isna().all()
        ]

        if missing_columns or all_nan_columns:
            context_text = ""
            if training_context:
                context_text = f"\nTraining context: {training_context}"

            raise ValueError(
                "Required attention linear op columns are missing or all-NaN in "
                f"{file_path}."
                f"\nMissing columns: {missing_columns}"
                f"\nAll-NaN columns: {all_nan_columns}"
                f"{context_text}"
            )

    def _load_attention_df(
        self,
        file_path: str,
        replica_config,
        replica_scheduler_config,
        cluster_type: Optional[ClusterType] = None,
    ) -> pd.DataFrame:
        """
        Load attention dataframe (attention.csv) with model configuration filtering.

        Args:
            file_path: Path to the attention profiling CSV file
            replica_config: Replica configuration for filtering
            replica_scheduler_config: Replica scheduler configuration for block size
            cluster_type: Cluster type for policy warning context

        Returns:
            Filtered DataFrame

        Raises:
            FileNotFoundError: If the input file does not exist
            ValueError: If no data matches filtering criteria
        """
        # Check file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Attention input file does not exist: {file_path}\n"
                f"Please run attention profiling first to generate this file.\n"
                f"Suggested command: bash frontier/profiling/example/test_profiling_attention.sh"
            )

        df = pd.read_csv(file_path)
        df = df.drop_duplicates()
        logger.info(f"Original attention data: {len(df)} rows, {len(df.columns)} columns")

        enforce_mixed_attention_input_contract(
            attention_file_path=file_path,
            available_columns=df.columns,
        )

        # Fill missing cache-write column for older attention profiling CSVs.
        cache_write_median_column = get_enabled_predictor_median_column_by_role(
            DENSE_ATTENTION_FAMILY,
            AttentionOperatorRole.CACHE_WRITE,
        )
        for column in [cache_write_median_column]:
            if column not in df.columns:
                df[column] = 0
            else:
                df.fillna({column: 0}, inplace=True)

        model_config = replica_config.model_config
        requested_tp = replica_config.attn_tensor_parallel_size
        prefill_op_name = get_enabled_predictor_metric_name_by_role(
            DENSE_ATTENTION_FAMILY,
            AttentionOperatorRole.PREFILL_KERNEL,
        )
        effective_tp = resolve_effective_attention_tp_size(
            op_name=prefill_op_name,
            requested_tp_size=requested_tp,
            num_kv_heads=model_config.num_kv_heads,
            cluster_type=cluster_type,
            warning_cache=getattr(self, "_attention_tp_warning_cache", None),
            include_linear_ops=False,
        )

        # Show filtering conditions
        logger.info(f"Filtering conditions:")
        logger.info(f"  - n_embd == {model_config.embedding_dim}")
        logger.info(f"  - n_q_head == {model_config.num_q_heads}")
        logger.info(f"  - n_kv_head == {model_config.num_kv_heads}")
        logger.info(f"  - block_size == {replica_scheduler_config.block_size}")
        logger.info(
            "  - num_tensor_parallel_workers == %s (requested_tp=%s)",
            effective_tp,
            requested_tp,
        )

        filtered_df = df[
            (df["n_embd"] == model_config.embedding_dim)
            & (df["n_q_head"] == model_config.num_q_heads)
            & (df["n_kv_head"] == model_config.num_kv_heads)
            & (df["block_size"] == replica_scheduler_config.block_size)
            & (df["num_tensor_parallel_workers"] == effective_tp)
        ]

        logger.info(f"After filtering: {len(filtered_df)} rows")

        if len(filtered_df) == 0:
            # Surface what is available to make debugging explicit.
            available = {
                "n_embd": sorted(df["n_embd"].unique().tolist()) if "n_embd" in df else [],
                "n_q_head": sorted(df["n_q_head"].unique().tolist()) if "n_q_head" in df else [],
                "n_kv_head": sorted(df["n_kv_head"].unique().tolist()) if "n_kv_head" in df else [],
                "block_size": sorted(df["block_size"].unique().tolist()) if "block_size" in df else [],
                "num_tensor_parallel_workers": sorted(df["num_tensor_parallel_workers"].unique().tolist()) if "num_tensor_parallel_workers" in df else [],
            }

            logger.error(
                "Attention profiling rows are missing for the requested configuration. "
                "Available values: %s", available
            )

            raise ValueError(
                f"No data matches the filtering criteria in {file_path}\n"
                f"Required configuration:\n"
                f"  - n_embd: {model_config.embedding_dim}\n"
                f"  - n_q_head: {model_config.num_q_heads}\n"
                f"  - n_kv_head: {model_config.num_kv_heads}\n"
                f"  - block_size: {replica_scheduler_config.block_size}\n"
                f"  - tensor_parallel_size(requested): {requested_tp}\n"
                f"  - tensor_parallel_size(effective): {effective_tp}\n"
                f"Available values: {available}\n"
                f"Please run attention profiling with the correct configuration."
            )

        return filtered_df

    def _load_all_reduce_df(self, file_path: str, replica_config, tensor_parallel_size: int) -> pd.DataFrame:
        """
        Load all_reduce dataframe with cluster-specific tensor parallel size.

        Args:
            file_path: Path to the communication profiling CSV file
            replica_config: Replica configuration
            tensor_parallel_size: Required tensor parallel size for filtering

        Returns:
            Filtered DataFrame

        Raises:
            FileNotFoundError: If the input file does not exist
            ValueError: If no data matches filtering criteria
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"All-reduce input file does not exist: {file_path}\n"
                f"Please run communication profiling first.\n"
                f"Suggested command: bash frontier/profiling/example/test_profiling_communication.sh"
            )

        df = pd.read_csv(file_path)
        logger.info(f"Original all_reduce data: {len(df)} rows")

        # Show filtering conditions
        logger.info(f"Filtering conditions:")
        logger.info(f"  - num_workers == {tensor_parallel_size}")
        logger.info(f"  - devices_per_node == {tensor_parallel_size}")
        logger.info(f"  - collective == 'all_reduce'")

        filtered_df = df[
            (df["num_workers"] == tensor_parallel_size)
            & (df["devices_per_node"] == tensor_parallel_size)
            & (df["collective"] == "all_reduce")
        ]

        logger.info(f"After filtering: {len(filtered_df)} rows")

        if len(filtered_df) == 0:
            available_info = ""
            if len(df) > 0:
                available_info = (
                    f"Available values in file:\n"
                    f"  - num_workers: {sorted(df['num_workers'].unique())}\n"
                    f"  - devices_per_node: {sorted(df['devices_per_node'].unique())}\n"
                    f"  - collective: {sorted(df['collective'].unique())}"
                )
            raise ValueError(
                f"No data matches the filtering criteria in {file_path}\n"
                f"Required: num_workers={tensor_parallel_size}, devices_per_node={tensor_parallel_size}, collective='all_reduce'\n"
                f"{available_info}"
            )

        return filtered_df

    def _load_send_recv_df(self, file_path: str, replica_config) -> pd.DataFrame:
        """
        Load send_recv dataframe for pipeline parallel communication.

        Args:
            file_path: Path to the communication profiling CSV file
            replica_config: Replica configuration

        Returns:
            Filtered DataFrame

        Raises:
            FileNotFoundError: If the input file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Send/recv input file does not exist: {file_path}\n"
                f"Please run communication profiling first.\n"
                f"Suggested command: bash frontier/profiling/example/test_profiling_communication.sh"
            )

        num_workers = replica_config.num_pipeline_stages * replica_config.attn_tensor_parallel_size
        devices_per_node = replica_config.node_config.num_devices_per_node
        is_multi_node = num_workers > devices_per_node

        if is_multi_node:
            devices_per_node = 1
        else:
            devices_per_node = 2

        df = pd.read_csv(file_path)
        logger.info(f"Original send_recv data: {len(df)} rows")
        logger.info(f"Filtering conditions: collective='send_recv', devices_per_node={devices_per_node}")

        filtered_df = df[
            (df["collective"] == "send_recv")
            & (df["devices_per_node"] == devices_per_node)
        ]

        logger.info(f"After filtering: {len(filtered_df)} rows")
        return filtered_df

    def _load_cpu_overhead_df(self, file_path: str, replica_config) -> pd.DataFrame:
        """
        Load CPU overhead dataframe with model configuration filtering.

        Args:
            file_path: Path to the CPU overhead profiling CSV file
            replica_config: Replica configuration

        Returns:
            Filtered DataFrame

        Raises:
            FileNotFoundError: If the input file does not exist
        """
        if not os.path.exists(file_path):
            logger.warning(
                "CPU overhead input file does not exist: %s. "
                "Skipping CPU overhead model training for this cluster.",
                file_path,
            )
            return pd.DataFrame()

        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning(
                "CPU overhead input file is empty: %s. "
                "Skipping CPU overhead model training for this cluster.",
                file_path,
            )
            return pd.DataFrame()

        df = apply_cpu_overhead_schema_v2_defaults(
            df,
            warn_fn=logger.warning,
            context=file_path,
        )
        df = validate_cpu_overhead_dataframe(df)

        model_config = replica_config.model_config

        logger.info(f"Original CPU overhead data: {len(df)} rows")
        logger.info(f"Filtering conditions: model_name='{model_config.get_name()}', tensor_parallel_degree={replica_config.attn_tensor_parallel_size}")

        filtered_df = df[
            (df["model_name"] == model_config.get_name())
            & (df["tensor_parallel_degree"] == replica_config.attn_tensor_parallel_size)
        ]

        logger.info(f"After filtering: {len(filtered_df)} rows")
        if filtered_df.empty:
            logger.warning(
                "No CPU overhead profiling rows found for model_name='%s', "
                "tensor_parallel_degree=%s in file '%s'.",
                model_config.get_name(),
                replica_config.attn_tensor_parallel_size,
                file_path,
            )
        return filtered_df

    # Load imbalance feature columns used for MoE training
    # These features describe the load distribution across experts
    # Reference: frontier/training/moe_trainer.py lines 224-239 (authoritative source)
    # Reference: frontier/profiling/moe/LOAD_IMBALANCE_GUIDE.md
    MOE_LOAD_IMBALANCE_FEATURES = [
        # Config features (6) - describe model configuration
        "total_routed_tokens",      # Total tokens after routing (num_tokens * router_topk)
        "num_experts_per_device",   # Number of experts per device after EP sharding
        "hidden_dim",               # Model hidden dimension
        "expert_hidden_dim",        # Expert FFN hidden dimension
        "router_topk",              # Number of experts each token is routed to
        "model_expansion_ratio",    # expert_hidden_dim / hidden_dim
        # Derived features (2) - derived from config and routing
        "tokens_per_expert_avg",    # Average tokens per expert
        "tokens_to_experts_ratio",  # tokens / num_experts ratio
        # Load features (6) - describe load distribution characteristics
        "expert_utilization",       # Proportion of experts with non-zero load
        "min_load_ratio",           # Min load / average load
        "load_imbalance_cv",        # Coefficient of Variation: std/mean, key imbalance metric
        "max_load_ratio",           # Max load / average load
        "load_entropy",             # Entropy of load distribution (higher = more uniform)
        "load_gini_coefficient",    # Gini coefficient: 0=equality, 1=inequality
    ]

    # Feature columns for mixed-batch attention prefill model
    # These features capture batch heterogeneity characteristics together with
    # the uniform KV-cache context used by MixedAttentionInput profiling.
    # Reference: frontier/training/attention_trainer.py lines 362-375 (authoritative source)
    ATTN_PREFILL_MIXED_FEATURES = [
        # Core features (7)
        "batch_size",               # Number of sequences in batch
        "kv_cache_size",            # Uniform KV cache context for the mixed batch
        "total_tokens",             # Total tokens across all sequences
        "avg_seq_len",              # Average sequence length
        "min_seq_len",              # Minimum sequence length
        "max_seq_len",              # Maximum sequence length
        "total_tokens_squared",     # Computational complexity proxy
        # Heterogeneity features (3)
        "seq_len_variance",         # Variance of sequence lengths
        "seq_len_cv",               # Coefficient of variation (std/mean)
        "seq_len_range",            # max_seq_len - min_seq_len
        # Interaction features (2)
        "batch_variance_interaction",   # batch_size * seq_len_variance
        "batch_cv_interaction",         # batch_size * seq_len_cv
    ]

    ATTN_DECODE_IN_MIXED_FEATURES = [
        "decode_batch_size",
        "decode_avg_kv_cache_size",
        "num_prefill_seqs",
        "total_prefill_tokens",
        "total_batch_size",
        "batch_composition_ratio",
        "total_tokens",
    ]

    def _load_moe_df(
        self,
        file_path: str,
        replica_config,
        load_imbalance: bool = True,
        tensor_parallel_size: Optional[int] = None,
        expert_parallel_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load MoE dataframe with cluster-specific configuration filtering.

        This function loads and filters MoE profiling data based on the model configuration
        and parallelism settings. It supports two training modes controlled by `load_imbalance`:

        1. **Load Imbalance Mode (default, load_imbalance=True)**:
           - Uses profiling data that includes load imbalance features
           - Training will use features like `load_imbalance_cv`, `load_gini_coefficient`, etc.
           - Recommended for accurate MoE execution time prediction under real-world scenarios
           - Requires profiling with `--enable_load_imbalance` flag

        2. **Standard Mode (load_imbalance=False)**:
           - Uses basic profiling data without load imbalance features
           - Training only uses `num_tokens` as feature
           - Simpler but less accurate for imbalanced workloads
           - Compatible with legacy profiling data

        The difference is in the **training features used**, not data row filtering.
        Load imbalance mode uses additional features to capture expert load distribution.

        Reference: frontier/profiling/moe/LOAD_IMBALANCE_GUIDE.md

        Args:
            file_path: Path to the MoE profiling CSV file
            replica_config: Replica configuration containing model and parallelism settings
            load_imbalance: Training mode flag:
                - True (default): Load imbalance mode - use load imbalance features
                - False: Standard mode - only use basic num_tokens feature
            tensor_parallel_size: Optional TP override for op-specific MoE training.
                If None, uses replica_config.moe_tensor_parallel_size.
            expert_parallel_size: Optional EP filter for op-specific MoE training.
                If None, EP filtering is skipped (used for EP-agnostic replicated ops).

        Returns:
            Filtered DataFrame ready for MoE model training

        Raises:
            FileNotFoundError: If the input file does not exist
            ValueError: If no data matches filtering criteria or required features are missing
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"MoE input file does not exist: {file_path}\n"
                f"Please run MoE profiling first.\n"
                f"Suggested command: bash frontier/profiling/example/test_profiling_moe.sh"
            )

        df = pd.read_csv(file_path)
        logger.info(f"Original MoE data: {len(df)} rows, {len(df.columns)} columns")

        model_config = replica_config.model_config
        training_mode = "load_imbalance (load_imbalance=True)" if load_imbalance else "standard (load_imbalance=False)"
        if tensor_parallel_size is None:
            tensor_parallel_size = replica_config.moe_tensor_parallel_size
        if tensor_parallel_size <= 0:
            raise ValueError(
                f"Invalid tensor_parallel_size for MoE data loading: {tensor_parallel_size}"
            )

        # Display filtering conditions
        logger.info(f"Filtering conditions:")
        logger.info(f"  - num_experts == {model_config.num_experts}")
        logger.info(f"  - router_topk == {model_config.num_experts_per_tok}")
        logger.info(f"  - hidden_dim == {model_config.embedding_dim}")
        logger.info(f"  - expert_hidden_dim == {model_config.mlp_hidden_dim}")
        logger.info(f"  - num_tensor_parallel_workers == {tensor_parallel_size}")
        if expert_parallel_size is None:
            logger.info("  - expert_parallel_size == ANY (EP-agnostic op)")
        else:
            logger.info(f"  - expert_parallel_size == {expert_parallel_size}")
        logger.info(f"  - training_mode: {training_mode}")

        # Display available values in the dataset
        available_info = []
        if len(df) > 0:
            if 'num_experts' in df.columns:
                available_info.append(f"  - Available num_experts: {sorted(df['num_experts'].unique())}")
            if 'router_topk' in df.columns:
                available_info.append(f"  - Available router_topk: {sorted(df['router_topk'].unique())}")
            if 'num_tensor_parallel_workers' in df.columns:
                available_info.append(f"  - Available num_tensor_parallel_workers: {sorted(df['num_tensor_parallel_workers'].unique())}")
            if 'expert_parallel_size' in df.columns:
                available_info.append(f"  - Available expert_parallel_size: {sorted(df['expert_parallel_size'].unique())}")
            if 'load_distribution' in df.columns:
                available_info.append(f"  - Available load_distribution: {sorted(df['load_distribution'].unique())}")

        for info in available_info:
            logger.info(info)

        # Apply filtering based on MoE configuration
        filtered_df = df[
            (df["num_experts"] == model_config.num_experts)
            & (df["router_topk"] == model_config.num_experts_per_tok)
            & (df["hidden_dim"] == model_config.embedding_dim)
            & (df["expert_hidden_dim"] == model_config.mlp_hidden_dim)
            & (df["num_tensor_parallel_workers"] == tensor_parallel_size)
        ]
        if expert_parallel_size is not None:
            filtered_df = filtered_df[
                filtered_df["expert_parallel_size"] == expert_parallel_size
            ]

        logger.info(f"After config filtering: {len(filtered_df)} rows")

        # Check for load imbalance features if load_imbalance mode is enabled
        if load_imbalance:
            missing_features = [
                f for f in self.MOE_LOAD_IMBALANCE_FEATURES
                if f not in filtered_df.columns
            ]
            if missing_features:
                logger.warning(
                    f"Load imbalance mode requested but missing features: {missing_features}\n"
                    f"Available columns: {list(filtered_df.columns)}\n"
                    f"Please run MoE profiling with --enable_load_imbalance flag.\n"
                    f"Use load_imbalance=False (standard mode) explicitly if you want to train without load imbalance features."
                )
                raise ValueError("Missing load imbalance features")
                # Note: We don't change load_imbalance here, caller should handle feature selection
            else:
                logger.info(f"Load imbalance features available: {self.MOE_LOAD_IMBALANCE_FEATURES}")

        if len(filtered_df) == 0:
            ep_requirement = "ANY" if expert_parallel_size is None else expert_parallel_size
            available_info_text = "\n".join(available_info)
            message = (
                f"No data matches the filtering criteria in {file_path}\n"
                f"Required MoE configuration:\n"
                f"  - num_experts: {model_config.num_experts}\n"
                f"  - router_topk: {model_config.num_experts_per_tok}\n"
                f"  - hidden_dim: {model_config.embedding_dim}\n"
                f"  - expert_hidden_dim: {model_config.mlp_hidden_dim}\n"
                f"  - tensor_parallel_size: {tensor_parallel_size}\n"
                f"  - expert_parallel_size: {ep_requirement}\n"
                f"  - training_mode: {training_mode}\n"
            )
            if available_info_text:
                message += available_info_text
            raise ValueError(
                message
            )

        return filtered_df

    def _get_attention_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to attention dataframe.

        Standard features for attn_prefill and attn_decode:
            - num_tokens: max(prefill_chunk_size, batch_size)
            - is_decode: derived from is_prefill when available, else prefill_chunk_size == 0
            - prefill_chunk_size_squared: prefill_chunk_size ** 2

        Mixed-batch features for attn_prefill_mixed (12 features):
            Reference: frontier/training/attention_trainer.py lines 362-375
            These features capture batch heterogeneity for accurate prefill time prediction.
        """
        df_with_derived_features = df.copy()

        # Standard attention features
        df_with_derived_features["num_tokens"] = df_with_derived_features[["prefill_chunk_size", "batch_size"]].max(axis=1)
        if "is_prefill" in df_with_derived_features.columns:
            normalized_prefill_values = (
                df_with_derived_features["is_prefill"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"1", "true", "t", "yes", "y"})
            )
            df_with_derived_features["is_decode"] = ~normalized_prefill_values
        else:
            df_with_derived_features["is_decode"] = (df_with_derived_features["prefill_chunk_size"] == 0)
        df_with_derived_features["prefill_chunk_size_squared"] = (df_with_derived_features["prefill_chunk_size"] ** 2)

        def _normalize_bool_series(series: pd.Series) -> pd.Series:
            return (
                series.astype(str)
                .str.strip()
                .str.lower()
                .isin({"1", "true", "t", "yes", "y"})
            )

        if "is_mixed_batch" in df_with_derived_features.columns:
            df_with_derived_features["is_mixed_batch"] = _normalize_bool_series(
                df_with_derived_features["is_mixed_batch"]
            )
        else:
            df_with_derived_features["is_mixed_batch"] = False

        if "is_true_mixed_batch" in df_with_derived_features.columns:
            df_with_derived_features["is_true_mixed_batch"] = _normalize_bool_series(
                df_with_derived_features["is_true_mixed_batch"]
            )
        else:
            df_with_derived_features["is_true_mixed_batch"] = False

        # Mixed-batch features for attn_prefill_mixed (if applicable)
        # Check if the profiling data contains mixed-batch specific columns
        has_mixed_batch_data = "total_tokens" in df_with_derived_features.columns

        if has_mixed_batch_data:
            logger.info("Adding mixed-batch derived features for attn_prefill_mixed")

            # total_tokens_squared for computational complexity
            if "total_tokens" in df_with_derived_features.columns:
                df_with_derived_features["total_tokens_squared"] = (
                    df_with_derived_features["total_tokens"] ** 2
                )

            # seq_len_range = max_seq_len - min_seq_len
            if "max_seq_len" in df_with_derived_features.columns and "min_seq_len" in df_with_derived_features.columns:
                df_with_derived_features["seq_len_range"] = (
                    df_with_derived_features["max_seq_len"] -
                    df_with_derived_features["min_seq_len"]
                )

            # Interaction features: batch_size * heterogeneity metrics
            if "seq_len_variance" in df_with_derived_features.columns:
                df_with_derived_features["batch_variance_interaction"] = (
                    df_with_derived_features["batch_size"] *
                    df_with_derived_features["seq_len_variance"]
                )

            if "seq_len_cv" in df_with_derived_features.columns:
                df_with_derived_features["batch_cv_interaction"] = (
                    df_with_derived_features["batch_size"] *
                    df_with_derived_features["seq_len_cv"]
                )

            if {
                "num_prefill_seqs",
                "num_decode_seqs",
            }.issubset(df_with_derived_features.columns) and (
                "total_batch_size" not in df_with_derived_features.columns
            ):
                df_with_derived_features["total_batch_size"] = (
                    df_with_derived_features["num_prefill_seqs"]
                    + df_with_derived_features["num_decode_seqs"]
                )

            if {
                "num_prefill_seqs",
                "total_batch_size",
            }.issubset(df_with_derived_features.columns) and (
                "batch_composition_ratio" not in df_with_derived_features.columns
            ):
                total_batch_size = df_with_derived_features["total_batch_size"].replace(0, pd.NA)
                df_with_derived_features["batch_composition_ratio"] = (
                    df_with_derived_features["num_prefill_seqs"] / total_batch_size
                ).fillna(0.0)

            if (
                "num_decode_seqs" in df_with_derived_features.columns
                and "decode_batch_size" not in df_with_derived_features.columns
            ):
                df_with_derived_features["decode_batch_size"] = df_with_derived_features[
                    "num_decode_seqs"
                ]

        return df_with_derived_features

    def _get_all_reduce_df_with_derived_features(self, df: pd.DataFrame, replica_config) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        df_with_derived_features["num_tokens"] = (
            df_with_derived_features["size"] / replica_config.model_config.embedding_dim / 2
        )
        return df_with_derived_features

    def _get_send_recv_df_with_derived_features(self, df: pd.DataFrame, replica_config) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        df_with_derived_features["num_tokens"] = (
            df_with_derived_features["size"] / replica_config.model_config.embedding_dim / 2
        )
        return df_with_derived_features

    def _get_moe_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features to MoE dataframe.

        The MoE profiling data already contains num_tokens as a direct column,
        so we just ensure it exists and return the dataframe.
        Additional derived features can be added here if needed in the future.
        """
        df_with_derived_features = df.copy()

        # Verify that num_tokens column exists (it should be in the profiling output)
        if "num_tokens" not in df_with_derived_features.columns:
            logger.warning("num_tokens column not found in MoE dataframe")
            logger.warning(f"Available columns: {list(df_with_derived_features.columns)}")
            # If num_tokens is missing, we cannot proceed with training
            raise ValueError("MoE profiling data must contain 'num_tokens' column")

        return df_with_derived_features

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
            'moe_shuffling_calibration_scale': config.moe_shuffling_calibration_scale,
            'decode_phase_moe_shuffling_calibration_scale': config.decode_phase_moe_shuffling_calibration_scale,
            'moe_grouped_gemm_calibration_scale': config.moe_grouped_gemm_calibration_scale,
            'decode_phase_moe_grouped_gemm_calibration_scale': config.decode_phase_moe_grouped_gemm_calibration_scale,
            'short_decode_request_length_threshold': getattr(config, 'short_decode_request_length_threshold', None),
            'short_decode_request_length_calibration_scale': getattr(config, 'short_decode_request_length_calibration_scale', None),
            'long_decode_request_length_threshold': getattr(config, 'long_decode_request_length_threshold', None),
            'long_decode_request_length_calibration_scale': getattr(config, 'long_decode_request_length_calibration_scale', None),
            'low_prefill_short_decode_request_prefill_threshold': getattr(config, 'low_prefill_short_decode_request_prefill_threshold', None),
            'low_prefill_short_decode_request_decode_threshold': getattr(config, 'low_prefill_short_decode_request_decode_threshold', None),
            'low_prefill_short_decode_request_calibration_scale': getattr(config, 'low_prefill_short_decode_request_calibration_scale', None),
            'low_prefill_decode_mix_request_prefill_threshold': getattr(config, 'low_prefill_decode_mix_request_prefill_threshold', None),
            'low_prefill_decode_mix_request_decode_min': getattr(config, 'low_prefill_decode_mix_request_decode_min', None),
            'low_prefill_decode_mix_request_decode_max': getattr(config, 'low_prefill_decode_mix_request_decode_max', None),
            'low_prefill_decode_mix_request_min_match_ratio': getattr(config, 'low_prefill_decode_mix_request_min_match_ratio', None),
            'low_prefill_decode_mix_request_max_match_ratio': getattr(config, 'low_prefill_decode_mix_request_max_match_ratio', None),
            'low_prefill_decode_mix_request_calibration_scale': getattr(config, 'low_prefill_decode_mix_request_calibration_scale', None),
            'low_prefill_decode_mix_request_include_mixed_batches': getattr(config, 'low_prefill_decode_mix_request_include_mixed_batches', False),
            'low_prefill_long_decode_request_prefill_threshold': getattr(config, 'low_prefill_long_decode_request_prefill_threshold', None),
            'low_prefill_long_decode_request_decode_threshold': getattr(config, 'low_prefill_long_decode_request_decode_threshold', None),
            'low_prefill_long_decode_request_calibration_scale': getattr(config, 'low_prefill_long_decode_request_calibration_scale', None),
            'low_prefill_long_decode_request_include_mixed_batches': getattr(config, 'low_prefill_long_decode_request_include_mixed_batches', False),
            'high_prefill_mid_decode_request_prefill_threshold': getattr(config, 'high_prefill_mid_decode_request_prefill_threshold', None),
            'high_prefill_mid_decode_request_decode_min': getattr(config, 'high_prefill_mid_decode_request_decode_min', None),
            'high_prefill_mid_decode_request_decode_max': getattr(config, 'high_prefill_mid_decode_request_decode_max', None),
            'high_prefill_mid_decode_request_calibration_scale': getattr(config, 'high_prefill_mid_decode_request_calibration_scale', None),
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

    def _get_model_hash(self, model_name: str, df: pd.DataFrame, execution_time_predictor_config, profiling_precision: str, measurement_type: MeasurementType) -> str:
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

        # Combine all components
        combined_str = f"{config_str}_{model_name}_{df_hash_str}_{profiling_precision}_{measurement_type.value}"
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

    def _store_model_precision(self, model_name: str, precision: str, model: BaseEstimator) -> None:
        precision_key = precision.upper()
        family_name = self._measurement_family_name(self._active_measurement_type)
        if family_name == "eager":
            self._trained_models_eager[model_name] = model
            self._models_by_precision_eager.setdefault(precision_key, {})[model_name] = model
            self._model_profiling_precision_eager[model_name] = precision_key
        elif family_name == "kernel_only":
            self._trained_models_kernel_only[model_name] = model
            self._models_by_precision_kernel_only.setdefault(precision_key, {})[model_name] = model
            self._model_profiling_precision_kernel_only[model_name] = precision_key
        else:
            raise ValueError(f"Unsupported family_name={family_name!r}")

        self._models_by_precision.setdefault(precision_key, {})[f"{family_name}:{model_name}"] = model
        self._model_profiling_precision[f"{family_name}:{model_name}"] = precision_key

    def get_model(self, model_name: str, precision: Optional[str] = None) -> Optional[BaseEstimator]:
        """Get a trained prediction model by name and precision."""
        if self._all_dummy_mode:
            return None

        if precision:
            precision_key = precision.upper()
            for registry in (self._models_by_precision_eager, self._models_by_precision_kernel_only):
                model = registry.get(precision_key, {}).get(model_name)
                if model is not None:
                    return model

            available_precisions = sorted(
                set(self._models_by_precision_eager.keys()) | set(self._models_by_precision_kernel_only.keys())
            )
            raise ValueError(
                f"Model '{model_name}' not available for precision '{precision_key}'. "
                f"Available precisions: {available_precisions}. "
                f"Ensure profiling data matches the requested precision."
            )

        return self._trained_models_eager.get(model_name) or self._trained_models_kernel_only.get(model_name)

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
            pickle.dump(model, open(cache_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"✓ Saved trained model '{model_name}' to cache (hash: {model_hash})")
            logger.info(f"  Cache file: {cache_file}")

    def get_models(self) -> Dict[str, Dict[str, BaseEstimator]]:
        """Return the trained models grouped by measurement family."""
        if self._all_dummy_mode:
            logger.debug("Returning empty models dict for dummy mode")
            return {"eager": {}, "kernel_only": {}}
        return {
            "eager": dict(self._trained_models_eager),
            "kernel_only": dict(self._trained_models_kernel_only),
        }

    def get_models_for_cluster(self, cluster_type: ClusterType) -> Dict[str, Dict[str, BaseEstimator]]:
        """Return a cluster-specific view of trained models grouped by measurement family."""
        if self._all_dummy_mode:
            return {"eager": {}, "kernel_only": {}}

        if cluster_type == ClusterType.PREFILL:
            return {"eager": dict(self._trained_models_eager), "kernel_only": {}}
        if cluster_type in [ClusterType.DECODE, ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN]:
            if not self._is_kernel_only_measurement_enabled_for_cluster(cluster_type):
                return {"eager": dict(self._trained_models_eager), "kernel_only": {}}
            return {"eager": {}, "kernel_only": dict(self._trained_models_kernel_only)}
        if cluster_type == ClusterType.MONOLITHIC:
            kernel_only_models = {}
            if self._is_kernel_only_measurement_enabled_for_cluster(cluster_type):
                kernel_only_models = dict(self._trained_models_kernel_only)
            return {
                "eager": dict(self._trained_models_eager),
                "kernel_only": kernel_only_models,
            }
        raise ValueError(f"Unsupported cluster_type={cluster_type!r}")

    def get_required_capabilities(self) -> Dict[str, Any]:
        """Return the analyzed requirements."""
        return self._required_capabilities

    def get_training_file_paths(self, cluster_type: ClusterType) -> Dict[str, str]:
        """Get the resolved profiling file paths for a specific cluster type."""
        if cluster_type not in self._cluster_configs:
            return {}

        cluster_config = self._cluster_configs[cluster_type]
        replica_config = cluster_config.replica_config
        execution_time_predictor_config = cluster_config.execution_time_predictor_config

        def _resolve(path_template: str) -> str:
            return (
                path_template
                .replace("{DEVICE}", replica_config.device)
                .replace("{MODEL}", replica_config.model_config.get_name())
                .replace("{NETWORK_DEVICE}", replica_config.network_device)
            )

        linear_op_file = execution_time_predictor_config.linear_op_input_file
        if not linear_op_file and execution_time_predictor_config.mlp_input_file:
            linear_op_file = execution_time_predictor_config.mlp_input_file

        return {
            'compute_input_file': _resolve(linear_op_file),
            'attention_input_file': _resolve(execution_time_predictor_config.atten_input_file),
            'moe_input_file': _resolve(execution_time_predictor_config.moe_input_file),
            'all_reduce_input_file': _resolve(execution_time_predictor_config.all_reduce_input_file),
            'send_recv_input_file': _resolve(execution_time_predictor_config.send_recv_input_file),
            'cpu_overhead_input_file': _resolve(execution_time_predictor_config.cpu_overhead_input_file),
            'cpu_overhead_kernel_only_input_file': _resolve(
                getattr(
                    execution_time_predictor_config,
                    'cpu_overhead_kernel_only_input_file',
                    execution_time_predictor_config.cpu_overhead_input_file,
                )
            ),
            'pp_stage_boundary_input_file': _resolve(execution_time_predictor_config.pp_stage_boundary_input_file),
            'pp_receiver_head_input_file': _resolve(execution_time_predictor_config.pp_receiver_head_input_file),
            'pp_producer_send_path_input_file': _resolve(execution_time_predictor_config.pp_producer_send_path_input_file),
            'pp_prefill_consumer_active_input_file': _resolve(
                execution_time_predictor_config.pp_prefill_consumer_active_input_file
            ),
            'compute_kernel_only_input_file': _resolve(execution_time_predictor_config.linear_op_kernel_only_input_file),
            'attention_kernel_only_input_file': _resolve(execution_time_predictor_config.atten_kernel_only_input_file),
            'moe_kernel_only_input_file': _resolve(execution_time_predictor_config.moe_kernel_only_input_file),
        }

    def get_training_context(self, cluster_type: ClusterType) -> Dict[str, Any]:
        """
        Get comprehensive training context for a specific cluster type.
        
        Args:
            cluster_type: The cluster type to get context for
        
        Returns:
            Dictionary containing training context information
        """
        if cluster_type not in self._cluster_configs:
            return {}
        
        cluster_config = self._cluster_configs[cluster_type]
        replica_config = cluster_config.replica_config
        replica_scheduler_config = cluster_config.replica_scheduler_config
        
        file_paths = self.get_training_file_paths(cluster_type)
        
        return {
            'cluster_type': cluster_type,
            'device': replica_config.device,
            'model_name': replica_config.model_name,
            'attn_tensor_parallel_size': replica_config.attn_tensor_parallel_size,
            'moe_tensor_parallel_size': replica_config.moe_tensor_parallel_size,
            'num_pipeline_stages': replica_config.num_pipeline_stages,
            'network_device': replica_config.network_device,
            'block_size': replica_scheduler_config.block_size,
            'file_paths': file_paths,
            'max_tokens': getattr(replica_config, 'max_tokens', None),
            'max_batch_size': getattr(replica_config, 'max_batch_size', None)
        }
