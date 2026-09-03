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
from frontier.attention.model_binding import bind_attention_family
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

logger = init_logger(__name__)
MIGRATION_HELP_COMMAND = (
    "python -m frontier.profiling.migrate_csv_metadata --help"
)


def _get_moe_family_model_names() -> List[str]:
    return list(get_family_profiling_names(MOE_FAMILY))


def _get_moe_family_operator_by_model_name(model_name: str):
    moe_ops = {
        operator.profiling_name(): operator
        for operator in MOE_FAMILY.profiling_ops()
    }
    if model_name not in moe_ops:
        raise ValueError(f"Unsupported MoE op: {model_name}")
    return moe_ops[model_name]


def _get_moe_gating_family_model_names() -> List[str]:
    return [
        operator.profiling_name()
        for operator in MOE_FAMILY.profiling_ops()
        if operator.precision_name() == "moe_gating"
    ]


def _get_prefill_hot_moe_gating_model_names() -> List[str]:
    return [
        f"{model_name}__prefill_hot"
        for model_name in _get_moe_gating_family_model_names()
    ]


def _resolve_model_architecture_profile(
    model_config: Any,
    *,
    allow_generic: bool = False,
) -> Optional[ModelArchitectureProfile]:
    if model_config is None:
        return None
    getter = getattr(model_config, "get_model_architecture_profile", None)
    if callable(getter):
        return cast(Optional[ModelArchitectureProfile], getter())

    # Lightweight test/config adapters that predate the typed contract do not
    # expose a profile accessor or typed widths.  Keep those callers on the
    # scalar compatibility path instead of treating the generic fallback as a
    # complete typed declaration.  An explicit profile or typed width opts the
    # adapter into strict profile-owned resolution.
    typed_fields = (
        "model_architecture_profile",
        "dense_mlp_hidden_dim",
        "routed_mlp_hidden_dim",
        "share_expert_dim",
    )
    if not allow_generic and not any(
        getattr(model_config, field_name, None) is not None
        for field_name in typed_fields
    ):
        return None
    return get_model_architecture_profile(model_config)


def _resolve_model_architecture_profile_id(model_config) -> str:
    architecture_profile = _resolve_model_architecture_profile(model_config)
    if architecture_profile is None:
        return "generic"
    return architecture_profile.profile_id


def _resolve_profile_typed_family_for_query(
    architecture_profile: ModelArchitectureProfile,
    op_name: str,
) -> Optional[Tuple[str, LayerKind]]:
    """Resolve a query to the profile-owned typed operator family."""

    if not isinstance(op_name, str) or not op_name:
        raise ValueError("typed operator query name must be a non-empty string")
    matches: list[Tuple[str, LayerKind]] = []
    for layer_contract in architecture_profile.layer_contracts:
        for family_id in layer_contract.operator_family_ids:
            family = get_operator_family(family_id)
            if any(
                op_name == operator.name
                or op_name == operator.profiling_name()
                for operator in family.operators
            ):
                matches.append((family_id, layer_contract.layer_kind))
    if len(matches) > 1:
        raise ValueError(
            f"Operator query {op_name!r} belongs to multiple typed layer families: "
            f"{sorted(family_id for family_id, _ in matches)}"
        )
    return matches[0] if matches else None


def _serialize_selected_layer_cache_identity(
    layer_contract: Optional[ResolvedLayerContract],
) -> Optional[str]:
    """Serialize the selected semantic domain used by a model cache.

    Physical ``layer_id`` and producer-side domain envelopes do not identify a
    trained estimator.  Keep only the selected fields that affect estimator
    admission and reuse, in deterministic JSON form.
    """

    if layer_contract is None:
        return None
    if not isinstance(layer_contract, ResolvedLayerContract):
        raise TypeError(
            "layer_contract must be a ResolvedLayerContract when provided"
        )
    metadata = layer_contract.typed_metadata_identity()
    selected_fields = (
        "profile_id",
        "operator_family_id",
        "layer_kind",
        "dimension_source",
        "effective_ffn_width",
        "tensor_parallel_mode",
        "expert_parallel_mode",
        "selected_expert_parallel_size",
        "selected_tensor_parallel_size",
        "selected_padded_ffn_width",
    )
    return json.dumps(
        {field_name: metadata[field_name] for field_name in selected_fields},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _get_contract_hash(
    layer_contract: Optional[ResolvedLayerContract],
) -> str:
    """Return a short deterministic hash for an optional selected contract."""

    identity = _serialize_selected_layer_cache_identity(layer_contract)
    if identity is None:
        return "none"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _validate_typed_parallel_selection(
    layer_contract: ResolvedLayerContract,
    *,
    tensor_parallel_size: Optional[int] = None,
    expert_parallel_size: Optional[int] = None,
) -> None:
    """Validate explicit loader selectors against a resolved contract."""

    if not isinstance(layer_contract, ResolvedLayerContract):
        raise TypeError("layer_contract must be a ResolvedLayerContract")
    contract_tp = layer_contract.tensor_parallel_size
    if (
        contract_tp is not None
        and tensor_parallel_size is not None
        and contract_tp != tensor_parallel_size
    ):
        raise ValueError(
            f"typed layer contract TP {contract_tp} conflicts with "
            f"tensor_parallel_size {tensor_parallel_size}"
        )
    contract_ep = layer_contract.expert_parallel_size
    if (
        contract_ep is not None
        and expert_parallel_size is not None
        and contract_ep != expert_parallel_size
    ):
        raise ValueError(
            f"typed layer contract EP {contract_ep} conflicts with "
            f"expert_parallel_size {expert_parallel_size}"
        )


def _typed_row_matches_contract(
    raw_contracts: Any,
    layer_contract: ResolvedLayerContract,
    *,
    operator_name: Optional[str],
) -> bool:
    """Match one parsed or serialized row to its exact typed operator contract."""

    if not isinstance(operator_name, str) or not operator_name:
        raise ValueError(
            "typed profiling loading requires a non-empty operator_name when "
            f"the canonical {TYPED_OPERATOR_CONTRACTS_COLUMN!r} column is present"
        )
    if not isinstance(layer_contract.operator_family_id, str) or not layer_contract.operator_family_id:
        raise ValueError(
            "typed profiling loading requires a layer contract with an operator family id"
        )
    return matches_resolved_layer_contract(
        raw_contracts,
        layer_contract,
        operator_name=operator_name,
    )


def _normalize_layer_contract_context(
    training_context: Optional[Mapping[str, Any]],
    explicit_layer_contract: Optional[ResolvedLayerContract] = None,
) -> Tuple[Optional[ResolvedLayerContract], Dict[str, Any]]:
    """Resolve one contract and keep every context representation consistent."""

    context = dict(training_context or {})
    context_contract = context.get("layer_contract")
    if context_contract is not None and not isinstance(
        context_contract, ResolvedLayerContract
    ):
        raise TypeError(
            "training_context['layer_contract'] must be a ResolvedLayerContract"
        )
    if explicit_layer_contract is not None and not isinstance(
        explicit_layer_contract, ResolvedLayerContract
    ):
        raise TypeError("layer_contract must be a ResolvedLayerContract")

    if context_contract is not None and explicit_layer_contract is not None:
        if not context_contract.is_semantically_equivalent(explicit_layer_contract):
            raise ValueError(
                "conflicting layer_contract values were provided through the "
                "explicit argument and training_context"
            )

    resolved_contract = explicit_layer_contract or context_contract
    context_identity = context.get("layer_contract_identity")
    if context_identity is not None and not isinstance(context_identity, str):
        raise TypeError(
            "training_context['layer_contract_identity'] must be a string"
        )
    selected_identity = _serialize_selected_layer_cache_identity(resolved_contract)
    if context_identity is not None and context_identity != selected_identity:
        raise ValueError(
            "training_context['layer_contract_identity'] does not match the "
            "supplied layer_contract"
        )
    if resolved_contract is None:
        return None, context

    context["layer_contract"] = resolved_contract
    context["layer_contract_identity"] = selected_identity
    context["layer_kind"] = resolved_contract.layer_kind.value
    context["effective_ffn_width"] = resolved_contract.effective_ffn_width
    context["tensor_parallel_mode"] = resolved_contract.tensor_parallel_mode.value
    context["expert_parallel_mode"] = resolved_contract.expert_parallel_mode.value
    return resolved_contract, context


def _add_layer_contract_to_training_context(
    training_context: Mapping[str, Any],
    layer_contract: Optional[ResolvedLayerContract],
) -> Dict[str, Any]:
    """Copy a training context and attach a resolved typed contract."""

    _, context = _normalize_layer_contract_context(
        training_context,
        explicit_layer_contract=layer_contract,
    )
    return context


def _layer_contract_kwargs(
    layer_contract: Optional[ResolvedLayerContract],
    *,
    operator_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Return typed keyword arguments only for an opted-in contract path."""

    if layer_contract is None:
        return {}
    kwargs: Dict[str, Any] = {"layer_contract": layer_contract}
    if operator_name is not None:
        kwargs["operator_name"] = operator_name
    return kwargs


def _is_moe_gating_family_model_name(model_name: str) -> bool:
    base_model_name = get_moe_gating_base_model_name(model_name)
    return _get_moe_family_operator_by_model_name(
        base_model_name
    ).precision_name() == "moe_gating"


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
        # Typed FFN models are keyed by their selected semantic domain.  Keep
        # the historical bare registries for untyped/legacy models only.
        self._trained_models_eager_by_contract = {}
        self._trained_models_kernel_only_by_contract = {}
        self._models_by_precision_eager_by_contract = {}
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
        if measurement_type == MeasurementType.KERNEL_ONLY:
            return "kernel_only"
        raise ValueError(f"Unsupported measurement_type={measurement_type!r}")

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
        self, cluster_type: ClusterType
    ) -> List[MeasurementType]:
        if global_vars.get_sys_arch() == "pd-af-disaggregation":
            if cluster_type == ClusterType.PREFILL:
                return [MeasurementType.CUDA_EVENT]
            if cluster_type == ClusterType.DECODE_ATTN:
                return [MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY]
            if cluster_type in (ClusterType.DECODE, ClusterType.DECODE_FFN):
                return [MeasurementType.KERNEL_ONLY]
            if cluster_type == ClusterType.MONOLITHIC:
                return [MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY]
            raise ValueError(f"Unsupported cluster_type={cluster_type!r}")

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
        if cluster_type == ClusterType.DECODE_FFN:
            # In the FFN-only PD-AF cluster, dense FFN tensor parallelism is
            # carried by moe_tensor_parallel_size as the FFN-domain TP field.
            # attn_tensor_parallel_size can remain at its default because this
            # cluster owns no attention weights.  Use the FFN-domain TP for
            # both dense and MoE DECODE_FFN profiling selection.
            return replica_config.moe_tensor_parallel_size
        if (
            is_moe_model
            and cluster_type in {
                ClusterType.PREFILL,
                ClusterType.DECODE,
                ClusterType.MONOLITHIC,
            }
        ):
            return replica_config.moe_tensor_parallel_size
        return replica_config.attn_tensor_parallel_size

    def _resolve_typed_layer_contract(
        self,
        op_name: str,
        cluster_type: ClusterType,
        replica_config,
        *,
        is_moe_model: bool,
        layer_id: Optional[int] = None,
    ) -> Optional[ResolvedLayerContract]:
        """Resolve a typed FFN contract through the architecture profile."""

        model_config = getattr(replica_config, "model_config", None)
        architecture_profile = _resolve_model_architecture_profile(model_config)
        if architecture_profile is None:
            return None

        typed_family = _resolve_profile_typed_family_for_query(
            architecture_profile, op_name
        )
        if typed_family is None:
            return None
        typed_family_id, _ = typed_family

        # DECODE_ATTN is attention-only. Its exact zero domain is a deliberate
        # sentinel; any non-zero value indicates a malformed configuration.
        if cluster_type == ClusterType.DECODE_ATTN:
            zero_fields = (
                "attn_tensor_parallel_size",
                "moe_tensor_parallel_size",
                "moe_expert_parallel_size",
            )
            invalid = {
                field_name: getattr(replica_config, field_name, None)
                for field_name in zero_fields
                if getattr(replica_config, field_name, None) != 0
            }
            if invalid:
                raise ValueError(
                    "DECODE_ATTN typed FFN resolution requires exact zero "
                    f"parallel sizes, got {invalid!r}"
                )
            return None

        from frontier.operators.binding import bind_operator_query

        binding = bind_operator_query(op_name, family_id=typed_family_id)
        if binding.family_id != typed_family_id:
            raise ValueError(
                f"Operator query {op_name!r} resolved to family "
                f"{binding.family_id!r}, expected {typed_family_id!r}"
            )

        moe_tp_size = getattr(replica_config, "moe_tensor_parallel_size", None)
        attention_tp_size = getattr(
            replica_config, "attn_tensor_parallel_size", None
        )
        if cluster_type == ClusterType.DECODE_FFN:
            # The FFN-only role stores its domain size in the existing MoE TP
            # field, while the profile still owns the semantic TP mode.
            attention_tp_size = moe_tp_size
        ffn_tp_size = self._get_ffn_tp_key(
            cluster_type, replica_config, is_moe_model
        )
        return architecture_profile.resolve_layer_contract(
            model_config,
            layer_id=layer_id,
            operator_name=op_name,
            attention_tp_size=attention_tp_size,
            moe_tp_size=moe_tp_size,
            ffn_tp_size=ffn_tp_size,
            expert_parallel_size=getattr(
                replica_config, "moe_expert_parallel_size", None
            ),
        )

    def _resolve_ffn_layer_contracts(
        self,
        cluster_type: ClusterType,
        replica_config,
        is_moe_model: bool,
    ) -> Tuple[Tuple[str, ResolvedLayerContract], ...]:
        """Resolve each profile-owned FFN domain used by one training pass."""

        if cluster_type == ClusterType.DECODE_ATTN:
            zero_fields = (
                "attn_tensor_parallel_size",
                "moe_tensor_parallel_size",
                "moe_expert_parallel_size",
            )
            invalid = {
                field_name: getattr(replica_config, field_name, None)
                for field_name in zero_fields
                if getattr(replica_config, field_name, None) != 0
            }
            if invalid:
                raise ValueError(
                    "DECODE_ATTN FFN contract resolution requires exact zero "
                    f"parallel sizes, got {invalid!r}"
                )
            return ()

        model_config = getattr(replica_config, "model_config", None)
        if model_config is None:
            return ()
        architecture_profile = _resolve_model_architecture_profile(model_config)
        if architecture_profile is None:
            return ()
        if bool(is_moe_model) != bool(getattr(model_config, "is_moe", False)):
            raise ValueError(
                "is_moe_model does not match model configuration while resolving "
                "typed FFN contracts"
            )

        contracts: list[Tuple[str, ResolvedLayerContract]] = []
        for spec in architecture_profile.iter_active_layer_contracts(model_config):
            family_is_moe = spec.layer_kind is not LayerKind.DENSE
            for family_id in spec.operator_family_ids:
                family = get_operator_family(family_id)
                profiling_ops = tuple(family.profiling_ops())
                if not profiling_ops:
                    raise ValueError(
                        f"Typed operator family {family_id!r} has no profiling operators"
                    )
                contract = self._resolve_typed_layer_contract(
                    profiling_ops[0].name,
                    cluster_type,
                    replica_config,
                    is_moe_model=family_is_moe,
                )
                if contract is None:
                    raise ValueError(
                        f"Missing typed layer contract for operator family {family_id!r}"
                    )
                if contract.operator_family_id != family_id:
                    raise ValueError(
                        f"Operator family {family_id!r} resolved to "
                        f"{contract.operator_family_id!r}"
                    )
                contracts.append((family_id, contract))
        return tuple(contracts)

    def _get_ffn_contract_signature(
        self,
        cluster_type: ClusterType,
        replica_config,
        is_moe_model: bool,
    ) -> str:
        """Return a deterministic signature for the active FFN domains."""

        entries = self._resolve_ffn_layer_contracts(
            cluster_type, replica_config, is_moe_model
        )
        if not entries:
            return "none"
        payload = []
        for family_id, contract in entries:
            family = get_operator_family(family_id)
            profiling_ops = tuple(family.profiling_ops())
            if not profiling_ops:
                raise ValueError(
                    f"Typed operator family {family_id!r} has no profiling operators"
                )

            # The first operator is the compatibility representative returned
            # by _resolve_ffn_layer_contracts().  Include every sibling as
            # well: a family may mix EP-agnostic routing operators with an
            # EP-sensitive grouped GEMM, and the cache signature must retain
            # both semantics.
            for operator in profiling_ops:
                operator_contract = contract
                if operator is not profiling_ops[0]:
                    operator_contract = self._resolve_typed_layer_contract(
                        operator.profiling_name(),
                        cluster_type,
                        replica_config,
                        is_moe_model=contract.layer_kind is not LayerKind.DENSE,
                    )
                    if operator_contract is None:
                        raise ValueError(
                            "Missing typed layer contract for operator "
                            f"{operator.profiling_name()!r} in family {family_id!r}"
                        )
                payload.append(
                    {
                        "family_id": family_id,
                        "operator_name": operator.profiling_name(),
                        "identity": _serialize_selected_layer_cache_identity(
                            operator_contract
                        ),
                    }
                )
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _is_mixed_layer_moe_model(model_config, is_moe_model: bool) -> bool:
        """Return whether a model needs both MoE and dense FFN predictors.

        Some MoE architectures keep dense FFN layers at the model boundaries.
        Their runtime dispatch is layer-specific, so model-level ``is_moe`` is
        insufficient to decide which predictor families must be materialized.
        Keep the legacy pure-MoE path unchanged when the layer-count contract
        is unavailable.
        """
        if not is_moe_model or model_config is None:
            return False
        get_num_moe_layers = getattr(model_config, "get_num_moe_layers", None)
        num_layers = getattr(model_config, "num_layers", None)
        if callable(get_num_moe_layers) and isinstance(num_layers, int):
            return int(get_num_moe_layers()) < int(num_layers)
        return False

    def _get_linear_op_tp_key(self, op_name: str, cluster_type: ClusterType, replica_config, is_moe_model: bool) -> int:
        model_config = getattr(replica_config, "model_config", None)
        # Lightweight configs retain the scalar FFN compatibility path, but
        # generic linear attention names still require a profile declaration
        # for TP-mode lookup.
        architecture_profile = _resolve_model_architecture_profile(
            model_config,
            allow_generic=True,
        )
        if op_name in get_target_embedded_mtp_linear_ops():
            return resolve_effective_attention_tp_size(
                op_name="attn_pre_proj",
                requested_tp_size=replica_config.attn_tensor_parallel_size,
                num_kv_heads=replica_config.model_config.num_kv_heads,
                cluster_type=cluster_type,
                warning_cache=getattr(self, "_attention_tp_warning_cache", None),
                include_linear_ops=True,
            )

        try:
            tp_mode = resolve_operator_query_tp_mode(
                op_name,
                architecture_profile=architecture_profile,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported linear op for TP mapping: {op_name}") from exc

        if tp_mode is TensorParallelMode.REPLICATED:
            if (
                is_target_embedded_mtp_enabled(
                    getattr(replica_config, "speculative_decoding_config", None)
                )
                and is_target_embedded_mtp_same_tp_linear_op(op_name)
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

        if tp_mode is TensorParallelMode.FFN_TP:
            return self._get_ffn_tp_key(cluster_type, replica_config, is_moe_model)

        if tp_mode is TensorParallelMode.ATTENTION_TP:
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
    def _get_moe_op_tp_key(
        op_name: str,
        replica_config,
        cluster_type: ClusterType | None = None,
    ) -> int:
        try:
            return resolve_moe_operator_tp_key(
                op_name,
                moe_tp_size=replica_config.moe_tensor_parallel_size,
                cluster_type=cluster_type,
                family=MOE_FAMILY,
            )
        except ValueError as exc:
            if str(exc).startswith("Unsupported MoE op:"):
                raise ValueError(
                    f"Unsupported MoE op for TP mapping: {op_name}"
                ) from exc
            raise

    @staticmethod
    def _is_moe_op_ep_agnostic(op_name: str) -> bool:
        try:
            return is_moe_operator_ep_agnostic(op_name, family=MOE_FAMILY)
        except ValueError as exc:
            if str(exc).startswith("Unsupported MoE op:"):
                raise ValueError(
                    f"Unsupported MoE op for EP mapping: {op_name}"
                ) from exc
            raise

    def _validate_moe_dataset_contract(
        self,
        file_path: str,
        replica_config,
        model_names: List[str],
        cluster_type: ClusterType,
        layer_contract: Optional[ResolvedLayerContract] = None,
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
        if layer_contract is None:
            # A legacy caller has no profile-owned contract to validate.  Keep
            # the historical scalar admission rule, while refusing to guess
            # when the dataset advertises typed metadata.
            if TYPED_OPERATOR_CONTRACTS_COLUMN in df.columns:
                raise ValueError(
                    "typed MoE profiling data requires an explicit routed layer contract"
                )
            expected_expert_width = getattr(model_config, "mlp_hidden_dim", None)
            if type(expected_expert_width) is not int or expected_expert_width <= 0:
                raise ValueError(
                    "legacy MoE dataset validation requires a positive model_config.mlp_hidden_dim"
                )
        else:
            if layer_contract.layer_kind is not LayerKind.ROUTED:
                raise ValueError(
                    "MoE dataset contract validation requires a routed layer contract"
                )
            expected_expert_width = layer_contract.effective_ffn_width
        base_df = df[
            (df["num_experts"] == model_config.num_experts)
            & (df["router_topk"] == model_config.num_experts_per_tok)
            & (df["hidden_dim"] == model_config.embedding_dim)
            & (df["expert_hidden_dim"] == expected_expert_width)
        ]

        if len(base_df) == 0:
            raise ValueError(
                "MoE dataset contract validation failed: no rows match model configuration in "
                f"{file_path}. Required: num_experts={model_config.num_experts}, "
                f"router_topk={model_config.num_experts_per_tok}, hidden_dim={model_config.embedding_dim}, "
                f"expert_hidden_dim={expected_expert_width}."
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
            getattr(replica_config, "moe_routing_distribution_type", "balanced")
        )

        missing_requirements: List[str] = []
        for model_name in model_names:
            tp_key = self._get_moe_op_tp_key(
                model_name,
                replica_config,
                cluster_type,
            )
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

        # Create a signature for this FFN model configuration.
        model_config = replica_config.model_config
        model_arch = model_config.get_model_arch() if model_config is not None else "generic"
        architecture_profile_id = _resolve_model_architecture_profile_id(model_config)
        primary_contract = self._resolve_typed_layer_contract(
            "moe_grouped_gemm" if is_moe_model else "mlp_up_proj",
            cluster_type,
            replica_config,
            is_moe_model=is_moe_model,
        )
        typed_contract_hash = self._get_ffn_contract_signature(
            cluster_type,
            replica_config,
            is_moe_model,
        )
        active_measurement_type = getattr(
            self, "_active_measurement_type", MeasurementType.CUDA_EVENT
        )
        ffn_signature = (
            f"ffn_{replica_config.device}_{replica_config.model_name}_{tp_size}"
            f"_moe{is_moe_model}_arch_profile{architecture_profile_id}"
            f"_layer_contracts{typed_contract_hash}"
            f"_family{self._measurement_family_name(active_measurement_type)}"
        )

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
            'model_architecture_profile': architecture_profile_id,
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
            base_moe_model_names = _get_moe_family_model_names()
            moe_model_names = list(base_moe_model_names)
            if should_enable_prefill_hot_moe_gating_contract(
                model_config=model_config,
                model_arch=model_arch,
                model_name=replica_config.model_name,
            ):
                prefill_hot_probe_df = pd.read_csv(moe_input_file)
                include_prefill_hot_models = has_prefill_hot_moe_gating_rows(
                    prefill_hot_probe_df
                )

                if include_prefill_hot_models:
                    moe_model_names.extend(_get_prefill_hot_moe_gating_model_names())
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
                base_moe_model_names,
                cluster_type,
                **_layer_contract_kwargs(primary_contract),
            )
            requested_routing_runtime_path = resolve_moe_gating_routing_runtime_path(
                getattr(replica_config, "moe_routing_distribution_type", "balanced")
            )

            moe_df_cache: Dict[
                Tuple[
                    int,
                    Optional[int],
                    Optional[str],
                    Optional[str],
                    Optional[str],
                ],
                pd.DataFrame,
            ] = {}

            def _get_moe_df_for_op(
                model_name: str,
            ) -> Tuple[
                pd.DataFrame,
                int,
                Optional[int],
                Optional[ResolvedLayerContract],
            ]:
                base_model_name = get_moe_gating_base_model_name(model_name)
                op_layer_contract = self._resolve_typed_layer_contract(
                    base_model_name,
                    cluster_type,
                    replica_config,
                    is_moe_model=True,
                )
                tp_key = self._get_moe_op_tp_key(
                    base_model_name,
                    replica_config,
                    cluster_type,
                )
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
                if _is_moe_gating_family_model_name(base_model_name):
                    gating_context_key = DEFAULT_MOE_GATING_RUNTIME_CONTEXT
                    if model_name.endswith("__prefill_hot"):
                        gating_context_key = PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT

                contract_identity = _serialize_selected_layer_cache_identity(
                    op_layer_contract
                )
                cache_key = (
                    tp_key,
                    ep_key,
                    runtime_path_key,
                    gating_context_key,
                    contract_identity,
                )
                if cache_key not in moe_df_cache:
                    op_df = self._load_moe_df(
                        moe_input_file,
                        replica_config,
                        load_imbalance=False,
                        tensor_parallel_size=tp_key,
                        expert_parallel_size=ep_key,
                        **_layer_contract_kwargs(
                            op_layer_contract,
                            operator_name=base_model_name,
                        ),
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
                return moe_df_cache[cache_key], tp_key, ep_key, op_layer_contract

            for model_name in moe_model_names:
                model_signature = f"{model_name}_{ffn_signature}"
                if model_signature not in trained_model_signatures:
                    try:
                        (
                            op_moe_df,
                            moe_tp_key,
                            moe_ep_key,
                            op_layer_contract,
                        ) = _get_moe_df_for_op(model_name)
                    except PrefillHotRowsUnavailableError as exc:
                        logger.warning(
                            "Skipping %s because prefill-hot gating rows are unavailable "
                            "for the requested TP/EP slice (%s).",
                            model_name,
                            exc,
                        )
                        continue
                    op_training_context = _add_layer_contract_to_training_context(
                        training_context,
                        op_layer_contract,
                    )
                    op_training_context['tensor_parallel_size'] = moe_tp_key
                    op_training_context['expert_parallel_size'] = (
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
                    op_training_context['feature_cols'] = op_feature_cols

                    target_op_name = get_moe_gating_base_model_name(model_name)
                    train_kwargs: Dict[str, Any] = dict(
                        model_name=model_name,
                        df=op_moe_df,
                        feature_cols=op_feature_cols,
                        target_col=f"time_stats.{target_op_name}.median",
                        execution_time_predictor_config=execution_time_predictor_config,
                        training_context=op_training_context,
                    )
                    train_kwargs.update(_layer_contract_kwargs(op_layer_contract))
                    models[model_name] = self._train_single_model(
                        **train_kwargs,
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

                step2mini_share_expert_model_names = list(
                    get_family_profiling_names(SHARE_EXPERT_FAMILY)
                )
                if not step2mini_share_expert_model_names:
                    raise ValueError("Shared-expert operator family has no profiling names")
                share_expert_tp_key = self._get_linear_op_tp_key(
                    step2mini_share_expert_model_names[0],
                    cluster_type,
                    replica_config,
                    is_moe_model,
                )
                shared_layer_contract = self._resolve_typed_layer_contract(
                    step2mini_share_expert_model_names[0],
                    cluster_type,
                    replica_config,
                    is_moe_model=True,
                )
                if (
                    shared_layer_contract is None
                    and _resolve_model_architecture_profile(model_config) is not None
                ):
                    raise ValueError(
                        "Missing shared layer contract for share-expert training"
                    )
                share_expert_linear_ops_df = self._load_linear_op_df(
                    linear_ops_file,
                    share_expert_tp_key,
                    **_layer_contract_kwargs(
                        shared_layer_contract,
                        operator_name=step2mini_share_expert_model_names[0],
                    ),
                )
                logger.info(f"Loaded {len(share_expert_linear_ops_df)} rows for share_expert training")

                for model_name in step2mini_share_expert_model_names:
                    model_signature = f"{model_name}_{ffn_signature}"
                    if model_signature not in trained_model_signatures:
                        # Update training context to reflect linear_op.csv source.
                        shared_training_context = _add_layer_contract_to_training_context(
                            training_context,
                            shared_layer_contract,
                        )
                        shared_training_context['input_file'] = linear_ops_file
                        shared_training_context['tensor_parallel_size'] = share_expert_tp_key
                        target_col = f"time_stats.{model_name}.median"
                        if target_col not in share_expert_linear_ops_df.columns:
                            raise ValueError(
                                f"share_expert operation '{model_name}' column '{target_col}' not found in profiling data. "
                                f"Ensure profiling was run with a model architecture that includes share_expert. "
                                f"Available columns: {list(share_expert_linear_ops_df.columns)}"
                            )
                        train_kwargs: Dict[str, Any] = dict(
                            model_name=model_name,
                            df=share_expert_linear_ops_df,
                            feature_cols=["num_tokens"],
                            target_col=target_col,
                            execution_time_predictor_config=execution_time_predictor_config,
                            training_context=shared_training_context,
                        )
                        train_kwargs.update(
                            _layer_contract_kwargs(shared_layer_contract)
                        )
                        models[model_name] = self._train_single_model(
                            **train_kwargs,
                        )
                        trained_model_signatures.add(model_signature)
                        logger.info(f"Trained {model_name} for {cluster_type}")

            # Mixed-layer MoE models (for example step-moe-noquant) also have
            # dense boundary layers.  Train these additions after the legacy
            # MoE/share-expert families so RandomForest training order remains
            # compatible with the historical predictor artifact contract.
            if self._is_mixed_layer_moe_model(model_config, is_moe_model):
                dense_ffn_tp_key = self._get_ffn_tp_key(
                    cluster_type, replica_config, is_moe_model=False
                )
                dense_layer_contract = self._resolve_typed_layer_contract(
                    "mlp_up_proj",
                    cluster_type,
                    replica_config,
                    is_moe_model=False,
                )
                dense_ffn_signature = (
                    f"ffn_{replica_config.device}_{replica_config.model_name}_{dense_ffn_tp_key}"
                    f"_moeFalse_arch_profile{architecture_profile_id}"
                    f"_layer_contracts{_get_contract_hash(dense_layer_contract)}"
                    f"_family{self._measurement_family_name(active_measurement_type)}"
                )
                dense_training_context = dict(training_context)
                dense_training_context["is_moe_model"] = False
                dense_training_context["tensor_parallel_size"] = dense_ffn_tp_key
                dense_training_context = _add_layer_contract_to_training_context(
                    dense_training_context,
                    dense_layer_contract,
                )
                self._train_dense_mlp_models_for_cluster(
                    cluster_type=cluster_type,
                    replica_config=replica_config,
                    execution_time_predictor_config=execution_time_predictor_config,
                    linear_ops_file=linear_ops_file,
                    ffn_signature=dense_ffn_signature,
                    ffn_tp_key=dense_ffn_tp_key,
                    training_context=dense_training_context,
                    trained_model_signatures=trained_model_signatures,
                    models=models,
                    layer_contract=dense_layer_contract,
                )
        else:
            self._train_dense_mlp_models_for_cluster(
                cluster_type=cluster_type,
                replica_config=replica_config,
                execution_time_predictor_config=execution_time_predictor_config,
                linear_ops_file=linear_ops_file,
                ffn_signature=ffn_signature,
                ffn_tp_key=ffn_tp_key,
                training_context=training_context,
                trained_model_signatures=trained_model_signatures,
                models=models,
                layer_contract=primary_contract,
            )

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

    def _train_dense_mlp_models_for_cluster(
        self,
        *,
        cluster_type: ClusterType,
        replica_config,
        execution_time_predictor_config,
        linear_ops_file: str,
        ffn_signature: str,
        ffn_tp_key: int,
        training_context: Dict[str, Any],
        trained_model_signatures: set,
        models: Dict[str, BaseEstimator],
        layer_contract: Optional[ResolvedLayerContract] = None,
    ) -> None:
        """Materialize dense MLP predictors from the linear-op profile."""
        if not os.path.exists(linear_ops_file):
            raise FileNotFoundError(f"Linear ops input file {linear_ops_file} not found")
        if (
            layer_contract is None
            and _resolve_model_architecture_profile(
                getattr(replica_config, "model_config", None)
            )
            is not None
        ):
            layer_contract = self._resolve_typed_layer_contract(
                "mlp_up_proj",
                cluster_type,
                replica_config,
                is_moe_model=False,
            )
        if (
            layer_contract is not None
            and layer_contract.tensor_parallel_size is not None
            and layer_contract.tensor_parallel_size != ffn_tp_key
        ):
            raise ValueError(
                "Dense FFN training TP conflicts with its typed layer contract: "
                f"ffn_tp_key={ffn_tp_key}, "
                f"contract_tp={layer_contract.tensor_parallel_size}"
            )
        logger.info(f"Loading MLP data for {cluster_type} from: {linear_ops_file}")
        dense_model_names = tuple(get_family_profiling_names(FFN_FAMILY))
        if not dense_model_names:
            raise ValueError("FFN operator family has no profiling names")
        linear_ops_df = self._load_linear_op_df(
            linear_ops_file,
            ffn_tp_key,
            **_layer_contract_kwargs(
                layer_contract,
                operator_name=dense_model_names[0],
            ),
        )
        logger.info(f"Loaded {len(linear_ops_df)} rows for MLP training")
        dense_training_context = _add_layer_contract_to_training_context(
            training_context,
            layer_contract,
        )
        dense_training_context["input_file"] = linear_ops_file
        dense_training_context["tensor_parallel_size"] = ffn_tp_key

        missing_standard_columns = [
            f"time_stats.{model_name}.median"
            for model_name in dense_model_names
            if f"time_stats.{model_name}.median" not in linear_ops_df.columns
        ]
        if missing_standard_columns:
            model_config = getattr(replica_config, "model_config", None)
            supports_share_expert = bool(
                model_config is not None
                and model_config.supports_share_expert()
            )
            if supports_share_expert:
                logger.info(
                    "Skipping standard dense MLP training for %s: profile provides "
                    "shared-expert operations instead; missing columns=%s",
                    cluster_type,
                    missing_standard_columns,
                )
                return
            raise ValueError(
                "Dense MLP profiling data is incomplete; missing columns: "
                + ", ".join(missing_standard_columns)
            )

        for model_name in dense_model_names:
            model_signature = f"{model_name}_{ffn_signature}"
            if model_signature in trained_model_signatures:
                continue
            train_kwargs: Dict[str, Any] = dict(
                model_name=model_name,
                df=linear_ops_df,
                feature_cols=["num_tokens"],
                target_col=f"time_stats.{model_name}.median",
                execution_time_predictor_config=execution_time_predictor_config,
                training_context=dense_training_context,
            )
            train_kwargs.update(_layer_contract_kwargs(layer_contract))
            models[model_name] = self._train_single_model(**train_kwargs)
            trained_model_signatures.add(model_signature)
            logger.info(f"Trained {model_name} for {cluster_type}")

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

        model_config = replica_config.model_config
        model_arch = model_config.get_model_arch() if model_config is not None else "generic"
        architecture_profile_id = _resolve_model_architecture_profile_id(model_config)
        attention_signature = (
            f"attention_{replica_config.device}_{replica_config.model_name}_{tp_size}"
            f"_{replica_scheduler_config.block_size}_arch_profile{architecture_profile_id}"
            f"_family{self._measurement_family_name(self._active_measurement_type)}"
        )

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
            'model_architecture_profile': architecture_profile_id,
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

        model_config = replica_config.model_config
        architecture_profile = _resolve_model_architecture_profile(model_config)
        predictor_attention_extra_ops = (
            architecture_profile.predictor_attention_extra_ops
            if architecture_profile is not None
            else ()
        )
        for model_name in predictor_attention_extra_ops:
                model_signature = f"{model_name}_{attention_signature}"
                if model_signature not in trained_model_signatures:
                    target_col = f"time_stats.{model_name}.median"
                    if target_col not in attn_linear_ops_df.columns:
                        raise ValueError(
                            f"Architecture-profile operation '{model_name}' column '{target_col}' not found in profiling data. "
                            f"Ensure profiling was run with the selected model architecture profile. "
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
                    logger.info("Trained architecture-profile %s for %s", model_name, cluster_type)

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
        training_context['input_file'] = attn_file

        # Family-aware attention-core training. Latent-MLA profiles carry six
        # ``attn_mla_*`` operators with a structural layout the dense block cannot
        # consume; route them through the MLA branch before the dense derive (which
        # assumes dense feature columns such as ``prefill_chunk_size``).
        if self._is_mla_family(replica_config.model_config):
            attention_df = self._get_mla_attention_df_with_derived_features(
                attention_df
            )
            logger.info(
                f"Loaded {len(attention_df)} rows for latent-MLA attention core training"
            )
            models.update(
                self._train_mla_attention_core_models(
                    attention_df=attention_df,
                    attention_signature=attention_signature,
                    cluster_type=cluster_type,
                    execution_time_predictor_config=execution_time_predictor_config,
                    training_context=training_context,
                    trained_model_signatures=trained_model_signatures,
                )
            )
            trained_model_signatures.add(attention_signature)
            return models

        attention_df = self._get_attention_df_with_derived_features(attention_df)
        logger.info(f"Loaded {len(attention_df)} rows for attention core training")
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
                persist_exact_lookup=True,
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
                        persist_exact_lookup=True,
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
                        persist_exact_lookup=True,
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

    @staticmethod
    def _is_mla_family(model_config) -> bool:
        """Return True when the model binds to the latent-MLA attention family."""
        if model_config is None:
            return False
        return (
            bind_attention_family(model_config).family.family_id
            == LATENT_MLA_ATTENTION_FAMILY.family_id
        )

    def _get_mla_attention_df_with_derived_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Derive latent-MLA attention features (normalize ``is_prefill`` to int).

        Mirrors the monolithic ``SklearnExecutionTimePredictor`` MLA early-return:
        latent-MLA training keys on the imported structural columns directly and must
        NOT add the dense ``num_tokens`` / ``prefill_chunk_size`` derived features.
        """
        df_with_derived_features = df.copy()
        if "is_prefill" in df_with_derived_features.columns:
            df_with_derived_features["is_prefill"] = coerce_truthy_int(
                df_with_derived_features["is_prefill"]
            )
        return df_with_derived_features

    def _filter_mla_attention_df(
        self,
        df: pd.DataFrame,
        file_path: str,
        replica_config,
        replica_scheduler_config,
    ) -> pd.DataFrame:
        """Filter an imported latent-MLA profile to the requested structural layout.

        Verbatim port of the monolithic ``_filter_mla_attention_df`` (adapted to the
        shared-manager's per-cluster ``replica_config`` / ``replica_scheduler_config``
        instead of instance state). Fail-fast on missing structural columns or an empty
        post-filter frame per §7 (no silent fallback).
        """
        validate_attention_profiling_dataframe(
            df,
            LATENT_MLA_ATTENTION_FAMILY,
            measurement_type=self._active_measurement_type,
        )

        model_config = replica_config.model_config
        expected_values = {
            "n_q_head": int(getattr(model_config, "num_q_heads")),
            "n_kv_head": int(
                model_config.get_runtime_num_kv_heads()
                if hasattr(model_config, "get_runtime_num_kv_heads")
                else 1
            ),
            "head_size": int(
                model_config.get_runtime_head_size()
                if hasattr(model_config, "get_runtime_head_size")
                else int(getattr(model_config, "kv_lora_rank"))
                + int(getattr(model_config, "qk_rope_head_dim"))
            ),
            "qk_nope_head_dim": int(getattr(model_config, "qk_nope_head_dim")),
            "qk_rope_head_dim": int(getattr(model_config, "qk_rope_head_dim")),
            "qk_head_dim": int(model_config.get_qk_head_dim()),
            "kv_lora_rank": int(getattr(model_config, "kv_lora_rank")),
            "v_head_dim": int(getattr(model_config, "v_head_dim")),
            "block_size": int(replica_scheduler_config.block_size),
            "num_tensor_parallel_workers": int(
                replica_config.attn_tensor_parallel_size
            ),
        }
        missing_columns = [
            column for column in expected_values if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                "MLA attention profiling data is missing structural columns: "
                f"{missing_columns}. file={file_path}"
            )

        filtered = df.copy()
        for column, expected_value in expected_values.items():
            filtered = filtered[filtered[column].astype(int) == expected_value]

        if filtered.empty:
            raise ValueError(
                "No MLA attention profiling rows remain after structural filtering. "
                f"file={file_path}, expected={expected_values}"
            )
        return filtered

    def _train_mla_attention_core_models(
        self,
        attention_df: pd.DataFrame,
        attention_signature: str,
        cluster_type: ClusterType,
        execution_time_predictor_config,
        training_context: Dict[str, Any],
        trained_model_signatures: set,
    ) -> Dict[str, BaseEstimator]:
        """Train the six latent-MLA attention-core operators (training-only A2 fix).

        Mirrors the monolithic ``_train_mla_attention_layer_models`` (sparse-by-target
        row filtering + exact-row memoization). The on-demand consumer pairs each
        estimator's ``_frontier_exact_lookup`` with the same module-level builder, so
        the disaggregation prediction path works unchanged once these models exist.
        """
        model_names = list(
            get_enabled_predictor_metric_names(LATENT_MLA_ATTENTION_FAMILY)
        )
        target_columns = dict(
            zip(
                model_names,
                get_enabled_predictor_median_columns(LATENT_MLA_ATTENTION_FAMILY),
            )
        )
        feature_columns = get_enabled_shared_predictor_feature_columns(
            LATENT_MLA_ATTENTION_FAMILY
        )

        models: Dict[str, BaseEstimator] = {}
        for model_name in model_names:
            model_signature = f"{model_name}_{attention_signature}"
            if model_signature in trained_model_signatures:
                continue

            feature_cols = list(feature_columns[model_name])
            target_col = target_columns[model_name]
            required_columns = [*feature_cols, target_col]
            missing_columns = [
                column
                for column in required_columns
                if column not in attention_df.columns
            ]
            all_nan_columns = [
                column
                for column in required_columns
                if column in attention_df.columns
                and attention_df[column].isna().all()
            ]
            if missing_columns or all_nan_columns:
                raise ValueError(
                    "MLA attention profiling data cannot train "
                    f"{model_name}."
                    f"\nMissing columns: {missing_columns}"
                    f"\nAll-NaN columns: {all_nan_columns}"
                )

            op_attention_df = attention_df.dropna(subset=[target_col]).copy()
            if op_attention_df.empty:
                raise ValueError(
                    "MLA attention profiling data cannot train "
                    f"{model_name}: target column {target_col!r} has no "
                    "observed timing rows."
                )
            nan_feature_columns = [
                column
                for column in feature_cols
                if op_attention_df[column].isna().any()
            ]
            if nan_feature_columns:
                raise ValueError(
                    "MLA attention profiling data cannot train "
                    f"{model_name}: feature columns contain NaN after "
                    f"target filtering: {nan_feature_columns}"
                )

            model = self._train_single_model(
                model_name=model_name,
                df=op_attention_df,
                feature_cols=feature_cols,
                target_col=target_col,
                execution_time_predictor_config=execution_time_predictor_config,
                training_context=training_context,
                persist_exact_lookup=True,
            )
            if not hasattr(model, "_frontier_exact_lookup"):
                model._frontier_exact_lookup = _build_exact_feature_lookup(
                    op_attention_df,
                    feature_cols,
                    target_col,
                )
            models[model_name] = model
            trained_model_signatures.add(model_signature)
            logger.info(f"Trained {model_name} for {cluster_type}")

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
                    persist_exact_lookup=True,
                )
                if not hasattr(model, "_frontier_exact_lookup"):
                    model._frontier_exact_lookup = _build_exact_feature_lookup(
                        cpu_overhead_df,
                        feature_cols,
                        target_col,
                    )
                models[model_name] = model
                trained_model_signatures.add(model_signature)

        trained_model_signatures.add(cpu_signature)
        return models

    def _train_single_model(
        self,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        execution_time_predictor_config,
        training_context: Optional[Dict[str, Any]] = None,
        persist_exact_lookup: bool = True,
        layer_contract: Optional[ResolvedLayerContract] = None,
    ) -> BaseEstimator:
        """Train a single model with given data and configuration."""
        layer_contract, training_context = _normalize_layer_contract_context(
            training_context,
            explicit_layer_contract=layer_contract,
        )
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
        hash_args = (
            model_name,
            df,
            execution_time_predictor_config,
            profiling_precision,
            measurement_type,
        )
        model_hash = self._get_model_hash(
            *hash_args,
            **_layer_contract_kwargs(layer_contract),
        )
        cached_model = self._load_model_from_cache(model_name, model_hash)
        if cached_model is not None:
            if layer_contract is not None:
                requested_identity = _serialize_selected_layer_cache_identity(
                    layer_contract
                )
                if requested_identity is None:
                    raise ValueError(
                        "resolved layer contract did not produce a cache identity"
                    )
                self._validate_cached_layer_cache_identity(
                    model_name=model_name,
                    model=cached_model,
                    requested_identity=requested_identity,
                )
            if persist_exact_lookup:
                self._ensure_exact_lookup_metadata(
                    model_name=model_name,
                    model_hash=model_hash,
                    model=cached_model,
                    df=df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                )
            self._store_model_precision(
                model_name,
                profiling_precision,
                cached_model,
                **_layer_contract_kwargs(layer_contract),
            )
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
        if layer_contract is not None:
            self._model_contract_identity(best_estimator, layer_contract)

        if persist_exact_lookup:
            setattr(
                best_estimator,
                "_frontier_exact_lookup",
                _build_exact_feature_lookup(df, feature_cols, target_col),
            )

        self._store_model_in_cache(model_name, model_hash, best_estimator)
        self._store_model_precision(
            model_name,
            profiling_precision,
            best_estimator,
            **_layer_contract_kwargs(layer_contract),
        )
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
        layer_contract: Optional[ResolvedLayerContract] = None,
        operator_name: Optional[str] = None,
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
        if layer_contract is not None:
            _validate_typed_parallel_selection(
                layer_contract,
                tensor_parallel_size=tensor_parallel_size,
            )

        # Check file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Linear ops input file does not exist: {file_path}\n"
                f"Please run profiling first to generate this file.\n"
                f"Suggested command: bash frontier/profiling/example/test_profiling_linear_op.sh"
            )

        df = pd.read_csv(file_path)
        logger.info(f"Original linear ops data: {len(df)} rows, {len(df.columns)} columns")
        expected_profile = (
            layer_contract.profile_id
            if layer_contract is not None
            else infer_single_runtime_profile(self)
        )
        if expected_profile is not None and "model_architecture_profile" in df.columns:
            validate_model_architecture_profile(
                df,
                file_path=file_path,
                expected_profile=expected_profile,
            )

        # Check required column
        if 'num_tensor_parallel_workers' not in df.columns:
            raise ValueError(
                f"Column 'num_tensor_parallel_workers' not found in {file_path}\n"
                f"Available columns: {list(df.columns)}\n"
                f"This may indicate a corrupted or incompatible profiling file."
            )

        has_typed_contracts = TYPED_OPERATOR_CONTRACTS_COLUMN in df.columns
        parsed_typed_contracts: Optional[pd.Series] = None
        if has_typed_contracts:
            if not operator_name and layer_contract is not None:
                raise ValueError(
                    "typed profiling loading requires operator_name when the "
                    f"canonical {TYPED_OPERATOR_CONTRACTS_COLUMN!r} column is present"
                )
            if operator_name is not None and layer_contract is None:
                raise ValueError(
                    "typed profiling loading requires layer_contract when the "
                    f"canonical {TYPED_OPERATOR_CONTRACTS_COLUMN!r} column is present"
                )
            # Parse every row before applying scalar filters so malformed metadata
            # cannot be hidden by an unrelated TP or width selector.
            parsed_typed_contracts = cast(
                pd.Series,
                df[TYPED_OPERATOR_CONTRACTS_COLUMN].map(
                    lambda raw_contracts: validate_typed_operator_contracts(
                        raw_contracts,
                        model_config=infer_single_runtime_model_config(self),
                    )
                ),
            )

        # Show filtering conditions
        available_tp = sorted(df['num_tensor_parallel_workers'].unique())
        logger.info(f"Filtering conditions:")
        logger.info(f"  - num_tensor_parallel_workers == {tensor_parallel_size}")
        logger.info(f"  - Available num_tensor_parallel_workers: {available_tp}")

        # Apply filtering
        filtered_df: pd.DataFrame = cast(
            pd.DataFrame,
            df[df["num_tensor_parallel_workers"] == tensor_parallel_size],
        )
        if parsed_typed_contracts is not None and layer_contract is not None:
            selected_layer_contract = layer_contract
            if not isinstance(operator_name, str) or not operator_name:
                raise ValueError(
                    "typed profiling loading requires a non-empty operator_name "
                    "for contract matching"
                )
            typed_mask = parsed_typed_contracts.loc[filtered_df.index].map(
                lambda raw_contracts: _typed_row_matches_contract(
                    raw_contracts,
                    selected_layer_contract,
                    operator_name=operator_name,
                )
            )
            filtered_df = cast(pd.DataFrame, filtered_df[typed_mask])
            if filtered_df.empty:
                raise ValueError(
                    "No linear-op rows match the selected typed layer contract "
                    f"for operator={operator_name!r}, TP={tensor_parallel_size} "
                    f"in {file_path}"
                )
        elif layer_contract is not None:
            if "n_expanded_embd" not in filtered_df.columns:
                raise ValueError(
                    "Legacy linear-op profiling data is missing 'n_expanded_embd' "
                    f"for typed contract loading in {file_path}"
                )
            filtered_df = cast(
                pd.DataFrame,
                filtered_df[
                    filtered_df["n_expanded_embd"]
                    == layer_contract.effective_ffn_width
                ],
            )
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
            width_requirement = (
                layer_contract.effective_ffn_width
                if layer_contract is not None
                else "legacy model width"
            )
            raise ValueError(
                f"No data matches the filtering criteria in {file_path}\n"
                f"Required tensor_parallel_size: {tensor_parallel_size}\n"
                f"Available tensor_parallel_sizes: {available_tp}\n"
                f"Required effective_ffn_width: {width_requirement}\n"
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
        architecture_profile = _resolve_model_architecture_profile(model_config)
        if architecture_profile is not None:
            required_columns.extend(
                f"time_stats.{op_name}.median"
                for op_name in architecture_profile.predictor_attention_extra_ops
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

        # Latent-MLA profiles use a distinct structural schema (runtime kv heads = 1,
        # head size = kv_lora_rank + qk_rope_head_dim); route them to the MLA
        # structural filter before the dense cache-write fill / dense filter.
        model_config = replica_config.model_config
        if self._is_mla_family(model_config):
            return self._filter_mla_attention_df(
                df, file_path, replica_config, replica_scheduler_config
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
        layer_contract: Optional[ResolvedLayerContract] = None,
        operator_name: Optional[str] = None,
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
        if layer_contract is not None:
            _validate_typed_parallel_selection(
                layer_contract,
                tensor_parallel_size=tensor_parallel_size,
                expert_parallel_size=expert_parallel_size,
            )

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"MoE input file does not exist: {file_path}\n"
                f"Please run MoE profiling first.\n"
                f"Suggested command: bash frontier/profiling/example/test_profiling_moe.sh"
            )

        df = pd.read_csv(file_path)
        logger.info(f"Original MoE data: {len(df)} rows, {len(df.columns)} columns")
        expected_profile = (
            layer_contract.profile_id
            if layer_contract is not None
            else infer_single_runtime_profile(self)
        )
        if expected_profile is not None and "model_architecture_profile" in df.columns:
            validate_model_architecture_profile(
                df,
                file_path=file_path,
                expected_profile=expected_profile,
            )

        has_typed_contracts = TYPED_OPERATOR_CONTRACTS_COLUMN in df.columns
        parsed_typed_contracts: Optional[pd.Series] = None
        if has_typed_contracts:
            if not operator_name:
                raise ValueError(
                    "typed profiling loading requires operator_name when the "
                    f"canonical {TYPED_OPERATOR_CONTRACTS_COLUMN!r} column is present"
                )
            if layer_contract is None:
                raise ValueError(
                    "typed profiling loading requires layer_contract when the "
                    f"canonical {TYPED_OPERATOR_CONTRACTS_COLUMN!r} column is present"
                )
            # Parse every row before applying scalar filters so malformed metadata
            # cannot be hidden by an unrelated TP, EP, or width selector.
            parsed_typed_contracts = cast(
                pd.Series,
                df[TYPED_OPERATOR_CONTRACTS_COLUMN].map(
                    lambda raw_contracts: validate_typed_operator_contracts(
                        raw_contracts,
                        model_config=replica_config.model_config,
                    )
                ),
            )

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
        expected_expert_width = (
            layer_contract.effective_ffn_width
            if layer_contract is not None
            else model_config.mlp_hidden_dim
        )
        logger.info(f"  - expert_hidden_dim == {expected_expert_width}")
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
        filtered_df = cast(pd.DataFrame, df[
            (df["num_experts"] == model_config.num_experts)
            & (df["router_topk"] == model_config.num_experts_per_tok)
            & (df["hidden_dim"] == model_config.embedding_dim)
            & (df["num_tensor_parallel_workers"] == tensor_parallel_size)
        ])
        if not has_typed_contracts:
            filtered_df = filtered_df[
                filtered_df["expert_hidden_dim"] == expected_expert_width
            ]
        else:
            if parsed_typed_contracts is None:
                raise RuntimeError(
                    "typed MoE metadata column was detected but could not be parsed"
                )
            if layer_contract is None:
                raise ValueError(
                    "typed MoE filtering requires a resolved layer contract"
                )
            selected_layer_contract = layer_contract
            typed_mask = parsed_typed_contracts.loc[filtered_df.index].map(
                lambda raw_contracts: _typed_row_matches_contract(
                    raw_contracts,
                    selected_layer_contract,
                    operator_name=operator_name,
                )
            )
            filtered_df = cast(pd.DataFrame, filtered_df[typed_mask])
        filtered_df = cast(pd.DataFrame, filtered_df)
        if expert_parallel_size is not None:
            if "expert_parallel_size" not in filtered_df.columns:
                raise ValueError(
                    "MoE profiling data is missing 'expert_parallel_size' while "
                    f"EP={expert_parallel_size} is required in {file_path}"
                )
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
                f"  - expert_hidden_dim: {expected_expert_width}\n"
                f"  - tensor_parallel_size: {tensor_parallel_size}\n"
                f"  - expert_parallel_size: {ep_requirement}\n"
                f"  - training_mode: {training_mode}\n"
            )
            if has_typed_contracts:
                message += (
                    f"  - typed operator: {operator_name!r}\n"
                    "  - typed layer contract admission: required\n"
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
            normalized_prefill_values = coerce_truthy_bool(
                df_with_derived_features["is_prefill"]
            )
            df_with_derived_features["is_decode"] = ~normalized_prefill_values
        else:
            df_with_derived_features["is_decode"] = (df_with_derived_features["prefill_chunk_size"] == 0)
        df_with_derived_features["prefill_chunk_size_squared"] = (df_with_derived_features["prefill_chunk_size"] ** 2)

        def _normalize_bool_series(series: pd.Series) -> pd.Series:
            return coerce_truthy_bool(series)

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
            "kernel_only": "_trained_models_kernel_only_by_contract",
        }.get(family_name)
        if registry_attr is None:
            raise ValueError(f"Unsupported family_name={family_name!r}")
        registry = getattr(self, registry_attr, None)
        if registry is None:
            registry = {}
            setattr(self, registry_attr, registry)
        return registry

    def _contract_precision_registry(
        self, family_name: str
    ) -> Dict[str, Dict[Tuple[str, Optional[str]], BaseEstimator]]:
        registry_attr = {
            "eager": "_models_by_precision_eager_by_contract",
            "kernel_only": "_models_by_precision_kernel_only_by_contract",
        }.get(family_name)
        if registry_attr is None:
            raise ValueError(f"Unsupported family_name={family_name!r}")
        registry = getattr(self, registry_attr, None)
        if registry is None:
            registry = {}
            setattr(self, registry_attr, registry)
        return registry

    def _legacy_model_registry(self, family_name: str) -> Dict[str, BaseEstimator]:
        registry_attr = {
            "eager": "_trained_models_eager",
            "kernel_only": "_trained_models_kernel_only",
        }.get(family_name)
        if registry_attr is None:
            raise ValueError(f"Unsupported family_name={family_name!r}")
        registry = getattr(self, registry_attr, None)
        if registry is None:
            registry = {}
            setattr(self, registry_attr, registry)
        return registry

    def _legacy_precision_registry(
        self, family_name: str
    ) -> Dict[str, Dict[str, BaseEstimator]]:
        registry_attr = {
            "eager": "_models_by_precision_eager",
            "kernel_only": "_models_by_precision_kernel_only",
        }.get(family_name)
        if registry_attr is None:
            raise ValueError(f"Unsupported family_name={family_name!r}")
        registry = getattr(self, registry_attr, None)
        if registry is None:
            registry = {}
            setattr(self, registry_attr, registry)
        return registry

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
        cluster_config = (getattr(self, "_cluster_configs", None) or {}).get(
            cluster_type
        )
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
        for family_name in ("eager", "kernel_only"):
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
                    for value in self._contract_precision_registry("kernel_only")
                }
                | {
                    str(value).upper()
                    for value in self._legacy_precision_registry("eager")
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

    def get_models(self) -> Dict[str, Dict[str, BaseEstimator]]:
        """Return the trained models grouped by measurement family."""
        if self._all_dummy_mode:
            logger.debug("Returning empty models dict for dummy mode")
            return {"eager": {}, "kernel_only": {}}
        return {
            "eager": self._models_view_for_family("eager"),
            "kernel_only": self._models_view_for_family("kernel_only"),
        }

    def get_models_for_cluster(self, cluster_type: ClusterType) -> Dict[str, Dict[str, BaseEstimator]]:
        """Return a cluster-specific view of trained models grouped by measurement family."""
        if self._all_dummy_mode:
            return {"eager": {}, "kernel_only": {}}

        if cluster_type == ClusterType.PREFILL:
            return {
                "eager": self._models_view_for_family("eager", cluster_type),
                "kernel_only": {},
            }
        if cluster_type in [ClusterType.DECODE, ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN]:
            if (
                global_vars.get_sys_arch() == "pd-af-disaggregation"
                and cluster_type == ClusterType.DECODE_ATTN
            ):
                return {
                    "eager": self._models_view_for_family("eager", cluster_type),
                    "kernel_only": self._models_view_for_family(
                        "kernel_only", cluster_type
                    ),
                }
            if not self._is_kernel_only_measurement_enabled_for_cluster(cluster_type):
                return {
                    "eager": self._models_view_for_family("eager", cluster_type),
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
                "eager": self._models_view_for_family("eager", cluster_type),
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
