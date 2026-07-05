import ast
import copy
import hashlib
import json
import os
import pickle
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import math

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
from frontier.attention.ops import AttentionPhase
from frontier.attention.string_coercion import coerce_truthy_bool, coerce_truthy_int
from frontier.attention.profiling_mapping import (
    get_enabled_predictor_feature_columns,
    get_enabled_predictor_median_column_by_role,
    get_enabled_predictor_median_columns,
    get_enabled_predictor_metric_name_by_role,
    get_enabled_predictor_metric_names,
    get_enabled_shared_predictor_feature_columns,
    validate_attention_profiling_dataframe,
)
from frontier.config import (
    BaseExecutionTimePredictorConfig,
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    PrecisionType,
    ReplicaConfig,
    global_vars,
    get_quantization_manager,
)
from frontier.entities import Batch
from frontier.entities.time_components import (
    AttentionTime,
    AttentionOperatorTimes,
    CommunicationOperatorTimes,
    MLPOperatorTimes,
    MLPTime,
    MoETime,
)
from frontier.execution_time_predictor.base_execution_time_predictor import (
    BaseExecutionTimePredictor,
)
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.attention_tp_policy import (
    resolve_effective_attention_tp_size,
)
from frontier.execution_time_predictor.attention_dataset_contract import (
    enforce_mixed_attention_input_contract,
)
from frontier.logger import init_logger
from frontier.moe_gating_runtime import get_moe_gating_base_model_name
from frontier.model_architectures import ModelArchitectureProfile
from frontier.operators.families import (
    FFN_FAMILY,
    MEMORY_FAMILY,
    SHARE_EXPERT_FAMILY,
    get_family_profiling_names,
    get_family_profiling_name_set,
    get_comm_operator,
)
from frontier.operators.spec import CommOperatorSpec, CommPayloadContext, OperatorSpec
from frontier.profiling.cpu_overhead.schema import (
    DEFAULT_NUM_DECODE_TOKENS_AMPLIFICATION_FACTOR,
    DEFAULT_NUM_PREFILL_TOKENS,
)
from frontier.profiling.cpu_overhead.validation import (
    apply_cpu_overhead_schema_v2_defaults,
    validate_cpu_overhead_dataframe,
)
from frontier.profiling.other_overhead.validation import (
    validate_pp_producer_send_path_dataframe,
    validate_pp_receiver_head_dataframe,
    validate_pp_stage_boundary_dataframe,
)
from frontier.spec_decode import (
    build_mtp_runtime_contract,
    get_decode_draft_proposer_latency_ms,
    get_mtp_method_family,
    is_target_embedded_mtp_enabled,
)
from frontier.spec_decode.mtp_registry import is_target_embedded_mtp_same_tp_linear_op
from frontier.spec_decode.mtp_runtime import load_mtp_structural_model_config
from frontier.entities import ExecutionTime
from frontier.types import ClusterType, MeasurementType


@dataclass(frozen=True)
class ProfilingMetadata:
    profiling_precision: PrecisionType
    quant_signature: str
    model_arch: str
    model_architecture_profile: str
    measurement_type: MeasurementType


def _build_exact_feature_lookup(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Dict[Tuple[float, ...], float]:
    """Build exact profiling-row lookups before falling back to regression."""
    if df.empty:
        return {}
    for feature_col in feature_cols:
        non_scalar_rows = df[feature_col].map(
            lambda value: isinstance(value, (list, tuple, dict, set))
        )
        if bool(non_scalar_rows.any()):
            raise ValueError(
                "Exact feature lookup requires scalar numeric feature values; "
                f"column {feature_col!r} contains non-scalar values. "
                "Keep request-token vectors such as batch_request_num_tokens "
                "out of numeric exact keys until a vector-key schema is designed."
            )
    grouped = df.groupby(feature_cols, dropna=False)[target_col].mean()
    lookup: Dict[Tuple[float, ...], float] = {}
    for key, value in grouped.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        lookup[tuple(float(item) for item in key_tuple)] = float(value)
    return lookup


if TYPE_CHECKING:
    from frontier.entities import EPBatchGroup
    from frontier.cc_backend import BaseCCBackend

logger = init_logger(__name__)
MIGRATION_HELP_COMMAND = (
    "python -m frontier.profiling.migrate_csv_metadata --help"
)


def _get_operator_spec_by_name(family, op_name: str) -> OperatorSpec:
    matches = tuple(operator for operator in family.operators if operator.name == op_name)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one operator named {op_name!r} in family "
            f"{family.family_id!r}; found {len(matches)}"
        )
    return matches[0]


class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):
    @staticmethod
    def _dense_attention_cache_write_op_name() -> str:
        return get_enabled_predictor_metric_name_by_role(
            DENSE_ATTENTION_FAMILY,
            AttentionOperatorRole.CACHE_WRITE,
        )

    @staticmethod
    def _dense_attention_prefill_op_name() -> str:
        return get_enabled_predictor_metric_name_by_role(
            DENSE_ATTENTION_FAMILY,
            AttentionOperatorRole.PREFILL_KERNEL,
        )

    @staticmethod
    def _dense_attention_decode_op_name() -> str:
        return get_enabled_predictor_metric_name_by_role(
            DENSE_ATTENTION_FAMILY,
            AttentionOperatorRole.DECODE_KERNEL,
        )

    def _get_attention_family(self):
        return bind_attention_family(self._model_config).family

    def _is_mla_attention_family(self) -> bool:
        return (
            self._get_attention_family().family_id
            == LATENT_MLA_ATTENTION_FAMILY.family_id
        )

    def _get_model_architecture_profile(self) -> ModelArchitectureProfile:
        getter = getattr(self._model_config, "get_model_architecture_profile", None)
        if not callable(getter):
            raise TypeError(
                "SklearnExecutionTimePredictor requires "
                "model_config.get_model_architecture_profile()"
            )
        profile = getter()
        if not isinstance(profile, ModelArchitectureProfile):
            raise TypeError(
                "model_config.get_model_architecture_profile() must return "
                "ModelArchitectureProfile"
            )
        return profile

    def _get_predictor_attention_extra_ops(self) -> tuple[str, ...]:
        return tuple(
            self._get_model_architecture_profile().predictor_attention_extra_ops
        )

    def __init__(
        self,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
        model_manager: "ExecutionTimePredictionModelManager" = None,
        cluster_type: ClusterType = None,
        training_file_paths: Dict[str, str] = None,
        cc_backend: Optional["BaseCCBackend"] = None,
    ) -> None:
        super().__init__(
            predictor_config=predictor_config,
            replica_config=replica_config,
            replica_scheduler_config=replica_scheduler_config,
            metrics_config=metrics_config,
        )

        self._cluster_type = cluster_type
        self._model_manager = model_manager
        self._cc_backend = cc_backend  # CC Backend for communication predictions
        self._attention_tp_warning_cache: Set[str] = set()

        self._initialize_file_paths(training_file_paths)
        self._pp_stage_boundary_lookup: Dict[Tuple[int, int, int, int, int, int], float] = {}
        self._pp_stage_boundary_profile_rows: List[
            Tuple[Tuple[int, int, int, int, int, int], float]
        ] = []
        self._pp_receiver_head_lookup: Dict[Tuple[int, int, int, int, int, int], float] = {}
        self._pp_producer_send_path_lookup: Dict[
            Tuple[int, int, int, int, int, int], float
        ] = {}
        self._pp_producer_send_path_profile_rows: List[
            Tuple[Tuple[int, int, int, int, int, int], float]
        ] = []
        self._pp_prefill_consumer_active_lookup: Dict[
            Tuple[int, int, int, int, int, int], float
        ] = {}
        self._pp_prefill_consumer_active_profile_rows: List[
            Tuple[Tuple[int, int, int, int, int, int], float]
        ] = []
        self._missing_pp_stage_boundary_predictions_logged: Set[str] = set()
        self._missing_pp_stage_boundary_metrics_logged = (
            self._missing_pp_stage_boundary_predictions_logged
        )
        self._missing_pp_receiver_head_predictions_logged: Set[str] = set()
        self._missing_pp_producer_send_path_predictions_logged: Set[str] = set()
        self._missing_pp_prefill_consumer_active_predictions_logged: Set[str] = set()
        self._initialize_pp_stage_boundary_lookup()
        self._initialize_pp_receiver_head_lookup()
        self._initialize_pp_producer_send_path_lookup()
        self._initialize_pp_prefill_consumer_active_lookup()
        get_quantization_manager().configure_from_model_config(self._model_config)
        self._log_quantization_details()

        self._attention_prefill_batching_overhead_fraction = (
            (self._config.attention_prefill_batching_overhead_fraction)
            if self._model_config.num_q_heads > self._model_config.num_kv_heads
            else 0
        )
        self._attention_decode_batching_overhead_fraction = (
            (self._config.attention_decode_batching_overhead_fraction)
            if self._model_config.num_q_heads > self._model_config.num_kv_heads
            else 0
        )
        self._attn_pre_proj_calibration_scale = self._get_calibration_scale(
            "_attn_pre_proj_calibration_scale", "attn_pre_proj_calibration_scale"
        )
        self._attn_post_proj_calibration_scale = self._get_calibration_scale(
            "_attn_post_proj_calibration_scale", "attn_post_proj_calibration_scale"
        )
        self._attn_decode_calibration_scale = self._get_calibration_scale(
            "_attn_decode_calibration_scale", "attn_decode_calibration_scale"
        )
        self._attn_kv_cache_save_calibration_scale = self._get_calibration_scale(
            "_attn_kv_cache_save_calibration_scale",
            "attn_kv_cache_save_calibration_scale",
        )
        self._mlp_up_proj_calibration_scale = self._get_operator_calibration_scale(
            _get_operator_spec_by_name(FFN_FAMILY, "mlp_up_proj")
        )
        self._mlp_down_proj_calibration_scale = self._get_operator_calibration_scale(
            _get_operator_spec_by_name(FFN_FAMILY, "mlp_down_proj")
        )
        self._moe_shuffling_calibration_scale = self._get_calibration_scale(
            "_moe_shuffling_calibration_scale", "moe_shuffling_calibration_scale"
        )
        self._moe_grouped_gemm_calibration_scale = self._get_calibration_scale(
            "_moe_grouped_gemm_calibration_scale",
            "moe_grouped_gemm_calibration_scale",
        )
        self._expert_parallel_communication_calibration_scale = (
            self._get_calibration_scale(
                "_expert_parallel_communication_calibration_scale",
                "expert_parallel_communication_calibration_scale",
            )
        )
        # Predictor caches are indexed by Batch.get_effective_total_tokens_rounded(), which is a
        # *batch-level* compute-effective token count (e.g., batch_size * seq_len for prefill).
        # The helper name is backward-compatible; token values follow exact compute semantics.
        #
        # For schedulers with an explicit batch token budget (e.g., vLLM v1 max_num_batched_tokens),
        # we must ensure the cache range covers that budget to avoid KeyError at runtime.
        if self._replica_scheduler_provider == "orca":
            self._max_tokens = (
                self._config.prediction_max_tokens_per_request
                * self._config.prediction_max_batch_size
            )
        else:
            self._max_tokens = self._config.prediction_max_tokens_per_request

        max_tokens_in_batch = getattr(replica_scheduler_config, "max_tokens_in_batch", None)
        if max_tokens_in_batch is not None:
            self._max_tokens = max(self._max_tokens, int(max_tokens_in_batch))

        # Design note: PREFILL/MONOLITHIC cache sizing keeps the current shared max-token budget.
        if cluster_type in [
            ClusterType.PREFILL,
            ClusterType.DECODE_ATTN,
            ClusterType.MONOLITHIC,
        ]:
            tp_size = replica_config.attn_tensor_parallel_size
        else:  # ClusterType.DECODE_FFN
            tp_size = replica_config.moe_tensor_parallel_size
        num_workers = self._replica_config.num_pipeline_stages * tp_size
        devices_per_node = self._replica_config.node_config.num_devices_per_node
        assert (
            num_workers < devices_per_node or num_workers % devices_per_node == 0
        ), "Number of workers should be less than devices per node or a multiple of devices per node"

        self._is_multi_node = num_workers > devices_per_node

        # Call parent class initialization first
        super()._initialize_normal_mode()

        self._models_eager: Dict[str, BaseEstimator] = {}
        self._models_kernel_only: Dict[str, BaseEstimator] = {}
        self._predictions_eager: Dict[str, Any] = {}
        self._predictions_kernel_only: Dict[str, Any] = {}
        self._models = {}
        self._predictions = {}
        self._mtp_secondary_predictors: Dict[str, BaseExecutionTimePredictor] = {}
        self._active_measurement_type = self._get_default_measurement_type_for_cluster()

        if not self._enable_dummy_mode:
            if model_manager is not None:
                logger.info(
                    f"Using shared execution time prediction models from ExecutionTimePredictionModelManager for cluster {cluster_type}"
                )
                models_by_family = (
                    model_manager.get_models()
                    if cluster_type is None
                    else model_manager.get_models_for_cluster(cluster_type)
                )
                self._models_eager = dict(models_by_family.get("eager", {}))
                self._models_kernel_only = dict(models_by_family.get("kernel_only", {}))
            else:
                logger.info(
                    "Training execution time prediction models independently with eager/kernel-only families"
                )
                should_load_eager = self._should_enable_measurement_family(
                    MeasurementType.CUDA_EVENT
                )
                should_load_kernel_only = self._should_enable_measurement_family(
                    MeasurementType.KERNEL_ONLY
                )
                if should_load_eager:
                    self._models_eager = self._train_models_for_family(MeasurementType.CUDA_EVENT)
                if should_load_kernel_only:
                    self._models_kernel_only = self._train_models_for_family(MeasurementType.KERNEL_ONLY)

            self._predictions_eager = self._predict_from_models_for_family(
                MeasurementType.CUDA_EVENT, self._models_eager
            )
            self._predictions_kernel_only = self._predict_from_models_for_family(
                MeasurementType.KERNEL_ONLY, self._models_kernel_only
            )
        else:
            logger.info(
                "Skipped ML model training and prediction cache generation in dummy mode"
            )

        if not self._enable_dummy_mode and model_manager is not None:
            self._register_profiling_metadata_from_files()

        from collections import defaultdict

        self._runtime_cache: Dict[str, Dict[str, Dict[tuple, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self._activate_measurement_type(self._get_default_measurement_type_for_cluster())


    @staticmethod
    def _validate_positive_scale(scale: float, field_name: str) -> float:
        if scale <= 0.0:
            raise ValueError(f"{field_name} must be > 0, got={scale!r}")
        return scale

    def _get_calibration_scale(self, attr_name: str, field_name: str) -> float:
        attr_value = getattr(self, attr_name, None)
        if attr_value is not None:
            return self._validate_positive_scale(float(attr_value), field_name)

        config = getattr(self, "_config", None)
        if config is None:
            return 1.0

        config_value = getattr(config, field_name, 1.0)
        return self._validate_positive_scale(float(config_value), field_name)

    def _get_optional_calibration_scale(
        self, attr_name: str, field_name: str
    ) -> Optional[float]:
        attr_value = getattr(self, attr_name, None)
        if attr_value is not None:
            return self._validate_positive_scale(float(attr_value), field_name)

        config = getattr(self, "_config", None)
        if config is None:
            return None

        config_value = getattr(config, field_name, None)
        if config_value is None:
            return None
        return self._validate_positive_scale(float(config_value), field_name)

    def _get_operator_calibration_scale(self, operator: OperatorSpec) -> float:
        attr_name = operator.calibration_attr_name()
        field_name = operator.calibration_field_name()
        if attr_name is None or field_name is None:
            return 1.0
        return self._get_calibration_scale(attr_name, field_name)

    def _get_optional_operator_phase_calibration_scale(
        self,
        operator: OperatorSpec,
        phase_prefix: str,
    ) -> Optional[float]:
        if operator.calibration_key is None:
            return None
        field_name = f"{phase_prefix}_{operator.calibration_key}_calibration_scale"
        return self._get_optional_calibration_scale(f"_{field_name}", field_name)

    def _get_decode_phase_only_calibration_scale(
        self, batch: Batch, attr_name: str, field_name: str
    ) -> Optional[float]:
        if getattr(batch, "num_prefill_tokens", 0) != 0:
            return None
        return self._get_optional_calibration_scale(attr_name, field_name)

    def _get_late_decode_only_calibration_scale(
        self, batch: Batch, attr_name: str, field_name: str
    ) -> Optional[float]:
        if getattr(batch, "num_prefill_tokens", 0) != 0:
            return None

        requests = getattr(batch, "requests", None)
        if not requests:
            return None

        # Apply this override only when the entire decode-only batch is already
        # past the first pure decode token, so TTFT / decode-first semantics stay
        # anchored to the existing global scale.
        for request in requests:
            if int(getattr(request, "num_processed_decode_tokens", 0)) <= 1:
                return None

        return self._get_optional_calibration_scale(attr_name, field_name)

    def _get_decode_request_length_calibration_scale(
        self, batch: Batch
    ) -> Optional[float]:
        config = getattr(self, "_config", None)
        if config is None:
            return None
        request_length_calibration_fields = (
            "short_decode_request_length_threshold",
            "short_decode_request_length_calibration_scale",
            "long_decode_request_length_threshold",
            "long_decode_request_length_calibration_scale",
            "low_prefill_short_decode_request_prefill_threshold",
            "low_prefill_short_decode_request_decode_threshold",
            "low_prefill_short_decode_request_calibration_scale",
            "low_prefill_decode_mix_request_prefill_threshold",
            "low_prefill_decode_mix_request_decode_min",
            "low_prefill_decode_mix_request_decode_max",
            "low_prefill_decode_mix_request_min_match_ratio",
            "low_prefill_decode_mix_request_max_match_ratio",
            "low_prefill_decode_mix_request_calibration_scale",
            "low_prefill_long_decode_request_prefill_threshold",
            "low_prefill_long_decode_request_decode_threshold",
            "low_prefill_long_decode_request_calibration_scale",
            "high_prefill_mid_decode_request_prefill_threshold",
            "high_prefill_mid_decode_request_decode_min",
            "high_prefill_mid_decode_request_decode_max",
            "high_prefill_mid_decode_request_calibration_scale",
        )
        if not any(
            getattr(config, field_name, None) is not None
            for field_name in request_length_calibration_fields
        ):
            return None

        requests = getattr(batch, "requests", None)
        if not requests:
            return None

        num_prefill_tokens = int(getattr(batch, "num_prefill_tokens", 0))
        raw_num_decode_tokens = getattr(batch, "num_decode_tokens", None)
        if raw_num_decode_tokens is None:
            num_decode_tokens = sum(
                1
                for request in requests
                if int(getattr(request, "num_decode_tokens")) > 0
            )
        else:
            num_decode_tokens = int(raw_num_decode_tokens)
        is_mixed_batch = num_prefill_tokens > 0 and num_decode_tokens > 0
        include_low_mix_mixed_batches = bool(
            getattr(
                config,
                "low_prefill_decode_mix_request_include_mixed_batches",
                False,
            )
        )
        include_low_long_mixed_batches = bool(
            getattr(
                config,
                "low_prefill_long_decode_request_include_mixed_batches",
                False,
            )
        )
        if num_prefill_tokens != 0 and not (
            is_mixed_batch
            and (include_low_mix_mixed_batches or include_low_long_mixed_batches)
        ):
            return None

        request_shapes = [
            (
                int(getattr(request, "num_prefill_tokens")),
                int(getattr(request, "num_decode_tokens")),
            )
            for request in requests
        ]
        scale = 1.0
        matched = False

        low_short_prefill_threshold = getattr(
            config, "low_prefill_short_decode_request_prefill_threshold", None
        )
        low_short_decode_threshold = getattr(
            config, "low_prefill_short_decode_request_decode_threshold", None
        )
        if (
            not is_mixed_batch
            and low_short_prefill_threshold is not None
            and low_short_decode_threshold is not None
            and any(
                prefill_length <= int(low_short_prefill_threshold)
                and decode_length <= int(low_short_decode_threshold)
                for prefill_length, decode_length in request_shapes
            )
        ):
            shape_scale = self._get_optional_calibration_scale(
                "_low_prefill_short_decode_request_calibration_scale",
                "low_prefill_short_decode_request_calibration_scale",
            )
            if shape_scale is not None:
                scale *= shape_scale
                matched = True

        low_mix_prefill_threshold = getattr(
            config, "low_prefill_decode_mix_request_prefill_threshold", None
        )
        low_mix_decode_min = getattr(
            config, "low_prefill_decode_mix_request_decode_min", None
        )
        low_mix_decode_max = getattr(
            config, "low_prefill_decode_mix_request_decode_max", None
        )
        low_mix_min_match_ratio = getattr(
            config, "low_prefill_decode_mix_request_min_match_ratio", None
        )
        low_mix_max_match_ratio = getattr(
            config, "low_prefill_decode_mix_request_max_match_ratio", None
        )
        if (
            (not is_mixed_batch or include_low_mix_mixed_batches)
            and low_mix_prefill_threshold is not None
            and low_mix_decode_min is not None
            and low_mix_decode_max is not None
            and low_mix_min_match_ratio is not None
            and low_mix_max_match_ratio is not None
        ):
            matched_request_count = sum(
                prefill_length <= int(low_mix_prefill_threshold)
                and int(low_mix_decode_min) <= decode_length <= int(low_mix_decode_max)
                for prefill_length, decode_length in request_shapes
            )
            match_ratio = matched_request_count / len(request_shapes)
            if (
                float(low_mix_min_match_ratio)
                <= match_ratio
                <= float(low_mix_max_match_ratio)
            ):
                shape_scale = self._get_optional_calibration_scale(
                    "_low_prefill_decode_mix_request_calibration_scale",
                    "low_prefill_decode_mix_request_calibration_scale",
                )
                if shape_scale is not None:
                    scale *= shape_scale
                    matched = True

        low_prefill_threshold = getattr(
            config, "low_prefill_long_decode_request_prefill_threshold", None
        )
        low_decode_threshold = getattr(
            config, "low_prefill_long_decode_request_decode_threshold", None
        )
        if (
            (not is_mixed_batch or include_low_long_mixed_batches)
            and low_prefill_threshold is not None
            and low_decode_threshold is not None
            and any(
                prefill_length <= int(low_prefill_threshold)
                and decode_length >= int(low_decode_threshold)
                for prefill_length, decode_length in request_shapes
            )
        ):
            shape_scale = self._get_optional_calibration_scale(
                "_low_prefill_long_decode_request_calibration_scale",
                "low_prefill_long_decode_request_calibration_scale",
            )
            if shape_scale is not None:
                scale *= shape_scale
                matched = True

        high_prefill_threshold = getattr(
            config, "high_prefill_mid_decode_request_prefill_threshold", None
        )
        high_decode_min = getattr(
            config, "high_prefill_mid_decode_request_decode_min", None
        )
        high_decode_max = getattr(
            config, "high_prefill_mid_decode_request_decode_max", None
        )
        if (
            not is_mixed_batch
            and high_prefill_threshold is not None
            and high_decode_min is not None
            and high_decode_max is not None
            and any(
                prefill_length >= int(high_prefill_threshold)
                and int(high_decode_min) <= decode_length <= int(high_decode_max)
                for prefill_length, decode_length in request_shapes
            )
        ):
            shape_scale = self._get_optional_calibration_scale(
                "_high_prefill_mid_decode_request_calibration_scale",
                "high_prefill_mid_decode_request_calibration_scale",
            )
            if shape_scale is not None:
                scale *= shape_scale
                matched = True

        original_decode_lengths = [
            decode_length for _, decode_length in request_shapes
        ]
        long_threshold = getattr(config, "long_decode_request_length_threshold", None)
        if not is_mixed_batch and long_threshold is not None and any(
            length >= int(long_threshold) for length in original_decode_lengths
        ):
            length_scale = self._get_optional_calibration_scale(
                "_long_decode_request_length_calibration_scale",
                "long_decode_request_length_calibration_scale",
            )
            if length_scale is not None:
                scale *= length_scale
                matched = True

        short_threshold = getattr(config, "short_decode_request_length_threshold", None)
        if not is_mixed_batch and short_threshold is not None and all(
            length <= int(short_threshold) for length in original_decode_lengths
        ):
            length_scale = self._get_optional_calibration_scale(
                "_short_decode_request_length_calibration_scale",
                "short_decode_request_length_calibration_scale",
            )
            if length_scale is not None:
                scale *= length_scale
                matched = True

        return scale if matched else None

    def _log_quantization_details(self) -> None:
        quant_manager = get_quantization_manager()
        cluster_name = self._cluster_type.name if self._cluster_type else "NONE"
        model_signature = getattr(
            self._model_config, "get_quant_signature", lambda: "none"
        )()
        logger.info(
            "[QUANT] Model quant_signature=%s cluster=%s",
            model_signature,
            cluster_name,
        )
        quantization_config = getattr(self._model_config, "quantization_config", None)
        if quantization_config is not None:
            quant_config = quantization_config.to_dict()
            logger.info(
                "[QUANT] Model quantization_config=%s",
                json.dumps(quant_config, sort_keys=True),
            )

        supported_ops = quant_manager.get_supported_operations()
        op_precision = {}
        for op_name in (
            supported_ops["compute_operations"]
            + supported_ops["communication_operations"]
        ):
            op_precision[op_name] = quant_manager.get_precision(
                op_name, self._cluster_type
            ).name

        op_speedups = {}
        for op_name in supported_ops["compute_operations"]:
            speedup = quant_manager.get_compute_speedup_factor(
                op_name, self._cluster_type
            )
            if speedup != 1.0:
                op_speedups[op_name] = speedup

        logger.info(
            "[QUANT] Op precision map (cluster=%s): %s",
            cluster_name,
            json.dumps(op_precision, sort_keys=True),
        )
        logger.info(
            "[QUANT] Op speedup factors (non-1.0): %s",
            json.dumps(op_speedups, sort_keys=True),
        )

    def _initialize_file_paths(self, training_file_paths: Dict[str, str] = None):
        """Initialize eager and kernel-only file path attributes."""
        if training_file_paths:
            self._compute_input_file_eager = training_file_paths.get("compute_input_file", "")
            self._attention_input_file_eager = training_file_paths.get(
                "attention_input_file", ""
            )
            self._moe_input_file_eager = training_file_paths.get("moe_input_file", "")
            self._compute_input_file_kernel_only = training_file_paths.get(
                "compute_kernel_only_input_file", ""
            )
            self._attention_input_file_kernel_only = training_file_paths.get(
                "attention_kernel_only_input_file", ""
            )
            self._moe_input_file_kernel_only = training_file_paths.get(
                "moe_kernel_only_input_file", ""
            )
            self._all_reduce_input_file = training_file_paths.get(
                "all_reduce_input_file", ""
            )
            self._send_recv_input_file = training_file_paths.get(
                "send_recv_input_file", ""
            )
            self._cpu_overhead_input_file = training_file_paths.get(
                "cpu_overhead_input_file", ""
            )
            self._pp_stage_boundary_input_file = training_file_paths.get(
                "pp_stage_boundary_input_file", ""
            )
            self._pp_receiver_head_input_file = training_file_paths.get(
                "pp_receiver_head_input_file", ""
            )
            self._pp_producer_send_path_input_file = training_file_paths.get(
                "pp_producer_send_path_input_file", ""
            )
            self._pp_prefill_consumer_active_input_file = training_file_paths.get(
                "pp_prefill_consumer_active_input_file", ""
            )
        else:
            eager_files = self._get_input_files(MeasurementType.CUDA_EVENT)
            kernel_only_files = self._get_input_files(MeasurementType.KERNEL_ONLY)
            self._compute_input_file_eager = eager_files[0]
            self._attention_input_file_eager = eager_files[1]
            self._moe_input_file_eager = eager_files[2]
            self._compute_input_file_kernel_only = kernel_only_files[0]
            self._attention_input_file_kernel_only = kernel_only_files[1]
            self._moe_input_file_kernel_only = kernel_only_files[2]
            self._all_reduce_input_file = eager_files[3]
            self._send_recv_input_file = eager_files[4]
            self._cpu_overhead_input_file = eager_files[5]
            self._pp_stage_boundary_input_file = (
                self._config.pp_stage_boundary_input_file
                .replace("{DEVICE}", self._replica_config.device)
                .replace("{MODEL}", self._model_config.get_name())
                .replace("{NETWORK_DEVICE}", self._replica_config.network_device)
            )
            self._pp_receiver_head_input_file = (
                self._config.pp_receiver_head_input_file
                .replace("{DEVICE}", self._replica_config.device)
                .replace("{MODEL}", self._model_config.get_name())
                .replace("{NETWORK_DEVICE}", self._replica_config.network_device)
            )
            self._pp_producer_send_path_input_file = (
                self._config.pp_producer_send_path_input_file
                .replace("{DEVICE}", self._replica_config.device)
                .replace("{MODEL}", self._model_config.get_name())
                .replace("{NETWORK_DEVICE}", self._replica_config.network_device)
            )
            self._pp_prefill_consumer_active_input_file = (
                self._config.pp_prefill_consumer_active_input_file
                .replace("{DEVICE}", self._replica_config.device)
                .replace("{MODEL}", self._model_config.get_name())
                .replace("{NETWORK_DEVICE}", self._replica_config.network_device)
            )

        self._compute_input_file = self._compute_input_file_eager
        self._attention_input_file = self._attention_input_file_eager
        self._moe_input_file = self._moe_input_file_eager

    def _get_input_files(
        self, measurement_type: MeasurementType = MeasurementType.CUDA_EVENT
    ) -> Tuple[str, str, str, str, str, str]:
        if measurement_type == MeasurementType.CUDA_EVENT:
            compute_file = self._config.linear_op_input_file
            if not compute_file and self._config.mlp_input_file:
                compute_file = self._config.mlp_input_file
            attention_file = self._config.atten_input_file
            moe_file = self._config.moe_input_file
        elif measurement_type == MeasurementType.KERNEL_ONLY:
            compute_file = self._config.linear_op_kernel_only_input_file
            attention_file = self._config.atten_kernel_only_input_file
            moe_file = self._config.moe_kernel_only_input_file
        else:
            raise ValueError(f"Unsupported measurement_type={measurement_type!r}")

        input_files = [
            compute_file,
            attention_file,
            moe_file,
            self._config.all_reduce_input_file,
            self._config.send_recv_input_file,
            self._config.cpu_overhead_input_file,
        ]
        for i in range(len(input_files)):
            input_files[i] = (
                input_files[i]
                .replace("{DEVICE}", self._replica_config.device)
                .replace("{MODEL}", self._model_config.get_name())
                .replace("{NETWORK_DEVICE}", self._replica_config.network_device)
            )

        return tuple(input_files)

    @staticmethod
    def _measurement_family_name(measurement_type: MeasurementType) -> str:
        if measurement_type == MeasurementType.CUDA_EVENT:
            return "eager"
        if measurement_type == MeasurementType.KERNEL_ONLY:
            return "kernel_only"
        raise ValueError(f"Unsupported measurement_type={measurement_type!r}")

    def _is_kernel_only_measurement_enabled_for_cluster(self) -> bool:
        decode_cuda_graph_mode = str(global_vars.get_decode_cuda_graph_mode()).lower()
        use_cuda_graph = bool(global_vars.get_use_cuda_graph())

        if self._cluster_type == ClusterType.PREFILL:
            return False
        if self._cluster_type in (ClusterType.MONOLITHIC, ClusterType.DECODE):
            return decode_cuda_graph_mode != "none"
        if self._cluster_type in (ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN):
            return use_cuda_graph
        if self._cluster_type is None:
            return use_cuda_graph or decode_cuda_graph_mode != "none"
        raise ValueError(f"Unsupported cluster_type={self._cluster_type!r}")

    def _get_default_measurement_type_for_cluster(self) -> MeasurementType:
        if self._should_enable_measurement_family(MeasurementType.KERNEL_ONLY) and not (
            self._should_enable_measurement_family(MeasurementType.CUDA_EVENT)
        ):
            if self._is_kernel_only_measurement_enabled_for_cluster():
                return MeasurementType.KERNEL_ONLY
            return MeasurementType.CUDA_EVENT
        return MeasurementType.CUDA_EVENT

    def _should_enable_measurement_family(
        self, measurement_type: MeasurementType
    ) -> bool:
        decode_graph_mode = str(global_vars.get_decode_cuda_graph_mode()).strip().lower()
        use_cuda_graph = bool(global_vars.get_use_cuda_graph())

        if measurement_type == MeasurementType.CUDA_EVENT:
            if self._cluster_type == ClusterType.DECODE:
                return decode_graph_mode == "none"
            if self._cluster_type in (
                ClusterType.DECODE_ATTN,
                ClusterType.DECODE_FFN,
            ):
                return not use_cuda_graph
            return True

        if measurement_type == MeasurementType.KERNEL_ONLY:
            if self._cluster_type in (None, ClusterType.MONOLITHIC, ClusterType.DECODE):
                return decode_graph_mode != "none"
            if self._cluster_type in (
                ClusterType.DECODE_ATTN,
                ClusterType.DECODE_FFN,
            ):
                return use_cuda_graph
            return False

        raise ValueError(f"Unsupported measurement_type={measurement_type!r}")

    def _select_measurement_type_for_batch(self, batch: Batch) -> MeasurementType:
        if self._cluster_type in (
            ClusterType.PREFILL,
            ClusterType.DECODE,
            ClusterType.DECODE_ATTN,
            ClusterType.DECODE_FFN,
        ):
            return self._get_default_measurement_type_for_cluster()

        if getattr(batch, "num_prefill_tokens", 0) > 0:
            return MeasurementType.CUDA_EVENT

        if getattr(batch, "num_decode_tokens", 0) > 0:
            runtime_mode = self._get_decode_cuda_graph_runtime_mode(batch)
            if runtime_mode != "NONE":
                return MeasurementType.KERNEL_ONLY

        return MeasurementType.CUDA_EVENT

    def _activate_measurement_type(self, measurement_type: MeasurementType) -> None:
        self._active_measurement_type = measurement_type
        if measurement_type == MeasurementType.CUDA_EVENT:
            self._compute_input_file = self._compute_input_file_eager
            self._attention_input_file = self._attention_input_file_eager
            self._moe_input_file = self._moe_input_file_eager
            self._models = self._models_eager
            self._predictions = self._predictions_eager
        elif measurement_type == MeasurementType.KERNEL_ONLY:
            self._compute_input_file = self._compute_input_file_kernel_only
            self._attention_input_file = self._attention_input_file_kernel_only
            self._moe_input_file = self._moe_input_file_kernel_only
            self._models = self._models_kernel_only
            self._predictions = self._predictions_kernel_only
        else:
            raise ValueError(f"Unsupported measurement_type={measurement_type!r}")

    @contextmanager
    def _temporary_measurement_type(self, measurement_type: MeasurementType):
        previous_measurement_type = getattr(self, "_active_measurement_type", None)
        if previous_measurement_type == measurement_type:
            yield
            return

        self._activate_measurement_type(measurement_type)
        try:
            yield
        finally:
            if previous_measurement_type is not None:
                self._activate_measurement_type(previous_measurement_type)

    def _should_use_hybrid_attention_measurement_for_spec_piecewise(
        self, batch: Batch
    ) -> bool:
        spec_metadata = getattr(batch, "spec_decode_metadata", None)
        if spec_metadata is None:
            return False
        return self._get_decode_cuda_graph_runtime_mode(batch) == "PIECEWISE"

    def _predict_from_models_for_family(
        self, measurement_type: MeasurementType, models: Dict[str, BaseEstimator]
    ) -> Dict[str, Any]:
        if not models:
            return {}
        previous_measurement_type = getattr(self, "_active_measurement_type", None)
        previous_models = getattr(self, "_models", {})
        previous_predictions = getattr(self, "_predictions", {})
        self._models = models
        self._predictions = {}
        self._activate_measurement_type(measurement_type)
        try:
            return self._predict_from_models()
        finally:
            self._models = previous_models
            self._predictions = previous_predictions
            if previous_measurement_type is not None:
                self._activate_measurement_type(previous_measurement_type)

    def _train_models_for_family(
        self, measurement_type: MeasurementType
    ) -> Dict[str, BaseEstimator]:
        previous_measurement_type = getattr(self, "_active_measurement_type", None)
        self._activate_measurement_type(measurement_type)
        try:
            return self._train_models()
        finally:
            if previous_measurement_type is not None:
                self._activate_measurement_type(previous_measurement_type)

    def _require_predictions_for_measurement_type(
        self, measurement_type: MeasurementType, batch: Batch
    ) -> None:
        predictions = (
            self._predictions_eager
            if measurement_type == MeasurementType.CUDA_EVENT
            else self._predictions_kernel_only
        )
        if predictions:
            return
        raise ValueError(
            f"No {self._measurement_family_name(measurement_type)} predictor family is loaded for "
            f"cluster_type={self._cluster_type} batch_id={getattr(batch, 'id', 'unknown')}."
        )

    def _load_compute_df(
        self,
        file_path: str,
        tensor_parallel_size: Optional[int] = None,
        required_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        df = self._read_input_file(file_path)
        df = df.drop_duplicates()

        metadata = self._get_profiling_metadata(df, file_path)
        self._validate_active_measurement_type(metadata, file_path)
        self._register_profiling_metadata_for_ops(
            self._get_compute_model_names(), metadata, file_path
        )

        logger.debug(f"Length of complete compute df: {len(df)} {file_path}")
        logger.debug(f"self._num_q_heads: {self._model_config.num_q_heads}")
        logger.debug(f"self._embedding_dim: {self._model_config.embedding_dim}")
        logger.debug(f"self._mlp_hidden_dim: {self._model_config.mlp_hidden_dim}")
        logger.debug(f"self._use_gated_mlp: {self._model_config.use_gated_mlp}")
        logger.debug(f"self._vocab_size: {self._model_config.vocab_size}")
        # NOTE: linear_op.csv compute profiling now uses op-specific TP assignment:
        # - Replicated ops: TP=1
        # - Attention-sharded ops: attn_tensor_parallel_size
        # - FFN-sharded ops: ffn_tp (moe_tp for DECODE_FFN MoE, else attn_tp)
        if tensor_parallel_size is None:
            tensor_parallel_size = self._replica_config.attn_tensor_parallel_size
        logger.debug(
            f"Filtering linear-op compute data by num_tensor_parallel_workers={tensor_parallel_size}"
        )

        df = df[
            (df["n_head"] == self._model_config.num_q_heads)
            & (df["n_kv_head"] == self._model_config.num_kv_heads)
            & (df["n_embd"] == self._model_config.embedding_dim)
            & (df["n_expanded_embd"] == self._model_config.mlp_hidden_dim)
            & (df["use_gated_mlp"] == self._model_config.use_gated_mlp)
            & (df["vocab_size"] == self._model_config.vocab_size)
            & (
                df["num_tensor_parallel_workers"]
                == tensor_parallel_size
            )
        ]

        expected_use_qk_norm = bool(getattr(self._model_config, "use_qk_norm", False))
        if expected_use_qk_norm and "use_qk_norm" not in df.columns:
            raise ValueError(
                "linear_op profiling data is missing 'use_qk_norm' column for a model "
                "that requires QK-norm-aware filtering. "
                f"file={file_path}, model={self._model_config.get_name()}"
            )
        if "use_qk_norm" in df.columns:
            df = df[df["use_qk_norm"].astype(bool) == expected_use_qk_norm]

        expected_attn_output_gate = bool(
            getattr(self._model_config, "attn_output_gate", False)
        )
        if expected_attn_output_gate and "attn_output_gate" not in df.columns:
            raise ValueError(
                "linear_op profiling data is missing 'attn_output_gate' column for "
                "a model that requires gated-attention-aware filtering. "
                f"file={file_path}, model={self._model_config.get_name()}"
            )
        if "attn_output_gate" in df.columns:
            gate_mask = df["attn_output_gate"].map(
                lambda value: bool(value) if pd.notna(value) else False
            )
            df = (
                df[gate_mask == expected_attn_output_gate]
                .copy()
            )

        if len(df) == 0:
            raise ValueError(
                "No compute profiling rows remain after model and TP filtering. "
                f"file={file_path}, tp={tensor_parallel_size}, "
                f"use_qk_norm={expected_use_qk_norm}, "
                f"attn_output_gate={expected_attn_output_gate}"
            )

        if required_columns:
            self._verify_required_attn_linear_op_columns(df, required_columns, file_path)

        return df

    @staticmethod
    def _verify_required_attn_linear_op_columns(
        df: pd.DataFrame, required_columns: List[str], file_path: str
    ) -> None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        all_nan_columns = [
            col for col in required_columns if col in df.columns and df[col].isna().all()
        ]
        if missing_columns or all_nan_columns:
            raise ValueError(
                "Required attention linear op columns are missing or all-NaN in "
                f"{file_path}."
                f"\nMissing columns: {missing_columns}"
                f"\nAll-NaN columns: {all_nan_columns}"
            )

    def _load_attention_df(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df = df.drop_duplicates()

        enforce_mixed_attention_input_contract(
            attention_file_path=file_path,
            available_columns=df.columns,
        )

        metadata = self._get_profiling_metadata(df, file_path)
        self._validate_active_measurement_type(metadata, file_path)
        self._register_profiling_metadata_for_ops(
            self._get_attention_model_names(), metadata, file_path
        )

        if self._is_mla_attention_family():
            return self._filter_mla_attention_df(df, file_path)

        cache_write_median_column = get_enabled_predictor_median_column_by_role(
            DENSE_ATTENTION_FAMILY,
            AttentionOperatorRole.CACHE_WRITE,
        )
        for column in [cache_write_median_column]:
            if column not in df.columns:
                df[column] = 0
            else:
                df.fillna({column: 0}, inplace=True)

        prefill_op_name = get_enabled_predictor_metric_name_by_role(
            DENSE_ATTENTION_FAMILY,
            AttentionOperatorRole.PREFILL_KERNEL,
        )
        effective_tp = resolve_effective_attention_tp_size(
            op_name=prefill_op_name,
            requested_tp_size=self._replica_config.attn_tensor_parallel_size,
            num_kv_heads=self._model_config.num_kv_heads,
            cluster_type=self._cluster_type,
            warning_cache=getattr(self, "_attention_tp_warning_cache", None),
            include_linear_ops=False,
        )

        return df[
            (df["n_embd"] == self._model_config.embedding_dim)
            & (df["n_q_head"] == self._model_config.num_q_heads)
            & (df["n_kv_head"] == self._model_config.num_kv_heads)
            & (df["block_size"] == self._block_size)
            & (df["num_tensor_parallel_workers"] == effective_tp)
        ]

    def _filter_mla_attention_df(
        self, df: pd.DataFrame, file_path: str
    ) -> pd.DataFrame:
        validate_attention_profiling_dataframe(
            df,
            LATENT_MLA_ATTENTION_FAMILY,
            measurement_type=self._active_measurement_type,
        )

        model_config = self._model_config
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
            "block_size": int(self._block_size),
            "num_tensor_parallel_workers": int(
                self._replica_config.attn_tensor_parallel_size
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

    def _load_all_reduce_df(self, file_path: str) -> pd.DataFrame:
        df = self._read_input_file(file_path)
        return df[
            (df["num_workers"] == self._replica_config.attn_tensor_parallel_size)
            & (df["devices_per_node"] == self._replica_config.attn_tensor_parallel_size)
            & (df["collective"] == "all_reduce")
        ]

    def _load_send_recv_df(self, file_path: str) -> pd.DataFrame:
        if self._is_multi_node:
            devices_per_node = 1
        else:
            devices_per_node = 2

        df = self._read_input_file(file_path)
        filtered_df = df[
            (df["collective"] == "send_recv")
            & (df["devices_per_node"] == devices_per_node)
        ]
        return filtered_df

    def _load_cpu_overhead_df(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            logger.warning(
                "CPU overhead profiling file not found: %s. "
                "CPU overhead model training will be skipped.",
                file_path,
            )
            logger.warning(
                "No CPU overhead profiling rows found for model_name='%s', "
                "tensor_parallel_degree=%s in file '%s'. "
                "CPU overhead predictions will default to 0 in this run.",
                self._model_config.get_name(),
                self._replica_config.attn_tensor_parallel_size,
                file_path,
            )
            return pd.DataFrame()

        df = self._read_input_file(file_path)
        if df.empty:
            logger.warning(
                "CPU overhead profiling file is empty: %s. "
                "CPU overhead model training will be skipped.",
                file_path,
            )
            logger.warning(
                "No CPU overhead profiling rows found for model_name='%s', "
                "tensor_parallel_degree=%s in file '%s'. "
                "CPU overhead predictions will default to 0 in this run.",
                self._model_config.get_name(),
                self._replica_config.attn_tensor_parallel_size,
                file_path,
            )
            return pd.DataFrame()

        df = apply_cpu_overhead_schema_v2_defaults(
            df,
            warn_fn=logger.warning,
            context=file_path,
        )
        df = validate_cpu_overhead_dataframe(df)

        filtered_df = df[
            (df["model_name"] == self._model_config.get_name())
            & (
                df["tensor_parallel_degree"]
                == self._replica_config.attn_tensor_parallel_size
            )
        ]

        if filtered_df.empty:
            logger.warning(
                "No CPU overhead profiling rows found for model_name='%s', "
                "tensor_parallel_degree=%s in file '%s'. "
                "CPU overhead predictions will default to 0 in this run.",
                self._model_config.get_name(),
                self._replica_config.attn_tensor_parallel_size,
                file_path,
            )

        return filtered_df

    def _load_pp_stage_boundary_df(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            logger.warning(
                "PP stage-boundary overhead input file does not exist: %s. "
                "PP handoff overhead predictions will default to 0 in this run.",
                file_path,
            )
            return pd.DataFrame()

        df = self._read_input_file(file_path)
        if df.empty:
            logger.warning(
                "PP stage-boundary overhead input file is empty: %s. "
                "PP handoff overhead predictions will default to 0 in this run.",
                file_path,
            )
            return pd.DataFrame()

        df = validate_pp_stage_boundary_dataframe(df)
        filtered_df = df[
            (df["model_name"] == self._model_config.get_name())
            & (
                df["tensor_parallel_degree"]
                == self._replica_config.attn_tensor_parallel_size
            )
            & (df["pp_world_size"] == self._replica_config.num_pipeline_stages)
        ]

        if filtered_df.empty:
            logger.warning(
                "No PP stage-boundary overhead rows found for model_name='%s', "
                "tensor_parallel_degree=%s, pp_world_size=%s in file '%s'. "
                "PP handoff overhead predictions will default to 0 in this run.",
                self._model_config.get_name(),
                self._replica_config.attn_tensor_parallel_size,
                self._replica_config.num_pipeline_stages,
                file_path,
            )

        return filtered_df

    def _build_pp_stage_boundary_lookup(
        self,
        df: pd.DataFrame,
    ) -> Dict[Tuple[int, int, int, int, int, int], float]:
        if df.empty:
            return {}

        lookup: Dict[Tuple[int, int, int, int, int, int], float] = {}
        for row in df.to_dict("records"):
            key = (
                int(row["batch_size"]),
                int(row["num_prefill_tokens"]),
                int(row["num_decode_tokens"]),
                int(row["producer_pp_rank"]),
                int(row["consumer_pp_rank"]),
                int(row["pp_world_size"]),
            )
            lookup[key] = float(row["pp_stage_boundary_overhead_ms"])
        return lookup

    def _build_pp_overhead_profile_rows(
        self,
        df: pd.DataFrame,
        *,
        identity_columns: Tuple[str, str, str, str, str, str],
        value_column: str,
    ) -> List[Tuple[Tuple[int, int, int, int, int, int], float]]:
        if df.empty:
            return []

        rows: List[Tuple[Tuple[int, int, int, int, int, int], float]] = []
        for row in df.to_dict("records"):
            key = tuple(int(row[column]) for column in identity_columns)
            rows.append((key, float(row[value_column])))  # type: ignore[arg-type]
        return rows

    def _predict_pp_overhead_from_nearest_profile_row(
        self,
        lookup_key: Tuple[int, int, int, int, int, int],
        profile_rows: List[Tuple[Tuple[int, int, int, int, int, int], float]],
        *,
        static_feature_indexes: Tuple[int, ...],
    ) -> Optional[float]:
        if not profile_rows:
            return None

        candidates = [
            (profile_key, value)
            for profile_key, value in profile_rows
            if all(profile_key[index] == lookup_key[index] for index in static_feature_indexes)
        ]
        if not candidates:
            return None

        dynamic_indexes = [
            index for index in range(len(lookup_key)) if index not in static_feature_indexes
        ]
        feature_ranges: Dict[int, float] = {}
        for index in dynamic_indexes:
            values = [profile_key[index] for profile_key, _ in candidates]
            values.append(lookup_key[index])
            span = max(values) - min(values)
            feature_ranges[index] = float(span if span > 0 else 1)

        def _distance(profile_key: Tuple[int, int, int, int, int, int]) -> float:
            return sum(
                abs(profile_key[index] - lookup_key[index]) / feature_ranges[index]
                for index in dynamic_indexes
            )

        nearest_key, nearest_value = min(candidates, key=lambda item: _distance(item[0]))
        return nearest_value

    def _initialize_pp_stage_boundary_lookup(self) -> None:
        if self._replica_config.num_pipeline_stages <= 1:
            self._pp_stage_boundary_lookup = {}
            self._pp_stage_boundary_profile_rows = []
            return

        df = self._load_pp_stage_boundary_df(self._pp_stage_boundary_input_file)
        self._pp_stage_boundary_lookup = self._build_pp_stage_boundary_lookup(df)
        self._pp_stage_boundary_profile_rows = self._build_pp_overhead_profile_rows(
            df,
            identity_columns=(
                "batch_size",
                "num_prefill_tokens",
                "num_decode_tokens",
                "producer_pp_rank",
                "consumer_pp_rank",
                "pp_world_size",
            ),
            value_column="pp_stage_boundary_overhead_ms",
        )

    def _load_pp_receiver_head_df(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            logger.warning(
                "PP receiver-head input file does not exist: %s. "
                "PP receiver-head predictions will default to 0 in this run.",
                file_path,
            )
            return pd.DataFrame()

        df = self._read_input_file(file_path)
        if df.empty:
            logger.warning(
                "PP receiver-head input file is empty: %s. "
                "PP receiver-head predictions will default to 0 in this run.",
                file_path,
            )
            return pd.DataFrame()

        df = validate_pp_receiver_head_dataframe(df)
        filtered_df = df[
            (df["model_name"] == self._model_config.get_name())
            & (
                df["tensor_parallel_degree"]
                == self._replica_config.attn_tensor_parallel_size
            )
            & (df["pp_world_size"] == self._replica_config.num_pipeline_stages)
            & (df["phase_label"] == "decode")
        ]

        if filtered_df.empty:
            logger.warning(
                "No PP receiver-head rows found for model_name='%s', "
                "tensor_parallel_degree=%s, pp_world_size=%s in file '%s'. "
                "PP receiver-head predictions will default to 0 in this run.",
                self._model_config.get_name(),
                self._replica_config.attn_tensor_parallel_size,
                self._replica_config.num_pipeline_stages,
                file_path,
            )

        return filtered_df

    def _build_pp_receiver_head_lookup(
        self,
        df: pd.DataFrame,
    ) -> Dict[Tuple[int, int, int, int, int, int], float]:
        if df.empty:
            return {}

        lookup: Dict[Tuple[int, int, int, int, int, int], float] = {}
        for row in df.to_dict("records"):
            key = (
                int(row["batch_size"]),
                int(row["num_prefill_tokens"]),
                int(row["num_decode_tokens"]),
                int(row["consumer_pp_rank"]),
                int(row["pp_world_size"]),
                int(row["activation_bytes_per_rank"]),
            )
            lookup[key] = float(row["pp_receiver_head_runtime_ms"])
        return lookup

    def _initialize_pp_receiver_head_lookup(self) -> None:
        if self._replica_config.num_pipeline_stages <= 1:
            self._pp_receiver_head_lookup = {}
            return

        df = self._load_pp_receiver_head_df(self._pp_receiver_head_input_file)
        self._pp_receiver_head_lookup = self._build_pp_receiver_head_lookup(df)

    def _load_pp_producer_send_path_df(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            logger.warning(
                "PP producer send-path input file does not exist: %s. "
                "PP producer send-path predictions will default to 0 in this run.",
                file_path,
            )
            return pd.DataFrame()

        df = self._read_input_file(file_path)
        if df.empty:
            logger.warning(
                "PP producer send-path input file is empty: %s. "
                "PP producer send-path predictions will default to 0 in this run.",
                file_path,
            )
            return pd.DataFrame()

        df = validate_pp_producer_send_path_dataframe(df)
        filtered_df = df[
            (df["model_name"] == self._model_config.get_name())
            & (
                df["tensor_parallel_degree"]
                == self._replica_config.attn_tensor_parallel_size
            )
            & (df["pp_world_size"] == self._replica_config.num_pipeline_stages)
            & (df["phase_label"] == "prefill")
        ]

        if filtered_df.empty:
            logger.warning(
                "No PP producer send-path rows found for model_name='%s', "
                "tensor_parallel_degree=%s, pp_world_size=%s in file '%s'. "
                "PP producer send-path predictions will default to 0 in this run.",
                self._model_config.get_name(),
                self._replica_config.attn_tensor_parallel_size,
                self._replica_config.num_pipeline_stages,
                file_path,
            )

        return filtered_df

    def _build_pp_producer_send_path_lookup(
        self,
        df: pd.DataFrame,
    ) -> Dict[Tuple[int, int, int, int, int, int], float]:
        if df.empty:
            return {}

        lookup: Dict[Tuple[int, int, int, int, int, int], float] = {}
        for row in df.to_dict("records"):
            key = (
                int(row["batch_size"]),
                int(row["num_prefill_tokens"]),
                int(row["num_decode_tokens"]),
                int(row["producer_pp_rank"]),
                int(row["pp_world_size"]),
                int(row["activation_bytes_per_rank"]),
            )
            lookup[key] = float(row["pp_producer_send_path_runtime_ms"])
        return lookup

    def _initialize_pp_producer_send_path_lookup(self) -> None:
        if self._replica_config.num_pipeline_stages <= 1:
            self._pp_producer_send_path_lookup = {}
            self._pp_producer_send_path_profile_rows = []
            return

        df = self._load_pp_producer_send_path_df(
            self._pp_producer_send_path_input_file
        )
        self._pp_producer_send_path_lookup = (
            self._build_pp_producer_send_path_lookup(df)
        )
        self._pp_producer_send_path_profile_rows = (
            self._build_pp_overhead_profile_rows(
                df,
                identity_columns=(
                    "batch_size",
                    "num_prefill_tokens",
                    "num_decode_tokens",
                    "producer_pp_rank",
                    "pp_world_size",
                    "activation_bytes_per_rank",
                ),
                value_column="pp_producer_send_path_runtime_ms",
            )
        )

    def _load_pp_prefill_consumer_active_df(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            logger.warning(
                "PP prefill consumer-active input file does not exist: %s. "
                "PP prefill consumer-active predictions will default to 0 in this run.",
                file_path,
            )
            return pd.DataFrame()

        df = self._read_input_file(file_path)
        if df.empty:
            logger.warning(
                "PP prefill consumer-active input file is empty: %s. "
                "PP prefill consumer-active predictions will default to 0 in this run.",
                file_path,
            )
            return pd.DataFrame()

        df = validate_pp_receiver_head_dataframe(df)
        is_prefill_consumer_active = df["phase_label"] == "prefill"
        is_mtp_lookahead_consumer_active = (
            df["other_overhead_source"]
            == "vllm_mtp_lookahead_consumer_active_replay"
        )
        filtered_df = df[
            (df["model_name"] == self._model_config.get_name())
            & (
                df["tensor_parallel_degree"]
                == self._replica_config.attn_tensor_parallel_size
            )
            & (df["pp_world_size"] == self._replica_config.num_pipeline_stages)
            & (is_prefill_consumer_active | is_mtp_lookahead_consumer_active)
        ]

        if filtered_df.empty:
            logger.warning(
                "No PP prefill consumer-active rows found for model_name='%s', "
                "tensor_parallel_degree=%s, pp_world_size=%s in file '%s'. "
                "PP prefill consumer-active predictions will default to 0 in this run.",
                self._model_config.get_name(),
                self._replica_config.attn_tensor_parallel_size,
                self._replica_config.num_pipeline_stages,
                file_path,
            )

        return filtered_df

    def _build_pp_prefill_consumer_active_lookup(
        self,
        df: pd.DataFrame,
    ) -> Dict[Tuple[int, int, int, int, int, int], float]:
        if df.empty:
            return {}

        lookup: Dict[Tuple[int, int, int, int, int, int], float] = {}
        for row in df.to_dict("records"):
            key = (
                int(row["batch_size"]),
                int(row["num_prefill_tokens"]),
                int(row["num_decode_tokens"]),
                int(row["consumer_pp_rank"]),
                int(row["pp_world_size"]),
                int(row["activation_bytes_per_rank"]),
            )
            lookup[key] = float(row["pp_receiver_head_runtime_ms"])
        return lookup

    def _initialize_pp_prefill_consumer_active_lookup(self) -> None:
        if self._replica_config.num_pipeline_stages <= 1:
            self._pp_prefill_consumer_active_lookup = {}
            self._pp_prefill_consumer_active_profile_rows = []
            return

        df = self._load_pp_prefill_consumer_active_df(
            self._pp_prefill_consumer_active_input_file
        )
        self._pp_prefill_consumer_active_lookup = (
            self._build_pp_prefill_consumer_active_lookup(df)
        )
        self._pp_prefill_consumer_active_profile_rows = (
            self._build_pp_overhead_profile_rows(
                df,
                identity_columns=(
                    "batch_size",
                    "num_prefill_tokens",
                    "num_decode_tokens",
                    "consumer_pp_rank",
                    "pp_world_size",
                    "activation_bytes_per_rank",
                ),
                value_column="pp_receiver_head_runtime_ms",
            )
        )

    def _get_pp_receiver_head_payload_tensor_multiplier(self) -> int:
        model_type = str(getattr(self._model_config, "model_type", "")).strip().lower()
        if model_type in {"llama", "qwen3_moe"}:
            # vLLM PP traces for dense Llama-family models log the full
            # IntermediateTensors payload, which carries both hidden_states
            # and residual on non-first PP consumer stages. Qwen3-MoE PP
            # traces use the same tensor-dict-total payload contract.
            return 2

        model_name_getter = getattr(self._model_config, "get_name", None)
        model_name = ""
        if callable(model_name_getter):
            model_name = str(model_name_getter()).strip().lower()
        if "llama" in model_name or "qwen3-a3b-30b-moe" in model_name:
            return 2

        return 1

    def _get_pp_producer_send_path_payload_tensor_multiplier(self) -> int:
        return self._get_pp_receiver_head_payload_tensor_multiplier()

    def _get_pp_stage_boundary_features(
        self, batch: Batch, stage_id: int
    ) -> Tuple[int, int, int, int, int, int]:
        producer_pp_rank = int(stage_id)
        consumer_pp_rank = producer_pp_rank + 1
        num_prefill_tokens, num_decode_tokens = (
            self._get_pp_stage_boundary_semantic_token_counts(batch)
        )
        return (
            int(getattr(batch, "size", 0)),
            num_prefill_tokens,
            num_decode_tokens,
            producer_pp_rank,
            consumer_pp_rank,
            int(getattr(self._replica_config, "num_pipeline_stages", 1)),
        )

    def _get_pp_stage_boundary_semantic_token_counts(
        self, batch: Batch
    ) -> Tuple[int, int]:
        num_prefill_tokens = int(getattr(batch, "num_prefill_tokens", 0))
        num_decode_tokens = int(getattr(batch, "num_decode_tokens", 0))
        if not self._is_spec_lookahead_verification_batch(batch):
            return num_prefill_tokens, num_decode_tokens
        if num_decode_tokens <= 0:
            return num_prefill_tokens, num_decode_tokens

        decode_sequence_count = 0
        for request in getattr(batch, "requests", []):
            if bool(getattr(request, "is_prefill_complete", False)):
                decode_sequence_count += 1
        if decode_sequence_count <= 0:
            decode_sequence_count = int(getattr(batch, "size", 0))

        return num_prefill_tokens, decode_sequence_count

    def _get_pp_receiver_head_runtime_features(
        self, batch: Batch, stage_id: int
    ) -> Tuple[int, int, int, int, int, int]:
        quant_manager = get_quantization_manager()
        effective_tokens_transfer = batch.get_effective_total_tokens_for_transfer(
            self._cluster_type
        )
        dtype_bytes = quant_manager.get_bytes_per_element(
            "send_recv",
            self._cluster_type,
        )
        activation_bytes_per_rank = math.ceil(
            self._model_config.embedding_dim
            * dtype_bytes
            * effective_tokens_transfer
        ) * self._get_pp_receiver_head_payload_tensor_multiplier()
        return (
            int(getattr(batch, "size", 0)),
            int(getattr(batch, "num_prefill_tokens", 0)),
            int(getattr(batch, "num_decode_tokens", 0)),
            int(stage_id),
            int(getattr(self._replica_config, "num_pipeline_stages", 1)),
            int(activation_bytes_per_rank),
        )

    def _is_spec_lookahead_verification_batch(self, batch: Batch) -> bool:
        metadata = getattr(batch, "spec_decode_metadata", None)
        if metadata is None:
            return False
        if not bool(getattr(metadata, "uses_lookahead_slots", False)):
            return False
        return int(getattr(batch, "num_decode_tokens", 0)) > 0

    def _get_pp_prefill_consumer_active_runtime_features(
        self, batch: Batch, stage_id: int
    ) -> Tuple[int, int, int, int, int, int]:
        if not self._is_spec_lookahead_verification_batch(batch):
            return self._get_pp_receiver_head_runtime_features(batch, stage_id)

        quant_manager = get_quantization_manager()
        effective_tokens_transfer = batch.get_effective_total_tokens_for_transfer(
            self._cluster_type
        )
        dtype_bytes = quant_manager.get_bytes_per_element(
            "send_recv",
            self._cluster_type,
        )
        activation_bytes_per_rank = math.ceil(
            self._model_config.embedding_dim
            * dtype_bytes
            * effective_tokens_transfer
        ) * self._get_pp_receiver_head_payload_tensor_multiplier()
        semantic_prefill_tokens, semantic_decode_tokens = (
            self._get_pp_stage_boundary_semantic_token_counts(batch)
        )
        return (
            int(getattr(batch, "size", 0)),
            int(semantic_prefill_tokens),
            int(semantic_decode_tokens),
            int(stage_id),
            int(getattr(self._replica_config, "num_pipeline_stages", 1)),
            int(activation_bytes_per_rank),
        )

    def _get_pp_producer_send_path_runtime_features(
        self, batch: Batch, stage_id: int
    ) -> Tuple[int, int, int, int, int, int]:
        quant_manager = get_quantization_manager()
        effective_tokens_transfer = batch.get_effective_total_tokens_for_transfer(
            self._cluster_type
        )
        dtype_bytes = quant_manager.get_bytes_per_element(
            "send_recv",
            self._cluster_type,
        )
        activation_bytes_per_rank = math.ceil(
            self._model_config.embedding_dim
            * dtype_bytes
            * effective_tokens_transfer
        ) * self._get_pp_producer_send_path_payload_tensor_multiplier()
        return (
            int(getattr(batch, "size", 0)),
            int(getattr(batch, "num_prefill_tokens", 0)),
            int(getattr(batch, "num_decode_tokens", 0)),
            int(stage_id),
            int(getattr(self._replica_config, "num_pipeline_stages", 1)),
            int(activation_bytes_per_rank),
        )

    def _log_missing_pp_stage_boundary_prediction_once(
        self, lookup_key: Tuple[int, int, int, int, int, int]
    ) -> None:
        missing_cache = getattr(
            self, "_missing_pp_stage_boundary_predictions_logged", None
        )
        if missing_cache is None:
            missing_cache = getattr(
                self, "_missing_pp_stage_boundary_metrics_logged", None
            )
        if missing_cache is None:
            missing_cache = set()
            self._missing_pp_stage_boundary_predictions_logged = missing_cache
            self._missing_pp_stage_boundary_metrics_logged = missing_cache

        cache_key = str(lookup_key)
        if cache_key in missing_cache:
            return

        logger.warning(
            "PP stage-boundary overhead prediction is unavailable. Falling back to 0. "
            "pp_stage_boundary_input_file=%s, lookup_key=%s",
            getattr(self, "_pp_stage_boundary_input_file", ""),
            lookup_key,
        )
        missing_cache.add(cache_key)

    def _log_missing_pp_receiver_head_prediction_once(
        self, lookup_key: Tuple[int, int, int, int, int, int]
    ) -> None:
        cache_key = str(lookup_key)
        if cache_key in self._missing_pp_receiver_head_predictions_logged:
            return

        logger.warning(
            "PP receiver-head runtime prediction is unavailable. Falling back to 0. "
            "pp_receiver_head_input_file=%s, lookup_key=%s",
            getattr(self, "_pp_receiver_head_input_file", ""),
            lookup_key,
        )
        self._missing_pp_receiver_head_predictions_logged.add(cache_key)

    def _log_missing_pp_producer_send_path_prediction_once(
        self, lookup_key: Tuple[int, int, int, int, int, int]
    ) -> None:
        cache_key = str(lookup_key)
        if cache_key in self._missing_pp_producer_send_path_predictions_logged:
            return

        logger.warning(
            "PP producer send-path runtime prediction is unavailable. Falling back to 0. "
            "pp_producer_send_path_input_file=%s, lookup_key=%s",
            getattr(self, "_pp_producer_send_path_input_file", ""),
            lookup_key,
        )
        self._missing_pp_producer_send_path_predictions_logged.add(cache_key)

    def _log_missing_pp_prefill_consumer_active_prediction_once(
        self, lookup_key: Tuple[int, int, int, int, int, int]
    ) -> None:
        missing_cache = getattr(
            self, "_missing_pp_prefill_consumer_active_predictions_logged", None
        )
        if missing_cache is None:
            missing_cache = set()
            self._missing_pp_prefill_consumer_active_predictions_logged = (
                missing_cache
            )

        cache_key = str(lookup_key)
        if cache_key in missing_cache:
            return

        logger.warning(
            "PP prefill consumer-active runtime prediction is unavailable. Falling back to 0. "
            "pp_prefill_consumer_active_input_file=%s, lookup_key=%s",
            getattr(self, "_pp_prefill_consumer_active_input_file", ""),
            lookup_key,
        )
        missing_cache.add(cache_key)

    def _get_pp_stage_boundary_handoff_time(
        self, batch: Batch, stage_id: int
    ) -> float:
        pp_world_size = int(getattr(self._replica_config, "num_pipeline_stages", 1))
        if pp_world_size <= 1:
            return 0.0
        if stage_id < 0 or stage_id >= pp_world_size - 1:
            return 0.0

        lookup_key = self._get_pp_stage_boundary_features(batch, stage_id)
        lookup = getattr(self, "_pp_stage_boundary_lookup", {})
        if lookup_key in lookup:
            return float(lookup[lookup_key])

        nearest_value = self._predict_pp_overhead_from_nearest_profile_row(
            lookup_key,
            getattr(self, "_pp_stage_boundary_profile_rows", []),
            static_feature_indexes=(3, 4, 5),
        )
        if nearest_value is not None:
            return float(nearest_value)

        self._log_missing_pp_stage_boundary_prediction_once(lookup_key)
        return 0.0

    def _get_pp_receiver_head_runtime_time(
        self, batch: Batch, stage_id: int
    ) -> float:
        pp_world_size = int(getattr(self._replica_config, "num_pipeline_stages", 1))
        if pp_world_size <= 1:
            return 0.0
        if stage_id <= 0 or stage_id >= pp_world_size:
            return 0.0
        if self._is_spec_lookahead_verification_batch(batch):
            return 0.0
        if int(getattr(batch, "num_decode_tokens", 0)) <= 0:
            return 0.0
        if int(getattr(batch, "num_prefill_tokens", 0)) != 0:
            return 0.0

        lookup_key = self._get_pp_receiver_head_runtime_features(batch, stage_id)
        if lookup_key in self._pp_receiver_head_lookup:
            return float(self._pp_receiver_head_lookup[lookup_key])

        self._log_missing_pp_receiver_head_prediction_once(lookup_key)
        return 0.0

    def _get_pp_producer_send_path_runtime_time(
        self, batch: Batch, stage_id: int
    ) -> float:
        pp_world_size = int(getattr(self._replica_config, "num_pipeline_stages", 1))
        if pp_world_size <= 1:
            return 0.0
        if stage_id < 0 or stage_id >= pp_world_size - 1:
            return 0.0
        if int(getattr(batch, "num_prefill_tokens", 0)) <= 0:
            return 0.0
        if int(getattr(batch, "num_decode_tokens", 0)) != 0:
            return 0.0

        lookup_key = self._get_pp_producer_send_path_runtime_features(batch, stage_id)
        lookup = getattr(self, "_pp_producer_send_path_lookup", {})
        if lookup_key in lookup:
            return float(lookup[lookup_key])

        nearest_value = self._predict_pp_overhead_from_nearest_profile_row(
            lookup_key,
            getattr(self, "_pp_producer_send_path_profile_rows", []),
            static_feature_indexes=(3, 4),
        )
        if nearest_value is not None:
            return float(nearest_value)

        self._log_missing_pp_producer_send_path_prediction_once(lookup_key)
        return 0.0

    def _get_pp_prefill_consumer_active_runtime_time(
        self, batch: Batch, stage_id: int
    ) -> float:
        pp_world_size = int(getattr(self._replica_config, "num_pipeline_stages", 1))
        if pp_world_size <= 1:
            return 0.0
        if stage_id <= 0 or stage_id >= pp_world_size:
            return 0.0
        is_spec_lookahead_verification = self._is_spec_lookahead_verification_batch(
            batch
        )
        if (
            int(getattr(batch, "num_prefill_tokens", 0)) <= 0
            and not is_spec_lookahead_verification
        ):
            return 0.0
        if (
            int(getattr(batch, "num_decode_tokens", 0)) != 0
            and not is_spec_lookahead_verification
        ):
            return 0.0

        lookup_key = self._get_pp_prefill_consumer_active_runtime_features(
            batch, stage_id
        )
        lookup = getattr(self, "_pp_prefill_consumer_active_lookup", {})
        if lookup_key in lookup:
            return float(lookup[lookup_key])

        nearest_value = self._predict_pp_overhead_from_nearest_profile_row(
            lookup_key,
            getattr(self, "_pp_prefill_consumer_active_profile_rows", []),
            static_feature_indexes=(3, 4),
        )
        if nearest_value is not None:
            return float(nearest_value)

        self._log_missing_pp_prefill_consumer_active_prediction_once(lookup_key)
        return 0.0

    def _read_input_file(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df = df.drop_duplicates()
        return df

    def _get_profiling_metadata(
        self, df: pd.DataFrame, file_path: str
    ) -> ProfilingMetadata:
        """Validate profiling metadata columns and return the parsed metadata."""
        # Check for profiling_precision column
        if "profiling_precision" not in df.columns:
            raise ValueError(
                f"profiling_precision column is missing from '{file_path}'. "
                f"Run '{MIGRATION_HELP_COMMAND}' to add required metadata columns."
            )

        precision_values = df["profiling_precision"].dropna().unique().tolist()
        if not precision_values:
            raise ValueError(f"profiling_precision column is empty in '{file_path}'")
        if len(precision_values) > 1:
            raise ValueError(
                f"Multiple profiling_precision values found in '{file_path}': {precision_values}. "
                "Profiling data should have consistent precision."
            )

        # Check for model_arch column (legacy metadata retained for diagnostics)
        if "model_arch" not in df.columns:
            raise ValueError(
                f"model_arch column is missing from '{file_path}'. "
                f"Run '{MIGRATION_HELP_COMMAND}' to add required metadata columns."
            )

        arch_values = df["model_arch"].dropna().unique().tolist()
        if not arch_values:
            raise ValueError(f"model_arch column is empty in '{file_path}'")
        if len(arch_values) > 1:
            raise ValueError(
                f"Multiple model_arch values found in '{file_path}': {arch_values}. "
                "Profiling data should have consistent architecture."
            )

        # Validate model_arch matches model config
        expected_arch = getattr(self._model_config, "model_arch", "generic")
        actual_arch = str(arch_values[0])
        if actual_arch != expected_arch:
            raise ValueError(
                f"model_arch mismatch: expected '{expected_arch}' but profiling data has '{actual_arch}'. "
                f"File: '{file_path}'"
            )

        if "model_architecture_profile" not in df.columns:
            raise ValueError(
                f"model_architecture_profile column is missing from '{file_path}'. "
                f"Run '{MIGRATION_HELP_COMMAND}' to add required metadata columns."
            )

        profile_values = df["model_architecture_profile"].dropna().unique().tolist()
        if not profile_values:
            raise ValueError(
                f"model_architecture_profile column is empty in '{file_path}'"
            )
        if len(profile_values) > 1:
            raise ValueError(
                "Multiple model_architecture_profile values found in "
                f"'{file_path}': {profile_values}. Profiling data should have "
                "consistent architecture profile."
            )

        expected_profile = self._get_model_architecture_profile().profile_id
        actual_profile = str(profile_values[0])
        if actual_profile != expected_profile:
            raise ValueError(
                "model_architecture_profile mismatch: expected "
                f"'{expected_profile}' but profiling data has '{actual_profile}'. "
                f"File: '{file_path}'"
            )

        profiling_precision = PrecisionType.from_string(precision_values[0])

        # Check for quant_signature column
        if "quant_signature" not in df.columns:
            raise ValueError(
                f"quant_signature column is missing from '{file_path}'. "
                f"Run '{MIGRATION_HELP_COMMAND}' to add required metadata columns."
            )

        quant_values = df["quant_signature"].dropna().unique().tolist()
        if not quant_values:
            raise ValueError(f"quant_signature column is empty in '{file_path}'")
        if len(quant_values) > 1:
            raise ValueError(
                f"Multiple quant_signature values found in '{file_path}': {quant_values}. "
                "Profiling data should have consistent quantization configuration."
            )

        # Log quant_signature mismatch (do not fail-fast; approximation may apply)
        expected_quant = getattr(self._model_config, "get_quant_signature", lambda: "none")()
        actual_quant = str(quant_values[0])
        if actual_quant != expected_quant:
            logger.warning(
                "quant_signature mismatch: expected '%s' but profiling data has '%s'. File: '%s'",
                expected_quant,
                actual_quant,
                file_path,
            )

        if "measurement_type" not in df.columns:
            raise ValueError(
                f"measurement_type column is missing from '{file_path}'. "
                f"Run '{MIGRATION_HELP_COMMAND}' to add required metadata columns."
            )

        measurement_values = df["measurement_type"].dropna().unique().tolist()
        if not measurement_values:
            raise ValueError(f"measurement_type column is empty in '{file_path}'")
        if len(measurement_values) > 1:
            raise ValueError(
                f"Multiple measurement_type values found in '{file_path}': {measurement_values}. "
                "Profiling data should have consistent measurement semantics."
            )

        measurement_type = MeasurementType.from_string(measurement_values[0])

        return ProfilingMetadata(
            profiling_precision=profiling_precision,
            quant_signature=actual_quant,
            model_arch=actual_arch,
            model_architecture_profile=actual_profile,
            measurement_type=measurement_type,
        )

    def _validate_active_measurement_type(
        self, metadata: ProfilingMetadata, file_path: str
    ) -> None:
        expected_measurement_type = getattr(self, "_active_measurement_type", None)
        if expected_measurement_type is None:
            return
        if metadata.measurement_type != expected_measurement_type:
            raise ValueError(
                f"measurement_type mismatch for '{file_path}': expected "
                f"{expected_measurement_type.value} but found {metadata.measurement_type.value}."
            )

    def _register_profiling_metadata_for_ops(
        self,
        operation_names: List[str],
        metadata: ProfilingMetadata,
        file_path: str,
    ) -> None:
        quant_manager = get_quantization_manager()
        expected_quant = getattr(self._model_config, "get_quant_signature", lambda: "none")()
        normalized_operation_names = [
            get_moe_gating_base_model_name(operation_name)
            for operation_name in operation_names
        ]
        normalized_operation_names = list(dict.fromkeys(normalized_operation_names))
        quant_manager.register_profiling_metadata(
            operation_names=normalized_operation_names,
            profiling_precision=metadata.profiling_precision,
            profiling_quant_signature=metadata.quant_signature,
            expected_quant_signature=expected_quant,
            file_path=file_path,
        )

    def _register_missing_profiling_metadata(
        self, operation_names: List[str], file_path: str
    ) -> None:
        quant_manager = get_quantization_manager()
        expected_quant = getattr(self._model_config, "get_quant_signature", lambda: "none")()
        default_precision = self._model_config.get_default_precision()
        quant_manager.register_profiling_metadata(
            operation_names=operation_names,
            profiling_precision=default_precision,
            profiling_quant_signature="missing",
            expected_quant_signature=expected_quant,
            file_path=file_path,
        )

    def _register_profiling_metadata_from_file(
        self, file_path: str, operation_names: List[str]
    ) -> None:
        if not file_path or not os.path.exists(file_path):
            logger.warning("Profiling file missing: %s", file_path)
            self._register_missing_profiling_metadata(operation_names, file_path)
            return
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            logger.warning("Failed to read profiling file %s: %s", file_path, exc)
            self._register_missing_profiling_metadata(operation_names, file_path)
            return
        metadata = self._get_profiling_metadata(df, file_path)
        self._validate_active_measurement_type(metadata, file_path)
        self._register_profiling_metadata_for_ops(operation_names, metadata, file_path)

    def _register_additional_profiling_metadata_from_files(self) -> None:
        return

    def _register_profiling_metadata_from_files(self) -> None:
        self._register_profiling_metadata_from_file(
            self._compute_input_file, self._get_compute_model_names()
        )
        self._register_profiling_metadata_from_file(
            self._attention_input_file, self._get_attention_model_names()
        )
        self._register_additional_profiling_metadata_from_files()

    def _get_compute_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        return df_with_derived_features

    def _get_attention_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        if self._is_mla_attention_family():
            if "is_prefill" in df_with_derived_features.columns:
                df_with_derived_features["is_prefill"] = coerce_truthy_int(
                    df_with_derived_features["is_prefill"]
                )
            return df_with_derived_features

        df_with_derived_features["num_tokens"] = df_with_derived_features[
            ["prefill_chunk_size", "batch_size"]
        ].max(axis=1)

        if "is_prefill" in df_with_derived_features.columns:
            normalized_prefill_values = coerce_truthy_bool(
                df_with_derived_features["is_prefill"]
            )
            df_with_derived_features["is_decode"] = ~normalized_prefill_values
        else:
            df_with_derived_features["is_decode"] = (
                df_with_derived_features["prefill_chunk_size"] == 0
            )

        df_with_derived_features["prefill_chunk_size_squared"] = (
            df_with_derived_features["prefill_chunk_size"] ** 2
        )

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

        def _parse_numeric_list(value: Any, column_name: str) -> List[float]:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped == "" or stripped.lower() in {"nan", "none", "<na>"}:
                    return []
                try:
                    parsed_value = ast.literal_eval(stripped)
                except (SyntaxError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid true-mixed attention profile list column "
                        f"{column_name}: {value!r}"
                    ) from exc
            else:
                parsed_value = value

            if isinstance(parsed_value, pd.Series):
                parsed_value = parsed_value.tolist()
            elif isinstance(parsed_value, np.ndarray):
                parsed_value = parsed_value.tolist()

            if isinstance(parsed_value, (list, tuple)):
                parsed_list = []
                for item in parsed_value:
                    if pd.isna(item):
                        continue
                    parsed_list.append(float(item))
                return parsed_list

            if pd.isna(parsed_value):
                return []
            return [float(parsed_value)]

        def _round_kv_cache_size(avg_kv_cache_size: int) -> int:
            granularity = self._config.kv_cache_prediction_granularity
            return (
                (avg_kv_cache_size + granularity - 1) // granularity
            ) * granularity

        if {
            "prefill_seq_lens",
            "prefill_kv_cache_sizes",
        }.issubset(df_with_derived_features.columns):
            true_mixed_mask = df_with_derived_features["is_true_mixed_batch"]
            for row_idx, row in df_with_derived_features.loc[true_mixed_mask].iterrows():
                prefill_seq_lens = _parse_numeric_list(
                    row["prefill_seq_lens"],
                    "prefill_seq_lens",
                )
                prefill_kv_cache_sizes = _parse_numeric_list(
                    row["prefill_kv_cache_sizes"],
                    "prefill_kv_cache_sizes",
                )
                if not prefill_seq_lens:
                    raise ValueError(
                        "True-mixed attention profile row is missing "
                        f"prefill_seq_lens at index {row_idx}."
                    )
                if len(prefill_seq_lens) != len(prefill_kv_cache_sizes):
                    raise ValueError(
                        "True-mixed attention profile row has mismatched "
                        f"prefill_seq_lens/prefill_kv_cache_sizes lengths "
                        f"at index {row_idx}: {len(prefill_seq_lens)} vs "
                        f"{len(prefill_kv_cache_sizes)}."
                    )

                seq_lens_arr = np.array(prefill_seq_lens, dtype=np.float64)
                batch_size = len(prefill_seq_lens)
                avg_kv_cache_size = int(np.mean(prefill_kv_cache_sizes))
                avg_seq_len = float(seq_lens_arr.mean())
                seq_len_variance = (
                    float(seq_lens_arr.var()) if batch_size > 1 else 0.0
                )
                seq_len_std = float(seq_lens_arr.std())
                seq_len_cv = seq_len_std / avg_seq_len if avg_seq_len > 0 else 0.0
                min_seq_len = int(seq_lens_arr.min())
                max_seq_len = int(seq_lens_arr.max())
                total_tokens = int(seq_lens_arr.sum())

                df_with_derived_features.loc[
                    row_idx,
                    "prefill_mixed_batch_size",
                ] = batch_size
                df_with_derived_features.loc[
                    row_idx,
                    "prefill_mixed_kv_cache_size",
                ] = _round_kv_cache_size(avg_kv_cache_size)
                df_with_derived_features.loc[
                    row_idx,
                    "prefill_mixed_total_tokens",
                ] = total_tokens
                df_with_derived_features.loc[
                    row_idx,
                    "prefill_mixed_avg_seq_len",
                ] = avg_seq_len
                df_with_derived_features.loc[
                    row_idx,
                    "prefill_mixed_min_seq_len",
                ] = min_seq_len
                df_with_derived_features.loc[
                    row_idx,
                    "prefill_mixed_max_seq_len",
                ] = max_seq_len
                df_with_derived_features.loc[
                    row_idx,
                    "prefill_mixed_total_tokens_squared",
                ] = total_tokens**2
                df_with_derived_features.loc[
                    row_idx,
                    "prefill_mixed_seq_len_variance",
                ] = seq_len_variance
                df_with_derived_features.loc[
                    row_idx,
                    "prefill_mixed_seq_len_cv",
                ] = seq_len_cv
                df_with_derived_features.loc[
                    row_idx,
                    "prefill_mixed_seq_len_range",
                ] = max_seq_len - min_seq_len
                df_with_derived_features.loc[
                    row_idx,
                    "prefill_mixed_batch_variance_interaction",
                ] = batch_size * seq_len_variance
                df_with_derived_features.loc[
                    row_idx,
                    "prefill_mixed_batch_cv_interaction",
                ] = batch_size * seq_len_cv

        # Mixed-batch derived features for optional attn_prefill_mixed training.
        if "total_tokens" in df_with_derived_features.columns:
            df_with_derived_features["total_tokens_squared"] = (
                df_with_derived_features["total_tokens"] ** 2
            )
            if {
                "max_seq_len",
                "min_seq_len",
            }.issubset(df_with_derived_features.columns):
                df_with_derived_features["seq_len_range"] = (
                    df_with_derived_features["max_seq_len"]
                    - df_with_derived_features["min_seq_len"]
                )
            if {
                "batch_size",
                "seq_len_variance",
            }.issubset(df_with_derived_features.columns):
                df_with_derived_features["batch_variance_interaction"] = (
                    df_with_derived_features["batch_size"]
                    * df_with_derived_features["seq_len_variance"]
                )
            if {"batch_size", "seq_len_cv"}.issubset(df_with_derived_features.columns):
                df_with_derived_features["batch_cv_interaction"] = (
                    df_with_derived_features["batch_size"]
                    * df_with_derived_features["seq_len_cv"]
                )

        # True mixed (prefill+decode) decode model features.
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
            total_batch_size = df_with_derived_features["total_batch_size"].replace(0, np.nan)
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

    def _get_all_reduce_df_with_derived_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        # convert bytes to num tokens
        # each token is of size 2 * h bytes
        df_with_derived_features["num_tokens"] = (
            df_with_derived_features["size"] / self._model_config.embedding_dim / 2
        )
        return df_with_derived_features

    def _get_send_recv_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        df_with_derived_features["num_tokens"] = (
            df_with_derived_features["size"] / self._model_config.embedding_dim / 2
        )
        return df_with_derived_features

    def _get_cpu_overhead_df_with_derived_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        return df_with_derived_features

    def _requires_target_embedded_mtp_compute_models(self) -> bool:
        if bool(
            getattr(
                self._replica_config,
                "requires_mtp_structural_compute_models",
                False,
            )
        ):
            return True
        return is_target_embedded_mtp_enabled(
            getattr(self._replica_config, "speculative_decoding_config", None)
        )

    def _should_include_spec_decode_proposer_overhead(self, batch: Batch) -> bool:
        if bool(
            getattr(
                self._replica_config,
                "suppress_spec_decode_proposer_overhead",
                False,
            )
        ):
            return False
        return getattr(batch, "spec_decode_metadata", None) is not None

    def _requires_dense_mlp_compute_models(self) -> bool:
        if not getattr(self._model_config, "is_moe", False):
            return True

        get_num_moe_layers = getattr(self._model_config, "get_num_moe_layers", None)
        num_layers = getattr(self._model_config, "num_layers", None)
        if callable(get_num_moe_layers) and isinstance(num_layers, int):
            return int(get_num_moe_layers()) < int(num_layers)

        return True

    def _get_compute_model_names(self) -> List[str]:
        model_names = [
            "emb",
            "attn_pre_proj",
            "attn_post_proj",
            "input_layernorm",
            "post_attention_layernorm",
            "attn_rope",
        ]
        if self._requires_dense_mlp_compute_models():
            model_names.extend(get_family_profiling_names(FFN_FAMILY))
        if self._requires_target_embedded_mtp_compute_models():
            model_names.extend(["mtp_fusion_proj", "lm_head_linear"])

        # add is only a separate operation for non-fused LayerNorm models
        if not self._model_config.uses_fused_add_norm:
            model_names.append("add")

        model_names.extend(self._get_predictor_attention_extra_ops())

        # Shared-expert operations are required when the architecture profile exposes that path.
        if self._model_config.supports_share_expert() and self._model_config.is_moe:
            model_names.extend(get_family_profiling_names(SHARE_EXPERT_FAMILY))
        return model_names

    def _get_ffn_tp_key_for_linear_op(self) -> int:
        if (
            self._model_config.is_moe
            and self._cluster_type in {
                ClusterType.PREFILL,
                ClusterType.DECODE_FFN,
                ClusterType.DECODE,
                ClusterType.MONOLITHIC,
            }
        ):
            return self._replica_config.moe_tensor_parallel_size
        return self._replica_config.attn_tensor_parallel_size

    def _get_linear_op_tp_key(self, op_name: str) -> int:
        replicated_ops = set(get_family_profiling_name_set(MEMORY_FAMILY))
        replicated_ops.update(
            self._get_model_architecture_profile().linear_attention.replicated_ops
        )
        if op_name in replicated_ops:
            if (
                self._requires_target_embedded_mtp_compute_models()
                and is_target_embedded_mtp_same_tp_linear_op(op_name)
            ):
                return resolve_effective_attention_tp_size(
                    op_name="attn_pre_proj",
                    requested_tp_size=self._replica_config.attn_tensor_parallel_size,
                    num_kv_heads=self._model_config.num_kv_heads,
                    cluster_type=self._cluster_type,
                    warning_cache=getattr(self, "_attention_tp_warning_cache", None),
                    include_linear_ops=True,
                )
            return 1

        ffn_tp_key = self._get_ffn_tp_key_for_linear_op()

        if op_name in get_family_profiling_name_set(FFN_FAMILY):
            return ffn_tp_key

        if op_name in {
            "mtp_fusion_proj",
            "lm_head_linear",
        }:
            return resolve_effective_attention_tp_size(
                op_name="attn_pre_proj",
                requested_tp_size=self._replica_config.attn_tensor_parallel_size,
                num_kv_heads=self._model_config.num_kv_heads,
                cluster_type=self._cluster_type,
                warning_cache=getattr(self, "_attention_tp_warning_cache", None),
                include_linear_ops=True,
            )

        if op_name in get_family_profiling_name_set(SHARE_EXPERT_FAMILY):
            return ffn_tp_key

        if op_name.startswith("attn_"):
            return resolve_effective_attention_tp_size(
                op_name=op_name,
                requested_tp_size=self._replica_config.attn_tensor_parallel_size,
                num_kv_heads=self._model_config.num_kv_heads,
                cluster_type=self._cluster_type,
                warning_cache=getattr(self, "_attention_tp_warning_cache", None),
                include_linear_ops=True,
            )

        raise ValueError(f"Unsupported linear op for TP mapping: {op_name}")

    def _get_attention_model_names(self) -> List[str]:
        return list(get_enabled_predictor_metric_names(self._get_attention_family()))

    @staticmethod
    def mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Handling the case where y_true is 0 separately to avoid division by zero
        zero_true_mask = y_true == 0
        non_zero_true_mask = ~zero_true_mask

        # For non-zero true values, calculate the absolute percentage error
        error = np.zeros_like(y_true, dtype=float)  # using float instead of np.float
        error[non_zero_true_mask] = (
            np.abs(
                (y_true[non_zero_true_mask] - y_pred[non_zero_true_mask])
                / y_true[non_zero_true_mask]
            )
            * 100
        )

        # For zero true values, if prediction is also 0, error is 0, else it is 100
        error[zero_true_mask] = np.where(y_pred[zero_true_mask] == 0, 0, 100)

        # Return the mean of the absolute percentage errors
        return np.mean(error)

    def _get_scorer(self) -> Any:
        return make_scorer(
            SklearnExecutionTimePredictor.mean_absolute_percentage_error,
            greater_is_better=False,
        )

    @abstractmethod
    def _get_grid_search_params(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _get_estimator(self) -> BaseEstimator:
        pass

    def _get_model_hash(self, model_name: str, df: pd.DataFrame = None) -> str:
        config_str = str(self.to_dict())

        if df is None:
            combined_str = f"{config_str}_{model_name}_{self._active_measurement_type.value}"
        else:
            df_hash_str = hashlib.md5(df.to_json().encode("utf-8")).hexdigest()
            combined_str = f"{config_str}_{model_name}_{df_hash_str}_{self._active_measurement_type.value}"

        return hashlib.md5(combined_str.encode("utf-8")).hexdigest()[0:8]

    def _load_model_from_cache(self, model_name: str, model_hash: str) -> BaseEstimator:
        with InterProcessReaderWriterLock(
            f"{self._cache_dir}/{model_hash}_model_lock.file"
        ).read_lock():
            if self._config.no_cache:
                return
            # check if model is in cache
            cache_file = f"{self._cache_dir}/{model_name}_{model_hash}.pkl"
            if not os.path.exists(cache_file):
                return

            logger.debug(f"Found model {model_name} in cache")
            model = pickle.load(open(cache_file, "rb"))
            return model

    def _store_model_in_cache(
        self, model_name: str, model_hash: str, model: BaseEstimator
    ) -> None:
        with InterProcessReaderWriterLock(
            f"{self._cache_dir}/{model_hash}_model_lock.file"
        ).write_lock():
            # store model in cache
            cache_file = f"{self._cache_dir}/{model_name}_{model_hash}.pkl"
            pickle.dump(model, open(cache_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    def _store_training_prediction_data(
        self,
        model_name: str,
        model_hash: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        model: BaseEstimator,
    ) -> None:
        df = df.copy()

        # convert the df to list of tuples
        df["prediction"] = model.predict(df[feature_cols])

        # store the prediction data
        df[feature_cols + [target_col, "prediction"]].to_csv(
            f"{self._cache_dir}/{model_name}_{model_hash}_training_predictions.csv",
            index=False,
        )

    def _train_model(
        self,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> BaseEstimator:
        if len(df) == 0:
            raise Exception(f"Training data for model {model_name} is empty")

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

        model_hash = self._get_model_hash(model_name, df)

        cached_model = self._load_model_from_cache(model_name, model_hash)
        if cached_model:
            return cached_model

        model = self._get_estimator()
        grid_search_params = self._get_grid_search_params()

        if len(df) < self._config.k_fold_cv_splits:
            cv = 2
        else:
            cv = self._config.k_fold_cv_splits

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid_search_params,
            scoring=self._get_scorer(),
            cv=cv,
            n_jobs=self._config.num_training_job_threads,
        )

        # we don't create a train/test split, because we want to use all data for training
        # and we don't care about overfitting, because we only want to predict execution time within the same domain
        X, y = df[feature_cols], df[target_col]

        grid_search.fit(X, y)
        score = grid_search.score(X, y)

        logger.info(
            f"Trained model {model_name} and found best parameters: {grid_search.best_params_} "
            f"with mean absolute percentage error (MEAP) {-score}%"
        )

        best_estimator = grid_search.best_estimator_
        # Attach model identity so prediction cache can stay in sync with the actual estimator.
        setattr(best_estimator, "_frontier_model_hash", model_hash)
        setattr(best_estimator, "_frontier_feature_names", list(feature_cols))

        self._store_model_in_cache(model_name, model_hash, best_estimator)

        self._store_training_prediction_data(
            model_name=model_name,
            model_hash=model_hash,
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            model=best_estimator,
        )
        return best_estimator

    def _store_model_predication_cache(
        self, model_name: str, prediction_hash: str, predictions: Dict[Tuple, float]
    ) -> None:
        with InterProcessReaderWriterLock(
            f"{self._cache_dir}/{prediction_hash}_prediction_lock.file"
        ).write_lock():
            cache_file = (
                f"{self._cache_dir}/{model_name}_{prediction_hash}_predictions.pkl"
            )
            pickle.dump(
                predictions, open(cache_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL
            )

    def _load_model_predication_cache(
        self, model_name: str, prediction_hash: str
    ) -> Optional[Dict[Tuple, float]]:
        with InterProcessReaderWriterLock(
            f"{self._cache_dir}/{prediction_hash}_prediction_lock.file"
        ).read_lock():
            if self._config.no_cache:
                return None
            cache_file = (
                f"{self._cache_dir}/{model_name}_{prediction_hash}_predictions.pkl"
            )

            if not os.path.exists(cache_file):
                return None

            logger.debug(f"Found model {model_name} predictions in cache")

            predictions = pickle.load(open(cache_file, "rb"))
            return predictions

    def _get_prediction_cache_hash(self, model_name: str, model: BaseEstimator) -> str:
        """
        Build a prediction-cache hash that binds together:
        1) predictor configuration (same as the old behavior)
        2) model identity (to prevent stale caches when model changes)
        """
        import pickle as _pkl

        config_hash = self._get_model_hash(model_name, df=None)

        # Prefer explicit model hash attached during training/cache load
        model_identity = getattr(model, "_frontier_model_hash", None)
        if model_identity is None:
            # Fallback: hash model bytes to avoid reusing old prediction cache across model changes
            try:
                model_identity = _pkl.dumps(model, protocol=_pkl.HIGHEST_PROTOCOL)
                model_identity = hashlib.md5(model_identity).hexdigest()[0:8]
            except Exception:
                model_identity = "no_model_hash"

        combined = f"{config_hash}_{model_identity}"
        return hashlib.md5(combined.encode("utf-8")).hexdigest()[0:8]

    def _get_model_prediction(
        self, model_name: str, model: BaseEstimator, X: pd.DataFrame
    ) -> Dict[Tuple, float]:
        prediction_hash = self._get_prediction_cache_hash(model_name, model)

        predictions = self._load_model_predication_cache(model_name, prediction_hash)
        if predictions is not None:
            return predictions

        logger.info(f"Predicting execution time for model {model_name}")

        model = self._models[model_name]
        X = X.copy()
        predictions_array = model.predict(X)

        # turn this into a dict, so we can store use it as a cache
        # the key is tuple for each row of X
        predictions = dict(zip([tuple(x) for x in X.values], predictions_array))

        self._store_model_predication_cache(model_name, prediction_hash, predictions)

        X["prediction"] = predictions_array
        X.to_csv(
            f"{self._cache_dir}/{model_name}_{prediction_hash}_predictions.csv",
            index=False,
        )

        return predictions

    def _train_compute_models(self) -> Dict[str, BaseEstimator]:
        models = {}
        compute_df_cache: Dict[int, pd.DataFrame] = {}

        def _get_compute_df_for_tp(tp_size: int) -> pd.DataFrame:
            if tp_size not in compute_df_cache:
                compute_df_cache[tp_size] = self._get_compute_df_with_derived_features(
                    self._load_compute_df(
                        self._compute_input_file, tensor_parallel_size=tp_size
                    )
                )
            return compute_df_cache[tp_size]

        model_names = self._get_compute_model_names()
        predictor_attention_extra_ops = self._get_predictor_attention_extra_ops()
        if predictor_attention_extra_ops:
            logger.info(
                "Architecture profile requires additional attention models: %s%s",
                ", ".join(predictor_attention_extra_ops),
                ", share_expert_*" if self._model_config.is_moe else "",
            )

        for model_name in model_names:
            target_col = f"time_stats.{model_name}.median"
            tp_key = self._get_linear_op_tp_key(model_name)
            compute_df = _get_compute_df_for_tp(tp_key)
            if target_col not in compute_df.columns:
                # For model-arch-required operations, raise error instead of warning.
                # - Architecture profile attention extras, e.g. attn_inter_norm, attn_wq_proj
                # - Shared-expert FFN operations when supported
                if model_name in self._get_predictor_attention_extra_ops():
                    raise ValueError(
                        f"Column '{target_col}' not found in compute dataframe. "
                        f"Architecture profile requires {model_name} profiling data. "
                        f"Re-run profiling with the selected model_architecture_profile."
                    )
                if (
                    model_name in ["share_expert_up_proj", "share_expert_down_proj", "share_expert_act"]
                    and self._model_config.supports_share_expert()
                    and self._model_config.is_moe
                ):
                    raise ValueError(
                        f"Column '{target_col}' not found in compute dataframe. "
                        f"Model requires {model_name} profiling data (supports_share_expert=True). "
                        f"Re-run linear-op profiling with share_expert enabled."
                    )
                if model_name in ["mtp_fusion_proj", "lm_head_linear"]:
                    raise ValueError(
                        "target-embedded MTP compute profiling columns are missing from "
                        f"{self._compute_input_file}. Missing column: '{target_col}'. "
                        "Re-run linear-op profiling with --include_target_embedded_mtp."
                    )
                logger.warning(
                    f"Column '{target_col}' not found in compute dataframe. "
                    f"Skipping {model_name} model training."
                )
                continue
            if compute_df[target_col].isna().all():
                raise ValueError(
                    f"Column '{target_col}' is all-NaN in compute dataframe for TP={tp_key}. "
                    "Please re-run linear-op profiling for this TP configuration."
                )
            logger.debug(
                f"Training model {model_name} at TP={tp_key}, size of training data: {len(compute_df)}"
            )
            models[model_name] = self._train_model(
                model_name=model_name,
                df=compute_df,
                feature_cols=["num_tokens"],
                target_col=target_col,
            )

        model_names = self._get_attention_model_names()
        if (
            model_names
            and self._get_attention_family().family_id
            == DENSE_ATTENTION_FAMILY.family_id
        ):
            attention_df = self._load_attention_df(self._attention_input_file)
            attention_df = self._get_attention_df_with_derived_features(attention_df)
            target_columns = dict(
                zip(
                    model_names,
                    get_enabled_predictor_median_columns(DENSE_ATTENTION_FAMILY),
                )
            )
            feature_columns = get_enabled_shared_predictor_feature_columns(
                DENSE_ATTENTION_FAMILY
            )
            kv_cache_model_name = get_enabled_predictor_metric_name_by_role(
                DENSE_ATTENTION_FAMILY,
                AttentionOperatorRole.CACHE_WRITE,
            )

            kv_cache_feature_cols = list(feature_columns[kv_cache_model_name])
            missing_cols = [
                col for col in kv_cache_feature_cols if col not in attention_df.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Missing columns for {kv_cache_model_name} training: {missing_cols}. "
                    "Re-run attention profiling with mixed-batch metadata."
                )
            models[kv_cache_model_name] = self._train_model(
                model_name=kv_cache_model_name,
                df=attention_df,
                feature_cols=kv_cache_feature_cols,
                target_col=target_columns[kv_cache_model_name],
            )

        # NOTE: Communication models (all_reduce, send_recv) are no longer trained here.
        # Communication predictions are now delegated to CC Backend.
        # See: _get_tensor_parallel_communication_time() and _get_pipeline_parallel_communication_time()

        return models

    def _train_mla_attention_layer_models(self) -> Dict[str, BaseEstimator]:
        attention_df = self._load_attention_df(self._attention_input_file)
        attention_df = self._get_attention_df_with_derived_features(attention_df)

        model_names = list(get_enabled_predictor_metric_names(LATENT_MLA_ATTENTION_FAMILY))
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
            feature_cols = list(feature_columns[model_name])
            target_col = target_columns[model_name]
            required_columns = [*feature_cols, target_col]
            missing_columns = [
                column for column in required_columns if column not in attention_df.columns
            ]
            all_nan_columns = [
                column
                for column in required_columns
                if column in attention_df.columns and attention_df[column].isna().all()
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

            model = self._train_model(
                model_name=model_name,
                df=op_attention_df,
                feature_cols=feature_cols,
                target_col=target_col,
            )
            model._frontier_exact_lookup = _build_exact_feature_lookup(
                op_attention_df,
                feature_cols,
                target_col,
            )
            models[model_name] = model
        return models

    def _train_cpu_overhead_models(self) -> Dict[str, BaseEstimator]:
        if self._config.skip_cpu_overhead_modeling:
            return {}

        models = {}
        model_names = [
            "schedule",
            "sampler_e2e",
            "prepare_inputs_e2e",
            "process_model_outputs",
            "ray_comm_time",
        ]

        cpu_overhead_df = self._load_cpu_overhead_df(self._cpu_overhead_input_file)
        if cpu_overhead_df.empty:
            logger.warning(
                "Skipping CPU overhead model training because no matching profiling rows "
                "were found for model_name='%s', tensor_parallel_degree=%s.",
                self._model_config.get_name(),
                self._replica_config.attn_tensor_parallel_size,
            )
            return {}
        cpu_overhead_df = self._get_cpu_overhead_df_with_derived_features(
            cpu_overhead_df
        )

        for model_name in model_names:
            if model_name == "ray_comm_time":
                target_col = "ray_comm_time_mean"
            else:
                target_col = f"{model_name}_median"

            model = self._train_model(
                model_name=model_name,
                df=cpu_overhead_df,
                feature_cols=[
                    "batch_size",
                    "num_prefill_tokens",
                    "num_decode_tokens",
                ],
                target_col=target_col,
            )
            model._frontier_exact_lookup = _build_exact_feature_lookup(
                cpu_overhead_df,
                ["batch_size", "num_prefill_tokens", "num_decode_tokens"],
                target_col,
            )
            models[model_name] = model

        return models

    def _train_attention_layer_models(self) -> Dict[str, BaseEstimator]:
        if self._is_mla_attention_family():
            return self._train_mla_attention_layer_models()

        attention_df = self._load_attention_df(self._attention_input_file)
        attention_df = self._get_attention_df_with_derived_features(attention_df)
        true_mixed_df = attention_df[attention_df["is_true_mixed_batch"]].copy()
        standard_df = attention_df[~attention_df["is_true_mixed_batch"]].copy()
        prefill_df = standard_df[~standard_df["is_decode"]].copy()
        decode_df = standard_df[standard_df["is_decode"]].copy()

        models = {}
        measurement_type = getattr(self, "_active_measurement_type", MeasurementType.CUDA_EVENT)
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
        prefill_model_name = get_enabled_predictor_metric_name_by_role(
            DENSE_ATTENTION_FAMILY,
            AttentionOperatorRole.PREFILL_KERNEL,
        )
        decode_model_name = get_enabled_predictor_metric_name_by_role(
            DENSE_ATTENTION_FAMILY,
            AttentionOperatorRole.DECODE_KERNEL,
        )

        if measurement_type == MeasurementType.CUDA_EVENT:
            if "prefill_chunk_size" not in prefill_df.columns:
                raise ValueError(
                    "Missing required column 'prefill_chunk_size' in attention profiling data."
                )
            standard_prefill_df = prefill_df[prefill_df["prefill_chunk_size"] > 0].copy()
            if len(standard_prefill_df) == 0:
                raise ValueError(
                    "No standard prefill rows (prefill_chunk_size > 0) found in eager attention profiling data."
                )

            models[prefill_model_name] = self._train_model(
                model_name=prefill_model_name,
                df=standard_prefill_df,
                feature_cols=list(dense_attention_feature_columns[prefill_model_name]),
                target_col=dense_attention_target_columns[prefill_model_name],
            )

            if len(decode_df) > 0:
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
                        "Skipping eager %s training: missing decode feature columns %s",
                        decode_model_name,
                        missing_decode_cols,
                    )
                else:
                    models[decode_model_name] = self._train_model(
                        model_name=decode_model_name,
                        df=decode_df,
                        feature_cols=decode_feature_cols,
                        target_col=dense_attention_target_columns[decode_model_name],
                    )
            else:
                logger.info(
                    "Skipping eager %s training: no standard decode rows",
                    decode_model_name,
                )

            mixed_feature_cols = [
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
            if all(col in prefill_df.columns for col in mixed_feature_cols):
                mixed_prefill_sources = [
                    prefill_df[
                        prefill_df["is_mixed_batch"] | (prefill_df["batch_size"] > 1)
                    ].copy()
                ]
                true_mixed_prefill_feature_map = {
                    "avg_seq_len": "prefill_mixed_avg_seq_len",
                    "batch_cv_interaction": "prefill_mixed_batch_cv_interaction",
                    "batch_size": "prefill_mixed_batch_size",
                    "batch_variance_interaction": (
                        "prefill_mixed_batch_variance_interaction"
                    ),
                    "kv_cache_size": "prefill_mixed_kv_cache_size",
                    "max_seq_len": "prefill_mixed_max_seq_len",
                    "min_seq_len": "prefill_mixed_min_seq_len",
                    "seq_len_cv": "prefill_mixed_seq_len_cv",
                    "seq_len_range": "prefill_mixed_seq_len_range",
                    "seq_len_variance": "prefill_mixed_seq_len_variance",
                    "total_tokens": "prefill_mixed_total_tokens",
                    "total_tokens_squared": "prefill_mixed_total_tokens_squared",
                }
                true_mixed_required_cols = [
                    *true_mixed_prefill_feature_map.values(),
                    "time_stats.attn_prefill.median",
                ]
                if len(true_mixed_df) > 0 and all(
                    col in true_mixed_df.columns for col in true_mixed_required_cols
                ):
                    true_mixed_prefill_df = true_mixed_df[
                        true_mixed_df["time_stats.attn_prefill.median"].notna()
                    ].copy()
                    if len(true_mixed_prefill_df) > 0:
                        for (
                            training_col,
                            source_col,
                        ) in true_mixed_prefill_feature_map.items():
                            true_mixed_prefill_df[training_col] = (
                                true_mixed_prefill_df[source_col]
                            )
                        valid_feature_mask = true_mixed_prefill_df[
                            [*mixed_feature_cols, "time_stats.attn_prefill.median"]
                        ].notna().all(axis=1)
                        invalid_count = int((~valid_feature_mask).sum())
                        if invalid_count:
                            logger.warning(
                                "Dropping %d/%d true-mixed rows with incomplete "
                                "prefill-side mixed features before training "
                                "attn_prefill_mixed.",
                                invalid_count,
                                len(true_mixed_prefill_df),
                            )
                        true_mixed_prefill_df = true_mixed_prefill_df.loc[
                            valid_feature_mask
                        ].copy()
                        if len(true_mixed_prefill_df) == 0:
                            raise ValueError(
                                "True-mixed attention profiling data contains "
                                "attn_prefill targets but no rows with complete "
                                "prefill-side mixed features for attn_prefill_mixed."
                            )
                        mixed_prefill_sources.append(true_mixed_prefill_df)
                elif len(true_mixed_df) > 0 and all(
                    col in true_mixed_df.columns
                    for col in [*mixed_feature_cols, "time_stats.attn_prefill.median"]
                ):
                    true_mixed_prefill_df = true_mixed_df[
                        true_mixed_df["time_stats.attn_prefill.median"].notna()
                    ].copy()
                    valid_feature_mask = true_mixed_prefill_df[
                        [*mixed_feature_cols, "time_stats.attn_prefill.median"]
                    ].notna().all(axis=1)
                    true_mixed_prefill_df = true_mixed_prefill_df.loc[
                        valid_feature_mask
                    ].copy()
                    if len(true_mixed_prefill_df) > 0:
                        mixed_prefill_sources.append(true_mixed_prefill_df)
                mixed_prefill_df = pd.concat(
                    mixed_prefill_sources,
                    ignore_index=True,
                )
                if len(mixed_prefill_df) > 0:
                    models["attn_prefill_mixed"] = self._train_model(
                        model_name="attn_prefill_mixed",
                        df=mixed_prefill_df,
                        feature_cols=mixed_feature_cols,
                        target_col="time_stats.attn_prefill.median",
                    )
                    logger.info(
                        "Trained model attn_prefill_mixed with %d mixed-batch samples",
                        len(mixed_prefill_df),
                    )
                else:
                    logger.info(
                        "Skipping attn_prefill_mixed training: no rows with batch_size > 1"
                    )
            else:
                missing_cols = [c for c in mixed_feature_cols if c not in prefill_df.columns]
                logger.info(
                    "Skipping attn_prefill_mixed training: missing mixed-batch feature columns %s",
                    missing_cols,
                )

            decode_in_mixed_feature_cols = [
                "decode_batch_size",
                "decode_avg_kv_cache_size",
                "num_prefill_seqs",
                "total_prefill_tokens",
                "total_batch_size",
                "batch_composition_ratio",
                "total_tokens",
            ]
            if all(col in true_mixed_df.columns for col in decode_in_mixed_feature_cols):
                if len(true_mixed_df) > 0:
                    models["attn_decode_in_mixed"] = self._train_model(
                        model_name="attn_decode_in_mixed",
                        df=true_mixed_df,
                        feature_cols=decode_in_mixed_feature_cols,
                        target_col="time_stats.attn_decode.median",
                    )
                    logger.info(
                        "Trained model attn_decode_in_mixed with %d true mixed samples",
                        len(true_mixed_df),
                    )
                else:
                    logger.info(
                        "Skipping attn_decode_in_mixed training: no true mixed rows"
                    )
            else:
                missing_cols = [
                    c for c in decode_in_mixed_feature_cols if c not in true_mixed_df.columns
                ]
                logger.info(
                    "Skipping attn_decode_in_mixed training: missing true mixed feature columns %s",
                    missing_cols,
                )

            return models

        if measurement_type != MeasurementType.KERNEL_ONLY:
            raise ValueError(f"Unsupported measurement_type={measurement_type!r}")

        if len(decode_df) == 0:
            raise ValueError(
                "No standard decode rows found in kernel-only attention profiling data."
            )

        models[decode_model_name] = self._train_model(
            model_name=decode_model_name,
            df=decode_df,
            feature_cols=list(dense_attention_feature_columns[decode_model_name]),
            target_col=dense_attention_target_columns[decode_model_name],
        )

        decode_in_mixed_feature_cols = [
            "decode_batch_size",
            "decode_avg_kv_cache_size",
            "num_prefill_seqs",
            "total_prefill_tokens",
            "total_batch_size",
            "batch_composition_ratio",
            "total_tokens",
        ]
        if all(col in true_mixed_df.columns for col in decode_in_mixed_feature_cols):
            if len(true_mixed_df) > 0:
                models["attn_decode_in_mixed"] = self._train_model(
                    model_name="attn_decode_in_mixed",
                    df=true_mixed_df,
                    feature_cols=decode_in_mixed_feature_cols,
                    target_col="time_stats.attn_decode.median",
                )
                logger.info(
                    "Trained kernel-only model attn_decode_in_mixed with %d true mixed samples",
                    len(true_mixed_df),
                )
            else:
                logger.info(
                    "Skipping kernel-only attn_decode_in_mixed training: no true mixed rows"
                )
        else:
            missing_cols = [
                c for c in decode_in_mixed_feature_cols if c not in true_mixed_df.columns
            ]
            logger.info(
                "Skipping kernel-only attn_decode_in_mixed training: missing true mixed feature columns %s",
                missing_cols,
            )
        return models

    def _train_models(self) -> Dict[str, BaseEstimator]:
        models = self._train_compute_models()
        models.update(self._train_cpu_overhead_models())
        models.update(self._train_attention_layer_models())

        return models

    # Predict per-operation lookup caches from trained sklearn models.
    # This method supports both dense (MLP) and MoE models; MoE vs dense is determined by ReplicaConfig.model_config.is_moe.
    def _predict_for_compute_models(self) -> Dict[str, Any]:
        predictions = {}

        # Determine which models to predict based on cluster type
        model_names = []

        # Add attention-related compute models for clusters that support attention
        if self._cluster_type in [
            ClusterType.PREFILL,
            ClusterType.DECODE_ATTN,
            ClusterType.DECODE,
            ClusterType.MONOLITHIC,
        ]:
            model_names.extend(
                [
                    "attn_pre_proj",
                    "attn_post_proj",
                    "attn_rope",
                    self._dense_attention_cache_write_op_name(),
                ]
            )

            model_names.extend(self._get_predictor_attention_extra_ops())

        # Add FFN-related compute models for clusters that support FFN.
        # Distinguish MoE vs dense models based on ReplicaConfig.model_config.is_moe.
        model_config = getattr(self._replica_config, "model_config", None)
        if model_config is None or not hasattr(model_config, "is_moe"):
            raise ValueError(
                "ReplicaConfig.model_config.is_moe is required to distinguish MoE vs dense models"
            )
        is_moe_model = bool(model_config.is_moe)

        if self._cluster_type in [
            ClusterType.PREFILL,
            ClusterType.DECODE_FFN,
            ClusterType.DECODE,
            ClusterType.MONOLITHIC,
        ]:
            if is_moe_model:
                model_names.extend(
                    [
                        "moe_gating_linear",       # Split gating: linear layer
                        "moe_gating_routing_topk", # Split gating: topk + softmax
                        "moe_shuffling",
                        "moe_grouped_gemm",
                    ]
                )

                # Mixed-layer MoE models require dense MLP predictors on non-MoE layers.
                if self._requires_dense_mlp_compute_models():
                    model_names.extend(
                        [
                            "mlp_up_proj",
                            "mlp_down_proj",
                            "mlp_act",
                        ]
                    )

                # Shared-expert operations follow the architecture profile capability.
                if self._model_config.supports_share_expert():
                    model_names.extend([
                        "share_expert_up_proj",
                        "share_expert_down_proj",
                        "share_expert_act",
                    ])
            else:
                model_names.extend(
                    [
                        "mlp_up_proj",
                        "mlp_down_proj",
                        "mlp_act",
                    ]
                )

        # Common models needed by all clusters
        model_names.extend(
            [
                "input_layernorm",
                "post_attention_layernorm",
            ]
        )
        if self._requires_target_embedded_mtp_compute_models():
            model_names.extend(
                [
                    "emb",
                    "mtp_fusion_proj",
                    "lm_head_linear",
                ]
            )
        # add is only a separate operation for non-fused LayerNorm models
        if not self._model_config.uses_fused_add_norm:
            model_names.append("add")

        # NOTE: Communication models (all_reduce, send_recv) are no longer added here.
        # Communication predictions are now delegated to CC Backend.
        # See: _get_tensor_parallel_communication_time() and _get_pipeline_parallel_communication_time()

        num_token_range = np.arange(1, self._max_tokens + 1)
        X = pd.DataFrame({"num_tokens": num_token_range})

        for model_name in model_names:
            if model_name not in self._models:
                continue

            model = self._models[model_name]
            n_features = getattr(model, "n_features_in_", None)
            if n_features is None:
                raise ValueError(
                    f"Model {model_name} is missing n_features_in_; cannot determine feature shape"
                )

            # Single-feature models are cached as a lookup table over num_tokens.
            if int(n_features) == 1:
                predictions[model_name] = self._get_model_prediction(
                    model_name, model, X
                )
                continue

            # Multi-feature models are stored for on-demand prediction.
            # Example: moe_grouped_gemm in load imbalance mode uses 14 features.
            feature_names = getattr(model, "_frontier_feature_names", None)
            if feature_names is None:
                feature_names = getattr(model, "feature_names_in_", None)

            if feature_names is None or len(feature_names) != int(n_features):
                raise ValueError(
                    f"Model {model_name} has inconsistent feature metadata: "
                    f"n_features_in_={n_features}, feature_names_in_={feature_names}"
                )

            predictions[model_name] = {
                "_on_demand_prediction": True,
                "_n_features": int(n_features),
                "_model": model,
                "_feature_names": list(feature_names),
            }

        return predictions

    def _predict_for_cpu_overhead_models(self) -> Dict[str, Any]:
        if self._config.skip_cpu_overhead_modeling:
            return {}

        predictions = {}

        model_names = [
            "schedule",
            "sampler_e2e",
            "prepare_inputs_e2e",
            "process_model_outputs",
            "ray_comm_time",
        ]

        batch_size_range = np.arange(1, self._config.prediction_max_batch_size + 1)
        X = pd.DataFrame({"batch_size": batch_size_range})

        for model_name in model_names:
            if model_name in self._models:
                model = self._models[model_name]
                n_features = getattr(model, "n_features_in_", None)
                if n_features is None:
                    raise ValueError(
                        f"CPU overhead model {model_name} is missing n_features_in_."
                    )

                if int(n_features) == 1:
                    predictions[model_name] = self._get_model_prediction(
                        model_name, model, X
                    )
                    continue

                feature_names = getattr(model, "_frontier_feature_names", None)
                if feature_names is None:
                    feature_names = getattr(model, "feature_names_in_", None)

                if feature_names is None or len(feature_names) != int(n_features):
                    raise ValueError(
                        f"CPU overhead model {model_name} has inconsistent feature metadata: "
                        f"n_features_in_={n_features}, feature_names_in_={feature_names}"
                    )

                predictions[model_name] = {
                    "_on_demand_prediction": True,
                    "_n_features": int(n_features),
                    "_model": model,
                    "_feature_names": list(feature_names),
                    "_exact_lookup": getattr(model, "_frontier_exact_lookup", {}),
                }

        return predictions

    def _predict_for_attention_layer_models(self) -> Dict[str, Any]:
        predictions = {}

        # Only predict attention layer models for clusters that support attention
        # Note: ClusterType.DECODE is the unified decode cluster in PD-disaggregation mode
        if self._cluster_type not in [
            ClusterType.PREFILL,
            ClusterType.DECODE_ATTN,
            ClusterType.DECODE,
            ClusterType.MONOLITHIC,
        ]:
            return predictions

        if self._is_mla_attention_family():
            feature_columns = get_enabled_shared_predictor_feature_columns(
                LATENT_MLA_ATTENTION_FAMILY
            )
            for model_name in get_enabled_predictor_metric_names(
                LATENT_MLA_ATTENTION_FAMILY
            ):
                if model_name not in self._models:
                    continue
                model = self._models[model_name]
                feature_names = getattr(model, "_frontier_feature_names", None)
                if feature_names is None:
                    feature_names = getattr(model, "feature_names_in_", None)
                expected_feature_names = list(feature_columns[model_name])
                if list(feature_names or []) != expected_feature_names:
                    raise ValueError(
                        "MLA attention model feature schema mismatch for "
                        f"{model_name}: expected {expected_feature_names}, got "
                        f"{list(feature_names or [])}"
                    )
                n_features = getattr(model, "n_features_in_", None)
                if n_features is None:
                    n_features = len(expected_feature_names)
                if int(n_features) != len(expected_feature_names):
                    raise ValueError(
                        "MLA attention model feature count mismatch for "
                        f"{model_name}: expected {len(expected_feature_names)}, "
                        f"got {n_features}"
                    )
                predictions[model_name] = {
                    "_on_demand_prediction": True,
                    "_n_features": int(n_features),
                    "_model": model,
                    "_feature_names": expected_feature_names,
                    "_exact_lookup": getattr(model, "_frontier_exact_lookup", {}),
                }
            return predictions

        measurement_type = getattr(self, "_active_measurement_type", MeasurementType.CUDA_EVENT)

        # Cluster-specific needs with measurement-aware family split.
        need_prefill = measurement_type == MeasurementType.CUDA_EVENT and self._cluster_type in [
            ClusterType.PREFILL,
            ClusterType.MONOLITHIC,
        ]
        need_decode = self._cluster_type in [
            ClusterType.DECODE_ATTN,
            ClusterType.DECODE,
            ClusterType.MONOLITHIC,
        ] and (
            measurement_type == MeasurementType.KERNEL_ONLY
            or (
                measurement_type == MeasurementType.CUDA_EVENT
                and self._cluster_type
                in [
                    ClusterType.DECODE_ATTN,
                    ClusterType.DECODE,
                    ClusterType.MONOLITHIC,
                ]
            )
        )

        # Build only the grids we actually need to avoid unnecessary compute/memory.
        decode_df = None
        prefill_df = None

        decode_op_name = self._dense_attention_decode_op_name()
        prefill_op_name = self._dense_attention_prefill_op_name()

        if need_decode and decode_op_name in self._models:
            decode_batch_size_range = np.arange(
                1, self._config.prediction_max_batch_size + 1
            )
            decode_kv_cache_size_range = np.arange(
                0,
                self._config.prediction_max_tokens_per_request + 1,
                self._config.kv_cache_prediction_granularity,
            )
            decode_df = pd.DataFrame(
                {
                    "batch_size": np.repeat(
                        decode_batch_size_range, len(decode_kv_cache_size_range)
                    ),
                    "kv_cache_size": np.tile(
                        decode_kv_cache_size_range, len(decode_batch_size_range)
                    ),
                }
            )
            predictions[decode_op_name] = self._get_model_prediction(
                decode_op_name,
                self._models[decode_op_name],
                decode_df[["batch_size", "kv_cache_size"]],
            )

        if need_prefill and prefill_op_name in self._models:
            prefill_kv_cache_size_range = np.arange(
                0,
                self._config.prediction_max_tokens_per_request + 1,
                self._config.kv_cache_prediction_granularity,
            )
            prefill_prefill_chunk_size_range = np.arange(
                1, self._config.prediction_max_prefill_chunk_size + 1
            )
            # PREFILL training data uses batch_size=1 for per-request prediction in this cache.
            prefill_df = pd.DataFrame(
                {
                    "kv_cache_size": np.repeat(
                        prefill_kv_cache_size_range,
                        len(prefill_prefill_chunk_size_range),
                    ),
                    "prefill_chunk_size_squared": np.tile(
                        prefill_prefill_chunk_size_range,
                        len(prefill_kv_cache_size_range),
                    )
                    ** 2,
                }
            )
            predictions[prefill_op_name] = self._get_model_prediction(
                prefill_op_name,
                self._models[prefill_op_name],
                prefill_df[["kv_cache_size", "prefill_chunk_size_squared"]],
            )

        # Handle attn_prefill_mixed: high-dimensional model requiring on-demand prediction
        # This model uses 12 features and cannot be pre-computed efficiently
        if need_prefill and "attn_prefill_mixed" in self._models:
            model = self._models["attn_prefill_mixed"]
            n_features = getattr(model, "n_features_in_", 12)
            feature_names = list(
                getattr(
                    model,
                    "_frontier_feature_names",
                    [
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
                    ],
                )
            )
            logger.info(
                f"attn_prefill_mixed detected with {n_features} features. "
                f"Prediction will be computed on-demand instead of using a lookup cache."
            )
            # Store the model and feature names for on-demand prediction
            predictions["attn_prefill_mixed"] = {
                "_on_demand_prediction": True,
                "_n_features": n_features,
                "_model": model,  # Store model for on-demand prediction
                "_feature_names": feature_names,
            }

        need_true_mixed_decode = self._cluster_type == ClusterType.MONOLITHIC
        if need_true_mixed_decode and "attn_decode_in_mixed" in self._models:
            model = self._models["attn_decode_in_mixed"]
            feature_names = list(
                getattr(
                    model,
                    "_frontier_feature_names",
                    [
                        "decode_batch_size",
                        "decode_avg_kv_cache_size",
                        "num_prefill_seqs",
                        "total_prefill_tokens",
                        "total_batch_size",
                        "batch_composition_ratio",
                        "total_tokens",
                    ],
                )
            )
            predictions["attn_decode_in_mixed"] = {
                "_on_demand_prediction": True,
                "_n_features": getattr(model, "n_features_in_", len(feature_names)),
                "_model": model,
                "_feature_names": feature_names,
            }

        return predictions

    def _predict_from_models(self) -> Dict[str, Any]:
        predictions = {}

        # Always predict compute models (cluster-specific filtering happens inside)
        predictions.update(self._predict_for_compute_models())

        # Only predict CPU overhead models if not skipped
        if not self._config.skip_cpu_overhead_modeling:
            predictions.update(self._predict_for_cpu_overhead_models())

        # Only predict attention layer models for clusters that support attention
        # Note: ClusterType.DECODE is the unified decode cluster in PD-disaggregation mode
        if self._cluster_type in [
            ClusterType.PREFILL,
            ClusterType.DECODE_ATTN,
            ClusterType.DECODE,
            ClusterType.MONOLITHIC,
        ]:
            predictions.update(self._predict_for_attention_layer_models())

        logger.info(
            f"Generated predictions for {len(predictions)} model types for cluster {self._cluster_type}"
        )
        return predictions

    def _get_batch_decode_attention_params(self, batch: Batch) -> Tuple[int, int]:
        if hasattr(batch, "_decode_params"):
            return batch._decode_params

        decode_kv_cache_sizes = []

        for request in batch.requests:
            if request._is_prefill_complete:
                decode_kv_cache_sizes.append(request.num_processed_tokens)

        if not decode_kv_cache_sizes:
            batch._decode_params = (0, 0)
            return batch._decode_params

        if hasattr(batch, "get_effective_decode_batch_size_for_attention"):
            decode_batch_size = int(
                batch.get_effective_decode_batch_size_for_attention()
            )
        else:
            decode_batch_size = len(decode_kv_cache_sizes)

        decode_avg_kv_cache_size = int(np.mean(decode_kv_cache_sizes))
        # round up to the nearest multiple of kv_cache_prediction_granularity in csv file
        decode_avg_kv_cache_size = (
            (
                decode_avg_kv_cache_size
                + self._config.kv_cache_prediction_granularity
                - 1
            )
            // self._config.kv_cache_prediction_granularity
        ) * self._config.kv_cache_prediction_granularity

        batch._decode_params = (decode_batch_size, decode_avg_kv_cache_size)

        return batch._decode_params

    def _get_batch_prefill_attention_params(
        self, batch: Batch
    ) -> List[Tuple[int, int]]:
        if hasattr(batch, "_prefill_params"):
            return batch._prefill_params

        prefill_params = []

        for request, num_tokens_to_process in zip(batch.requests, batch.num_tokens):
            if request._is_prefill_complete:
                continue

            prefill_chunk_size = num_tokens_to_process
            kv_cache_size = (
                (
                    request.num_processed_tokens
                    + self._config.kv_cache_prediction_granularity
                    - 1
                )
                // self._config.kv_cache_prediction_granularity
            ) * self._config.kv_cache_prediction_granularity

            prefill_params.append((kv_cache_size, prefill_chunk_size))

        batch._prefill_params = prefill_params

        return prefill_params

    def _get_batch_prefill_mixed_features(self, batch: Batch) -> Dict[str, float]:
        """
        Extract mixed-prefill features for attn_prefill_mixed from a mixed prefill batch.
        Feature order must match ATTN_PREFILL_MIXED_FEATURES exactly.

        Features:
        - batch_size: Number of prefill requests in the batch
        - kv_cache_size: Rounded average KV cache context for the batch
        - total_tokens: Total number of tokens across all prefill requests
        - avg_seq_len: Average sequence length
        - min_seq_len: Minimum sequence length
        - max_seq_len: Maximum sequence length
        - total_tokens_squared: total_tokens^2 for quadratic attention
        - seq_len_variance: Variance of sequence lengths
        - seq_len_cv: Coefficient of variation (std/mean)
        - seq_len_range: max_seq_len - min_seq_len
        - batch_variance_interaction: batch_size * seq_len_variance
        - batch_cv_interaction: batch_size * seq_len_cv
        """
        if hasattr(batch, "_prefill_mixed_features"):
            return batch._prefill_mixed_features

        # Collect sequence lengths and live cache context for prefill requests only.
        seq_lens = []
        kv_cache_sizes = []
        for request, num_tokens in zip(batch.requests, batch.num_tokens):
            if not request._is_prefill_complete:
                seq_lens.append(num_tokens)
                kv_cache_sizes.append(request.num_processed_tokens)

        if not seq_lens:
            # No prefill requests - return zeros (should not happen in normal flow)
            return {
                "batch_size": 0,
                "kv_cache_size": 0,
                "total_tokens": 0,
                "avg_seq_len": 0,
                "min_seq_len": 0,
                "max_seq_len": 0,
                "total_tokens_squared": 0,
                "seq_len_variance": 0,
                "seq_len_cv": 0,
                "seq_len_range": 0,
                "batch_variance_interaction": 0,
                "batch_cv_interaction": 0,
            }

        import numpy as np

        seq_lens_arr = np.array(seq_lens, dtype=np.float64)

        batch_size = len(seq_lens)
        avg_kv_cache_size = int(np.mean(kv_cache_sizes))
        avg_kv_cache_size = (
            (
                avg_kv_cache_size
                + self._config.kv_cache_prediction_granularity
                - 1
            )
            // self._config.kv_cache_prediction_granularity
        ) * self._config.kv_cache_prediction_granularity
        total_tokens = int(seq_lens_arr.sum())
        avg_seq_len = float(seq_lens_arr.mean())
        min_seq_len = int(seq_lens_arr.min())
        max_seq_len = int(seq_lens_arr.max())
        total_tokens_squared = total_tokens**2
        seq_len_variance = float(seq_lens_arr.var()) if batch_size > 1 else 0.0
        seq_len_cv = (
            (float(seq_lens_arr.std()) / avg_seq_len) if avg_seq_len > 0 else 0.0
        )
        seq_len_range = max_seq_len - min_seq_len
        batch_variance_interaction = batch_size * seq_len_variance
        batch_cv_interaction = batch_size * seq_len_cv

        features = {
            "batch_size": batch_size,
            "kv_cache_size": avg_kv_cache_size,
            "total_tokens": total_tokens,
            "avg_seq_len": avg_seq_len,
            "min_seq_len": min_seq_len,
            "max_seq_len": max_seq_len,
            "total_tokens_squared": total_tokens_squared,
            "seq_len_variance": seq_len_variance,
            "seq_len_cv": seq_len_cv,
            "seq_len_range": seq_len_range,
            "batch_variance_interaction": batch_variance_interaction,
            "batch_cv_interaction": batch_cv_interaction,
        }

        batch._prefill_mixed_features = features
        return features

    def _get_batch_decode_mixed_features(self, batch: Batch) -> Dict[str, float]:
        """Extract decode-in-mixed features for true mixed prefill+decode batches."""
        if hasattr(batch, "_decode_mixed_features"):
            return batch._decode_mixed_features

        prefill_tokens = []
        decode_kv_cache_sizes = []
        for request, num_tokens in zip(batch.requests, batch.num_tokens):
            if request._is_prefill_complete:
                decode_kv_cache_sizes.append(request.num_processed_tokens)
            else:
                prefill_tokens.append(num_tokens)

        decode_batch_size = len(decode_kv_cache_sizes)
        num_prefill_seqs = len(prefill_tokens)
        total_batch_size = decode_batch_size + num_prefill_seqs
        total_prefill_tokens = int(sum(prefill_tokens))
        total_tokens = int(batch.total_num_tokens)

        if decode_batch_size == 0:
            decode_avg_kv_cache_size = 0
        else:
            decode_avg_kv_cache_size = int(np.mean(decode_kv_cache_sizes))
            decode_avg_kv_cache_size = (
                (
                    decode_avg_kv_cache_size
                    + self._config.kv_cache_prediction_granularity
                    - 1
                )
                // self._config.kv_cache_prediction_granularity
            ) * self._config.kv_cache_prediction_granularity

        batch_composition_ratio = (
            float(num_prefill_seqs) / float(total_batch_size)
            if total_batch_size > 0
            else 0.0
        )

        features = {
            "decode_batch_size": decode_batch_size,
            "decode_avg_kv_cache_size": decode_avg_kv_cache_size,
            "num_prefill_seqs": num_prefill_seqs,
            "total_prefill_tokens": total_prefill_tokens,
            "total_batch_size": total_batch_size,
            "batch_composition_ratio": batch_composition_ratio,
            "total_tokens": total_tokens,
        }
        batch._decode_mixed_features = features
        return features

    def _get_on_demand_prediction(
        self, model_name: str, features: Dict[str, float]
    ) -> float:
        """
        Perform on-demand prediction with runtime caching for high-dimensional models.

        This method is used for models with many features (e.g., attn_prefill_mixed with 12 features)
        where pre-computing all combinations is impractical.

        Args:
            model_name: Name of the prediction model (e.g., "attn_prefill_mixed")
            features: Dictionary of feature names to values

        Returns:
            Predicted execution time in seconds
        """
        # Create cache key from features (must be hashable)
        # Use sorted keys for consistent ordering
        cache_key = tuple(features[k] for k in sorted(features.keys()))

        family_name = self._measurement_family_name(self._active_measurement_type)

        # Check runtime cache first
        if cache_key in self._runtime_cache[family_name][model_name]:
            return self._runtime_cache[family_name][model_name][cache_key]

        # Get model from predictions dict
        model_info = self._predictions.get(model_name)
        if model_info is None:
            raise ValueError(f"Model {model_name} not found in predictions")

        # Extract model and feature names from the on-demand prediction marker
        if (
            not isinstance(model_info, dict)
            or "_on_demand_prediction" not in model_info
        ):
            raise ValueError(
                f"Model {model_name} is not configured for on-demand prediction"
            )

        model = model_info.get("_model")
        feature_names = model_info.get("_feature_names", [])

        # Build feature vector in correct order.
        # Fail-fast if any required feature is missing (no silent defaults).
        missing = [fn for fn in feature_names if fn not in features]
        if missing:
            raise ValueError(
                f"On-demand prediction missing required features for {model_name}: {missing}. "
                f"Provided keys: {sorted(list(features.keys()))}"
            )

        feature_key = tuple(float(features[fn]) for fn in feature_names)
        exact_lookup = model_info.get("_exact_lookup") or {}
        if feature_key in exact_lookup:
            prediction = float(exact_lookup[feature_key])
            self._runtime_cache[family_name][model_name][cache_key] = prediction
            return prediction

        if model is None:
            raise ValueError(f"Model {model_name} has no trained model available")

        feature_vector = pd.DataFrame(
            [[features[fn] for fn in feature_names]],
            columns=feature_names,
        )

        # Make prediction
        try:
            prediction = float(model.predict(feature_vector)[0])
            # Ensure non-negative prediction
            prediction = max(0.0, prediction)
        except Exception as e:
            raise ValueError(f"On-demand prediction failed for {model_name}: {e}")

        # Cache the result
        self._runtime_cache[family_name][model_name][cache_key] = prediction

        return prediction

    def _get_tensor_parallel_size_for_comm(self) -> int:
        if self._cluster_type == ClusterType.DECODE_FFN:
            return self._replica_config.moe_tensor_parallel_size
        return self._replica_config.attn_tensor_parallel_size

    def _get_decode_cuda_graph_runtime_mode(self, batch: Batch) -> str:
        if hasattr(batch, "get_decode_cuda_graph_runtime_mode"):
            return str(batch.get_decode_cuda_graph_runtime_mode())

        metadata = getattr(batch, "decode_cuda_graph_metadata", None)
        if metadata is None:
            return "NONE"
        return str(getattr(metadata, "runtime_mode", "NONE"))

    def _should_strip_collective_sim_allreduce_launch_overhead(
        self, batch: Batch
    ) -> bool:
        measurement_type = getattr(self, "_active_measurement_type", None)
        if measurement_type is None:
            measurement_type = self._select_measurement_type_for_batch(batch)
        if measurement_type != MeasurementType.KERNEL_ONLY:
            return False
        if getattr(batch, "num_decode_tokens", 0) <= 0:
            return False
        runtime_mode = self._get_decode_cuda_graph_runtime_mode(batch)
        if runtime_mode == "FULL":
            return True
        if runtime_mode != "PIECEWISE":
            return False
        # PIECEWISE mixed batches still pay explicit communication. Only the
        # pure-decode captured path should strip eager-only launch overhead.
        return getattr(batch, "num_prefill_tokens", 0) <= 0

    def _strip_collective_sim_allreduce_launch_overhead_if_needed(
        self,
        *,
        batch: Batch,
        predicted_ms: float,
        num_devices: int,
        comm_domain: str,
    ) -> float:
        if predicted_ms <= 0.0:
            return predicted_ms
        if not self._should_strip_collective_sim_allreduce_launch_overhead(batch):
            return predicted_ms
        if self._cc_backend is None:
            return predicted_ms

        launch_overhead_fn = getattr(
            type(self._cc_backend),
            "estimate_intra_server_allreduce_launch_overhead_ms",
            None,
        )
        if not callable(launch_overhead_fn):
            return predicted_ms

        launch_overhead_ms = float(
            launch_overhead_fn(
                self._cc_backend,
                num_devices=num_devices,
                comm_domain=comm_domain,
            )
        )
        if launch_overhead_ms <= 0.0:
            return predicted_ms

        adjusted_ms = max(predicted_ms - launch_overhead_ms, 0.0)
        logger.debug(
            "[CUDA_GRAPH][COMM] stripped collective-sim allreduce launch overhead: "
            "predicted_ms=%s, launch_overhead_ms=%s, adjusted_ms=%s, "
            "cluster_type=%s, comm_domain=%s, num_devices=%s",
            predicted_ms,
            launch_overhead_ms,
            adjusted_ms,
            self._cluster_type,
            comm_domain,
            num_devices,
        )
        return adjusted_ms

    def _supports_operation(self, operation: str) -> bool:
        """
        Check if the current cluster configuration supports a specific operation.

        Step2Mini/Step3-specific operations:
        - architecture-profile attention extras: supported by attention clusters
        - share_expert_up_proj, share_expert_down_proj, share_expert_act: Part of FFN (forward_3),
          supported by FFN clusters
        """
        if operation in [
            "attention",
            "attn_pre_proj",
            "attn_post_proj",
            "attn_rope",
            self._dense_attention_cache_write_op_name(),
        ]:
            return self._cluster_type in [
                ClusterType.PREFILL,
                ClusterType.DECODE_ATTN,
                ClusterType.DECODE,
                ClusterType.MONOLITHIC,
            ]

        if operation in self._get_predictor_attention_extra_ops():
            return self._cluster_type in [
                ClusterType.PREFILL,
                ClusterType.DECODE_ATTN,
                ClusterType.DECODE,
                ClusterType.MONOLITHIC,
            ]

        if operation == self._dense_attention_prefill_op_name():
            return self._cluster_type in [ClusterType.PREFILL, ClusterType.MONOLITHIC]

        if operation == self._dense_attention_decode_op_name():
            return self._cluster_type in [
                ClusterType.DECODE_ATTN,
                ClusterType.DECODE,
                ClusterType.MONOLITHIC,
            ]

        if operation in {
            *get_family_profiling_name_set(FFN_FAMILY),
            "moe_grouped_gemm",
            "expert_parallel_communication",
            "moe_gating_linear",  # Split gating: linear layer
            "moe_gating_routing_topk",  # Split gating: topk + softmax
            "moe_shuffling",
        }:
            return self._cluster_type in [
                ClusterType.PREFILL,
                ClusterType.DECODE_FFN,
                ClusterType.DECODE,
                ClusterType.MONOLITHIC,
            ]

        # Step2Mini/Step3 share_expert operations (forward_3: shared expert alongside routed experts)
        # These are part of the FFN path and should be handled by FFN clusters
        if operation in get_family_profiling_name_set(SHARE_EXPERT_FAMILY):
            # Only supported for Step2Mini/Step3 MoE models
            if not self._model_config.supports_share_expert():
                return False
            if not self._model_config.is_moe:
                return False
            return self._cluster_type in [
                ClusterType.PREFILL,
                ClusterType.DECODE_FFN,
                ClusterType.DECODE,
                ClusterType.MONOLITHIC,
            ]

        if operation in ["pipeline_parallel_communication", "send_recv"]:
            return self._replica_config.num_pipeline_stages > 1

        if operation in ["tensor_parallel_communication", "all_reduce"]:
            return self._get_tensor_parallel_size_for_comm() > 1

        # Common operations supported by all clusters
        if operation in [
            "input_layernorm",
            "post_attention_layernorm",
            "schedule",
            "sampler_e2e",
            "prepare_inputs_e2e",
            "process_model_outputs",
            "ray_comm_time",
        ]:
            return True

        # add is only a separate operation for non-fused LayerNorm models
        if operation == "add":
            return not self._model_config.uses_fused_add_norm

        return False

    def _get_attention_layer_pre_proj_execution_time(self, batch: Batch) -> float:
        if not self._supports_operation("attn_pre_proj"):
            raise ValueError(
                f"attention pre-projection operation not supported for cluster {self._cluster_type}"
            )
        effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
        raw_time = self._predictions["attn_pre_proj"][(effective_tokens,)]
        if getattr(batch, "num_prefill_tokens", 0) > 0:
            prefill_phase_scale = self._get_optional_calibration_scale(
                "_prefill_phase_attn_pre_proj_calibration_scale",
                "prefill_phase_attn_pre_proj_calibration_scale",
            )
            if prefill_phase_scale is not None:
                return raw_time * prefill_phase_scale
        scale = self._get_calibration_scale(
            "_attn_pre_proj_calibration_scale", "attn_pre_proj_calibration_scale"
        )
        return raw_time * scale

    def _get_attention_layer_post_proj_execution_time(self, batch: Batch) -> float:
        if not self._supports_operation("attn_post_proj"):
            raise ValueError(
                f"attention post-projection operation not supported for cluster {self._cluster_type}"
            )
        effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
        raw_time = self._predictions["attn_post_proj"][(effective_tokens,)]
        if getattr(batch, "num_prefill_tokens", 0) > 0:
            prefill_phase_scale = self._get_optional_calibration_scale(
                "_prefill_phase_attn_post_proj_calibration_scale",
                "prefill_phase_attn_post_proj_calibration_scale",
            )
            if prefill_phase_scale is not None:
                return raw_time * prefill_phase_scale
        scale = self._get_calibration_scale(
            "_attn_post_proj_calibration_scale", "attn_post_proj_calibration_scale"
        )
        return raw_time * scale

    def _get_mlp_layer_up_proj_execution_time(self, batch: Batch) -> float:
        if not self._supports_operation("mlp_up_proj"):
            raise ValueError(
                f"MLP up-projection operation not supported for cluster {self._cluster_type}"
            )
        operator = _get_operator_spec_by_name(FFN_FAMILY, "mlp_up_proj")
        effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
        raw_time = self._predictions[operator.profiling_name()][(effective_tokens,)]
        if getattr(batch, "num_prefill_tokens", 0) > 0:
            prefill_phase_scale = self._get_optional_operator_phase_calibration_scale(
                operator,
                "prefill_phase",
            )
            if prefill_phase_scale is not None:
                return raw_time * prefill_phase_scale
        scale = self._get_operator_calibration_scale(operator)
        return raw_time * scale

    def _get_mlp_layer_down_proj_execution_time(self, batch: Batch) -> float:
        if not self._supports_operation("mlp_down_proj"):
            raise ValueError(
                f"MLP down-projection operation not supported for cluster {self._cluster_type}"
            )
        operator = _get_operator_spec_by_name(FFN_FAMILY, "mlp_down_proj")
        effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
        raw_time = self._predictions[operator.profiling_name()][(effective_tokens,)]
        decode_phase_scale = None
        if getattr(batch, "num_prefill_tokens", 0) == 0:
            decode_phase_scale = self._get_optional_operator_phase_calibration_scale(
                operator,
                "decode_phase",
            )
        if decode_phase_scale is not None:
            return raw_time * decode_phase_scale
        scale = self._get_operator_calibration_scale(operator)
        return raw_time * scale

    def _get_mlp_layer_act_execution_time(self, batch: Batch) -> float:
        if not self._supports_operation("mlp_act"):
            raise ValueError(
                f"MLP activation operation not supported for cluster {self._cluster_type}"
            )
        effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
        return self._predictions["mlp_act"][(effective_tokens,)]

    def _get_attn_norm_layer_act_execution_time(self, batch: Batch) -> float:
        if not self._supports_operation("input_layernorm"):
            raise ValueError(
                f"input layernorm operation not supported for cluster {self._cluster_type}"
            )
        effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
        return self._predictions["input_layernorm"][(effective_tokens,)]

    def _get_mlp_norm_layer_act_execution_time(self, batch: Batch) -> float:
        if not self._model_config.post_attn_norm:
            raise ValueError(
                f"post-attention layernorm operation not supported for cluster {self._cluster_type}"
            )
        if not self._supports_operation("post_attention_layernorm"):
            raise ValueError(
                f"post-attention layernorm operation not supported for cluster {self._cluster_type}"
            )
        effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
        return self._predictions["post_attention_layernorm"][(effective_tokens,)]

    def _get_add_layer_act_execution_time(self, batch: Batch) -> float:
        if self._model_config.uses_fused_add_norm:
            return 0.0  # add is fused into layernorm
        if not self._supports_operation("add"):
            raise ValueError(
                f"add operation not supported for cluster {self._cluster_type}"
            )
        effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
        return self._predictions["add"][(effective_tokens,)]

    def _get_named_linear_op_execution_time(
        self,
        *,
        op_name: str,
        num_tokens: int,
    ) -> float:
        if op_name not in self._predictions:
            raise ValueError(
                f"{op_name} prediction cache not found. "
                "Ensure matching profiling data is available."
            )
        key = (int(num_tokens),)
        if key not in self._predictions[op_name]:
            raise ValueError(
                f"{op_name} prediction key not found for num_tokens={num_tokens}. "
                "Ensure matching profiling rows exist."
            )
        return float(self._predictions[op_name][key])

    @staticmethod
    def _get_mtp_active_request_indices(
        metadata,
        *,
        mtp_n_predict: int,
    ) -> List[List[int]]:
        if metadata is None:
            return []
        if int(mtp_n_predict) <= 0:
            raise ValueError(
                f"mtp_n_predict must be > 0, got={mtp_n_predict!r}"
            )
        active_indices_by_block: List[List[int]] = []
        max_planned_drafts = max(
            int(value) for value in metadata.planned_draft_tokens_per_request
        )
        for block_start in range(0, max_planned_drafts, int(mtp_n_predict)):
            active_indices = [
                idx
                for idx, (planned_drafts, verify_tokens) in enumerate(
                    zip(
                        metadata.planned_draft_tokens_per_request,
                        metadata.verify_tokens_per_request,
                    )
                )
                if int(verify_tokens) > 1 and int(planned_drafts) > block_start
            ]
            if active_indices:
                active_indices_by_block.append(active_indices)
        return active_indices_by_block

    def _build_mtp_synthetic_batch(
        self,
        *,
        source_batch: Batch,
        active_indices: List[int],
        is_moe: bool,
        num_tokens: Optional[List[int]] = None,
        copy_spec_decode_metadata: bool = False,
    ) -> Batch:
        requests = [source_batch.requests[idx] for idx in active_indices]
        synthetic_batch = Batch(
            replica_id=source_batch.replica_id,
            requests=requests,
            num_tokens=(
                list(num_tokens)
                if num_tokens is not None
                else [1 for _ in active_indices]
            ),
            is_moe=bool(is_moe),
        )
        if copy_spec_decode_metadata:
            metadata = getattr(source_batch, "spec_decode_metadata", None)
            if metadata is not None:
                synthetic_batch.spec_decode_metadata = metadata.__class__(
                    method=str(metadata.method),
                    planned_draft_tokens_per_request=[
                        int(metadata.planned_draft_tokens_per_request[idx])
                        for idx in active_indices
                    ],
                    verify_tokens_per_request=[
                        int(metadata.verify_tokens_per_request[idx])
                        for idx in active_indices
                    ],
                    accepted_draft_tokens_per_request=[
                        int(metadata.accepted_draft_tokens_per_request[idx])
                        for idx in active_indices
                    ],
                    rejected_draft_tokens_per_request=[
                        int(metadata.rejected_draft_tokens_per_request[idx])
                        for idx in active_indices
                    ],
                    committed_tokens_per_request=[
                        int(metadata.committed_tokens_per_request[idx])
                        for idx in active_indices
                    ],
                    uses_lookahead_slots=bool(metadata.uses_lookahead_slots),
                    terminal_overshoot_planned_draft_tokens_per_request=[
                        []
                        for _ in active_indices
                    ],
                    terminal_overshoot_verify_tokens_per_request=[
                        []
                        for _ in active_indices
                    ],
                    terminal_overshoot_accepted_draft_tokens_per_request=[
                        []
                        for _ in active_indices
                    ],
                    terminal_overshoot_rejected_draft_tokens_per_request=[
                        []
                        for _ in active_indices
                    ],
                    terminal_overshoot_raw_committed_tokens_per_request=[
                        []
                        for _ in active_indices
                    ],
                )
        return synthetic_batch

    @staticmethod
    def _copy_request_after_terminal_progress(
        request: Any,
        committed_tokens: int,
    ) -> Any:
        request_copy = copy.copy(request)
        processed_tokens = int(getattr(request_copy, "_num_processed_tokens", 0))
        total_tokens = int(getattr(request_copy, "total_tokens", processed_tokens))
        committed_tokens_int = max(int(committed_tokens), 0)
        if processed_tokens > total_tokens:
            request_copy._num_processed_tokens = (
                processed_tokens + committed_tokens_int
            )
        else:
            request_copy._num_processed_tokens = min(
                processed_tokens + committed_tokens_int,
                total_tokens,
            )
        current_decode_token_index = int(
            getattr(request_copy, "_current_decode_token_index", 0)
        )
        request_copy._current_decode_token_index = current_decode_token_index + (
            committed_tokens_int
        )
        return request_copy

    def _get_mtp_terminal_overshoot_time(
        self,
        batch: Batch,
        *,
        stage_id: int,
        cluster_type: ClusterType,
        num_layers: int,
        layer_id: int,
    ) -> float:
        if getattr(self, "_suppress_mtp_terminal_overshoot_overhead", False):
            return 0.0
        metadata = getattr(batch, "spec_decode_metadata", None)
        if metadata is None:
            return 0.0
        metadata.validate(len(batch.requests))

        planned_rows = getattr(
            metadata,
            "terminal_overshoot_planned_draft_tokens_per_request",
            None,
        )
        verify_rows = getattr(
            metadata,
            "terminal_overshoot_verify_tokens_per_request",
            None,
        )
        accepted_rows = getattr(
            metadata,
            "terminal_overshoot_accepted_draft_tokens_per_request",
            None,
        )
        rejected_rows = getattr(
            metadata,
            "terminal_overshoot_rejected_draft_tokens_per_request",
            None,
        )
        raw_committed_rows = getattr(
            metadata,
            "terminal_overshoot_raw_committed_tokens_per_request",
            None,
        )
        nested_vectors = [
            planned_rows,
            verify_rows,
            accepted_rows,
            rejected_rows,
            raw_committed_rows,
        ]
        if all(values is None for values in nested_vectors):
            return 0.0
        if any(values is None for values in nested_vectors):
            raise ValueError(
                "MTP terminal overshoot metadata must provide all nested token vectors"
            )

        max_terminal_rows = max(len(rows) for rows in planned_rows)
        if max_terminal_rows <= 0:
            return 0.0

        total_time_ms = 0.0
        previous_suppression = getattr(
            self,
            "_suppress_mtp_terminal_overshoot_overhead",
            False,
        )
        self._suppress_mtp_terminal_overshoot_overhead = True
        try:
            for terminal_row_index in range(max_terminal_rows):
                active_indices = [
                    idx
                    for idx, rows in enumerate(planned_rows)
                    if terminal_row_index < len(rows)
                ]
                if not active_indices:
                    continue

                terminal_requests = [
                    self._copy_request_after_terminal_progress(
                        batch.requests[idx],
                        int(metadata.committed_tokens_per_request[idx]),
                    )
                    for idx in active_indices
                ]
                terminal_verify_tokens = [
                    int(verify_rows[idx][terminal_row_index])
                    for idx in active_indices
                ]
                terminal_batch = Batch(
                    replica_id=batch.replica_id,
                    requests=terminal_requests,
                    num_tokens=terminal_verify_tokens,
                    is_moe=bool(batch.is_moe),
                )
                terminal_batch.spec_decode_metadata = metadata.__class__(
                    method=str(metadata.method),
                    planned_draft_tokens_per_request=[
                        int(planned_rows[idx][terminal_row_index])
                        for idx in active_indices
                    ],
                    verify_tokens_per_request=terminal_verify_tokens,
                    accepted_draft_tokens_per_request=[
                        int(accepted_rows[idx][terminal_row_index])
                        for idx in active_indices
                    ],
                    rejected_draft_tokens_per_request=[
                        int(rejected_rows[idx][terminal_row_index])
                        for idx in active_indices
                    ],
                    committed_tokens_per_request=[
                        0
                        for _ in active_indices
                    ],
                    uses_lookahead_slots=bool(metadata.uses_lookahead_slots),
                    terminal_overshoot_planned_draft_tokens_per_request=[
                        []
                        for _ in active_indices
                    ],
                    terminal_overshoot_verify_tokens_per_request=[
                        []
                        for _ in active_indices
                    ],
                    terminal_overshoot_accepted_draft_tokens_per_request=[
                        []
                        for _ in active_indices
                    ],
                    terminal_overshoot_rejected_draft_tokens_per_request=[
                        []
                        for _ in active_indices
                    ],
                    terminal_overshoot_raw_committed_tokens_per_request=[
                        []
                        for _ in active_indices
                    ],
                )
                terminal_execution_time = self.predict_stage_execution_time(
                    batch=terminal_batch,
                    stage_id=stage_id,
                    cluster_type=cluster_type,
                    num_layers=num_layers,
                    layer_id=layer_id,
                )
                total_time_ms += terminal_execution_time.total_time * 1e3
        finally:
            self._suppress_mtp_terminal_overshoot_overhead = previous_suppression

        return total_time_ms

    def _predict_mtp_decoder_layer_time_ms(
        self,
        *,
        predictor: "BaseExecutionTimePredictor",
        batch: Batch,
    ) -> float:
        execution_time = predictor.predict_stage_execution_time(
            batch=batch,
            stage_id=0,
            cluster_type=self._cluster_type,
            num_layers=1,
        )
        return float(execution_time.model_time_ms)

    def _predict_mtp_structural_step_time_ms(
        self,
        *,
        predictor: "BaseExecutionTimePredictor",
        contract,
        synthetic_batch: Batch,
        active_request_count: int,
    ) -> float:
        total_forward_tokens = int(sum(int(token_count) for token_count in synthetic_batch.num_tokens))
        active_tokens = int(active_request_count)
        if total_forward_tokens <= 0:
            raise ValueError(
                "target-embedded MTP structural replay requires positive forward token count, "
                f"got synthetic_batch.num_tokens={synthetic_batch.num_tokens!r}"
            )
        hidden_dim = int(predictor._model_config.embedding_dim)
        attn_tp_size = int(contract.attn_tp_size)
        step_time_ms = 0.0

        step_time_ms += predictor._get_named_linear_op_execution_time(
            op_name="emb",
            num_tokens=total_forward_tokens,
        )
        for _ in range(int(contract.num_pre_fusion_norms)):
            step_time_ms += predictor._get_named_linear_op_execution_time(
                op_name=str(contract.norm_op_name),
                num_tokens=total_forward_tokens,
            )
        step_time_ms += predictor._get_named_linear_op_execution_time(
            op_name=contract.fusion_op_name,
            num_tokens=total_forward_tokens,
        )
        step_time_ms += self._predict_mtp_decoder_layer_time_ms(
            predictor=predictor,
            batch=synthetic_batch,
        )
        for _ in range(int(contract.num_post_decoder_norms)):
            step_time_ms += predictor._get_named_linear_op_execution_time(
                op_name=str(contract.norm_op_name),
                num_tokens=total_forward_tokens,
            )
        step_time_ms += predictor._get_named_linear_op_execution_time(
            op_name=contract.lm_head_op_name,
            num_tokens=active_tokens,
        )

        if contract.embedding_requires_allreduce and attn_tp_size > 1:
            emb_bytes = hidden_dim * 2 * total_forward_tokens
            step_time_ms += predictor.predict_allreduce_time(
                data_size_bytes=emb_bytes,
                num_devices=attn_tp_size,
                cluster_type=self._cluster_type,
                comm_domain="ATTN_TP",
            )

        if contract.fusion_requires_allgather and attn_tp_size > 1:
            per_device_fusion_bytes = (hidden_dim * 2 * total_forward_tokens) // attn_tp_size
            step_time_ms += predictor.predict_allgather_time(
                data_size_bytes=per_device_fusion_bytes,
                num_devices=attn_tp_size,
                cluster_type=self._cluster_type,
                comm_domain="ATTN_TP",
            )

        if contract.lm_head_requires_allgather and attn_tp_size > 1:
            vocab_size = int(predictor._model_config.vocab_size)
            if vocab_size % attn_tp_size != 0:
                raise ValueError(
                    "lm_head all-gather requires vocab_size divisible by attn_tp_size, "
                    f"got vocab_size={vocab_size}, attn_tp_size={attn_tp_size}"
                )
            per_device_logits_bytes = (vocab_size // attn_tp_size) * 2 * active_tokens
            step_time_ms += predictor.predict_allgather_time(
                data_size_bytes=per_device_logits_bytes,
                num_devices=attn_tp_size,
                cluster_type=self._cluster_type,
                comm_domain="ATTN_TP",
            )

        return step_time_ms

    def _discover_mtp_model_training_file_paths(
        self,
        *,
        model_name: str,
    ) -> Dict[str, str]:
        def _existing_overhead_input_paths() -> Dict[str, str]:
            input_attrs = {
                "cpu_overhead_input_file": "_cpu_overhead_input_file",
                "pp_stage_boundary_input_file": "_pp_stage_boundary_input_file",
                "pp_receiver_head_input_file": "_pp_receiver_head_input_file",
                "pp_producer_send_path_input_file": "_pp_producer_send_path_input_file",
                "pp_prefill_consumer_active_input_file": (
                    "_pp_prefill_consumer_active_input_file"
                ),
            }
            existing_paths = {}
            for key, attr_name in input_attrs.items():
                input_path = str(getattr(self, attr_name, ""))
                if input_path and os.path.exists(input_path):
                    existing_paths[key] = input_path
            return existing_paths

        current_model_name = str(getattr(self._replica_config, "model_name", ""))
        if str(model_name) == current_model_name:
            compute_input_file = str(getattr(self, "_compute_input_file", ""))
            attention_input_file = str(getattr(self, "_attention_input_file", ""))
            moe_input_file = str(getattr(self, "_moe_input_file", ""))
            if not compute_input_file or not os.path.exists(compute_input_file):
                raise ValueError(
                    "MTP structural proposer cannot reuse current compute "
                    f"profiling file for model={model_name!r}: "
                    f"{compute_input_file!r}"
                )
            if not attention_input_file or not os.path.exists(attention_input_file):
                raise ValueError(
                    "MTP structural proposer cannot reuse current attention "
                    f"profiling file for model={model_name!r}: "
                    f"{attention_input_file!r}"
                )
            training_file_paths = {
                "compute_input_file": compute_input_file,
                "attention_input_file": attention_input_file,
                "moe_input_file": moe_input_file if os.path.exists(moe_input_file) else "",
            }
            training_file_paths.update(_existing_overhead_input_paths())
            return training_file_paths

        model_root = os.path.join(
            "data",
            "profiling",
            "compute",
            self._replica_config.device,
            model_name,
        )
        linear_op_input_file = os.path.join(model_root, "linear_op.csv")
        legacy_linear_op_input_file = os.path.join(model_root, "mlp.csv")
        if not os.path.exists(linear_op_input_file) and os.path.exists(
            legacy_linear_op_input_file
        ):
            linear_op_input_file = legacy_linear_op_input_file
        attention_input_file = os.path.join(model_root, "attention.csv")
        moe_input_file = os.path.join(model_root, "moe.csv")
        if not os.path.exists(linear_op_input_file):
            raise ValueError(
                f"MTP structural proposer compute profiling file not found for model={model_name!r}: "
                f"{linear_op_input_file!r}"
            )
        if not os.path.exists(attention_input_file):
            raise ValueError(
                f"MTP structural proposer attention profiling file not found for model={model_name!r}: "
                f"{attention_input_file!r}"
            )
        training_file_paths = {
            "compute_input_file": linear_op_input_file,
            "attention_input_file": attention_input_file,
            "moe_input_file": moe_input_file if os.path.exists(moe_input_file) else "",
        }
        training_file_paths.update(_existing_overhead_input_paths())
        return training_file_paths

    def _get_or_create_mtp_secondary_predictor(self, *, contract):
        cache_key = str(contract.proposer_model_name)
        if cache_key in self._mtp_secondary_predictors:
            return self._mtp_secondary_predictors[cache_key]

        from types import SimpleNamespace

        from frontier.config.config import SpeculativeDecodingConfig
        from frontier.execution_time_predictor.execution_time_predictor_registry import (
            ExecutionTimePredictorRegistry,
        )

        disabled_spec_config = SpeculativeDecodingConfig(enabled=False)
        proposer_model_config = load_mtp_structural_model_config(
            str(contract.proposer_model_name)
        )
        total_expert_num = int(
            getattr(
                self._replica_config,
                "total_expert_num",
                getattr(proposer_model_config, "num_experts", 1),
            )
            or getattr(proposer_model_config, "num_experts", 1)
            or 1
        )
        moe_ep_size = int(getattr(self._replica_config, "moe_expert_parallel_size", 1))
        local_expert_num = getattr(self._replica_config, "local_expert_num", None)
        if local_expert_num is None and total_expert_num > 1 and moe_ep_size > 0:
            local_expert_num = total_expert_num // moe_ep_size
        secondary_replica_config = SimpleNamespace(
            model_name=str(contract.proposer_model_name),
            model_config=proposer_model_config,
            speculative_decoding_config=disabled_spec_config,
            num_pipeline_stages=1,
            attn_tensor_parallel_size=int(self._replica_config.attn_tensor_parallel_size),
            attn_data_parallel_size=int(getattr(self._replica_config, "attn_data_parallel_size", 1)),
            moe_tensor_parallel_size=int(getattr(self._replica_config, "moe_tensor_parallel_size", 1)),
            moe_expert_parallel_size=moe_ep_size,
            total_expert_num=total_expert_num,
            local_expert_num=local_expert_num,
            router_load_balancing_type=getattr(
                self._replica_config,
                "router_load_balancing_type",
                "None",
            ),
            router_topk=int(
                getattr(
                    self._replica_config,
                    "router_topk",
                    getattr(proposer_model_config, "num_experts_per_tok", 1),
                )
                or getattr(proposer_model_config, "num_experts_per_tok", 1)
                or 1
            ),
            moe_routing_mode=getattr(
                self._replica_config,
                "moe_routing_mode",
                "simulation",
            ),
            moe_routing_seed=int(
                getattr(self._replica_config, "moe_routing_seed", 42)
            ),
            extend_ep_across_dp=bool(
                getattr(self._replica_config, "extend_ep_across_dp", False)
            ),
            data_parallel_size=int(
                getattr(self._replica_config, "data_parallel_size", 1) or 1
            ),
            cluster_prefix=getattr(self._replica_config, "cluster_prefix", None),
            requires_mtp_structural_compute_models=True,
            suppress_spec_decode_proposer_overhead=True,
            node_config=getattr(self._replica_config, "node_config", None),
            device=str(getattr(self._replica_config, "device", "")),
            network_device=str(getattr(self._replica_config, "network_device", "")),
        )
        scheduler_config = SimpleNamespace(
            get_type=lambda: self._replica_scheduler_provider,
            block_size=self._block_size,
            max_tokens_in_batch=getattr(self._config, "prediction_max_tokens_per_request", 0),
        )
        metrics_config = SimpleNamespace(cache_dir=self._cache_dir)
        predictor_type = str(self._config.get_type())
        training_file_paths = self._discover_mtp_model_training_file_paths(
            model_name=str(contract.proposer_model_name),
        )
        predictor = ExecutionTimePredictorRegistry.get(
            predictor_type=predictor_type,
            predictor_config=self._config,
            replica_config=secondary_replica_config,
            replica_scheduler_config=scheduler_config,
            metrics_config=metrics_config,
            cluster_type=self._cluster_type,
            training_file_paths=training_file_paths,
            cc_backend=self._cc_backend,
        )
        self._mtp_secondary_predictors[cache_key] = predictor
        return predictor

    def _get_structural_mtp_proposer_time(
        self,
        batch: Batch,
        *,
        method_name: str,
    ) -> float:
        metadata = getattr(batch, "spec_decode_metadata", None)
        if metadata is None:
            return 0.0
        metadata.validate(len(batch.requests))
        spec_config = getattr(self._replica_config, "speculative_decoding_config", None)
        if spec_config is None:
            raise ValueError("Speculative decoding config is not initialized")

        contract = build_mtp_runtime_contract(
            method=method_name,
            target_model_name=str(self._replica_config.model_name),
            spec_model_name=str(getattr(spec_config, "spec_model_name", "")),
            attn_tp_size=int(self._replica_config.attn_tensor_parallel_size),
            mtp_n_predict=int(spec_config.mtp_n_predict),
            mtp_num_layers=int(spec_config.mtp_num_layers),
        )
        active_indices_by_block = self._get_mtp_active_request_indices(
            metadata,
            mtp_n_predict=contract.mtp_n_predict,
        )
        if not active_indices_by_block:
            return 0.0

        if str(contract.proposer_model_name) != str(self._replica_config.model_name):
            raise ValueError(
                "draft-model MTP structural proposer path is not implemented yet; "
                "provide decode_draft_proposer_latency_profile_file for now"
            )

        predictor = (
            self
            if (
                str(contract.proposer_model_name) == str(self._replica_config.model_name)
                and int(getattr(self._replica_config, "num_pipeline_stages", 1)) == 1
            )
            else self._get_or_create_mtp_secondary_predictor(contract=contract)
        )
        total_time_ms = 0.0
        for block_index, active_indices in enumerate(active_indices_by_block):
            predictor_model_config = getattr(predictor, "_model_config", None)
            use_verify_window_shape = block_index == 0
            verify_step_num_tokens = None
            if use_verify_window_shape:
                verify_step_num_tokens = []
                for idx in active_indices:
                    step_tokens = int(metadata.verify_tokens_per_request[idx]) - int(
                        metadata.rejected_draft_tokens_per_request[idx]
                    )
                    if step_tokens <= 0:
                        raise ValueError(
                            "target-embedded MTP structural replay requires positive "
                            "pruned verify tokens per request, "
                            f"got verify_tokens={metadata.verify_tokens_per_request[idx]!r}, "
                            f"rejected_tokens={metadata.rejected_draft_tokens_per_request[idx]!r}, "
                            f"request_index={idx}"
                        )
                    verify_step_num_tokens.append(step_tokens)
            synthetic_batch = self._build_mtp_synthetic_batch(
                source_batch=batch,
                active_indices=active_indices,
                is_moe=bool(
                    getattr(
                        predictor_model_config,
                        "is_moe",
                        getattr(batch, "is_moe", False),
                    )
                ),
                num_tokens=verify_step_num_tokens,
                copy_spec_decode_metadata=use_verify_window_shape,
            )
            total_time_ms += self._predict_mtp_structural_step_time_ms(
                predictor=predictor,
                contract=contract,
                synthetic_batch=synthetic_batch,
                active_request_count=len(active_indices),
            )
        return total_time_ms

    # ========== Architecture-profile attention extra operation getters ==========

    def _get_attn_inter_norm_execution_time(self, batch: Batch) -> float:
        """Get architecture-profile inter_norm execution time."""
        if not self._supports_operation("attn_inter_norm"):
            raise ValueError(
                f"attn_inter_norm operation not supported for cluster {self._cluster_type} "
                f"(requires matching model architecture profile)"
            )
        if "attn_inter_norm" not in self._predictions:
            raise ValueError(
                f"attn_inter_norm prediction cache not found. "
                f"Ensure architecture-profile profiling data is available."
            )
        effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
        return self._predictions["attn_inter_norm"][(effective_tokens,)]

    def _get_attn_wq_proj_execution_time(self, batch: Batch) -> float:
        """Get architecture-profile WQ projection execution time."""
        if not self._supports_operation("attn_wq_proj"):
            raise ValueError(
                f"attn_wq_proj operation not supported for cluster {self._cluster_type} "
                f"(requires matching model architecture profile)"
            )
        if "attn_wq_proj" not in self._predictions:
            raise ValueError(
                f"attn_wq_proj prediction cache not found. "
                f"Ensure architecture-profile profiling data is available."
            )
        effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
        return self._predictions["attn_wq_proj"][(effective_tokens,)]

    def _get_share_expert_up_proj_execution_time(self, batch: Batch) -> float:
        """Get shared-expert up projection execution time."""
        if not self._supports_operation("share_expert_up_proj"):
            raise ValueError(
                f"share_expert_up_proj operation not supported for cluster {self._cluster_type} "
                f"(requires shared-expert MoE profile)"
            )
        if "share_expert_up_proj" not in self._predictions:
            raise ValueError(
                f"share_expert_up_proj prediction cache not found. "
                f"Ensure shared-expert MoE profiling data is available."
            )
        effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
        return self._predictions["share_expert_up_proj"][(effective_tokens,)]

    def _get_share_expert_down_proj_execution_time(self, batch: Batch) -> float:
        """Get shared-expert down projection execution time."""
        if not self._supports_operation("share_expert_down_proj"):
            raise ValueError(
                f"share_expert_down_proj operation not supported for cluster {self._cluster_type} "
                f"(requires shared-expert MoE profile)"
            )
        if "share_expert_down_proj" not in self._predictions:
            raise ValueError(
                f"share_expert_down_proj prediction cache not found. "
                f"Ensure shared-expert MoE profiling data is available."
            )
        effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
        return self._predictions["share_expert_down_proj"][(effective_tokens,)]

    def _get_share_expert_act_execution_time(self, batch: Batch) -> float:
        """Get shared-expert activation execution time."""
        if not self._supports_operation("share_expert_act"):
            raise ValueError(
                f"share_expert_act operation not supported for cluster {self._cluster_type} "
                f"(requires shared-expert MoE profile)"
            )
        if "share_expert_act" not in self._predictions:
            raise ValueError(
                f"share_expert_act prediction cache not found. "
                f"Ensure shared-expert MoE profiling data is available."
            )
        effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
        return self._predictions["share_expert_act"][(effective_tokens,)]

    def _validate_prediction_value(
        self, value: float, operation_name: str, batch: Batch, context: str = ""
    ) -> float:
        """
        Validate that a predicted execution time value is valid (not NaN, not Inf, not negative).

        Args:
            value: The predicted execution time value
            operation_name: Name of the operation being predicted
            batch: The batch being processed
            context: Additional context for error messages

        Returns:
            The validated value

        Raises:
            ValueError: If the value is invalid
        """
        if math.isnan(value):
            error_msg = (
                f"[EXEC_TIME_ERROR] NaN prediction detected!\n"
                f"  Operation: {operation_name}\n"
                f"  Batch ID: {batch.id}\n"
                f"  Batch size: {batch.size}\n"
                f"  Num tokens: {batch.num_tokens}\n"
                f"  Context: {context}\n"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if math.isinf(value):
            error_msg = (
                f"[EXEC_TIME_ERROR] Infinite prediction detected!\n"
                f"  Operation: {operation_name}\n"
                f"  Batch ID: {batch.id}\n"
                f"  Batch size: {batch.size}\n"
                f"  Num tokens: {batch.num_tokens}\n"
                f"  Value: {value}\n"
                f"  Context: {context}\n"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if value < 0:
            error_msg = (
                f"[EXEC_TIME_ERROR] Negative prediction detected!\n"
                f"  Operation: {operation_name}\n"
                f"  Batch ID: {batch.id}\n"
                f"  Batch size: {batch.size}\n"
                f"  Num tokens: {batch.num_tokens}\n"
                f"  Value: {value}\n"
                f"  Context: {context}\n"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Log the valid prediction for debugging
        logger.debug(
            f"[EXEC_TIME_VALID] {operation_name}: {value:.6f}ms "
            f"(batch_id={batch.id}, tokens={batch.num_tokens})"
        )

        return value

    def _get_tensor_parallel_communication_time(self, batch: Batch) -> float:
        if not self._supports_operation("tensor_parallel_communication"):
            raise ValueError(
                f"tensor parallel communication operation not supported for cluster {self._cluster_type}"
            )

        # When CC Backend is not available, require explicit dummy mode
        if self._enable_dummy_mode:
            logger.info(
                f"[CC-FALLBACK] _get_tensor_parallel_communication_time: CC Backend not available, "
                f"falling back to dummy mode value={self._dummy_execution_time} ms"
            )
            return self._dummy_execution_time

        tp_size = self._get_tensor_parallel_size_for_comm()

        # Use CC Backend if available for communication predictions
        if self._cc_backend is not None:
            # Calculate data size for all-reduce: embedding_dim * 2 bytes (FP16) * num_tokens
            # Use compute-effective tokens so AFD CUDA Graph padding is reflected
            # when present, while non-CUDA-Graph paths keep exact token counts.
            # Aligned with StepFun-vLLM hidden_states communication volume semantics.
            effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
            data_size_bytes = self._model_config.embedding_dim * 2 * effective_tokens
            quant_manager = get_quantization_manager()
            data_size_bytes = quant_manager.adjust_tensor_size(
                "allreduce", data_size_bytes, self._cluster_type
            )
            num_devices = tp_size

            result = self._cc_backend.predict_allreduce(
                data_size_bytes=data_size_bytes,
                num_devices=num_devices,
                cluster_type=self._cluster_type,
                comm_domain="ATTN_TP",
            )
            result = self._strip_collective_sim_allreduce_launch_overhead_if_needed(
                batch=batch,
                predicted_ms=result,
                num_devices=num_devices,
                comm_domain="ATTN_TP",
            )
            logger.debug(
                f"_get_tensor_parallel_communication_time: using CC Backend, "
                f"data_size={data_size_bytes}, num_devices={num_devices}, result={result:.6f} ms"
            )
            return result

        # Fail fast: CC Backend is required for tensor parallel communication predictions
        # unless dummy mode is explicitly enabled
        raise RuntimeError(
            f"CC Backend is required for tensor parallel communication prediction "
            f"but was not provided. Either:\n"
            f"  1. Configure a CC Backend (e.g., --cc_backend vidur or --cc_backend analytical)\n"
            f"  2. Enable dummy mode explicitly (--enable_dummy_mode)\n"
            f"Current state: cc_backend=None, enable_dummy_mode={self._enable_dummy_mode}"
        )

    def _get_pipeline_parallel_communication_time(self, batch: Batch) -> float:
        if not self._supports_operation("pipeline_parallel_communication"):
            raise ValueError(
                f"pipeline parallel communication operation not supported for cluster {self._cluster_type}"
            )

        # Use CC Backend if available for communication predictions
        if self._cc_backend is not None:
            # Calculate data size for send/recv: embedding_dim * 2 bytes (FP16) * num_tokens
            # Use compute-effective tokens so AFD CUDA Graph padding is reflected
            # when present, while non-CUDA-Graph paths keep exact token counts.
            # Aligned with StepFun-vLLM PP communication volume semantics.
            effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
            data_size_bytes = (
                self._model_config.embedding_dim * 2 * effective_tokens
            )
            quant_manager = get_quantization_manager()
            data_size_bytes = quant_manager.adjust_tensor_size(
                "send_recv", data_size_bytes, self._cluster_type
            )

            result = self._cc_backend.predict_send_recv(
                data_size_bytes=data_size_bytes,
                cluster_type=self._cluster_type,
                comm_domain="PP",
            )
            logger.debug(
                f"_get_pipeline_parallel_communication_time: using CC Backend, "
                f"data_size={data_size_bytes}, result={result:.6f} ms"
            )
            return result

        # When CC Backend is not available, require explicit dummy mode
        if self._enable_dummy_mode:
            logger.info(
                f"[CC-FALLBACK] _get_pipeline_parallel_communication_time: CC Backend not available, "
                f"falling back to dummy mode value={self._dummy_execution_time} ms"
            )
            return self._dummy_execution_time

        # Fail fast: CC Backend is required for pipeline parallel communication predictions
        # unless dummy mode is explicitly enabled
        raise RuntimeError(
            f"CC Backend is required for pipeline parallel communication prediction "
            f"but was not provided. Either:\n"
            f"  1. Configure a CC Backend (e.g., --cc_backend vidur or --cc_backend analytical)\n"
            f"  2. Enable dummy mode explicitly (--enable_dummy_mode)\n"
            f"Current state: cc_backend=None, enable_dummy_mode={self._enable_dummy_mode}"
        )

    def _predict_comm_operator(
        self,
        operator: CommOperatorSpec,
        batch: Batch,
    ) -> float:
        """Predict one first-class communication operator through the thin CC wrappers."""

        ctx = CommPayloadContext(
            batch=batch,
            model_config=self._model_config,
            replica_config=self._replica_config,
            cluster_type=self._cluster_type,
            quantization_manager=get_quantization_manager(),
        )
        data_size_bytes = operator.build_payload_bytes(ctx)
        num_devices = operator.num_devices(ctx)

        if operator.collective_alias == "allreduce":
            if num_devices is None:
                raise ValueError(
                    f"CommOperator {operator.name} requires num_devices for allreduce"
                )
            predicted_ms = self.predict_allreduce_time(
                data_size_bytes=data_size_bytes,
                num_devices=num_devices,
                cluster_type=self._cluster_type,
                comm_domain=operator.comm_domain,
            )
            if operator.apply_allreduce_launch_overhead_strip:
                if operator.comm_domain is None:
                    raise ValueError(
                        f"CommOperator {operator.name} requires comm_domain for "
                        "allreduce launch-overhead stripping"
                    )
                predicted_ms = (
                    self._strip_collective_sim_allreduce_launch_overhead_if_needed(
                        batch=batch,
                        predicted_ms=predicted_ms,
                        num_devices=num_devices,
                        comm_domain=operator.comm_domain,
                    )
                )
            return predicted_ms

        if operator.collective_alias == "allgather":
            if num_devices is None:
                raise ValueError(
                    f"CommOperator {operator.name} requires num_devices for allgather"
                )
            return self.predict_allgather_time(
                data_size_bytes=data_size_bytes,
                num_devices=num_devices,
                cluster_type=self._cluster_type,
                comm_domain=operator.comm_domain,
            )

        if operator.collective_alias == "alltoall":
            if num_devices is None:
                raise ValueError(
                    f"CommOperator {operator.name} requires num_devices for alltoall"
                )
            return self.predict_alltoall_time(
                data_size_bytes=data_size_bytes,
                num_devices=num_devices,
                cluster_type=self._cluster_type,
                comm_domain=operator.comm_domain,
            )

        if operator.collective_alias == "send_recv":
            return self.predict_p2p_time(
                data_size_bytes=data_size_bytes,
                cluster_type=self._cluster_type,
                comm_domain=operator.comm_domain,
            )

        raise ValueError(
            f"Unsupported communication collective alias for {operator.name}: "
            f"{operator.collective_alias}"
        )

    # Expert-parallel communication timing is modeled through COMM family operators.

    def _get_attention_rope_execution_time(self, batch: Batch) -> float:
        if not self._supports_operation("attn_rope"):
            raise ValueError(
                f"attention rope operation not supported for cluster {self._cluster_type}"
            )
        if "attn_rope" not in self._predictions:
            raise ValueError(
                f"attention rope prediction cache not found for cluster {self._cluster_type}"
            )
        effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
        return self._predictions["attn_rope"][(effective_tokens,)]

    def _get_attention_kv_cache_save_execution_time(self, batch: Batch) -> float:
        cache_write_op_name = self._dense_attention_cache_write_op_name()
        if not self._supports_operation(cache_write_op_name):
            raise ValueError(
                f"attention kv cache save operation not supported for cluster {self._cluster_type}"
            )
        if cache_write_op_name not in self._predictions:
            raise ValueError(
                f"attention kv cache save prediction cache not found for cluster {self._cluster_type}"
            )
        prediction_info = self._predictions[cache_write_op_name]
        if isinstance(prediction_info, dict) and prediction_info.get(
            "_on_demand_prediction", False
        ):
            total_tokens = batch.total_num_tokens
            batch_size = len(batch.requests)
            kv_cache_size = 0
            if batch.num_decode_tokens > 0:
                _, kv_cache_size = self._get_batch_decode_attention_params(batch)
            features = {
                "total_tokens": total_tokens,
                "kv_cache_size": kv_cache_size,
                "batch_size": batch_size,
            }
            raw_time = self._get_on_demand_prediction(cache_write_op_name, features)
            if getattr(batch, "num_prefill_tokens", 0) > 0:
                prefill_phase_scale = self._get_optional_calibration_scale(
                    "_prefill_phase_attn_kv_cache_save_calibration_scale",
                    "prefill_phase_attn_kv_cache_save_calibration_scale",
                )
                if prefill_phase_scale is not None:
                    return raw_time * prefill_phase_scale
            return raw_time * self._get_calibration_scale(
                "_attn_kv_cache_save_calibration_scale",
                "attn_kv_cache_save_calibration_scale",
            )

        # don't use round up to the nearest multiple of 8 here, because we want to
        # predict the execution time for the exact number of tokens
        num_tokens = batch.total_num_tokens
        raw_time = prediction_info[(num_tokens,)]
        if getattr(batch, "num_prefill_tokens", 0) > 0:
            prefill_phase_scale = self._get_optional_calibration_scale(
                "_prefill_phase_attn_kv_cache_save_calibration_scale",
                "prefill_phase_attn_kv_cache_save_calibration_scale",
            )
            if prefill_phase_scale is not None:
                return raw_time * prefill_phase_scale
        return raw_time * self._get_calibration_scale(
            "_attn_kv_cache_save_calibration_scale",
            "attn_kv_cache_save_calibration_scale",
        )

    def _get_attention_decode_execution_time(self, batch: Batch) -> float:
        (
            decode_batch_size,
            decode_avg_kv_cache_size,
        ) = self._get_batch_decode_attention_params(batch)
        if decode_batch_size == 0:
            return 0.0

        if batch.num_prefill_tokens > 0:
            if self._cluster_type != ClusterType.MONOLITHIC:
                raise ValueError(
                    "True mixed prefill+decode batches are only supported in co-location "
                    f"(ClusterType.MONOLITHIC), but got cluster={self._cluster_type}."
                )
            if "attn_decode_in_mixed" not in self._predictions:
                raise ValueError(
                    "attn_decode_in_mixed prediction is required for true mixed batches "
                    f"but not found for cluster {self._cluster_type}. "
                    "Please provide merged attention profiling data via atten_input_file "
                    "(Option A) and train attn_decode_in_mixed."
                )
            features = self._get_batch_decode_mixed_features(batch)
            raw_time = self._get_on_demand_prediction("attn_decode_in_mixed", features)
            mixed_scale = self._get_optional_calibration_scale(
                "_attn_decode_in_mixed_calibration_scale",
                "attn_decode_in_mixed_calibration_scale",
            )
            if mixed_scale is not None:
                return raw_time * mixed_scale
            return raw_time

        decode_op_name = self._dense_attention_decode_op_name()
        if not self._supports_operation(decode_op_name):
            raise ValueError(
                f"attention decode operation not supported for cluster {self._cluster_type}"
            )
        if decode_op_name not in self._predictions:
            raise ValueError(
                f"attention decode prediction cache not found for cluster {self._cluster_type}"
            )

        raw_time = self._predictions[decode_op_name][
            (decode_batch_size, decode_avg_kv_cache_size)
        ] * (
            1
            + self._attention_decode_batching_overhead_fraction
            * int(decode_batch_size > 1)
        )
        late_decode_scale = self._get_late_decode_only_calibration_scale(
            batch,
            "_late_decode_attn_decode_calibration_scale",
            "late_decode_attn_decode_calibration_scale",
        )
        if late_decode_scale is not None:
            return raw_time * late_decode_scale
        return raw_time * self._get_calibration_scale(
            "_attn_decode_calibration_scale", "attn_decode_calibration_scale"
        )

    def _get_attention_prefill_execution_time(self, batch: Batch) -> float:
        prefill_params = self._get_batch_prefill_attention_params(batch)

        if len(prefill_params) == 0:
            # Decode-only batches legitimately have no prefill requests.
            # In that case, attention prefill time should be 0 by definition.
            if batch.num_prefill_tokens == 0:
                return 0.0
            raise ValueError(
                f"no prefill parameters found for batch {batch.id} "
                f"(num_prefill_tokens={batch.num_prefill_tokens}, num_tokens={batch.num_tokens})"
            )

        prefill_op_name = self._dense_attention_prefill_op_name()
        if not self._supports_operation(prefill_op_name):
            raise ValueError(
                f"attention prefill operation not supported for cluster {self._cluster_type}"
            )

        # Check if we should use attn_prefill_mixed for mixed-batch prediction
        # Use attn_prefill_mixed when:
        # 1. The model is available
        # 2. There are multiple prefill requests (mixed batch)
        if len(prefill_params) > 1 and "attn_prefill_mixed" in self._predictions:
            model_info = self._predictions["attn_prefill_mixed"]
            if isinstance(model_info, dict) and model_info.get("_on_demand_prediction"):
                # Extract mixed-prefill features, including live KV cache context
                features = self._get_batch_prefill_mixed_features(batch)
                return self._get_on_demand_prediction("attn_prefill_mixed", features)

        # Fall back to original attn_prefill model for single-request prefill
        if prefill_op_name not in self._predictions:
            raise ValueError(
                f"attention prefill prediction cache not found for cluster {self._cluster_type}"
            )

        kv_cache_sizes, prefill_chunk_sizes = zip(*prefill_params)

        agg_kv_cache_size = sum(kv_cache_sizes)
        agg_prefill_chunk_size = sum([x**2 for x in prefill_chunk_sizes]) ** 0.5

        return self._predictions[prefill_op_name][
            (agg_kv_cache_size, round(agg_prefill_chunk_size) ** 2)
        ] * (
            1
            + self._attention_prefill_batching_overhead_fraction
            * int(len(prefill_params) > 1)
        )

    def _get_spec_verify_attention_prefill_execution_time(self, batch: Batch) -> float:
        metadata = getattr(batch, "spec_decode_metadata", None)
        if metadata is None:
            return 0.0
        metadata.validate(len(batch.requests))

        verify_entries: List[Tuple[Any, int]] = []
        normal_decode_request_count = 0
        for request, verify_tokens in zip(
            batch.requests, metadata.verify_tokens_per_request
        ):
            if not request.is_prefill_complete:
                continue
            verify_tokens_int = int(verify_tokens)
            if verify_tokens_int <= 1:
                normal_decode_request_count += 1
                continue
            verify_entries.append((request, verify_tokens_int))

        if not verify_entries:
            return 0.0

        # In co-location, verify requests execute together as one batched
        # prefill-style kernel. That batching benefit exists both for:
        # 1. true speculative mixed batches (verify + normal decode), and
        # 2. pure verify batches with multiple verify requests.
        if (
            self._cluster_type == ClusterType.MONOLITHIC
            and len(verify_entries) > 1
        ):
            if "attn_prefill_mixed" not in self._predictions:
                raise ValueError(
                    "attn_prefill_mixed prediction is required for speculative verify "
                    "multi-request batches in co-location mode but was not found."
                )
            model_info = self._predictions["attn_prefill_mixed"]
            if not (isinstance(model_info, dict) and model_info.get("_on_demand_prediction")):
                raise ValueError(
                    "attn_prefill_mixed must be configured for on-demand prediction "
                    "when used by speculative verify multi-request batches."
                )
            features = self._get_spec_verify_prefill_mixed_features(verify_entries)
            return self._get_on_demand_prediction("attn_prefill_mixed", features)

        prefill_op_name = self._dense_attention_prefill_op_name()
        if not self._supports_operation(prefill_op_name):
            raise ValueError(
                f"attention prefill operation not supported for cluster {self._cluster_type}"
            )
        if prefill_op_name not in self._predictions:
            raise ValueError(
                f"attention prefill prediction cache not found for cluster {self._cluster_type}"
            )

        total_verify_prefill_time = 0.0
        for request, verify_tokens in verify_entries:
            kv_cache_size = (
                (
                    request.num_processed_tokens
                    + self._config.kv_cache_prediction_granularity
                    - 1
                )
                // self._config.kv_cache_prediction_granularity
            ) * self._config.kv_cache_prediction_granularity

            # attn_prefill cache uses prefill_chunk_size_squared as its second
            # dimension in this codebase. For speculative verify, query_len is
            # verify_tokens, so we map to verify_tokens**2 here.
            key = (int(kv_cache_size), int(verify_tokens**2))
            if key not in self._predictions[prefill_op_name]:
                raise ValueError(
                    "Speculative verify prefill key missing from attn_prefill cache: "
                    f"key={key}, request_id={request.id}, verify_tokens={verify_tokens}"
                )
            total_verify_prefill_time += self._predictions[prefill_op_name][key]

        return total_verify_prefill_time

    def _get_spec_verify_prefill_mixed_features(
        self, verify_entries: List[Tuple[Any, int]]
    ) -> Dict[str, float]:
        kv_cache_sizes = [request.num_processed_tokens for request, _ in verify_entries]
        seq_lens_arr = np.array([tokens for _, tokens in verify_entries], dtype=np.float64)
        batch_size = len(verify_entries)
        avg_kv_cache_size = int(np.mean(kv_cache_sizes))
        avg_kv_cache_size = (
            (
                avg_kv_cache_size
                + self._config.kv_cache_prediction_granularity
                - 1
            )
            // self._config.kv_cache_prediction_granularity
        ) * self._config.kv_cache_prediction_granularity
        total_tokens = int(seq_lens_arr.sum())
        avg_seq_len = float(seq_lens_arr.mean())
        min_seq_len = int(seq_lens_arr.min())
        max_seq_len = int(seq_lens_arr.max())
        total_tokens_squared = total_tokens**2
        seq_len_variance = float(seq_lens_arr.var()) if batch_size > 1 else 0.0
        seq_len_cv = (
            (float(seq_lens_arr.std()) / avg_seq_len) if avg_seq_len > 0 else 0.0
        )
        seq_len_range = max_seq_len - min_seq_len
        batch_variance_interaction = batch_size * seq_len_variance
        batch_cv_interaction = batch_size * seq_len_cv

        return {
            "batch_size": batch_size,
            "kv_cache_size": avg_kv_cache_size,
            "total_tokens": total_tokens,
            "avg_seq_len": avg_seq_len,
            "min_seq_len": min_seq_len,
            "max_seq_len": max_seq_len,
            "total_tokens_squared": total_tokens_squared,
            "seq_len_variance": seq_len_variance,
            "seq_len_cv": seq_len_cv,
            "seq_len_range": seq_len_range,
            "batch_variance_interaction": batch_variance_interaction,
            "batch_cv_interaction": batch_cv_interaction,
        }

    def _get_spec_normal_decode_attention_execution_time(self, batch: Batch) -> float:
        metadata = getattr(batch, "spec_decode_metadata", None)
        if metadata is None:
            return self._get_attention_decode_execution_time(batch)
        metadata.validate(len(batch.requests))

        verify_tokens_list: List[int] = []
        decode_kv_cache_sizes = []
        for request, verify_tokens in zip(
            batch.requests, metadata.verify_tokens_per_request
        ):
            if not request.is_prefill_complete:
                continue
            verify_tokens_int = int(verify_tokens)
            if verify_tokens_int > 1:
                verify_tokens_list.append(verify_tokens_int)
                continue
            decode_kv_cache_sizes.append(request.num_processed_tokens)

        if not decode_kv_cache_sizes:
            return 0.0

        if verify_tokens_list and self._cluster_type == ClusterType.MONOLITHIC:
            if "attn_decode_in_mixed" not in self._predictions:
                raise ValueError(
                    "attn_decode_in_mixed prediction is required for speculative "
                    "mixed decode batches in co-location mode but was not found."
                )
            model_info = self._predictions["attn_decode_in_mixed"]
            if not (isinstance(model_info, dict) and model_info.get("_on_demand_prediction")):
                raise ValueError(
                    "attn_decode_in_mixed must be configured for on-demand prediction "
                    "when used by speculative mixed decode batches."
                )
            features = self._get_spec_decode_mixed_features(
                decode_kv_cache_sizes=decode_kv_cache_sizes,
                verify_tokens_list=verify_tokens_list,
            )
            return self._get_on_demand_prediction("attn_decode_in_mixed", features)

        decode_op_name = self._dense_attention_decode_op_name()
        if not self._supports_operation(decode_op_name):
            raise ValueError(
                f"attention decode operation not supported for cluster {self._cluster_type}"
            )
        if decode_op_name not in self._predictions:
            raise ValueError(
                f"attention decode prediction cache not found for cluster {self._cluster_type}"
            )

        decode_batch_size = len(decode_kv_cache_sizes)
        decode_avg_kv_cache_size = int(np.mean(decode_kv_cache_sizes))
        decode_avg_kv_cache_size = (
            (
                decode_avg_kv_cache_size
                + self._config.kv_cache_prediction_granularity
                - 1
            )
            // self._config.kv_cache_prediction_granularity
        ) * self._config.kv_cache_prediction_granularity

        key = (int(decode_batch_size), int(decode_avg_kv_cache_size))
        if key not in self._predictions[decode_op_name]:
            raise ValueError(
                "Speculative decode key missing from attn_decode cache: "
                f"key={key}, decode_batch_size={decode_batch_size}"
            )

        raw_time = self._predictions[decode_op_name][key] * (
            1
            + self._attention_decode_batching_overhead_fraction
            * int(decode_batch_size > 1)
        )
        return raw_time * self._get_calibration_scale(
            "_attn_decode_calibration_scale", "attn_decode_calibration_scale"
        )

    def _get_spec_decode_mixed_features(
        self,
        *,
        decode_kv_cache_sizes: List[int],
        verify_tokens_list: List[int],
    ) -> Dict[str, float]:
        decode_batch_size = len(decode_kv_cache_sizes)
        decode_avg_kv_cache_size = int(np.mean(decode_kv_cache_sizes))
        decode_avg_kv_cache_size = (
            (
                decode_avg_kv_cache_size
                + self._config.kv_cache_prediction_granularity
                - 1
            )
            // self._config.kv_cache_prediction_granularity
        ) * self._config.kv_cache_prediction_granularity

        num_prefill_seqs = len(verify_tokens_list)
        total_prefill_tokens = int(sum(verify_tokens_list))
        total_batch_size = decode_batch_size + num_prefill_seqs
        batch_composition_ratio = (
            float(num_prefill_seqs) / float(total_batch_size)
            if total_batch_size > 0
            else 0.0
        )
        # In speculative mixed decode, normal decode requests still contribute 1 token
        # each, while verify requests contribute verify_tokens as prefill-like work.
        total_tokens = int(total_prefill_tokens + decode_batch_size)

        return {
            "decode_batch_size": decode_batch_size,
            "decode_avg_kv_cache_size": decode_avg_kv_cache_size,
            "num_prefill_seqs": num_prefill_seqs,
            "total_prefill_tokens": total_prefill_tokens,
            "total_batch_size": total_batch_size,
            "batch_composition_ratio": batch_composition_ratio,
            "total_tokens": total_tokens,
        }

    def _get_spec_decode_method_proposer_overhead_ms(self, method_name: str) -> float:
        replica_config = getattr(self, "_replica_config", None)
        spec_config = getattr(replica_config, "speculative_decoding_config", None)
        if spec_config is None:
            return 0.0

        raw_overhead_map = getattr(spec_config, "proposer_overhead_ms_by_method", {}) or {}
        if not isinstance(raw_overhead_map, dict):
            raise ValueError(
                "SpeculativeDecodingConfig.proposer_overhead_ms_by_method must be a dict, "
                f"got={type(raw_overhead_map).__name__}"
            )
        overhead_ms = float(raw_overhead_map.get(method_name, 0.0))
        if overhead_ms < 0.0:
            raise ValueError(
                "SpeculativeDecodingConfig.proposer_overhead_ms_by_method "
                f"must be >= 0, got method={method_name!r}, value={overhead_ms!r}"
            )
        return overhead_ms

    def _get_spec_decode_proposer_overhead_time(
        self,
        batch: Batch,
        *,
        method_name: str,
    ) -> float:
        metadata = getattr(batch, "spec_decode_metadata", None)
        if metadata is None:
            return 0.0
        metadata.validate(len(batch.requests))

        speculative_verify_request_count = 0
        for request, verify_tokens in zip(
            batch.requests, metadata.verify_tokens_per_request
        ):
            if not getattr(request, "is_prefill_complete", False):
                continue
            if int(verify_tokens) <= 1:
                continue
            speculative_verify_request_count += 1

        if speculative_verify_request_count == 0:
            return 0.0

        replica_config = getattr(self, "_replica_config", None)
        spec_config = getattr(replica_config, "speculative_decoding_config", None)
        if spec_config is None:
            return 0.0

        profile_file = getattr(
            spec_config,
            "decode_draft_proposer_latency_profile_file",
            "",
        )
        is_mtp_method = False
        try:
            get_mtp_method_family(method_name)
            is_mtp_method = True
        except ValueError:
            is_mtp_method = False
        method_host_overhead_ms = (
            self._get_spec_decode_method_proposer_overhead_ms(method_name)
            * speculative_verify_request_count
        )

        if profile_file:
            model_name = str(
                getattr(
                    getattr(self, "_model_config", None),
                    "get_name",
                    lambda: getattr(replica_config, "model_name", ""),
                )()
            ).strip()
            if not model_name:
                model_name = str(getattr(replica_config, "model_name", "")).strip()
            if not model_name:
                raise ValueError(
                    "decode draft proposer profile lookup requires a non-empty model_name"
                )
            attn_tp_size = int(
                getattr(replica_config, "attn_tensor_parallel_size", 0)
            )
            if attn_tp_size <= 0:
                raise ValueError(
                    "decode draft proposer profile lookup requires attn_tensor_parallel_size > 0"
                )
            return get_decode_draft_proposer_latency_ms(
                lookup=getattr(
                    spec_config,
                    "_decode_draft_proposer_latency_profile",
                    None,
                ),
                method=method_name,
                model_name=model_name,
                attn_tp_size=attn_tp_size,
                num_speculative_tokens=int(spec_config.num_speculative_tokens),
                spec_verify_request_count=speculative_verify_request_count,
            )

        if is_mtp_method:
            return method_host_overhead_ms + self._get_structural_mtp_proposer_time(
                batch,
                method_name=method_name,
            )

        if method_host_overhead_ms == 0.0:
            return 0.0
        return method_host_overhead_ms

    def _get_cpu_overhead_features(self, batch: Batch) -> Dict[str, int]:
        batch_size = int(getattr(batch, "size", 0))

        num_prefill_tokens = int(
            getattr(batch, "num_prefill_tokens", DEFAULT_NUM_PREFILL_TOKENS)
        )
        if not hasattr(batch, "num_prefill_tokens") and batch_size > 0:
            num_prefill_tokens = DEFAULT_NUM_PREFILL_TOKENS

        if hasattr(batch, "num_decode_tokens"):
            num_decode_tokens = int(getattr(batch, "num_decode_tokens"))
        else:
            num_decode_tokens = (
                batch_size * DEFAULT_NUM_DECODE_TOKENS_AMPLIFICATION_FACTOR
            )

        return {
            "batch_size": batch_size,
            "num_prefill_tokens": num_prefill_tokens,
            "num_decode_tokens": num_decode_tokens,
        }

    def _log_missing_cpu_overhead_prediction_once(self, metric_name: str) -> None:
        missing_cache = getattr(self, "_missing_cpu_overhead_metrics_logged", None)
        if missing_cache is None:
            missing_cache = set()
            self._missing_cpu_overhead_metrics_logged = missing_cache

        if metric_name in missing_cache:
            return

        logger.warning(
            "CPU overhead prediction '%s' is unavailable. Falling back to 0. "
            "cpu_overhead_input_file=%s",
            metric_name,
            getattr(self, "_cpu_overhead_input_file", ""),
        )
        missing_cache.add(metric_name)

    def _get_cpu_overhead_prediction_or_default(
        self,
        metric_name: str,
        batch: Batch,
    ) -> float:
        if self._config.skip_cpu_overhead_modeling:
            return 0.0

        metric_predictions = self._predictions.get(metric_name)
        if metric_predictions is None:
            self._log_missing_cpu_overhead_prediction_once(metric_name)
            return 0.0

        if isinstance(metric_predictions, dict) and metric_predictions.get(
            "_on_demand_prediction"
        ):
            features = self._get_cpu_overhead_features(batch)
            feature_key = (
                float(features["batch_size"]),
                float(features["num_prefill_tokens"]),
                float(features["num_decode_tokens"]),
            )
            exact_lookup = metric_predictions.get("_exact_lookup") or {}
            if feature_key in exact_lookup:
                return float(exact_lookup[feature_key])
            try:
                return self._get_on_demand_prediction(metric_name, features)
            except Exception:
                self._log_missing_cpu_overhead_prediction_once(metric_name)
                return 0.0

        batch_size_key = (int(getattr(batch, "size", 0)),)
        if batch_size_key in metric_predictions:
            return float(metric_predictions[batch_size_key])

        features = self._get_cpu_overhead_features(batch)
        feature_key = (
            float(features["batch_size"]),
            float(features["num_prefill_tokens"]),
            float(features["num_decode_tokens"]),
        )
        if feature_key in metric_predictions:
            return float(metric_predictions[feature_key])

        self._log_missing_cpu_overhead_prediction_once(metric_name)
        return 0.0

    def _get_schedule_time(self, batch: Batch) -> float:
        return self._get_cpu_overhead_prediction_or_default("schedule", batch)

    def _log_architecture_attention_shape(self, batch: Batch) -> None:
        architecture_profile = self._get_model_architecture_profile()
        if architecture_profile.attention_shape_log_kind is None:
            return
        num_tokens = batch.total_num_tokens
        hidden_dim = self._model_config.embedding_dim
        share_q_dim = self._model_config.share_q_dim
        head_dim = self._model_config.get_head_dim()
        num_q_heads = self._model_config.num_q_heads
        num_kv_heads = self._model_config.num_kv_heads
        kv_size = num_kv_heads * head_dim
        qkv_out_dim = None
        if share_q_dim is not None:
            qkv_out_dim = share_q_dim + 2 * kv_size
        logger.debug(
            "[ARCH_ATTENTION_SHAPE] profile=%s kind=%s tokens=%s hidden=%s "
            "share_q_dim=%s head_dim=%s num_q_heads=%s num_kv_heads=%s "
            "qkv_out_dim=%s",
            architecture_profile.profile_id,
            architecture_profile.attention_shape_log_kind,
            num_tokens,
            hidden_dim,
            share_q_dim,
            head_dim,
            num_q_heads,
            num_kv_heads,
            qkv_out_dim,
        )

    def _get_sampler_e2e_time(self, batch: Batch) -> float:
        return self._get_cpu_overhead_prediction_or_default("sampler_e2e", batch)

    def _get_prepare_inputs_e2e_time(self, batch: Batch) -> float:
        return self._get_cpu_overhead_prediction_or_default(
            "prepare_inputs_e2e", batch
        )

    def _get_process_model_outputs_time(self, batch: Batch) -> float:
        return self._get_cpu_overhead_prediction_or_default(
            "process_model_outputs", batch
        )

    def _get_ray_comm_time(self, batch: Batch) -> float:
        return self._get_cpu_overhead_prediction_or_default("ray_comm_time", batch)

    # Phase 2.5: Removed deprecated get_moe_stage_execution_details() method
    # MoE models now use predict_moe_layer_time() and other fine-grained APIs

    def to_dict(self) -> dict:
        return {
            "model_provider": str(self._config.get_type()),
            "cluster_type": str(self._cluster_type) if self._cluster_type else None,
            "num_tensor_parallel_workers": self._replica_config.attn_tensor_parallel_size,
            "k_fold_cv_splits": self._config.k_fold_cv_splits,
            "num_q_heads": self._model_config.num_q_heads,
            "num_kv_heads": self._model_config.num_kv_heads,
            "embedding_dim": self._model_config.embedding_dim,
            "mlp_hidden_dim": self._model_config.mlp_hidden_dim,
            "use_gated_mlp": self._model_config.use_gated_mlp,
            "vocab_size": self._model_config.vocab_size,
            "block_size": self._block_size,
            "max_tokens": self._max_tokens,
            "active_measurement_type": self._active_measurement_type.value,
            "compute_input_file": self._compute_input_file,
            "attention_input_file": self._attention_input_file,
            "compute_input_file_eager": self._compute_input_file_eager,
            "attention_input_file_eager": self._attention_input_file_eager,
            "compute_input_file_kernel_only": self._compute_input_file_kernel_only,
            "attention_input_file_kernel_only": self._attention_input_file_kernel_only,
            "all_reduce_input_file": self._all_reduce_input_file,
            "send_recv_input_file": self._send_recv_input_file,
            "cpu_overhead_input_file": self._cpu_overhead_input_file,
            "pp_stage_boundary_input_file": self._pp_stage_boundary_input_file,
            "pp_receiver_head_input_file": self._pp_receiver_head_input_file,
            "pp_producer_send_path_input_file": self._pp_producer_send_path_input_file,
            "pp_prefill_consumer_active_input_file": self._pp_prefill_consumer_active_input_file,
            "prediction_max_prefill_chunk_size": self._config.prediction_max_prefill_chunk_size,
            "max_batch_size": self._config.prediction_max_batch_size,
            "using_shared_models": self._model_manager is not None,
        }

    # Phase 2.5: Removed deprecated get_execution_time() method
    # All active code paths now use predict_stage_execution_time() instead

    # ========================================================================
    # New unified API implementation (Phase 0)
    # ========================================================================

    @staticmethod
    def _get_mla_batch_runtime_shape_components(batch: Batch) -> Dict[str, Any]:
        requests = getattr(batch, "requests", None)
        request_token_counts = getattr(batch, "num_tokens", None)
        if not requests or request_token_counts is None:
            raise ValueError(
                "MLA exact-row prediction requires batch.requests and "
                "batch.num_tokens to derive vLLM max_seqlen_k."
            )
        if not hasattr(request_token_counts, "__len__"):
            raise ValueError(
                "MLA exact-row prediction requires batch.num_tokens to be a "
                f"per-request sequence, got {request_token_counts!r}."
            )
        if len(requests) != len(request_token_counts):
            raise ValueError(
                "MLA exact-row prediction requires one token count per request: "
                f"requests={len(requests)}, num_tokens={len(request_token_counts)}"
            )
        missing_batch_fields = [
            field
            for field in (
                "total_num_tokens",
                "num_prefill_tokens",
                "num_decode_tokens",
            )
            if not hasattr(batch, field)
        ]
        if missing_batch_fields:
            missing_batch_attrs = [f"batch.{field}" for field in missing_batch_fields]
            raise ValueError(
                "MLA exact-row prediction requires batch token total fields: "
                f"{missing_batch_attrs}."
            )

        max_seqlen_k = 0
        batch_num_tokens = 0
        current_tokens_by_request: list[tuple[Any, int]] = []
        prefill_active_token_counts: list[int] = []
        decode_active_token_counts: list[int] = []
        for request, num_tokens in zip(requests, request_token_counts):
            current_tokens = int(num_tokens)
            if current_tokens <= 0:
                raise ValueError(
                    "MLA exact-row prediction requires positive per-request "
                    f"token counts, got num_tokens={num_tokens}."
                )
            processed_tokens = getattr(request, "num_processed_tokens", None)
            if processed_tokens is None:
                raise ValueError(
                    "MLA exact-row prediction requires request.num_processed_tokens "
                    "to derive vLLM max_seqlen_k."
                )
            current_seq_len = int(processed_tokens) + current_tokens
            if current_seq_len <= 0:
                raise ValueError(
                    "MLA exact-row prediction requires positive runtime sequence "
                    f"lengths, got processed={processed_tokens}, "
                    f"num_tokens={num_tokens}."
                )
            max_seqlen_k = max(max_seqlen_k, current_seq_len)
            batch_num_tokens += current_tokens
            current_tokens_by_request.append((request, current_tokens))
            if not hasattr(request, "is_prefill_complete"):
                raise ValueError(
                    "MLA exact-row prediction requires request.is_prefill_complete "
                    "to derive the current vLLM prefill/decode phase partition."
                )
            if bool(getattr(request, "is_prefill_complete")):
                decode_active_token_counts.append(current_tokens)
            else:
                prefill_active_token_counts.append(current_tokens)

        if int(getattr(batch, "total_num_tokens")) != batch_num_tokens:
            raise ValueError(
                "MLA exact-row prediction requires batch.total_num_tokens to match "
                f"sum(batch.num_tokens): total_num_tokens="
                f"{getattr(batch, 'total_num_tokens')}, "
                f"sum_num_tokens={batch_num_tokens}."
            )

        batch_num_prefill_tokens = int(getattr(batch, "num_prefill_tokens"))
        batch_num_decode_tokens = int(getattr(batch, "num_decode_tokens"))
        if batch_num_prefill_tokens + batch_num_decode_tokens != batch_num_tokens:
            raise ValueError(
                "MLA exact-row prediction requires prefill/decode token counts to "
                "sum to batch_num_tokens: "
                f"prefill={batch_num_prefill_tokens}, decode={batch_num_decode_tokens}, "
                f"batch_num_tokens={batch_num_tokens}."
            )
        prefill_active_token_sum = sum(prefill_active_token_counts)
        decode_active_token_sum = sum(decode_active_token_counts)
        if prefill_active_token_sum != batch_num_prefill_tokens:
            raise ValueError(
                "MLA exact-row prediction requires request.is_prefill_complete "
                "partition to match batch.num_prefill_tokens: "
                f"partition_prefill={prefill_active_token_sum}, "
                f"batch_num_prefill_tokens={batch_num_prefill_tokens}."
            )
        if decode_active_token_sum != batch_num_decode_tokens:
            raise ValueError(
                "MLA exact-row prediction requires request.is_prefill_complete "
                "partition to match batch.num_decode_tokens: "
                f"partition_decode={decode_active_token_sum}, "
                f"batch_num_decode_tokens={batch_num_decode_tokens}."
            )

        return {
            "requests": requests,
            "current_tokens_by_request": current_tokens_by_request,
            "prefill_active_token_counts": prefill_active_token_counts,
            "decode_active_token_counts": decode_active_token_counts,
            "batch_size": len(requests),
            "batch_num_tokens": batch_num_tokens,
            "batch_num_prefill_tokens": batch_num_prefill_tokens,
            "batch_num_decode_tokens": batch_num_decode_tokens,
            "max_seqlen_k": max_seqlen_k,
        }

    @staticmethod
    def _get_mla_runtime_dynamic_shape_features(batch: Batch) -> Dict[str, int]:
        components = SklearnExecutionTimePredictor._get_mla_batch_runtime_shape_components(
            batch
        )
        max_seqlen_q = max(
            current_tokens
            for _, current_tokens in components["current_tokens_by_request"]
        )
        batch_num_prefill_tokens = int(components["batch_num_prefill_tokens"])
        max_seqlen_k = int(components["max_seqlen_k"])
        batch_num_tokens = int(components["batch_num_tokens"])

        return {
            "batch_size": int(components["batch_size"]),
            "batch_num_tokens": batch_num_tokens,
            "batch_num_prefill_tokens": batch_num_prefill_tokens,
            "batch_num_decode_tokens": int(components["batch_num_decode_tokens"]),
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "num_actual_tokens": batch_num_tokens,
            "is_prefill": int(batch_num_prefill_tokens > 0),
            "max_seq_len": max_seqlen_k,
        }

    @staticmethod
    def _get_mla_runtime_max_seq_len(batch: Batch) -> int:
        return SklearnExecutionTimePredictor._get_mla_runtime_dynamic_shape_features(
            batch
        )["max_seqlen_k"]

    @staticmethod
    def _mla_operator_phase_kind(op_name: str) -> str:
        for operator in LATENT_MLA_ATTENTION_FAMILY.predictor_ops():
            if operator.name != op_name:
                continue
            if operator.role is AttentionOperatorRole.CACHE_WRITE:
                return "cache_write"
            if (
                AttentionPhase.PREFILL in operator.phases
                and AttentionPhase.DECODE not in operator.phases
            ):
                return "prefill"
            if (
                AttentionPhase.DECODE in operator.phases
                and AttentionPhase.PREFILL not in operator.phases
            ):
                return "decode"
            raise ValueError(
                "Unsupported MLA operator phase contract for "
                f"{op_name}: {operator.phases}"
            )
        raise ValueError(f"Unknown MLA predictor operator: {op_name}")

    @staticmethod
    def _is_mla_operator_applicable_to_batch(op_name: str, batch: Batch) -> bool:
        phase_kind = SklearnExecutionTimePredictor._mla_operator_phase_kind(op_name)
        if phase_kind == "cache_write":
            return int(getattr(batch, "total_num_tokens")) > 0
        if phase_kind == "prefill":
            return int(getattr(batch, "num_prefill_tokens")) > 0
        if phase_kind == "decode":
            return int(getattr(batch, "num_decode_tokens")) > 0
        raise ValueError(f"Unsupported MLA operator phase kind: {phase_kind}")

    @staticmethod
    def _get_mla_runtime_dynamic_shape_features_for_op(
        batch: Batch,
        op_name: str,
    ) -> Dict[str, int]:
        components = SklearnExecutionTimePredictor._get_mla_batch_runtime_shape_components(
            batch
        )
        phase_kind = SklearnExecutionTimePredictor._mla_operator_phase_kind(op_name)
        current_tokens_by_request = components["current_tokens_by_request"]
        if phase_kind == "cache_write":
            active_token_counts = [
                current_tokens for _, current_tokens in current_tokens_by_request
            ]
            num_actual_tokens = int(components["batch_num_tokens"])
        elif phase_kind == "prefill":
            active_token_counts = list(components["prefill_active_token_counts"])
            num_actual_tokens = int(components["batch_num_prefill_tokens"])
        elif phase_kind == "decode":
            active_token_counts = list(components["decode_active_token_counts"])
            num_actual_tokens = int(components["batch_num_decode_tokens"])
        else:
            raise ValueError(f"Unsupported MLA operator phase kind: {phase_kind}")

        if not active_token_counts or num_actual_tokens <= 0:
            raise ValueError(
                "MLA operator dynamic shape requested for an inactive operator: "
                f"op_name={op_name}, phase_kind={phase_kind}, "
                f"batch_num_prefill_tokens={components['batch_num_prefill_tokens']}, "
                f"batch_num_decode_tokens={components['batch_num_decode_tokens']}."
            )
        active_token_sum = sum(active_token_counts)
        if active_token_sum != num_actual_tokens:
            raise ValueError(
                "MLA operator dynamic shape active token sum must match "
                f"num_actual_tokens for {op_name}: active_token_sum="
                f"{active_token_sum}, num_actual_tokens={num_actual_tokens}."
            )

        max_seqlen_k = int(components["max_seqlen_k"])
        return {
            "batch_size": int(components["batch_size"]),
            "batch_num_tokens": int(components["batch_num_tokens"]),
            "batch_num_prefill_tokens": int(
                components["batch_num_prefill_tokens"]
            ),
            "batch_num_decode_tokens": int(components["batch_num_decode_tokens"]),
            "max_seqlen_q": max(active_token_counts),
            "max_seqlen_k": max_seqlen_k,
            "num_actual_tokens": num_actual_tokens,
            "is_prefill": int(int(components["batch_num_prefill_tokens"]) > 0),
            "max_seq_len": max_seqlen_k,
        }

    def _get_mla_imported_predictor_features(
        self,
        batch: Batch,
        op_name: str | None = None,
    ) -> Dict[str, float]:
        feature_values: Dict[str, float] = {}
        model_config = self._model_config
        dynamic_shape_features = (
            self._get_mla_runtime_dynamic_shape_features(batch)
            if op_name is None
            else self._get_mla_runtime_dynamic_shape_features_for_op(batch, op_name)
        )
        getters = {
            "n_q_head": lambda: getattr(model_config, "num_q_heads"),
            "n_kv_head": lambda: model_config.get_runtime_num_kv_heads()
            if hasattr(model_config, "get_runtime_num_kv_heads")
            else 1,
            "head_size": lambda: model_config.get_runtime_head_size()
            if hasattr(model_config, "get_runtime_head_size")
            else int(getattr(model_config, "kv_lora_rank"))
            + int(getattr(model_config, "qk_rope_head_dim")),
            "qk_nope_head_dim": lambda: getattr(model_config, "qk_nope_head_dim"),
            "qk_rope_head_dim": lambda: getattr(model_config, "qk_rope_head_dim"),
            "qk_head_dim": lambda: model_config.get_qk_head_dim(),
            "kv_lora_rank": lambda: getattr(model_config, "kv_lora_rank"),
            "v_head_dim": lambda: getattr(model_config, "v_head_dim"),
            "block_size": lambda: self._block_size,
            "num_tensor_parallel_workers": lambda: (
                self._replica_config.attn_tensor_parallel_size
            ),
            "batch_size": lambda: dynamic_shape_features["batch_size"],
            "batch_num_tokens": lambda: dynamic_shape_features["batch_num_tokens"],
            "batch_num_prefill_tokens": lambda: (
                dynamic_shape_features["batch_num_prefill_tokens"]
            ),
            "batch_num_decode_tokens": lambda: (
                dynamic_shape_features["batch_num_decode_tokens"]
            ),
            "max_seqlen_q": lambda: dynamic_shape_features["max_seqlen_q"],
            "max_seqlen_k": lambda: dynamic_shape_features["max_seqlen_k"],
            "num_actual_tokens": lambda: dynamic_shape_features["num_actual_tokens"],
            "is_prefill": lambda: dynamic_shape_features["is_prefill"],
            "max_seq_len": lambda: dynamic_shape_features["max_seq_len"],
        }
        feature_columns = get_enabled_predictor_feature_columns(
            LATENT_MLA_ATTENTION_FAMILY
        )
        required_columns = next(iter(feature_columns.values()))
        for column in required_columns:
            try:
                value = getters[column]()
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported MLA predictor feature column: {column!r}"
                ) from exc
            if value is None:
                raise ValueError(
                    f"MLA predictor feature {column!r} is not configured"
                )
            feature_values[column] = float(value)
        return feature_values

    def _get_mla_attention_operator_times(self, batch: Batch) -> AttentionOperatorTimes:
        self._get_mla_batch_runtime_shape_components(batch)
        feature_columns_by_op = get_enabled_predictor_feature_columns(
            LATENT_MLA_ATTENTION_FAMILY
        )
        op_times: Dict[str, float] = {}
        for op_name in get_enabled_predictor_metric_names(LATENT_MLA_ATTENTION_FAMILY):
            if not self._is_mla_operator_applicable_to_batch(op_name, batch):
                op_times[op_name] = 0.0
                continue
            features = self._get_mla_imported_predictor_features(batch, op_name)
            model_info = self._predictions.get(op_name)
            if not isinstance(model_info, dict) or not model_info.get(
                "_on_demand_prediction"
            ):
                raise ValueError(
                    "MLA predictor requires imported-row on-demand metadata for "
                    f"{op_name}"
                )
            feature_names = list(model_info.get("_feature_names", ()))
            expected_feature_names = list(feature_columns_by_op[op_name])
            if feature_names != expected_feature_names:
                raise ValueError(
                    "MLA predictor feature schema mismatch for "
                    f"{op_name}: expected {expected_feature_names}, got "
                    f"{feature_names}"
                )
            exact_key = tuple(float(features[name]) for name in feature_names)
            exact_lookup = model_info.get("_exact_lookup") or {}
            if exact_key in exact_lookup:
                op_times[op_name] = float(exact_lookup[exact_key])
                continue

            model = model_info.get("_model")
            if model is None or getattr(model, "_frontier_model_hash", None) is None:
                raise ValueError(
                    f"No exact MLA profiling row for {op_name}: "
                    f"features={dict(zip(feature_names, exact_key))}. "
                    "No trained MLA prediction model is available for exact-miss "
                    "prediction."
                )
            op_times[op_name] = self._get_on_demand_prediction(op_name, features)
        return AttentionOperatorTimes(op_times)

    def _predict_mla_attention_layer_time(
        self,
        batch: Batch,
        layer_id: int,
        cluster_type: ClusterType,
    ) -> AttentionTime:
        operator_times = self._get_mla_attention_operator_times(batch)
        cluster_name = cluster_type.name
        batch_input_lens = [req.num_prefill_tokens for req in batch.requests]
        logger.info(
            f"[OP-TRACE][{cluster_name}][ATTENTION] batch_id={batch.id}, layer_id={layer_id}, "
            f"num_tokens={batch.total_num_tokens}, batch_size={len(batch.requests)}, "
            f"batch_input_lens={batch_input_lens}, model_type=mla"
        )
        for op_name, predicted_time_ms in operator_times.op_times.items():
            logger.info(
                f"[OP-TRACE][{cluster_name}][ATTENTION][{op_name}] batch_id={batch.id}, layer_id={layer_id}, "
                f"predicted_time_ms={predicted_time_ms:.6f}"
            )
        logger.info(
            f"[OP-TRACE][{cluster_name}][ATTENTION][TOTAL] batch_id={batch.id}, layer_id={layer_id}, "
            f"total_attention_time_ms={operator_times.total_time():.6f}"
        )
        return AttentionTime(operator_times=operator_times)

    def predict_attention_layer_time(
        self, batch: Batch, layer_id: int, cluster_type: ClusterType
    ) -> AttentionTime:
        """
        Predict attention execution time for a single transformer layer.

        For dense models, layer_id is not used (all layers are uniform).
        Reuses existing internal _get_attention_* methods.

        Special handling for idle batches:
        - Idle batches skip attention computation (raise error)
        - Idle batches are created when num_requests < attn_dp_size
        - They only participate in MoE synchronization, not attention
        """
        # Check if this is an idle batch
        if batch.is_idle:
            logger.debug(
                f"Idle batch detected (batch_id={batch.id}), skipping attention computation"
            )
            raise ValueError("Idle batch detected, skipping attention computation")
            # return AttentionTime(
            #     attention_prefill_execution_time=0.0,
            #     attention_decode_execution_time=0.0,
            #     attention_layer_pre_proj_execution_time=0.0,
            #     attention_layer_post_proj_execution_time=0.0,
            #     attention_rope_execution_time=0.0,
            #     attention_kv_cache_save_execution_time=0.0,
            #     attn_norm_time=0.0,
            # )

        self._log_architecture_attention_shape(batch)

        attention_family = self._get_attention_family()
        attention_family.require_enabled_for_execution()

        if self._enable_dummy_mode:
            base_time = self._dummy_execution_time
            return AttentionTime(
                attention_prefill_execution_time=base_time,
                attention_decode_execution_time=base_time,
                attention_layer_pre_proj_execution_time=base_time,
                attention_layer_post_proj_execution_time=base_time,
                attention_rope_execution_time=base_time,
                attention_kv_cache_save_execution_time=base_time,
                attn_norm_time=base_time,
            )

        if not self._supports_operation("attention"):
            raise NotImplementedError(
                f"Attention operations not supported for cluster type {cluster_type}"
            )

        if attention_family.family_id == LATENT_MLA_ATTENTION_FAMILY.family_id:
            return self._predict_mla_attention_layer_time(
                batch=batch,
                layer_id=layer_id,
                cluster_type=cluster_type,
            )

        logger.debug(
            f"Predicting attention layer time for layer_id={layer_id}, cluster_type={cluster_type}"
        )

        # Get individual operation times for detailed tracing
        prefill_op_name = self._dense_attention_prefill_op_name()
        decode_op_name = self._dense_attention_decode_op_name()
        cache_write_op_name = self._dense_attention_cache_write_op_name()
        attn_prefill_time = (
            self._get_attention_prefill_execution_time(batch)
            if batch.num_prefill_tokens > 0 and self._supports_operation(prefill_op_name)
            else 0.0
        )
        attn_decode_time = (
            self._get_attention_decode_execution_time(batch)
            if batch.num_decode_tokens > 0 and self._supports_operation(decode_op_name)
            else 0.0
        )

        # Phase 1 speculative decode modeling:
        # verify requests (verify_tokens > 1) are routed to attn_prefill predictor,
        # normal decode requests stay on attn_decode predictor.
        # Aggregate with SUM semantics to match serial kernel launches.
        spec_metadata = getattr(batch, "spec_decode_metadata", None)
        if spec_metadata is not None:
            if self._should_use_hybrid_attention_measurement_for_spec_piecewise(batch):
                with self._temporary_measurement_type(MeasurementType.CUDA_EVENT):
                    verify_prefill_time = (
                        self._get_spec_verify_attention_prefill_execution_time(batch)
                    )
                    normal_decode_time = (
                        self._get_spec_normal_decode_attention_execution_time(batch)
                    )
            else:
                verify_prefill_time = (
                    self._get_spec_verify_attention_prefill_execution_time(batch)
                )
                normal_decode_time = (
                    self._get_spec_normal_decode_attention_execution_time(batch)
                )
            attn_prefill_time = attn_prefill_time + verify_prefill_time
            attn_decode_time = normal_decode_time
        attn_pre_proj_time = self._get_attention_layer_pre_proj_execution_time(batch)
        attn_post_proj_time = self._get_attention_layer_post_proj_execution_time(batch)
        attn_rope_time = self._get_attention_rope_execution_time(batch)
        if self._should_use_hybrid_attention_measurement_for_spec_piecewise(batch):
            with self._temporary_measurement_type(MeasurementType.CUDA_EVENT):
                attn_kv_cache_save_time = (
                    self._get_attention_kv_cache_save_execution_time(batch)
                )
        else:
            attn_kv_cache_save_time = self._get_attention_kv_cache_save_execution_time(
                batch
            )
        attn_norm_time = self._get_attn_norm_layer_act_execution_time(batch)

        # Architecture-profile attention extras are 0.0 when not declared by the profile.
        attn_inter_norm_time = 0.0
        attn_wq_proj_time = 0.0
        if "attn_inter_norm" in self._get_predictor_attention_extra_ops():
            attn_inter_norm_time = self._get_attn_inter_norm_execution_time(batch)
            attn_wq_proj_time = self._get_attn_wq_proj_execution_time(batch)

        # Operation-level tracing for attention operations
        # This enables comparison with real vLLM operation-level GPU execution traces
        # Refactored to use dynamic cluster_type.name instead of if/elif branching
        cluster_name = cluster_type.name
        batch_input_lens = [req.num_prefill_tokens for req in batch.requests]

        logger.info(
            f"[OP-TRACE][{cluster_name}][ATTENTION] batch_id={batch.id}, layer_id={layer_id}, "
            f"num_tokens={batch.total_num_tokens}, batch_size={len(batch.requests)}, "
            f"batch_input_lens={batch_input_lens}, model_type=dense"
        )

        logger.info(
            f"[OP-TRACE][{cluster_name}][ATTENTION] batch_id={batch.id}, layer_id={layer_id}, "
            f"num_tokens={batch.total_num_tokens}, batch_size={len(batch.requests)}, "
            f"num_prefill_tokens={batch.num_prefill_tokens}, num_decode_tokens={batch.num_decode_tokens}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][ATTENTION][input_layernorm] batch_id={batch.id}, layer_id={layer_id}, "
            f"predicted_time_ms={attn_norm_time:.6f}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][ATTENTION][attn_pre_proj] batch_id={batch.id}, layer_id={layer_id}, "
            f"predicted_time_ms={attn_pre_proj_time:.6f}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][ATTENTION][attn_rope] batch_id={batch.id}, layer_id={layer_id}, "
            f"predicted_time_ms={attn_rope_time:.6f}"
        )
        if batch.num_prefill_tokens > 0:
            logger.info(
                f"[OP-TRACE][{cluster_name}][ATTENTION][{prefill_op_name}] batch_id={batch.id}, layer_id={layer_id}, "
                f"predicted_time_ms={attn_prefill_time:.6f}"
            )
        if batch.num_decode_tokens > 0:
            logger.info(
                f"[OP-TRACE][{cluster_name}][ATTENTION][{decode_op_name}] batch_id={batch.id}, layer_id={layer_id}, "
                f"predicted_time_ms={attn_decode_time:.6f}"
            )
        logger.info(
            f"[OP-TRACE][{cluster_name}][ATTENTION][{cache_write_op_name}] batch_id={batch.id}, layer_id={layer_id}, "
            f"predicted_time_ms={attn_kv_cache_save_time:.6f}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][ATTENTION][attn_post_proj] batch_id={batch.id}, layer_id={layer_id}, "
            f"predicted_time_ms={attn_post_proj_time:.6f}"
        )

        # Architecture-profile attention extra operation tracing.
        if "attn_inter_norm" in self._get_predictor_attention_extra_ops():
            logger.info(
                f"[OP-TRACE][{cluster_name}][ATTENTION][attn_inter_norm] batch_id={batch.id}, layer_id={layer_id}, "
                f"predicted_time_ms={attn_inter_norm_time:.6f}"
            )
            logger.info(
                f"[OP-TRACE][{cluster_name}][ATTENTION][attn_wq_proj] batch_id={batch.id}, layer_id={layer_id}, "
                f"predicted_time_ms={attn_wq_proj_time:.6f}"
            )

        total_attention_time = (
            attn_norm_time
            + attn_pre_proj_time
            + attn_rope_time
            + attn_prefill_time
            + attn_decode_time
            + attn_kv_cache_save_time
            + attn_post_proj_time
            # Architecture-profile attention extras are 0.0 when absent.
            + attn_inter_norm_time
            + attn_wq_proj_time
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][ATTENTION][TOTAL] batch_id={batch.id}, layer_id={layer_id}, "
            f"total_attention_time_ms={total_attention_time:.6f}"
        )

        return AttentionTime(
            attention_prefill_execution_time=attn_prefill_time,
            attention_decode_execution_time=attn_decode_time,
            attention_layer_pre_proj_execution_time=attn_pre_proj_time,
            attention_layer_post_proj_execution_time=attn_post_proj_time,
            attention_rope_execution_time=attn_rope_time,
            attention_kv_cache_save_execution_time=attn_kv_cache_save_time,
            attn_norm_time=attn_norm_time,
            # Architecture-profile attention extras are 0.0 when absent.
            attn_inter_norm_time=attn_inter_norm_time,
            attn_wq_proj_time=attn_wq_proj_time,
        )

    def predict_mlp_layer_time(
        self, batch: Batch, layer_id: int, cluster_type: ClusterType
    ) -> MLPTime:
        """
        Predict dense MLP execution time for a single transformer layer.

        For dense models, layer_id is not used (all layers are uniform).
        Reuses existing internal _get_mlp_* methods.
        """
        if self._enable_dummy_mode:
            base_time = self._dummy_execution_time
            return MLPTime(
                mlp_layer_up_proj_execution_time=base_time,
                mlp_layer_down_proj_execution_time=base_time,
                mlp_layer_act_execution_time=base_time,
                mlp_norm_time=base_time,
            )

        if not self._supports_operation("mlp_up_proj"):
            raise NotImplementedError(
                f"MLP operations not supported for cluster type {cluster_type}"
            )

        # Extract detailed batch information for logging
        batch_input_lens = [req.num_prefill_tokens for req in batch.requests]
        batch_request_ids = [req.id for req in batch.requests]

        logger.debug(
            f"Predicting MLP layer time for layer_id={layer_id}, cluster_type={cluster_type.name}, "
            f"batch_id={batch.id}, num_tokens={batch.total_num_tokens}, batch_size={len(batch.requests)}, "
            f"batch_input_lens={batch_input_lens}, batch_request_ids={batch_request_ids}"
        )

        # Get individual MLP operation times for detailed tracing
        mlp_up_proj_time = self._get_mlp_layer_up_proj_execution_time(batch)
        mlp_down_proj_time = self._get_mlp_layer_down_proj_execution_time(batch)
        mlp_act_time = self._get_mlp_layer_act_execution_time(batch)
        mlp_norm_time = self._get_mlp_norm_layer_act_execution_time(batch)

        # Operation-level tracing for MLP (dense model FFN) operations
        # This enables comparison with real vLLM operation-level GPU execution traces
        # Refactored to use dynamic cluster_type.name instead of if/elif branching
        cluster_name = cluster_type.name

        # Header log with batch details
        logger.info(
            f"[OP-TRACE][{cluster_name}][MLP] batch_id={batch.id}, layer_id={layer_id}, "
            f"num_tokens={batch.total_num_tokens}, batch_size={len(batch.requests)}, "
            f"batch_input_lens={batch_input_lens}, model_type=dense"
        )

        # Individual operation logs
        logger.info(
            f"[OP-TRACE][{cluster_name}][MLP][post_attention_layernorm] batch_id={batch.id}, layer_id={layer_id}, "
            f"predicted_time_ms={mlp_norm_time:.6f}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][MLP][mlp_up_proj] batch_id={batch.id}, layer_id={layer_id}, "
            f"predicted_time_ms={mlp_up_proj_time:.6f}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][MLP][mlp_act] batch_id={batch.id}, layer_id={layer_id}, "
            f"predicted_time_ms={mlp_act_time:.6f}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][MLP][mlp_down_proj] batch_id={batch.id}, layer_id={layer_id}, "
            f"predicted_time_ms={mlp_down_proj_time:.6f}"
        )

        # Total time log
        total_mlp_time = (
            mlp_norm_time + mlp_up_proj_time + mlp_act_time + mlp_down_proj_time
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][MLP][TOTAL] batch_id={batch.id}, layer_id={layer_id}, "
            f"total_mlp_time_ms={total_mlp_time:.6f}"
        )

        return MLPTime(
            mlp_layer_up_proj_execution_time=mlp_up_proj_time,
            mlp_layer_down_proj_execution_time=mlp_down_proj_time,
            mlp_layer_act_execution_time=mlp_act_time,
            mlp_norm_time=mlp_norm_time,
            operator_times=MLPOperatorTimes(
                op_times={
                    "post_attention_layernorm": mlp_norm_time,
                    "mlp_up_proj": mlp_up_proj_time,
                    "mlp_act": mlp_act_time,
                    "mlp_down_proj": mlp_down_proj_time,
                }
            ),
        )

    def predict_moe_layer_time(
        self,
        batch_or_group: "Batch | EPBatchGroup",
        layer_id: int,
        cluster_type: ClusterType,
        per_expert_tokens: Optional[Dict[int, int]] = None,
    ) -> MoETime:
        """
        Predict MoE execution time for a single transformer layer.

        Not supported for dense models - raises NotImplementedError.
        This method is implemented in frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py module

        """
        raise NotImplementedError(
            "MoE operations are not supported by SklearnExecutionTimePredictor. "
            "Use SklearnMoEExecutionTimePredictor for MoE models."
        )

    def predict_allreduce_time(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: ClusterType,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict tensor parallel all-reduce communication time.

        Delegates to CC Backend if available, otherwise falls back to dummy mode.

        Args:
            data_size_bytes: Size of data in bytes
            num_devices: Number of participating devices
            cluster_type: Type of cluster for context-aware prediction

        Returns:
            Predicted execution time in milliseconds
        """
        # Use CC Backend if available for communication predictions
        if self._cc_backend is not None:
            result = self._cc_backend.predict_allreduce(
                data_size_bytes=data_size_bytes,
                num_devices=num_devices,
                cluster_type=cluster_type,
                comm_domain=comm_domain,
            )
            logger.debug(
                f"predict_allreduce_time: using CC Backend, "
                f"data_size={data_size_bytes}, num_devices={num_devices}, result={result:.6f} ms"
            )
            return result

        # When CC Backend is not available, require explicit dummy mode
        if self._enable_dummy_mode:
            logger.info(
                f"[CC-FALLBACK] predict_allreduce_time: CC Backend not available, "
                f"falling back to dummy mode value={self._dummy_execution_time} ms"
            )
            return self._dummy_execution_time

        # Fail fast: CC Backend is required for all-reduce communication predictions
        # unless dummy mode is explicitly enabled
        raise RuntimeError(
            f"CC Backend is required for all-reduce communication prediction "
            f"but was not provided. Either:\n"
            f"  1. Configure a CC Backend (e.g., --cc_backend vidur or --cc_backend analytical)\n"
            f"  2. Enable dummy mode explicitly (--enable_dummy_mode)\n"
            f"Current state: cc_backend=None, enable_dummy_mode={self._enable_dummy_mode}"
        )

    def predict_allgather_time(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: ClusterType,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict expert parallel all-gather communication time.

        Delegates to CC Backend if available, otherwise falls back to dummy mode.

        Args:
            data_size_bytes: Size of data per device in bytes
            num_devices: Number of participating devices
            cluster_type: Type of cluster for context-aware prediction

        Returns:
            Predicted execution time in milliseconds
        """
        # Use CC Backend if available for communication predictions
        if self._cc_backend is not None:
            result = self._cc_backend.predict_allgather(
                data_size_bytes=data_size_bytes,
                num_devices=num_devices,
                cluster_type=cluster_type,
                comm_domain=comm_domain,
            )
            logger.debug(
                f"predict_allgather_time: using CC Backend, "
                f"data_size={data_size_bytes}, num_devices={num_devices}, result={result:.6f} ms"
            )
            return result

        # When CC Backend is not available, require explicit dummy mode
        if self._enable_dummy_mode:
            logger.info(
                f"[CC-FALLBACK] predict_allgather_time: CC Backend not available, "
                f"falling back to dummy mode value={self._dummy_execution_time} ms"
            )
            return self._dummy_execution_time

        # Fail fast: CC Backend is required for all-gather communication predictions
        # unless dummy mode is explicitly enabled
        raise RuntimeError(
            f"CC Backend is required for all-gather communication prediction "
            f"but was not provided. Either:\n"
            f"  1. Configure a CC Backend (e.g., --cc_backend vidur or --cc_backend analytical)\n"
            f"  2. Enable dummy mode explicitly (--enable_dummy_mode)\n"
            f"Current state: cc_backend=None, enable_dummy_mode={self._enable_dummy_mode}"
        )

    def predict_alltoall_time(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: ClusterType,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict expert parallel all-to-all communication time.

        Delegates to CC Backend if available, otherwise falls back to dummy mode.

        Args:
            data_size_bytes: Total size of data in bytes
            num_devices: Number of participating devices
            cluster_type: Type of cluster for context-aware prediction

        Returns:
            Predicted execution time in milliseconds
        """
        # Use CC Backend if available for communication predictions
        if self._cc_backend is not None:
            result = self._cc_backend.predict_all_to_all(
                data_size_bytes=data_size_bytes,
                num_devices=num_devices,
                cluster_type=cluster_type,
                comm_domain=comm_domain,
            )
            logger.debug(
                f"predict_alltoall_time: using CC Backend, "
                f"data_size={data_size_bytes}, num_devices={num_devices}, result={result:.6f} ms"
            )
            return result

        # When CC Backend is not available, require explicit dummy mode
        if self._enable_dummy_mode:
            logger.info(
                f"[CC-FALLBACK] predict_alltoall_time: CC Backend not available, "
                f"falling back to dummy mode value={self._dummy_execution_time} ms"
            )
            return self._dummy_execution_time

        # Fail fast: CC Backend is required for all-to-all communication predictions
        # unless dummy mode is explicitly enabled
        raise RuntimeError(
            f"CC Backend is required for all-to-all communication prediction "
            f"but was not provided. Either:\n"
            f"  1. Configure a CC Backend (e.g., --cc_backend vidur or --cc_backend analytical)\n"
            f"  2. Enable dummy mode explicitly (--enable_dummy_mode)\n"
            f"Current state: cc_backend=None, enable_dummy_mode={self._enable_dummy_mode}"
        )

    def predict_p2p_time(
        self,
        data_size_bytes: int,
        cluster_type: ClusterType,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict pipeline parallel point-to-point communication time.

        Delegates to CC Backend if available, otherwise falls back to dummy mode.

        Args:
            data_size_bytes: Size of data in bytes
            cluster_type: Type of cluster for context-aware prediction

        Returns:
            Predicted execution time in milliseconds
        """
        # Use CC Backend if available for communication predictions
        if self._cc_backend is not None:
            result = self._cc_backend.predict_send_recv(
                data_size_bytes=data_size_bytes,
                cluster_type=cluster_type,
                comm_domain=comm_domain,
            )
            logger.debug(
                f"predict_p2p_time: using CC Backend, "
                f"data_size={data_size_bytes}, result={result:.6f} ms"
            )
            return result

        # When CC Backend is not available, require explicit dummy mode
        if self._enable_dummy_mode:
            logger.info(
                f"[CC-FALLBACK] predict_p2p_time: CC Backend not available, "
                f"falling back to dummy mode value={self._dummy_execution_time} ms"
            )
            return self._dummy_execution_time

        # Fail fast: CC Backend is required for P2P communication predictions
        # unless dummy mode is explicitly enabled
        raise RuntimeError(
            f"CC Backend is required for P2P (send/recv) communication prediction "
            f"but was not provided. Either:\n"
            f"  1. Configure a CC Backend (e.g., --cc_backend vidur or --cc_backend analytical)\n"
            f"  2. Enable dummy mode explicitly (--enable_dummy_mode)\n"
            f"Current state: cc_backend=None, enable_dummy_mode={self._enable_dummy_mode}"
        )

    def predict_stage_execution_time(
        self,
        batch: Batch,
        stage_id: int,
        cluster_type: ClusterType,
        num_layers: int = 1,
        layer_id: int = 0,
    ) -> ExecutionTime:
        """
        Predict aggregated execution time for one or more transformer layers.

        This is the main entry point for execution time prediction. It composes
        attention, MLP, communication, overhead, and residual times.

        For dense models:
        - Single-layer prediction (num_layers=1): Used by PD+AF disaggregation
        - Multi-layer aggregation (num_layers>1): Used by monolithic systems

        Implementation strategy:
        - For now, delegate to existing get_execution_time() and scale by num_layers
        - Future: Use fine-grained predict_attention_layer_time() + predict_mlp_layer_time()

        Communication skip rules:
        - Pipeline parallel send/recv is skipped when stage_id is the last pipeline stage.
        - Attention tensor parallel all-reduce is skipped when attn_tensor_parallel_size == 1.

        Notes:
        - layer_id is accepted for API compatibility with MoE per-layer routing flows.
          Dense execution-time prediction is layer-homogeneous and ignores this value.
        """
        if self._enable_dummy_mode:
            return self._get_dummy_execution_time(batch, stage_id)

        logger.debug(
            f"[EXEC_TIME_PREDICT] Predicting stage execution time: stage_id={stage_id}, "
            f"cluster_type={cluster_type}, num_layers={num_layers}, layer_id={layer_id}, batch_id={batch.id}, "
            f"batch_size={batch.size}, num_tokens={batch.num_tokens}"
        )

        assert num_layers >= 1, f"num_layers must be >= 1, got {num_layers}"

        measurement_type = self._select_measurement_type_for_batch(batch)
        self._require_predictions_for_measurement_type(measurement_type, batch)
        self._activate_measurement_type(measurement_type)

        # Calculate first-class communication operators for the dense live path.
        communication_operator_times: dict[str, float] = {}
        if stage_id == self._replica_config.num_pipeline_stages - 1:
            pipeline_parallel_communication_time = 0
        else:
            pipeline_parallel_communication_time = (
                self._predict_comm_operator(
                    get_comm_operator("pipeline_parallel_send_recv"),
                    batch,
                )
            )
            communication_operator_times["pipeline_parallel_send_recv"] = (
                pipeline_parallel_communication_time
            )

        if self._replica_config.attn_tensor_parallel_size == 1:
            tensor_parallel_communication_time = 0
        else:
            tensor_parallel_communication_time = (
                self._predict_comm_operator(
                    get_comm_operator("attn_tensor_parallel_allreduce"),
                    batch,
                )
            )
            communication_operator_times["attn_tensor_parallel_allreduce"] = (
                tensor_parallel_communication_time
            )
            communication_operator_times["mlp_tensor_parallel_allreduce"] = (
                tensor_parallel_communication_time
            )

        # Build base ExecutionTime for single layer.
        # IMPORTANT: attention must come from predict_attention_layer_time()
        # so the main execution path consumes speculative verify routing and
        # proposer-overhead logic implemented there.
        attention_time = self.predict_attention_layer_time(
            batch=batch,
            layer_id=layer_id,
            cluster_type=cluster_type,
        )
        attn_rope_time = self._validate_prediction_value(
            attention_time.attention_rope_execution_time,
            "attention_rope",
            batch,
            f"stage={stage_id}",
        )
        attn_kv_save_time = self._validate_prediction_value(
            attention_time.attention_kv_cache_save_execution_time,
            "attention_kv_cache_save",
            batch,
            f"stage={stage_id}",
        )
        attn_decode_time = self._validate_prediction_value(
            attention_time.attention_decode_execution_time,
            "attention_decode",
            batch,
            f"stage={stage_id}",
        )
        attn_prefill_time = self._validate_prediction_value(
            attention_time.attention_prefill_execution_time,
            "attention_prefill",
            batch,
            f"stage={stage_id}",
        )
        attn_pre_proj_time = self._validate_prediction_value(
            attention_time.attention_layer_pre_proj_execution_time,
            "attention_pre_proj",
            batch,
            f"stage={stage_id}",
        )
        attn_post_proj_time = self._validate_prediction_value(
            attention_time.attention_layer_post_proj_execution_time,
            "attention_post_proj",
            batch,
            f"stage={stage_id}",
        )
        mlp_up_proj_time = self._validate_prediction_value(
            self._get_mlp_layer_up_proj_execution_time(batch),
            "mlp_up_proj",
            batch,
            f"stage={stage_id}",
        )
        mlp_down_proj_time = self._validate_prediction_value(
            self._get_mlp_layer_down_proj_execution_time(batch),
            "mlp_down_proj",
            batch,
            f"stage={stage_id}",
        )
        mlp_act_time = self._validate_prediction_value(
            self._get_mlp_layer_act_execution_time(batch),
            "mlp_act",
            batch,
            f"stage={stage_id}",
        )
        attn_norm_time = self._validate_prediction_value(
            attention_time.attn_norm_time,
            "attn_norm",
            batch,
            f"stage={stage_id}",
        )
        mlp_norm_time = self._validate_prediction_value(
            self._get_mlp_norm_layer_act_execution_time(batch),
            "mlp_norm",
            batch,
            f"stage={stage_id}",
        )
        add_time = self._validate_prediction_value(
            self._get_add_layer_act_execution_time(batch),
            "add",
            batch,
            f"stage={stage_id}",
        )
        schedule_time = self._validate_prediction_value(
            self._get_schedule_time(batch), "schedule", batch, f"stage={stage_id}"
        )
        sampler_time = self._validate_prediction_value(
            self._get_sampler_e2e_time(batch), "sampler", batch, f"stage={stage_id}"
        )
        prepare_inputs_time = self._validate_prediction_value(
            self._get_prepare_inputs_e2e_time(batch),
            "prepare_inputs",
            batch,
            f"stage={stage_id}",
        )
        process_outputs_time = self._validate_prediction_value(
            self._get_process_model_outputs_time(batch),
            "process_outputs",
            batch,
            f"stage={stage_id}",
        )
        ray_comm_time = self._validate_prediction_value(
            self._get_ray_comm_time(batch), "ray_comm", batch, f"stage={stage_id}"
        )
        pp_producer_send_path_runtime_time = self._validate_prediction_value(
            self._get_pp_producer_send_path_runtime_time(batch, stage_id),
            "pp_producer_send_path_runtime",
            batch,
            f"stage={stage_id}",
        )
        pp_receiver_head_runtime_time = self._validate_prediction_value(
            self._get_pp_receiver_head_runtime_time(batch, stage_id),
            "pp_receiver_head_runtime",
            batch,
            f"stage={stage_id}",
        )
        pp_prefill_consumer_active_runtime_time = self._validate_prediction_value(
            self._get_pp_prefill_consumer_active_runtime_time(batch, stage_id),
            "pp_prefill_consumer_active_runtime",
            batch,
            f"stage={stage_id}",
        )
        pp_stage_boundary_handoff_time = self._validate_prediction_value(
            self._get_pp_stage_boundary_handoff_time(batch, stage_id),
            "pp_stage_boundary_handoff",
            batch,
            f"stage={stage_id}",
        )
        decode_draft_proposer_time = 0.0
        spec_metadata = getattr(batch, "spec_decode_metadata", None)
        if self._should_include_spec_decode_proposer_overhead(batch):
            decode_draft_proposer_time = self._validate_prediction_value(
                self._get_spec_decode_proposer_overhead_time(
                    batch,
                    method_name=str(spec_metadata.method),
                ),
                "decode_draft_proposer",
                batch,
                f"stage={stage_id}",
            )
        mtp_terminal_overshoot_time = self._validate_prediction_value(
            self._get_mtp_terminal_overshoot_time(
                batch,
                stage_id=stage_id,
                cluster_type=cluster_type,
                num_layers=num_layers,
                layer_id=layer_id,
            ),
            "mtp_terminal_overshoot",
            batch,
            f"stage={stage_id}",
        )

        quant_manager = get_quantization_manager()
        attn_rope_time = quant_manager.adjust_compute_time(
            "attn_rope", attn_rope_time, self._cluster_type
        )
        attn_kv_save_time = quant_manager.adjust_compute_time(
            self._dense_attention_cache_write_op_name(),
            attn_kv_save_time,
            self._cluster_type,
        )
        attn_decode_time = quant_manager.adjust_compute_time(
            self._dense_attention_decode_op_name(),
            attn_decode_time,
            self._cluster_type,
        )
        attn_prefill_time = quant_manager.adjust_compute_time(
            self._dense_attention_prefill_op_name(),
            attn_prefill_time,
            self._cluster_type,
        )
        attn_pre_proj_time = quant_manager.adjust_compute_time(
            "attn_pre_proj", attn_pre_proj_time, self._cluster_type
        )
        attn_post_proj_time = quant_manager.adjust_compute_time(
            "attn_post_proj", attn_post_proj_time, self._cluster_type
        )
        mlp_up_proj_time = quant_manager.adjust_compute_time(
            "mlp_up_proj", mlp_up_proj_time, self._cluster_type
        )
        mlp_down_proj_time = quant_manager.adjust_compute_time(
            "mlp_down_proj", mlp_down_proj_time, self._cluster_type
        )
        mlp_act_time = quant_manager.adjust_compute_time(
            "mlp_act", mlp_act_time, self._cluster_type
        )
        attn_norm_time = quant_manager.adjust_compute_time(
            "input_layernorm", attn_norm_time, self._cluster_type
        )
        mlp_norm_time = quant_manager.adjust_compute_time(
            "post_attention_layernorm", mlp_norm_time, self._cluster_type
        )
        add_time = quant_manager.adjust_compute_time(
            "add", add_time, self._cluster_type
        )

        # Communication times already predicted by CC backend paths (or explicit dummy mode)
        tp_comm_time = tensor_parallel_communication_time
        attn_tp_allreduce_time = tp_comm_time
        ffn_tp_allreduce_time = tp_comm_time
        pp_comm_time = pipeline_parallel_communication_time

        logger.debug(
            f"[EXEC_TIME_SUMMARY] batch_id={batch.id}, stage={stage_id}: "
            f"attn_total={(attn_rope_time + attn_kv_save_time + attn_decode_time + attn_prefill_time + attn_pre_proj_time + attn_post_proj_time):.6f}ms, "
            f"mlp_total={(mlp_up_proj_time + mlp_down_proj_time + mlp_act_time):.6f}ms, "
            f"comm_total={(tp_comm_time + pp_comm_time):.6f}ms"
        )

        base_execution_time = ExecutionTime(
            num_layers_per_pipeline_stage=1,  # Single layer
            attention_rope_execution_time=attn_rope_time,
            attention_kv_cache_save_execution_time=attn_kv_save_time,
            attention_decode_execution_time=attn_decode_time,
            attention_prefill_execution_time=attn_prefill_time,
            attention_layer_pre_proj_execution_time=attn_pre_proj_time,
            attention_layer_post_proj_execution_time=attn_post_proj_time,
            mlp_layer_up_proj_execution_time=mlp_up_proj_time,
            mlp_layer_down_proj_execution_time=mlp_down_proj_time,
            mlp_layer_act_execution_time=mlp_act_time,
            attn_norm_time=attn_norm_time,
            mlp_norm_time=mlp_norm_time,
            add_time=add_time,
            tensor_parallel_communication_time=tp_comm_time,
            attn_tensor_parallel_allreduce_time=attn_tp_allreduce_time,
            moe_tensor_parallel_allreduce_time=ffn_tp_allreduce_time,
            pipeline_parallel_communication_time=pp_comm_time,
            expert_parallel_communication_time=0.0,
            moe_gating_time=0.0,
            moe_shuffling_time=0.0,
            schedule_time=schedule_time,
            sampler_e2e_time=sampler_time,
            prepare_inputs_e2e_time=prepare_inputs_time,
            process_model_outputs_time=process_outputs_time,
            ray_comm_time=ray_comm_time,
            is_moe=False,
            pp_producer_send_path_runtime_time=pp_producer_send_path_runtime_time,
            pp_receiver_head_runtime_time=pp_receiver_head_runtime_time,
            pp_prefill_consumer_active_runtime_time=pp_prefill_consumer_active_runtime_time,
            pp_stage_boundary_handoff_time=pp_stage_boundary_handoff_time,
            decode_draft_proposer_time=decode_draft_proposer_time,
            mtp_terminal_overshoot_time=mtp_terminal_overshoot_time,
            attention_operator_times=attention_time.operator_times,
            communication_operator_times=CommunicationOperatorTimes(
                communication_operator_times
            ),
            mlp_operator_times=MLPOperatorTimes(
                op_times={
                    "post_attention_layernorm": mlp_norm_time,
                    "mlp_up_proj": mlp_up_proj_time,
                    "mlp_act": mlp_act_time,
                    "mlp_down_proj": mlp_down_proj_time,
                }
            ),
        )

        # If num_layers is 1, return as-is
        if num_layers == 1:
            return base_execution_time

        logger.debug(
            "Aggregating dense execution time with num_layers_per_pipeline_stage=%s",
            num_layers,
        )

        # Keep component fields at single-layer granularity and let ExecutionTime
        # apply layer aggregation via num_layers_per_pipeline_stage.
        return ExecutionTime(
            num_layers_per_pipeline_stage=num_layers,
            attention_rope_execution_time=base_execution_time._attention_rope_execution_time,
            attention_kv_cache_save_execution_time=base_execution_time._attention_kv_cache_save_execution_time,
            attention_decode_execution_time=base_execution_time._attention_decode_execution_time,
            attention_prefill_execution_time=base_execution_time._attention_prefill_execution_time,
            attention_layer_pre_proj_execution_time=base_execution_time._attention_layer_pre_proj_execution_time,
            attention_layer_post_proj_execution_time=base_execution_time._attention_layer_post_proj_execution_time,
            attn_norm_time=base_execution_time._attn_norm_time,
            mlp_norm_time=base_execution_time._mlp_norm_time,
            add_time=base_execution_time._add_time,
            tensor_parallel_communication_time=base_execution_time._tensor_parallel_communication_time,
            attn_tensor_parallel_allreduce_time=(
                base_execution_time._attn_tensor_parallel_allreduce_time
                if base_execution_time._has_attn_tensor_parallel_allreduce_time
                else None
            ),
            moe_tensor_parallel_allreduce_time=(
                base_execution_time._moe_tensor_parallel_allreduce_time
                if base_execution_time._has_moe_tensor_parallel_allreduce_time
                else None
            ),
            pipeline_parallel_communication_time=base_execution_time._pipeline_parallel_communication_time,
            expert_parallel_communication_time=base_execution_time._expert_parallel_communication_time,
            moe_gating_time=0.0,
            moe_shuffling_time=0.0,
            schedule_time=base_execution_time._schedule_time,
            sampler_e2e_time=base_execution_time._sampler_e2e_time,
            prepare_inputs_e2e_time=base_execution_time._prepare_inputs_e2e_time,
            process_model_outputs_time=base_execution_time._process_model_outputs_time,
            ray_comm_time=base_execution_time._ray_comm_time,
            is_moe=False,
            pp_producer_send_path_runtime_time=base_execution_time._pp_producer_send_path_runtime_time,
            pp_receiver_head_runtime_time=base_execution_time._pp_receiver_head_runtime_time,
            pp_prefill_consumer_active_runtime_time=base_execution_time._pp_prefill_consumer_active_runtime_time,
            pp_stage_boundary_handoff_time=base_execution_time._pp_stage_boundary_handoff_time,
            mlp_layer_up_proj_execution_time=base_execution_time._mlp_layer_up_proj_execution_time,
            mlp_layer_down_proj_execution_time=base_execution_time._mlp_layer_down_proj_execution_time,
            mlp_layer_act_execution_time=base_execution_time._mlp_layer_act_execution_time,
            decode_draft_proposer_time=base_execution_time._decode_draft_proposer_time,
            mtp_terminal_overshoot_time=base_execution_time._mtp_terminal_overshoot_time,
            attention_operator_times=base_execution_time.attention_operator_times,
            communication_operator_times=(
                base_execution_time.communication_operator_times
            ),
            mlp_operator_times=base_execution_time.mlp_operator_times,
        )
