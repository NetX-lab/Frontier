from __future__ import annotations

import hashlib
import math
import json
import threading
from pathlib import Path
from typing import Dict, Optional, Set, Any, List

from frontier.attention.families import (
    DENSE_ATTENTION_FAMILY,
    iter_execution_enabled_families,
)
from frontier.config.model_config import BaseModelConfig
from frontier.config.precision_type import PrecisionType, PrecisionMismatchInfo
from frontier.logger import init_logger
from frontier.types import ClusterType

logger = init_logger(__name__)


def _attention_compute_operations() -> List[str]:
    operation_names: List[str] = []
    for family in iter_execution_enabled_families():
        active_family = (
            DENSE_ATTENTION_FAMILY
            if family.family_id == DENSE_ATTENTION_FAMILY.family_id
            else family
        )
        operation_names.extend(
            operator.name for operator in active_family.profiling_ops()
        )
    return operation_names


class QuantizationManager:
    """Thread-safe manager for model-config-driven precision settings.

    Precision and quantization settings are derived from model config only.
    Operation-level quantization JSON files are no longer supported.
    """

    _instance: Optional["QuantizationManager"] = None
    _instance_lock = threading.Lock()
    _initialized = False

    DEFAULT_CONFIG_DIR = Path("data/config/op_quantization")
    DEFAULT_CONFIG_FILE = "default.json"
    REGISTRY_FILE = "supported_operations.json"
    MODEL_CONFIG_QUANT_OPS = {
        "attn_pre_proj",
        "attn_post_proj",
        "mlp_up_proj",
        "mlp_down_proj",
        "moe_grouped_gemm",
    }

    def __new__(cls) -> "QuantizationManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self) -> None:
        with self._instance_lock:
            if self._initialized:
                return

            self._lock = threading.RLock()
            self._config_path: Optional[str] = None
            self._default_precision = PrecisionType.FP16
            self._profiling_precision = PrecisionType.FP16
            self._operation_precisions: Dict[str, PrecisionType] = {}
            self._operation_speedup_factors: Dict[str, float] = {}
            self._operation_precision_sources: Dict[str, str] = {}
            self._operation_data_sources: Dict[str, str] = {}
            self._operation_approximation_factors: Dict[str, float] = {}
            self._operation_profiling_precision: Dict[str, PrecisionType] = {}
            self._cluster_overrides: Dict[ClusterType, Dict[str, PrecisionType]] = {}
            self._supported_operations: Dict[str, Any] = {}
            self._warned_mismatches: Set[tuple] = set()
            self._warned_approximations: Set[tuple] = set()
            self._config: Dict[str, Any] = {}
            self._precision_mismatches: Set[PrecisionMismatchInfo] = set()

            self._load_registry()
            self._initialized = True

    # note: currently omit the registry：supported_operations.json from data/config/op_quantization
    def _get_default_registry(self) -> Dict[str, Any]:
        return {
            "compute_operations": [
                "attn_inter_norm",
                "attn_pre_proj",
                "attn_post_proj",
                "attn_rope",
                *_attention_compute_operations(),
                "attn_wq_proj",
                "mlp_up_proj",
                "mlp_down_proj",
                "mlp_act",
                "moe_gating",
                "moe_gating_linear",
                "moe_gating_routing_topk",
                "moe_shuffling",
                "moe_grouped_gemm",
                "share_expert_up_proj",
                "share_expert_down_proj",
                "share_expert_act",
                "input_layernorm",
                "post_attention_layernorm",
                "add",
                "emb",
                "mtp_fusion_proj",
                "lm_head_linear",
            ],
            "communication_operations": [
                "allreduce",
                "allgather",
                "broadcast",
                "send_recv",
                "reduce_scatter",
                "all_to_all",
                "kv_cache_transfer",
                "m2n_transfer",
                "expert_parallel_communication",
            ]
        }

    def _load_registry(self) -> None:
        registry_path = self.DEFAULT_CONFIG_DIR / self.REGISTRY_FILE
        if registry_path.exists():
            logger.info(
                "Ignoring op_quantization registry at %s; using built-in defaults",
                registry_path,
            )
        self._supported_operations = self._get_default_registry()

    def load_config(self, config_path: Optional[str] = None) -> None:
        if config_path:
            raise NotImplementedError(
                "Operation-level quantization configs are deprecated. "
                "Use model config (torch_dtype + quantization_config) only."
            )
        with self._lock:
            self._config_path = None
            self._default_precision = PrecisionType.FP16
            self._profiling_precision = PrecisionType.FP16
            self._operation_precisions = {}
            self._operation_speedup_factors = {}
            self._operation_precision_sources = {}
            self._operation_data_sources = {}
            self._operation_approximation_factors = {}
            self._operation_profiling_precision = {}
            self._cluster_overrides = {}
            self._precision_mismatches = set()
            self._warned_mismatches = set()
            self._warned_approximations = set()
            self._config = {}
            logger.info(
                "Op-level quantization config disabled; awaiting model config setup"
            )

    def configure_from_model_config(self, model_config: BaseModelConfig) -> None:
        if model_config is None:
            raise ValueError("Model config is required for quantization setup.")
        with self._lock:
            self._config_path = "model_config"
            self._default_precision = model_config.get_default_precision()
            self._profiling_precision = self._default_precision
            self._operation_precisions = {}
            self._operation_speedup_factors = {}
            self._operation_precision_sources = {}
            self._operation_data_sources = {}
            self._operation_approximation_factors = {}
            self._operation_profiling_precision = {}
            self._cluster_overrides = {}
            self._precision_mismatches = set()
            self._warned_mismatches = set()
            self._warned_approximations = set()

            compute_ops = self._supported_operations.get("compute_operations", [])
            for op_name in compute_ops:
                self._operation_precision_sources[op_name] = "torch_dtype"
                self._operation_data_sources[op_name] = "profiling"

            quant_config = model_config.quantization_config
            if quant_config is not None and quant_config.quant_method is not None:
                quant_precision = PrecisionType.from_string(quant_config.quant_method)
                for op_name in self.MODEL_CONFIG_QUANT_OPS:
                    if not self._is_operation_supported(op_name):
                        registry_path = self.DEFAULT_CONFIG_DIR / self.REGISTRY_FILE
                        raise ValueError(
                            f"Unsupported operation '{op_name}' in model quantization config. "
                            f"See registry: {registry_path}"
                        )
                    self._operation_precisions[op_name] = quant_precision
                    self._operation_precision_sources[op_name] = "quantization_config"

            self._config = {
                "source": "model_config",
                "model_name": model_config.get_name(),
                "torch_dtype": model_config.torch_dtype,
                "quantization_config": (
                    model_config.quantization_config.to_dict()
                    if model_config.quantization_config is not None
                    else None
                ),
                "quant_signature": model_config.get_quant_signature(),
            }

            logger.info(
                "Quantization configured from model config: model=%s default=%s quant_method=%s",
                model_config.get_name(),
                self._default_precision.name,
                quant_config.quant_method if quant_config is not None else None,
            )

    def _resolve_config_path(self, config_path: Optional[str]) -> Optional[str]:
        if config_path:
            return config_path

        default_path = self.DEFAULT_CONFIG_DIR / self.DEFAULT_CONFIG_FILE
        if default_path.exists():
            return str(default_path)
        return None

    def _validate_config(self, config: Dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise ValueError("Quantization config must be a JSON object")

        if "version" not in config:
            raise ValueError("Quantization config missing required field: version")
        if "default_precision" not in config:
            raise ValueError("Quantization config missing required field: default_precision")

        operations = config.get("operations", {})
        if not isinstance(operations, dict):
            raise ValueError("Quantization config field 'operations' must be a JSON object")

        for op_name, op_config in operations.items():
            if not self._is_operation_supported(op_name):
                registry_path = self.DEFAULT_CONFIG_DIR / self.REGISTRY_FILE
                logger.warning("Unsupported operation in config: %s", op_name)
                raise ValueError(
                    f"Unsupported operation '{op_name}'. See registry: {registry_path}"
                )
            if not isinstance(op_config, dict):
                raise ValueError(
                    f"Operation '{op_name}' must map to a JSON object with precision settings"
                )
            if "precision" in op_config:
                PrecisionType.from_string(op_config["precision"])
            if "compute_speedup_factor" in op_config:
                speedup = op_config["compute_speedup_factor"]
                if not isinstance(speedup, (int, float)) or speedup <= 0:
                    raise ValueError(
                        f"Operation '{op_name}' compute_speedup_factor must be > 0"
                    )

        cluster_overrides = config.get("cluster_overrides", {})
        if not isinstance(cluster_overrides, dict):
            raise ValueError("Quantization config field 'cluster_overrides' must be a JSON object")

        for cluster_name, cluster_ops in cluster_overrides.items():
            if cluster_name.upper() not in ClusterType.__members__:
                raise ValueError(
                    f"Unsupported cluster type '{cluster_name}' in cluster_overrides"
                )
            if not isinstance(cluster_ops, dict):
                raise ValueError(
                    f"cluster_overrides['{cluster_name}'] must be a JSON object"
                )
            for op_name, op_config in cluster_ops.items():
                if not self._is_operation_supported(op_name):
                    registry_path = self.DEFAULT_CONFIG_DIR / self.REGISTRY_FILE
                    logger.warning(
                        "Unsupported operation in cluster_overrides: %s", op_name
                    )
                    raise ValueError(
                        f"Unsupported operation '{op_name}'. See registry: {registry_path}"
                    )
                if not isinstance(op_config, dict):
                    raise ValueError(
                        f"cluster_overrides['{cluster_name}']['{op_name}'] must be a JSON object"
                    )
                if "precision" not in op_config:
                    raise ValueError(
                        f"cluster_overrides['{cluster_name}']['{op_name}'] missing precision"
                    )
                PrecisionType.from_string(op_config["precision"])
                extra_keys = set(op_config.keys()) - {"precision"}
                if extra_keys:
                    raise ValueError(
                        f"cluster_overrides['{cluster_name}']['{op_name}'] has unsupported fields: {sorted(extra_keys)}"
                    )

    def _extract_config(self, config: Dict[str, Any]) -> None:
        self._default_precision = PrecisionType.from_string(
            config.get("default_precision", "FP16")
        )
        self._profiling_precision = PrecisionType.from_string(
            config.get("profiling_precision", "FP16")
        )

        operations = config.get("operations", {})
        self._operation_precisions = {}
        self._operation_speedup_factors = {}
        for op_name, op_config in operations.items():
            if "precision" in op_config:
                self._operation_precisions[op_name] = PrecisionType.from_string(
                    op_config["precision"]
                )
            if "compute_speedup_factor" in op_config:
                self._operation_speedup_factors[op_name] = float(
                    op_config["compute_speedup_factor"]
                )

        self._cluster_overrides = {}
        cluster_overrides = config.get("cluster_overrides", {})
        for cluster_name, cluster_ops in cluster_overrides.items():
            cluster_type = ClusterType[cluster_name.upper()]
            self._cluster_overrides[cluster_type] = {}
            for op_name, op_config in cluster_ops.items():
                self._cluster_overrides[cluster_type][op_name] = PrecisionType.from_string(
                    op_config["precision"]
                )

    def _is_operation_supported(self, operation_name: str) -> bool:
        compute_ops = set(self._supported_operations.get("compute_operations", []))
        comm_ops = set(self._supported_operations.get("communication_operations", []))
        return operation_name in compute_ops or operation_name in comm_ops

    def is_operation_supported(self, operation_name: str) -> bool:
        return self._is_operation_supported(operation_name)

    def is_compute_operation(self, operation_name: str) -> bool:
        return operation_name in set(self._supported_operations.get("compute_operations", []))

    def is_communication_operation(self, operation_name: str) -> bool:
        return operation_name in set(
            self._supported_operations.get("communication_operations", [])
        )

    def get_supported_operations(self) -> Dict[str, List[str]]:
        with self._lock:
            compute_ops = sorted(
                self._supported_operations.get("compute_operations", [])
            )
            comm_ops = sorted(
                self._supported_operations.get("communication_operations", [])
            )
        return {
            "compute_operations": compute_ops,
            "communication_operations": comm_ops,
        }

    def register_profiling_metadata(
        self,
        operation_names: List[str],
        profiling_precision: PrecisionType,
        profiling_quant_signature: str,
        expected_quant_signature: str,
        file_path: str,
    ) -> None:
        for op_name in operation_names:
            if not self._is_operation_supported(op_name):
                registry_path = self.DEFAULT_CONFIG_DIR / self.REGISTRY_FILE
                raise ValueError(
                    f"Unsupported operation '{op_name}'. See registry: {registry_path}"
                )
            target_precision = self.get_precision(op_name)
            precision_match = target_precision == profiling_precision
            quant_match = profiling_quant_signature == expected_quant_signature
            with self._lock:
                self._operation_profiling_precision[op_name] = profiling_precision
                if precision_match:
                    self._operation_data_sources[op_name] = "profiling"
                    self._operation_approximation_factors.pop(op_name, None)
                    self._operation_speedup_factors.pop(op_name, None)
                    if not quant_match:
                        warning_key = (
                            op_name,
                            profiling_precision.name,
                            target_precision.name,
                            profiling_quant_signature,
                            expected_quant_signature,
                            "quant_signature",
                        )
                        if warning_key not in self._warned_approximations:
                            self._warned_approximations.add(warning_key)
                            logger.warning(
                                "Quant signature mismatch for op=%s (profiling=%s expected=%s). "
                                "Using profiling data without approximation. File=%s",
                                op_name,
                                profiling_quant_signature,
                                expected_quant_signature,
                                file_path,
                            )
                    continue

                scale = self._get_approximation_scale(
                    target_precision, profiling_precision
                )
                self._operation_data_sources[op_name] = "approximation"
                self._operation_approximation_factors[op_name] = scale
                if scale <= 0:
                    raise ValueError(
                        f"Invalid approximation scale {scale} for op '{op_name}'"
                    )
                self._operation_speedup_factors[op_name] = 1.0 / scale
                warning_key = (
                    op_name,
                    profiling_precision.name,
                    target_precision.name,
                    profiling_quant_signature,
                    expected_quant_signature,
                    "precision_mismatch",
                )
                if warning_key not in self._warned_approximations:
                    self._warned_approximations.add(warning_key)
                    reason_parts = [
                        f"precision mismatch ({profiling_precision.name} -> {target_precision.name})"
                    ]
                    if not quant_match:
                        reason_parts.append(
                            f"quant_signature mismatch ({profiling_quant_signature} != {expected_quant_signature})"
                        )
                    logger.warning(
                        "Profiling metadata mismatch for op=%s (%s). Using approximation factor=%.3f. File=%s",
                        op_name,
                        "; ".join(reason_parts),
                        scale,
                        file_path,
                    )

    def get_operation_precision_metadata(self) -> List[Dict[str, Any]]:
        metadata = []
        compute_ops = self._supported_operations.get("compute_operations", [])
        for op_name in sorted(compute_ops):
            precision = self.get_precision(op_name)
            data_source = self._operation_data_sources.get(op_name, "profiling")
            approx_factor = self._operation_approximation_factors.get(op_name)
            metadata.append(
                {
                    "operation": op_name,
                    "precision": precision.name,
                    "data_source": data_source,
                    "approximation_factor": approx_factor,
                }
            )
        return metadata

    def _get_approximation_scale(
        self, target_precision: PrecisionType, profiling_precision: PrecisionType
    ) -> float:
        if target_precision in {PrecisionType.FP8, PrecisionType.INT8}:
            return 0.5
        return target_precision.get_compute_scaling_factor(profiling_precision)

    def has_explicit_precision(
        self, operation_name: str, cluster_type: Optional[ClusterType] = None
    ) -> bool:
        if cluster_type in self._cluster_overrides:
            if operation_name in self._cluster_overrides[cluster_type]:
                return True
        return operation_name in self._operation_precisions

    def get_precision(
        self, operation_name: str, cluster_type: Optional[ClusterType] = None
    ) -> PrecisionType:
        if not self._is_operation_supported(operation_name):
            registry_path = self.DEFAULT_CONFIG_DIR / self.REGISTRY_FILE
            raise ValueError(
                f"Unsupported operation '{operation_name}'. See registry: {registry_path}"
            )

        with self._lock:
            if cluster_type in self._cluster_overrides:
                cluster_ops = self._cluster_overrides[cluster_type]
                if operation_name in cluster_ops:
                    precision = cluster_ops[operation_name]
                    logger.debug(
                        "Precision lookup: op=%s cluster=%s precision=%s",
                        operation_name,
                        cluster_type.name,
                        precision.name,
                    )
                    return precision

            if operation_name in self._operation_precisions:
                precision = self._operation_precisions[operation_name]
                logger.debug(
                    "Precision lookup: op=%s precision=%s",
                    operation_name,
                    precision.name,
                )
                return precision

            logger.debug(
                "Precision lookup: op=%s using default precision=%s",
                operation_name,
                self._default_precision.name,
            )
            return self._default_precision

    def get_bytes_per_element(
        self, operation_name: str, cluster_type: Optional[ClusterType] = None
    ) -> float:
        return self.get_precision(operation_name, cluster_type).bytes_per_element

    def get_tensor_size_multiplier(
        self, operation_name: str, cluster_type: Optional[ClusterType] = None
    ) -> float:
        precision = self.get_precision(operation_name, cluster_type)
        return precision.get_size_multiplier()

    def get_compute_speedup_factor(
        self, operation_name: str, cluster_type: Optional[ClusterType] = None
    ) -> float:
        _ = cluster_type
        return float(self._operation_speedup_factors.get(operation_name, 1.0))

    def get_profiling_precision(self) -> PrecisionType:
        return self._profiling_precision

    def get_default_precision(self) -> PrecisionType:
        return self._default_precision

    def get_config_path(self) -> Optional[str]:
        return self._config_path

    def get_custom_operation_count(self) -> int:
        return len(self._operation_precisions)

    def has_precision_mismatch(
        self, operation_name: str, cluster_type: Optional[ClusterType] = None
    ) -> bool:
        precision = self.get_precision(operation_name, cluster_type)
        profiling_precision = self._operation_profiling_precision.get(
            operation_name, self._profiling_precision
        )
        return precision != profiling_precision

    def adjust_tensor_size(
        self,
        operation_name: str,
        original_size_bytes: int,
        cluster_type: Optional[ClusterType] = None,
    ) -> int:
        if original_size_bytes < 0:
            raise ValueError("original_size_bytes must be >= 0")
        multiplier = self.get_tensor_size_multiplier(operation_name, cluster_type)
        scaled = original_size_bytes * multiplier
        if original_size_bytes == 0:
            adjusted = 0
        else:
            adjusted = max(1, int(math.ceil(scaled)))
        logger.debug(
            "Tensor size adjustment: op=%s cluster=%s original=%d adjusted=%d multiplier=%.4f",
            operation_name,
            cluster_type.name if cluster_type else None,
            original_size_bytes,
            adjusted,
            multiplier,
        )
        return adjusted

    def adjust_compute_time(
        self,
        operation_name: str,
        predicted_time: float,
        cluster_type: Optional[ClusterType] = None,
    ) -> float:
        if operation_name in self._operation_speedup_factors:
            speedup = self._operation_speedup_factors[operation_name]
            adjusted = predicted_time / speedup
            logger.debug(
                "Compute time adjustment (speedup): op=%s original=%.6f adjusted=%.6f factor=%.4f",
                operation_name,
                predicted_time,
                adjusted,
                speedup,
            )
            return adjusted

        if not self.has_precision_mismatch(operation_name, cluster_type):
            return predicted_time

        precision = self.get_precision(operation_name, cluster_type)
        profiling_precision = self._profiling_precision
        scaling = precision.get_compute_scaling_factor(profiling_precision)
        adjusted = predicted_time * scaling
        self.check_and_warn_mismatch(operation_name, cluster_type)
        logger.debug(
            "Compute time adjustment (scaling): op=%s original=%.6f adjusted=%.6f scaling=%.4f",
            operation_name,
            predicted_time,
            adjusted,
            scaling,
        )
        return adjusted

    def check_and_warn_mismatch(
        self, operation_name: str, cluster_type: Optional[ClusterType]
    ) -> None:
        mismatch = self.get_precision_mismatch_info(operation_name, cluster_type)
        if mismatch is None:
            return

        key = (
            mismatch.operation_name,
            mismatch.cluster_type,
            mismatch.profiling_precision.name,
            mismatch.configured_precision.name,
        )
        if key in self._warned_mismatches:
            return
        self._warned_mismatches.add(key)
        logger.warning(mismatch.get_warning_message())

    def get_precision_mismatch_info(
        self, operation_name: str, cluster_type: Optional[ClusterType] = None
    ) -> Optional[PrecisionMismatchInfo]:
        if not self.has_precision_mismatch(operation_name, cluster_type):
            return None
        precision = self.get_precision(operation_name, cluster_type)
        profiling_precision = self._operation_profiling_precision.get(
            operation_name, self._profiling_precision
        )
        return PrecisionMismatchInfo(
            operation_name=operation_name,
            configured_precision=precision,
            profiling_precision=profiling_precision,
            cluster_type=cluster_type.name if cluster_type else None,
        )

    def get_precision_mismatch_summary(self) -> List[PrecisionMismatchInfo]:
        return list(self._precision_mismatches)

    def _detect_precision_mismatches(self) -> None:
        self._precision_mismatches = set()
        supported_ops = (
            self._supported_operations.get("compute_operations", [])
            + self._supported_operations.get("communication_operations", [])
        )
        cluster_types = [None] + list(self._cluster_overrides.keys())
        for operation in supported_ops:
            for cluster_type in cluster_types:
                if self.has_precision_mismatch(operation, cluster_type):
                    mismatch = self.get_precision_mismatch_info(operation, cluster_type)
                    if mismatch is not None:
                        self._precision_mismatches.add(mismatch)
                        self.check_and_warn_mismatch(operation, cluster_type)

    def print_config_summary(self) -> None:
        config_path = self._config_path or "none"
        custom_operations = sorted(self._operation_precisions.keys())
        cluster_overrides = {
            cluster.name: sorted(ops.keys())
            for cluster, ops in self._cluster_overrides.items()
        }
        logger.info(
            "Quantization config summary: source=%s default=%s profiling=%s custom_ops=%d",
            config_path,
            self._default_precision.name,
            self._profiling_precision.name,
            len(custom_operations),
        )
        if custom_operations:
            logger.info("Quantization custom operations: %s", custom_operations)
        if cluster_overrides:
            logger.info("Quantization cluster overrides: %s", cluster_overrides)

    def _build_config_dict(self) -> Dict[str, Any]:
        config = {
            "version": self._config.get("version", "1.0"),
            "default_precision": self._default_precision.name,
            "profiling_precision": self._profiling_precision.name,
            "operations": {},
        }

        for op_name in sorted(
            set(self._operation_precisions.keys())
            | set(self._operation_speedup_factors.keys())
        ):
            config["operations"][op_name] = {}
            if op_name in self._operation_precisions:
                config["operations"][op_name]["precision"] = self._operation_precisions[
                    op_name
                ].name
            if op_name in self._operation_speedup_factors:
                config["operations"][op_name][
                    "compute_speedup_factor"
                ] = self._operation_speedup_factors[op_name]

        if self._cluster_overrides:
            config["cluster_overrides"] = {}
            for cluster_type, ops in self._cluster_overrides.items():
                config["cluster_overrides"][cluster_type.name] = {
                    op_name: {"precision": precision.name}
                    for op_name, precision in ops.items()
                }

        return config

    def save_config(self, path: str) -> None:
        if not path:
            raise ValueError("Config save path must be provided")
        config = self._config if self._config else self._build_config_dict()
        with Path(path).open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2, sort_keys=True)

    def get_config_hash(self) -> str:
        config = self._config if self._config else self._build_config_dict()
        config_json = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    @classmethod
    def reset(cls) -> None:
        with cls._instance_lock:
            cls._instance = None
            cls._initialized = False


def get_quantization_manager() -> QuantizationManager:
    return QuantizationManager()
