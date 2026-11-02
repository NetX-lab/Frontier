from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PrecisionType(Enum):
    """Supported precision types.

    Note: BF16 and FP16 are equivalent for tensor size calculation, but must
    remain distinct enum members for accurate logging and configuration flow.
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    FP4 = "fp4"
    INT4 = "int4"

    @property
    def bytes_per_element(self) -> float:
        """Return bytes per element for this precision type."""
        return float(_PRECISION_BYTES[self.name])

    @classmethod
    def from_string(cls, name: str) -> "PrecisionType":
        """Parse precision type from string (case-insensitive)."""
        if not isinstance(name, str):
            valid_types = [t.name for t in cls]
            raise ValueError(
                f"Unsupported precision type: '{name}'. Supported types: {valid_types}"
            )
        normalized = name.strip().lower().replace("torch.", "")
        alias_map = {
            "float32": "fp32",
            "float16": "fp16",
            "half": "fp16",
            "bfloat16": "bf16",
            "fbgemm_fp8": "fp8",
        }
        normalized = alias_map.get(normalized, normalized)
        try:
            return cls[normalized.upper()]
        except KeyError as exc:
            valid_types = [t.name for t in cls]
            raise ValueError(
                f"Unsupported precision type: '{name}'. Supported types: {valid_types}"
            ) from exc

    def get_size_multiplier(self, baseline: Optional["PrecisionType"] = None) -> float:
        """Return size multiplier relative to baseline (default: FP16)."""
        if baseline is None:
            baseline = PrecisionType.FP16
        return self.bytes_per_element / baseline.bytes_per_element

    def get_compute_scaling_factor(self, profiling_precision: "PrecisionType") -> float:
        """Return compute time scaling factor relative to profiling precision."""
        return self.bytes_per_element / profiling_precision.bytes_per_element

    @classmethod
    def from_torch_dtype(cls, dtype: str) -> "PrecisionType":
        """Map torch dtype strings to PrecisionType."""
        if not isinstance(dtype, str):
            valid_types = [t.name for t in cls]
            raise ValueError(
                f"Unsupported torch dtype: '{dtype}'. Supported precision types: {valid_types}"
            )
        normalized = dtype.strip().lower().replace("torch.", "")
        mapping = {
            "float32": cls.FP32,
            "fp32": cls.FP32,
            "float16": cls.FP16,
            "fp16": cls.FP16,
            "half": cls.FP16,
            "bfloat16": cls.BF16,
            "bf16": cls.BF16,
        }
        if normalized in mapping:
            return mapping[normalized]
        valid_types = [t.name for t in cls]
        raise ValueError(
            f"Unsupported torch dtype: '{dtype}'. Supported precision types: {valid_types}"
        )


_PRECISION_BYTES = {
    "FP32": 4.0,
    "FP16": 2.0,
    "BF16": 2.0,
    "FP8": 1.0,
    "INT8": 1.0,
    "FP4": 0.5,
    "INT4": 0.5,
}


@dataclass(frozen=True)
class PrecisionMismatchInfo:
    """Structured metadata for precision mismatches against profiling data.

    Example:
        >>> info = PrecisionMismatchInfo(
        ...     operation_name="mlp_up_proj",
        ...     configured_precision=PrecisionType.FP8,
        ...     profiling_precision=PrecisionType.FP16,
        ...     cluster_type="PREFILL",
        ... )
        >>> info.size_ratio
        0.5
        >>> info.get_warning_message()
        'Precision mismatch for mlp_up_proj (cluster=PREFILL): profiling=FP16 simulation=FP8. Collect precision-specific profiling data for best accuracy.'
    """
    operation_name: str
    configured_precision: PrecisionType
    profiling_precision: PrecisionType
    cluster_type: Optional[str] = None

    @property
    def size_ratio(self) -> float:
        return (
            self.configured_precision.bytes_per_element
            / self.profiling_precision.bytes_per_element
        )

    @property
    def estimated_speedup(self) -> float:
        return (
            self.profiling_precision.bytes_per_element
            / self.configured_precision.bytes_per_element
        )

    def get_warning_message(self) -> str:
        cluster = self.cluster_type or "None"
        return (
            f"Precision mismatch for {self.operation_name} (cluster={cluster}): "
            f"profiling={self.profiling_precision.name} "
            f"simulation={self.configured_precision.name}. "
            "Collect precision-specific profiling data for best accuracy."
        )
