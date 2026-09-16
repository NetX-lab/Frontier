"""CPU-safe feature construction shared by GDN training and prediction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from frontier.entities.batch import Batch

from frontier.attention.families import GATED_DELTA_NET_ATTENTION_FAMILY
from frontier.attention.ops import AttentionPhase
from frontier.types import MeasurementType


GDN_TASKS = tuple(
    (operator.name, phase.value)
    for phase in (AttentionPhase.PREFILL, AttentionPhase.DECODE)
    for operator in GATED_DELTA_NET_ATTENTION_FAMILY.predictor_ops()
    if phase in operator.phases
)
GDN_ARTIFACT_SCHEMA_VERSION = 1
GDN_IDENTITY_COLUMNS = tuple(
    column
    for column in GATED_DELTA_NET_ATTENTION_FAMILY.required_profiling_feature_columns
    if column not in {
        "batch_size", "batch_num_tokens", "batch_num_prefill_tokens",
        "batch_num_decode_tokens", "max_query_len", "has_initial_state",
    }
)
_GDN_INTEGER_IDENTITY_FIELDS = frozenset({
    "tensor_parallel_size", "hidden_size", "conv_kernel_size", "key_head_dim",
    "value_head_dim", "num_key_heads", "num_value_heads",
})


def validate_gdn_artifact_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the complete selected model/runtime identity at the boundary."""

    normalized = dict(identity)
    required = tuple(
        "tensor_parallel_size" if column == "num_tensor_parallel_workers" else column
        for column in GDN_IDENTITY_COLUMNS
    ) + ("dataset_fingerprint",)
    for name in required:
        value = identity.get(name)
        if value is None or not str(value).strip() or (
            isinstance(value, float) and not math.isfinite(value)
        ):
            raise ValueError(f"GDN identity requires nonempty {name}")
        if name in _GDN_INTEGER_IDENTITY_FIELDS:
            try:
                integer = int(value)
                valid = integer > 0 and float(value) == integer
            except (ValueError, TypeError, OverflowError):
                valid = False
            if not valid:
                raise ValueError(f"GDN identity {name} must be a positive integer")
            normalized[name] = integer
        else:
            normalized[name] = str(value).strip()
    normalized["measurement_type"] = MeasurementType.from_string(
        normalized["measurement_type"]
    ).value
    return normalized


GDN_FEATURE_COLUMNS = (
    "batch_size",
    "batch_num_tokens",
    "max_query_len",
    "query_len_cv",
    "num_stateful_requests",
)


def _positive_int(value: Any, field_name: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if integer <= 0:
        raise ValueError(f"{field_name} must be positive")
    return integer


def _non_negative_int(value: Any, field_name: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if integer < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return integer


@dataclass(frozen=True)
class GDNBatchFeatures:
    """Phase-aware physical batch features for one GDN prediction query.

    Decode features intentionally contain no history/context-length value. GDN
    recurrent state has a fixed shape; only the physical batch and current
    query work belong in this contract.
    """

    batch_size: int
    batch_num_tokens: int
    max_query_len: int
    query_len_cv: float
    num_stateful_requests: int
    phase: str

    def __post_init__(self) -> None:
        batch_size = _positive_int(self.batch_size, "batch_size")
        batch_num_tokens = _positive_int(self.batch_num_tokens, "batch_num_tokens")
        max_query_len = _positive_int(self.max_query_len, "max_query_len")
        num_stateful_requests = _non_negative_int(
            self.num_stateful_requests, "num_stateful_requests"
        )
        if num_stateful_requests > batch_size:
            raise ValueError("num_stateful_requests cannot exceed batch_size")
        query_len_cv = float(self.query_len_cv)
        if not math.isfinite(query_len_cv) or query_len_cv < 0.0:
            raise ValueError("query_len_cv must be finite and non-negative")
        phase = str(self.phase).strip().lower()
        if phase not in {"prefill", "decode"}:
            raise ValueError(
                "GDN prediction supports only pure prefill or pure decode batches; "
                f"got phase={self.phase!r}"
            )
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "batch_num_tokens", batch_num_tokens)
        object.__setattr__(self, "max_query_len", max_query_len)
        object.__setattr__(self, "query_len_cv", query_len_cv)
        object.__setattr__(self, "num_stateful_requests", num_stateful_requests)
        object.__setattr__(self, "phase", phase)

    @property
    def feature_values(self) -> dict[str, float]:
        return {
            "batch_size": float(self.batch_size),
            "batch_num_tokens": float(self.batch_num_tokens),
            "max_query_len": float(self.max_query_len),
            "query_len_cv": float(self.query_len_cv),
            "num_stateful_requests": float(self.num_stateful_requests),
        }

    def as_vector(self, feature_columns: tuple[str, ...] = GDN_FEATURE_COLUMNS) -> list[float]:
        values = self.feature_values
        unknown = [name for name in feature_columns if name not in values]
        if unknown:
            raise ValueError(f"Unknown GDN feature columns: {unknown}")
        return [values[name] for name in feature_columns]

    def exact_key(self, feature_columns: tuple[str, ...] = GDN_FEATURE_COLUMNS) -> tuple[float, ...]:
        return tuple(self.as_vector(feature_columns))

    @classmethod
    def from_batch(cls, batch: Batch) -> "GDNBatchFeatures":
        requests = batch.requests
        query_lengths = tuple(_positive_int(value, "query_len") for value in batch.num_tokens)
        if not query_lengths:
            raise ValueError("GDN prediction requires at least one request")
        if len(requests) != len(query_lengths):
            raise ValueError(
                "GDN prediction requires one scheduled query length per request"
            )
        batch_size = len(query_lengths)
        batch_num_tokens = _positive_int(
            batch.total_num_tokens,
            "batch_num_tokens",
        )
        if batch_num_tokens != sum(query_lengths):
            raise ValueError(
                "GDN batch_num_tokens must equal the sum of scheduled query lengths"
            )
        mean_query_len = sum(query_lengths) / batch_size
        variance = sum((value - mean_query_len) ** 2 for value in query_lengths) / batch_size
        query_len_cv = math.sqrt(variance) / mean_query_len
        prefill_tokens = _non_negative_int(
            batch.num_prefill_tokens, "num_prefill_tokens"
        )
        decode_tokens = _non_negative_int(
            batch.num_decode_tokens, "num_decode_tokens"
        )
        if prefill_tokens + decode_tokens != batch_num_tokens:
            raise ValueError(
                "GDN prefill/decode token counts must sum to batch_num_tokens"
            )
        if prefill_tokens and decode_tokens:
            raise ValueError(
                "GDN predictor does not support same-batch prefill+decode mixing"
            )
        phase = "prefill" if prefill_tokens else "decode"
        num_stateful_requests = sum(
            request.is_prefill_complete
            or _non_negative_int(
                request.num_processed_tokens,
                "num_processed_tokens",
            )
            > 0
            for request in requests
        )
        return cls(
            batch_size=batch_size,
            batch_num_tokens=batch_num_tokens,
            max_query_len=max(query_lengths),
            query_len_cv=query_len_cv,
            num_stateful_requests=num_stateful_requests,
            phase=phase,
        )

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "GDNBatchFeatures":
        batch_size = _positive_int(row["batch_size"], "batch_size")
        batch_num_tokens = _positive_int(row["batch_num_tokens"], "batch_num_tokens")
        prefill_tokens = _non_negative_int(
            row.get("batch_num_prefill_tokens", 0), "batch_num_prefill_tokens"
        )
        decode_tokens = _non_negative_int(
            row.get("batch_num_decode_tokens", 0), "batch_num_decode_tokens"
        )
        if prefill_tokens and decode_tokens:
            raise ValueError("GDN training rows cannot mix prefill and decode work")
        if prefill_tokens + decode_tokens != batch_num_tokens:
            raise ValueError(
                "GDN training row prefill/decode token counts must sum to total tokens"
            )
        phase = "prefill" if prefill_tokens else "decode"
        stateful_default = batch_size if bool(row.get("has_initial_state", False)) else 0
        return cls(
            batch_size=batch_size,
            batch_num_tokens=batch_num_tokens,
            max_query_len=_positive_int(row["max_query_len"], "max_query_len"),
            query_len_cv=float(row.get("query_len_cv", 0.0)),
            num_stateful_requests=_non_negative_int(
                row.get("num_stateful_requests", stateful_default),
                "num_stateful_requests",
            ),
            phase=phase,
        )


__all__ = [
    "GDNBatchFeatures",
    "GDN_FEATURE_COLUMNS",
    "GDN_TASKS",
    "GDN_ARTIFACT_SCHEMA_VERSION",
    "GDN_IDENTITY_COLUMNS",
    "validate_gdn_artifact_identity",
]
