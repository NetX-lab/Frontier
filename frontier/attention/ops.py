from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, ClassVar

from frontier.operators.spec import (
    OperatorFamilySpec,
    OperatorPhase,
    OperatorRole,
    OperatorSpec,
    ProjectionOwnership,
    TraceKind,
)


AttentionOperatorRole = OperatorRole
AttentionPhase = OperatorPhase
AttentionTraceKind = TraceKind


class AttentionMemoryLayout(Enum):
    """Runtime KV-cache layout represented by an attention family."""

    DENSE_KV = "dense_kv"
    LATENT_MLA = "latent_mla"
    FROZEN_DSA = "frozen_dsa"


@dataclass(frozen=True)
class AttentionRuntimeMetaContract:
    """Runtime metadata contract for imported attention profiler rows."""

    expected_runtime_num_kv_heads: int
    runtime_head_size_formula: str
    supported_block_sizes: tuple[int, ...]
    expected_n_q_head: int | None = None

    def __post_init__(self) -> None:
        if self.expected_runtime_num_kv_heads <= 0:
            raise ValueError(
                "expected_runtime_num_kv_heads must be positive, "
                f"got={self.expected_runtime_num_kv_heads!r}"
            )
        if not self.runtime_head_size_formula:
            raise ValueError("runtime_head_size_formula must be non-empty")
        if not self.supported_block_sizes:
            raise ValueError("supported_block_sizes must be non-empty")
        invalid_block_sizes = [
            block_size
            for block_size in self.supported_block_sizes
            if int(block_size) <= 0
        ]
        if invalid_block_sizes:
            raise ValueError(
                "supported_block_sizes must be positive, "
                f"got={invalid_block_sizes!r}"
            )
        if self.expected_n_q_head is not None and self.expected_n_q_head <= 0:
            raise ValueError(
                "expected_n_q_head must be positive when declared, "
                f"got={self.expected_n_q_head!r}"
            )


@dataclass(frozen=True)
class AttentionOperatorSpec(OperatorSpec):
    """Declarative contract for one physical attention operator."""

    _operator_label: ClassVar[str] = "Attention operator"


@dataclass(frozen=True)
class AttentionFamilySpec(OperatorFamilySpec):
    """Declarative contract for one attention operator family."""

    memory_layout: AttentionMemoryLayout
    dense_compatible: bool
    requires_runtime_kv_helpers: bool
    disjoint_model_projection_attrs: tuple[str, ...] = ()
    dsa_frozen: bool = False
    kv_factor: int | None = None
    required_profiling_feature_columns: tuple[str, ...] = ()
    imported_predictor_excluded_feature_columns: tuple[str, ...] = ()
    runtime_num_kv_heads_resolver: Callable[[Any], int] | None = None
    runtime_head_size_resolver: Callable[[Any], int] | None = None
    runtime_meta_contract: AttentionRuntimeMetaContract | None = None

    _family_label: ClassVar[str] = "Attention family"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dsa_frozen and self.operators:
            raise ValueError("Frozen DSA family must not enable operator targets")
        if self.dsa_frozen and self.execution_enabled:
            raise ValueError("Frozen DSA family must not be execution enabled")
        projection_attrs = list(self.disjoint_model_projection_attrs)
        duplicate_projection_attrs = sorted(
            {
                attr_name
                for attr_name in projection_attrs
                if projection_attrs.count(attr_name) > 1
            }
        )
        if duplicate_projection_attrs:
            raise ValueError(
                f"Attention family {self.family_id} has duplicate disjoint "
                f"model projection attrs: {duplicate_projection_attrs}"
            )
        if any(not attr_name for attr_name in projection_attrs):
            raise ValueError(
                f"Attention family {self.family_id} has an empty disjoint "
                "model projection attr"
            )
        if self.kv_factor is not None and self.kv_factor <= 0:
            raise ValueError(
                f"Attention family {self.family_id} kv_factor must be positive "
                f"when declared, got={self.kv_factor!r}"
            )
        missing_excluded = tuple(
            column
            for column in self.imported_predictor_excluded_feature_columns
            if column not in self.required_profiling_feature_columns
        )
        if missing_excluded:
            raise ValueError(
                f"Attention family {self.family_id} excludes imported predictor "
                f"columns absent from its profiling schema: {list(missing_excluded)}"
            )

    def resolve_runtime_num_kv_heads(self, config: Any) -> int:
        if self.runtime_num_kv_heads_resolver is None:
            raise ValueError(
                f"Attention family {self.family_id} does not declare a "
                "runtime_num_kv_heads_resolver"
            )
        value = int(self.runtime_num_kv_heads_resolver(config))
        if value <= 0:
            raise ValueError(
                f"Attention family {self.family_id} resolved non-positive "
                f"runtime_num_kv_heads={value!r}"
            )
        return value

    def resolve_runtime_head_size(self, config: Any) -> int:
        if self.runtime_head_size_resolver is None:
            raise ValueError(
                f"Attention family {self.family_id} does not declare a "
                "runtime_head_size_resolver"
            )
        value = int(self.runtime_head_size_resolver(config))
        if value <= 0:
            raise ValueError(
                f"Attention family {self.family_id} resolved non-positive "
                f"runtime_head_size={value!r}"
            )
        return value

    def require_enabled_for_execution(self) -> None:
        if self.dsa_frozen:
            raise NotImplementedError(
                "DSA attention is frozen until a real vLLM/FlashInfer truth "
                "backend and Frontier mapping are approved."
            )
        super().require_enabled_for_execution()
