from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class AttentionOperatorRole(Enum):
    """Semantic role of a physical attention operator."""

    CACHE_WRITE = "cache_write"
    PREFILL_KERNEL = "prefill_kernel"
    DECODE_KERNEL = "decode_kernel"
    PROJECTION = "projection"
    POSITION_ENCODING = "position_encoding"
    RESHAPE = "reshape"
    COMMUNICATION = "communication"


class AttentionPhase(Enum):
    """Batch phase in which an attention operator can execute."""

    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"


class AttentionTraceKind(Enum):
    """Trace event class used by Frontier E2E operation traces."""

    COMPUTE = "COMPUTE"
    COMM = "COMM"


class ProjectionOwnership(Enum):
    """Whether an operator accounts for projection work."""

    OUTSIDE_ATTENTION = "outside_attention"
    INSIDE_ATTENTION_PHYSICAL_SCOPE = "inside_attention_physical_scope"
    NOT_PROJECTION = "not_projection"


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
class AttentionOperatorSpec:
    """Declarative contract for one physical attention operator."""

    name: str
    role: AttentionOperatorRole
    phases: tuple[AttentionPhase, ...]
    trace_kind: AttentionTraceKind = AttentionTraceKind.COMPUTE
    predictor_target: bool = True
    profiling_target: bool = True
    e2e_trace_target: bool = True
    execution_time_attr: str | None = None
    projection_ownership: ProjectionOwnership = ProjectionOwnership.NOT_PROJECTION

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Attention operator name must be non-empty")
        if not self.phases:
            raise ValueError(f"Attention operator {self.name} must declare phases")
        if self.e2e_trace_target and not self.execution_time_attr:
            raise ValueError(
                f"Attention operator {self.name} must declare execution_time_attr "
                "when it is an E2E trace target"
            )
        if (
            self.role is AttentionOperatorRole.PROJECTION
            and self.projection_ownership
            is ProjectionOwnership.NOT_PROJECTION
        ):
            raise ValueError(
                f"Projection operator {self.name} must declare projection ownership"
            )


@dataclass(frozen=True)
class AttentionFamilySpec:
    """Declarative contract for one attention operator family."""

    family_id: str
    display_name: str
    supported_variants: tuple[str, ...]
    operators: tuple[AttentionOperatorSpec, ...]
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

    def __post_init__(self) -> None:
        if not self.family_id:
            raise ValueError("Attention family_id must be non-empty")
        if not self.supported_variants:
            raise ValueError(
                f"Attention family {self.family_id} must declare variants"
            )
        names = [operator.name for operator in self.operators]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"Attention family {self.family_id} has duplicate operators: "
                f"{duplicates}"
            )
        if self.dsa_frozen and self.operators:
            raise ValueError("Frozen DSA family must not enable operator targets")
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

    def profiling_ops(self) -> tuple[AttentionOperatorSpec, ...]:
        return tuple(operator for operator in self.operators if operator.profiling_target)

    def predictor_ops(self) -> tuple[AttentionOperatorSpec, ...]:
        return tuple(operator for operator in self.operators if operator.predictor_target)

    def e2e_trace_ops(self) -> tuple[AttentionOperatorSpec, ...]:
        return tuple(operator for operator in self.operators if operator.e2e_trace_target)

    def projection_ops(self) -> tuple[AttentionOperatorSpec, ...]:
        return tuple(
            operator
            for operator in self.operators
            if operator.projection_ownership
            is not ProjectionOwnership.NOT_PROJECTION
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
