from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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

    def require_enabled_for_execution(self) -> None:
        if self.dsa_frozen:
            raise NotImplementedError(
                "DSA attention is frozen until a real vLLM/FlashInfer truth "
                "backend and Frontier mapping are approved."
            )
