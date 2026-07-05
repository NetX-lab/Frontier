from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, ClassVar


class ResourceClass(Enum):
    """High-level resource bucket for a physical operator."""

    COMP = "COMP"
    COMM = "COMM"
    MEMORY = "MEMORY"


class OperatorRole(Enum):
    """Semantic role of a physical operator."""

    CACHE_WRITE = "cache_write"
    PREFILL_KERNEL = "prefill_kernel"
    DECODE_KERNEL = "decode_kernel"
    PROJECTION = "projection"
    ACTIVATION = "activation"
    NORMALIZATION = "normalization"
    RESIDUAL = "residual"
    EMBEDDING = "embedding"
    POSITION_ENCODING = "position_encoding"
    RESHAPE = "reshape"
    COMMUNICATION = "communication"


class OperatorPhase(Enum):
    """Batch phase in which an operator can execute."""

    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"


class TraceKind(Enum):
    """Trace event class used by Frontier E2E operation traces."""

    COMPUTE = "COMPUTE"
    COMM = "COMM"


class ProjectionOwnership(Enum):
    """Whether an operator accounts for projection work."""

    OUTSIDE_ATTENTION = "outside_attention"
    INSIDE_ATTENTION_PHYSICAL_SCOPE = "inside_attention_physical_scope"
    NOT_PROJECTION = "not_projection"


class TensorParallelMode(Enum):
    """How a predictor resolves the tensor-parallel key for an operator."""

    REPLICATED = "replicated"
    ATTENTION_TP = "attention_tp"
    FFN_TP = "ffn_tp"
    MOE_TP = "moe_tp"


@dataclass(frozen=True)
class OperatorSpec:
    """Declarative contract for one physical operator."""

    name: str
    role: OperatorRole
    phases: tuple[OperatorPhase, ...]
    trace_kind: TraceKind = TraceKind.COMPUTE
    predictor_target: bool = True
    profiling_target: bool = True
    profiling_key: str | None = None
    precision_op: str | None = None
    calibration_key: str | None = None
    e2e_trace_target: bool = True
    execution_time_attr: str | None = None
    resource_class: ResourceClass | None = None
    tp_mode: TensorParallelMode | None = None
    ep_agnostic: bool = False
    projection_ownership: ProjectionOwnership = ProjectionOwnership.NOT_PROJECTION

    _operator_label: ClassVar[str] = "Operator"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError(f"{self._operator_label} name must be non-empty")
        if self.resource_class is not None and not isinstance(
            self.resource_class, ResourceClass
        ):
            raise ValueError(
                f"{self._operator_label} {self.name} resource_class must be "
                f"ResourceClass, got {self.resource_class!r}"
            )
        if self.profiling_key == "":
            raise ValueError(f"{self._operator_label} {self.name} profiling_key must be non-empty")
        if self.precision_op == "":
            raise ValueError(f"{self._operator_label} {self.name} precision_op must be non-empty")
        if self.calibration_key == "":
            raise ValueError(f"{self._operator_label} {self.name} calibration_key must be non-empty")
        if not self.phases:
            raise ValueError(f"{self._operator_label} {self.name} must declare phases")
        if self.e2e_trace_target and not self.execution_time_attr:
            raise ValueError(
                f"{self._operator_label} {self.name} must declare execution_time_attr "
                "when it is an E2E trace target"
            )
        if (
            self.role is OperatorRole.PROJECTION
            and self.projection_ownership is ProjectionOwnership.NOT_PROJECTION
        ):
            raise ValueError(
                f"Projection operator {self.name} must declare projection ownership"
            )

    def profiling_name(self) -> str:
        """Return the profiler-facing key for this physical operator."""

        return self.profiling_key or self.name

    def precision_name(self) -> str:
        """Return the quantization/precision-facing key for this physical operator."""

        return self.precision_op or self.name

    def calibration_field_name(self) -> str | None:
        """Return the config field name for this operator's base calibration scale."""

        if self.calibration_key is None:
            return None
        return f"{self.calibration_key}_calibration_scale"

    def calibration_attr_name(self) -> str | None:
        """Return the predictor attribute name for this operator's base calibration scale."""

        field_name = self.calibration_field_name()
        if field_name is None:
            return None
        return f"_{field_name}"


@dataclass(frozen=True)
class CommPayloadContext:
    """Inputs required to build a collective payload for one batch."""

    batch: Any
    model_config: Any
    replica_config: Any
    cluster_type: Any
    quantization_manager: Any


CommPayloadBuilder = Callable[[CommPayloadContext], int]
CommNumDevicesBuilder = Callable[[CommPayloadContext], int]


@dataclass(frozen=True)
class CommOperatorSpec(OperatorSpec):
    """Declarative contract for one collective communication operator."""

    collective_alias: str = ""
    comm_group: str = ""
    payload_builder: CommPayloadBuilder | None = None
    num_devices_builder: CommNumDevicesBuilder | None = None
    comm_domain: str | None = None
    apply_allreduce_launch_overhead_strip: bool = False

    _operator_label: ClassVar[str] = "CommOperator"

    def __post_init__(self) -> None:
        super().__post_init__()
        valid_collectives = {"allreduce", "allgather", "alltoall", "send_recv"}
        if self.collective_alias not in valid_collectives:
            raise ValueError(
                f"CommOperator {self.name} collective_alias must be one of "
                f"{sorted(valid_collectives)}, got {self.collective_alias!r}"
            )
        if not self.comm_group:
            raise ValueError(f"CommOperator {self.name} comm_group must be non-empty")
        if self.payload_builder is None:
            raise ValueError(f"CommOperator {self.name} must declare payload_builder")
        if self.trace_kind is not TraceKind.COMM:
            raise ValueError(f"CommOperator {self.name} trace_kind must be COMM")
        if self.resource_class is not ResourceClass.COMM:
            raise ValueError(f"CommOperator {self.name} resource_class must be COMM")

    def build_payload_bytes(self, ctx: CommPayloadContext) -> int:
        if self.payload_builder is None:
            raise ValueError(f"CommOperator {self.name} has no payload_builder")
        payload_bytes = int(self.payload_builder(ctx))
        if payload_bytes < 0:
            raise ValueError(
                f"CommOperator {self.name} produced negative payload: {payload_bytes}"
            )
        return payload_bytes

    def num_devices(self, ctx: CommPayloadContext) -> int | None:
        if self.num_devices_builder is None:
            return None
        num_devices = int(self.num_devices_builder(ctx))
        if num_devices <= 0:
            raise ValueError(
                f"CommOperator {self.name} produced invalid num_devices: {num_devices}"
            )
        return num_devices


@dataclass(frozen=True)
class OperatorFamilySpec:
    """Declarative contract for one operator family."""

    family_id: str
    display_name: str
    supported_variants: tuple[str, ...]
    operators: tuple[OperatorSpec, ...]
    resource_class: ResourceClass = field(default=ResourceClass.COMP, kw_only=True)
    profiling_order: tuple[str, ...] = field(default=(), kw_only=True)
    execution_enabled: bool = field(default=True, kw_only=True)
    disabled_reason: str = field(default="", kw_only=True)

    _family_label: ClassVar[str] = "Operator family"

    def __post_init__(self) -> None:
        if not self.family_id:
            raise ValueError(f"{self._family_label} family_id must be non-empty")
        if not isinstance(self.resource_class, ResourceClass):
            raise ValueError(
                f"{self._family_label} {self.family_id} resource_class must be "
                f"ResourceClass, got {self.resource_class!r}"
            )
        if not self.supported_variants:
            raise ValueError(
                f"{self._family_label} {self.family_id} must declare variants"
            )
        names = [operator.name for operator in self.operators]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"{self._family_label} {self.family_id} has duplicate operators: "
                f"{duplicates}"
            )
        missing_resource_class = sorted(
            operator.name for operator in self.operators if operator.resource_class is None
        )
        if missing_resource_class:
            raise ValueError(
                f"{self._family_label} {self.family_id} has operators missing "
                f"resource_class: {missing_resource_class}"
            )
        invalid_resource_class = sorted(
            (operator.name, repr(operator.resource_class))
            for operator in self.operators
            if not isinstance(operator.resource_class, ResourceClass)
        )
        if invalid_resource_class:
            raise ValueError(
                f"{self._family_label} {self.family_id} has operators with invalid "
                f"resource_class: {invalid_resource_class}"
            )
        profiling_order_duplicates = sorted(
            {name for name in self.profiling_order if self.profiling_order.count(name) > 1}
        )
        if profiling_order_duplicates:
            raise ValueError(
                f"{self._family_label} {self.family_id} has duplicate profiling_order "
                f"entries: {profiling_order_duplicates}"
            )
        profiling_target_names = {
            operator.name for operator in self.operators if operator.profiling_target
        }
        unknown_profiling_order_names = sorted(
            set(self.profiling_order) - profiling_target_names
        )
        if unknown_profiling_order_names:
            raise ValueError(
                f"{self._family_label} {self.family_id} profiling_order references "
                f"unknown profiling operators: {unknown_profiling_order_names}"
            )
        if not self.execution_enabled and not self.disabled_reason:
            raise ValueError(
                f"{self._family_label} {self.family_id} must declare disabled_reason "
                "when execution_enabled is False"
            )

    def require_enabled_for_execution(self) -> None:
        if not self.execution_enabled:
            raise NotImplementedError(
                f"{self._family_label} {self.family_id} is not enabled for execution: "
                f"{self.disabled_reason}"
            )

    def profiling_ops(self) -> tuple[OperatorSpec, ...]:
        operators = tuple(operator for operator in self.operators if operator.profiling_target)
        if not self.profiling_order:
            return operators

        by_name = {operator.name: operator for operator in operators}
        ordered_names = set(self.profiling_order)
        ordered = tuple(by_name[name] for name in self.profiling_order)
        remaining = tuple(operator for operator in operators if operator.name not in ordered_names)
        return ordered + remaining

    def predictor_ops(self) -> tuple[OperatorSpec, ...]:
        return tuple(operator for operator in self.operators if operator.predictor_target)

    def e2e_trace_ops(self) -> tuple[OperatorSpec, ...]:
        return tuple(operator for operator in self.operators if operator.e2e_trace_target)

    def projection_ops(self) -> tuple[OperatorSpec, ...]:
        return tuple(
            operator
            for operator in self.operators
            if operator.projection_ownership is not ProjectionOwnership.NOT_PROJECTION
        )
