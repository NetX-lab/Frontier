"""Plugin-style model architecture contracts for model-specific runtime semantics."""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal

from frontier.operators.spec import TensorParallelMode
from frontier.types import ClusterType


logger = logging.getLogger(__name__)


class LinearAttentionImplementation(Enum):
    """Linear-op profiling attention implementation selected by architecture."""

    GENERIC = "generic"
    STEP2_MINI = "step2_mini"
    STEP3_TEXT = "step3_text"


class ExpertParallelCollective(Enum):
    """Collective semantic used for expert-parallel synchronization."""

    ALLGATHER = "allgather"
    ALLTOALL = "alltoall"


class ResidualAddPolicy(Enum):
    """Residual add accounting policy selected by architecture."""

    STANDARD = "standard"
    FFN_RESIDUAL_ONLY = "ffn_residual_only"


class LayerKind(Enum):
    """Typed FFN layer domains exposed by a model architecture profile."""

    DENSE = "dense"
    ROUTED = "routed"
    ROUTED_MOE = "routed"
    SHARED = "shared"
    SHARED_EXPERT = "shared"


class LayerDimensionSource(Enum):
    """Declarative model-config source for a layer's effective FFN width."""

    DENSE = "dense_mlp_hidden_dim"
    ROUTED = "routed_mlp_hidden_dim"
    SHARED = "share_expert_dim"
    DENSE_INTERMEDIATE_SIZE = "dense_mlp_hidden_dim"
    MOE_INTERMEDIATE_SIZE = "routed_mlp_hidden_dim"


class ExpertParallelMode(Enum):
    """Whether a typed layer runs through the routed expert-parallel domain."""

    OFF = "off"
    ON = "on"


@dataclass(frozen=True)
class LayerContractSpec:
    """Declarative contract for one dense, routed, or shared FFN domain."""

    layer_kind: LayerKind
    dimension_source: LayerDimensionSource
    tensor_parallel_mode: TensorParallelMode
    expert_parallel_mode: ExpertParallelMode = ExpertParallelMode.OFF
    operator_family_ids: tuple[str, ...] = ()
    base_layer_kinds: tuple[LayerKind, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.layer_kind, LayerKind):
            object.__setattr__(self, "layer_kind", LayerKind(self.layer_kind))
        if not isinstance(self.dimension_source, LayerDimensionSource):
            object.__setattr__(
                self, "dimension_source", LayerDimensionSource(self.dimension_source)
            )
        if not isinstance(self.tensor_parallel_mode, TensorParallelMode):
            object.__setattr__(
                self, "tensor_parallel_mode", TensorParallelMode(self.tensor_parallel_mode)
            )
        if not isinstance(self.expert_parallel_mode, ExpertParallelMode):
            object.__setattr__(
                self, "expert_parallel_mode", ExpertParallelMode(self.expert_parallel_mode)
            )

        family_ids = tuple(self.operator_family_ids)
        if any(not isinstance(family_id, str) or not family_id for family_id in family_ids):
            raise ValueError(
                f"{self.layer_kind.value} layer contract operator_family_ids must "
                "contain non-empty strings"
            )
        if len(set(family_ids)) != len(family_ids):
            raise ValueError(
                f"{self.layer_kind.value} layer contract declares duplicate "
                f"operator families: {family_ids}"
            )
        object.__setattr__(self, "operator_family_ids", family_ids)

        base_kinds = tuple(self.base_layer_kinds) or (self.layer_kind,)
        normalized_base_kinds = []
        for base_kind in base_kinds:
            if not isinstance(base_kind, LayerKind):
                try:
                    base_kind = LayerKind(base_kind)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Unknown base layer kind in {self.layer_kind.value} "
                        f"layer contract: {base_kind!r}"
                    ) from exc
            normalized_base_kinds.append(base_kind)
        if len(set(normalized_base_kinds)) != len(normalized_base_kinds):
            raise ValueError(
                f"{self.layer_kind.value} layer contract declares duplicate base "
                f"layer kinds: {normalized_base_kinds}"
            )
        object.__setattr__(self, "base_layer_kinds", tuple(normalized_base_kinds))

        expected_source = {
            LayerKind.DENSE: LayerDimensionSource.DENSE,
            LayerKind.ROUTED: LayerDimensionSource.ROUTED,
            LayerKind.SHARED: LayerDimensionSource.SHARED,
        }[self.layer_kind]
        if self.dimension_source is not expected_source:
            raise ValueError(
                f"{self.layer_kind.value} layer contract must use "
                f"dimension source {expected_source.value}, got "
                f"{self.dimension_source.value}"
            )

    @property
    def tp_mode(self) -> TensorParallelMode:
        """Compatibility alias for callers that use the shorter TP name."""

        return self.tensor_parallel_mode

    @property
    def ep_mode(self) -> ExpertParallelMode:
        """Compatibility alias for callers that use the shorter EP name."""

        return self.expert_parallel_mode

    def resolve_width(self, config: Any, *, mixed_model: bool = False) -> int:
        """Resolve and validate the width declared by this contract."""

        source_attributes = {
            LayerDimensionSource.DENSE: ("dense_mlp_hidden_dim", "intermediate_size"),
            LayerDimensionSource.ROUTED: (
                "routed_mlp_hidden_dim",
                "moe_intermediate_size",
                "mlp_hidden_dim",
            ),
            LayerDimensionSource.SHARED: (
                "share_expert_dim",
                "shared_expert_intermediate_size",
            ),
        }[self.dimension_source]
        value = None
        for attribute in source_attributes:
            candidate = getattr(config, attribute, None)
            if candidate is not None:
                value = candidate
                break
        if value is None and self.dimension_source is LayerDimensionSource.DENSE:
            # A mixed model must carry an explicit dense width. Falling back to
            # the legacy model-wide field would silently reuse the routed width.
            if mixed_model:
                raise ValueError(
                    "dense layer width requires dense_mlp_hidden_dim or "
                    "intermediate_size for a mixed model"
                )
            value = getattr(config, "mlp_hidden_dim", None)
        if type(value) is not int or value <= 0:
            raise ValueError(
                f"{self.layer_kind.value} layer width must be a positive int, got {value!r}"
            )
        return value


@dataclass(frozen=True)
class ResolvedLayerContract:
    """Immutable layer contract after config and parallel sizes are resolved."""

    profile_id: str
    layer_id: int | None
    layer_kind: LayerKind
    dimension_source: LayerDimensionSource
    effective_ffn_width: int
    tensor_parallel_mode: TensorParallelMode
    expert_parallel_mode: ExpertParallelMode
    tensor_parallel_size: int | None = None
    expert_parallel_size: int | None = None
    operator_family_id: str | None = None

    @property
    def width(self) -> int:
        return self.effective_ffn_width

    @property
    def tp_mode(self) -> TensorParallelMode:
        return self.tensor_parallel_mode

    @property
    def ep_mode(self) -> ExpertParallelMode:
        return self.expert_parallel_mode

    @property
    def is_expert_parallel(self) -> bool:
        return self.expert_parallel_mode is ExpertParallelMode.ON


def _default_layer_contracts() -> tuple[LayerContractSpec, ...]:
    return (
        LayerContractSpec(
            LayerKind.DENSE,
            LayerDimensionSource.DENSE,
            TensorParallelMode.FFN_TP,
            operator_family_ids=("ffn",),
        ),
        LayerContractSpec(
            LayerKind.ROUTED,
            LayerDimensionSource.ROUTED,
            TensorParallelMode.MOE_TP,
            ExpertParallelMode.ON,
            operator_family_ids=("moe",),
        ),
        LayerContractSpec(
            LayerKind.SHARED,
            LayerDimensionSource.SHARED,
            TensorParallelMode.FFN_TP,
            operator_family_ids=("share_expert",),
            base_layer_kinds=(LayerKind.ROUTED,),
        ),
    )


@dataclass(frozen=True)
class LinearAttentionProfile:
    """Declarative linear-op profiling contract for attention-related ops."""

    sharded_impl: LinearAttentionImplementation
    sharded_ops: tuple[str, ...]
    replicated_ops: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.sharded_ops:
            raise ValueError("linear attention profile must declare sharded_ops")
        if len(set(self.sharded_ops)) != len(self.sharded_ops):
            raise ValueError(
                f"linear attention sharded_ops contains duplicates: {self.sharded_ops}"
            )
        if len(set(self.replicated_ops)) != len(self.replicated_ops):
            raise ValueError(
                "linear attention replicated_ops contains duplicates: "
                f"{self.replicated_ops}"
            )

    def has_replicated_pre_projection(self, enabled_ops: set[str] | None) -> bool:
        """Return whether a replicated-only attention pre-projection path is needed."""

        if not self.replicated_ops or enabled_ops is None:
            return False
        return bool(set(self.replicated_ops).intersection(enabled_ops))

    @property
    def additional_sharded_ops(self) -> tuple[str, ...]:
        """Return architecture-specific sharded attention ops beyond the generic path."""

        generic_ops = ("attn_pre_proj", "attn_rope", "attn_post_proj")
        return tuple(op_name for op_name in self.sharded_ops if op_name not in generic_ops)


ArchitectureMatcher = Callable[[Any], bool]
StructuralPredicate = Callable[[Any], bool]
StructuralMessage = Callable[["ModelArchitectureProfile", Any], str]
AttentionShapeLogKind = Literal["mla", "mfa"]


@dataclass(frozen=True)
class StructuralRequirement:
    """Profile-owned validation rule for structural model config facts."""

    name: str
    predicate: StructuralPredicate
    message: StructuralMessage

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("structural requirement name must be non-empty")

    def validate(self, profile: "ModelArchitectureProfile", config: Any) -> None:
        try:
            passed = self.predicate(config)
        except ValueError as exc:
            raise ValueError(self.message(profile, config)) from exc
        if not passed:
            raise ValueError(self.message(profile, config))


@dataclass(frozen=True)
class ModelArchitectureProfile:
    """Declarative contract for model-specific architecture behavior."""

    profile_id: str
    display_name: str
    linear_attention: LinearAttentionProfile
    expert_parallel_collective: ExpertParallelCollective
    target_embedded_mtp: bool = False
    predictor_attention_extra_ops: tuple[str, ...] = ()
    attention_shape_log_kind: AttentionShapeLogKind | None = None
    residual_add_policy: ResidualAddPolicy = ResidualAddPolicy.STANDARD
    skip_decode_ffn_attn_norm_residual: bool = False
    skip_decode_attn_residual: bool = False
    moe_tensor_parallel_allgather_op: str | None = None
    share_expert_tensor_parallel_allreduce_op: str | None = None
    always_supports_share_expert: bool = False
    counts_share_expert_param_memory: bool = False
    structural_requirements: tuple[StructuralRequirement, ...] = ()
    match: ArchitectureMatcher = field(default=lambda _config: False, repr=False, compare=False)
    layer_contracts: tuple[LayerContractSpec, ...] = field(
        default_factory=_default_layer_contracts
    )

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("model architecture profile_id must be non-empty")
        if not self.display_name:
            raise ValueError("model architecture display_name must be non-empty")
        if self.attention_shape_log_kind is not None and not self.attention_shape_log_kind:
            raise ValueError(
                "model architecture attention_shape_log_kind must be non-empty "
                "when provided"
            )
        unknown_predictor_ops = set(self.predictor_attention_extra_ops).difference(
            self.linear_attention.sharded_ops
        )
        if unknown_predictor_ops:
            raise ValueError(
                "predictor_attention_extra_ops must be declared in linear_attention.sharded_ops, "
                f"got unknown ops: {sorted(unknown_predictor_ops)}"
            )
        if not self.layer_contracts:
            raise ValueError(
                f"model architecture profile {self.profile_id!r} must declare "
                "at least one layer contract"
            )
        kinds = [contract.layer_kind for contract in self.layer_contracts]
        if len(set(kinds)) != len(kinds):
            raise ValueError(
                f"model architecture profile {self.profile_id!r} declares duplicate "
                f"layer contracts: {kinds}"
            )
        family_ids = [
            family_id
            for contract in self.layer_contracts
            for family_id in contract.operator_family_ids
        ]
        if len(set(family_ids)) != len(family_ids):
            raise ValueError(
                f"model architecture profile {self.profile_id!r} declares duplicate "
                f"operator family ownership: {family_ids}"
            )

    @classmethod
    def generic(
        cls,
        profile_id: str = "generic",
        match: ArchitectureMatcher | None = None,
    ) -> "ModelArchitectureProfile":
        return cls(
            profile_id=profile_id,
            display_name="Generic Transformer",
            linear_attention=LinearAttentionProfile(
                sharded_impl=LinearAttentionImplementation.GENERIC,
                sharded_ops=(
                    "attn_pre_proj",
                    "attn_rope",
                    "attn_post_proj",
                ),
            ),
            expert_parallel_collective=ExpertParallelCollective.ALLTOALL,
            match=match or (lambda _config: False),
        )

    @classmethod
    def step2_mini(
        cls,
        profile_id: str = "step2_mini",
        match: ArchitectureMatcher | None = None,
    ) -> "ModelArchitectureProfile":
        return cls(
            profile_id=profile_id,
            display_name="Step2Mini",
            linear_attention=LinearAttentionProfile(
                sharded_impl=LinearAttentionImplementation.STEP2_MINI,
                sharded_ops=(
                    "attn_pre_proj",
                    "attn_rope",
                    "attn_post_proj",
                    "attn_inter_norm",
                    "attn_wq_proj",
                ),
            ),
            expert_parallel_collective=ExpertParallelCollective.ALLTOALL,
            target_embedded_mtp=True,
            predictor_attention_extra_ops=(
                "attn_inter_norm",
                "attn_wq_proj",
            ),
            always_supports_share_expert=True,
            counts_share_expert_param_memory=True,
            structural_requirements=_moe_share_expert_requirements(),
            match=match or _matches_step2_mini,
        )

    @classmethod
    def step3_text(
        cls,
        profile_id: str = "step3_text",
        match: ArchitectureMatcher | None = None,
    ) -> "ModelArchitectureProfile":
        return cls(
            profile_id=profile_id,
            display_name="Step3Text MFA",
            linear_attention=LinearAttentionProfile(
                sharded_impl=LinearAttentionImplementation.STEP3_TEXT,
                sharded_ops=(
                    "attn_pre_proj",
                    "attn_rope",
                    "attn_post_proj",
                    "attn_pre_proj_wq",
                ),
                replicated_ops=(
                    "attn_pre_proj_qkv",
                    "attn_pre_proj_q_norm",
                ),
            ),
            expert_parallel_collective=ExpertParallelCollective.ALLTOALL,
            target_embedded_mtp=True,
            attention_shape_log_kind="mfa",
            residual_add_policy=ResidualAddPolicy.FFN_RESIDUAL_ONLY,
            skip_decode_ffn_attn_norm_residual=True,
            skip_decode_attn_residual=True,
            moe_tensor_parallel_allgather_op="moe_tensor_parallel_allgather",
            share_expert_tensor_parallel_allreduce_op=(
                "share_expert_tensor_parallel_allreduce"
            ),
            always_supports_share_expert=True,
            counts_share_expert_param_memory=True,
            structural_requirements=(
                *_moe_share_expert_requirements(),
                _requires_step3_mfa_attention_contract(),
            ),
            match=match or _matches_step3_text,
            layer_contracts=(
                LayerContractSpec(
                    LayerKind.DENSE,
                    LayerDimensionSource.DENSE,
                    TensorParallelMode.ATTENTION_TP,
                    operator_family_ids=("ffn",),
                ),
                LayerContractSpec(
                    LayerKind.ROUTED,
                    LayerDimensionSource.ROUTED,
                    TensorParallelMode.MOE_TP,
                    ExpertParallelMode.ON,
                    operator_family_ids=("moe",),
                ),
                LayerContractSpec(
                    LayerKind.SHARED,
                    LayerDimensionSource.SHARED,
                    TensorParallelMode.ATTENTION_TP,
                    operator_family_ids=("share_expert",),
                    base_layer_kinds=(LayerKind.ROUTED,),
                ),
            ),
        )

    def validate_structural_requirements(self, config: Any) -> None:
        """Validate profile-owned structural requirements against a config."""

        for requirement in self.structural_requirements:
            requirement.validate(self, config)
        if self.attention_shape_log_kind == "mla":
            _requires_attention_family("latent_mla_attention").validate(self, config)

    def supports_share_expert(self, config: Any) -> bool:
        """Return whether this architecture exposes a shared expert FFN path."""

        if self.always_supports_share_expert:
            return True
        return bool(getattr(config, "is_moe", False)) and int(
            getattr(config, "share_expert_dim", 0) or 0
        ) > 0

    def get_layer_contract(self, layer_kind: LayerKind) -> LayerContractSpec:
        """Return the profile-owned contract for one typed FFN domain."""

        if not isinstance(layer_kind, LayerKind):
            try:
                layer_kind = LayerKind(layer_kind)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Unknown layer kind: {layer_kind!r}") from exc
        for contract in self.layer_contracts:
            if contract.layer_kind is layer_kind:
                return contract
        raise ValueError(
            f"Profile {self.profile_id!r} does not declare layer kind "
            f"{layer_kind.value!r}"
        )

    def get_layer_contract_for_family(self, family_id: str) -> LayerContractSpec:
        """Return the profile contract that owns one registered operator family."""

        if not isinstance(family_id, str) or not family_id:
            raise ValueError(f"Operator family ID must be a non-empty string: {family_id!r}")
        matches = tuple(
            contract
            for contract in self.layer_contracts
            if family_id in contract.operator_family_ids
        )
        if len(matches) != 1:
            raise ValueError(
                f"Profile {self.profile_id!r} declares {len(matches)} layer contracts "
                f"for operator family {family_id!r}; expected exactly one"
            )
        return matches[0]

    def resolve_typed_layer_contract(self, config: Any, **kwargs: Any) -> ResolvedLayerContract:
        """Alias for :meth:`resolve_layer_contract` used by typed consumers."""

        return self.resolve_layer_contract(config, **kwargs)

    def resolve_layer_contract(
        self,
        config: Any,
        *,
        layer_id: int | None = None,
        operator_name: str | None = None,
        attention_tp_size: int | None = None,
        attn_tp_size: int | None = None,
        moe_tp_size: int | None = None,
        ffn_tp_size: int | None = None,
        tensor_parallel_size: int | None = None,
        expert_parallel_size: int | None = None,
        ep_size: int | None = None,
    ) -> ResolvedLayerContract:
        """Resolve layer kind, width, and parallel domains from one profile.

        ``operator_name`` is bound through the existing operator registry. A
        layer ID additionally validates that the requested operator is legal
        for that layer. Mixed models without either an operator family or a
        concrete layer ID fail fast instead of collapsing domains.
        """

        if config is None:
            raise ValueError("layer contract resolution requires a model config")
        for size_name, size_value in (
            ("attention_tp_size", attention_tp_size),
            ("attn_tp_size", attn_tp_size),
            ("moe_tp_size", moe_tp_size),
            ("ffn_tp_size", ffn_tp_size),
            ("tensor_parallel_size", tensor_parallel_size),
            ("expert_parallel_size", expert_parallel_size),
            ("ep_size", ep_size),
        ):
            self._validate_positive_int(size_name, size_value)
        if attention_tp_size is not None and attn_tp_size is not None:
            if attention_tp_size != attn_tp_size:
                raise ValueError(
                    "attention_tp_size and attn_tp_size disagree: "
                    f"{attention_tp_size} != {attn_tp_size}"
                )
        attention_tp_size = (
            attention_tp_size if attention_tp_size is not None else attn_tp_size
        )
        if expert_parallel_size is not None and ep_size is not None:
            if expert_parallel_size != ep_size:
                raise ValueError(
                    "expert_parallel_size and ep_size disagree: "
                    f"{expert_parallel_size} != {ep_size}"
                )
        expert_parallel_size = (
            expert_parallel_size if expert_parallel_size is not None else ep_size
        )

        num_layers = getattr(config, "num_layers", None)
        if layer_id is not None:
            if type(layer_id) is not int:
                raise ValueError(f"layer_id must be an int, got {layer_id!r}")
            if type(num_layers) is not int or num_layers <= 0:
                raise ValueError(
                    "layer contract resolution requires a positive integer num_layers"
                )
            if layer_id < 0 or layer_id >= num_layers:
                raise ValueError(
                    f"layer_id {layer_id} out of range [0, {num_layers})"
                )

        mixed_model = self._is_mixed_model(config)
        operator_family_id: str | None = None
        operator_contract: LayerContractSpec | None = None
        if operator_name is not None:
            if not isinstance(operator_name, str) or not operator_name:
                raise ValueError("operator_name must be a non-empty string")
            try:
                from frontier.operators.binding import bind_operator_query

                operator_family_id = bind_operator_query(operator_name).family_id
            except (ImportError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Unable to bind operator {operator_name!r} to a typed layer contract"
                ) from exc
            operator_contract = self.get_layer_contract_for_family(operator_family_id)

        config_kind = self._resolve_config_layer_kind(config, layer_id)
        if operator_contract is not None:
            if layer_id is not None and config_kind not in operator_contract.base_layer_kinds:
                allowed = ", ".join(kind.value for kind in operator_contract.base_layer_kinds)
                raise ValueError(
                    f"{operator_contract.layer_kind.value} operator {operator_name!r} "
                    "requires base layer kind "
                    f"{allowed}, but layer_id={layer_id} is {config_kind.value}"
                )
            if operator_contract.layer_kind is LayerKind.SHARED:
                supports_shared = getattr(config, "supports_share_expert", None)
                if callable(supports_shared):
                    if not bool(supports_shared()):
                        raise ValueError(
                            "shared-expert operator requested for a config that "
                            "does not support shared experts"
                        )
                elif getattr(config, "share_expert_dim", None) is None:
                    raise ValueError(
                        "shared-expert operator requires share_expert_dim"
                    )
            selected_kind = operator_contract.layer_kind
        elif layer_id is not None:
            selected_kind = config_kind
        else:
            if mixed_model:
                raise ValueError(
                    "mixed-model FFN contract resolution requires layer_id or operator_name"
                )
            selected_kind = config_kind

        spec = operator_contract or self.get_layer_contract(selected_kind)
        width = spec.resolve_width(config, mixed_model=mixed_model)
        tp_size = self._resolve_tensor_parallel_size(
            spec,
            selected_kind,
            attention_tp_size=attention_tp_size,
            moe_tp_size=moe_tp_size,
            ffn_tp_size=ffn_tp_size,
            tensor_parallel_size=tensor_parallel_size,
        )
        if tp_size is not None:
            if type(tp_size) is not int or tp_size <= 0:
                raise ValueError(
                    f"tensor parallel size must be a positive int, got {tp_size!r}"
                )
            if spec.tensor_parallel_mode is not TensorParallelMode.REPLICATED:
                if width % tp_size != 0:
                    raise ValueError(
                        f"{selected_kind.value} width {width} must be divisible by "
                        f"tensor parallel size {tp_size}"
                    )

        if spec.expert_parallel_mode is ExpertParallelMode.ON:
            num_experts = getattr(config, "num_experts", None)
            if num_experts is not None and (type(num_experts) is not int or num_experts <= 0):
                raise ValueError(
                    "num_experts must be a positive int, got " f"{num_experts!r}"
                )
            if expert_parallel_size is not None:
                if type(expert_parallel_size) is not int or expert_parallel_size <= 0:
                    raise ValueError(
                        "expert_parallel_size must be a positive int, got "
                        f"{expert_parallel_size!r}"
                    )
                if type(num_experts) is not int or num_experts <= 0:
                    raise ValueError(
                        "num_experts must be a positive int when "
                        "expert_parallel_size is provided, got "
                        f"{num_experts!r}"
                    )
                if num_experts % expert_parallel_size != 0:
                    raise ValueError(
                        f"num_experts ({num_experts}) must be divisible by "
                        f"expert_parallel_size ({expert_parallel_size})"
                    )
        else:
            expert_parallel_size = None

        return ResolvedLayerContract(
            profile_id=self.profile_id,
            layer_id=layer_id,
            layer_kind=selected_kind,
            dimension_source=spec.dimension_source,
            effective_ffn_width=width,
            tensor_parallel_mode=spec.tensor_parallel_mode,
            expert_parallel_mode=spec.expert_parallel_mode,
            tensor_parallel_size=tp_size,
            expert_parallel_size=expert_parallel_size,
            operator_family_id=operator_family_id,
        )

    @staticmethod
    def _validate_positive_int(name: str, value: Any) -> None:
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be a positive int, got {value!r}")

    @staticmethod
    def _resolve_tensor_parallel_size(
        spec: LayerContractSpec,
        layer_kind: LayerKind,
        *,
        attention_tp_size: int | None,
        moe_tp_size: int | None,
        ffn_tp_size: int | None,
        tensor_parallel_size: int | None,
    ) -> int | None:
        ModelArchitectureProfile._validate_positive_int(
            "tensor_parallel_size", tensor_parallel_size
        )
        ModelArchitectureProfile._validate_positive_int(
            "attention_tp_size", attention_tp_size
        )
        ModelArchitectureProfile._validate_positive_int("moe_tp_size", moe_tp_size)
        ModelArchitectureProfile._validate_positive_int("ffn_tp_size", ffn_tp_size)

        domain_tp_size: int | None
        if spec.tensor_parallel_mode is TensorParallelMode.REPLICATED:
            domain_tp_size = 1
        elif spec.tensor_parallel_mode is TensorParallelMode.ATTENTION_TP:
            domain_tp_size = attention_tp_size
        elif spec.tensor_parallel_mode is TensorParallelMode.MOE_TP:
            domain_tp_size = moe_tp_size
        elif spec.tensor_parallel_mode is TensorParallelMode.FFN_TP:
            domain_tp_size = ffn_tp_size if ffn_tp_size is not None else attention_tp_size
        else:
            raise ValueError(
                f"Unsupported tensor parallel mode for {layer_kind.value}: "
                f"{spec.tensor_parallel_mode!r}"
            )

        if tensor_parallel_size is not None and domain_tp_size is not None:
            if tensor_parallel_size != domain_tp_size:
                domain_name = {
                    TensorParallelMode.ATTENTION_TP: "attention_tp_size",
                    TensorParallelMode.MOE_TP: "moe_tp_size",
                    TensorParallelMode.FFN_TP: "ffn_tp_size",
                    TensorParallelMode.REPLICATED: "replicated_tp_size",
                }[spec.tensor_parallel_mode]
                raise ValueError(
                    "tensor_parallel_size conflicts with "
                    f"{domain_name}: {tensor_parallel_size} != {domain_tp_size}"
                )
        selected = tensor_parallel_size if tensor_parallel_size is not None else domain_tp_size
        if selected is not None and selected <= 0:
            raise ValueError(
                "tensor parallel size must be a positive int, got " f"{selected!r}"
            )
        return selected

    @classmethod
    def _is_mixed_model(cls, config: Any) -> bool:
        if not bool(getattr(config, "is_moe", False)):
            return False
        num_layers = getattr(config, "num_layers", None)
        get_ids = getattr(config, "get_moe_layer_ids", None)
        if callable(get_ids) and type(num_layers) is int:
            ids = tuple(get_ids())
            return bool(ids) and len(ids) < num_layers
        get_count = getattr(config, "get_num_moe_layers", None)
        if callable(get_count) and type(num_layers) is int:
            count = int(get_count())
            return 0 < count < num_layers
        raw_layers = getattr(config, "moe_layers_enum", None)
        if raw_layers is None or str(raw_layers).strip() == "":
            return False
        ids = cls._parse_moe_layer_ids(raw_layers, num_layers)
        return bool(ids) and len(ids) < num_layers

    @classmethod
    def _resolve_config_layer_kind(
        cls, config: Any, layer_id: int | None
    ) -> LayerKind:
        if not bool(getattr(config, "is_moe", False)):
            return LayerKind.DENSE
        num_layers = getattr(config, "num_layers", None)
        if layer_id is None:
            return LayerKind.ROUTED
        predicate = getattr(config, "is_moe_layer", None)
        if callable(predicate):
            return LayerKind.ROUTED if bool(predicate(layer_id)) else LayerKind.DENSE
        get_ids = getattr(config, "get_moe_layer_ids", None)
        if callable(get_ids):
            return (
                LayerKind.ROUTED
                if layer_id in set(get_ids())
                else LayerKind.DENSE
            )
        raw_layers = getattr(config, "moe_layers_enum", None)
        if raw_layers is not None and str(raw_layers).strip():
            parsed_layer_ids = set(cls._parse_moe_layer_ids(raw_layers, num_layers))
            return (
                LayerKind.ROUTED
                if layer_id in parsed_layer_ids
                else LayerKind.DENSE
            )
        return LayerKind.ROUTED

    @staticmethod
    def _parse_moe_layer_ids(raw_layers: Any, num_layers: Any) -> tuple[int, ...]:
        """Parse an explicit MoE layer map with one strict fail-fast contract."""

        if type(num_layers) is not int or num_layers <= 0:
            raise ValueError(
                "moe_layers_enum validation requires a positive integer num_layers"
            )
        if not isinstance(raw_layers, str):
            raise ValueError(
                f"moe_layers_enum must be a comma-separated string, got {raw_layers!r}"
            )
        if raw_layers.strip() == "":
            return tuple(range(num_layers))

        parsed: list[int] = []
        seen: set[int] = set()
        for raw_token in raw_layers.split(","):
            token = raw_token.strip()
            if not re.fullmatch(r"[+-]?\d+", token):
                raise ValueError(
                    f"Invalid moe_layers_enum token {token!r} in {raw_layers!r}"
                )
            layer_id = int(token)
            if layer_id < 0 or layer_id >= num_layers:
                raise ValueError(
                    f"moe_layers_enum layer id {layer_id} out of range "
                    f"[0, {num_layers})"
                )
            if layer_id in seen:
                raise ValueError(
                    f"moe_layers_enum contains duplicate layer id {layer_id}"
                )
            seen.add(layer_id)
            parsed.append(layer_id)
        return tuple(sorted(parsed))

    def uses_expert_parallel_alltoall(
        self,
        cluster_type: ClusterType,
        expected_ep_size: int,
    ) -> bool:
        """Return whether EP synchronization should use alltoall semantics."""

        if self.expert_parallel_collective is not ExpertParallelCollective.ALLTOALL:
            return False
        return cluster_type in (
            ClusterType.PREFILL,
            ClusterType.DECODE,
            ClusterType.DECODE_FFN,
            ClusterType.MONOLITHIC,
        )


def _model_identifier(config: Any) -> str:
    identifier_parts = []
    get_name = getattr(config, "get_name", None)
    if callable(get_name):
        identifier_parts.append(f"name={get_name()}")
    for attr_name in ("_model_name", "name", "model_type", "model_arch"):
        attr_value = getattr(config, attr_name, None)
        if attr_value:
            identifier_parts.append(f"{attr_name}={attr_value}")
    if not identifier_parts:
        return "unknown model"
    return ", ".join(str(part) for part in identifier_parts)


def _moe_share_expert_requirements() -> tuple[StructuralRequirement, ...]:
    return (
        StructuralRequirement(
            name="requires_moe",
            predicate=lambda config: bool(getattr(config, "is_moe", False)),
            message=lambda profile, config: (
                f"{profile.display_name} profile {profile.profile_id} "
                f"requires is_moe=True. Model: {_model_identifier(config)}"
            ),
        ),
        StructuralRequirement(
            name="requires_share_expert_dim",
            predicate=lambda config: getattr(config, "share_expert_dim", None)
            is not None,
            message=lambda profile, config: (
                f"{profile.display_name} profile {profile.profile_id} "
                f"requires share_expert_dim. Model: {_model_identifier(config)}"
            ),
        ),
    )


def _requires_attention_family(expected_family_id: str) -> StructuralRequirement:
    def predicate(config: Any) -> bool:
        from frontier.attention.model_binding import bind_attention_family

        return bind_attention_family(config).family_id == expected_family_id

    def message(profile: ModelArchitectureProfile, config: Any) -> str:
        from frontier.attention.model_binding import bind_attention_family

        try:
            actual_family_id = bind_attention_family(config).family_id
        except ValueError as exc:
            actual_family_id = f"invalid attention binding ({exc})"
        return (
            f"{profile.display_name} profile {profile.profile_id} requires "
            f"attention family {expected_family_id}, got {actual_family_id}. "
            f"Model: {_model_identifier(config)}"
        )

    return StructuralRequirement(
        name=f"requires_attention_family_{expected_family_id}",
        predicate=predicate,
        message=message,
    )


def _requires_step3_mfa_attention_contract() -> StructuralRequirement:
    def predicate(config: Any) -> bool:
        from frontier.attention.model_binding import bind_attention_family

        binding = bind_attention_family(config)
        return (
            bool(getattr(config, "use_mfa", False))
            and binding.family_id == "dense_attention"
            and binding.variant_id == "mqa"
        )

    def message(profile: ModelArchitectureProfile, config: Any) -> str:
        from frontier.attention.model_binding import bind_attention_family

        try:
            binding = bind_attention_family(config)
            actual = f"{binding.family_id}/{binding.variant_id}"
        except ValueError as exc:
            actual = f"invalid attention binding ({exc})"
        return (
            f"{profile.display_name} profile {profile.profile_id} requires "
            "use_mfa=True with dense_attention/mqa attention binding, got "
            f"{actual}. Model: {_model_identifier(config)}"
        )

    return StructuralRequirement(
        name="requires_step3_mfa_attention_contract",
        predicate=predicate,
        message=message,
    )


def _normalized_attr(config: Any, attr_name: str) -> str:
    return str(getattr(config, attr_name, None) or "").lower()


def _matches_step2_mini(config: Any) -> bool:
    return _normalized_attr(config, "model_arch") == "step2_mini" or _normalized_attr(
        config, "model_type"
    ) == "step2_mini"


def _matches_step3_text(config: Any) -> bool:
    return _normalized_attr(config, "model_type") == "step3_text"


class ModelArchitectureRegistry:
    """Ordered plugin registry for model architecture profiles."""

    def __init__(self) -> None:
        self._profiles_by_id: OrderedDict[str, ModelArchitectureProfile] = OrderedDict()

    def register(self, profile: ModelArchitectureProfile) -> None:
        if profile.profile_id in self._profiles_by_id:
            raise ValueError(f"Duplicate model architecture profile: {profile.profile_id}")
        if profile.expert_parallel_collective is not ExpertParallelCollective.ALLTOALL:
            raise ValueError(
                "Model architecture profiles must declare "
                "expert_parallel_collective=ALLTOALL: "
                f"{profile.profile_id} declares {profile.expert_parallel_collective.value}"
            )
        self._profiles_by_id[profile.profile_id] = profile

    def get(self, profile_id: str) -> ModelArchitectureProfile:
        try:
            return self._profiles_by_id[profile_id]
        except KeyError as exc:
            raise ValueError(f"Unknown model architecture profile: {profile_id}") from exc

    def iter_profiles(self) -> tuple[ModelArchitectureProfile, ...]:
        return tuple(self._profiles_by_id.values())

    def resolve(self, config: Any) -> ModelArchitectureProfile:
        explicit_profile = getattr(config, "model_architecture_profile", None)
        if explicit_profile:
            return self.get(str(explicit_profile).lower())
        for profile in self.iter_profiles():
            if profile.match(config):
                return profile
        generic_profile = self.get("generic")
        logger.warning(
            "Model architecture profile fallback selected generic for %s",
            _model_identifier(config),
        )
        return generic_profile


MODEL_ARCHITECTURE_REGISTRY = ModelArchitectureRegistry()
for _profile in (
    ModelArchitectureProfile.step3_text(),
    ModelArchitectureProfile.step2_mini(),
    ModelArchitectureProfile.generic(),
):
    MODEL_ARCHITECTURE_REGISTRY.register(_profile)


def get_model_architecture_profile(config: Any) -> ModelArchitectureProfile:
    """Resolve a profile from the config's current declarative state.

    Runtime ``BaseModelConfig`` consumers use the config-owned accessor so one
    simulation instance retains its construction-time resolved identity.
    """

    return MODEL_ARCHITECTURE_REGISTRY.resolve(config)
