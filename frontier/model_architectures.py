"""Plugin-style model architecture contracts for model-specific runtime semantics."""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Literal, cast

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
    """Typed FFN domain exposed by a model architecture profile."""

    DENSE = "dense"
    ROUTED = "routed"
    SHARED = "shared"


class LayerDimensionSource(Enum):
    """Configuration field that supplies a typed FFN width."""

    DENSE = "dense_mlp_hidden_dim"
    ROUTED = "routed_mlp_hidden_dim"
    SHARED = "share_expert_dim"


class ExpertParallelMode(Enum):
    """Whether a typed FFN domain uses expert parallelism."""

    OFF = "off"
    ON = "on"


LayerActivationPredicate = Callable[["ModelArchitectureProfile", Any], bool]


def parse_moe_layer_ids(raw_layers: Any, num_layers: Any) -> tuple[int, ...]:
    """Parse a model's explicit MoE layer map with strict validation."""

    if type(num_layers) is not int or num_layers <= 0:
        raise ValueError("moe layer validation requires a positive num_layers")
    if raw_layers is None or (isinstance(raw_layers, str) and not raw_layers.strip()):
        return tuple(range(num_layers))
    if not isinstance(raw_layers, str):
        raise ValueError(
            f"moe_layers_enum must be a comma-separated string, got {raw_layers!r}"
        )
    parsed: list[int] = []
    seen: set[int] = set()
    for raw_token in raw_layers.split(","):
        token = raw_token.strip()
        if not re.fullmatch(r"[+-]?\d+", token):
            raise ValueError(f"Invalid moe_layers_enum token {token!r}")
        layer_id = int(token)
        if layer_id < 0 or layer_id >= num_layers:
            raise ValueError(
                f"moe_layers_enum layer id {layer_id} out of range [0, {num_layers})"
            )
        if layer_id in seen:
            raise ValueError(f"moe_layers_enum contains duplicate layer id {layer_id}")
        seen.add(layer_id)
        parsed.append(layer_id)
    return tuple(sorted(parsed))


@dataclass(frozen=True)
class LayerContractSpec:
    """Declarative contract for one dense, routed, or shared FFN domain."""

    layer_kind: LayerKind
    dimension_source: LayerDimensionSource
    tensor_parallel_mode: TensorParallelMode
    expert_parallel_mode: ExpertParallelMode = ExpertParallelMode.OFF
    operator_family_ids: tuple[str, ...] = ()
    base_layer_kinds: tuple[LayerKind, ...] = ()
    activation_predicate: LayerActivationPredicate | None = field(
        default=None, repr=False, compare=False
    )

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
        if any(not isinstance(value, str) or not value for value in family_ids):
            raise ValueError("layer contract operator family IDs must be non-empty strings")
        if len(set(family_ids)) != len(family_ids):
            raise ValueError(f"layer contract has duplicate operator families: {family_ids}")
        object.__setattr__(self, "operator_family_ids", family_ids)
        base_kinds = tuple(self.base_layer_kinds) or (self.layer_kind,)
        normalized_base_kinds = tuple(
            value if isinstance(value, LayerKind) else LayerKind(value)
            for value in base_kinds
        )
        if len(set(normalized_base_kinds)) != len(normalized_base_kinds):
            raise ValueError(f"layer contract has duplicate base layer kinds: {base_kinds}")
        object.__setattr__(self, "base_layer_kinds", normalized_base_kinds)
        expected_source = {
            LayerKind.DENSE: LayerDimensionSource.DENSE,
            LayerKind.ROUTED: LayerDimensionSource.ROUTED,
            LayerKind.SHARED: LayerDimensionSource.SHARED,
        }[self.layer_kind]
        if self.dimension_source is not expected_source:
            raise ValueError(
                f"{self.layer_kind.value} contract must use {expected_source.value}"
            )
        if self.activation_predicate is not None and not callable(self.activation_predicate):
            raise ValueError("layer contract activation_predicate must be callable")

    def is_active(self, profile: "ModelArchitectureProfile", config: Any) -> bool:
        if self.activation_predicate is None:
            return True
        return bool(self.activation_predicate(profile, config))

    @property
    def tp_mode(self) -> TensorParallelMode:
        return self.tensor_parallel_mode

    @property
    def ep_mode(self) -> ExpertParallelMode:
        return self.expert_parallel_mode

    def resolve_width(self, config: Any, *, mixed_model: bool = False) -> int:
        """Resolve this contract's width without coercing malformed values."""

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
        value = next(
            (getattr(config, name, None) for name in source_attributes
             if getattr(config, name, None) is not None),
            None,
        )
        if (
            mixed_model
            and self.dimension_source is LayerDimensionSource.ROUTED
            and getattr(config, "routed_mlp_hidden_dim", None) is None
        ):
            raise ValueError(
                "routed layer width requires routed_mlp_hidden_dim for a mixed model"
            )
        if value is None and self.dimension_source is LayerDimensionSource.DENSE:
            if mixed_model:
                raise ValueError(
                    "dense layer width requires dense_mlp_hidden_dim for a mixed model"
                )
            value = getattr(config, "mlp_hidden_dim", None)
        if type(value) is not int or value <= 0:
            raise ValueError(
                f"{self.layer_kind.value} layer width must be a positive int, got {value!r}"
            )
        return value


@dataclass(frozen=True)
class ResolvedLayerContract:
    """Immutable typed layer contract after configuration resolution."""

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
    operator_family_ids: tuple[str, ...] = ()
    tensor_parallel_sizes: tuple[int, ...] = ()
    selected_padded_ffn_width: int | None = None

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

    def semantic_identity(self) -> tuple[object, ...]:
        """Return semantic identity fields while ignoring physical layer occurrence."""

        return (
            self.profile_id,
            self.layer_kind.value,
            self.dimension_source.value,
            self.effective_ffn_width,
            self.tensor_parallel_mode.value,
            self.expert_parallel_mode.value,
            self.tensor_parallel_size,
            self.expert_parallel_size,
            self.operator_family_id,
            tuple(self.operator_family_ids),
            tuple(self.tensor_parallel_sizes),
            self.selected_padded_ffn_width,
        )

    def is_semantically_equivalent(self, other: object) -> bool:
        return isinstance(other, ResolvedLayerContract) and self.semantic_identity() == other.semantic_identity()

    def typed_metadata_identity(self) -> dict[str, object]:
        """Return the canonical typed metadata fields for one operator row."""

        family_ids = self.operator_family_ids
        family_id = self.operator_family_id
        if not family_ids and family_id is not None:
            family_ids = (family_id,)
        if family_id is None and len(family_ids) == 1:
            family_id = family_ids[0]
        if family_id is None:
            raise ValueError(
                "typed metadata identity requires one operator family"
            )
        tp_sizes = self.tensor_parallel_sizes
        if not tp_sizes and self.tensor_parallel_size is not None:
            tp_sizes = (self.tensor_parallel_size,)
        padded_width = self.selected_padded_ffn_width
        if padded_width is None and self.tensor_parallel_size is not None:
            padded_width = _pad_width(self.effective_ffn_width, self.tensor_parallel_size)
        return {
            "profile_id": self.profile_id,
            "operator_family_id": family_id,
            "operator_family_ids": list(family_ids),
            "layer_kind": self.layer_kind.value,
            "dimension_source": self.dimension_source.value,
            "effective_ffn_width": self.effective_ffn_width,
            "tensor_parallel_mode": self.tensor_parallel_mode.value,
            "expert_parallel_mode": self.expert_parallel_mode.value,
            "selected_expert_parallel_size": self.expert_parallel_size,
            "tensor_parallel_sizes": list(tp_sizes),
            "selected_tensor_parallel_size": self.tensor_parallel_size,
            "selected_padded_ffn_width": padded_width,
        }


def _pad_width(width: int, tp_size: int) -> int:
    if type(tp_size) is not int or tp_size <= 0:
        raise ValueError(f"tensor parallel size must be positive, got {tp_size!r}")
    return ((width + tp_size - 1) // tp_size) * tp_size


def _dense_layer_contract_active(profile: "ModelArchitectureProfile", config: Any) -> bool:
    return not bool(getattr(config, "is_moe", False)) or profile._is_mixed_model(config)


def _routed_layer_contract_active(_profile: "ModelArchitectureProfile", config: Any) -> bool:
    return bool(getattr(config, "is_moe", False))


def _shared_layer_contract_active(profile: "ModelArchitectureProfile", config: Any) -> bool:
    if not bool(getattr(config, "is_moe", False)):
        return False
    supports = getattr(config, "supports_share_expert", None)
    return bool(supports()) if callable(supports) else profile.supports_share_expert(config)


def _default_layer_contracts() -> tuple[LayerContractSpec, ...]:
    return (
        LayerContractSpec(
            LayerKind.DENSE,
            LayerDimensionSource.DENSE,
            TensorParallelMode.FFN_TP,
            operator_family_ids=("ffn",),
            activation_predicate=_dense_layer_contract_active,
        ),
        LayerContractSpec(
            LayerKind.ROUTED,
            LayerDimensionSource.ROUTED,
            TensorParallelMode.MOE_TP,
            ExpertParallelMode.ON,
            operator_family_ids=("moe",),
            activation_predicate=_routed_layer_contract_active,
        ),
        LayerContractSpec(
            LayerKind.SHARED,
            LayerDimensionSource.SHARED,
            TensorParallelMode.FFN_TP,
            operator_family_ids=("share_expert",),
            base_layer_kinds=(LayerKind.ROUTED,),
            activation_predicate=_shared_layer_contract_active,
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
                f"model architecture profile {self.profile_id!r} must declare layer contracts"
            )
        layer_kinds = [contract.layer_kind for contract in self.layer_contracts]
        if len(set(layer_kinds)) != len(layer_kinds):
            raise ValueError(
                f"model architecture profile {self.profile_id!r} declares duplicate layer kinds"
            )
        family_ids = [
            family_id
            for contract in self.layer_contracts
            for family_id in contract.operator_family_ids
        ]
        if len(set(family_ids)) != len(family_ids):
            raise ValueError(
                f"model architecture profile {self.profile_id!r} declares duplicate family ownership"
            )
        try:
            from frontier.operators.families import get_operator_family

            for family_id in family_ids:
                get_operator_family(family_id)
        except (ImportError, TypeError, ValueError) as exc:
            raise ValueError(
                f"model architecture profile {self.profile_id!r} references unknown operator family"
            ) from exc

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
                    activation_predicate=_dense_layer_contract_active,
                ),
                LayerContractSpec(
                    LayerKind.ROUTED,
                    LayerDimensionSource.ROUTED,
                    TensorParallelMode.MOE_TP,
                    ExpertParallelMode.ON,
                    operator_family_ids=("moe",),
                    activation_predicate=_routed_layer_contract_active,
                ),
                LayerContractSpec(
                    LayerKind.SHARED,
                    LayerDimensionSource.SHARED,
                    TensorParallelMode.ATTENTION_TP,
                    operator_family_ids=("share_expert",),
                    base_layer_kinds=(LayerKind.ROUTED,),
                    activation_predicate=_shared_layer_contract_active,
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

    def iter_active_layer_contracts(self, config: Any) -> tuple[LayerContractSpec, ...]:
        """Return the profile-owned contracts active for a model config."""

        if config is None:
            raise ValueError("model config is required for layer contract resolution")
        return tuple(
            contract
            for contract in self.layer_contracts
            if contract.is_active(self, config)
        )

    def get_layer_contract(self, layer_kind: LayerKind | str) -> LayerContractSpec:
        if not isinstance(layer_kind, LayerKind):
            try:
                layer_kind = LayerKind(layer_kind)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Unknown layer kind: {layer_kind!r}") from exc
        for contract in self.layer_contracts:
            if contract.layer_kind is layer_kind:
                return contract
        raise ValueError(
            f"Profile {self.profile_id!r} does not declare layer kind {layer_kind.value!r}"
        )

    def get_layer_contract_for_family(self, family_id: str) -> LayerContractSpec:
        if not isinstance(family_id, str) or not family_id:
            raise ValueError("operator family ID must be a non-empty string")
        matches = tuple(
            contract
            for contract in self.layer_contracts
            if family_id in contract.operator_family_ids
        )
        if len(matches) != 1:
            raise ValueError(
                f"Profile {self.profile_id!r} declares {len(matches)} contracts for family {family_id!r}"
            )
        return matches[0]

    def resolve_layer_contract(
        self,
        config: Any,
        *,
        layer_id: int | None = None,
        layer_kind: LayerKind | str | None = None,
        operator_name: str | None = None,
        attention_tp_size: int | None = None,
        attn_tp_size: int | None = None,
        moe_tp_size: int | None = None,
        ffn_tp_size: int | None = None,
        tensor_parallel_size: int | None = None,
        expert_parallel_size: int | None = None,
        ep_size: int | None = None,
    ) -> ResolvedLayerContract:
        """Resolve layer kind, width, and parallel domains from one profile."""

        if config is None:
            raise ValueError("model config is required for layer contract resolution")
        for name, value in (
            ("attention_tp_size", attention_tp_size),
            ("attn_tp_size", attn_tp_size),
            ("moe_tp_size", moe_tp_size),
            ("ffn_tp_size", ffn_tp_size),
            ("tensor_parallel_size", tensor_parallel_size),
            ("expert_parallel_size", expert_parallel_size),
            ("ep_size", ep_size),
        ):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if attention_tp_size is not None and attn_tp_size is not None:
            if attention_tp_size != attn_tp_size:
                raise ValueError("attention_tp_size and attn_tp_size disagree")
        attention_tp_size = attention_tp_size or attn_tp_size
        if expert_parallel_size is not None and ep_size is not None:
            if expert_parallel_size != ep_size:
                raise ValueError("expert_parallel_size and ep_size disagree")
        expert_parallel_size = expert_parallel_size or ep_size

        if layer_id is not None:
            if type(layer_id) is not int:
                raise ValueError(f"layer_id must be an int, got {layer_id!r}")
            num_layers = getattr(config, "num_layers", None)
            if type(num_layers) is not int or num_layers <= 0:
                raise ValueError("config must declare a positive num_layers")
            if layer_id < 0 or layer_id >= num_layers:
                raise ValueError(f"layer_id {layer_id} out of range [0, {num_layers})")
        if layer_kind is not None and not isinstance(layer_kind, LayerKind):
            try:
                layer_kind = LayerKind(layer_kind)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Unknown layer kind: {layer_kind!r}") from exc

        operator_family_id = None
        operator_contract = None
        operator_ep_agnostic = False
        if operator_name is not None:
            if not isinstance(operator_name, str) or not operator_name:
                raise ValueError("operator_name must be a non-empty string")
            try:
                from frontier.operators.binding import bind_operator_query

                operator_binding = bind_operator_query(operator_name)
                operator_family_id = operator_binding.family_id
                # EP semantics belong to the operator registry, not to the
                # layer kind.  A routed family can contain both EP-agnostic
                # routing work and EP-sensitive grouped GEMM.
                operator_ep_agnostic = bool(operator_binding.operator.ep_agnostic)
            except (ImportError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Unable to bind operator {operator_name!r} to a layer contract"
                ) from exc
            operator_contract = self.get_layer_contract_for_family(operator_family_id)

        if layer_kind is not None:
            kind_contract = self.get_layer_contract(layer_kind)
            if operator_contract is not None and kind_contract is not operator_contract:
                raise ValueError("layer_kind conflicts with operator_name contract")
            operator_contract = kind_contract

        config_kind = self._resolve_config_layer_kind(config, layer_id)
        mixed_model = self._is_mixed_model(config)
        if operator_contract is not None:
            if operator_contract.layer_kind in (LayerKind.ROUTED, LayerKind.SHARED) and not bool(
                getattr(config, "is_moe", False)
            ):
                raise ValueError(
                    f"{operator_contract.layer_kind.value} operator requires an MoE model"
                )
            if layer_id is not None and config_kind not in operator_contract.base_layer_kinds:
                allowed = ", ".join(kind.value for kind in operator_contract.base_layer_kinds)
                raise ValueError(
                    f"operator {operator_name!r} requires base layer kind {allowed}; "
                    f"layer_id={layer_id} is {config_kind.value}"
                )
            if operator_contract.layer_kind is LayerKind.SHARED:
                supports = getattr(config, "supports_share_expert", None)
                if callable(supports) and not bool(supports()):
                    raise ValueError("shared-expert operator requires shared-expert support")
            selected_kind = operator_contract.layer_kind
        elif layer_id is not None:
            selected_kind = config_kind
        elif mixed_model:
            raise ValueError("mixed-model resolution requires layer_id or operator_name")
        else:
            selected_kind = config_kind

        spec = operator_contract or self.get_layer_contract(selected_kind)
        if not spec.is_active(self, config):
            raise ValueError(
                f"{spec.layer_kind.value} layer contract is inactive for profile {self.profile_id!r}"
            )
        width = spec.resolve_width(config, mixed_model=mixed_model)
        if spec.tensor_parallel_mode is TensorParallelMode.REPLICATED:
            selected_tp = 1
        elif spec.tensor_parallel_mode is TensorParallelMode.ATTENTION_TP:
            selected_tp = attention_tp_size
        elif spec.tensor_parallel_mode is TensorParallelMode.MOE_TP:
            selected_tp = moe_tp_size
        elif spec.tensor_parallel_mode is TensorParallelMode.FFN_TP:
            selected_tp = ffn_tp_size if ffn_tp_size is not None else attention_tp_size
        else:
            raise ValueError(f"Unsupported tensor parallel mode: {spec.tensor_parallel_mode!r}")
        if tensor_parallel_size is not None and selected_tp is not None and tensor_parallel_size != selected_tp:
            raise ValueError("tensor_parallel_size conflicts with the selected domain TP")
        selected_tp = tensor_parallel_size if tensor_parallel_size is not None else selected_tp
        if selected_tp is not None and spec.tensor_parallel_mode is not TensorParallelMode.REPLICATED:
            if width % selected_tp != 0:
                raise ValueError(
                    f"{selected_kind.value} width {width} must be divisible by TP={selected_tp}"
                )

        if spec.expert_parallel_mode is ExpertParallelMode.ON:
            num_experts = getattr(config, "num_experts", None)
            if type(num_experts) is not int or num_experts <= 0:
                raise ValueError("routed contract requires a positive num_experts")
            if expert_parallel_size is not None and num_experts % expert_parallel_size != 0:
                raise ValueError(
                    f"num_experts ({num_experts}) must be divisible by EP={expert_parallel_size}"
                )
            if operator_ep_agnostic:
                # Keep validating the runtime topology above, but omit EP from
                # the selected estimator identity for operations whose timing
                # is independent of expert placement.
                expert_parallel_size = None
        else:
            expert_parallel_size = None

        domain_sizes = ()
        if selected_tp is not None:
            domain_sizes = (selected_tp,)
        return ResolvedLayerContract(
            profile_id=self.profile_id,
            layer_id=layer_id,
            layer_kind=selected_kind,
            dimension_source=spec.dimension_source,
            effective_ffn_width=width,
            tensor_parallel_mode=spec.tensor_parallel_mode,
            expert_parallel_mode=spec.expert_parallel_mode,
            tensor_parallel_size=selected_tp,
            expert_parallel_size=expert_parallel_size,
            operator_family_id=operator_family_id,
            operator_family_ids=spec.operator_family_ids,
            tensor_parallel_sizes=domain_sizes,
            selected_padded_ffn_width=(
                _pad_width(width, selected_tp) if selected_tp is not None else None
            ),
        )

    @classmethod
    def _is_mixed_model(cls, config: Any) -> bool:
        if not bool(getattr(config, "is_moe", False)):
            return False
        num_layers = getattr(config, "num_layers", None)
        if type(num_layers) is not int or num_layers <= 0:
            if any(
                callable(getattr(config, name, None))
                for name in ("get_moe_layer_ids", "is_moe_layer")
            ) or getattr(config, "moe_layers_enum", None) is not None:
                raise ValueError("moe layer validation requires a positive num_layers")
            return False
        ids = cls._resolve_moe_layer_ids(config)
        return 0 < len(ids) < num_layers

    @classmethod
    def _resolve_moe_layer_ids(cls, config: Any) -> tuple[int, ...]:
        """Resolve one canonical MoE layer set and verify custom adapters."""

        num_layers = getattr(config, "num_layers", None)
        if type(num_layers) is not int or num_layers <= 0:
            raise ValueError("moe layer validation requires a positive num_layers")
        getter = getattr(config, "get_moe_layer_ids", None)
        predicate = getattr(config, "is_moe_layer", None)
        if callable(getter):
            try:
                typed_getter = cast(Callable[[], Iterable[int]], getter)
                raw_ids = tuple(typed_getter())
            except TypeError as exc:
                raise ValueError("get_moe_layer_ids() must return an iterable") from exc
            if any(type(layer_id) is not int for layer_id in raw_ids):
                raise ValueError("get_moe_layer_ids() must return integer layer IDs")
            if len(set(raw_ids)) != len(raw_ids):
                raise ValueError("get_moe_layer_ids() returned duplicate layer IDs")
            if any(layer_id < 0 or layer_id >= num_layers for layer_id in raw_ids):
                raise ValueError(
                    f"get_moe_layer_ids() returned an out-of-range layer ID for [0, {num_layers})"
                )
            canonical = tuple(sorted(raw_ids))
            if callable(predicate):
                predicate_ids = tuple(
                    layer_id for layer_id in range(num_layers) if bool(predicate(layer_id))
                )
                if predicate_ids != canonical:
                    raise ValueError(
                        "custom MoE layer getter and predicate disagree: "
                        f"get_moe_layer_ids={list(canonical)}, is_moe_layer={list(predicate_ids)}"
                    )
            return canonical
        if callable(predicate):
            return tuple(
                layer_id for layer_id in range(num_layers) if bool(predicate(layer_id))
            )
        return parse_moe_layer_ids(getattr(config, "moe_layers_enum", None), num_layers)

    @classmethod
    def _resolve_config_layer_kind(cls, config: Any, layer_id: int | None) -> LayerKind:
        if not bool(getattr(config, "is_moe", False)):
            return LayerKind.DENSE
        if layer_id is None:
            return LayerKind.ROUTED
        ids = cls._resolve_moe_layer_ids(config)
        return LayerKind.ROUTED if layer_id in ids else LayerKind.DENSE

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
