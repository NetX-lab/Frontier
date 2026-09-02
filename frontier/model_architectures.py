"""Plugin-style model architecture contracts for model-specific runtime semantics."""

from __future__ import annotations

import logging
import inspect
import json
import re
from collections.abc import Iterable, Mapping
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


LayerActivationPredicate = Callable[["ModelArchitectureProfile", Any], bool]


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


# Dimension aliases are architecture-contract data.  Consumers resolve a
# source through this table instead of interpreting model-config field names
# independently.
_LAYER_DIMENSION_SOURCE_ATTRIBUTES: dict[LayerDimensionSource, tuple[str, ...]] = {
    LayerDimensionSource.DENSE: (
        "dense_mlp_hidden_dim",
        "intermediate_size",
        "mlp_hidden_dim",
    ),
    LayerDimensionSource.ROUTED: (
        "routed_mlp_hidden_dim",
        "moe_intermediate_size",
        "mlp_hidden_dim",
    ),
    LayerDimensionSource.SHARED: (
        "share_expert_dim",
        "shared_expert_intermediate_size",
    ),
}

_MISSING_CONFIG_ATTRIBUTE = object()
_DECLARED_CONFIG_ATTRIBUTE_PROVIDER = "get_declared_config_attribute"


def _get_declared_config_attribute(
    config: Any,
    attribute_name: str,
) -> Any:
    """Read an attribute only when the config actually declares it.

    Duck-typed test/config adapters may expose fields through ``__getattr__``.
    Treating every dynamically synthesized attribute as a contract would turn
    an arbitrary object (for example a bare mock) into a fake architecture
    profile. ``inspect.getattr_static`` lets the resolver distinguish a real
    instance/class declaration from such synthesized values. An adapter that
    deliberately delegates declared fields may implement the explicit
    ``get_declared_config_attribute`` protocol; the provider itself must be a
    statically declared method, so a dynamic ``__getattr__`` cannot opt an
    arbitrary value into the architecture contract.
    """

    try:
        inspect.getattr_static(config, attribute_name)
    except AttributeError:
        try:
            inspect.getattr_static(config, _DECLARED_CONFIG_ATTRIBUTE_PROVIDER)
        except AttributeError:
            return _MISSING_CONFIG_ATTRIBUTE
        provider = getattr(config, _DECLARED_CONFIG_ATTRIBUTE_PROVIDER)
        if not callable(provider):
            raise TypeError(
                f"{_DECLARED_CONFIG_ATTRIBUTE_PROVIDER} must be callable when "
                "declared"
            )
        declaration = provider(attribute_name)
        if (
            not isinstance(declaration, tuple)
            or len(declaration) != 2
            or type(declaration[0]) is not bool
        ):
            raise TypeError(
                f"{_DECLARED_CONFIG_ATTRIBUTE_PROVIDER} must return "
                "(declared: bool, value: object)"
            )
        declared, value = declaration
        return value if declared else _MISSING_CONFIG_ATTRIBUTE
    return getattr(config, attribute_name)


def _get_config_value(config: Any, attribute_name: str, default: Any = None) -> Any:
    """Return a declared config value, ignoring synthesized duck-typed attrs."""

    value = _get_declared_config_attribute(config, attribute_name)
    return default if value is _MISSING_CONFIG_ATTRIBUTE else value


def _pad_to_multiple(value: int, multiple: int) -> int:
    """Return ``value`` rounded up to the selected parallel shard multiple."""

    if type(value) is not int or value <= 0:
        raise ValueError(f"value must be a positive int, got {value!r}")
    if type(multiple) is not int or multiple <= 0:
        raise ValueError(f"multiple must be a positive int, got {multiple!r}")
    return ((value + multiple - 1) // multiple) * multiple


def _normalize_dimension_source(
    dimension_source: LayerDimensionSource | str,
) -> LayerDimensionSource:
    if isinstance(dimension_source, LayerDimensionSource):
        return dimension_source
    try:
        return LayerDimensionSource(dimension_source)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unknown layer dimension source: {dimension_source!r}"
        ) from exc


def get_declared_layer_width(
    config: Any,
    dimension_source: LayerDimensionSource | str,
    *,
    mixed_model: bool = False,
    reject_zero_sentinel: bool = False,
) -> int | None:
    """Read one profile-owned layer width from its canonical field or alias.

    ``None`` means the source is not declared.  A present but malformed value
    raises immediately so callers cannot silently fall back to a different
    typed domain.  Mixed dense layers intentionally require the explicit
    ``dense_mlp_hidden_dim`` field; the legacy ``intermediate_size`` property
    on model-config adapters may otherwise mirror the routed width.
    """

    source = _normalize_dimension_source(dimension_source)
    attributes = _LAYER_DIMENSION_SOURCE_ATTRIBUTES[source]
    if source is LayerDimensionSource.DENSE and mixed_model:
        attributes = ("dense_mlp_hidden_dim",)

    zero_attribute: str | None = None
    for attribute in attributes:
        candidate = _get_config_value(config, attribute)
        if candidate is None:
            continue
        if (
            source is LayerDimensionSource.SHARED
            and type(candidate) is int
            and candidate == 0
        ):
            zero_attribute = attribute
            continue
        if type(candidate) is not int or candidate <= 0:
            raise ValueError(
                f"{source.value} layer width must be a positive int, "
                f"got {candidate!r} from {attribute}"
            )
        return candidate
    if reject_zero_sentinel and zero_attribute is not None:
        raise ValueError(
            f"{source.value} layer width must be a positive int, got 0 "
            f"from {zero_attribute}"
        )
    return None


def _get_optional_shared_expert_width(config: Any) -> int | None:
    """Resolve the optional shared-expert width while honoring the legacy zero sentinel.

    A zero value means that the optional shared-expert path is not declared.
    Positive values still use :func:`get_declared_layer_width` for strict type
    validation, and malformed non-positive values continue to fail fast.
    Explicit shared-layer resolution intentionally calls the strict helper
    directly so a selected shared contract can never materialize with width 0.
    """

    for attribute in _LAYER_DIMENSION_SOURCE_ATTRIBUTES[LayerDimensionSource.SHARED]:
        candidate = _get_config_value(config, attribute)
        if candidate is None:
            continue
        if type(candidate) is int and candidate == 0:
            continue
        return get_declared_layer_width(config, LayerDimensionSource.SHARED)
    return None


def resolve_model_architecture_profile(config: Any) -> "ModelArchitectureProfile":
    """Return the construction-time profile when a config owns one.

    Lightweight config adapters may expose only the registry-facing profile
    selector.  Runtime configs expose a construction-time accessor so a
    single simulation keeps a stable profile snapshot; both paths converge on
    the same profile-owned contract implementation.
    """

    profile_getter = _get_declared_config_attribute(
        config,
        "get_model_architecture_profile",
    )
    if profile_getter is not _MISSING_CONFIG_ATTRIBUTE:
        if not callable(profile_getter):
            raise TypeError(
                "get_model_architecture_profile must be callable when declared"
            )
        profile = profile_getter()
        if not isinstance(profile, ModelArchitectureProfile):
            raise TypeError(
                "get_model_architecture_profile() must return "
                f"ModelArchitectureProfile, got {type(profile).__name__}"
            )
        return profile

    profile = get_model_architecture_profile(config)
    if not isinstance(profile, ModelArchitectureProfile):
        raise TypeError(
            "architecture registry resolution must return "
            f"ModelArchitectureProfile, got {type(profile).__name__}"
        )
    return profile


def normalize_moe_layer_ids(
    raw_ids: Any,
    num_layers: Any,
    *,
    source: str = "model config MoE layer IDs",
) -> tuple[int, ...]:
    """Validate and canonicalize layer IDs returned by a config adapter.

    Config files use the comma-separated ``moe_layers_enum`` representation,
    while lightweight adapters expose ``get_moe_layer_ids()`` as an iterable.
    Both representations converge here so every typed-contract consumer sees
    exact integer IDs, a sorted order, and the same duplicate/range failures.
    """

    if not isinstance(source, str) or not source.strip():
        raise ValueError("layer-map validation source must be a non-empty string")
    if type(num_layers) is not int or num_layers <= 0:
        raise ValueError(
            f"{source} validation requires a positive integer num_layers"
        )
    if isinstance(raw_ids, (str, bytes)) or isinstance(raw_ids, Mapping):
        raise ValueError(
            f"{source} must return an iterable of exact integer IDs, "
            f"got {raw_ids!r}"
        )
    if not isinstance(raw_ids, Iterable):
        raise ValueError(
            f"{source} must return an iterable of exact integer IDs, "
            f"got {raw_ids!r}"
        )
    normalized = tuple(raw_ids)
    if any(type(layer_id) is not int for layer_id in normalized):
        raise ValueError(
            f"{source} must be exact integers, got {normalized!r}"
        )
    if any(layer_id < 0 or layer_id >= num_layers for layer_id in normalized):
        raise ValueError(
            f"{source} contains a layer ID out of range; values must fall "
            f"within the model range [0, {num_layers}), "
            f"got {normalized!r}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"{source} contains duplicate layer IDs; IDs must be unique, "
            f"got {normalized!r}"
        )
    return tuple(sorted(normalized))


def parse_moe_layer_ids(raw_layers: Any, num_layers: Any) -> tuple[int, ...]:
    """Parse an explicit MoE layer map using one strict contract.

    ``None`` and an empty string represent the conventional all-layer MoE
    configuration.  Explicit maps reject empty tokens, duplicates, malformed
    values, and IDs outside the model depth so every config consumer observes
    the same layer identity.
    """

    if type(num_layers) is not int or num_layers <= 0:
        raise ValueError(
            "moe_layers_enum validation requires a positive integer num_layers"
        )
    if raw_layers is None:
        return normalize_moe_layer_ids(
            range(num_layers),
            num_layers,
            source="moe_layers_enum",
        )
    if not isinstance(raw_layers, str):
        raise ValueError(
            f"moe_layers_enum must be a comma-separated string, got {raw_layers!r}"
        )
    if raw_layers.strip() == "":
        return normalize_moe_layer_ids(
            range(num_layers),
            num_layers,
            source="moe_layers_enum",
        )

    parsed: list[int] = []
    for raw_token in raw_layers.split(","):
        token = raw_token.strip()
        if not re.fullmatch(r"[+-]?\d+", token):
            raise ValueError(
                f"Invalid moe_layers_enum token {token!r} in {raw_layers!r}"
            )
        layer_id = int(token)
        parsed.append(layer_id)
    return normalize_moe_layer_ids(
        parsed,
        num_layers,
        source="moe_layers_enum",
    )


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
        default=None,
        repr=False,
        compare=False,
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

        if self.activation_predicate is not None and not callable(
            self.activation_predicate
        ):
            raise ValueError(
                f"{self.layer_kind.value} layer contract activation_predicate "
                "must be callable when provided"
            )

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

    def is_active(self, profile: "ModelArchitectureProfile", config: Any) -> bool:
        """Evaluate the profile-owned activation rule for this layer domain."""

        if self.activation_predicate is None:
            return True
        return bool(self.activation_predicate(profile, config))

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

        value = get_declared_layer_width(
            config,
            self.dimension_source,
            mixed_model=mixed_model,
        )
        if value is None and self.dimension_source is LayerDimensionSource.DENSE:
            if mixed_model:
                raise ValueError(
                    "dense layer width requires dense_mlp_hidden_dim for a mixed "
                    "model"
                )
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
    # Keep the declared family and TP domain alongside the selected values.
    # A resolver call that supplies one concrete TP value materializes a
    # singleton domain; producers that own a wider domain can attach it when
    # they construct a contract in a later migration.
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
        """Return the identity used when two transport paths share one domain.

        ``layer_id`` is intentionally absent: a profile-level operator query
        and a concrete layer query describe the same typed domain when every
        width, parallel, and operator-family field agrees.  Cache and
        provenance callers must continue using
        :func:`serialize_layer_contract_identity`, which retains ``layer_id``.
        """

        return (
            self.profile_id,
            self.layer_kind,
            self.dimension_source,
            self.effective_ffn_width,
            self.tensor_parallel_mode,
            self.expert_parallel_mode,
            self.tensor_parallel_size,
            self.expert_parallel_size,
            self.operator_family_id,
        )

    def is_semantically_equivalent(self, other: object) -> bool:
        """Compare typed-domain semantics while ignoring layer identity scope."""

        if not isinstance(other, ResolvedLayerContract):
            return False
        return self.semantic_identity() == other.semantic_identity()

    def typed_metadata_identity(self) -> dict[str, object]:
        """Return the canonical fields required in a typed profiling row."""

        family_ids = self.operator_family_ids
        if not family_ids and self.operator_family_id is not None:
            family_ids = (self.operator_family_id,)
        operator_family_id = self.operator_family_id
        if operator_family_id is None and len(family_ids) == 1:
            operator_family_id = family_ids[0]
        if operator_family_id is None:
            raise ValueError(
                "typed metadata identity requires one resolved operator family; "
                "pass operator_name when a layer contract owns multiple families"
            )

        tensor_parallel_sizes = self.tensor_parallel_sizes
        if not tensor_parallel_sizes and self.tensor_parallel_size is not None:
            tensor_parallel_sizes = (self.tensor_parallel_size,)

        padded_width = self.selected_padded_ffn_width
        if padded_width is None and self.tensor_parallel_size is not None:
            padded_width = _pad_to_multiple(
                self.effective_ffn_width,
                self.tensor_parallel_size,
            )

        metadata = {
            "profile_id": self.profile_id,
            "operator_family_id": operator_family_id,
            "operator_family_ids": list(family_ids),
            "layer_kind": self.layer_kind.value,
            "dimension_source": self.dimension_source.value,
            "effective_ffn_width": self.effective_ffn_width,
            "tensor_parallel_mode": self.tensor_parallel_mode.value,
            "expert_parallel_mode": self.expert_parallel_mode.value,
            "selected_expert_parallel_size": self.expert_parallel_size,
            "tensor_parallel_sizes": list(tensor_parallel_sizes),
            "selected_tensor_parallel_size": self.tensor_parallel_size,
            "selected_padded_ffn_width": padded_width,
        }

        # Keep the architecture-owned value object and the shared serializer
        # on one field source.  The import is local so the architecture module
        # does not acquire pandas during package initialization.
        from frontier.operators.typed_contracts import TYPED_METADATA_REQUIRED_FIELDS

        missing_fields = tuple(
            field_name
            for field_name in TYPED_METADATA_REQUIRED_FIELDS
            if field_name not in metadata
        )
        if missing_fields:
            raise ValueError(
                "typed metadata identity is missing required fields: "
                f"{', '.join(missing_fields)}"
            )
        return {
            field_name: metadata[field_name]
            for field_name in TYPED_METADATA_REQUIRED_FIELDS
        }


def serialize_layer_contract_identity(
    layer_contract: ResolvedLayerContract | None,
) -> str | None:
    """Serialize one resolved contract for stable cache and provenance identity.

    Cache producers and consumers must use the same representation. Keeping
    this serialization next to the profile-owned value object prevents a
    training-only or runtime-only identity from silently diverging.
    """

    if layer_contract is None:
        return None
    if not isinstance(layer_contract, ResolvedLayerContract):
        raise TypeError(
            "layer_contract must be a ResolvedLayerContract when provided"
        )
    payload = {
        "profile_id": layer_contract.profile_id,
        "layer_id": layer_contract.layer_id,
        "layer_kind": layer_contract.layer_kind.value,
        "dimension_source": layer_contract.dimension_source.value,
        "effective_ffn_width": layer_contract.effective_ffn_width,
        "tensor_parallel_mode": layer_contract.tensor_parallel_mode.value,
        "expert_parallel_mode": layer_contract.expert_parallel_mode.value,
        "tensor_parallel_size": layer_contract.tensor_parallel_size,
        "expert_parallel_size": layer_contract.expert_parallel_size,
        "operator_family_id": layer_contract.operator_family_id,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _dense_layer_contract_active(
    profile: "ModelArchitectureProfile", config: Any
) -> bool:
    """Activate dense layers for dense models and mixed-layer MoE boundaries."""

    return not bool(_get_config_value(config, "is_moe", False)) or profile._is_mixed_model(config)


def _routed_layer_contract_active(
    _profile: "ModelArchitectureProfile", config: Any
) -> bool:
    """Activate routed layers for MoE model configurations."""

    return bool(_get_config_value(config, "is_moe", False))


def _shared_layer_contract_active(
    profile: "ModelArchitectureProfile", config: Any
) -> bool:
    """Activate shared-expert layers only when the profile supports them."""

    if not bool(_get_config_value(config, "is_moe", False)):
        return False
    configured_support = _get_declared_config_attribute(
        config,
        "supports_share_expert",
    )
    if configured_support is not _MISSING_CONFIG_ATTRIBUTE and callable(
        configured_support
    ):
        return bool(configured_support())
    return profile.supports_share_expert(config)


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
            raise ValueError(
                f"{self.message(profile, config)}: {exc}"
            ) from exc
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
        family_id: str | None = None
        try:
            from frontier.operators.families import get_operator_family

            for family_id in family_ids:
                get_operator_family(family_id)
        except (ImportError, TypeError, ValueError) as exc:
            raise ValueError(
                f"model architecture profile {self.profile_id!r} references an "
                f"unknown operator family: {family_id!r}"
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
        if not bool(_get_config_value(config, "is_moe", False)):
            return False
        return (
            _get_optional_shared_expert_width(config) is not None
        )

    def iter_active_layer_contracts(
        self,
        config: Any,
    ) -> tuple[LayerContractSpec, ...]:
        """Return the typed layer contracts materialized by ``config``.

        The profile owns activation of its declared domains. This keeps the
        shared prediction manager independent of a fixed dense/routed/shared
        family tuple while preserving the established pure-dense, pure-MoE,
        and mixed-layer semantics.
        """

        if config is None:
            raise ValueError("active layer contract resolution requires a model config")

        active_contracts: list[LayerContractSpec] = []
        for contract in self.layer_contracts:
            if contract.is_active(self, config):
                active_contracts.append(contract)
        return tuple(active_contracts)

    def get_layer_contract_identity(self, config: Any) -> tuple[object, ...]:
        """Return a stable identity for all active typed layer contracts.

        Runtime caches must distinguish configurations that retain the same
        legacy ``mlp_hidden_dim`` but declare different typed widths or layer
        maps or model depth.  The profile owns both the semantic contract
        fields and the width aliases, so cache callers only need this one
        identity method.
        Parallel sizes remain in the caller's topology key; the identity here
        records the semantic TP/EP modes without creating a second topology
        source of truth.
        """

        if config is None:
            raise ValueError("layer contract identity requires a model config")

        mixed_model = self._is_mixed_model(config)
        contract_identity = tuple(
            (
                contract.layer_kind.value,
                contract.dimension_source.value,
                contract.resolve_width(config, mixed_model=mixed_model),
                contract.tensor_parallel_mode.value,
                contract.expert_parallel_mode.value,
                tuple(contract.operator_family_ids),
                tuple(base_kind.value for base_kind in contract.base_layer_kinds),
            )
            for contract in self.iter_active_layer_contracts(config)
        )

        if not bool(_get_config_value(config, "is_moe", False)):
            moe_layer_ids: tuple[int, ...] = ()
        else:
            raw_layers = _get_config_value(config, "moe_layers_enum")
            if raw_layers is not None:
                moe_layer_ids = parse_moe_layer_ids(
                    raw_layers,
                    _get_config_value(config, "num_layers"),
                )
            else:
                get_layer_ids = _get_declared_config_attribute(
                    config,
                    "get_moe_layer_ids",
                )
                if get_layer_ids is not _MISSING_CONFIG_ATTRIBUTE and callable(
                    get_layer_ids
                ):
                    num_layers = _get_config_value(config, "num_layers")
                    moe_layer_ids = normalize_moe_layer_ids(
                        get_layer_ids(),
                        num_layers,
                    )
                else:
                    moe_layer_ids = parse_moe_layer_ids(
                        None,
                        _get_config_value(config, "num_layers"),
                    )

        # Keep the existing leading tuple members stable for callers that
        # inspect profile, contract, and layer-map components positionally.
        # Model depth remains an independent identity dimension: the same
        # explicit MoE IDs can be valid in models with different trailing
        # dense layers.
        num_layers = _get_config_value(config, "num_layers")
        return (self.profile_id, contract_identity, moe_layer_ids, num_layers)

    def serialize_layer_contract_identity(
        self,
        config: Any,
        layer_contract: ResolvedLayerContract | None = None,
    ) -> str:
        """Serialize the profile-wide and optional resolved contract identity.

        The profile-wide component includes every active typed domain and the
        complete MoE layer map.  A resolved operator contract is included when
        a cache entry or model artifact represents one concrete domain.  This
        method is the single serialization boundary for typed cache identity;
        callers do not reconstruct architecture fields independently.
        """

        return serialize_profile_layer_contract_identity(
            self,
            config,
            layer_contract=layer_contract,
        )

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

        num_layers = _get_config_value(config, "num_layers")
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

        if layer_kind is not None and not isinstance(layer_kind, LayerKind):
            try:
                layer_kind = LayerKind(layer_kind)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Unknown layer kind: {layer_kind!r}") from exc

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
                    f"Unable to bind operator {operator_name!r} to a typed layer "
                    f"domain: {exc}"
                ) from exc
            operator_contract = self.get_layer_contract_for_family(operator_family_id)

        if layer_kind is not None:
            kind_contract = self.get_layer_contract(layer_kind)
            if operator_contract is not None and kind_contract is not operator_contract:
                raise ValueError(
                    "layer_kind conflicts with operator_name contract: "
                    f"{layer_kind.value} != {operator_contract.layer_kind.value}"
                )
            operator_contract = kind_contract

        config_kind = self._resolve_config_layer_kind(config, layer_id)
        if operator_contract is not None:
            if operator_contract.layer_kind in (
                LayerKind.ROUTED,
                LayerKind.SHARED,
            ) and not bool(_get_config_value(config, "is_moe", False)):
                raise ValueError(
                    f"{operator_contract.layer_kind.value} operator "
                    f"{operator_name!r} requires an MoE model configuration"
                )
            if layer_id is not None and config_kind not in operator_contract.base_layer_kinds:
                allowed = ", ".join(kind.value for kind in operator_contract.base_layer_kinds)
                raise ValueError(
                    f"{operator_contract.layer_kind.value} operator {operator_name!r} "
                    "requires base layer kind "
                    f"{allowed}, but layer_id={layer_id} is {config_kind.value}"
                )
            if operator_contract.layer_kind is LayerKind.SHARED:
                supports_shared = _get_declared_config_attribute(
                    config,
                    "supports_share_expert",
                )
                if supports_shared is not _MISSING_CONFIG_ATTRIBUTE and callable(
                    supports_shared
                ):
                    if not bool(supports_shared()):
                        raise ValueError(
                            "shared-expert operator requested for a config that "
                            "does not support shared experts"
                        )
                elif (
                    get_declared_layer_width(
                        config,
                        LayerDimensionSource.SHARED,
                        reject_zero_sentinel=True,
                    )
                    is None
                ):
                    raise ValueError(
                        "shared-expert operator requires a declared shared width "
                        "(share_expert_dim or shared_expert_intermediate_size); "
                        "share_expert_dim must be set"
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
        if not spec.is_active(self, config):
            raise ValueError(
                f"{spec.layer_kind.value} layer contract is inactive for "
                f"profile {self.profile_id!r} and the supplied model configuration"
            )
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
            num_experts = _get_config_value(config, "num_experts")
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

        family_ids = tuple(spec.operator_family_ids)
        if operator_family_id is None and len(family_ids) > 1:
            raise ValueError(
                f"{spec.layer_kind.value} layer contract for profile "
                f"{self.profile_id!r} owns multiple operator families "
                f"{family_ids}; pass operator_name to select one"
            )
        resolved_family_id = operator_family_id
        if resolved_family_id is None and len(family_ids) == 1:
            resolved_family_id = family_ids[0]

        # The resolver receives one selected value at a time. Preserve that
        # value as a singleton domain so typed metadata never invents a wider
        # profiling envelope than the caller supplied.
        tensor_parallel_sizes = () if tp_size is None else (tp_size,)
        selected_padded_ffn_width = (
            None
            if tp_size is None
            else _pad_to_multiple(width, tp_size)
        )

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
            operator_family_id=resolved_family_id,
            operator_family_ids=family_ids,
            tensor_parallel_sizes=tensor_parallel_sizes,
            selected_padded_ffn_width=selected_padded_ffn_width,
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
        if not bool(_get_config_value(config, "is_moe", False)):
            return False
        num_layers = _get_config_value(config, "num_layers")
        raw_layers = _get_config_value(config, "moe_layers_enum")
        if raw_layers is not None:
            ids = parse_moe_layer_ids(raw_layers, num_layers)
            return bool(ids) and len(ids) < num_layers
        get_ids = _get_declared_config_attribute(config, "get_moe_layer_ids")
        if (
            get_ids is not _MISSING_CONFIG_ATTRIBUTE
            and callable(get_ids)
            and type(num_layers) is int
        ):
            ids = normalize_moe_layer_ids(get_ids(), num_layers)
            return bool(ids) and len(ids) < num_layers
        get_count = _get_declared_config_attribute(config, "get_num_moe_layers")
        if get_count is not _MISSING_CONFIG_ATTRIBUTE:
            if not callable(get_count):
                raise TypeError("get_num_moe_layers must be callable when declared")
            if type(num_layers) is not int or num_layers <= 0:
                raise ValueError(
                    "get_num_moe_layers requires a positive integer num_layers"
                )
            count = get_count()
            if type(count) is not int or count < 0 or count > num_layers:
                raise ValueError(
                    "get_num_moe_layers() must return an integer in the range "
                    f"[0, {num_layers}], got {count!r}"
                )
            return 0 < count < num_layers
        return False

    @classmethod
    def _resolve_config_layer_kind(
        cls, config: Any, layer_id: int | None
    ) -> LayerKind:
        if not bool(_get_config_value(config, "is_moe", False)):
            return LayerKind.DENSE
        num_layers = _get_config_value(config, "num_layers")
        if layer_id is None:
            return LayerKind.ROUTED
        raw_layers = _get_config_value(config, "moe_layers_enum")
        if raw_layers is not None:
            parsed_layer_ids = set(parse_moe_layer_ids(raw_layers, num_layers))
            return (
                LayerKind.ROUTED
                if layer_id in parsed_layer_ids
                else LayerKind.DENSE
            )
        predicate = _get_declared_config_attribute(config, "is_moe_layer")
        if predicate is not _MISSING_CONFIG_ATTRIBUTE and callable(predicate):
            return LayerKind.ROUTED if bool(predicate(layer_id)) else LayerKind.DENSE
        get_ids = _get_declared_config_attribute(config, "get_moe_layer_ids")
        if get_ids is not _MISSING_CONFIG_ATTRIBUTE and callable(get_ids):
            parsed_layer_ids = set(
                normalize_moe_layer_ids(get_ids(), num_layers)
            )
            return (
                LayerKind.ROUTED
                if layer_id in parsed_layer_ids
                else LayerKind.DENSE
            )
        return LayerKind.ROUTED

    @staticmethod
    def _parse_moe_layer_ids(raw_layers: Any, num_layers: Any) -> tuple[int, ...]:
        """Parse an explicit MoE layer map with one strict fail-fast contract."""
        return parse_moe_layer_ids(raw_layers, num_layers)

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


def serialize_profile_layer_contract_identity(
    profile: ModelArchitectureProfile,
    config: Any,
    *,
    layer_contract: ResolvedLayerContract | None = None,
) -> str:
    """Return canonical JSON for a profile-owned typed contract identity.

    The profile-wide identity and the concrete contract use the serializers
    defined in this module, so cache consumers share one representation.  A
    contract from another profile is rejected instead of producing an
    identity that appears valid but cannot be resolved by the requested
    architecture.
    """

    if not isinstance(profile, ModelArchitectureProfile):
        raise TypeError(
            "profile must be a ModelArchitectureProfile when serializing a "
            "typed layer identity"
        )
    if layer_contract is not None:
        if not isinstance(layer_contract, ResolvedLayerContract):
            raise TypeError(
                "layer_contract must be a ResolvedLayerContract when provided"
            )
        if layer_contract.profile_id != profile.profile_id:
            raise ValueError(
                "layer_contract profile does not match the architecture profile: "
                f"{layer_contract.profile_id!r} != {profile.profile_id!r}"
            )

    contract_payload = None
    if layer_contract is not None:
        serialized_contract = serialize_layer_contract_identity(layer_contract)
        if serialized_contract is None:
            raise ValueError("typed layer contract serialization unexpectedly returned None")
        contract_payload = json.loads(serialized_contract)

    payload = {
        "profile_identity": profile.get_layer_contract_identity(config),
        "resolved_contract": contract_payload,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


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
            predicate=lambda config: bool(_get_config_value(config, "is_moe", False)),
            message=lambda profile, config: (
                f"{profile.display_name} profile {profile.profile_id} "
                f"requires is_moe=True. Model: {_model_identifier(config)}"
            ),
        ),
        StructuralRequirement(
            name="requires_share_expert_dim",
            predicate=lambda config: get_declared_layer_width(
                config,
                LayerDimensionSource.SHARED,
            )
            is not None,
            message=lambda profile, config: (
                f"{profile.display_name} profile {profile.profile_id} "
                "requires a declared shared width "
                "(share_expert_dim or shared_expert_intermediate_size). "
                f"Model: {_model_identifier(config)}"
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
        explicit_profile = _get_declared_config_attribute(
            config,
            "model_architecture_profile",
        )
        if explicit_profile is not _MISSING_CONFIG_ATTRIBUTE and explicit_profile:
            if not isinstance(explicit_profile, str):
                raise TypeError(
                    "model_architecture_profile must be a string when declared, "
                    f"got {type(explicit_profile).__name__}"
                )
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
