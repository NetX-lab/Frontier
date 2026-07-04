"""Plugin-style model architecture contracts for model-specific runtime semantics."""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal

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
    expert_parallel_collective: ExpertParallelCollective = ExpertParallelCollective.ALLGATHER
    target_embedded_mtp: bool = False
    predictor_attention_extra_ops: tuple[str, ...] = ()
    attention_shape_log_kind: AttentionShapeLogKind | None = None
    residual_add_policy: ResidualAddPolicy = ResidualAddPolicy.STANDARD
    skip_decode_ffn_attn_norm_residual: bool = False
    skip_decode_attn_residual: bool = False
    moe_tensor_parallel_allgather_op: str | None = None
    share_expert_tensor_parallel_allreduce_op: str | None = None
    share_expert_tp_allreduce_visibility_scale: float | None = None
    always_supports_share_expert: bool = False
    counts_share_expert_param_memory: bool = False
    structural_requirements: tuple[StructuralRequirement, ...] = ()
    match: ArchitectureMatcher = field(default=lambda _config: False, repr=False, compare=False)

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
            share_expert_tp_allreduce_visibility_scale=2.0 / 3.0,
            always_supports_share_expert=True,
            counts_share_expert_param_memory=True,
            structural_requirements=(
                *_moe_share_expert_requirements(),
                _requires_step3_mfa_attention_contract(),
            ),
            match=match or _matches_step3_text,
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

    def uses_expert_parallel_alltoall(
        self,
        cluster_type: ClusterType,
        expected_ep_size: int,
    ) -> bool:
        """Return whether EP synchronization should use alltoall semantics."""

        if expected_ep_size <= 1:
            return False
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
    """Resolve the model architecture profile for a runtime/profiling config."""

    return MODEL_ARCHITECTURE_REGISTRY.resolve(config)
