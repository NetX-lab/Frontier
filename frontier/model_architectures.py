"""Plugin-style model architecture contracts for model-specific runtime semantics."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from frontier.types import ClusterType


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


@dataclass(frozen=True)
class ModelArchitectureProfile:
    """Declarative contract for model-specific architecture behavior."""

    profile_id: str
    display_name: str
    linear_attention: LinearAttentionProfile
    expert_parallel_collective: ExpertParallelCollective = ExpertParallelCollective.ALLGATHER
    target_embedded_mtp: bool = False
    predictor_attention_extra_ops: tuple[str, ...] = ()
    attention_shape_log_kind: str | None = None
    residual_add_policy: ResidualAddPolicy = ResidualAddPolicy.STANDARD
    skip_decode_ffn_attn_norm_residual: bool = False
    skip_decode_attn_residual: bool = False
    moe_tensor_parallel_allgather_op: str | None = None
    share_expert_tensor_parallel_allreduce_op: str | None = None
    share_expert_tp_allreduce_visibility_scale: float | None = None
    always_supports_share_expert: bool = False
    counts_share_expert_param_memory: bool = False
    step2_mini_compatible: bool = False
    step3_text_compatible: bool = False
    requires_share_expert_dim: bool = False
    requires_moe: bool = False
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
            step2_mini_compatible=True,
            requires_share_expert_dim=True,
            requires_moe=True,
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
            display_name="Step3Text MLA",
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
            attention_shape_log_kind="mla",
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
            step3_text_compatible=True,
            requires_share_expert_dim=True,
            requires_moe=True,
            match=match or _matches_step3_text,
        )

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
        return self.get("generic")


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
