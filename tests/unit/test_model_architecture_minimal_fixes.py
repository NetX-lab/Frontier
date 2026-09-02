"""Focused regressions for profile resolver error visibility and field reads."""

from types import SimpleNamespace

import pytest

from frontier.model_architectures import (
    ExpertParallelCollective,
    LayerKind,
    LinearAttentionImplementation,
    LinearAttentionProfile,
    ModelArchitectureProfile,
    StructuralRequirement,
)


def test_structural_requirement_preserves_predicate_error_detail() -> None:
    profile = ModelArchitectureProfile(
        profile_id="unit_wrapped_error_profile",
        display_name="Unit Wrapped Error Profile",
        linear_attention=LinearAttentionProfile(
            sharded_impl=LinearAttentionImplementation.GENERIC,
            sharded_ops=("attn_pre_proj", "attn_rope", "attn_post_proj"),
        ),
        expert_parallel_collective=ExpertParallelCollective.ALLTOALL,
        structural_requirements=(
            StructuralRequirement(
                name="requires_unit_contract",
                predicate=lambda config: (_ for _ in ()).throw(
                    ValueError("low-level binding failed")
                ),
                message=lambda profile, config: (
                    f"{profile.profile_id} requires unit structural contract"
                ),
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "unit_wrapped_error_profile requires unit structural contract: "
            "low-level binding failed"
        ),
    ):
        profile.validate_structural_requirements(SimpleNamespace(model_type="invalid"))


def test_resolver_does_not_accept_synthesized_num_experts() -> None:
    """EP validation reads declared config fields, not dynamic __getattr__."""

    class DynamicNumExpertsConfig:
        is_moe = True
        num_layers = 1
        mlp_hidden_dim = 128
        embedding_dim = 64

        def __getattr__(self, name: str) -> object:
            if name == "num_experts":
                return 8
            raise AttributeError(name)

    with pytest.raises(ValueError, match="num_experts must be a positive int"):
        ModelArchitectureProfile.generic().resolve_layer_contract(
            DynamicNumExpertsConfig(),
            layer_kind=LayerKind.ROUTED,
            tensor_parallel_size=1,
            expert_parallel_size=2,
        )
