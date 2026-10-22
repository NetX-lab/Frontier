from __future__ import annotations

import pytest

from frontier.moe_routing_runtime import (
    STANDARD_MOE_GATING_ROUTING_RUNTIME_PATH,
    UNIFORM_MOE_GATING_ROUTING_RUNTIME_PATH,
    resolve_moe_gating_routing_runtime_path,
)


@pytest.mark.parametrize("distribution_type", ["balanced", "skewed", "zipf"])
def test_structured_distributions_use_standard_runtime_path(
    distribution_type: str,
) -> None:
    assert (
        resolve_moe_gating_routing_runtime_path(distribution_type)
        == STANDARD_MOE_GATING_ROUTING_RUNTIME_PATH
    )


def test_random_distribution_uses_uniform_runtime_path() -> None:
    assert (
        resolve_moe_gating_routing_runtime_path("random")
        == UNIFORM_MOE_GATING_ROUTING_RUNTIME_PATH
    )


def test_removed_routing_mode_values_fail_fast() -> None:
    with pytest.raises(ValueError, match="moe_routing_distribution_type"):
        resolve_moe_gating_routing_runtime_path("simulation")


@pytest.mark.parametrize("distribution", ["balanced", "random", "skewed", "zipf"])
def test_glm_router_kernel_is_not_the_workload_distribution(distribution):
    from frontier.config.model_config import BaseModelConfig
    profile = BaseModelConfig.create_from_name("zai-org/GLM-5.3").get_model_architecture_profile()
    assert resolve_moe_gating_routing_runtime_path(distribution, profile) == "grouped_topk_sigmoid"


def test_glm_grouped_router_parameters_come_from_model_config():
    from frontier.profiling.common.model_config import ModelConfig
    model = ModelConfig.from_model_name("zai-org/GLM-5.3")
    assert (model.n_group, model.topk_group, model.scoring_func) == (1, 1, "sigmoid")
    assert model.norm_topk_prob is True
    assert model.routed_scaling_factor == 2.5
