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


def test_position_policy_resolves_before_parent_initialization(monkeypatch):
    from frontier.config import global_vars
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor
    from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import SklearnMoEExecutionTimePredictor
    from tests.unit.predictor_cache_fixtures import predictor_fixture_config
    assert resolve_moe_gating_routing_runtime_path("position_cyclic") == "position_cyclic"
    kwargs = predictor_fixture_config()
    kwargs["replica_config"].moe_routing_distribution_type = "position_cyclic"
    monkeypatch.setattr(global_vars, "SYS_ARCH", "co-location")
    monkeypatch.setattr(SklearnExecutionTimePredictor, "__init__", lambda *a, **k: pytest.fail("Unsupported scope reached parent init"))
    with pytest.raises(ValueError, match="only Step P/D"):
        class SourcePredictor(SklearnMoEExecutionTimePredictor):
            def _get_estimator(self):
                pytest.fail("Unsupported scope must not train")
            def _get_grid_search_params(self):
                return {}
        SourcePredictor(**kwargs)
