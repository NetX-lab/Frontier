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


def test_explicit_runtime_preserves_distribution_validation() -> None:
    assert resolve_moe_gating_routing_runtime_path("balanced", "uniform_topk") == "uniform_topk"
    assert resolve_moe_gating_routing_runtime_path("random", "standard_fused_topk") == "standard_fused_topk"
    assert resolve_moe_gating_routing_runtime_path("balanced", "") == "standard_fused_topk"
    with pytest.raises(ValueError, match="routing_runtime_path"):
        resolve_moe_gating_routing_runtime_path("balanced", "unknown")
    with pytest.raises(ValueError, match="moe_routing_distribution_type"):
        resolve_moe_gating_routing_runtime_path("simulation", "uniform_topk")


def test_runtime_config_copy_and_predictor_keep_balanced_loads() -> None:
    from frontier.config.config import ClusterConfig, OrcaSchedulerConfig, ReplicaConfig
    source = ReplicaConfig(
        model_name="Phi-tiny-MoE-instruct",
        moe_gating_routing_runtime_path="uniform_topk",
    )
    cluster = ClusterConfig(num_replicas=1, replica_config=source,
                            replica_scheduler_config=OrcaSchedulerConfig())
    copied = cluster._create_replica_config_copy()
    assert copied.moe_gating_routing_runtime_path == "uniform_topk"
    assert copied.moe_routing_distribution_type == "balanced"

    from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
        SklearnMoEExecutionTimePredictor,
    )
    from types import SimpleNamespace

    predictor = SimpleNamespace(
        _replica_config=copied, _model_config=copied.model_config,
        _moe_routing_distribution_type="balanced", _moe_routing_seed=42,
        _moe_ep_size=copied.moe_expert_parallel_size,
    )
    assert SklearnMoEExecutionTimePredictor._get_requested_moe_gating_routing_runtime_path(predictor) == "uniform_topk"
    allocations = SklearnMoEExecutionTimePredictor._init_global_routing_allocations(predictor)
    assert all(set(layer.values()) == {1 / copied.total_expert_num}
               for layer in allocations.values())


def test_runtime_override_reaches_effective_cli_config(tmp_path) -> None:
    import subprocess
    import sys

    script = """
from frontier.config.config import SimulationConfig
config = SimulationConfig.create_from_cli_args()
assert config.cluster_config.replica_config.moe_gating_routing_runtime_path == 'uniform_topk'
assert config.cluster_config.replica_config.moe_routing_distribution_type == 'balanced'
"""
    subprocess.run([
        sys.executable, "-c", script,
        "--replica_config_model_name", "Phi-tiny-MoE-instruct",
        "--replica_config_moe_gating_routing_runtime_path", "uniform_topk",
        "--metrics_config_output_dir", str(tmp_path / "metrics"),
        "--metrics_config_cache_dir", str(tmp_path / "cache"),
    ], check=True, capture_output=True, text=True)
