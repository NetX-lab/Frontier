"""PD-disaggregation transfer enum and config contract tests."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pd_transfer_types_are_exported_and_parse_analytical() -> None:
    from frontier.types import KVCacheTransferType, M2NTransferType

    assert KVCacheTransferType.ANALYTICAL.value == 1
    assert str(KVCacheTransferType.ANALYTICAL) == "analytical"
    assert KVCacheTransferType.from_str("analytical") is KVCacheTransferType.ANALYTICAL

    assert M2NTransferType.ANALYTICAL.value == 1
    assert str(M2NTransferType.ANALYTICAL) == "analytical"
    assert M2NTransferType.from_str("analytical") is M2NTransferType.ANALYTICAL


def test_analytical_transfer_configs_return_enum_types() -> None:
    from frontier.config.kv_cache_transfer_config import AnalyticalKVCacheTransferConfig
    from frontier.config.m2n_transfer_config import AnalyticalM2NTransferConfig
    from frontier.types import KVCacheTransferType, M2NTransferType

    assert AnalyticalKVCacheTransferConfig.get_type() is KVCacheTransferType.ANALYTICAL
    assert AnalyticalKVCacheTransferConfig.get_name() == "analytical"

    assert AnalyticalM2NTransferConfig.get_type() is M2NTransferType.ANALYTICAL
    assert AnalyticalM2NTransferConfig.get_name() == "analytical"


def test_pd_disaggregation_release_guard_rejects_parallel_cluster_default() -> None:
    from frontier.config.config import SimulationConfig

    config = object.__new__(SimulationConfig)
    config.sys_arch = "pd-disaggregation"
    config.use_cuda_graph = False
    config.enable_parallel_clusters = True

    with pytest.raises(ValueError, match="--no-enable_parallel_clusters"):
        config._validate_open_source_release_architecture_guard()


def test_pd_disaggregation_release_guard_allows_explicit_sequential_mode() -> None:
    from frontier.config.config import SimulationConfig

    config = object.__new__(SimulationConfig)
    config.sys_arch = "pd-disaggregation"
    config.use_cuda_graph = False
    config.enable_parallel_clusters = False

    config._validate_open_source_release_architecture_guard()


def test_pd_disaggregation_cli_release_guard_exits_without_traceback() -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "frontier.main", "--sys_arch", "pd-disaggregation"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 1
    assert "--no-enable_parallel_clusters" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("kwargs", "expected_field"),
    [
        ({"network_bandwidth_gbps": 0.0}, "network_bandwidth_gbps"),
        ({"network_bandwidth_gbps": -1.0}, "network_bandwidth_gbps"),
        ({"network_latency_ms": -0.1}, "network_latency_ms"),
        ({"compression_ratio": 0.0}, "compression_ratio"),
        ({"compression_ratio": -1.0}, "compression_ratio"),
        ({"kv_cache_dtype_size_bytes": 0}, "kv_cache_dtype_size_bytes"),
        ({"kv_cache_dtype_size_bytes": -1}, "kv_cache_dtype_size_bytes"),
        ({"override_num_layers": 0}, "override_num_layers"),
        ({"override_num_heads": -1}, "override_num_heads"),
        ({"override_head_dim": 0}, "override_head_dim"),
    ],
)
def test_kv_transfer_config_rejects_invalid_numeric_boundaries(
    kwargs: dict[str, float | int],
    expected_field: str,
) -> None:
    from frontier.config.kv_cache_transfer_config import AnalyticalKVCacheTransferConfig

    with pytest.raises(ValueError, match=expected_field):
        AnalyticalKVCacheTransferConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "expected_field"),
    [
        ({"memory_bandwidth_gbps": 0.0}, "memory_bandwidth_gbps"),
        ({"memory_bandwidth_gbps": -1.0}, "memory_bandwidth_gbps"),
        ({"network_latency_ms": -0.1}, "network_latency_ms"),
        ({"compression_ratio": 0.0}, "compression_ratio"),
        ({"compression_ratio": -1.0}, "compression_ratio"),
        ({"activation_dtype_size_bytes": 0}, "activation_dtype_size_bytes"),
        ({"activation_dtype_size_bytes": -1}, "activation_dtype_size_bytes"),
        ({"override_hidden_size": 0}, "override_hidden_size"),
        ({"override_intermediate_size": -1}, "override_intermediate_size"),
    ],
)
def test_m2n_transfer_config_rejects_invalid_numeric_boundaries(
    kwargs: dict[str, float | int],
    expected_field: str,
) -> None:
    from frontier.config.m2n_transfer_config import AnalyticalM2NTransferConfig

    with pytest.raises(ValueError, match=expected_field):
        AnalyticalM2NTransferConfig(**kwargs)
